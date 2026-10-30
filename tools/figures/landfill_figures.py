#!/usr/bin/env python3
"""Draw the four data figures for projects/landfill-survey.html.

With no arguments everything is synthetic: one invented dumpsite with "before"
and "after" DEMs, a material-zone raster and a thermal field, all generated from
a fixed seed, so the PNGs come out the same on every run.

    python tools/figures/landfill_figures.py

Pass real GeoTIFFs to draw the same figures from survey data. The rasters can
have any resolution and CRS; they are resampled onto a north-up grid aligned
with the "before" DEM (--cell metres, default 0.5).

    python tools/figures/landfill_figures.py \\
        --before dem_t1.tif --after dem_t2.tif \\
        --zones zones.tif --thermal thermal_celsius.tif \\
        --boundary site.geojson --out figures/

--zones is an integer raster using the codes in ZONES below (e.g. the
segmentation model's output) and --thermal is surface temperature in °C. A
figure whose input is missing is skipped. --boundary is a GeoJSON polygon in
the "before" DEM's CRS; without it the outline is traced from the DEM footprint.
Real-data figures leave out the ILLUSTRATIVE tag.

Requires numpy, matplotlib and scipy; rasterio for --before/--after; pyoxipng
(or Pillow) for lossless PNG compression. Fonts are fetched from Google Fonts
into tools/figures/fonts/ on first run.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "assets/img/projects/landfill"
FONT_DIR = HERE / "fonts"

SEED = 2026
CELL = 0.5          # m, grid spacing
SITE_W, SITE_H = 300.0, 170.0  # m, synthetic grid extent
LOD = 0.15          # m, level of detection for the DEM of difference
HOT_C = 50.0        # °C, hotspots above this are flagged
MOVE_CM = 2.0       # cm, movement arrows drawn above this

INK = "#2B1B18"
ACCENT = "#A51D2D"
TAG_GREY = "#8C8C8C"
LOD_GREY = "#DADADA"
HATCH_GREY = "#7D7D7D"

# Zone codes for --zones rasters. Colours step from dark to light so they stay
# apart without hue; checked for protan/deutan separation across all pairs.
ZONES = [
    (1, "Fresh waste", "#F2A33A"),
    (2, "Legacy waste", "#7B3F8C"),
    (3, "Soil cover", "#2E9C7E"),
    (4, "Cleared ground", "#F3E7C2"),
    (5, "Leachate pond", "#1B3A5C"),
]
FRESH, LEGACY, SOIL, CLEARED, POND = (code for code, _, _ in ZONES)
ZONE_ALPHA = 0.55

FONTS = {"Unbounded": (600,), "Outfit": (400, 500)}
TITLE_FONT = "Unbounded"


# --------------------------------------------------------------------------
# Data


@dataclass
class Site:
    before: np.ndarray  # DEMs in metres, north-up (row 0 = north), NaN outside the site
    after: np.ndarray
    cell: float
    boundary: np.ndarray  # (n, 2) polygon in metres from the grid's south-west corner
    zones: np.ndarray | None = None  # ZONES codes, 0 = none
    thermal: np.ndarray | None = None  # °C
    synthetic: bool = True

    @property
    def extent(self):
        ny, nx = self.before.shape
        return (0.0, nx * self.cell, 0.0, ny * self.cell)

    @property
    def valid(self):
        return np.isfinite(self.before) & np.isfinite(self.after)


def grid_xy(shape, cell):
    ny, nx = shape
    x = (np.arange(nx) + 0.5) * cell
    y = ny * cell - (np.arange(ny) + 0.5) * cell
    return np.meshgrid(x, y)


def polygon_mask(boundary, shape, cell):
    X, Y = grid_xy(shape, cell)
    inside = MplPath(boundary).contains_points(np.column_stack([X.ravel(), Y.ravel()]))
    return inside.reshape(shape)


def fill_nan(z):
    """Nearest-neighbour fill, so filters and gradients don't bleed NaNs."""
    bad = ~np.isfinite(z)
    if not bad.any():
        return z
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def gradient_xy(z, cell):
    """d/dx (east) and d/dy (north) of a north-up grid."""
    d_row, d_col = np.gradient(z, cell)
    return d_col, -d_row


def fractal_noise(rng, shape, beta=3.2):
    """Random surface with a 1/f^beta power spectrum, zero mean and unit std."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    k = np.hypot(fx, fy)
    k[0, 0] = np.inf
    spectrum = k ** (-beta / 2) * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape))
    z = np.fft.irfft2(spectrum, s=shape)
    return (z - z.mean()) / z.std()


def smooth_noise(rng, shape, sigma):
    n = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return n / n.std()


def rotated(X, Y, x0, y0, rot_deg):
    t = np.deg2rad(rot_deg)
    u = (X - x0) * np.cos(t) + (Y - y0) * np.sin(t)
    v = -(X - x0) * np.sin(t) + (Y - y0) * np.cos(t)
    return u, v


def gaussian_mound(X, Y, x0, y0, amp, sx, sy, rot_deg=0.0):
    u, v = rotated(X, Y, x0, y0, rot_deg)
    return amp * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))


def heap_mound(X, Y, x0, y0, amp, sx, sy, rot_deg, power, wobble):
    """Waste heap: power > 2 flattens the top and steepens the flanks, and
    `wobble` (a smooth field around 0) makes the footprint irregular."""
    u, v = rotated(X, Y, x0, y0, rot_deg)
    q = ((u / sx) ** 2 + (v / sy) ** 2) * np.exp(wobble)
    return amp * np.exp(-0.5 * q ** (power / 2))


def blob(X, Y, x0, y0, rx, ry, rot_deg=0.0, wobble=0.0):
    """Ellipse with an irregular outline."""
    u, v = rotated(X, Y, x0, y0, rot_deg)
    return ((u / rx) ** 2 + (v / ry) ** 2) * np.exp(wobble) <= 1


def smooth_labels(labels, sigma):
    """Majority-style smoothing so zones read like a segmentation output."""
    codes = np.unique(labels)
    stack = np.stack([ndimage.gaussian_filter((labels == c).astype(float), sigma) for c in codes])
    return codes[stack.argmax(0)]


def synthetic_boundary(rng):
    t = np.linspace(0, 2 * np.pi, 30, endpoint=False) + rng.uniform(-0.06, 0.06, 30)
    wobble = 1 + sum(
        a * np.cos(k * t + p)
        for k, a, p in zip((2, 3, 5), rng.uniform(0.02, 0.045, 3), rng.uniform(0, 2 * np.pi, 3))
    )
    wobble *= 1 + rng.uniform(-0.015, 0.015, t.size)
    c, s = np.cos(t), np.sin(t)
    # Superellipse: boxier than an ellipse, like a fenced plot.
    r = (np.abs(c / 132) ** 3 + np.abs(s / 72) ** 3) ** (-1 / 3) * wobble
    xy = np.column_stack([150 + r * c, 86 + r * s])
    return np.clip(xy, [6, 6], [SITE_W - 6, SITE_H - 6])


def synthetic_site(seed=SEED):
    """One invented dumpsite surveyed twice."""
    rng = np.random.default_rng(seed)
    shape = (int(SITE_H / CELL), int(SITE_W / CELL))
    X, Y = grid_xy(shape, CELL)
    boundary = synthetic_boundary(rng)
    inside = polygon_mask(boundary, shape, CELL)

    # Ground tilts down to the south-west, where the leachate pond collects.
    base = 2.0 + 0.010 * (X - 150) + 0.008 * (Y - 85)
    lobes = 0.3 * smooth_noise(rng, shape, 24)  # low-frequency field for irregular outlines
    jag = 0.5 * smooth_noise(rng, shape, 8)  # finer one for cut lines and pond edges
    mounds = {  # x, y, height, sx, sy, rotation, flank steepness
        "west": (72, 98, 10.0, 24, 18, 20, 2.8),
        "north": (150, 114, 12.0, 30, 20, -8, 3.0),
        "southeast": (200, 62, 8.5, 21, 15, 35, 2.6),
        "northeast": (238, 116, 7.5, 17, 13, 0, 2.6),
        "south": (112, 50, 5.5, 15, 11, 10, 2.4),
    }
    heaps = {name: heap_mound(X, Y, *m, wobble=lobes) for name, m in mounds.items()}
    # A thin, patchy blanket of older waste, absent along the southern access strip.
    blanket = (smooth_noise(rng, shape, 30) > -0.3) & ~blob(X, Y, 122, 22, 80, 13, 3, lobes + jag)
    heap = sum(heaps.values()) + 1.2 * ndimage.gaussian_filter(blanket.astype(float), 6)

    capped = blob(X, Y, 150, 114, 40, 27, -8, lobes + 0.5 * jag)  # north mound has a soil cap
    cap_soft = ndimage.gaussian_filter(capped.astype(float), 4)
    fine = smooth_noise(rng, shape, 0.8)
    texture = (0.3 * fractal_noise(rng, shape, 2.8) + 0.06 * fine) * (1 - 0.35 * cap_soft)
    texture *= 0.5 + 0.5 * np.clip(heap / 2, 0, 1)
    terrain = base + heap + texture

    pond = blob(X, Y, 50, 40, 15, 9, -15, jag)
    terrain -= gaussian_mound(X, Y, 50, 40, 1.6, 11, 7, -15)
    terrain[pond] = np.percentile(terrain[pond], 30)

    # --- between surveys ---
    after = terrain.copy()
    batter = np.tan(np.deg2rad(35))
    # West mound: an excavation bay taken down to near ground level.
    bay = blob(X, Y, 90, 92, 19, 14, 25, lobes + jag)
    dist = ndimage.distance_transform_edt(~bay) * CELL
    after = np.minimum(after, base + 0.25 + 0.05 * fine + batter * dist)
    # South-east mound: summit cut back to a bench.
    bench = blob(X, Y, 200, 62, 14, 9, 35, lobes + jag)
    dist = ndimage.distance_transform_edt(~bench) * CELL
    after = np.minimum(after, base + 4.0 + 0.05 * fine + batter * dist)
    cut = terrain - after
    after += 0.12 * fractal_noise(rng, shape, 3.0) * ndimage.gaussian_filter((cut > 0.3).astype(float), 2)

    # Fresh dumping on the east side: overlapping truck-load piles.
    piles = np.zeros(shape)
    centres = rng.normal([255, 72], [9, 8], size=(60, 2))
    centres = centres[MplPath(boundary).contains_points(centres)][:34]
    for x0, y0 in centres:
        h, r = rng.uniform(0.9, 1.9), rng.uniform(1.8, 3.0)
        piles = np.maximum(piles, h * np.exp(-0.5 * ((X - x0) ** 2 + (Y - y0) ** 2) / r**2))
    after += piles

    # Creep on the north mound's south flank: material slides a few cm
    # downslope, the head drops and the toe bulges.
    gx, gy = gradient_xy(heaps["north"], CELL)
    slope = np.hypot(gx, gy) + 1e-9
    cx, cy = 148.0, 92.0
    env = np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / 12.0**2)
    r0, c0 = int((SITE_H - cy) / CELL), int(cx / CELL)
    d0 = -np.array([gx[r0, c0], gy[r0, c0]]) / slope[r0, c0]
    s = (X - cx) * d0[0] + (Y - cy) * d0[1]  # metres downslope of the centre
    ux, uy = -0.12 * env * gx / slope, -0.12 * env * gy / slope
    w = env * (-0.07 + 0.12 / (1 + np.exp(-s / 5)))
    rows, cols = np.indices(shape)
    after = ndimage.map_coordinates(after, [rows + uy / CELL, cols - ux / CELL], order=3, mode="nearest") + w

    # Each survey gets its own spatially correlated vertical error.
    before_dem = terrain + 0.03 * smooth_noise(rng, shape, 1.2)
    after_dem = after + 0.03 * smooth_noise(rng, shape, 1.2)

    # Material zones as a segmentation model would label the after survey.
    height = after - base
    zones = np.full(shape, LEGACY)
    zones[height < 0.6] = CLEARED
    zones[capped] = SOIL
    zones[piles > 0.3] = FRESH
    zones[pond] = POND
    zones = smooth_labels(zones, 2.5)

    # Surface temperature: ambient varies with the material, plus hotspots.
    surface = {CLEARED: 2.0, FRESH: 1.0, SOIL: -0.8, POND: -2.5}
    offset = sum(dt * (zones == code) for code, dt in surface.items())
    temp = 32.0 + 0.9 * fractal_noise(rng, shape, 2.6) + 0.01 * (X - 150)
    temp += ndimage.gaussian_filter(offset.astype(float), 1.5) + 0.3 * smooth_noise(rng, shape, 0.7)
    temp = np.clip(temp, 29.5, 35.5)
    hotspots = [  # x, y, peak °C, sx, sy, rotation
        (236, 118, 65.0, 4.5, 3.0, 30),
        (62, 106, 60.0, 5.0, 3.5, -20),
        (114, 54, 55.5, 4.0, 4.0, 0),
        (176, 128, 53.0, 3.5, 2.5, 60),
        (210, 92, 46.0, 4.0, 3.0, 0),
    ]
    lumpy = 1 + 0.15 * smooth_noise(rng, shape, 2)
    for x0, y0, peak, sx, sy, rot in hotspots:
        g = gaussian_mound(X, Y, x0, y0, 1.0, sx, sy, rot) * lumpy
        g /= g.max()
        r, c = np.unravel_index(g.argmax(), shape)
        temp += (peak - temp[r, c]) * g

    outside = ~inside
    for a in (before_dem, after_dem, temp):
        a[outside] = np.nan
    zones[outside] = 0
    return Site(before_dem, after_dem, CELL, boundary, zones, temp, synthetic=True)


def load_site(args):
    """Read real survey rasters onto one north-up grid aligned with --before."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    with rasterio.open(args.before) as src:
        if src.transform.b or src.transform.d:
            sys.exit("--before is rotated; warp it to a north-up grid first")
        crs = src.crs
        left, bottom, right, top = src.bounds
    cell = args.cell
    shape = (int(round((top - bottom) / cell)), int(round((right - left) / cell)))
    transform = from_origin(left, top, cell, cell)
    bottom = top - shape[0] * cell

    def read(path, resampling):
        dst = np.full(shape, np.nan)
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=dst,
                src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                resampling=resampling,
            )
        return dst

    before = read(args.before, Resampling.average)
    after = read(args.after, Resampling.average)
    thermal = read(args.thermal, Resampling.average) if args.thermal else None
    zones = None
    if args.zones:
        z = read(args.zones, Resampling.nearest)
        zones = np.where(np.isfinite(z), z, 0).astype(int)

    if args.boundary:
        boundary = read_boundary(args.boundary, left, bottom)
    else:
        boundary = footprint_boundary(np.isfinite(before) & np.isfinite(after), cell)
    outside = ~polygon_mask(boundary, shape, cell)
    for a in (before, after, thermal):
        if a is not None:
            a[outside] = np.nan
    if zones is not None:
        zones[outside] = 0
    return Site(before, after, cell, boundary, zones, thermal, synthetic=False)


def read_boundary(path, left, bottom):
    gj = json.loads(Path(path).read_text())
    if gj.get("type") == "FeatureCollection":
        gj = gj["features"][0]
    geom = gj.get("geometry", gj)
    rings = geom["coordinates"] if geom["type"] == "Polygon" else geom["coordinates"][0]
    return np.asarray(rings[0], float)[:, :2] - (left, bottom)


def footprint_boundary(valid, cell):
    from contourpy import contour_generator

    X, Y = grid_xy(valid.shape, cell)
    lines = contour_generator(X, Y, ndimage.binary_opening(valid, iterations=3).astype(float)).lines(0.5)
    return max(lines, key=len)


def write_synthetic(site, out):
    """Write the synthetic site as GeoTIFFs, to test the --before/--after path."""
    import rasterio
    from rasterio.transform import from_origin

    out.mkdir(parents=True, exist_ok=True)
    left, top = 500000.0, 3000000.0 + site.extent[3]  # arbitrary UTM 43N origin
    profile = dict(driver="GTiff", crs="EPSG:32643", transform=from_origin(left, top, site.cell, site.cell),
                   width=site.before.shape[1], height=site.before.shape[0], count=1, compress="deflate")
    for name, arr, dtype, nodata in [
        ("before", site.before, "float32", -9999.0), ("after", site.after, "float32", -9999.0),
        ("thermal", site.thermal, "float32", -9999.0), ("zones", site.zones, "uint8", 0),
    ]:
        with rasterio.open(out / f"{name}.tif", "w", dtype=dtype, nodata=nodata, **profile) as dst:
            dst.write(np.where(np.isfinite(arr), arr, nodata).astype(dtype), 1)
    ring = (site.boundary + (left, top - site.extent[3])).tolist()
    geom = {"type": "Polygon", "coordinates": [ring + ring[:1]]}
    (out / "boundary.geojson").write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": geom}))
    print(f"wrote synthetic GeoTIFFs to {out}")


# --------------------------------------------------------------------------
# Analysis


def estimate_displacement(site, window=12.0, earthworks=0.25, damping=0.02, noise_corr=0.8, min_area=60.0):
    """Fit after(p) = before(p - u) + w in a moving Gaussian window.

    A Lucas-Kanade style least-squares fit on the DEMs: each window solves for
    a horizontal shift (ux, uy) and vertical shift w, in metres, and flags
    windows where the shift is significant at 95% over at least `min_area`
    square metres. Cells within a window of
    earthworks (|DoD| > `earthworks` m) are excluded, since cut and fill are
    not slope movement.
    """
    valid = site.valid
    dod = np.where(valid, site.after - site.before, 0.0)
    disturbed = np.abs(ndimage.gaussian_filter(dod, 2)) > earthworks
    excluded = ndimage.distance_transform_edt(~disturbed) * site.cell < window
    weight = (valid & ~excluded).astype(float)

    gx, gy = gradient_xy(ndimage.gaussian_filter(fill_nan(site.before), 1.0), site.cell)
    a = (-gx, -gy, np.ones_like(gx))
    sigma = window / 2 / site.cell
    total = ndimage.gaussian_filter(weight, sigma) + 1e-9

    def mean(f):
        return ndimage.gaussian_filter(f * weight, sigma) / total

    A = np.empty(dod.shape + (3, 3))
    b = np.empty(dod.shape + (3,))
    for i in range(3):
        b[..., i] = mean(a[i] * dod)
        for j in range(i + 1):
            A[..., i, j] = A[..., j, i] = mean(a[i] * a[j])
    damped = A + np.diag([damping, damping, 1e-6])  # flat ground can't show sideways motion
    p = np.linalg.solve(damped, b[..., None])[..., 0]

    # 95% test from the fit residuals. Survey errors are correlated over about
    # `noise_corr` m, which sets how many independent samples a window holds.
    resid = mean(dod**2) - 2 * np.einsum("...i,...i", p, b) + np.einsum("...i,...ij,...j", p, A, p)
    n_eff = np.maximum((sigma * site.cell / noise_corr) ** 2 * total, 1.0)
    cov = np.maximum(resid, 0)[..., None, None] / n_eff[..., None, None] * np.linalg.inv(damped)
    uh = p[..., :2, None]
    chi2 = (np.swapaxes(uh, -1, -2) @ np.linalg.solve(cov[..., :2, :2] + 1e-12 * np.eye(2), uh))[..., 0, 0]
    significant = (chi2 > 5.99) | (np.abs(p[..., 2]) > 1.96 * np.sqrt(cov[..., 2, 2]))

    ux, uy, w = np.moveaxis(p, -1, 0)
    ok = valid & ~excluded & (total > 0.75)  # windows mostly clear of earthworks and the edge
    labels, n = ndimage.label(ok & significant)
    areas = ndimage.sum_labels(np.ones_like(w), labels, np.arange(1, n + 1)) * site.cell**2
    significant = np.isin(labels, 1 + np.flatnonzero(areas >= min_area))
    return ux, uy, w, ok, significant, excluded & valid


def find_hotspots(temp, cell, threshold=HOT_C, min_area=1.0):
    """Connected areas above `threshold`, hottest first: (peak, x, y, radius m)."""
    hot = np.nan_to_num(temp, nan=-np.inf) > threshold
    labels, n = ndimage.label(hot)
    ny = temp.shape[0]
    spots = []
    for i in range(1, n + 1):
        region = labels == i
        area = region.sum() * cell**2
        if area < min_area:
            continue
        r, c = np.unravel_index(np.where(region, temp, -np.inf).argmax(), temp.shape)
        spots.append((temp[r, c], (c + 0.5) * cell, (ny - r - 0.5) * cell, np.sqrt(area / np.pi)))
    return sorted(spots, reverse=True)


def hillshade(site, z, exaggeration=2.0):
    ls = LightSource(azdeg=315, altdeg=35)
    return ls.hillshade(fill_nan(z), vert_exag=exaggeration, dx=site.cell, dy=site.cell)


# --------------------------------------------------------------------------
# Figure furniture


def frame(site, title, subtitle):
    fig = plt.figure(figsize=(8, 4.5), dpi=200, facecolor="white")
    fig.text(0.035, 0.945, title, fontfamily=TITLE_FONT, fontweight=600, fontsize=15, va="top")
    fig.add_artist(Line2D([0.035, 0.11], [0.845, 0.845], color=ACCENT, lw=1.4))
    fig.text(0.035, 0.815, subtitle, fontsize=10.5, va="top")
    if site.synthetic:
        fig.text(0.965, 0.94, "ILLUSTRATIVE", fontfamily=TITLE_FONT, fontweight=600, fontsize=7,
                 color=TAG_GREY, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.9", fc="white", ec=TAG_GREY, lw=0.7))
    ax = fig.add_axes([0.035, 0.04, 0.70, 0.71])
    x0, x1, y0, y1 = site.extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_anchor("SW")
    ax.axis("off")
    ax.add_patch(Polygon(site.boundary, closed=True, fill=False, ec=INK, lw=1.1, zorder=6))
    add_scalebar(ax, site)
    add_north_arrow(ax, site)
    return fig, ax


def nice_length(target):
    return max(v for v in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000) if v <= max(target, 5))


def add_scalebar(ax, site):
    x0, x1, y0, y1 = site.extent
    length = nice_length((x1 - x0) / 6)
    half = length / 2
    bx, by = x0 + 0.02 * (x1 - x0), y0 + 0.025 * (y1 - y0)
    h = 0.018 * (y1 - y0)
    halo = [withStroke(linewidth=2.5, foreground="white")]
    ax.add_patch(Rectangle((bx, by), half, h, fc=INK, ec=INK, lw=0.6, zorder=7))
    ax.add_patch(Rectangle((bx + half, by), half, h, fc="white", ec=INK, lw=0.6, zorder=7))
    for i, label in enumerate(("0", f"{half:g}", f"{length:g} m")):
        ax.text(bx + i * half, by + h + 0.012 * (y1 - y0), label, ha="center", va="bottom",
                fontsize=8.5, zorder=7, path_effects=halo)


def add_north_arrow(ax, site):
    x0, x1, y0, y1 = site.extent
    size = 0.075 * (y1 - y0)
    cx, top = x0 + 0.035 * (x1 - x0), y1 - 0.12 * (y1 - y0)
    notch = top - 0.7 * size
    ax.add_patch(Polygon([(cx, top), (cx - 0.38 * size, top - size), (cx, notch)], fc=INK, ec=INK, lw=0.7, zorder=7))
    ax.add_patch(Polygon([(cx, top), (cx + 0.38 * size, top - size), (cx, notch)], fc="white", ec=INK, lw=0.7, zorder=7))
    ax.text(cx, top + 0.015 * (y1 - y0), "N", ha="center", va="bottom", fontfamily=TITLE_FONT,
            fontweight=600, fontsize=8.5, zorder=7)


def side_colorbar(fig, mappable, label, **kwargs):
    fig.text(0.785, 0.745, label, fontsize=10, fontweight=500, va="top", linespacing=0.95)
    cax = fig.add_axes([0.79, 0.265, 0.022, 0.32])
    cb = fig.colorbar(mappable, cax=cax, **kwargs)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cax.tick_params(labelsize=9, length=2.5, width=0.6, color=INK, pad=3)
    return cb


def side_legend(fig, handles, title=None, y=0.04, **kwargs):
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.778, y), frameon=False,
                     fontsize=9, handlelength=1.6, handleheight=1.0, labelspacing=0.55,
                     borderaxespad=0, title=title, alignment="left", **kwargs)
    if title:
        leg.get_title().set_fontsize(10)
        leg.get_title().set_fontweight(500)
    return leg


def boundary_handle():
    return Line2D([], [], color=INK, lw=1.1, label="Site boundary")


def minus(v):
    return f"{v:g}".replace("-", "−")


# --------------------------------------------------------------------------
# Figures


def fig_change_map(site, lod):
    dod = site.after - site.before
    edges = [e for e in (0.5, 1.0, 2.0, 4.0) if e > lod]
    bounds = [-e for e in edges[::-1]] + [-lod, lod] + edges
    n = len(edges) + 1  # classes per side, including the open-ended one
    rdbu = plt.get_cmap("RdBu")
    colors = list(rdbu(np.linspace(0.02, 0.37, n))) + [to_rgb(LOD_GREY)] + list(rdbu(np.linspace(0.63, 0.98, n)))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N, extend="both")

    fig, ax = frame(site, "Cut / fill change map", "DEM of difference, after − before survey")
    im = ax.imshow(np.ma.masked_invalid(dod), extent=site.extent, cmap=cmap, norm=norm, zorder=1)
    cb = side_colorbar(fig, im, "Elevation\nchange (m)", spacing="uniform", ticks=bounds)
    cb.ax.set_yticklabels([minus(b) for b in bounds])
    for y, text in ((1.14, "Fill (added)"), (-0.14, "Cut (removed)")):
        cb.ax.text(0, y, text, transform=cb.ax.transAxes, va="center", fontsize=9, fontweight=500)
    side_legend(fig, [
        Patch(fc=LOD_GREY, ec="none", label=f"Below LoD (±{lod:g} m)"),
        boundary_handle(),
    ])
    return fig


def fig_zone_map(site):
    hs = hillshade(site, site.after, 3.0)
    detail = np.clip(hs + 0.8 * (hs - ndimage.gaussian_filter(hs, 2)), 0, 1)  # bring out surface texture
    base = 0.22 + 0.72 * detail
    rgb = np.repeat(base[..., None], 3, axis=2)
    for code, _, color in ZONES:
        sel = site.zones == code
        rgb[sel] = (1 - ZONE_ALPHA) * rgb[sel] + ZONE_ALPHA * np.array(to_rgb(color))
    rgba = np.dstack([rgb, site.valid.astype(float)])

    fig, ax = frame(site, "Material zones", "Segmentation over the orthomosaic, after survey")
    ax.imshow(rgba, extent=site.extent, zorder=1)
    mid = 0.22 + 0.72 * 0.7  # a typical ortho grey, so swatches match the map
    handles = [
        Patch(fc=(1 - ZONE_ALPHA) * mid + ZONE_ALPHA * np.array(to_rgb(color)), ec=INK, lw=0.4, label=label)
        for _, label, color in ZONES
    ]
    side_legend(fig, handles + [boundary_handle()], title="Zone", y=0.18)
    return fig


def fig_thermal(site):
    temp = site.thermal
    vmin = min(28.0, np.floor(np.nanpercentile(temp, 1)))
    vmax = max(66.0, np.ceil(np.nanmax(temp)))
    spots = find_hotspots(temp, site.cell)

    fig, ax = frame(site, "Thermal hotspots", "Surface temperature, after survey")
    im = ax.imshow(np.ma.masked_invalid(temp), extent=site.extent, cmap="inferno", vmin=vmin, vmax=vmax, zorder=1)
    stroke = [withStroke(linewidth=3.0, foreground=INK)]
    for i, (_, x, y, radius) in enumerate(spots, 1):
        r = max(5.0, radius + 4.0)
        ax.add_patch(Circle((x, y), r, fill=False, ec="white", lw=1.4, zorder=8, path_effects=stroke))
        ax.text(x + r * 0.72, y + r * 0.72, str(i), ha="center", va="center", fontsize=8, fontweight=500,
                zorder=9, bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec=INK, lw=0.7))
    side_colorbar(fig, im, "Surface\ntemperature (°C)")
    side_legend(fig, [
        Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=INK, mew=1.2, label=f"Hotspot > {HOT_C:g} °C"),
        boundary_handle(),
    ])
    return fig


def fig_slope(site):
    ux, uy, w, ok, significant, excluded = estimate_displacement(site)
    ux, uy, w = ux * 100, uy * 100, w * 100  # cm
    limit = 8.0
    hs = hillshade(site, site.before, 3.0)
    cmap = plt.get_cmap("RdBu")
    norm = Normalize(-limit, limit)
    # Movement that isn't significant at 95% is shown as plain terrain.
    tint = ndimage.gaussian_filter(significant.astype(float), 2)
    color = cmap(norm(w))[..., :3]
    plain = np.repeat((0.80 + 0.18 * hs)[..., None], 3, axis=2)
    rgb = tint[..., None] * color * (0.62 + 0.38 * hs)[..., None] + (1 - tint[..., None]) * plain
    rgba = np.dstack([rgb, site.valid.astype(float)])

    fig, ax = frame(site, "Slope movement", "Displacement between the two surveys")
    ax.imshow(rgba, extent=site.extent, zorder=1)
    plt.rcParams["hatch.color"] = HATCH_GREY
    plt.rcParams["hatch.linewidth"] = 0.5
    ax.contourf(excluded.astype(float), levels=[0.5, 1.5], colors="none", hatches=["//////"],
                extent=site.extent, origin="upper", zorder=2)

    step = max(1, int(round(6.0 / site.cell)))
    X, Y = grid_xy(w.shape, site.cell)
    sl = (slice(step // 2, None, step), slice(step // 2, None, step))
    moving = significant & (np.sqrt(ux**2 + uy**2 + w**2) > MOVE_CM)
    sel = moving[sl]
    q = ax.quiver(X[sl][sel], Y[sl][sel], ux[sl][sel], uy[sl][sel], color=INK, angles="xy", scale_units="xy",
                  scale=1.0, width=0.003, headwidth=3.8, headlength=4, headaxislength=3.6, zorder=5)
    ax.quiverkey(q, 0.80, 0.215, 5, "", coordinates="figure", labelpos="E", color=INK)
    fig.text(0.825, 0.215, "5 cm horizontal", fontsize=9, va="center")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    side_colorbar(fig, sm, "Vertical\ndisplacement (cm)", extend="both")
    side_legend(fig, [
        Patch(fc="#E8E8E8", ec="none", label="Not significant (95%)"),
        Patch(fc="#E8E8E8", ec=HATCH_GREY, lw=0.5, hatch="//////", label="Earthworks (excluded)"),
        boundary_handle(),
    ])
    return fig


# --------------------------------------------------------------------------
# Output


def ensure_fonts():
    """Fetch static TTFs from Google Fonts once, then register them."""
    global TITLE_FONT
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for family, weights in FONTS.items():
        for weight in weights:
            path = FONT_DIR / f"{family}-{weight}.ttf"
            if not path.exists():
                try:
                    url = f"https://fonts.googleapis.com/css2?family={family}:wght@{weight}"
                    css = urllib.request.urlopen(url, timeout=30).read().decode()
                    ttf = re.search(r"url\((https://\S+?\.ttf)\)", css)
                    if not ttf:
                        raise OSError("no TTF link in the Google Fonts response")
                    path.write_bytes(urllib.request.urlopen(ttf.group(1), timeout=30).read())
                except OSError as exc:
                    print(f"warning: could not fetch {family} {weight}: {exc}", file=sys.stderr)
                    continue
            font_manager.fontManager.addfont(str(path))
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Unbounded" not in names:
        TITLE_FONT = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": "Outfit" if "Outfit" in names else "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })


def save_png(fig, path):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    data = buf.getvalue()
    try:
        import oxipng

        data = oxipng.optimize_from_memory(data, level=4)
    except ImportError:
        try:
            from PIL import Image

            out = io.BytesIO()
            Image.open(io.BytesIO(data)).save(out, format="PNG", optimize=True)
            data = out.getvalue()
        except ImportError:
            pass
    path.write_bytes(data)
    print(f"wrote {path} ({len(data) / 1024:.0f} KB)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--before", type=Path, help="GeoTIFF DEM of the first survey")
    p.add_argument("--after", type=Path, help="GeoTIFF DEM of the second survey")
    p.add_argument("--zones", type=Path, help="GeoTIFF of zone codes (see ZONES)")
    p.add_argument("--thermal", type=Path, help="GeoTIFF of surface temperature in °C")
    p.add_argument("--boundary", type=Path, help="GeoJSON site polygon, in the --before CRS")
    p.add_argument("--cell", type=float, default=CELL, help="grid spacing in metres (default %(default)s)")
    p.add_argument("--lod", type=float, default=LOD, help="level of detection in metres (default %(default)s)")
    p.add_argument("--seed", type=int, default=SEED, help="seed for the synthetic site")
    p.add_argument("--out", type=Path, default=OUT_DIR, help="output folder (default: the site's image folder)")
    p.add_argument("--write-synthetic", type=Path, metavar="DIR",
                   help="also save the synthetic rasters as GeoTIFFs (needs rasterio)")
    args = p.parse_args(argv)

    if bool(args.before) != bool(args.after):
        p.error("--before and --after go together")
    ensure_fonts()
    site = load_site(args) if args.before else synthetic_site(args.seed)
    if args.write_synthetic and site.synthetic:
        write_synthetic(site, args.write_synthetic)

    args.out.mkdir(parents=True, exist_ok=True)
    save_png(fig_change_map(site, args.lod), args.out / "change-map.png")
    if site.zones is not None:
        save_png(fig_zone_map(site), args.out / "zone-map.png")
    else:
        print("skipped zone-map.png: pass --zones", file=sys.stderr)
    if site.thermal is not None:
        save_png(fig_thermal(site), args.out / "thermal.png")
    else:
        print("skipped thermal.png: pass --thermal", file=sys.stderr)
    save_png(fig_slope(site), args.out / "slope.png")


if __name__ == "__main__":
    main()
