# Sai Dharshan — Portfolio

Personal portfolio website. Plain HTML, CSS and a small script; no build step.

## Structure

```
index.html                       the home page
projects/landfill-survey.html    detail page: Drone-Based Landfill Survey System
projects/counter-uas.html        detail page: High-Altitude Counter-UAS System
assets/css/tokens.css            colours, type families, spacing, radii, line weights
assets/css/components.css        reusable pieces (headings, buttons, stickers, cards, timeline…)
assets/css/site.css              page layout and responsive rules
assets/css/project.css           project detail page layout
assets/js/main.js                scroll fade-in, mobile menu, contact form
assets/img/favicon.svg
assets/img/projects/<project>/   project images (placeholders for now)
```

## Run locally

```sh
python3 -m http.server 8000
```

Then open http://localhost:8000.

## Deploy

The site is static, so it can be served from GitHub Pages: in the repository's
**Settings → Pages**, choose **Deploy from a branch**, branch `main`, folder `/ (root)`.

## Updating content

- **Email:** change the `mailto:` links in `index.html` and `CONTACT_EMAIL` in `assets/js/main.js`.
- **Portrait:** `assets/img/portrait.webp` is a transparent cut-out, bottom-aligned on the crimson
  block. To swap it, export a new cut-out at the same aspect (about 6:5) and update the `<img>`'s
  `width`/`height` in `index.html`.
- **Résumé:** add `assets/resume.pdf` and link it from the hero buttons.
- **Project images:** the detail pages use placeholder SVGs in `assets/img/projects/landfill/`
  and `assets/img/projects/counter-uas/`. Add the real image next to them (e.g. `hero.jpg`),
  then in the project page change that `<img>`'s `src`, `alt`, `width` and `height` to match.
- **New project:** add a card to the Projects section of `index.html`, copy one of the files in
  `projects/` for its detail page, and point the card's **View more** button at it.
