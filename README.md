# Sai Dharshan — Portfolio

Personal portfolio website. Plain HTML, CSS and a small script; no build step.

## Structure

```
index.html              the page
assets/css/tokens.css   colours, type families, spacing, radii, line weights
assets/css/components.css  reusable pieces (headings, buttons, stickers, cards, timeline…)
assets/css/site.css     page layout and responsive rules
assets/js/main.js       scroll fade-in, mobile menu, contact form
assets/img/favicon.svg
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
- **Portrait:** replace the silhouette inside `.photo` with a cut-out image, e.g.
  `<img src="assets/img/portrait.png" alt="Sai Dharshan" style="width:100%;height:100%;object-fit:contain;object-position:bottom">`.
- **Résumé:** add `assets/resume.pdf` and link it from the hero buttons.
