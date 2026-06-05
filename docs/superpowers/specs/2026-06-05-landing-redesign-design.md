# Landing page redesign — scientific-editorial (A→C)

**Date:** 2026-06-05
**Goal:** Replace the plain Primer-theme homepage at `sisyphus-pbpk.io/` with a
custom landing that matches the Sisyphus Console's scientific-editorial design
system, is product-forward up top and research-credible below, and fixes the
homepage's broken/raw links. The console stays at `/app/`.

## Approach

A **self-contained static `index.html`** at repo root, hand-authored with the
console's design tokens **inlined** (no build step, no framework). It has no
Jekyll front matter, so GitHub Pages serves it verbatim at `/` — full pixel
control. `index.md` is removed (avoids the index.html/index.md conflict).
`_config.yml` stays (harmless; theme does not apply to a front-matter-less page).
Fully responsive / mobile-first (unlike the console's fixed 1280×800 canvas).

Rejected: (a) custom Jekyll layout + markdown content — too little layout control;
(b) a second Vite entry — multi-entry/base-path complexity for a static page.

## Design system (reused from web/src/styles.css)

- Palette (oklch): paper `0.991 .004 95`, ink `0.24 .012 255`, blue `0.52 .11 248`,
  teal `0.55 .085 178`, clay `0.60 .105 52`, hairlines, soft tints. Light theme.
- Type: Newsreader (serif, headings/figures), IBM Plex Sans (UI), IBM Plex Mono
  (labels/data/code). Google Fonts.
- Motifs: brand mark = dark rounded square + clay dot; hairline-bordered panels
  with 12px radius; mono uppercase tracked labels; serif display numerals.

## Page structure

1. **Top bar** (sticky): brand mark + "Sisyphus" wordmark · nav (Console ·
   Validation · GitHub) · primary button `Launch console →` (`/app/`).
2. **Hero**: serif display headline + one-line subhead (the tagline); a mono stat
   row — **AAFE 2.78 · N=107 holdout · 43.9% within 2-fold · 34-compartment ODE**;
   CTAs `Launch the console →` (`/app/`) + `View on GitHub`; a framed **real
   console screenshot** (`docs/figures/console-predict.png`).
3. **Three ideas** (3 cards): the body is a graph · everything is a Distribution ·
   the engine knows types, not identities.
4. **Six workflows** (card grid, links to `/app/`): predict (SMILES→PK) · simulate
   (multi-dose/steady state) · tdm (Bayesian TDM) · ddi (interactions) ·
   dose-adjust (MIPD) · benchmark (N=107 holdout).
5. **Validation**: the scatter figure (`docs/figures/figure2_scatter.png`) + a
   compact AAFE table (Meta 2.78 / Engine 4.46 / ML 3.01; in-domain 2.83) with the
   honest framing (prospective 3.27 > retrospective — generalization is harder).
6. **Calibration / honesty callout**: the user-facing 90% PI is split-conformal,
   holdout-validated to 0.953 coverage at nominal 0.90 — but wide (÷×~13), the
   honest price of structural error.
7. **Quickstart**: code panel — `pip install -e ".[dev,ml,chem]"` and
   `sisyphus predict --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O" --dose 100`.
8. **Footer**: Console · Repository · Preprint PDF · docs (Reproducibility · SBI
   results · Engine advantage) · scope disclaimer · "open source".

## Link fixes folded in

- **Preprint 404** → commit `Sisyphus_Preprint.pdf` (no public DOI yet; bioRxiv
  declined, ChemRxiv unposted) and link it locally.
- **Raw-markdown doc links** (README/docs served as `text/markdown`) → repoint to
  GitHub **blob** URLs (`github.com/jam-sudo/Sisyphus/blob/main/...`) which render.
- Keep the `/app/` console link.

## Assets

- New: `docs/figures/console-predict.png` — clean predict-view screenshot (captured
  via Playwright from the live/local console).
- Reuse: `docs/figures/figure2_scatter.png`.

## Verification (before claiming done)

- Serve repo root locally; Playwright screenshots at desktop (~1280) and mobile
  (~390); assert 0 console errors and that the hero/sections/figure render.
- Check every link: `/app/` 200, preprint PDF 200, figure 200, GitHub blob 200.
- Commit (index.html, removed index.md, PDF, screenshot, spec) → push → confirm the
  live homepage renders and links resolve.

## Out of scope

- No change to the console (`web/`, `/app/`).
- No JS beyond anchor links (static page).
- Not adding front matter to the raw `.md` files (blob links supersede them).
