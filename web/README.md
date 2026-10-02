# Sisyphus PBPK Console

A scientific-editorial web interface for the Sisyphus structure-only Cmax core.
It is a research console, not a clinical dosing tool. Built from the design handoff
("Direction B — Workspace Console") and **wired to the real Sisyphus engine**.

Two supported workflows, driven by real model artifacts:

| Workflow      | What it shows                                                              |
| ------------- | ------------------------------------------------------------------------- |
| `predict`     | SMILES → C(t) curve (real ODE), 4-track meta-learner, body graph, development-residual 90% interval, pipeline log |
| `benchmark`   | N=107 retrospective development scatter + per-track AAFE; explicitly not an independent holdout |

## Stack

- **Vite + React 18 + TypeScript**, hand-rolled SVG charts (no chart library —
  preserves the design's typographic look).
- **No backend (Phase 1).** All numbers are pre-computed by the real engine and
  read from `public/data/console_data.json`. See the data layer below.

## Develop

```bash
cd web
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production bundle → dist/
npm run smoke      # headless jsdom render test of both workflows
```

## Data: real engine, two tiers

The console reads `public/data/console_data.json` — **every numeric field is a
genuine Sisyphus engine solve / `pipeline.predict` call, not a mock.** It is
produced offline by:

```bash
# from the repo root, with the locked deps available
/opt/miniconda3/bin/python scripts/gen_console_data.py
```

This regenerates `web/public/data/console_data.json` (curated core predictions)
and `web/public/data/benchmark.json` (the N=107 development evidence). Each
preset uses one strict prediction; its Cmax contract and matching engine curve
are serialized without a second solve or curve rescaling.

The data layer (`src/data.ts`) talks to the engine through an `EngineClient`
interface:

- **Static tier:** `SisyphusClient` reads the pre-computed JSON. Works on
  GitHub Pages with zero backend.
- **Live tier:** when `VITE_API_URL` is configured, the same client calls the
  FastAPI core for arbitrary SMILES.

Curves use the matching ODE single-dose response. The empirical development
90% residual interval (÷×~10.24; 91.1% coverage on the repeatedly used development set) and its calibration caveat are surfaced. Regimen,
TDM, DDI, and dose-adjustment workflows are deliberately outside this console.

## Deploy (GitHub Pages — sisyphus-pbpk.io/app/)

The console is served at **`/app/`**, alongside the Jekyll homepage at `/`, with
no change to the Pages source. The bundle uses `base: "./"` (path-agnostic), so
the build is committed to the **repo-root `/app/` folder** and the existing
branch-based Jekyll Pages deployment serves it verbatim. The root `CNAME` keeps
the custom domain.

**To update the deployed console:**

```bash
cd web
npm run build:pages   # build + sync web/dist → ../app
cd .. && git add app && git commit -m "chore(web): rebuild console" && git push
```

`.github/workflows/web-ci.yml` build- and smoke-tests `web/` on every PR/push so
broken bundles never land. (It does not deploy — `/app/` is the committed build.)

## Layout

```
src/
  types.ts                # data contracts (mirror gen_console_data.py output)
  data.ts                 # EngineClient + useConsoleData() hook
  pk.ts                   # core Cmax/engine-curve display helpers
  styles.css              # the scientific-editorial design system
  components/
    App.tsx               # rail + nav + run flow + state
    RailInputs.tsx        # contextual per-workflow inputs
    charts.tsx            # concentration and development-scatter charts
    panels.tsx            # Stat, TrackBars, BodyGraph, Pill, Legend, …
    workflows/            # supported PredictView + BenchmarkView only
public/data/              # real-engine JSON (generated)
scripts/smoke.mjs         # headless render smoke test
```
