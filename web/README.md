# Sisyphus PBPK Console

A scientific-editorial web interface for the Sisyphus PBPK platform — turns the
CLI into a clickable clinical tool. Built from the design handoff
("Direction B — Workspace Console") and **wired to the real Sisyphus engine**.

Six workflows, all driven by real-engine numbers:

| Workflow      | What it shows                                                              |
| ------------- | ------------------------------------------------------------------------- |
| `predict`     | SMILES → C(t) curve (real ODE), 4-track meta-learner, body graph, conformal 90% PI, pipeline log |
| `simulate`    | multi-dose superposition, steady-state metrics, FDA-label check           |
| `tdm`         | Bayesian prior→posterior, SBI/IBIS/IS routing, ESS / CV-reduction         |
| `ddi`         | victim ± perpetrator, **real** AUC/Cmax fold changes, enzyme mechanism    |
| `dose-adjust` | MIPD dose recommendation to a target Cₛₛ                                   |
| `benchmark`   | real N=107 holdout scatter + per-track AAFE (Meta 2.784)                   |

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
npm run smoke      # headless jsdom render test of all 6 workflows
```

## Data: real engine, two tiers

The console reads `public/data/console_data.json` — **every numeric field is a
genuine Sisyphus engine solve / `pipeline.predict` call, not a mock.** It is
produced offline by:

```bash
# from the repo root, with the locked deps available
/opt/miniconda3/bin/python scripts/gen_console_data.py
```

This regenerates `web/public/data/console_data.json` (curated drug set + DDI
matrix + TDM + steady-state) and `web/public/data/benchmark.json` (the N=107
holdout). Re-run it whenever the engine changes.

The data layer (`src/data.ts`) talks to the engine through an `EngineClient`
interface:

- **Phase 1 (now):** `StaticEngineClient` reads the pre-computed JSON. Works on
  GitHub Pages with zero backend.
- **Phase 2 (later):** an `ApiEngineClient` implementing the same interface will
  hit a FastAPI service wrapping `pipeline.predict` for arbitrary SMILES. The
  views import only the interface, so swapping clients needs no UI changes.
  (A dev proxy hook is already stubbed in `vite.config.ts`.)

Curves use the real ODE single-dose response; multi-dose / TDM / dose-adjust
use superposition of a 1-compartment fit to that curve (exact for linear PK);
DDI folds are the real `apply_inhibition` / `apply_induction` re-solves. The
honesty stays intact: the conformal 90% PI (÷×~13) and the calibration caveats
are surfaced, not hidden.

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
  pk.ts                   # PK math (real-anchored curves / superposition / DDI)
  styles.css              # the scientific-editorial design system
  components/
    App.tsx               # rail + nav + run flow + state
    RailInputs.tsx        # contextual per-workflow inputs
    charts.tsx            # ConcChart, ScatterChart, McHistogram, TroughChart
    panels.tsx            # Stat, TrackBars, BodyGraph, Pill, Legend, …
    workflows/            # PredictView, SimulateView, TdmView, DdiView,
                          #   DoseAdjustView, BenchmarkView + dispatcher
public/data/              # real-engine JSON (generated)
scripts/smoke.mjs         # headless render smoke test
```
