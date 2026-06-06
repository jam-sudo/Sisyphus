# Live engine API — arbitrary-SMILES predictions (Phase 2)

**Date:** 2026-06-05
**Goal:** Let the console predict ANY SMILES (not just the 8 presets) by wiring it
to the real engine through a small FastAPI backend. Scope **(b): predict +
simulate + ddi** live. Host on **Hugging Face Spaces (Docker, free)**. Build +
verify locally now; public deploy when the user creates the Space (needs their
account). Presets remain as quick-start examples + offline fallback.

## Why a backend (constraint)

Arbitrary SMILES needs RDKit + XGBoost + scipy ODE + conformal calibration — real
Python. GitHub Pages is static; Pyodide can't run XGBoost/RDKit/torch well. So a
Python backend is required. The frontend's `EngineClient` seam (data.ts) was built
for exactly this swap.

## Backend (`server/`)

FastAPI app reusing the verified entry-building logic from
`scripts/gen_console_data.py` (imported: `build_drug_and_graph`, `solve_single`,
`downsample`, `fit_pk`, `ddi_folds`, `_CAPTURE` weights, `CONFORMAL_FACTOR`; the
`MetaLearner.combine` weight-capture monkeypatch applies on import) plus
`pipeline.predict`.

- `GET /health` → `{status, engine}`.
- `POST /predict` `{smiles, dose_mg, route="oral", name?}` → a **full Drug entry**
  matching `console_data.drugs[i]` exactly (so every frontend view works unchanged):
  meta, cmax90ci (conformal), tracks, weights (from `_CAPTURE`), disposition,
  curve (engine ODE, scaled so peak == engine-track Cmax — same reconcile as the
  gen script), pkfit, ddi (real 4-perpetrator folds), tdm (placeholder — see
  below), plus metadata: formula (RDKit `CalcMolFormula`), mw, type
  (`compound_type`), primaryEnzyme + enzymeFraction (from `drug.enzyme_affinity`,
  normalized), confidence, inDomain, adFlags.
- **simulate** is derived client-side from the returned `pkfit`/curve (SimulateView
  already does this) — no endpoint needed. **ddi** folds come from `/predict`.
- **tdm/dose-adjust** are out of live scope (real Bayesian update is 12–100 s).
  `/predict` returns a `tdm` placeholder (`method:"ibis"`, cv fields null); the TDM
  view already labels its shrink "illustrative — full re-inference needs the live
  engine," so this stays honest.
- Invalid SMILES → `400`. CORS allows the Pages origin (and `*` in dev). Base graph
  built once at startup. **Single worker** (the `_CAPTURE` weight global is not
  thread-safe) — fine for low traffic; revisit with a lock if scaled.

## Frontend

- `EngineClient` gains `predict(smiles, dose, route, name?): Promise<Drug>`.
  `StaticEngineClient.predict` throws (not supported); a new `ApiEngineClient`
  (or a composite) implements it against `${VITE_API_URL}/predict`. Static
  `load()` still provides benchmark/inhibitors/constants/presets.
- **Liveness:** `VITE_API_URL` build env; on load, ping `/health`. If reachable →
  enable custom-SMILES; else static-only with a small "live engine offline" note.
  Empty `VITE_API_URL` (current Pages build) = static-only until the Space exists.
- **UX (predict rail):** a SMILES **text input** (prefilled from the selected
  preset) + a presets dropdown to load examples + a name field (optional). The Run
  button becomes functional in custom mode → calls `/predict`, shows the spinner,
  and on success sets the result as the current drug (one reused "custom" slot).
  Errors (invalid SMILES / backend down) surface inline. Dose/route reuse existing
  inputs. Non-predict workflows operate on the returned drug like a preset.

## Hosting (HF Spaces, Docker)

`server/Dockerfile` (build context = repo root) installs locked deps + fastapi +
uvicorn and runs `uvicorn server.app:app --port 7860`. `server/README.md`: create a
Docker Space, set it to this repo/Dockerfile, note the `*.hf.space` URL, then set
`VITE_API_URL` to it and rebuild/redeploy the frontend. Deploy itself needs the
user's HF account → deferred.

## Verification

- Backend (local, uvicorn): `GET /health` ok; `POST /predict` for a **non-preset**
  SMILES (e.g. ibuprofen `CC(C)Cc1ccc(cc1)C(C)C(=O)O`) returns a valid full entry
  (meta.cmax>0, 120-pt curve, 4 ddi folds, weights sum→1); invalid SMILES → 400.
- Frontend (local, `vite dev` + `VITE_API_URL=http://127.0.0.1:8000`): type a
  non-preset SMILES → real prediction renders across predict/simulate/ddi; stop
  backend → graceful static fallback; 0 console errors.

## Out of scope

- Live tdm/dose-adjust (slow; later, job-based). Benchmark stays static. No
  auth/rate-limiting yet (add before a public, uncapped deploy). No change to the
  landing page.
