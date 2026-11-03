# Sisyphus engine API

A small FastAPI service over the **real** Sisyphus engine that powers
arbitrary-SMILES predictions in the console (the Phase-2 "live tier"). It returns
a Drug entry compatible with the prediction view. One core call supplies the
final `CmaxPrediction` and the separate coherent `EngineSimulation`.

## Endpoints

- `GET /health` → `{status, version, profile, primary_output}`
- `GET /model-info` → versioned scientific model card and evidence status
- `POST /predict` `{ "smiles": "...", "dose_mg": 100, "route": "oral", "name": "…?" }`
  → Meta Cmax + residual interval + tracks/weights + source-labelled engine
  endpoints/curve + artifact provenance. Invalid SMILES → `400`.

`disposition.doseOverAuc0t` is dose divided by the engine's 0–24h AUC, not
terminal CL/F. A failed or missing engine solve returns `500`.

Scope: only structure-only Cmax prediction is supported live. DDI, TDM, and
dose recommendation are not part of this API contract.

## Run locally

```bash
# from the repo root, with the engine deps available
pip install -r server/requirements.txt
pip install -e . --no-deps
uvicorn server.app:app --port 8000 --workers 1   # http://127.0.0.1:8000

curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CC(C)Cc1ccc(cc1)C(C)C(=O)O","dose_mg":400,"name":"Ibuprofen"}'
```

Point the frontend at it: `cd web && VITE_API_URL=http://127.0.0.1:8000 npm run dev`.

The server has no global weight-capture state and may use multiple workers,
subject to memory limits for the loaded ML artifacts.

## Deploy to Hugging Face Spaces (Docker, free)

The live console uses the Space
[`jam-sudo/sisyphus`](https://huggingface.co/spaces/jam-sudo/sisyphus)
at `https://jam-sudo-sisyphus.hf.space`.

1. The Space holds only a `README.md` (`sdk: docker`, `app_port: 7860`) and a
   [`Dockerfile`](https://huggingface.co/spaces/jam-sudo/sisyphus/blob/main/Dockerfile).
   That Dockerfile fetches one pinned commit of this repository and runs the
   same install steps as [`server/Dockerfile`](./Dockerfile), which CI builds
   and smoke-tests.
2. To deploy a newer engine, change `SISYPHUS_COMMIT` in the Space Dockerfile to
   the target commit on `main`. HF rebuilds on push; the first build takes a few
   minutes (rdkit/xgboost wheels).
3. Keep the Space variable `SISYPHUS_TRUST_PROXY=1` (see below).
4. Verify `GET <url>/health` returns `"version": "0.4.x"`. The console only
   enables live prediction when the major.minor version matches.

The console's backend URL is committed in `web/.env.production`, so
`npm run build:pages` produces the same `/app` bundle locally and in CI's
bundle-sync check. Do not pass `VITE_API_URL` on the command line for a
production build. `npm run dev` ignores that file; set `VITE_API_URL` there to
use a local server.

### Security knobs (env-overridable, see `server/config.py`)
- **CORS** is restricted to `https://sisyphus-pbpk.io` (the production console
  origin) — arbitrary origins are no longer reflected. Add more origins (e.g. a
  localhost dev origin) via `SISYPHUS_CORS_ORIGINS` (comma-separated), no code
  change needed. **On Hugging Face Spaces this allow-list is not effective:** the
  `*.hf.space` proxy adds a reflected `Access-Control-Allow-Origin` to every
  response, so any site can call the API from a browser. The API carries no
  cookies or credentials, so this exposes compute, not user data; the per-client
  rate limit still applies. Hosts that pass the app's headers through, such as
  Cloud Run, enforce the allow-list.
- **Rate limiting:** `POST /predict` is capped per client (default
  `20/minute`) via slowapi; over-cap requests get `429`. Tune with
  `SISYPHUS_PREDICT_RATE_LIMIT` (e.g. `10/minute`, read once at startup).
  `GET /health` is unlimited.
- **Behind a reverse proxy (HF Spaces, Cloud Run): set `SISYPHUS_TRUST_PROXY=1`.**
  Otherwise the rate-limit key is the socket peer — which is the *proxy*, identical
  for every external caller — so the per-client cap collapses into one shared
  bucket (all users throttled together; one abuser can exhaust it for everyone).
  With it on, the key is the rightmost `X-Forwarded-For` hop (the address the
  trusted proxy saw, not spoofable by the client). Assumes exactly one proxy hop.
  Alternative: run uvicorn with `--proxy-headers --forwarded-allow-ips=<proxy>`.

> The security contract (CORS allow-list + rate-limit key) is gated in CI by the
> `server-security` job (`server/tests/`, `server/config.py`).

### Notes / before a public, uncapped deploy
- The free Space sleeps when idle; the first request after sleep cold-starts
  (~30–60 s to wake + load models). The frontend handles this with a loading state.
- The rate-limit counter is slowapi's in-memory store: it resets on each
  cold-start and is per-worker, so the effective cap is looser than the nominal
  string under sleep/multi-worker. Use a shared backend (e.g. Redis) for a
  durable, multi-instance cap.
- Still **no auth** — add a token/key before exposing a fully public, uncapped
  endpoint (the rate limit above bounds abuse but does not authenticate callers).
- Cloud Run (scale-to-zero, custom `api.sisyphus-pbpk.io`) is an alternative host;
  the same image works (`docker build -f server/Dockerfile .`).
