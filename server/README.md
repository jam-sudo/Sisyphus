# Sisyphus engine API

A small FastAPI service over the **real** Sisyphus engine that powers
arbitrary-SMILES predictions in the console (the Phase-2 "live tier"). It returns
a Drug entry in the exact shape the frontend's static data uses, so every view
works on a freshly-predicted compound.

## Endpoints

- `GET /health` → `{status, engine}`
- `POST /predict` `{ "smiles": "...", "dose_mg": 100, "route": "oral", "name": "…?" }`
  → full Drug entry (meta + conformal PI + 4 tracks + weights + disposition +
  engine curve + pkfit + real DDI folds + metadata). Invalid SMILES → `400`.

Scope: **predict + ddi** are computed live; **simulate** is derived client-side
from the returned `pkfit`/curve; **tdm/dose-adjust** are out of live scope (the
response carries a tdm placeholder the TDM view labels "illustrative").

## Run locally

```bash
# from the repo root, with the engine deps available
pip install -r server/requirements.txt          # fastapi + uvicorn
uvicorn server.app:app --port 8000 --workers 1   # http://127.0.0.1:8000

curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CC(C)Cc1ccc(cc1)C(C)C(=O)O","dose_mg":400,"name":"Ibuprofen"}'
```

Point the frontend at it: `cd web && VITE_API_URL=http://127.0.0.1:8000 npm run dev`.

> Run with **one worker**. The weight-capture path uses a module global (guarded
> by a lock); a single worker keeps predictions correct under concurrency.

## Deploy to Hugging Face Spaces (Docker, free)

1. Create a new **Space** → SDK: **Docker** → blank.
2. Add a root `Dockerfile` with the contents of [`server/Dockerfile`](./Dockerfile)
   (it clones this repo, installs deps, and serves on port 7860). Commit/push to
   the Space; HF builds it. First build takes a few minutes (rdkit/xgboost wheels).
3. Note the Space URL, e.g. `https://<user>-sisyphus.hf.space`. Verify
   `GET <url>/health` returns ok.
4. Wire the frontend to it and redeploy the console:
   ```bash
   cd web
   VITE_API_URL=https://<user>-sisyphus.hf.space npm run build:pages
   cd .. && git add app && git commit -m "chore(web): point console at live engine" && git push
   ```
   The console then enables the **✎ Custom SMILES** option (free-text SMILES →
   real prediction). Until then the public site runs the static preset tier.

### Security knobs (env-overridable, see `server/config.py`)
- **CORS** is restricted to `https://sisyphus-pbpk.io` (the production console
  origin) — arbitrary origins are no longer reflected. Add more origins (e.g. a
  localhost dev origin) via `SISYPHUS_CORS_ORIGINS` (comma-separated), no code
  change needed.
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
> engine-free `server-security` job (`server/tests/`, `server/config.py`).

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
