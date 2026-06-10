"""CORS + rate-limit policy for the engine API.

Kept free of any Sisyphus *engine* import so the security contract is unit-
testable without loading the (heavy) engine graph. ``server.app`` imports the
helpers below and applies them to the live FastAPI app.

Both knobs are env-overridable for deployment without code changes:

* ``SISYPHUS_CORS_ORIGINS`` — comma-separated allow-list (e.g. add a localhost
  dev origin). Defaults to the production console origin only.
* ``SISYPHUS_PREDICT_RATE_LIMIT`` — slowapi limit string for ``POST /predict``
  (e.g. ``"10/minute"``). Defaults to ``"20/minute"`` per client IP.
"""
from __future__ import annotations

import os

# Production console origin (the GitHub Pages app served at sisyphus-pbpk.io/app).
DEFAULT_ALLOWED_ORIGINS = ("https://sisyphus-pbpk.io",)

# Per-client-IP cap on the ~5 s engine solve behind POST /predict.
DEFAULT_PREDICT_RATE_LIMIT = "20/minute"


def allowed_origins() -> list[str]:
    """CORS allow-list. Env override is comma-separated; blanks are dropped."""
    raw = os.environ.get("SISYPHUS_CORS_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_ALLOWED_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def predict_rate_limit() -> str:
    """slowapi limit string applied to POST /predict (per client IP)."""
    return os.environ.get("SISYPHUS_PREDICT_RATE_LIMIT", "").strip() or DEFAULT_PREDICT_RATE_LIMIT
