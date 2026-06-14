"""CORS + rate-limit policy for the engine API.

Kept free of any Sisyphus *engine* (and slowapi/fastapi) import so the security
contract is unit-testable without loading the heavy engine graph or the web
stack. ``server.app`` imports the helpers below and applies them to the live
FastAPI app.

Env-overridable knobs (deployment without code changes):

* ``SISYPHUS_CORS_ORIGINS`` — comma-separated allow-list (e.g. add a localhost
  dev origin). Defaults to the production console origin only. An all-blank
  value falls back to the default rather than blocking every origin.
* ``SISYPHUS_PREDICT_RATE_LIMIT`` — slowapi limit string for ``POST /predict``
  (e.g. ``"10/minute"``). Read once at import time (decorator evaluation), so
  set it **before the process starts**. Defaults to ``"20/minute"`` per client.
* ``SISYPHUS_TRUST_PROXY`` — when truthy ("1"/"true"/"yes"/"on"), derive the
  rate-limit client key from the rightmost ``X-Forwarded-For`` hop (the address
  the trusted reverse proxy observed) instead of the socket peer. **Required
  behind a proxy** (HF Spaces, Cloud Run): there the socket peer is the proxy
  and is identical for every external caller, so per-client limiting would
  otherwise collapse to a single shared bucket. Assumes exactly one trusted
  proxy hop. Defaults off (use the socket peer). Alternative: run uvicorn with
  ``--proxy-headers --forwarded-allow-ips=<proxy>`` so the peer is rewritten.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing the web stack at module load
    from starlette.requests import Request

# Production console origin (the GitHub Pages app served at sisyphus-pbpk.io/app).
DEFAULT_ALLOWED_ORIGINS = ("https://sisyphus-pbpk.io",)

# Per-client cap on the ~5 s engine solve behind POST /predict.
DEFAULT_PREDICT_RATE_LIMIT = "20/minute"

_TRUTHY = {"1", "true", "yes", "on"}


def allowed_origins() -> list[str]:
    """CORS allow-list. Env override is comma-separated; blanks are dropped.

    An env value that parses to nothing (all blanks/commas) falls back to the
    default rather than returning ``[]`` — which would block every origin,
    including the production console (a silent foot-gun).
    """
    raw = os.environ.get("SISYPHUS_CORS_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_ALLOWED_ORIGINS)
    parsed = [o.strip() for o in raw.split(",") if o.strip()]
    return parsed or list(DEFAULT_ALLOWED_ORIGINS)


def predict_rate_limit() -> str:
    """slowapi limit string applied to POST /predict (per client key)."""
    return os.environ.get("SISYPHUS_PREDICT_RATE_LIMIT", "").strip() or DEFAULT_PREDICT_RATE_LIMIT


def trust_proxy() -> bool:
    """Whether to derive the rate-limit key from X-Forwarded-For (see module doc)."""
    return os.environ.get("SISYPHUS_TRUST_PROXY", "").strip().lower() in _TRUTHY


def client_ip(request: Request) -> str:
    """Rate-limit key: the real client IP.

    Behind a trusted reverse proxy (``SISYPHUS_TRUST_PROXY`` on), the socket
    peer is the proxy and is identical for every external caller, so keying on
    it would rate-limit all users as one bucket. We instead take the
    **rightmost** ``X-Forwarded-For`` entry — the address the single trusted
    proxy saw, which a client cannot spoof by sending its own XFF header (the
    proxy appends the observed peer to the right). Falls back to the socket peer
    when the header is absent or proxy-trust is off. Mirrors slowapi's
    ``get_remote_address`` for the fallback, without importing slowapi here.
    """
    if trust_proxy():
        xff = request.headers.get("x-forwarded-for", "")
        hops = [h.strip() for h in xff.split(",") if h.strip()]
        if hops:
            return hops[-1]
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "127.0.0.1"
