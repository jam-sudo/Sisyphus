"""CORS + rate-limit behavior for the engine API.

These tests are deliberately free of any Sisyphus *engine* import: the security
config lives in ``server.config`` and is exercised here against minimal FastAPI
apps wired exactly the way ``server.app`` wires them. That keeps the security
contract fast to verify without loading the (heavy) engine graph.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

# --- server.config: origin allow-list -------------------------------------

def test_allowed_origins_default_is_production_only(monkeypatch):
    monkeypatch.delenv("SISYPHUS_CORS_ORIGINS", raising=False)
    from server.config import allowed_origins

    assert allowed_origins() == ["https://sisyphus-pbpk.io"]


def test_allowed_origins_env_override_is_parsed(monkeypatch):
    monkeypatch.setenv(
        "SISYPHUS_CORS_ORIGINS", "https://sisyphus-pbpk.io, http://localhost:5173"
    )
    from server.config import allowed_origins

    assert allowed_origins() == [
        "https://sisyphus-pbpk.io",
        "http://localhost:5173",
    ]


def test_predict_rate_limit_default(monkeypatch):
    monkeypatch.delenv("SISYPHUS_PREDICT_RATE_LIMIT", raising=False)
    from server.config import predict_rate_limit

    assert predict_rate_limit() == "20/minute"


def test_predict_rate_limit_env_override(monkeypatch):
    monkeypatch.setenv("SISYPHUS_PREDICT_RATE_LIMIT", "5/minute")
    from server.config import predict_rate_limit

    assert predict_rate_limit() == "5/minute"


# --- CORS middleware wired with allowed_origins() --------------------------

def _cors_app() -> FastAPI:
    from server.config import allowed_origins

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/ping")
    def ping() -> dict:
        return {"ok": True}

    return app


def test_cors_allows_production_origin(monkeypatch):
    monkeypatch.delenv("SISYPHUS_CORS_ORIGINS", raising=False)
    client = TestClient(_cors_app())
    r = client.get("/ping", headers={"Origin": "https://sisyphus-pbpk.io"})
    assert r.headers.get("access-control-allow-origin") == "https://sisyphus-pbpk.io"


def test_cors_does_not_reflect_arbitrary_origin(monkeypatch):
    monkeypatch.delenv("SISYPHUS_CORS_ORIGINS", raising=False)
    client = TestClient(_cors_app())
    r = client.get("/ping", headers={"Origin": "https://evil.example.com"})
    acao = r.headers.get("access-control-allow-origin")
    assert acao != "https://evil.example.com"
    assert "evil.example.com" not in (acao or "")


# --- rate limiter wired the way server.app wires /predict ------------------

def _rate_limited_app(limit: str = "2/minute") -> FastAPI:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/ping")
    @limiter.limit(limit)
    def ping(request: Request) -> dict:
        return {"ok": True}

    return app


def test_rate_limit_returns_429_after_cap():
    client = TestClient(_rate_limited_app("2/minute"))
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 429
