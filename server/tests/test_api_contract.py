"""Production HTTP input contract is oral-only and resource bounded."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from server.app import app


def test_predict_rejects_iv_before_running_engine():
    client = TestClient(app)
    response = client.post(
        "/predict", json={"smiles": "CCO", "dose_mg": 10.0, "route": "iv"}
    )
    assert response.status_code == 422


def test_predict_rejects_nonpositive_dose_and_extra_fields():
    client = TestClient(app)
    assert client.post(
        "/predict", json={"smiles": "CCO", "dose_mg": 0, "route": "oral"}
    ).status_code == 422
    assert client.post(
        "/predict",
        json={"smiles": "CCO", "dose_mg": 10, "route": "oral", "unknown": True},
    ).status_code == 422


def test_predict_rejects_oversized_smiles():
    client = TestClient(app)
    response = client.post(
        "/predict", json={"smiles": "C" * 2001, "dose_mg": 10, "route": "oral"}
    )
    assert response.status_code == 422


def test_predict_rejects_missing_engine_even_with_meta_cmax(monkeypatch):
    monkeypatch.setattr(
        "server.app.predict",
        lambda *args, **kwargs: SimpleNamespace(
            cmax_prediction=SimpleNamespace(), engine_simulation=None,
        ),
    )
    response = TestClient(app).post(
        "/predict", json={"smiles": "CCO", "dose_mg": 10, "route": "oral"}
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Engine prediction unavailable."


def test_predict_reports_24h_exposure_proxy_as_such():
    response = TestClient(app).post(
        "/predict", json={"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "dose_mg": 100}
    )
    assert response.status_code == 200
    body = response.json()
    assert "clf" not in body["disposition"]
    assert body["disposition"]["doseOverAuc0t"] == pytest.approx(
        body["dose"] / body["meta"]["auc"], rel=1e-5
    )
