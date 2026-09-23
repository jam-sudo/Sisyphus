"""FastAPI adapter over the structured Sisyphus prediction contracts.

The API performs exactly one core prediction. It does not import experiment
scripts, monkeypatch the meta learner, change cwd, or rescale a second curve.
"""

from __future__ import annotations

import json
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from rdkit.Chem import MolFromSmiles, rdMolDescriptors
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from sisyphus import __version__
from sisyphus.pipeline.predict import predict
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.resources import get_resource_config

from .config import allowed_origins, client_ip, predict_rate_limit


def _sig(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else float(f"{float(value):.{digits}g}")


def _downsample(
    time_h: tuple[float, ...], concentration: tuple[float, ...], n: int = 120
) -> tuple[list[float], list[float]]:
    if len(time_h) <= n:
        return list(time_h), list(concentration)
    idx = np.unique(np.linspace(0, len(time_h) - 1, n, dtype=int))
    return [time_h[i] for i in idx], [concentration[i] for i in idx]


def _pkfit(tmax: float | None, half_life: float | None) -> dict[str, float | None]:
    ke = np.log(2.0) / half_life if half_life and half_life > 0 else None
    if not tmax or tmax <= 0 or not ke:
        return {"ka": None, "ke": _sig(ke), "thalf": _sig(half_life)}
    candidates = np.logspace(np.log10(ke * 1.01), 2, 2000)
    implied = np.log(candidates / ke) / (candidates - ke)
    ka = float(candidates[int(np.argmin(np.abs(implied - tmax)))])
    return {"ka": _sig(ka), "ke": _sig(ke), "thalf": _sig(half_life)}


limiter = Limiter(key_func=client_ip)
app = FastAPI(title="Sisyphus structure-only Cmax API", version=__version__)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smiles: str = Field(min_length=1, max_length=2000)
    dose_mg: float = Field(default=100.0, gt=0, le=100000, allow_inf_nan=False)
    route: Literal["oral"] = "oral"
    name: str | None = Field(default=None, max_length=200)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "profile": get_resource_config().profile,
        "primary_output": "CmaxPrediction",
    }


@app.get("/model-info")
def model_info() -> dict:
    path = get_resource_config().data("model_card.json")
    return json.loads(path.read_text())


@app.post("/predict")
@limiter.limit(predict_rate_limit())
def do_predict(request: Request, req: PredictRequest) -> dict:
    smiles = (req.smiles or "").strip()
    mol = MolFromSmiles(smiles)
    if not smiles or mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES.")
    if mol.GetNumAtoms() > 300:
        raise HTTPException(status_code=400, detail="SMILES exceeds the 300-atom limit.")
    dose = float(req.dose_mg)
    if not np.isfinite(dose) or dose <= 0:
        raise HTTPException(status_code=400, detail="Dose must be positive and finite.")
    route = req.route
    try:
        result = predict(smiles, dose, route=route, n_mc_samples=0, strict=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    cmax = result.cmax_prediction
    engine = result.engine_simulation
    if cmax is None:
        raise HTTPException(status_code=500, detail="Cmax prediction unavailable.")
    if engine is None or not engine.solver_success:
        raise HTTPException(status_code=500, detail="Engine prediction unavailable.")

    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    track_values = dict(cmax.tracks)
    weights = dict(cmax.weights)

    if engine is not None:
        curve_t, curve_c = _downsample(engine.time_h, engine.concentration_mg_l)
        endpoints = engine.endpoints
        engine_tmax = endpoints.tmax.mean
        engine_auc = endpoints.auc_0t.mean
        engine_half = endpoints.t_half.mean if endpoints.t_half else None
    else:
        curve_t, curve_c = [], []
        engine_tmax = engine_auc = engine_half = None

    formula = rdMolDescriptors.CalcMolFormula(mol)
    return {
        "id": "custom",
        "name": (req.name or "Custom compound").strip() or "Custom compound",
        "formula": formula,
        "mw": _sig(profile.mw),
        "smiles": smiles,
        "type": profile.compound_type,
        "dose": dose,
        "route": route,
        "primaryEnzyme": "not_reported",
        "enzymeFraction": {},
        "confidence": result.confidence,
        "applicabilityStatus": (
            "structurally_in_scope" if result.in_applicability_domain else "flagged"
        ),
        "inDomain": bool(result.in_applicability_domain),
        "adFlags": list(result.ad_flags),
        "executionStatus": result.execution_status,
        "artifactProvenance": dict(result.artifact_provenance),
        "meta": {
            "cmax": _sig(cmax.cmax.mean),
            "tmax": _sig(engine_tmax),
            "auc": _sig(engine_auc),
            "thalf": _sig(engine_half),
        },
        "endpointSources": {
            "cmax": cmax.method,
            "tmax": "engine" if engine is not None else None,
            "auc": "engine" if engine is not None else None,
            "thalf": "engine" if engine is not None else None,
        },
        "cmax90ci": (
            [_sig(cmax.interval_90[0]), _sig(cmax.interval_90[1])]
            if cmax.interval_90
            else None
        ),
        "intervalSource": cmax.interval_source,
        "residualInterval90": (
            [_sig(cmax.residual_interval_90[0]), _sig(cmax.residual_interval_90[1])]
            if cmax.residual_interval_90
            else None
        ),
        "residualIntervalSource": cmax.residual_interval_source,
        "parameterInterval90": (
            [_sig(cmax.parameter_interval_90[0]), _sig(cmax.parameter_interval_90[1])]
            if cmax.parameter_interval_90
            else None
        ),
        "parameterIntervalSource": cmax.parameter_interval_source,
        "tracks": {k: _sig(track_values.get(k)) for k in ("engine", "ml", "clf", "vdss")},
        "weights": {k: _sig(weights.get(k)) for k in ("engine", "ml", "clf", "vdss")},
        "disposition": {
            "doseOverAuc0t": _sig(dose / engine_auc) if engine_auc and engine_auc > 0 else None,
            "vdss": _sig(adme.vdss.mean),
            "fup": _sig(adme.fup.mean),
            "clint": _sig(adme.clint.mean),
        },
        "curve": {
            "t": [_sig(v, 5) for v in curve_t],
            "c": [_sig(v, 5) for v in curve_c],
        },
        "pkfit": _pkfit(engine_tmax, engine_half),
        "engineDiagnostics": (
            {
                "observationNode": engine.observation_node,
                "solverSuccess": engine.solver_success,
                "massBalanceError": _sig(engine.mass_balance_error),
            }
            if engine is not None
            else None
        ),
    }
