"""
Sisyphus engine API — arbitrary-SMILES predictions for the console.

A thin FastAPI layer over the REAL engine. It reuses the verified entry-building
helpers from scripts/gen_console_data.py (importing it also installs the
MetaLearner.combine weight-capture monkeypatch and exposes _CAPTURE) plus
pipeline.predict, and returns a Drug entry in EXACTLY the shape the frontend's
static console_data.json uses — so every view works unchanged.

Scope: predict + ddi computed live; simulate is derived client-side from the
returned pkfit/curve; tdm/dose-adjust are out of live scope (the response carries
a tdm placeholder the TDM view already labels "illustrative").

Run locally:  uvicorn server.app:app --port 8000   (from the repo root)
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)  # engine resolves data/model paths relative to the repo root

import gen_console_data as gcd  # noqa: E402  (helpers + weight-capture monkeypatch + _CAPTURE)
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from rdkit.Chem import MolFromSmiles, rdMolDescriptors  # noqa: E402
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402

import sisyphus.engine.flux  # noqa: E402,F401  (registers flux specs)
from sisyphus.graph.builder import build_from_yaml  # noqa: E402
from sisyphus.pipeline.predict import predict  # noqa: E402
from sisyphus.pk.endpoints import compute_endpoints  # noqa: E402

from .config import allowed_origins, client_ip, predict_rate_limit  # noqa: E402

_trapz = getattr(np, "trapezoid", np.trapz)
_SIG = gcd._sig
_SIG_LIST = gcd._sig_list
_LOCK = threading.Lock()  # _CAPTURE is a module global; serialize predicts

# build the physiology graph once
BASE_GRAPH = build_from_yaml(ROOT / "data/physiology/reference_man.yaml")

# Per-client rate limiter (slowapi) — guards the heavy /predict solve. The key
# is config.client_ip (X-Forwarded-For-aware behind a trusted proxy; see
# server/config.py SISYPHUS_TRUST_PROXY) so the cap is per real client, not
# per reverse-proxy IP.
limiter = Limiter(key_func=client_ip)

app = FastAPI(title="Sisyphus engine API", version="0.4")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),  # production console origin(s) only (SISYPHUS_CORS_ORIGINS)
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    smiles: str
    dose_mg: float = 100.0
    route: str = "oral"
    name: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "Sisyphus v0.4 post-FLUX-1"}


@app.post("/predict")
@limiter.limit(predict_rate_limit())
def do_predict(request: Request, req: PredictRequest) -> dict:
    smiles = (req.smiles or "").strip()
    if not smiles or MolFromSmiles(smiles) is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES.")
    dose = float(req.dose_mg)
    if not (dose > 0):
        raise HTTPException(status_code=400, detail="Dose must be positive.")
    route = (req.route or "oral").lower()
    if route not in ("oral", "iv"):
        route = "oral"
    try:
        with _LOCK:
            return _build_entry(smiles, dose, route, req.name)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"engine error: {e}") from e


def _build_entry(smiles: str, dose: float, route: str, name: str | None) -> dict:
    # --- production predict(): meta / tracks / conformal PI / AD ---
    res = predict(smiles, dose, route=route, n_mc_samples=0)
    cap = dict(gcd._CAPTURE)  # weights captured during this predict()'s combine()

    # --- profile/adme/drug/graph for metadata, curve, ddi ---
    profile, adme, drug, dgraph = gcd.build_drug_and_graph(smiles, dose, route, BASE_GRAPH)

    meta_cmax = float(res.pk.cmax.mean)
    meta_thalf = float(res.pk.t_half.mean) if res.pk.t_half else None
    eng_cmax = res.engine_pk.cmax.mean if res.engine_pk else None
    ml_cmax = res.ml_pk.cmax.mean if res.ml_pk else None
    clf_cmax = res.clf_pk.cmax.mean if res.clf_pk else None
    vdss_cmax = cap.get("vdss_cmax")
    w = cap.get("weights", {})

    # --- engine ODE curve, reconciled so its peak == engine-track Cmax ---
    thalf_for_t = meta_thalf if (meta_thalf and meta_thalf > 0) else 6.0
    t_end = max(4.0 * thalf_for_t, 24.0)
    _, _, sim = gcd.solve_single(dgraph, drug, t_end)
    t, c = sim.time_h, sim.concentrations["venous_blood"]
    tg, cg = gcd.downsample(t, c, 120)
    curve_t, curve_c = list(tg), list(cg)
    if eng_cmax and max(cg) > 0:
        f = eng_cmax / max(cg)
        curve_c = [v * f for v in cg]
    eng_auc = float(_trapz(curve_c, curve_t)) if curve_c else None
    eng_tmax = float(curve_t[int(np.argmax(curve_c))]) if curve_c else None
    clf_disp = (dose / eng_auc) if (eng_auc and eng_auc > 0) else None
    pkfit = gcd.fit_pk(eng_tmax, meta_thalf if (meta_thalf and meta_thalf > 0) else None)

    # --- DDI: real engine folds for the 4 perpetrators ---
    ep_b = compute_endpoints(sim, observation_node="venous_blood")
    ddi = gcd.ddi_folds(BASE_GRAPH, drug, t_end, float(ep_b.auc_0t.mean), float(ep_b.cmax.mean))

    # --- metadata ---
    formula = rdMolDescriptors.CalcMolFormula(MolFromSmiles(smiles))
    affs: dict[str, float] = {}
    try:
        for tag, dist in drug.enzyme_affinity.items():
            m = float(dist.mean)
            if m > 0:
                affs[tag] = m
    except Exception:  # noqa: BLE001
        pass
    if affs:
        total = sum(affs.values())
        ordered = sorted(affs.items(), key=lambda kv: -kv[1])
        enzyme_fraction = {k: round(v / total, 3) for k, v in ordered}
        primary_enzyme = ordered[0][0]
    else:
        enzyme_fraction = {}
        primary_enzyme = "renal"

    return {
        "id": "custom",
        "name": (name or "Custom compound").strip() or "Custom compound",
        "formula": formula,
        "mw": _SIG(profile.mw),
        "smiles": smiles,
        "type": profile.compound_type,
        "dose": dose,
        "route": route,
        "primaryEnzyme": primary_enzyme,
        "enzymeFraction": enzyme_fraction,
        "confidence": res.confidence,
        "inDomain": bool(res.in_applicability_domain),
        "adFlags": list(res.ad_flags),
        "meta": {
            "cmax": _SIG(meta_cmax),
            "tmax": _SIG(eng_tmax),
            "auc": _SIG(eng_auc),
            "thalf": _SIG(meta_thalf),
        },
        "cmax90ci": (
            [_SIG(res.cmax_90ci[0]), _SIG(res.cmax_90ci[1])] if res.cmax_90ci else None
        ),
        "tracks": {
            "engine": _SIG(eng_cmax),
            "ml": _SIG(ml_cmax),
            "clf": _SIG(clf_cmax),
            "vdss": _SIG(vdss_cmax),
        },
        "weights": {
            "engine": _SIG(w.get("engine")),
            "ml": _SIG(w.get("ml")),
            "clf": _SIG(w.get("clf")),
            "vdss": _SIG(w.get("vdss")),
        },
        "disposition": {
            "clf": _SIG(clf_disp),
            "vdss": _SIG(adme.vdss.mean),
            "fup": _SIG(adme.fup.mean),
            "clint": _SIG(adme.clint.mean),
        },
        "curve": {"t": _SIG_LIST(curve_t, 5), "c": _SIG_LIST(curve_c, 5)},
        "pkfit": pkfit,
        # tdm/dose-adjust are out of live scope (slow Bayesian update); the TDM view
        # labels its shrink "illustrative — full re-inference needs the live engine".
        "tdm": {"method": "ibis", "ess": None, "priorCv": None, "postCv": None, "reduction": None},
        "ddi": ddi,
    }
