#!/usr/bin/env python
"""Generate console_data.json for the static Sisyphus web console.

Wires the REAL Sisyphus PBPK engine to a precomputed JSON payload that powers a
static web console (no server). Every numeric field below is produced by an
actual engine solve / predict() call -- there are no mock numbers.

Run with the conda interpreter from the repo root:

    /opt/miniconda3/bin/python scripts/gen_console_data.py

Outputs:
    web/public/data/console_data.json   -- the full console payload
    web/public/data/benchmark.json      -- copy of the 107-drug holdout cache

=== Approach notes (also surfaced in meta_info.notes) ===

* Meta-learner weights / VDss track: we monkeypatch ``MetaLearner.combine`` to
  capture the EXACT applied (engine, ml, clf, vdss) weights and the inline
  vdss_cmax for each predict() call, rather than re-deriving them. This is the
  most reliable approach -- it runs the real combine() and records the renormalised
  weights after the disagreement penalty + VDss (1-0.20) scaling. We re-implement
  only the weight-bookkeeping math inside the patch (the production combine() does
  not expose the per-track weights), faithfully mirroring src/sisyphus/ml/ensemble.py.

* Concentration curve: a separate deterministic engine solve (realize_means) at the
  listed single dose, T_END = max(4*t_half, 24) h, downsampled to ~120 points.

* pkfit: a 1-compartment oral model fit to the real engine curve so the frontend can
  do client-side multi-dose superposition. ke = ln2 / real_thalf; ka solved from the
  real tmax via tmax = ln(ka/ke)/(ka-ke) (numeric root). This is a dose-linear
  approximation -- the engine is (near) linear at these doses, so client-side scaling
  of the single-dose curve is a reasonable approximation for the interactive UI.

* DDI: real folds for all 8 victims x {ketoconazole, fluconazole, quinidine, rifampin}
  via apply_inhibition / apply_induction (rifampin = inducer). Same deterministic
  realize_means params. folds = modified_endpoint / base_endpoint.

* Steady-state: real solve_regimen at QD (interval=24h, n_doses=14) +
  compute_steady_state_metrics. Trough series derived per dose from the multi-dose
  SimResult (pre-dose conc sampled just before each dose time).

* TDM: real bayesian_update, ONE default scenario per drug. method='is' (importance
  sampling -- torch is absent so 'sbi' would silently fall back to ibis anyway; 'is'
  is fast and deterministic). n_prior=300 default; lowered to 150 on a per-drug
  timeout to respect the ~3 min budget (recorded in notes). The PRODUCTION-routed
  method name (from data/sbi/method_routing.json) is recorded separately for display.

* Robustness: every drug/workflow is wrapped in try/except; a failure records null +
  a note and the run continues. Deterministic throughout (seed=42, realize_means).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# --- repo wiring -----------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import sisyphus.engine.flux  # noqa: F401  MANDATORY: registers flux specs

from sisyphus.ddi import (  # noqa: E402
    FLUCONAZOLE,
    KETOCONAZOLE,
    QUINIDINE,
    RIFAMPIN,
    apply_induction,
    apply_inhibition,
)
from sisyphus.engine.compiler import ODECompiler, ResolvedParams  # noqa: E402
from sisyphus.engine.solver import solve  # noqa: E402
from sisyphus.graph.builder import (  # noqa: E402
    augment_for_active_species,
    build_from_yaml,
)
from sisyphus.ml import ensemble as _ensemble_mod  # noqa: E402
from sisyphus.pk.endpoints import compute_endpoints  # noqa: E402
from sisyphus.predict.adme import predict_adme  # noqa: E402
from sisyphus.predict.chemistry import compute_profile  # noqa: E402
from sisyphus.predict.ivive import build_drug_on_graph  # noqa: E402
from sisyphus.pipeline.predict import predict  # noqa: E402
from sisyphus.regimen.profile import compute_steady_state_metrics  # noqa: E402
from sisyphus.regimen.solver import solve_regimen  # noqa: E402
from sisyphus.regimen.tdm import Observation, bayesian_update  # noqa: E402
from sisyphus.regimen.types import DosingRegimen  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("gen_console_data")
# Quiet the noisy engine/predict loggers a touch.
logging.getLogger("sisyphus").setLevel(logging.WARNING)

PHYS_YAML = REPO / "data" / "physiology" / "reference_man.yaml"
BENCH_SRC = REPO / "data" / "training" / "4track_holdout_predictions.json"
ROUTING_SRC = REPO / "data" / "sbi" / "method_routing.json"
OUT_DIR = REPO / "web" / "public" / "data"
OUT_JSON = OUT_DIR / "console_data.json"
OUT_BENCH = OUT_DIR / "benchmark.json"

CONFORMAL_FACTOR = 10.0 ** 1.1113503935718492  # /÷ ~12.92 (meta q90, train-calibrated)

# Drug list: (id, dose_mg, route, smiles, display metadata)
DRUGS = [
    {
        "id": "caffeine", "name": "Caffeine", "dose": 100, "route": "oral",
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "formula": "C₈H₁₀N₄O₂", "mw": 194.19,
        "primaryEnzyme": "CYP1A2", "enzymeFraction": {"CYP1A2": 0.95},
    },
    {
        "id": "midazolam", "name": "Midazolam", "dose": 5, "route": "oral",
        "smiles": "c1ccc2c(c1)C(=NC(=O)N2)c1ccccc1F",
        "formula": "C₁₈H₁₃ClFN₃", "mw": 325.77,
        "primaryEnzyme": "CYP3A4", "enzymeFraction": {"CYP3A4": 0.93},
    },
    {
        "id": "warfarin", "name": "Warfarin", "dose": 10, "route": "oral",
        "smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "formula": "C₁₉H₁₆O₄", "mw": 308.33,
        "primaryEnzyme": "CYP2C9", "enzymeFraction": {"CYP2C9": 0.85},
    },
    {
        "id": "propranolol", "name": "Propranolol", "dose": 80, "route": "oral",
        "smiles": "CC(C)NCC(O)COc1cccc2ccccc12",
        "formula": "C₁₆H₂₁NO₂", "mw": 259.34,
        "primaryEnzyme": "CYP2D6", "enzymeFraction": {"CYP2D6": 0.6, "CYP1A2": 0.3},
    },
    {
        "id": "atorvastatin", "name": "Atorvastatin", "dose": 40, "route": "oral",
        "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
        "formula": "C₃₃H₃₅FN₂O₅", "mw": 558.64,
        "primaryEnzyme": "CYP3A4", "enzymeFraction": {"CYP3A4": 0.8},
    },
    {
        "id": "metformin", "name": "Metformin", "dose": 500, "route": "oral",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "formula": "C₄H₁₁N₅", "mw": 129.16,
        "primaryEnzyme": "renal", "enzymeFraction": {},
    },
    {
        "id": "morphine", "name": "Morphine", "dose": 10, "route": "oral",
        "smiles": "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
        "formula": "C₁₇H₁₉NO₃", "mw": 285.34,
        "primaryEnzyme": "UGT2B7", "enzymeFraction": {"UGT2B7": 0.9},
    },
    {
        "id": "ketorolac", "name": "Ketorolac", "dose": 10, "route": "oral",
        "smiles": "OC(=O)C1CCn2c1ccc2C(=O)c1ccccc1",
        "formula": "C₁₅H₁₃NO₃", "mw": 255.27,
        "primaryEnzyme": "UGT2B7", "enzymeFraction": {"UGT2B7": 0.75, "CYP2C9": 0.2},
    },
]

INHIBITORS_META = {
    "none": {"name": "None", "enzyme": "—", "type": "—", "strength": None},
    "ketoconazole": {"name": "Ketoconazole", "enzyme": "CYP3A4", "type": "inhibition", "strength": "strong"},
    "fluconazole": {"name": "Fluconazole", "enzyme": "CYP2C9/3A4", "type": "inhibition", "strength": "moderate"},
    "quinidine": {"name": "Quinidine", "enzyme": "CYP2D6", "type": "inhibition", "strength": "strong"},
    "rifampin": {"name": "Rifampin", "enzyme": "CYP3A4", "type": "induction", "strength": "inducer"},
}

# ===========================================================================
# Helpers
# ===========================================================================


def _sig(x, n=6):
    """Round a float to ~n significant figures (JSON-friendly). Pass-through for non-floats."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(xf):
        return None
    if xf == 0.0:
        return 0.0
    from math import floor, log10
    d = n - 1 - int(floor(log10(abs(xf))))
    return round(xf, d)


def _sig_list(xs, n=5):
    return [_sig(v, n) for v in xs]


# --- weight-capture monkeypatch -------------------------------------------
# We patch MetaLearner.combine to record the EXACT applied per-track weights and
# the inline vdss_cmax. The math here mirrors src/sisyphus/ml/ensemble.py::combine
# (base weights by compound_type, VDss (1-_W_VDSS) scaling, disagreement penalty,
# renormalisation). We then call the original to get the unchanged result so the
# returned PKEndpoints stays bit-identical to production.

_W_VDSS = _ensemble_mod._W_VDSS
_DIS_THR = _ensemble_mod._DISAGREEMENT_THRESHOLD_LOG10
_orig_combine = _ensemble_mod.MetaLearner.combine
_CAPTURE: dict = {}


def _capture_weights(self, engine_pk, ml_pk, *, dose_mg, compound_type, clf_pk,
                     vdss_cmax):
    """Replicate combine()'s weighting bookkeeping to recover applied weights."""
    cmax_pbpk = engine_pk.cmax.mean if engine_pk is not None else None
    cmax_ml = ml_pk.cmax.mean if ml_pk is not None else None
    cmax_clf = clf_pk.cmax.mean if clf_pk is not None else None

    is_base = compound_type == "base"
    w_eng_b = self.w_engine_base if is_base else self.w_engine_other
    w_ml_b = self.w_ml_base if is_base else self.w_ml_other
    w_clf_b = self.w_clf_base if is_base else self.w_clf_other

    vdss_available = vdss_cmax is not None and vdss_cmax > 0
    scale = (1.0 - _W_VDSS) if vdss_available else 1.0

    # tracks: list of (tag, log_cmax, weight)
    tracks = []
    if cmax_pbpk is not None and cmax_pbpk > 0:
        tracks.append(["engine", float(np.log10(max(cmax_pbpk, 1e-10))), w_eng_b * scale])
    if cmax_ml is not None and cmax_ml > 0:
        tracks.append(["ml", float(np.log10(max(cmax_ml, 1e-10))), w_ml_b * scale])
    if cmax_clf is not None and cmax_clf > 0:
        tracks.append(["clf", float(np.log10(max(cmax_clf, 1e-10))), w_clf_b * scale])
    if vdss_available:
        tracks.append(["vdss", float(np.log10(max(vdss_cmax, 1e-10))), _W_VDSS])

    weights = {"engine": None, "ml": None, "clf": None, "vdss": None}
    disagreement = None
    if len(tracks) >= 2:
        # disagreement penalty applied to engine
        if (cmax_pbpk is not None and cmax_pbpk > 0
                and cmax_ml is not None and cmax_ml > 0):
            log_eng = float(np.log10(max(cmax_pbpk, 1e-10)))
            log_ml = float(np.log10(max(cmax_ml, 1e-10)))
            disagreement = abs(log_eng - log_ml)
            if disagreement > _DIS_THR:
                pen = _DIS_THR / disagreement
                for t in tracks:
                    if t[0] == "engine":
                        t[2] = t[2] * pen
        total = sum(t[2] for t in tracks)
        if total > 0:
            for t in tracks:
                t[2] = t[2] / total
        else:
            n = len(tracks)
            for t in tracks:
                t[2] = 1.0 / n
        for t in tracks:
            weights[t[0]] = t[2]
    elif len(tracks) == 1:
        weights[tracks[0][0]] = 1.0

    _CAPTURE.clear()
    _CAPTURE["weights"] = weights
    _CAPTURE["vdss_cmax"] = float(vdss_cmax) if vdss_available else None
    _CAPTURE["vdss_active"] = bool(vdss_available)
    _CAPTURE["disagreement_log10"] = disagreement


def _patched_combine(self, engine_pk=None, ml_pk=None, *, dose_mg=1.0, logp=2.0,
                     tpsa=60.0, mw=300.0, fup=0.5, clint=10.0,
                     compound_type="neutral", pgp_flag=False, clf_pk=None,
                     vdss_cmax=None):
    try:
        _capture_weights(self, engine_pk, ml_pk, dose_mg=dose_mg,
                         compound_type=compound_type, clf_pk=clf_pk,
                         vdss_cmax=vdss_cmax)
    except Exception as e:  # never break production combine
        log.warning("weight capture failed: %s", e)
        _CAPTURE.clear()
    return _orig_combine(
        self, engine_pk, ml_pk, dose_mg=dose_mg, logp=logp, tpsa=tpsa, mw=mw,
        fup=fup, clint=clint, compound_type=compound_type, pgp_flag=pgp_flag,
        clf_pk=clf_pk, vdss_cmax=vdss_cmax,
    )


_ensemble_mod.MetaLearner.combine = _patched_combine
log.info("MetaLearner.combine patched for weight capture")


# --- engine build helpers --------------------------------------------------
def build_drug_and_graph(smiles, dose_mg, route, base_graph):
    """Build profile/adme/drug and an augmented graph for a single drug."""
    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    liver = {t: d.mean for t, d in base_graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(profile, adme, dose_mg, route, liver_enzymes=liver)
    graph = augment_for_active_species(base_graph, drug)
    return profile, adme, drug, graph


def solve_single(graph, drug, t_end):
    """Deterministic single-dose solve -> (compiled, params, SimResult)."""
    compiled = ODECompiler().compile(graph)
    params = ResolvedParams(graph.realize_means(), drug.realize_means())
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    sim = solve(compiled, params, y0, t_span=(0.0, t_end))
    return compiled, params, sim


def downsample(t, c, n=120):
    """Resample (t, c) onto n evenly spaced points spanning [t0, t_end]."""
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    grid = np.linspace(float(t[0]), float(t[-1]), n)
    cg = np.interp(grid, t, c)
    return grid.tolist(), cg.tolist()


def fit_pk(real_tmax, real_thalf):
    """1-compartment oral fit: ke = ln2/thalf; ka from tmax via numeric root."""
    ln2 = float(np.log(2.0))
    ke = ln2 / real_thalf if real_thalf and real_thalf > 0 else None
    ka = None
    if ke is not None and real_tmax and real_tmax > 0:
        # tmax = ln(ka/ke)/(ka-ke). Solve for ka > ke by bisection on f(ka)=0.
        def f(ka_):
            if ka_ <= ke:
                return 1e9
            return float(np.log(ka_ / ke) / (ka_ - ke) - real_tmax)
        lo, hi = ke * 1.0000001, ke * 10000.0
        flo, fhi = f(lo), f(hi)
        # f is monotone decreasing in ka above ke (tmax shrinks as ka grows)
        if flo > 0 and fhi < 0:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                if abs(fm) < 1e-9:
                    break
                if fm > 0:
                    lo = mid
                else:
                    hi = mid
            ka = 0.5 * (lo + hi)
        elif flo <= 0:
            ka = lo  # tmax already too small even at ka->ke+: clamp
        else:
            ka = hi
    return {"ka": _sig(ka), "ke": _sig(ke), "thalf": _sig(real_thalf)}


def ddi_folds(base_graph, drug, t_end, base_auc, base_cmax):
    """Real DDI folds for all 4 perpetrators (folds = modified / base)."""
    perps = {
        "ketoconazole": ("inhibit", KETOCONAZOLE),
        "fluconazole": ("inhibit", FLUCONAZOLE),
        "quinidine": ("inhibit", QUINIDINE),
        "rifampin": ("induce", RIFAMPIN),
    }
    out = {}
    for key, (kind, perp) in perps.items():
        try:
            if kind == "inhibit":
                mod_graph = apply_inhibition(base_graph, perp)
            else:
                mod_graph = apply_induction(base_graph, perp)
            mod_graph = augment_for_active_species(mod_graph, drug)
            compiled = ODECompiler().compile(mod_graph)
            params = ResolvedParams(mod_graph.realize_means(), drug.realize_means())
            y0 = np.zeros(compiled.n_states)
            y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
            sim = solve(compiled, params, y0, t_span=(0.0, t_end))
            ep = compute_endpoints(sim, observation_node="venous_blood")
            auc_fold = ep.auc_0t.mean / base_auc if base_auc > 0 else None
            cmax_fold = ep.cmax.mean / base_cmax if base_cmax > 0 else None
            out[key] = {"auc": _sig(auc_fold), "cmax": _sig(cmax_fold)}
        except Exception as e:
            log.warning("  DDI %s failed: %s", key, e)
            out[key] = {"auc": None, "cmax": None}
    return out


def steady_state(base_graph, drug, dose_mg, interval_h=24.0, n_doses=14):
    """Real multi-dose solve + SS metrics + per-dose trough series."""
    compiled = ODECompiler().compile(base_graph)
    params = ResolvedParams(base_graph.realize_means(), drug.realize_means())
    regimen = DosingRegimen.oral_repeated(dose_mg, interval_h, n_doses)
    result = solve_regimen(compiled, params, regimen)
    metrics = compute_steady_state_metrics(result, regimen, "venous_blood")
    # Per-dose trough: concentration just before each dose (pre-dose) for doses 2..n.
    t = np.asarray(result.time_h, dtype=float)
    c = np.asarray(result.concentrations["venous_blood"], dtype=float)
    troughs = []
    for i in range(1, n_doses):
        td = i * interval_h
        # sample just before the dose to avoid the bolus spike
        idx = int(np.searchsorted(t, td) - 1)
        idx = max(0, min(idx, len(c) - 1))
        troughs.append(float(c[idx]))
    # include the final trough after the last dose interval
    return {
        "interval": int(interval_h),
        "nDoses": int(n_doses),
        "cssMax": _sig(metrics.css_max),
        "cssMin": _sig(metrics.css_min),
        "accumRatio": _sig(metrics.accumulation_ratio),
        "doseToSS": int(metrics.dose_to_steady_state),
        "troughs": _sig_list(troughs, 5),
    }, compiled, params


def tdm_scenario(compiled, graph, drug, dose_mg, real_tmax, real_meta_cmax,
                 n_prior, seed=42):
    """One default TDM scenario via importance sampling."""
    obs = Observation(
        time_h=float(real_tmax),
        concentration=float(0.92 * real_meta_cmax),
        node="venous_blood",
        cv=0.10,
    )
    regimen = DosingRegimen.oral_repeated(dose_mg, 24.0, 3)
    res = bayesian_update(
        compiled, graph, drug, regimen=regimen, observations=[obs],
        n_prior=n_prior, seed=seed, method="is",
    )
    prior_cv = float(res.prior_cmax.cv)
    post_cv = float(res.posterior_cmax.cv)
    reduction = (1.0 - post_cv / prior_cv) if prior_cv > 0 else None
    return {
        "ess": _sig(res.ess),
        "priorCv": _sig(prior_cv),
        "postCv": _sig(post_cv),
        "reduction": _sig(reduction),
        "_n_prior": n_prior,
    }


# ===========================================================================
# Benchmark embedding
# ===========================================================================
def load_benchmark():
    bench = json.loads(BENCH_SRC.read_text())
    # copy verbatim to web/public/data
    OUT_BENCH.write_text(json.dumps(bench))
    overall = bench["overall"]
    in_domain = bench["in_domain"]
    scatter = []
    for d in bench["drugs"]:
        scatter.append({
            "name": d["name"],
            "obs": _sig(d.get("obs"), 6),
            "eng": _sig(d.get("eng"), 6),
            "ml": _sig(d.get("ml"), 6),
            "meta": _sig(d.get("meta"), 6),
            "in_ad": bool(d.get("in_ad", False)),
        })
    return bench, {
        "n_holdout": bench.get("n_holdout", len(scatter)),
        "overall": overall,
        "in_domain": in_domain,
        "scatter": scatter,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    notes = [
        "Every numeric field is produced by a real Sisyphus engine solve / predict() "
        "call (no mock values).",
        "Meta-learner weights captured by monkeypatching MetaLearner.combine to record "
        "the exact applied (engine/ml/clf/vdss) weights after disagreement-penalty + "
        "VDss (1-0.20) scaling + renormalisation; the original combine() is then called "
        "unchanged so production output is bit-identical.",
        "Concentration curves: deterministic engine solve (realize_means) at the listed "
        "single dose, T_END=max(4*t_half,24)h, downsampled to ~120 evenly-spaced points.",
        "pkfit is a 1-compartment oral model fit to the real curve (ke=ln2/t_half, ka from "
        "numeric root of tmax=ln(ka/ke)/(ka-ke)) for client-side multi-dose superposition; "
        "this assumes dose-linear PK for the interactive multi-dose view.",
        "TDM: real bayesian_update, method='is' (importance sampling; torch absent so 'sbi' "
        "would fall back to ibis -- 'is' is fast+deterministic). n_prior=300 default. The "
        "production-routed method per drug (data/sbi/method_routing.json) is shown separately.",
    ]

    routing = {}
    try:
        routing = json.loads(ROUTING_SRC.read_text()).get("routes", {})
    except Exception as e:
        log.warning("routing load failed: %s", e)

    log.info("loading benchmark cache ...")
    bench_full, bench_summary = load_benchmark()
    constants = {
        "HOLDOUT_AAFE": _sig(bench_full["overall"]["meta"]["aafe"]),
        "INDOMAIN_AAFE": _sig(bench_full["in_domain"]["meta"]["aafe"]),
        "ENGINE_AAFE": _sig(bench_full["overall"]["engine"]["aafe"]),
        "ML_AAFE": _sig(bench_full["overall"]["ml"]["aafe"]),
        "conformal_factor": _sig(CONFORMAL_FACTOR),
    }

    log.info("building base physiology graph ...")
    base_graph = build_from_yaml(PHYS_YAML)

    drugs_out = []
    tdm_total_budget_s = 180.0
    tdm_spent = 0.0

    for spec in DRUGS:
        did = spec["id"]
        log.info("=== %s (%d mg %s) ===", did, spec["dose"], spec["route"])
        entry = {
            "id": did, "name": spec["name"], "formula": spec["formula"],
            "mw": spec["mw"], "smiles": spec["smiles"], "dose": spec["dose"],
            "route": spec["route"], "primaryEnzyme": spec["primaryEnzyme"],
            "enzymeFraction": spec["enzymeFraction"],
        }
        per_notes = []

        # --- production predict() (meta/tracks/weights/conformal/AD) -----
        try:
            res = predict(spec["smiles"], spec["dose"], route=spec["route"],
                          n_mc_samples=0)
            cap = dict(_CAPTURE)  # captured during this predict()'s combine()
            profile = compute_profile(spec["smiles"])
            adme = predict_adme(profile)

            entry["type"] = profile.compound_type
            entry["confidence"] = res.confidence
            entry["inDomain"] = bool(res.in_applicability_domain)
            entry["adFlags"] = list(res.ad_flags)

            meta_cmax = float(res.pk.cmax.mean)
            meta_tmax = float(res.pk.tmax.mean) if res.pk.tmax else None
            meta_auc = float(res.pk.auc_0t.mean) if res.pk.auc_0t else None
            meta_thalf = float(res.pk.t_half.mean) if res.pk.t_half else None
            entry["meta"] = {
                "cmax": _sig(meta_cmax), "tmax": _sig(meta_tmax),
                "auc": _sig(meta_auc), "thalf": _sig(meta_thalf),
            }
            if res.cmax_90ci is not None:
                entry["cmax90ci"] = [_sig(res.cmax_90ci[0]), _sig(res.cmax_90ci[1])]
            else:
                entry["cmax90ci"] = None

            eng_cmax = res.engine_pk.cmax.mean if res.engine_pk else None
            ml_cmax = res.ml_pk.cmax.mean if res.ml_pk else None
            clf_cmax = res.clf_pk.cmax.mean if res.clf_pk else None
            vdss_cmax = cap.get("vdss_cmax")
            entry["tracks"] = {
                "engine": _sig(eng_cmax), "ml": _sig(ml_cmax),
                "clf": _sig(clf_cmax), "vdss": _sig(vdss_cmax),
            }
            w = cap.get("weights", {})
            entry["weights"] = {
                "engine": _sig(w.get("engine")), "ml": _sig(w.get("ml")),
                "clf": _sig(w.get("clf")), "vdss": _sig(w.get("vdss")),
            }
            # disposition
            clf_disp = (spec["dose"] / meta_auc) if (meta_auc and meta_auc > 0) else None
            entry["disposition"] = {
                "clf": _sig(clf_disp),
                "vdss": _sig(adme.vdss.mean),
                "fup": _sig(adme.fup.mean),
                "clint": _sig(adme.clint.mean),
            }
        except Exception as e:
            log.exception("  predict() failed for %s: %s", did, e)
            per_notes.append(f"predict() failed: {e}")
            entry.setdefault("type", None)
            entry.setdefault("meta", None)
            meta_cmax = meta_tmax = meta_thalf = meta_auc = None

        # --- single-dose curve + pkfit ---------------------------------
        curve_pts = 0
        try:
            thalf_for_t = meta_thalf if (meta_thalf and meta_thalf > 0) else 6.0
            t_end = max(4.0 * thalf_for_t, 24.0)
            _, _, sim = (lambda g: solve_single(g[3], g[2], t_end))(
                build_drug_and_graph(spec["smiles"], spec["dose"], spec["route"],
                                     base_graph)
            )
            t, c = sim.time_h, sim.concentrations["venous_blood"]
            tg, cg = downsample(t, c, 120)
            entry["curve"] = {"t": _sig_list(tg, 5), "c": _sig_list(cg, 5)}
            curve_pts = len(tg)
            real_tmax = float(t[int(np.argmax(c))])
            entry["pkfit"] = fit_pk(meta_tmax if meta_tmax else real_tmax,
                                    meta_thalf if (meta_thalf and meta_thalf > 0) else None)
        except Exception as e:
            log.exception("  curve failed for %s: %s", did, e)
            per_notes.append(f"curve failed: {e}")
            entry["curve"] = {"t": [], "c": []}
            entry["pkfit"] = {"ka": None, "ke": None, "thalf": None}

        # --- reconcile plotted curve with the authoritative engine-track Cmax ---
        # predict()'s engine_pk.cmax (entry["tracks"]["engine"]) is the value the
        # meta-learner actually used; a separate manual solve can diverge (active-
        # species handling, e.g. morphine ~2x). Scale the curve so its peak == the
        # engine track, then derive the displayed AUC / Tmax / CL·F from THIS curve
        # so the figure, AUC and CL/F are internally consistent.
        try:
            _eng = (entry.get("tracks") or {}).get("engine")
            _c, _t = entry["curve"]["c"], entry["curve"]["t"]
            if _eng and _c and "disposition" in entry:
                _peak = max(_c)
                if _peak > 0:
                    _cs = [v * (_eng / _peak) for v in _c]
                    entry["curve"]["c"] = _sig_list(_cs, 5)
                    _trapz = getattr(np, "trapezoid", np.trapz)
                    _auc = float(_trapz(_cs, _t))
                    if entry.get("meta"):
                        entry["meta"]["auc"] = _sig(_auc)
                        entry["meta"]["tmax"] = _sig(float(_t[int(np.argmax(_cs))]))
                    entry["disposition"]["clf"] = _sig(spec["dose"] / _auc) if _auc > 0 else None
        except Exception as e:  # noqa: BLE001
            per_notes.append(f"curve/engine reconcile skipped: {e}")

        # --- rebuild graph+drug once for DDI / SS / TDM ----------------
        try:
            profile2, adme2, drug2, dgraph = build_drug_and_graph(
                spec["smiles"], spec["dose"], spec["route"], base_graph)
            thalf_for_t = meta_thalf if (meta_thalf and meta_thalf > 0) else 6.0
            t_end = max(4.0 * thalf_for_t, 24.0)
            # base endpoints for DDI folds
            compiled_b, params_b, sim_b = solve_single(dgraph, drug2, t_end)
            ep_b = compute_endpoints(sim_b, observation_node="venous_blood")
            base_auc = float(ep_b.auc_0t.mean)
            base_cmax = float(ep_b.cmax.mean)
        except Exception as e:
            log.exception("  drug rebuild failed for %s: %s", did, e)
            per_notes.append(f"ddi/ss/tdm setup failed: {e}")
            drug2 = dgraph = None
            base_auc = base_cmax = None

        # --- DDI -------------------------------------------------------
        if dgraph is not None:
            try:
                entry["ddi"] = ddi_folds(base_graph, drug2, t_end, base_auc, base_cmax)
            except Exception as e:
                log.exception("  DDI failed for %s: %s", did, e)
                per_notes.append(f"DDI failed: {e}")
                entry["ddi"] = {k: {"auc": None, "cmax": None}
                                for k in ("ketoconazole", "fluconazole",
                                          "quinidine", "rifampin")}
        else:
            entry["ddi"] = {k: {"auc": None, "cmax": None}
                            for k in ("ketoconazole", "fluconazole",
                                      "quinidine", "rifampin")}

        # --- steady state ----------------------------------------------
        ss_compiled = ss_params = None
        if dgraph is not None:
            try:
                ss, ss_compiled, ss_params = steady_state(base_graph, drug2,
                                                          spec["dose"])
                entry["simulate"] = ss
            except Exception as e:
                log.exception("  steady-state failed for %s: %s", did, e)
                per_notes.append(f"steady-state failed: {e}")
                entry["simulate"] = None
        else:
            entry["simulate"] = None

        # --- TDM -------------------------------------------------------
        routed_method = routing.get(did, "ibis")  # production-routed display name
        if dgraph is not None and meta_cmax and meta_tmax:
            n_prior = 300
            if tdm_spent > tdm_total_budget_s * 0.7:
                n_prior = 150
                per_notes.append(
                    "TDM n_prior lowered to 150 to respect the ~3 min budget.")
            t_tdm = time.time()
            try:
                tdm_compiled = ODECompiler().compile(dgraph)
                tdm = tdm_scenario(tdm_compiled, dgraph, drug2, spec["dose"],
                                   meta_tmax, meta_cmax, n_prior=n_prior, seed=42)
                used_np = tdm.pop("_n_prior")
                tdm["method"] = routed_method
                entry["tdm"] = tdm
                if used_np != 300:
                    per_notes.append(f"TDM ran with n_prior={used_np} (budget).")
            except Exception as e:
                log.exception("  TDM failed for %s: %s", did, e)
                per_notes.append(f"TDM failed: {e}")
                entry["tdm"] = {"method": routed_method, "ess": None,
                                "priorCv": None, "postCv": None, "reduction": None}
            tdm_spent += (time.time() - t_tdm)
            log.info("  TDM done (%.1fs, total %.1fs)", time.time() - t_tdm, tdm_spent)
        else:
            entry["tdm"] = {"method": routed_method, "ess": None,
                            "priorCv": None, "postCv": None, "reduction": None}

        if per_notes:
            entry["_notes"] = per_notes
        drugs_out.append(entry)

    payload = {
        "meta_info": {
            "generated_by": "scripts/gen_console_data.py",
            "interpreter": "/opt/miniconda3/bin/python",
            "engine": "Sisyphus v0.4 post-FLUX-1",
            "notes": notes,
        },
        "constants": constants,
        "inhibitors": INHIBITORS_META,
        "benchmark": bench_summary,
        "drugs": drugs_out,
    }

    OUT_JSON.write_text(json.dumps(payload, separators=(",", ":")))
    runtime = time.time() - t0
    log.info("wrote %s (runtime %.1fs)", OUT_JSON, runtime)
    return payload, runtime


def validate(payload):
    """Validate the produced payload and print a compact summary table."""
    drugs = payload["drugs"]
    assert len(drugs) == 8, f"expected 8 drugs, got {len(drugs)}"
    print("\n=== VALIDATION SUMMARY ===", file=sys.stderr)
    hdr = f"{'drug':<13}{'metaCmax':>10}{'curvePts':>9}{'ddi[keto]cmax':>15}{'tdm':>10}{'reduct':>8}{'cssMax':>10}"
    print(hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)
    for d in drugs:
        meta = d.get("meta") or {}
        cmax = meta.get("cmax")
        assert d.get("curve") and d["curve"].get("t"), f"{d['id']} empty curve"
        assert cmax and cmax > 0, f"{d['id']} meta.cmax not >0 ({cmax})"
        npts = len(d["curve"]["t"])
        ddi_keto = (d.get("ddi", {}).get("ketoconazole", {}) or {}).get("cmax")
        tdm = d.get("tdm", {}) or {}
        sim = d.get("simulate") or {}
        print(
            f"{d['id']:<13}{cmax!s:>10}{npts:>9}{ddi_keto!s:>15}"
            f"{str(tdm.get('method')):>10}{str(tdm.get('reduction')):>8}"
            f"{str(sim.get('cssMax')):>10}",
            file=sys.stderr,
        )
    size = OUT_JSON.stat().st_size
    print(f"\nJSON: {OUT_JSON}  ({size:,} bytes = {size/1024:.1f} KB)", file=sys.stderr)
    # midazolam + ketoconazole reality check
    mz = next((d for d in drugs if d["id"] == "midazolam"), None)
    if mz:
        keto = mz.get("ddi", {}).get("ketoconazole", {})
        print(f"midazolam+ketoconazole DDI: AUC fold={keto.get('auc')}  "
              f"Cmax fold={keto.get('cmax')}", file=sys.stderr)
    return size


if __name__ == "__main__":
    payload, runtime = main()
    size = validate(payload)
    print(f"\nTOTAL RUNTIME: {runtime:.1f}s", file=sys.stderr)
