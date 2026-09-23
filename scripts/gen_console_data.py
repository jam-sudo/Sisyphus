#!/usr/bin/env python
"""Generate core prediction data for the static Sisyphus web console.

Each preset is produced by exactly one strict ``predict()`` call.  The final
Cmax comes from ``CmaxPrediction`` and the curve/Tmax/AUC/half-life come from
the matching ``EngineSimulation``.  The generator never monkeypatches the
meta-learner, re-solves the engine, or rescales a curve.

Run with the locked Python environment from the repo root:

    python scripts/gen_console_data.py

Outputs:
    web/public/data/console_data.json   -- the full console payload
    web/public/data/benchmark.json      -- copy of the N=107 development cache
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

from sisyphus._version import __version__  # noqa: E402
from sisyphus.pipeline.predict import predict  # noqa: E402
from sisyphus.predict.adme import predict_adme  # noqa: E402
from sisyphus.predict.chemistry import compute_profile  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("gen_console_data")
# Quiet the noisy engine/predict loggers a touch.
logging.getLogger("sisyphus").setLevel(logging.WARNING)

BENCH_SRC = REPO / "data" / "training" / "4track_holdout_predictions.json"
OUT_DIR = REPO / "web" / "public" / "data"
OUT_JSON = OUT_DIR / "console_data.json"
OUT_BENCH = OUT_DIR / "benchmark.json"

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
        "n_development": bench.get("n_holdout", len(scatter)),
        "classification": "retrospective_development_benchmark",
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
        "Each preset is produced by one strict Sisyphus predict() call.",
        "Final Cmax, component tracks, effective weights, and the residual interval are "
        "read from CmaxPrediction.",
        "The curve, Tmax, AUC, half-life, solver status, and mass balance are read from the "
        "same EngineSimulation; no second solve or curve rescaling is performed.",
        "N=107 is a repeatedly accessed development benchmark, not an independent holdout.",
    ]

    log.info("loading benchmark cache ...")
    bench_full, bench_summary = load_benchmark()
    constants = {
        "DEVELOPMENT_AAFE": _sig(bench_full["overall"]["meta"]["aafe"]),
        "ENGINE_AAFE": _sig(bench_full["overall"]["engine"]["aafe"]),
        "ML_AAFE": _sig(bench_full["overall"]["ml"]["aafe"]),
    }

    drugs_out = []
    for spec in DRUGS:
        did = spec["id"]
        log.info("=== %s (%d mg %s) ===", did, spec["dose"], spec["route"])
        res = predict(
            spec["smiles"],
            spec["dose"],
            route=spec["route"],
            n_mc_samples=0,
            strict=True,
        )
        cmax = res.cmax_prediction
        engine = res.engine_simulation
        if cmax is None or engine is None:
            raise RuntimeError(f"{did}: strict prediction returned an incomplete contract")

        profile = compute_profile(spec["smiles"])
        adme = predict_adme(profile)
        ep = engine.endpoints
        tmax = float(ep.tmax.mean)
        auc = float(ep.auc_0t.mean)
        half = float(ep.t_half.mean) if ep.t_half else None
        tg, cg = downsample(engine.time_h, engine.concentration_mg_l, 120)
        tracks = dict(cmax.tracks)
        weights = dict(cmax.weights)
        entry = {
            "id": did,
            "name": spec["name"],
            "formula": spec["formula"],
            "mw": spec["mw"],
            "smiles": spec["smiles"],
            "type": profile.compound_type,
            "dose": spec["dose"],
            "route": spec["route"],
            "primaryEnzyme": spec["primaryEnzyme"],
            "enzymeFraction": spec["enzymeFraction"],
            "confidence": res.confidence,
            "applicabilityStatus": (
                "structurally_in_scope" if res.in_applicability_domain else "flagged"
            ),
            "inDomain": bool(res.in_applicability_domain),
            "adFlags": list(res.ad_flags),
            "executionStatus": res.execution_status,
            "artifactProvenance": dict(res.artifact_provenance),
            "meta": {
                "cmax": _sig(cmax.cmax.mean),
                "tmax": _sig(tmax),
                "auc": _sig(auc),
                "thalf": _sig(half),
            },
            "endpointSources": {
                "cmax": cmax.method,
                "tmax": "engine",
                "auc": "engine",
                "thalf": "engine",
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
            "tracks": {k: _sig(tracks.get(k)) for k in ("engine", "ml", "clf", "vdss")},
            "weights": {k: _sig(weights.get(k)) for k in ("engine", "ml", "clf", "vdss")},
            "disposition": {
                "doseOverAuc0t": _sig(spec["dose"] / auc) if auc > 0 else None,
                "vdss": _sig(adme.vdss.mean),
                "fup": _sig(adme.fup.mean),
                "clint": _sig(adme.clint.mean),
            },
            "curve": {"t": _sig_list(tg, 5), "c": _sig_list(cg, 5)},
            "pkfit": fit_pk(tmax, half),
            "engineDiagnostics": {
                "observationNode": engine.observation_node,
                "solverSuccess": engine.solver_success,
                "massBalanceError": _sig(engine.mass_balance_error),
            },
        }
        drugs_out.append(entry)

    payload = {
        "meta_info": {
            "generated_by": "scripts/gen_console_data.py",
            "interpreter": sys.executable,
            "engine": f"Sisyphus v{__version__}",
            "notes": notes,
        },
        "constants": constants,
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
    hdr = f"{'drug':<13}{'metaCmax':>12}{'engineCmax':>14}{'curvePts':>10}{'massErr':>13}"
    print(hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)
    for d in drugs:
        meta = d.get("meta") or {}
        cmax = meta.get("cmax")
        assert d.get("curve") and d["curve"].get("t"), f"{d['id']} empty curve"
        assert cmax and cmax > 0, f"{d['id']} meta.cmax not >0 ({cmax})"
        npts = len(d["curve"]["t"])
        eng = d["tracks"]["engine"]
        mass_error = d["engineDiagnostics"]["massBalanceError"]
        print(
            f"{d['id']:<13}{cmax!s:>12}{eng!s:>14}{npts:>10}{mass_error!s:>13}",
            file=sys.stderr,
        )
    size = OUT_JSON.stat().st_size
    print(f"\nJSON: {OUT_JSON}  ({size:,} bytes = {size/1024:.1f} KB)", file=sys.stderr)
    return size


if __name__ == "__main__":
    payload, runtime = main()
    size = validate(payload)
    print(f"\nTOTAL RUNTIME: {runtime:.1f}s", file=sys.stderr)
