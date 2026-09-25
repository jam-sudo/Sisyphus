"""Reproduce the legacy development-residual Cmax interval artifact.

This is not a valid split-conformal calibration: the direct ML component saw
part of the calibration outcomes during fitting, and development coverage is consumed
development evidence. Writes data/validation/development_residual_interval.json.

Do not use this script to claim independent coverage. A replacement must create
nested out-of-fold predictions or use a separate untouched calibration cohort.

Run: PYTHONPATH=src python scripts/calibrate_conformal.py
"""

from __future__ import annotations

import json
import pathlib
from hashlib import sha256

import numpy as np

from sisyphus.pipeline.predict import predict
from sisyphus.validation.conformal import (
    conformal_quantile,
    empirical_coverage,
    nonconformity_scores,
)
from sisyphus.validation.holdout_contract import _PRODUCTION_FITTED_MODELS
from sisyphus.validation.reference import load_reference

_LEVELS = {"0.5": 0.5, "0.2": 0.2, "0.1": 0.1, "0.05": 0.05}
_TRACKS = ("meta", "engine", "ml")
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "data/validation/development_residual_interval.json"
_HOLDOUT_CACHE = _ROOT / "data/training/4track_holdout_predictions.json"


def _sha(path: pathlib.Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _predict_track(result, track):
    if track == "meta":
        return result.pk.cmax.mean
    if track == "engine":
        return result.engine_pk.cmax.mean if result.engine_pk else None
    if track == "ml":
        return result.ml_pk.cmax.mean if result.ml_pk else None
    return None


def _collect_train():
    refs = [r for r in load_reference() if r.in_training]
    rows = {t: {"pred": [], "obs": []} for t in _TRACKS}
    n_ok = 0
    skipped = []
    for ref in refs:
        try:
            res = predict(ref.smiles, ref.dose_mg, ref.route)
        except Exception as exc:
            skipped.append({"name": ref.name, "reason": str(exc)})
            continue
        n_ok += 1
        for t in _TRACKS:
            p = _predict_track(res, t)
            if p and p > 0:
                rows[t]["pred"].append(p)
                rows[t]["obs"].append(ref.cmax_obs)
    return rows, n_ok, len(refs), skipped


def _holdout_arrays():
    d = json.loads(_HOLDOUT_CACHE.read_text())
    out = {}
    for t in _TRACKS:
        key = {"meta": "meta", "engine": "eng", "ml": "ml"}[t]
        rs = [r for r in d["drugs"] if r.get(key) and r[key] > 0 and r.get("obs") and r["obs"] > 0]
        out[t] = (np.array([r[key] for r in rs]), np.array([r["obs"] for r in rs]))
    return out


def main():
    print("Running predict() on train set (calibration; Invariant #5: not holdout)...")
    train, n_ok, n_tot, skipped = _collect_train()
    print(f"  train predicted: {n_ok}/{n_tot}")
    if skipped:
        raise RuntimeError(f"Calibration predictions failed for {len(skipped)} training references: {skipped}")

    holdout = _holdout_arrays()
    artifact = {
        "method": "development_empirical_residual_quantile",
        "score": "abs_log10_fold_error",
        "interval": "multiplicative: pred /÷ 10**q",
        "calibration_set": "partially_in_sample_development",
        "n_calibration_meta": len(train["meta"]["pred"]),
        "n_training_reference": n_tot,
        "skipped_training_reference": skipped,
        "generated_from": "scripts/calibrate_conformal.py",
        "source_cache_sha256": _sha(_HOLDOUT_CACHE),
        "calibration_reference_sha256": _sha(_ROOT / "data/reference/clinical_pk.json"),
        "holdout_membership_sha256": _sha(_ROOT / "data/reference/holdout.json"),
        "model_artifact_sha256": {
            path.replace(".meta.json", ".json"): _sha(
                _ROOT / path.replace(".meta.json", ".json")
            ) for path in _PRODUCTION_FITTED_MODELS
        },
        "validity": "not split-conformal; fitted components saw calibration outcomes",
        "tracks": {},
        "consumed_development_benchmark_coverage": {},
    }

    print(f"\n{'track':8s} {'level':6s} {'q(log10)':>9s} {'half-width':>11s} {'holdout-cov':>12s}")
    for t in _TRACKS:
        pred = np.array(train[t]["pred"])
        obs = np.array(train[t]["obs"])
        scores = nonconformity_scores(pred, obs)
        artifact["tracks"][t] = {}
        artifact["consumed_development_benchmark_coverage"][t] = {}
        h_pred, h_obs = holdout[t]
        for lvl, alpha in _LEVELS.items():
            q = conformal_quantile(scores, alpha)
            cov = empirical_coverage(h_pred, h_obs, q)
            artifact["tracks"][t][lvl] = q
            artifact["consumed_development_benchmark_coverage"][t][lvl] = round(cov, 4)
            hw = "inf" if not np.isfinite(q) else f"/÷{10 ** q:.2f}"
            print(f"{t:8s} {lvl:6s} {q:9.4f} {hw:>11s} {cov:12.3f}")

    _OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote {_OUT}")
    print("MC baseline at nominal 0.90: 0.299 (development diagnostic only).")


if __name__ == "__main__":
    main()
