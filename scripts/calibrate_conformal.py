"""Calibrate split-conformal Cmax prediction intervals on the TRAIN set.

Invariant #5: the deployed conformal quantile is fit on the train set (67 drugs),
NEVER the holdout. The holdout is used only as an honest out-of-sample coverage
*validation*. Writes data/validation/conformal_calibration.json.

Note: the ML track is trained on the train Cmax, so the meta's train residuals are
partially in-sample; the holdout-validation block reports the true deployed coverage.

Run: PYTHONPATH=src python scripts/calibrate_conformal.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from sisyphus.pipeline.predict import predict
from sisyphus.validation.conformal import (
    conformal_quantile,
    empirical_coverage,
    nonconformity_scores,
)
from sisyphus.validation.reference import load_reference

_LEVELS = {"0.5": 0.5, "0.2": 0.2, "0.1": 0.1, "0.05": 0.05}
_TRACKS = ("meta", "engine", "ml")
_OUT = pathlib.Path("data/validation/conformal_calibration.json")
_HOLDOUT_CACHE = pathlib.Path("data/training/4track_holdout_predictions.json")


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
    for ref in refs:
        try:
            res = predict(ref.smiles, ref.dose_mg, ref.route)
        except Exception:
            continue
        n_ok += 1
        for t in _TRACKS:
            p = _predict_track(res, t)
            if p and p > 0:
                rows[t]["pred"].append(p)
                rows[t]["obs"].append(ref.cmax_obs)
    return rows, n_ok, len(refs)


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
    train, n_ok, n_tot = _collect_train()
    print(f"  train predicted: {n_ok}/{n_tot}")

    holdout = _holdout_arrays()
    artifact = {
        "method": "split_conformal",
        "score": "abs_log10_fold_error",
        "interval": "multiplicative: pred /÷ 10**q",
        "calibration_set": "train",
        "n_calibration_meta": len(train["meta"]["pred"]),
        "generated_from": "scripts/calibrate_conformal.py",
        "tracks": {},
        "holdout_validation_coverage": {},
    }

    print(f"\n{'track':8s} {'level':6s} {'q(log10)':>9s} {'half-width':>11s} {'holdout-cov':>12s}")
    for t in _TRACKS:
        pred = np.array(train[t]["pred"])
        obs = np.array(train[t]["obs"])
        scores = nonconformity_scores(pred, obs)
        artifact["tracks"][t] = {}
        artifact["holdout_validation_coverage"][t] = {}
        h_pred, h_obs = holdout[t]
        for lvl, alpha in _LEVELS.items():
            q = conformal_quantile(scores, alpha)
            cov = empirical_coverage(h_pred, h_obs, q)
            artifact["tracks"][t][lvl] = q
            artifact["holdout_validation_coverage"][t][lvl] = round(cov, 4)
            hw = "inf" if not np.isfinite(q) else f"/÷{10 ** q:.2f}"
            print(f"{t:8s} {lvl:6s} {q:9.4f} {hw:>11s} {cov:12.3f}")

    _OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote {_OUT}")
    print("MC baseline at nominal 0.90: 0.299 (documented). Conformal target: ~0.90.")


if __name__ == "__main__":
    main()
