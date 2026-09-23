"""A/B: plain XGBoost vs censoring-aware (AFT interval) CLint on TDC Hepatocyte_AZ.

Hepatocyte_AZ labels are clipped at the assay limits: 3.0 (below LLOQ) and 150
(upper cap). Plain regression treats them as exact values, so the model cannot
predict below ~3 uL/min/1e6 cells. Arm B trains on interval labels instead:
Y == 3.0 -> (0, 3], Y == 150 -> [150, inf), else exact.

Both arms share data, holdout/reference exclusion, features, scaffold folds and
tree hyperparameters, so any difference is attributable to the loss alone.

Usage:
    PYTHONPATH=src python scripts/clint_censored_ab.py                 # scaffold-CV A/B
    PYTHONPATH=src python scripts/clint_censored_ab.py --benchmark B --holdout-only \
        --seed 42 --out /tmp/bench_B.json                               # 107-holdout arm

--benchmark patches adme._predict_clint at runtime (no tracked model or source
change) and runs scripts/run_engine_benchmark.py in-process. Results:
data/validation/clint_censored_ab_2026-09-22.json (DE-58).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train_clint_expanded import (  # noqa: E402
    _canonical_smiles,
    build_holdout_keys,
    compute_feature_matrix,
    is_holdout,
    scaffold_split_indices,
)

log = logging.getLogger(__name__)

HEP_TAB = ROOT / "data" / "training" / "clearance_hepatocyte_az.tab"
LLOQ = 3.0  # Hepatocyte_AZ lower reporting limit (16% of labels sit exactly here)
ULOQ = 150.0  # Hepatocyte_AZ upper cap (11% of labels)
FLOOR = 0.1  # production floor, adme._predict_clint
TREE = dict(max_depth=6, eta=0.1, subsample=0.8, colsample_bytree=0.8, seed=42, nthread=4)
N_ROUNDS = 500


def load_data(exclude_train: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    hold = json.loads((ROOT / "data/reference/holdout.json").read_text())
    clin = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())
    # production excludes the 107 holdout only; exclude_train also drops the 76 train refs
    keys = build_holdout_keys(hold["holdout"] + (hold["train"] if exclude_train else []), clin)
    df = pd.read_csv(HEP_TAB, sep="\t").rename(columns={"X": "Drug", "ID": "Drug_ID"})
    df["canonical_smiles"] = df["Drug"].apply(_canonical_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    df = df.groupby("canonical_smiles").agg({"Y": "mean", "Drug_ID": "first"}).reset_index()
    mask = df.apply(lambda r: is_holdout(r.canonical_smiles, str(r.Drug_ID), keys), axis=1)
    df = df[~mask].reset_index(drop=True)
    X, smiles, idx = compute_feature_matrix(df)
    return X, df.loc[idx, "Y"].to_numpy(float), smiles


def bounds(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = y.copy(), y.copy()
    lo[y <= LLOQ] = 0.0
    hi[y >= ULOQ] = np.inf
    return lo, hi


def fit_plain(X, y):
    d = xgb.DMatrix(X, label=np.log10(np.maximum(y, FLOOR)))
    return xgb.train({**TREE, "objective": "reg:squarederror"}, d, N_ROUNDS)


def fit_aft(X, y, sigma):
    d = xgb.DMatrix(X)
    lo, hi = bounds(y)
    d.set_float_info("label_lower_bound", lo)
    d.set_float_info("label_upper_bound", hi)
    params = {**TREE, "objective": "survival:aft", "eval_metric": "aft-nloglik",
              "aft_loss_distribution": "normal", "aft_loss_distribution_scale": sigma}
    return xgb.train(params, d, N_ROUNDS)


def predict_log10(model, X) -> np.ndarray:
    p = model.predict(xgb.DMatrix(X))
    if model.attr("aft") == "1":  # AFT predicts on the original scale
        p = np.log10(np.maximum(p, 1e-12))
    return np.maximum(p, np.log10(FLOOR))


def cv(X, y, smiles, make) -> np.ndarray:
    pred = np.zeros(len(y))
    folds = scaffold_split_indices(smiles, n_folds=5)
    for k in range(5):
        te = np.array(folds[k])
        tr = np.array([i for j in range(5) if j != k for i in folds[j]])
        m = make(X[tr], y[tr])
        pred[te] = predict_log10(m, X[te])
    return pred


def score(pred: np.ndarray, y: np.ndarray) -> dict:
    ly = np.log10(y)
    ex = (y > LLOQ) & (y < ULOQ)
    r = pred[ex] - ly[ex]
    low, high = y <= LLOQ, y >= ULOQ
    return {
        "exact_n": int(ex.sum()),
        "exact_r2": round(float(1 - np.sum(r**2) / np.sum((ly[ex] - ly[ex].mean()) ** 2)), 3),
        "exact_rmse_log10": round(float(np.sqrt(np.mean(r**2))), 3),
        # interval-consistent: prediction lands inside the censored interval
        "low_n": int(low.sum()),
        "low_pred_le_lloq": round(float(np.mean(pred[low] <= np.log10(LLOQ))), 3),
        "low_pred_median": round(float(10 ** np.median(pred[low])), 2),
        "high_pred_ge_uloq": round(float(np.mean(pred[high] >= np.log10(ULOQ))), 3),
        # censoring-aware error: distance outside the interval (0 if consistent)
        "interval_mae_log10": round(float(np.mean(np.where(
            low, np.maximum(pred - np.log10(LLOQ), 0),
            np.where(high, np.maximum(np.log10(ULOQ) - pred, 0), np.abs(pred - ly))))), 3),
    }


def benchmark(arm: str, sigma: float, exclude_train: bool, out: str) -> None:
    import runpy

    import sisyphus.predict.adme as adme
    from sisyphus.core import Distribution

    X, y, _ = load_data(exclude_train=exclude_train)
    model = fit_plain(X, y) if arm == "A" else fit_aft(X, y, sigma)
    if arm == "B":
        model.set_attr(aft="1")

    def _predict_clint(features: np.ndarray) -> Distribution:
        lc = float(predict_log10(model, np.asarray(features).reshape(1, -1))[0])
        return Distribution(mean=max(10**lc, FLOOR), cv=adme._CLINT_CV)

    adme._predict_clint = _predict_clint
    sys.argv = ["run_engine_benchmark.py", "--save-json", out]
    runpy.run_path(str(ROOT / "scripts" / "run_engine_benchmark.py"), run_name="__main__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.6,1.0,1.4")
    ap.add_argument("--save-model", help="write the AFT model (best sigma) to this path")
    ap.add_argument("--sigma", type=float, default=0.6, help="AFT sigma (benchmark/save)")
    ap.add_argument("--benchmark", choices=["A", "B"], help="run the 107-holdout with arm A or B")
    ap.add_argument("--holdout-only", action="store_true",
                    help="exclude only the 107 holdout (as production does)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="benchmark JSON output path")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    TREE["seed"] = a.seed
    if a.benchmark:
        benchmark(a.benchmark, a.sigma, not a.holdout_only, a.out)
        return
    X, y, smiles = load_data()
    print(f"N={len(y)} | at LLOQ {np.mean(y <= LLOQ):.3f} | at ULOQ {np.mean(y >= ULOQ):.3f}")

    rows = {"A_plain": score(cv(X, y, smiles, fit_plain), y)}
    for s in (float(v) for v in a.sigmas.split(",")):
        def make(Xt, yt, s=s):
            m = fit_aft(Xt, yt, s)
            m.set_attr(aft="1")
            return m
        rows[f"B_aft_sigma{s}"] = score(cv(X, y, smiles, make), y)
    print(pd.DataFrame(rows).T.to_string())

    if a.save_model:
        m = fit_aft(X, y, a.sigma)
        m.set_attr(aft="1")  # predict_log10 converts AFT's original-scale output
        m.save_model(a.save_model)
        print("saved", a.save_model)


if __name__ == "__main__":
    main()
