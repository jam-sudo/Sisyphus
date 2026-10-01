"""DE-58 gate-test: does censored (Tobit/AFT) CLint regression improve end-to-end Cmax?

Deep-research candidate (Svensson et al. 2025, AI in the Life Sciences 7:100128). The
CLint training set (clearance_hepatocyte_az.tab) is 27.4% censored — 16.1% left-censored
at the assay floor (CLint=3.0), 11.3% right-censored at the ceiling (150) — yet the
shipped model (reg:squarederror on log10) treats those piles as exact point labels.

Controlled AFT ablation isolates the CENSORING TREATMENT: both arms are XGBoost
survival:aft (normal distribution = Tobit on log time), identical features and
hyperparameters; only the label bounds differ.
  AFT-exact    : every point lower=upper=Y (censored treated as exact).
  AFT-censored : floor -> left-censored (0, 3.0]; ceiling -> right-censored [150, +inf).

Stage A: per-drug ΔCLint on the 107-holdout (does censoring move the input?).
Stage B: feed shipped / AFT-exact / AFT-censored CLint through the full pipeline
         (predict -> engine -> meta) on one local stack; compare AAFE. ml/clf/vdss are
         CLint-independent -> identical across arms (ml AAFE is a sanity check); the
         censoring effect is AFT-censored minus AFT-exact.

Dev-state run (drugbank may be present); the Δ is state-internal and valid because the
state is held constant across all three arms. Absolute meta here is NOT the public-clone
2.743 headline. Gate: PASS if AFT-censored improves meta AAFE beyond noise; else FAIL.

Reproduce: PYTHONPATH=src python scripts/clint_censored_regression.py
"""
from __future__ import annotations

import csv
import json

import numpy as np
import xgboost as xgb

import sisyphus.predict.adme as adme_mod
from sisyphus.core import Distribution
from sisyphus.descriptors import compute_features
from sisyphus.pipeline.predict import predict
from sisyphus.validation.metrics import aafe
from sisyphus.validation.reference import load_reference

SEED = 20260422
FLOOR = 3.0
CEIL = 150.0
CLINT_TAB = "data/training/clearance_hepatocyte_az.tab"
OUT = "data/validation/clint_censored_regression_2026-07-08.json"


def feats(smiles):
    try:
        return compute_features(smiles)
    except Exception:
        return None


# ---- training data ----
tr_smi, tr_y = [], []
with open(CLINT_TAB) as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for row in reader:
        tr_smi.append(row[1].strip('"'))
        tr_y.append(float(row[2]))
Xtr, Yr = [], []
for smi, y in zip(tr_smi, tr_y):
    fv = feats(smi)
    if fv is None:
        continue
    Xtr.append(fv)
    Yr.append(y)
Xtr = np.vstack(Xtr)
Yr = np.asarray(Yr)

params = {
    "objective": "survival:aft", "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0,
    "tree_method": "hist", "max_depth": 6, "eta": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "seed": SEED, "nthread": 4,
}


def train_aft(lower, upper):
    dm = xgb.DMatrix(Xtr)
    dm.set_float_info("label_lower_bound", lower)
    dm.set_float_info("label_upper_bound", upper)
    return xgb.train(params, dm, num_boost_round=200)


lo_c = Yr.copy().astype(float)
up_c = Yr.copy().astype(float)
floor_mask = Yr == FLOOR
ceil_mask = Yr == CEIL
lo_c[floor_mask] = 0.0
up_c[floor_mask] = FLOOR
lo_c[ceil_mask] = CEIL
up_c[ceil_mask] = np.inf
aft_exact = train_aft(Yr.copy(), Yr.copy())
aft_cens = train_aft(lo_c, up_c)
print("AFT boosters trained")

# ---- holdout ----
refs = [r for r in load_reference() if r.in_holdout]
print(f"holdout n={len(refs)}")

# ---- Stage A: per-drug ΔCLint ----
smis = {r.name: r.smiles for r in refs}
Xho = []
names = []
for r in refs:
    fv = feats(r.smiles)
    if fv is None:
        continue
    Xho.append(fv)
    names.append(r.name)
Xho = np.vstack(Xho)
dho = xgb.DMatrix(Xho)
lg_exact = np.log10(np.clip(aft_exact.predict(dho), 0.1, None))
lg_cens = np.log10(np.clip(aft_cens.predict(dho), 0.1, None))
delta = lg_cens - lg_exact
stage_a = {
    "pct_censored": round(100 * (floor_mask.sum() + ceil_mask.sum()) / len(Yr), 1),
    "median_abs_dlog10": round(float(np.median(np.abs(delta))), 4),
    "mean_abs_dlog10": round(float(np.mean(np.abs(delta))), 4),
    "max_abs_dlog10": round(float(np.max(np.abs(delta))), 4),
    "n_gt_0p1": int((np.abs(delta) > 0.1).sum()),
    "signed_mean": round(float(np.mean(delta)), 4),
}

# ---- Stage B: end-to-end 3 arms ----
_CV = adme_mod._CLINT_CV


def patched(booster):
    def _fn(features):
        clint = float(booster.predict(xgb.DMatrix(features))[0])
        return Distribution(mean=max(clint, 0.1), cv=_CV)
    return _fn


def run_arm(patch_fn, tag):
    orig = adme_mod._predict_clint
    if patch_fn is not None:
        adme_mod._predict_clint = patch_fn
    rows = []
    try:
        for ref in refs:
            try:
                res = predict(ref.smiles, ref.dose_mg, ref.route)
                rows.append({
                    "name": ref.name, "obs": ref.cmax_obs,
                    "eng": res.engine_pk.cmax.mean if res.engine_pk else None,
                    "ml": res.ml_pk.cmax.mean if res.ml_pk else None,
                    "meta": res.pk.cmax.mean,
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({"name": ref.name, "obs": ref.cmax_obs,
                             "eng": None, "ml": None, "meta": None, "err": str(exc)})
    finally:
        adme_mod._predict_clint = orig
    print(f"  arm {tag} done")
    return rows


arms = {
    "shipped": run_arm(None, "shipped"),
    "aft_exact": run_arm(patched(aft_exact), "aft_exact"),
    "aft_cens": run_arm(patched(aft_cens), "aft_cens"),
}


def arm_aafe(rows, key):
    pred = np.array([r[key] for r in rows if r.get(key) and r[key] > 0])
    obs = np.array([r["obs"] for r in rows if r.get(key) and r[key] > 0])
    return round(float(aafe(pred, obs)), 4)


result = {
    "seed": SEED, "n": len(refs),
    "state": "dev-state; Δ is state-internal (state constant across arms)",
    "stage_a": stage_a,
    "aafe": {tag: {k: arm_aafe(rows, k) for k in ("eng", "ml", "meta")}
             for tag, rows in arms.items()},
}
ex = result["aafe"]["aft_exact"]
ce = result["aafe"]["aft_cens"]
result["censoring_delta"] = {
    "d_engine_aafe": round(ce["eng"] - ex["eng"], 4),
    "d_meta_aafe": round(ce["meta"] - ex["meta"], 4),
}
ml_ex = {r["name"]: r["ml"] for r in arms["aft_exact"]}
ml_ce = {r["name"]: r["ml"] for r in arms["aft_cens"]}
result["ml_identical_maxdiff"] = round(
    float(max(abs((ml_ex[n] or 0) - (ml_ce[n] or 0)) for n in ml_ex)), 8)

# per-drug meta fold change
by = {}
for tag in ("aft_exact", "aft_cens"):
    for r in arms[tag]:
        by.setdefault(r["name"], {})[tag] = (r["meta"], r["obs"])
n_better = n_worse = 0
movers = []
for name, d in by.items():
    if d.get("aft_exact") and d.get("aft_cens") and d["aft_exact"][0] and d["aft_cens"][0]:
        me, obs = d["aft_exact"][0], d["aft_exact"][1]
        mc = d["aft_cens"][0]
        dfold = abs(np.log10(mc / obs)) - abs(np.log10(me / obs))
        movers.append({"name": name, "d_meta_logfold": round(float(dfold), 3)})
        if dfold < -0.01:
            n_better += 1
        elif dfold > 0.01:
            n_worse += 1
movers.sort(key=lambda x: -abs(x["d_meta_logfold"]))
result["per_drug"] = {"n_meta_better": n_better, "n_meta_worse": n_worse,
                      "n_unchanged": len(movers) - n_better - n_worse}
result["top_meta_movers"] = movers[:8]
result["verdict"] = ("PASS (meta improves)" if result["censoring_delta"]["d_meta_aafe"] < -0.02
                     else "FAIL (neutral/worse)")

print(json.dumps({k: result[k] for k in
                  ("stage_a", "aafe", "censoring_delta", "ml_identical_maxdiff",
                   "per_drug", "verdict")}, indent=2))
json.dump(result, open(OUT, "w"), indent=2)
print("wrote", OUT)
