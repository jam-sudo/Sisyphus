"""DE-57 decorrelation-gate: does per-drug CLint *epistemic* uncertainty predict
per-drug Cmax error — i.e. is there a signal an "uncertainty-aware meta" could
exploit to down-weight the engine track where CLint is unreliable?

This is the last diagnosis.md §9 CLint-floor candidate that (a) is sourceable from
repo data, (b) does not touch the production headline, and (c) structurally sidesteps
error-cancellation (it is a *weighting* signal, not a point-estimate change).

Distinct from DE-55, which killed divergence/AD/magnitude and the *propagated
MC-output* uncertainty (rho=0.039). The shipped CLint Distribution CV is a FIXED
constant (_CLINT_CV=1.0 in predict/adme.py) -> carries zero per-drug info, which is
exactly why the propagated signal was flat. Here we CONSTRUCT a genuine per-drug
epistemic uncertainty two independent ways:
  U_boot : SD of log10(CLint) across a K-model bootstrap ensemble (model/data
           disagreement -> high where the CLint training set is sparse).
  U_ad   : 1 - max Tanimoto similarity of the drug's Morgan FP to the CLint
           training set (applicability-domain distance).

Gate (DE-55 style): Spearman rho of each signal against per-drug residuals, with
bootstrap 95% CI (10000 resamples, seed 20260422 = project convention).
  r_meta  = |log10(meta/obs)|         (overall meta error)
  e_eng   = |log10(eng /obs)|         (engine error; engine is the CLint-fed track)
  diff    = e_eng - e_ml              (>0 => engine worse than ml => an
                                       uncertainty-aware meta SHOULD shift weight to
                                       ml; this is THE actionable correlation)
Verdict: PASS iff any |rho| >= 0.30 with CI excluding 0. Else FAIL.

Reproduce: PYTHONPATH=src python scripts/clint_uncertainty_gate.py
Deterministic; uses only repo data. Absolute residuals come from the canonical cache;
U is computed on the local stack, but the gate uses only *rank* correlation, which is
robust to CLint point-prediction stack drift.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import xgboost as xgb
from scipy import stats

from sisyphus.descriptors import compute_features

SEED = 20260422
K = 30
rng = np.random.default_rng(SEED)

CLINT_TAB = "data/training/clearance_hepatocyte_az.tab"
CACHE = "data/training/4track_holdout_predictions.json"
CPK = "data/reference/clinical_pk.json"
OUT = "data/validation/clint_uncertainty_gate_2026-07-08.json"


def feats(smiles: str):
    try:
        return compute_features(smiles)
    except Exception:
        return None


# ---- 1. CLint training set ----------------------------------------------------
tr_smi, tr_y = [], []
with open(CLINT_TAB) as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        tr_smi.append(row[1].strip('"'))
        tr_y.append(float(row[2]))
Xtr, ytt = [], []
for s, y in zip(tr_smi, tr_y):
    fv = feats(s)
    if fv is None:
        continue
    Xtr.append(fv)
    ytt.append(np.log10(max(y, 0.1)))  # mirror shipped model target transform
Xtr = np.vstack(Xtr)
ytt = np.asarray(ytt)
print(f"CLint train: {Xtr.shape[0]} compounds, {Xtr.shape[1]} feats, "
      f"y(log10) range [{ytt.min():.2f},{ytt.max():.2f}]")

# ---- 2. Holdout 107: residuals (cache) + SMILES (clinical_pk) -----------------
cache = json.load(open(CACHE))
drugs = cache["drugs"]
cpk = json.load(open(CPK))["drugs"]
smap = {k.strip().lower(): v["smiles"] for k, v in cpk.items() if v.get("smiles")}

names, obs, eng, ml, meta, in_ad = [], [], [], [], [], []
Xho = []
for d in drugs:
    sm = smap.get(d["name"].strip().lower())
    fv = feats(sm) if sm else None
    if fv is None:
        print("  !! no features for", d["name"])
        continue
    names.append(d["name"])
    obs.append(d["obs"])
    eng.append(d["eng"])
    ml.append(d["ml"])
    meta.append(d["meta"])
    in_ad.append(bool(d.get("in_ad", True)))
    Xho.append(fv)
Xho = np.vstack(Xho)
obs = np.asarray(obs)
eng = np.asarray(eng)
ml = np.asarray(ml)
meta = np.asarray(meta)
n = len(names)
print(f"Holdout: {n} drugs with features")

# ---- 3. Bootstrap ensemble -> U_boot -----------------------------------------
preds = np.zeros((K, n))
for k in range(K):
    idx = rng.integers(0, len(ytt), len(ytt))
    m = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED + k, n_jobs=4,
    )
    m.fit(Xtr[idx], ytt[idx])
    preds[k] = m.predict(Xho)
U_boot = preds.std(axis=0)          # per-drug epistemic SD, log10 units
clint_mean = preds.mean(axis=0)     # ensemble-mean log10(CLint)
print(f"U_boot range [{U_boot.min():.3f},{U_boot.max():.3f}] "
      f"median {np.median(U_boot):.3f}")

# ---- 4. AD distance -> U_ad ---------------------------------------------------
# Morgan bits = first 2048 features (0/1). Tanimoto on bit sets.
Btr = (Xtr[:, :2048] > 0.5)
Bho = (Xho[:, :2048] > 0.5)
tr_pop = Btr.sum(1).astype(float)
U_ad = np.zeros(n)
for i in range(n):
    inter = (Bho[i] & Btr).sum(1).astype(float)
    union = Bho[i].sum() + tr_pop - inter
    sim = np.where(union > 0, inter / union, 0.0)
    U_ad[i] = 1.0 - sim.max()       # distance to nearest training neighbour
print(f"U_ad range [{U_ad.min():.3f},{U_ad.max():.3f}] median {np.median(U_ad):.3f}")

# ---- 5. Residuals -------------------------------------------------------------
eps = 1e-12
r_meta = np.abs(np.log10(meta / obs + eps))
e_eng = np.abs(np.log10(eng / obs + eps))
e_ml = np.abs(np.log10(ml / obs + eps))
diff = e_eng - e_ml
mag = np.abs(np.log10(obs + eps))   # confound: obs magnitude


def boot_spearman(x, y, nb=10000):
    rho = stats.spearmanr(x, y).correlation
    idxs = rng.integers(0, len(x), (nb, len(x)))
    bs = np.array([stats.spearmanr(x[i], y[i]).correlation for i in idxs])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(rho), float(lo), float(hi)


def quartile_trend(u, resid):
    q = np.quantile(u, [0.25, 0.5, 0.75])
    bins = np.digitize(u, q)
    return [round(float(resid[bins == b].mean()), 3) if (bins == b).any() else None
            for b in range(4)]


targets = {"r_meta": r_meta, "e_eng": e_eng, "e_ml": e_ml, "diff": diff, "obs_mag": mag}
signals = {"U_boot": U_boot, "U_ad": U_ad, "clint_mean": clint_mean}

result = {"seed": SEED, "K": K, "n": n, "boot_resamples": 10000,
          "note": "gate: |rho|>=0.30 with CI excluding 0 = PASS", "corr": {}, "trend": {}}
print("\n=== Spearman rho (95% CI) ===")
for sname, sig in signals.items():
    result["corr"][sname] = {}
    for tname, t in targets.items():
        rho, lo, hi = boot_spearman(sig, t)
        result["corr"][sname][tname] = {"rho": round(rho, 3), "ci": [round(lo, 3), round(hi, 3)]}
        star = "  <-- PASS" if abs(rho) >= 0.30 and (lo > 0 or hi < 0) else ""
        print(f"  {sname:11s} vs {tname:8s}: rho={rho:+.3f} [{lo:+.3f},{hi:+.3f}]{star}")
    result["trend"][sname] = {
        t: quartile_trend(sig, targets[t]) for t in ("r_meta", "e_eng", "diff")
    }

# in_ad split of U (relation to DE-55 AD signal)
in_ad = np.asarray(in_ad)
result["u_boot_by_ad"] = {
    "in_ad_mean": round(float(U_boot[in_ad].mean()), 3),
    "out_ad_mean": round(float(U_boot[~in_ad].mean()), 3) if (~in_ad).any() else None,
    "n_out_ad": int((~in_ad).sum()),
}

# verdict
passed = any(
    abs(result["corr"][s][t]["rho"]) >= 0.30
    and (result["corr"][s][t]["ci"][0] > 0 or result["corr"][s][t]["ci"][1] < 0)
    for s in ("U_boot", "U_ad") for t in ("r_meta", "e_eng", "diff")
)
result["verdict"] = "PASS" if passed else "FAIL (null)"
print(f"\nVERDICT: {result['verdict']}")

json.dump(result, open(OUT, "w"), indent=2)
print("wrote", OUT)
