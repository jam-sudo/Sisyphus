"""Train F% (bioavailability) predictor from DrugBank absorption text.

Target: log10(F/100) in [-2, 0] range (F in 1-100%)
Features: 2057-element (Morgan FP + 9 descriptors)

Integration: Cmax_vdss_corrected = Cmax_vdss * F
"""
from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sisyphus.descriptors import compute_features
from sisyphus.validation.reference import load_reference

logging.disable(logging.WARNING)


def _ik14(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(m).split("-")[0] if m else None
    except Exception:
        return None


def main():
    df = pd.read_csv("data/training/bioavailability_v1.csv")
    print(f"Raw F data: {len(df)}")
    df["ik14"] = df.smiles.apply(_ik14)
    df = df[df.ik14.notna()]

    # Exclude holdout
    refs = load_reference()
    holdout_keys = set()
    for r in refs:
        if r.in_holdout:
            holdout_keys.add(r.name.lower().strip())
            ik = _ik14(r.smiles)
            if ik:
                holdout_keys.add(ik)
    df["name_lower"] = df["name"].astype(str).str.lower().str.strip()
    before = len(df)
    df = df[~(df.name_lower.isin(holdout_keys) | df.ik14.isin(holdout_keys))]
    print(f"After holdout exclusion: {len(df)} (removed {before - len(df)})")

    # Dedup by ik14
    df["log_f"] = np.log10(df.f_pct / 100.0)
    df = df.groupby("ik14").agg(
        smiles=("smiles", "first"),
        name=("name", "first"),
        log_f=("log_f", "mean"),
    ).reset_index()
    print(f"After dedup: {len(df)}")
    print(f"log(F): mean={df.log_f.mean():.2f}, std={df.log_f.std():.2f}, "
          f"range=[{df.log_f.min():.2f}, {df.log_f.max():.2f}]")

    # Compute features
    feats, valid = [], []
    for i, smi in enumerate(df.smiles):
        try:
            feats.append(compute_features(smi))
            valid.append(i)
        except Exception:
            continue
    X = np.vstack(feats)
    df = df.iloc[valid].reset_index(drop=True)
    y = df.log_f.to_numpy()
    print(f"Training: N={len(y)}")

    # Scaffold CV
    scaffolds = {}
    for i, smi in enumerate(df.smiles):
        m = Chem.MolFromSmiles(smi)
        if m:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            scaffolds.setdefault(s, []).append(i)
    sorted_s = sorted(scaffolds.items(), key=lambda x: -len(x[1]))
    folds = [[] for _ in range(5)]
    for j, (_, members) in enumerate(sorted_s):
        folds[j % 5].extend(members)
    r2s = []
    for f in range(5):
        test = folds[f]; train = [i for i in range(len(df)) if i not in test]
        if not test or not train: continue
        m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        m.fit(X[train], y[train])
        p = m.predict(X[test])
        ss_r = np.sum((y[test] - p) ** 2)
        ss_t = np.sum((y[test] - y[test].mean()) ** 2)
        if ss_t > 0:
            r2s.append(1 - ss_r / ss_t)
    print(f"5-fold scaffold CV R²: {np.mean(r2s):.3f}")

    # Train full
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X, y)
    p = model.predict(X)
    print(f"Train R²: {1 - np.sum((y-p)**2)/np.sum((y-y.mean())**2):.3f}")

    # Save
    model.get_booster().save_model("models/adme/xgboost_bioavailability.json")
    meta = {
        "n_train": int(len(y)),
        "scaffold_cv_r2": float(np.mean(r2s)),
        "target": "log10(F/100) where F is in percent",
        "source": "DrugBank absorption text regex parsing (holdout-excluded)",
        "notes": "F ranges 1-100%, log_f ∈ [-2, 0]",
    }
    Path("models/adme/xgboost_bioavailability_meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved model to models/adme/xgboost_bioavailability.json")


if __name__ == "__main__":
    main()
