"""Train CL + t_half XGBoost predictors from DrugBank.

Goal: Cmax_CL_thalf = dose × ln(2) / (CL × t_half)  [orthogonal to VDss track]
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

BW_KG = 70.0


def _ik14(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(m).split("-")[0] if m else None
    except Exception:
        return None


def load_holdout_keys():
    refs = load_reference()
    keys = set()
    for r in refs:
        if r.in_holdout:
            keys.add(r.name.lower().strip())
            ik = _ik14(r.smiles)
            if ik: keys.add(ik)
    return keys


def scaffold_cv(X, y, smiles_list, n_folds=5):
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            scaffolds.setdefault(s, []).append(i)
    sorted_s = sorted(scaffolds.items(), key=lambda x: -len(x[1]))
    folds = [[] for _ in range(n_folds)]
    for j, (_, m) in enumerate(sorted_s):
        folds[j % n_folds].extend(m)
    r2s = []
    for f in range(n_folds):
        test = folds[f]; train = [i for i in range(len(y)) if i not in test]
        if not test or not train: continue
        m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        m.fit(X[train], y[train])
        p = m.predict(X[test])
        ss_r = np.sum((y[test]-p)**2); ss_t = np.sum((y[test]-y[test].mean())**2)
        if ss_t > 0: r2s.append(1 - ss_r/ss_t)
    return float(np.mean(r2s)) if r2s else float("nan")


def build_dataset(field, target_log=True):
    """Build training data for 'clearance' or 'half_life' field."""
    pk = pd.read_csv("data/drugbank/pk_data.csv")
    drugs = pd.read_csv("data/drugbank/drugs.csv")
    src = pk[pk.field == field].copy()
    src = src[src.parsed_value.notna() & (src.parsed_value > 0)]
    print(f"DrugBank {field}: {len(src)} entries")

    # Normalize units - parsed_value is already in L/h for CL or hours for t_half
    # But need to check/parse unit strings
    # For CL: unit prefix in 'L/h' column - values are already in L/h, good
    # For half_life: may be h, min, days — let's normalize
    if field == "half_life":
        # Parse unit hints
        def to_hours(row):
            val = row.parsed_value
            unit = str(row.parsed_unit).lower() if row.parsed_unit else ""
            if "day" in unit:
                return val * 24.0
            if "min" in unit and "h" not in unit:
                return val / 60.0
            return val  # hours
        src["value_std"] = src.apply(to_hours, axis=1)
        unit_label = "hours"
    else:  # clearance
        # Already in L/h
        src["value_std"] = src.parsed_value
        unit_label = "L/h"

    # Join SMILES
    drugs_s = drugs[["drugbank_id", "canonical_smiles", "name"]].copy()
    m = src.merge(drugs_s, on="drugbank_id", how="left")
    m = m[m.canonical_smiles.notna() & (m.value_std > 0)]
    m["ik14"] = m.canonical_smiles.apply(_ik14)
    m = m[m.ik14.notna()]
    # Exclude holdout
    holdout_keys = load_holdout_keys()
    m["name_lower"] = m["name"].astype(str).str.lower().str.strip()
    before = len(m)
    m = m[~(m.name_lower.isin(holdout_keys) | m.ik14.isin(holdout_keys))]
    print(f"After holdout exclusion: {len(m)} (removed {before-len(m)})")
    # Dedup by ik14 (geometric mean)
    m["log_val"] = np.log10(m.value_std)
    d = m.groupby("ik14").agg(
        canonical_smiles=("canonical_smiles", "first"),
        name=("name", "first"),
        log_val=("log_val", "mean"),
        raw_val=("value_std", "mean"),
    ).reset_index()
    print(f"After dedup: {len(d)}")
    print(f"log({unit_label}): mean={d.log_val.mean():.2f}, std={d.log_val.std():.2f}, "
          f"range=[{d.log_val.min():.2f}, {d.log_val.max():.2f}]")
    return d


def train_model(df, model_name, description):
    feats, valid = [], []
    for i, smi in enumerate(df.canonical_smiles):
        try:
            feats.append(compute_features(smi))
            valid.append(i)
        except Exception:
            continue
    X = np.vstack(feats)
    df = df.iloc[valid].reset_index(drop=True)
    y = df.log_val.to_numpy()
    print(f"Training {model_name}: N={len(y)}")

    r2 = scaffold_cv(X, y, df.canonical_smiles.tolist(), 5)
    print(f"  5-fold scaffold CV R² = {r2:.3f}")

    m = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    m.fit(X, y)
    r2_train = 1 - np.sum((y-m.predict(X))**2) / np.sum((y-y.mean())**2)
    print(f"  Train R² = {r2_train:.3f}")

    path = f"models/adme/xgboost_{model_name}.json"
    m.get_booster().save_model(path)
    print(f"  Saved to {path}")
    return r2, r2_train, len(y)


def main():
    print("=== CL (clearance) model ===")
    df_cl = build_dataset("clearance")
    cl_r2, cl_train_r2, cl_n = train_model(df_cl, "clearance_v1", "Total clearance (L/h)")
    print()
    print("=== t_half model ===")
    df_th = build_dataset("half_life")
    th_r2, th_train_r2, th_n = train_model(df_th, "thalf_v1", "Terminal half-life (hours)")
    print()
    print("=== Summary ===")
    print(f"CL: N={cl_n}, scaffold CV R²={cl_r2:.3f}")
    print(f"t_half: N={th_n}, scaffold CV R²={th_r2:.3f}")

    out = {
        "clearance": {"n": cl_n, "cv_r2": cl_r2, "train_r2": cl_train_r2},
        "thalf": {"n": th_n, "cv_r2": th_r2, "train_r2": th_train_r2},
    }
    Path("data/validation/cl_thalf_training.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
