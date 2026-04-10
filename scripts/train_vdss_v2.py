"""Retrain VDss XGBoost model with DrugBank volume_of_distribution data.

Current model: xgboost_vdss.json (unknown origin, 2057 features).
Goal: augment training with DrugBank 1044 VDss values, validate on holdout.

Strategy:
- Load DrugBank pk_data.csv, field='volume_of_distribution'
- Join to drugs.csv for canonical SMILES
- Normalize values: L → L/kg (assume 70kg adult)
- Exclude 107 holdout drugs (name-based + InChIKey-14 matching)
- Compute 2057-element features + log10(VDss)
- Train XGBoost with scaffold CV
- Compare vs current model
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem

from sisyphus.descriptors import compute_features

logging.disable(logging.WARNING)

ASSUMED_BODY_WEIGHT_KG = 70.0


def _inchikey14(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


def load_holdout_keys() -> set:
    """Load names + InChIKey-14 for all 107 holdout drugs."""
    from sisyphus.validation.reference import load_reference
    refs = load_reference()
    holdout = [r for r in refs if r.in_holdout]
    keys = set()
    for r in holdout:
        keys.add(r.name.lower().strip())
        ik14 = _inchikey14(r.smiles)
        if ik14:
            keys.add(ik14)
    return keys


def build_training_set() -> pd.DataFrame:
    """Load DrugBank VD data + merge with SMILES, exclude holdout."""
    pk = pd.read_csv("data/drugbank/pk_data.csv")
    drugs = pd.read_csv("data/drugbank/drugs.csv")
    vd = pk[pk.field == "volume_of_distribution"].copy()
    vd = vd[vd.parsed_value.notna() & vd.parsed_unit.notna()]
    print(f"DrugBank VD entries: {len(vd)}")

    # Normalize to L/kg
    def to_l_per_kg(row):
        val = row.parsed_value
        unit = row.parsed_unit
        if unit == "L/kg":
            return val
        elif unit == "L":
            return val / ASSUMED_BODY_WEIGHT_KG
        return None
    vd["vdss_L_per_kg"] = vd.apply(to_l_per_kg, axis=1)
    vd = vd[vd.vdss_L_per_kg.notna() & (vd.vdss_L_per_kg > 0)]
    print(f"After unit normalization: {len(vd)}")

    # Join to SMILES via drugbank_id
    drugs_short = drugs[["drugbank_id", "canonical_smiles", "name"]].copy()
    merged = vd.merge(drugs_short, on="drugbank_id", how="left")
    merged = merged[merged.canonical_smiles.notna()]
    print(f"After SMILES join: {len(merged)}")

    # InChIKey-14 for dedup + holdout exclusion
    merged["ik14"] = merged.canonical_smiles.apply(_inchikey14)
    merged = merged[merged.ik14.notna()]

    # Exclude holdout
    holdout_keys = load_holdout_keys()
    merged["name_lower"] = merged["name"].str.lower().str.strip()
    contam = merged[merged.name_lower.isin(holdout_keys) | merged.ik14.isin(holdout_keys)]
    print(f"Holdout contamination found: {len(contam)} entries")
    for _, row in contam.head(5).iterrows():
        print(f"  {row['name']} (ik14={row['ik14']})")
    merged = merged[~(merged.name_lower.isin(holdout_keys) | merged.ik14.isin(holdout_keys))]
    print(f"After holdout exclusion: {len(merged)}")

    # Dedup by InChIKey-14 (geometric mean of VDss)
    merged["log_vdss"] = np.log10(merged.vdss_L_per_kg)
    dedup = merged.groupby("ik14").agg(
        canonical_smiles=("canonical_smiles", "first"),
        name=("name", "first"),
        log_vdss=("log_vdss", "mean"),
        n_sources=("log_vdss", "count"),
    ).reset_index()
    print(f"After dedup (unique ik14): {len(dedup)}")
    print(f"log_vdss distribution: mean={dedup.log_vdss.mean():.2f}, "
          f"std={dedup.log_vdss.std():.2f}, "
          f"range=[{dedup.log_vdss.min():.2f}, {dedup.log_vdss.max():.2f}]")
    return dedup


def compute_features_batch(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """Compute features for a batch of SMILES, skipping failures."""
    feats = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        try:
            f = compute_features(smi)
            feats.append(f)
            valid_idx.append(i)
        except Exception:
            continue
    return np.vstack(feats), valid_idx


def scaffold_cv_r2(X: np.ndarray, y: np.ndarray, smiles_list: list[str], n_folds: int = 5) -> float:
    """Rough scaffold-stratified CV R²."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    # Compute scaffolds
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds.setdefault(s, []).append(i)
    # Sort scaffolds by size (large → small), assign to folds
    scaffold_list = sorted(scaffolds.items(), key=lambda x: -len(x[1]))
    fold_assignments = [[] for _ in range(n_folds)]
    for j, (s, members) in enumerate(scaffold_list):
        fold_idx = j % n_folds
        fold_assignments[fold_idx].extend(members)
    # Run CV
    r2_scores = []
    for f in range(n_folds):
        test_idx = fold_assignments[f]
        train_idx = [i for i in range(len(smiles_list)) if i not in test_idx]
        if not test_idx or not train_idx:
            continue
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        ss_res = np.sum((y_te - y_pred) ** 2)
        ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
        if ss_tot > 0:
            r2_scores.append(1 - ss_res / ss_tot)
    return float(np.mean(r2_scores)) if r2_scores else float("nan")


def main():
    print("=== Building VDss training set ===")
    df = build_training_set()
    print()
    print("=== Computing features ===")
    X, valid_idx = compute_features_batch(df.canonical_smiles.tolist())
    df = df.iloc[valid_idx].reset_index(drop=True)
    y = df.log_vdss.to_numpy()
    print(f"Training: N={len(y)}, features shape={X.shape}")

    # Scaffold CV
    print()
    print("=== Scaffold CV ===")
    r2 = scaffold_cv_r2(X, y, df.canonical_smiles.tolist(), n_folds=5)
    print(f"5-fold scaffold CV R² = {r2:.3f}")

    # Train full model
    print()
    print("=== Training full model ===")
    m = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        random_state=42, verbosity=0,
    )
    m.fit(X, y)
    # In-sample performance
    y_pred_train = m.predict(X)
    ss_res = np.sum((y - y_pred_train) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_train = 1 - ss_res / ss_tot
    print(f"Train R² = {r2_train:.3f}")

    # Save
    out_path = Path("models/adme/xgboost_vdss_v2.json")
    m.get_booster().save_model(str(out_path))
    print(f"Saved to {out_path}")

    # Metadata
    meta = {
        "n_train": int(len(y)),
        "scaffold_cv_r2": r2,
        "train_r2": float(r2_train),
        "log_vdss_stats": {
            "mean": float(np.mean(y)), "std": float(np.std(y)),
            "min": float(np.min(y)), "max": float(np.max(y)),
        },
        "source": "DrugBank volume_of_distribution (holdout-excluded)",
        "target": "log10(VDss) in L/kg",
        "normalization": "70 kg assumed body weight for L→L/kg conversion",
    }
    Path("models/adme/xgboost_vdss_v2_meta.json").write_text(json.dumps(meta, indent=2))

    # Also save training data for reproducibility
    df[["canonical_smiles", "name", "log_vdss", "ik14"]].to_csv(
        "data/training/vdss_v2_training.csv", index=False
    )
    print(f"Saved training data to data/training/vdss_v2_training.csv")


if __name__ == "__main__":
    main()
