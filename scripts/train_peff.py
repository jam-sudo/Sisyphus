#!/usr/bin/env python3
"""train_peff.py — Train XGBoost Peff predictor from TDC Caco2_Wang.

Data source:
  - TDC Caco2_Wang (~910 compounds): Y = log10(Papp cm/s)

Target: log10(Peff [x10^-4 cm/s]) = TDC_Y + 4
  - TDC Y=-5.0 → log10(Peff) = -1.0 → Peff = 0.1 x10^-4 cm/s
  - TDC Y=-4.0 → log10(Peff) =  0.0 → Peff = 1.0 x10^-4 cm/s

Holdout drugs excluded via canonical SMILES + InChIKey-14 + name matching.

Output: models/adme/xgboost_peff.json

Usage:
    python3 scripts/train_peff.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sisyphus.descriptors import compute_features  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = ROOT / "data" / "reference" / "clinical_pk.json"
OUTPUT_MODEL = ROOT / "models" / "adme" / "xgboost_peff.json"

# ---------------------------------------------------------------------------
# RDKit helpers (same as train_fup_v2.py)
# ---------------------------------------------------------------------------

def _canonical_smiles(smiles: str) -> str | None:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _inchikey_prefix(smiles: str) -> str | None:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = MolToInchi(mol)
    if inchi is None:
        return None
    ik = InchiToInchiKey(inchi)
    if ik is None:
        return None
    return ik[:14]


# ---------------------------------------------------------------------------
# Holdout exclusion (reused from train_fup_v2.py pattern)
# ---------------------------------------------------------------------------

def build_holdout_keys(holdout_names: list[str], clinical_pk: dict) -> dict:
    canonical_smiles: set[str] = set()
    inchikey_prefixes: set[str] = set()
    names: set[str] = set()

    drugs = clinical_pk.get("drugs", {})
    for name in holdout_names:
        names.add(name.lower())
        entry = drugs.get(name) or drugs.get(name.replace(" ", "_"))
        if entry is None:
            continue
        smiles = entry.get("smiles", "")
        if not smiles:
            continue
        csmi = _canonical_smiles(smiles)
        if csmi:
            canonical_smiles.add(csmi)
        ik = _inchikey_prefix(smiles)
        if ik:
            inchikey_prefixes.add(ik)

    log.info(
        "Holdout keys: %d canonical SMILES, %d InChIKey prefixes, %d names",
        len(canonical_smiles), len(inchikey_prefixes), len(names),
    )
    return {
        "canonical_smiles": canonical_smiles,
        "inchikey_prefixes": inchikey_prefixes,
        "names": names,
    }


def is_holdout(smiles: str, drug_id: str, keys: dict) -> bool:
    if drug_id.lower() in keys["names"]:
        return True
    csmi = _canonical_smiles(smiles)
    if csmi and csmi in keys["canonical_smiles"]:
        return True
    ik = _inchikey_prefix(smiles)
    if ik and ik in keys["inchikey_prefixes"]:
        return True
    return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_caco2_wang(holdout_keys: dict) -> pd.DataFrame:
    """Load TDC Caco2_Wang, convert to log10(Peff [x10^-4 cm/s]), exclude holdout."""
    from tdc.single_pred import ADME

    log.info("Loading TDC Caco2_Wang ...")
    data = ADME(name="Caco2_Wang")
    df = data.get_data()
    log.info("Loaded %d rows", len(df))

    # Y = log10(Papp cm/s). Convert to log10(Peff [x10^-4 cm/s]) = Y + 4
    df["log_peff"] = df["Y"] + 4.0
    df["drug_id"] = df["Drug_ID"].astype(str)
    df["smiles"] = df["Drug"].astype(str)

    # Canonical SMILES
    log.info("Canonicalizing SMILES ...")
    df["canonical_smiles"] = df["smiles"].apply(_canonical_smiles)
    df = df.dropna(subset=["canonical_smiles"])

    # Exclude holdout
    n_before = len(df)
    holdout_mask = df.apply(
        lambda row: is_holdout(row["smiles"], row["drug_id"], holdout_keys), axis=1
    )
    df = df[~holdout_mask].reset_index(drop=True)
    n_excluded = n_before - len(df)
    log.info("Excluded %d holdout drugs, %d remain", n_excluded, len(df))

    # Dedup by canonical SMILES (keep first)
    n_before = len(df)
    df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    log.info("After dedup: %d compounds (removed %d)", len(df), n_before - len(df))

    return df[["canonical_smiles", "drug_id", "log_peff"]].copy()


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    log.info("Computing features for %d compounds ...", len(df))
    features, targets, valid_indices = [], [], []
    failed = 0

    for i, row in df.iterrows():
        try:
            feat = compute_features(row["canonical_smiles"])
            features.append(feat)
            targets.append(row["log_peff"])
            valid_indices.append(i)
        except Exception:
            failed += 1

    if failed:
        log.warning("Feature computation failed for %d compounds", failed)

    X = np.array(features, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    log.info("Feature matrix: %s, target: %s", X.shape, y.shape)
    return X, y, valid_indices


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    log.info("Training XGBoost Peff model (500 trees, max_depth=6, lr=0.05) ...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def evaluate_cv(X: np.ndarray, y: np.ndarray) -> dict:
    log.info("Running 5-fold cross-validation ...")
    cv_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    y_pred = cross_val_predict(cv_model, X, y, cv=5)

    # RMSE on log10 scale
    rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))

    # R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAE on log10 scale
    mae = float(np.mean(np.abs(y_pred - y)))

    # AAFE (on linear scale Peff)
    peff_pred = 10.0 ** y_pred
    peff_true = 10.0 ** y
    eps = 1e-10
    log_ratio = np.abs(np.log10(np.maximum(peff_pred, eps) / np.maximum(peff_true, eps)))
    aafe_val = float(10 ** np.mean(log_ratio))

    log.info("CV — RMSE: %.3f | R²: %.3f | MAE: %.3f | AAFE: %.3f", rmse, r2, mae, aafe_val)
    return {"rmse": rmse, "r2": r2, "mae": mae, "aafe": aafe_val, "n": len(y)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log.info("=" * 60)
    log.info("train_peff.py — XGBoost Peff predictor (Caco2_Wang)")
    log.info("=" * 60)

    # Holdout keys
    with open(HOLDOUT_JSON) as f:
        holdout_names = json.load(f)["holdout"]
    with open(CLINICAL_PK_JSON) as f:
        clinical_pk = json.load(f)

    holdout_keys = build_holdout_keys(holdout_names, clinical_pk)

    # Load data
    df = load_caco2_wang(holdout_keys)

    # Features
    X, y, _ = compute_feature_matrix(df)

    if len(X) < 100:
        log.error("Insufficient data: %d compounds. Aborting.", len(X))
        return 1

    # CV
    cv = evaluate_cv(X, y)

    # Train final model
    model = train_model(X, y)

    # Save
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_MODEL))
    log.info("Model saved to %s", OUTPUT_MODEL)

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Training compounds : %d", cv["n"])
    log.info("  CV RMSE (log10)    : %.3f", cv["rmse"])
    log.info("  CV R²              : %.3f", cv["r2"])
    log.info("  CV MAE (log10)     : %.3f", cv["mae"])
    log.info("  CV AAFE            : %.3f", cv["aafe"])
    log.info("  Model path         : %s", OUTPUT_MODEL)
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
