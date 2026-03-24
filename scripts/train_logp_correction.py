"""train_logp_correction.py — Train XGBoost logP residual correction model.

Learns a correction to Crippen (RDKit) logP using DrugBank experimental logP values.
Target: residual = experimental_logP - crippen_logP
Features: 6 descriptors [logP_crippen, MW, TPSA, HBD, HBA, rotatable_bonds]
  (NOT full 2057 — 6 features avoids overfitting on small dataset)

Holdout drugs are excluded via 3-key matching (canonical SMILES, InChIKey-14,
lowercase name), same pattern as train_fup_v2.py.

Output: models/adme/logp_correction.json

Usage:
    python3 scripts/train_logp_correction.py
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = ROOT / "data" / "reference" / "clinical_pk.json"
DRUGBANK_EXP_CSV = ROOT / "data" / "drugbank" / "experimental_properties.csv"
DRUGBANK_DRUGS_CSV = ROOT / "data" / "drugbank" / "drugs.csv"
OUTPUT_MODEL = ROOT / "models" / "adme" / "logp_correction.json"


# ---------------------------------------------------------------------------
# RDKit helpers (copied from train_fup_v2.py — scripts are standalone)
# ---------------------------------------------------------------------------

def _canonical_smiles(smiles: str) -> str | None:
    """Return RDKit canonical SMILES (isomericSmiles=True), or None on failure."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _inchikey_prefix(smiles: str) -> str | None:
    """Return first 14 characters of InChIKey (connectivity block), or None."""
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
# Holdout key construction (copied from train_fup_v2.py — scripts are standalone)
# ---------------------------------------------------------------------------

def build_holdout_keys(holdout_names: list[str], clinical_pk: dict) -> dict:
    """Build three exclusion key sets from holdout drug entries.

    Returns dict with keys:
        canonical_smiles: set[str]
        inchikey_prefixes: set[str]
        names: set[str]
    """
    canonical_smiles: set[str] = set()
    inchikey_prefixes: set[str] = set()
    names: set[str] = set()

    drugs = clinical_pk.get("drugs", {})
    missing_smiles: list[str] = []

    for name in holdout_names:
        names.add(name.lower())
        entry = drugs.get(name) or drugs.get(name.replace(" ", "_"))
        if entry is None:
            missing_smiles.append(name)
            continue

        smiles = entry.get("smiles", "")
        if not smiles:
            missing_smiles.append(name)
            continue

        csmi = _canonical_smiles(smiles)
        if csmi:
            canonical_smiles.add(csmi)

        ik = _inchikey_prefix(smiles)
        if ik:
            inchikey_prefixes.add(ik)

    if missing_smiles:
        log.warning(
            "No SMILES found in clinical_pk.json for %d holdout drugs "
            "(will use name matching only): %s",
            len(missing_smiles),
            missing_smiles[:10],
        )

    log.info(
        "Holdout keys built: %d canonical SMILES, %d InChIKey prefixes, %d names",
        len(canonical_smiles),
        len(inchikey_prefixes),
        len(names),
    )
    return {
        "canonical_smiles": canonical_smiles,
        "inchikey_prefixes": inchikey_prefixes,
        "names": names,
    }


def is_holdout(smiles: str, drug_id: str, keys: dict) -> bool:
    """Return True if the compound matches any holdout exclusion key."""
    # Key 3: lowercase name
    if drug_id.lower() in keys["names"]:
        return True

    # Key 1: canonical SMILES
    csmi = _canonical_smiles(smiles)
    if csmi and csmi in keys["canonical_smiles"]:
        return True

    # Key 2: InChIKey prefix
    ik = _inchikey_prefix(smiles)
    if ik and ik in keys["inchikey_prefixes"]:
        return True

    return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_logp_data(holdout_keys: dict) -> pd.DataFrame:
    """Load DrugBank experimental logP and match to SMILES via drugs.csv.

    Returns DataFrame with columns:
        canonical_smiles, drug_name, experimental_logp, drugbank_id
    """
    log.info("Loading DrugBank experimental logP ...")

    # Load experimental properties — keep only logP rows
    exp = pd.read_csv(DRUGBANK_EXP_CSV)
    logp_rows = exp[exp["property"] == "logP"].copy()
    log.info("DrugBank experimental: %d logP rows", len(logp_rows))

    # Parse value as float
    logp_rows["experimental_logp"] = pd.to_numeric(logp_rows["value"], errors="coerce")
    logp_rows = logp_rows.dropna(subset=["experimental_logp"])
    log.info("After float parsing: %d valid logP values", len(logp_rows))

    # Filter to plausible drug-like logP range [-10, 15] — removes clear data errors
    # (e.g., DB00461 Nabumetone has value=2400, DB11581 Venetoclax has value=99)
    n_before = len(logp_rows)
    logp_rows = logp_rows[(logp_rows["experimental_logp"] >= -10.0) & (logp_rows["experimental_logp"] <= 15.0)]
    n_filtered = n_before - len(logp_rows)
    if n_filtered > 0:
        log.info("Filtered %d out-of-range logP values (outside [-10, 15]), %d remain",
                 n_filtered, len(logp_rows))

    # Load drugs for canonical SMILES — rename to avoid collision with exp drug_name
    drugs = pd.read_csv(DRUGBANK_DRUGS_CSV, usecols=["drugbank_id", "name", "canonical_smiles"])
    drugs = drugs.rename(columns={"name": "drug_name_drugs"})

    # Merge on drugbank_id (experimental_properties already has drug_name col)
    merged = logp_rows.merge(drugs, on="drugbank_id", how="inner")
    merged = merged.dropna(subset=["canonical_smiles"])
    # Use drug_name from experimental_properties (already present), drop the duplicate
    merged = merged.drop(columns=["drug_name_drugs"], errors="ignore")
    log.info("After joining SMILES: %d rows", len(merged))

    # One logP per drug — keep first (DrugBank is curated, first entry is primary)
    merged = merged.drop_duplicates(subset=["drugbank_id"], keep="first")
    log.info("After dedup by drugbank_id: %d unique compounds", len(merged))

    # Exclude holdout
    n_before = len(merged)
    holdout_mask = merged.apply(
        lambda row: is_holdout(
            row["canonical_smiles"],
            str(row.get("drug_name", "")),
            holdout_keys,
        ),
        axis=1,
    )
    merged = merged[~holdout_mask].reset_index(drop=True)
    n_excluded = n_before - len(merged)
    log.info("Excluded %d holdout drugs, %d remain", n_excluded, len(merged))

    return merged[["canonical_smiles", "drug_name", "experimental_logp", "drugbank_id"]].copy()


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_logp_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Compute 6-feature matrix and residual target.

    Features: [logP_crippen, MW, TPSA, HBD, HBA, rotatable_bonds]
    Target: residual = experimental_logP - crippen_logP

    Returns:
        X: float64 array shape (n_valid, 6)
        y: float64 array shape (n_valid,) — residuals
        crippen_logp: float64 array shape (n_valid,) — for RMSE comparison
        valid_indices: list of row indices that succeeded
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    log.info("Computing features for %d compounds ...", len(df))
    features = []
    targets = []
    crippen_vals = []
    valid_indices = []
    failed = 0

    for i, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row["canonical_smiles"])
            if mol is None:
                failed += 1
                continue

            logp_crippen = float(Descriptors.MolLogP(mol))
            mw = float(Descriptors.MolWt(mol))
            tpsa = float(Descriptors.TPSA(mol))
            hbd = float(Descriptors.NumHDonors(mol))
            hba = float(Descriptors.NumHAcceptors(mol))
            rot = float(Descriptors.NumRotatableBonds(mol))

            residual = float(row["experimental_logp"]) - logp_crippen

            features.append([logp_crippen, mw, tpsa, hbd, hba, rot])
            targets.append(residual)
            crippen_vals.append(logp_crippen)
            valid_indices.append(i)

        except Exception as exc:
            log.debug("Feature computation failed for idx %d (%s): %s", i, row["drug_name"], exc)
            failed += 1

    if failed:
        log.warning("Feature computation failed for %d compounds (skipped)", failed)

    X = np.array(features, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    crippen_arr = np.array(crippen_vals, dtype=np.float64)
    log.info("Feature matrix: %s, residual vector: %s", X.shape, y.shape)
    return X, y, crippen_arr, valid_indices


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    """Train XGBoost regression model on logP residuals."""
    log.info("Training XGBoost model (200 trees, max_depth=4, lr=0.05) ...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    log.info("Training complete.")
    return model


def evaluate_cv(X: np.ndarray, y: np.ndarray, crippen_logp: np.ndarray, exp_logp: np.ndarray) -> dict:
    """5-fold CV evaluation comparing Crippen RMSE vs corrected RMSE.

    Args:
        X: feature matrix
        y: residuals (experimental - crippen)
        crippen_logp: Crippen logP values for each compound
        exp_logp: experimental logP values

    Returns:
        dict with rmse_crippen, rmse_corrected, r2_residual
    """
    log.info("Running 5-fold cross-validation ...")

    cv_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    y_pred_residuals = cross_val_predict(cv_model, X, y, cv=5)

    # Crippen RMSE: compare crippen_logp vs experimental_logp
    crippen_errors = exp_logp - crippen_logp
    rmse_crippen = float(np.sqrt(np.mean(crippen_errors ** 2)))

    # Corrected RMSE: compare (crippen + predicted_residual) vs experimental_logp
    corrected_logp = crippen_logp + y_pred_residuals
    corrected_errors = exp_logp - corrected_logp
    rmse_corrected = float(np.sqrt(np.mean(corrected_errors ** 2)))

    # R² on residuals
    ss_res = float(np.sum((y - y_pred_residuals) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_residual = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAE
    mae_crippen = float(np.mean(np.abs(crippen_errors)))
    mae_corrected = float(np.mean(np.abs(corrected_errors)))

    log.info(
        "Crippen RMSE: %.3f | Corrected RMSE: %.3f (improvement: %.3f)",
        rmse_crippen, rmse_corrected, rmse_crippen - rmse_corrected,
    )
    log.info(
        "Crippen MAE:  %.3f | Corrected MAE:  %.3f",
        mae_crippen, mae_corrected,
    )
    log.info("Residual R²: %.3f", r2_residual)

    return {
        "rmse_crippen": rmse_crippen,
        "rmse_corrected": rmse_corrected,
        "mae_crippen": mae_crippen,
        "mae_corrected": mae_corrected,
        "r2_residual": r2_residual,
        "n": len(y),
    }


# ---------------------------------------------------------------------------
# Post-training contamination verification
# ---------------------------------------------------------------------------

def verify_no_holdout_contamination(df: pd.DataFrame, holdout_keys: dict) -> int:
    """Re-verify training set contains no holdout drugs. Returns hit count."""
    log.info("Running post-training holdout contamination check ...")
    hits = 0
    for _, row in df.iterrows():
        if is_holdout(row["canonical_smiles"], str(row["drug_name"]), holdout_keys):
            log.error(
                "CONTAMINATION: %s (%s) matched holdout set",
                row["drug_name"],
                row["canonical_smiles"],
            )
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log.info("=" * 60)
    log.info("train_logp_correction.py — XGBoost logP residual correction")
    log.info("=" * 60)

    # Load holdout keys
    log.info("Loading holdout reference data ...")
    with open(HOLDOUT_JSON) as f:
        holdout_data = json.load(f)
    holdout_names: list[str] = holdout_data["holdout"]
    log.info("Holdout set: %d drugs", len(holdout_names))

    with open(CLINICAL_PK_JSON) as f:
        clinical_pk = json.load(f)

    holdout_keys = build_holdout_keys(holdout_names, clinical_pk)

    # Load DrugBank experimental logP
    df = load_logp_data(holdout_keys)

    if len(df) < 50:
        log.error("Insufficient training data: only %d compounds. Aborting.", len(df))
        return 1

    # Post-load holdout verification
    n_hits = verify_no_holdout_contamination(df, holdout_keys)
    if n_hits > 0:
        log.error(
            "CRITICAL: %d holdout drugs found in training set. Aborting.", n_hits
        )
        return 1
    log.info("Contamination check passed: 0 holdout drugs in training set")

    # Compute features and residuals
    X, y_residuals, crippen_logp, valid_indices = compute_logp_features(df)

    if len(X) < 50:
        log.error("Insufficient valid features: only %d compounds. Aborting.", len(X))
        return 1

    # Recover experimental logP for the valid subset
    # valid_indices are the integer index labels from df (after reset_index), so use .loc
    valid_df = df.loc[valid_indices].reset_index(drop=True)
    exp_logp = valid_df["experimental_logp"].values.astype(np.float64)

    log.info("Training set: %d compounds with valid features", len(X))

    # Cross-validation evaluation
    cv_metrics = evaluate_cv(X, y_residuals, crippen_logp, exp_logp)

    # Train final model on full dataset
    model = train_model(X, y_residuals)

    # Save model
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_MODEL))
    log.info("Model saved to %s", OUTPUT_MODEL)

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Training compounds  : %d", len(X))
    log.info("  Crippen RMSE        : %.3f log units", cv_metrics["rmse_crippen"])
    log.info("  Corrected RMSE      : %.3f log units", cv_metrics["rmse_corrected"])
    log.info("  RMSE improvement    : %.3f log units", cv_metrics["rmse_crippen"] - cv_metrics["rmse_corrected"])
    log.info("  Crippen MAE         : %.3f log units", cv_metrics["mae_crippen"])
    log.info("  Corrected MAE       : %.3f log units", cv_metrics["mae_corrected"])
    log.info("  Residual R²         : %.3f", cv_metrics["r2_residual"])
    log.info("  Model path          : %s", OUTPUT_MODEL)
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
