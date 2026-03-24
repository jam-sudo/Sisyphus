"""train_fup_v2.py — Train XGBoost fup v2 model with logit transform.

Data sources:
  - TDC PPBR_AZ (PyTDC package): ~1,614 compounds, Y = % bound (0-100)
  - DrugBank pk_data.csv + drugs.csv: curated fup values (fraction unbound, 0-1)

DrugBank overrides TDC when the same drug appears in both (more curated).
Holdout drugs are excluded via 3-key matching (canonical SMILES, InChIKey-14,
lowercase name). Post-training contamination check verifies 0 holdout drugs
remain.

Target transform: logit (logit(fup) for training, sigmoid for back-transform)

Output: models/adme/xgboost_fup_v2.json

Usage:
    python3 scripts/train_fup_v2.py
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
DRUGBANK_PK_CSV = ROOT / "data" / "drugbank" / "pk_data.csv"
DRUGBANK_DRUGS_CSV = ROOT / "data" / "drugbank" / "drugs.csv"
OUTPUT_MODEL = ROOT / "models" / "adme" / "xgboost_fup_v2.json"

# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logit transform: log(x / (1 - x)), clipped to [eps, 1-eps]."""
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Inverse logit: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# RDKit helpers (mirrors holdout_audit.py)
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
# Holdout key construction (mirrors holdout_audit.py)
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

def load_tdc_ppbr_az(holdout_keys: dict) -> pd.DataFrame:
    """Load TDC PPBR_AZ and convert to fup fraction, excluding holdout."""
    from tdc.single_pred import ADME

    log.info("Loading TDC PPBR_AZ ...")
    data = ADME(name="PPBR_AZ")
    df = data.get_data()
    log.info("Loaded %d rows from TDC PPBR_AZ", len(df))

    # Convert % bound → fup fraction unbound
    df["fup"] = (100.0 - df["Y"]) / 100.0
    df["drug_id"] = df["Drug_ID"].astype(str)
    df["smiles"] = df["Drug"].astype(str)

    # Compute canonical SMILES for dedup
    log.info("Canonicalizing SMILES for TDC data ...")
    df["canonical_smiles"] = df["smiles"].apply(_canonical_smiles)
    df = df.dropna(subset=["canonical_smiles"])

    # Exclude holdout
    n_before = len(df)
    holdout_mask = df.apply(
        lambda row: is_holdout(row["smiles"], row["drug_id"], holdout_keys), axis=1
    )
    df = df[~holdout_mask].reset_index(drop=True)
    n_excluded = n_before - len(df)
    log.info("TDC: excluded %d holdout drugs, %d remain", n_excluded, len(df))

    return df[["canonical_smiles", "drug_id", "fup"]].copy()


def load_drugbank_fup(holdout_keys: dict) -> pd.DataFrame:
    """Load DrugBank protein_binding fup values, excluding holdout."""
    log.info("Loading DrugBank fup data ...")

    # Load pk data — keep only protein_binding with valid fup
    pk = pd.read_csv(DRUGBANK_PK_CSV)
    pb = pk[(pk["field"] == "protein_binding") & (pk["parsed_unit"] == "fup")].copy()
    pb = pb.dropna(subset=["parsed_value"])
    log.info("DrugBank: %d protein_binding rows with parsed fup", len(pb))

    # Load drugs for canonical_smiles
    drugs = pd.read_csv(DRUGBANK_DRUGS_CSV, usecols=["drugbank_id", "name", "canonical_smiles"])
    drugs = drugs.rename(columns={"name": "drug_name_db"})

    # Merge on drugbank_id
    merged = pb.merge(drugs, on="drugbank_id", how="inner")
    merged = merged.dropna(subset=["canonical_smiles"])
    log.info("DrugBank: %d rows after joining canonical SMILES", len(merged))

    # fup is already fraction unbound in parsed_value
    merged["fup"] = pd.to_numeric(merged["parsed_value"], errors="coerce")
    merged = merged.dropna(subset=["fup"])

    # Exclude holdout
    n_before = len(merged)
    holdout_mask = merged.apply(
        lambda row: is_holdout(
            row["canonical_smiles"],
            str(row.get("drug_name_db", "")),
            holdout_keys,
        ),
        axis=1,
    )
    merged = merged[~holdout_mask].reset_index(drop=True)
    n_excluded = n_before - len(merged)
    log.info("DrugBank: excluded %d holdout drugs, %d remain", n_excluded, len(merged))

    # Keep one row per drugbank_id (first row, parsed_value is most reliable)
    merged = merged.drop_duplicates(subset=["drugbank_id"], keep="first")
    log.info("DrugBank: %d unique compounds after dedup by drugbank_id", len(merged))

    return merged[["canonical_smiles", "drug_name_db", "fup"]].rename(
        columns={"drug_name_db": "drug_id"}
    )


def merge_datasets(tdc_df: pd.DataFrame, db_df: pd.DataFrame) -> pd.DataFrame:
    """Merge TDC and DrugBank, with DrugBank overriding TDC for same canonical SMILES."""
    log.info("Merging TDC (%d) and DrugBank (%d) datasets ...", len(tdc_df), len(db_df))

    # Mark sources
    tdc_df = tdc_df.copy()
    tdc_df["source"] = "tdc"

    db_df = db_df.copy()
    db_df["source"] = "drugbank"

    # Combine — DrugBank first so it wins in drop_duplicates(keep='first')
    combined = pd.concat([db_df, tdc_df], ignore_index=True)

    # Dedup by canonical SMILES, DrugBank wins
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["canonical_smiles"], keep="first")
    n_dedup = n_before - len(combined)
    log.info("Dedup: removed %d duplicates (DrugBank override applied)", n_dedup)

    # Filter: 0.001 ≤ fup ≤ 0.999
    n_before = len(combined)
    combined = combined[(combined["fup"] >= 0.001) & (combined["fup"] <= 0.999)]
    n_filtered = n_before - len(combined)
    log.info("Filter [0.001, 0.999]: removed %d out-of-range fup values", n_filtered)

    log.info(
        "Final combined dataset: %d compounds (TDC: %d, DrugBank: %d)",
        len(combined),
        (combined["source"] == "tdc").sum(),
        (combined["source"] == "drugbank").sum(),
    )
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Post-training contamination verification
# ---------------------------------------------------------------------------

def verify_no_holdout_contamination(
    df: pd.DataFrame, holdout_keys: dict
) -> int:
    """Re-verify training set contains no holdout drugs. Returns hit count."""
    log.info("Running post-training holdout contamination check ...")
    hits = 0
    for _, row in df.iterrows():
        if is_holdout(row["canonical_smiles"], str(row["drug_id"]), holdout_keys):
            log.error(
                "CONTAMINATION: %s (%s) matched holdout set",
                row["drug_id"],
                row["canonical_smiles"],
            )
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Compute feature matrix and logit-transformed target.

    Returns:
        X: float64 array shape (n_valid, 2057)
        y: float64 array shape (n_valid,) — logit(fup)
        valid_indices: list of indices into df for rows that succeeded
    """
    log.info("Computing features for %d compounds ...", len(df))
    features = []
    targets = []
    valid_indices = []
    failed = 0

    for i, row in df.iterrows():
        try:
            feat = compute_features(row["canonical_smiles"])
            features.append(feat)
            targets.append(logit(np.array([row["fup"]])[0]))
            valid_indices.append(i)
        except (ValueError, Exception) as exc:
            log.debug("Feature computation failed for idx %d (%s): %s", i, row["drug_id"], exc)
            failed += 1

    if failed:
        log.warning("Feature computation failed for %d compounds (skipped)", failed)

    X = np.array(features, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    log.info("Feature matrix: %s, target vector: %s", X.shape, y.shape)
    return X, y, valid_indices


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    """Train XGBoost regression model on logit-transformed fup."""
    log.info("Training XGBoost model (500 trees, max_depth=6, lr=0.05) ...")
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
    log.info("Training complete.")
    return model


def evaluate_cv(model: xgb.XGBRegressor, X: np.ndarray, y: np.ndarray) -> dict:
    """5-fold CV evaluation with AAFE and R² on back-transformed fup."""
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

    y_pred_logit = cross_val_predict(cv_model, X, y, cv=5)
    y_pred = sigmoid(y_pred_logit)
    y_true = sigmoid(y)

    # AAFE: mean(|log10(pred/obs)|) as fold-error, then 10^mean
    # Use log10-based AFE: geometric mean of |log10(pred/obs)|
    eps = 1e-6
    log_ratio = np.abs(np.log10(np.maximum(y_pred, eps) / np.maximum(y_true, eps)))
    aafe = float(10 ** np.mean(log_ratio))

    # R² on fup
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAE on fup
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Bias
    bias = float(np.mean(y_pred - y_true))

    log.info("CV results — AAFE: %.3f | R²: %.3f | MAE: %.4f | Bias: %.4f",
             aafe, r2, mae, bias)
    return {"aafe": aafe, "r2": r2, "mae": mae, "bias": bias, "n": len(y)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log.info("=" * 60)
    log.info("train_fup_v2.py — XGBoost fup v2 training")
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

    # Load data sources
    tdc_df = load_tdc_ppbr_az(holdout_keys)
    db_df = load_drugbank_fup(holdout_keys)

    # Merge with DrugBank override
    combined = merge_datasets(tdc_df, db_df)

    # Post-merge holdout verification
    n_hits = verify_no_holdout_contamination(combined, holdout_keys)
    if n_hits > 0:
        log.error(
            "CRITICAL: %d holdout drugs found in training set after merge. Aborting.", n_hits
        )
        return 1
    log.info("Contamination check passed: 0 holdout drugs in training set")

    # Compute features
    X, y, valid_indices = compute_feature_matrix(combined)

    if len(X) < 100:
        log.error("Insufficient training data: only %d compounds. Aborting.", len(X))
        return 1

    log.info("Training set: %d compounds", len(X))

    # Cross-validation evaluation
    cv_metrics = evaluate_cv(None, X, y)

    # Train final model on full dataset
    model = train_model(X, y)

    # Save model
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_MODEL))
    log.info("Model saved to %s", OUTPUT_MODEL)

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Training compounds : %d", len(X))
    log.info("  CV AAFE            : %.3f", cv_metrics["aafe"])
    log.info("  CV R²              : %.3f", cv_metrics["r2"])
    log.info("  CV MAE (fup)       : %.4f", cv_metrics["mae"])
    log.info("  CV Bias (fup)      : %.4f", cv_metrics["bias"])
    log.info("  Model path         : %s", OUTPUT_MODEL)
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
