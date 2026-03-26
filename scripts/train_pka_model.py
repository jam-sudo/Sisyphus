"""train_pka_model.py — Train XGBoost pKa models (acidic + basic).

Data source:
  - DrugBank drugs.csv: ChemAxon-calculated pKa values (~9,974 acidic, ~10,987 basic)

Holdout drugs are excluded via 3-key matching (canonical SMILES, InChIKey-14,
lowercase name). Post-training contamination check verifies 0 holdout drugs
remain.

Features: Morgan FP (2048 bits) + 9 RDKit descriptors = 2057 features
CV: 5-fold Murcko scaffold split (grouping by chemical scaffold)
Models: Two separate XGBoost regressors for acidic and basic pKa

Output:
  - models/adme/xgboost_pka_acidic.json
  - models/adme/xgboost_pka_basic.json

Usage:
    python3 scripts/train_pka_model.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

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
DRUGBANK_DRUGS_CSV = ROOT / "data" / "drugbank" / "drugs.csv"
OUTPUT_MODEL_ACIDIC = ROOT / "models" / "adme" / "xgboost_pka_acidic.json"
OUTPUT_MODEL_BASIC = ROOT / "models" / "adme" / "xgboost_pka_basic.json"

# ---------------------------------------------------------------------------
# RDKit helpers (mirrors train_fup_v2.py)
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
# Holdout key construction (mirrors train_fup_v2.py)
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


def is_holdout(smiles: str, name: str, inchikey_14: str | None, keys: dict) -> bool:
    """Return True if the compound matches any holdout exclusion key."""
    # Key 3: lowercase name
    if name.lower() in keys["names"]:
        return True

    # Key 1: canonical SMILES
    csmi = _canonical_smiles(smiles)
    if csmi and csmi in keys["canonical_smiles"]:
        return True

    # Key 2: InChIKey-14 (use precomputed if available, else compute)
    if inchikey_14 and inchikey_14 in keys["inchikey_prefixes"]:
        return True
    if not inchikey_14:
        ik = _inchikey_prefix(smiles)
        if ik and ik in keys["inchikey_prefixes"]:
            return True

    return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pka_data(holdout_keys: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DrugBank pKa data, exclude holdout drugs.

    Returns:
        acidic_df: DataFrame with columns [canonical_smiles, name, pka_acidic]
        basic_df: DataFrame with columns [canonical_smiles, name, pka_basic]
    """
    log.info("Loading DrugBank drugs.csv ...")
    df = pd.read_csv(
        DRUGBANK_DRUGS_CSV,
        usecols=["drugbank_id", "name", "canonical_smiles", "inchikey_14",
                 "pka_acidic", "pka_basic"],
    )
    log.info("Loaded %d rows from DrugBank drugs.csv", len(df))

    # Drop rows without canonical SMILES
    df = df.dropna(subset=["canonical_smiles"])
    log.info("After dropping missing SMILES: %d rows", len(df))

    # Exclude holdout drugs
    n_before = len(df)
    holdout_mask = df.apply(
        lambda row: is_holdout(
            str(row["canonical_smiles"]),
            str(row.get("name", "")),
            str(row["inchikey_14"]) if pd.notna(row.get("inchikey_14")) else None,
            holdout_keys,
        ),
        axis=1,
    )
    df = df[~holdout_mask].reset_index(drop=True)
    n_excluded = n_before - len(df)
    log.info("Excluded %d holdout drugs, %d remain", n_excluded, len(df))

    # Split into acidic and basic subsets
    acidic_df = df.dropna(subset=["pka_acidic"])[
        ["canonical_smiles", "name", "pka_acidic"]
    ].copy()
    acidic_df = acidic_df.reset_index(drop=True)

    basic_df = df.dropna(subset=["pka_basic"])[
        ["canonical_smiles", "name", "pka_basic"]
    ].copy()
    basic_df = basic_df.reset_index(drop=True)

    log.info("Acidic pKa: %d compounds", len(acidic_df))
    log.info("Basic pKa: %d compounds", len(basic_df))

    return acidic_df, basic_df


# ---------------------------------------------------------------------------
# Post-training contamination verification
# ---------------------------------------------------------------------------


def verify_no_holdout_contamination(
    df: pd.DataFrame, holdout_keys: dict, label: str
) -> int:
    """Re-verify training set contains no holdout drugs. Returns hit count."""
    log.info("Running post-training holdout contamination check (%s) ...", label)
    hits = 0
    for _, row in df.iterrows():
        ik14 = str(row.get("inchikey_14", "")) if pd.notna(row.get("inchikey_14")) else None
        if is_holdout(
            str(row["canonical_smiles"]),
            str(row.get("name", "")),
            ik14,
            holdout_keys,
        ):
            log.error(
                "CONTAMINATION: %s (%s) matched holdout set",
                row.get("name", "?"),
                row["canonical_smiles"],
            )
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_feature_matrix(
    df: pd.DataFrame, smiles_col: str = "canonical_smiles"
) -> tuple[np.ndarray, list[str], list[int]]:
    """Compute feature matrix for all valid SMILES.

    Returns:
        X: float64 array shape (n_valid, 2057)
        valid_smiles: list of canonical SMILES for valid rows
        valid_indices: list of indices into df for rows that succeeded
    """
    log.info("Computing features for %d compounds ...", len(df))
    features = []
    valid_smiles: list[str] = []
    valid_indices: list[int] = []
    failed = 0

    for i, row in df.iterrows():
        smi = str(row[smiles_col])
        try:
            feat = compute_features(smi)
            features.append(feat)
            valid_smiles.append(smi)
            valid_indices.append(i)
        except (ValueError, Exception) as exc:
            log.debug(
                "Feature computation failed for idx %d (%s): %s",
                i, row.get("name", "?"), exc,
            )
            failed += 1

        if (len(valid_indices) + failed) % 2000 == 0:
            log.info("  Features: %d/%d", len(valid_indices) + failed, len(df))

    if failed:
        log.warning("Feature computation failed for %d compounds (skipped)", failed)

    X = np.array(features, dtype=np.float64)
    log.info("Feature matrix: %s", X.shape)
    return X, valid_smiles, valid_indices


# ---------------------------------------------------------------------------
# Scaffold split
# ---------------------------------------------------------------------------


def scaffold_split_indices(
    smiles_list: list[str], n_folds: int = 5, seed: int = 42
) -> list[list[int]]:
    """Generate n-fold scaffold split indices using Murcko scaffolds."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    scaffold_to_indices: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            scaffold = (
                MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
            )
        except Exception:
            scaffold = ""
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    rng = np.random.default_rng(seed)
    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for i, scaffold in enumerate(scaffolds):
        folds[i % n_folds].extend(scaffold_to_indices[scaffold])

    log.info(
        "Scaffold split: %d unique scaffolds → %d folds (sizes: %s)",
        len(scaffolds),
        n_folds,
        [len(f) for f in folds],
    )
    return folds


# ---------------------------------------------------------------------------
# Model training with scaffold CV
# ---------------------------------------------------------------------------


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    smiles_list: list[str],
    label: str,
) -> tuple[xgb.XGBRegressor, dict]:
    """Train XGBoost with 5-fold scaffold CV, report MAE/R2, return final model.

    Returns:
        model: XGBRegressor trained on ALL data
        metrics: dict with per-fold and overall MAE, R2
    """
    folds = scaffold_split_indices(smiles_list, n_folds=5)

    all_preds = np.zeros(len(y))
    all_true = np.zeros(len(y))
    fold_maes: list[float] = []
    fold_r2s: list[float] = []

    xgb_params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )

    for fold_idx in range(5):
        test_idx = folds[fold_idx]
        train_idx = [
            i for f_idx in range(5) if f_idx != fold_idx for i in folds[f_idx]
        ]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, verbose=False)

        preds = model.predict(X_test)
        mae = float(np.mean(np.abs(y_test - preds)))
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        fold_maes.append(mae)
        fold_r2s.append(r2)

        all_preds[test_idx] = preds
        all_true[test_idx] = y_test

        log.info(
            "  %s Fold %d: MAE=%.3f, R2=%.3f (N_train=%d, N_test=%d)",
            label, fold_idx + 1, mae, r2, len(train_idx), len(test_idx),
        )

    # Overall CV metrics
    overall_mae = float(np.mean(np.abs(all_true - all_preds)))
    overall_ss_res = np.sum((all_true - all_preds) ** 2)
    overall_ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    overall_r2 = float(1 - overall_ss_res / overall_ss_tot) if overall_ss_tot > 0 else 0.0

    log.info(
        "%s CV mean: MAE=%.3f (+-%.3f), R2=%.3f (+-%.3f)",
        label,
        np.mean(fold_maes), np.std(fold_maes),
        np.mean(fold_r2s), np.std(fold_r2s),
    )
    log.info("%s CV overall: MAE=%.3f, R2=%.3f", label, overall_mae, overall_r2)

    # Train final model on ALL data
    log.info("Training final %s model on all %d compounds ...", label, len(y))
    final_model = xgb.XGBRegressor(**xgb_params)
    final_model.fit(X, y, verbose=False)

    metrics = {
        "label": label,
        "n": len(y),
        "fold_maes": fold_maes,
        "fold_r2s": fold_r2s,
        "cv_mae_mean": float(np.mean(fold_maes)),
        "cv_mae_std": float(np.std(fold_maes)),
        "cv_r2_mean": float(np.mean(fold_r2s)),
        "cv_r2_std": float(np.std(fold_r2s)),
        "overall_mae": overall_mae,
        "overall_r2": overall_r2,
    }
    return final_model, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    log.info("=" * 60)
    log.info("train_pka_model.py — XGBoost pKa training (acidic + basic)")
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

    # Load pKa data
    acidic_df, basic_df = load_pka_data(holdout_keys)

    # Post-load holdout verification
    for label, df in [("acidic", acidic_df), ("basic", basic_df)]:
        n_hits = verify_no_holdout_contamination(df, holdout_keys, label)
        if n_hits > 0:
            log.error(
                "CRITICAL: %d holdout drugs found in %s training set. Aborting.",
                n_hits, label,
            )
            return 1
    log.info("Contamination check passed: 0 holdout drugs in both sets")

    # -----------------------------------------------------------------------
    # Acidic pKa model
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("ACIDIC pKa MODEL")
    log.info("-" * 60)

    X_acid, smiles_acid, valid_acid = compute_feature_matrix(acidic_df)
    y_acid = acidic_df.loc[valid_acid, "pka_acidic"].values.astype(np.float64)

    if len(X_acid) < 100:
        log.error("Insufficient acidic pKa data: %d compounds. Aborting.", len(X_acid))
        return 1

    acid_model, acid_metrics = train_and_evaluate(
        X_acid, y_acid, smiles_acid, "pKa_acidic"
    )

    # Save acidic model
    OUTPUT_MODEL_ACIDIC.parent.mkdir(parents=True, exist_ok=True)
    acid_model.save_model(str(OUTPUT_MODEL_ACIDIC))
    log.info("Acidic model saved to %s", OUTPUT_MODEL_ACIDIC)

    # -----------------------------------------------------------------------
    # Basic pKa model
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("BASIC pKa MODEL")
    log.info("-" * 60)

    X_basic, smiles_basic, valid_basic = compute_feature_matrix(basic_df)
    y_basic = basic_df.loc[valid_basic, "pka_basic"].values.astype(np.float64)

    if len(X_basic) < 100:
        log.error("Insufficient basic pKa data: %d compounds. Aborting.", len(X_basic))
        return 1

    basic_model, basic_metrics = train_and_evaluate(
        X_basic, y_basic, smiles_basic, "pKa_basic"
    )

    # Save basic model
    OUTPUT_MODEL_BASIC.parent.mkdir(parents=True, exist_ok=True)
    basic_model.save_model(str(OUTPUT_MODEL_BASIC))
    log.info("Basic model saved to %s", OUTPUT_MODEL_BASIC)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    print()
    for m in [acid_metrics, basic_metrics]:
        print(f"  {m['label']}:")
        print(f"    Training compounds : {m['n']}")
        print(f"    CV MAE (overall)   : {m['overall_mae']:.3f}")
        print(f"    CV R2 (overall)    : {m['overall_r2']:.3f}")
        print(f"    CV MAE (mean+-std) : {m['cv_mae_mean']:.3f} +- {m['cv_mae_std']:.3f}")
        print(f"    CV R2 (mean+-std)  : {m['cv_r2_mean']:.3f} +- {m['cv_r2_std']:.3f}")
        print(f"    Per-fold MAE       : {', '.join(f'{v:.3f}' for v in m['fold_maes'])}")
        print(f"    Per-fold R2        : {', '.join(f'{v:.3f}' for v in m['fold_r2s'])}")
        print()
    print(f"  Acidic model : {OUTPUT_MODEL_ACIDIC}")
    print(f"  Basic model  : {OUTPUT_MODEL_BASIC}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
