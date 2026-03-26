"""train_clint_expanded.py — Train XGBoost CLint model on expanded dataset.

Data sources:
  - TDC Clearance_Hepatocyte_AZ: 1,213 compounds (CLint, uL/min/10^6 cells)
  - TDC Clearance_Microsome_AZ: 1,102 compounds (CLint, uL/min/mg protein)

Merge strategy:
  - Canonicalize all SMILES with RDKit
  - For drugs in BOTH datasets (by canonical SMILES): use Hepatocyte value
  - For drugs in only one dataset: use that value
  - Result: ~1,633 unique compounds

Holdout exclusion via 3-key matching (canonical SMILES, InChIKey-14, lowercase name).

Features: Morgan FP (2048 bits) + 9 RDKit descriptors = 2057 features
CV: 5-fold Murcko scaffold split
Target: log10(CLint), floored at log10(0.1) = -1.0

Trains two models for comparison:
  - Model A: Hepatocyte_AZ only (baseline, same as current)
  - Model B: Hepatocyte_AZ + Microsome_AZ (expanded)

Then benchmarks both against holdout set using the engine pipeline.

Usage:
    python3 scripts/train_clint_expanded.py
"""

from __future__ import annotations

import json
import logging
import shutil
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
OUTPUT_MODEL = ROOT / "models" / "adme" / "xgboost_clint.json"
BACKUP_MODEL = ROOT / "models" / "adme" / "xgboost_clint_omega.json.bak"

# ---------------------------------------------------------------------------
# RDKit helpers (mirrors train_pka_model.py)
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
# Holdout key construction (mirrors train_pka_model.py)
# ---------------------------------------------------------------------------


def build_holdout_keys(holdout_names: list[str], clinical_pk: dict) -> dict:
    """Build three exclusion key sets from holdout drug entries."""
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


def is_holdout(smiles: str, name: str, holdout_keys: dict) -> bool:
    """Return True if the compound matches any holdout exclusion key."""
    # Key 3: lowercase name
    if name.lower() in holdout_keys["names"]:
        return True

    # Key 1: canonical SMILES
    csmi = _canonical_smiles(smiles)
    if csmi and csmi in holdout_keys["canonical_smiles"]:
        return True

    # Key 2: InChIKey-14
    ik = _inchikey_prefix(smiles)
    if ik and ik in holdout_keys["inchikey_prefixes"]:
        return True

    return False


# ---------------------------------------------------------------------------
# TDC data loading
# ---------------------------------------------------------------------------


def load_tdc_datasets(holdout_keys: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TDC Clearance datasets, canonicalize SMILES, exclude holdout.

    Returns:
        hep_df: Hepatocyte_AZ only (after holdout exclusion)
        combined_df: Hepatocyte_AZ + Microsome_AZ merged (after holdout exclusion)
    """
    from tdc.single_pred import ADME

    # Load raw datasets
    log.info("Loading TDC Clearance_Hepatocyte_AZ ...")
    hep_raw = ADME(name="Clearance_Hepatocyte_AZ").get_data()
    log.info("  Loaded %d compounds from Hepatocyte_AZ", len(hep_raw))

    log.info("Loading TDC Clearance_Microsome_AZ ...")
    mic_raw = ADME(name="Clearance_Microsome_AZ").get_data()
    log.info("  Loaded %d compounds from Microsome_AZ", len(mic_raw))

    # Canonicalize SMILES
    log.info("Canonicalizing SMILES ...")
    hep_raw["canonical_smiles"] = hep_raw["Drug"].apply(_canonical_smiles)
    mic_raw["canonical_smiles"] = mic_raw["Drug"].apply(_canonical_smiles)

    hep_raw = hep_raw.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    mic_raw = mic_raw.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    log.info("  After canonicalization: Hep=%d, Mic=%d", len(hep_raw), len(mic_raw))

    # Check overlap
    hep_smiles = set(hep_raw["canonical_smiles"])
    mic_smiles = set(mic_raw["canonical_smiles"])
    overlap = hep_smiles & mic_smiles
    hep_only = hep_smiles - mic_smiles
    mic_only = mic_smiles - hep_smiles
    log.info(
        "Overlap: %d | Hep-only: %d | Mic-only: %d | Total unique: %d",
        len(overlap), len(hep_only), len(mic_only),
        len(hep_smiles | mic_smiles),
    )

    # Build Hepatocyte-only DataFrame (Model A)
    # Deduplicate by canonical SMILES (take mean if duplicates exist)
    hep_dedup = (
        hep_raw.groupby("canonical_smiles")
        .agg({"Y": "mean", "Drug_ID": "first"})
        .reset_index()
    )
    log.info("Hep dedup: %d unique compounds", len(hep_dedup))

    # Build combined DataFrame (Model B)
    # Priority: Hepatocyte value if available, else Microsome
    mic_dedup = (
        mic_raw.groupby("canonical_smiles")
        .agg({"Y": "mean", "Drug_ID": "first"})
        .reset_index()
    )

    # Start with all hepatocyte entries
    combined = hep_dedup.copy()
    # Add microsome-only entries
    mic_only_df = mic_dedup[~mic_dedup["canonical_smiles"].isin(hep_smiles)]
    combined = pd.concat([combined, mic_only_df], ignore_index=True)
    log.info("Combined (Hep priority): %d unique compounds", len(combined))

    # Holdout exclusion
    n_before_hep = len(hep_dedup)
    hep_mask = hep_dedup.apply(
        lambda row: is_holdout(
            str(row["canonical_smiles"]),
            str(row.get("Drug_ID", "")),
            holdout_keys,
        ),
        axis=1,
    )
    hep_clean = hep_dedup[~hep_mask].reset_index(drop=True)
    n_excl_hep = n_before_hep - len(hep_clean)
    log.info("Hep holdout exclusion: %d excluded, %d remain", n_excl_hep, len(hep_clean))

    n_before_comb = len(combined)
    comb_mask = combined.apply(
        lambda row: is_holdout(
            str(row["canonical_smiles"]),
            str(row.get("Drug_ID", "")),
            holdout_keys,
        ),
        axis=1,
    )
    comb_clean = combined[~comb_mask].reset_index(drop=True)
    n_excl_comb = n_before_comb - len(comb_clean)
    log.info(
        "Combined holdout exclusion: %d excluded, %d remain",
        n_excl_comb, len(comb_clean),
    )

    return hep_clean, comb_clean


# ---------------------------------------------------------------------------
# Feature computation (mirrors train_pka_model.py)
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
                i, row.get("Drug_ID", "?"), exc,
            )
            failed += 1

        if (len(valid_indices) + failed) % 500 == 0:
            log.info("  Features: %d/%d", len(valid_indices) + failed, len(df))

    if failed:
        log.warning("Feature computation failed for %d compounds (skipped)", failed)

    X = np.array(features, dtype=np.float64)
    log.info("Feature matrix: %s", X.shape)
    return X, valid_smiles, valid_indices


# ---------------------------------------------------------------------------
# Scaffold split (mirrors train_pka_model.py)
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
        "Scaffold split: %d unique scaffolds -> %d folds (sizes: %s)",
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
    """Train XGBoost with 5-fold scaffold CV, report MAE/R2, return final model."""
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
    overall_r2 = (
        float(1 - overall_ss_res / overall_ss_tot) if overall_ss_tot > 0 else 0.0
    )

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
# Benchmark comparison
# ---------------------------------------------------------------------------


def run_benchmark_comparison() -> None:
    """Run engine benchmarks with old and new CLint models, print comparison."""
    # Suppress noisy logs during benchmark
    logging.getLogger("sisyphus").setLevel(logging.WARNING)

    from sisyphus.predict import adme
    from sisyphus.validation.benchmark import run_benchmark

    # --- Experiment A: old model (backed up) ---
    log.info("=" * 60)
    log.info("BENCHMARK A: Original CLint model")
    log.info("=" * 60)

    # Restore original model
    if BACKUP_MODEL.exists():
        shutil.copy2(BACKUP_MODEL, OUTPUT_MODEL)
        adme._model_cache.clear()
        log.info("Restored original model from backup")
    else:
        log.warning("No backup found — current model IS the original")

    result_a = run_benchmark(holdout_only=True)

    # --- Experiment B: new model ---
    log.info("=" * 60)
    log.info("BENCHMARK B: Expanded CLint model")
    log.info("=" * 60)

    # Restore new model (saved with .json extension during training)
    new_model_path = ROOT / "models" / "adme" / "xgboost_clint_expanded.json"
    if new_model_path.exists():
        shutil.copy2(new_model_path, OUTPUT_MODEL)
        adme._model_cache.clear()
        log.info("Installed expanded model")
    else:
        log.error("No .new model found — cannot benchmark expanded model")
        return

    result_b = run_benchmark(holdout_only=True)

    # --- Print comparison ---
    print()
    print("=" * 60)
    print("  ENGINE BENCHMARK COMPARISON")
    print("=" * 60)
    print()
    print(f"{'':20s} {'Old CLint':>12s}  {'New CLint':>12s}  {'Delta':>8s}")
    print("-" * 60)

    def _fmt(val: float | None) -> str:
        return f"{val:.3f}" if val is not None else "N/A"

    def _delta(a: float | None, b: float | None) -> str:
        if a is not None and b is not None:
            d = b - a
            return f"{d:+.3f}"
        return "N/A"

    print(
        f"{'Engine AAFE':20s} {_fmt(result_a.engine_aafe):>12s}  "
        f"{_fmt(result_b.engine_aafe):>12s}  "
        f"{_delta(result_a.engine_aafe, result_b.engine_aafe):>8s}"
    )
    print(
        f"{'Meta AAFE':20s} {_fmt(result_a.aafe):>12s}  "
        f"{_fmt(result_b.aafe):>12s}  "
        f"{_delta(result_a.aafe, result_b.aafe):>8s}"
    )
    print(
        f"{'%2-fold':20s} {result_a.pct_2fold:>11.1f}%  "
        f"{result_b.pct_2fold:>11.1f}%  "
        f"{result_b.pct_2fold - result_a.pct_2fold:>+7.1f}%"
    )
    print(f"{'N drugs':20s} {result_a.n_drugs:>12d}  {result_b.n_drugs:>12d}")
    print()

    # --- Decide which model to keep ---
    old_engine = result_a.engine_aafe or float("inf")
    new_engine = result_b.engine_aafe or float("inf")
    old_meta = result_a.aafe
    new_meta = result_b.aafe

    # Meta AAFE is the production metric. Only keep expanded model if meta improves.
    # Engine-only improvement without meta improvement means the meta-learner
    # compensated in the old regime — swapping hurts overall.
    if new_meta < old_meta and new_engine <= old_engine + 0.05:
        print("  DECISION: New expanded model is BETTER. Keeping it.")
        # new model is already in place (from Experiment B)
        new_model_path.unlink(missing_ok=True)
    else:
        print("  DECISION: Old model is BETTER (or no improvement). Restoring original.")
        shutil.copy2(BACKUP_MODEL, OUTPUT_MODEL)
        new_model_path.unlink(missing_ok=True)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    log.info("=" * 60)
    log.info("train_clint_expanded.py — XGBoost CLint training (Hep + Mic)")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Load holdout keys
    # ------------------------------------------------------------------
    log.info("Loading holdout reference data ...")
    with open(HOLDOUT_JSON) as f:
        holdout_data = json.load(f)
    # Combine both train and holdout lists for full exclusion
    holdout_names: list[str] = holdout_data.get("holdout", [])
    # Also exclude training drugs (they appear in clinical_pk.json reference)
    all_ref_names = holdout_names + holdout_data.get("train", [])
    log.info("Reference drugs to exclude: %d holdout + %d train = %d total",
             len(holdout_names), len(holdout_data.get("train", [])), len(all_ref_names))

    with open(CLINICAL_PK_JSON) as f:
        clinical_pk = json.load(f)

    holdout_keys = build_holdout_keys(all_ref_names, clinical_pk)

    # ------------------------------------------------------------------
    # Load TDC data
    # ------------------------------------------------------------------
    hep_df, combined_df = load_tdc_datasets(holdout_keys)

    # ------------------------------------------------------------------
    # Feature computation: Hepatocyte only
    # ------------------------------------------------------------------
    log.info("-" * 60)
    log.info("MODEL A: Hepatocyte-only CLint")
    log.info("-" * 60)

    X_hep, smiles_hep, valid_hep = compute_feature_matrix(hep_df)
    # Target: log10(CLint), floored at log10(0.1) = -1.0
    y_hep_raw = hep_df.loc[valid_hep, "Y"].values.astype(np.float64)
    y_hep = np.log10(np.maximum(y_hep_raw, 0.1))

    if len(X_hep) < 100:
        log.error("Insufficient Hep data: %d compounds. Aborting.", len(X_hep))
        return 1

    model_a, metrics_a = train_and_evaluate(X_hep, y_hep, smiles_hep, "CLint_Hep")

    # ------------------------------------------------------------------
    # Feature computation: Combined
    # ------------------------------------------------------------------
    log.info("-" * 60)
    log.info("MODEL B: Combined Hepatocyte + Microsome CLint")
    log.info("-" * 60)

    X_comb, smiles_comb, valid_comb = compute_feature_matrix(combined_df)
    y_comb_raw = combined_df.loc[valid_comb, "Y"].values.astype(np.float64)
    y_comb = np.log10(np.maximum(y_comb_raw, 0.1))

    if len(X_comb) < 100:
        log.error("Insufficient combined data: %d compounds. Aborting.", len(X_comb))
        return 1

    model_b, metrics_b = train_and_evaluate(X_comb, y_comb, smiles_comb, "CLint_Hep+Mic")

    # ------------------------------------------------------------------
    # Print CLint model comparison
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  CLint MODEL COMPARISON")
    print("=" * 60)
    print()
    print(f"{'':20s} {'Model A (Hep)':>15s}  {'Model B (Hep+Mic)':>17s}")
    print("-" * 60)
    print(f"{'N (train)':20s} {metrics_a['n']:>15d}  {metrics_b['n']:>17d}")
    print(f"{'CV R2':20s} {metrics_a['overall_r2']:>15.3f}  {metrics_b['overall_r2']:>17.3f}")
    print(f"{'CV MAE':20s} {metrics_a['overall_mae']:>15.3f}  {metrics_b['overall_mae']:>17.3f}")
    print(
        f"{'CV R2 (mean+-std)':20s} "
        f"{metrics_a['cv_r2_mean']:>6.3f}+-{metrics_a['cv_r2_std']:.3f}  "
        f"{metrics_b['cv_r2_mean']:>8.3f}+-{metrics_b['cv_r2_std']:.3f}"
    )
    print(
        f"{'CV MAE (mean+-std)':20s} "
        f"{metrics_a['cv_mae_mean']:>6.3f}+-{metrics_a['cv_mae_std']:.3f}  "
        f"{metrics_b['cv_mae_mean']:>8.3f}+-{metrics_b['cv_mae_std']:.3f}"
    )
    print()

    # ------------------------------------------------------------------
    # Backup original model and save both
    # ------------------------------------------------------------------
    if OUTPUT_MODEL.exists() and not BACKUP_MODEL.exists():
        shutil.copy2(OUTPUT_MODEL, BACKUP_MODEL)
        log.info("Backed up original model to %s", BACKUP_MODEL)
    elif OUTPUT_MODEL.exists():
        log.info("Backup already exists at %s", BACKUP_MODEL)

    # Save Model B (expanded) as temporary file for benchmark comparison
    # IMPORTANT: must use .json extension so XGBoost saves in JSON format (not UBJSON)
    new_model_path = ROOT / "models" / "adme" / "xgboost_clint_expanded.json"
    model_b.save_model(str(new_model_path))
    log.info("Saved expanded model to %s", new_model_path)

    # Also save Model A (hep-only retrained) for reference
    model_a_path = ROOT / "models" / "adme" / "xgboost_clint_hep_retrained.json"
    model_a.save_model(str(model_a_path))
    log.info("Saved hep-only retrained model to %s", model_a_path)

    # ------------------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------------------
    run_benchmark_comparison()

    # Cleanup temp files
    for p in [new_model_path, model_a_path]:
        p.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
