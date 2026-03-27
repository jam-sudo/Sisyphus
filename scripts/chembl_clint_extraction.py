"""chembl_clint_extraction.py — Extract and analyze CLint data from ChEMBL SQLite.

Phase 0:  SQL extraction of CLint/intrinsic clearance activities
Phase 0b: Homogeneity assessment (interlaboratory CV, single-protocol subsets)
Phase 1:  Build deduplicated training set (ChEMBL + TDC, holdout-safe)
Phase 2:  Train and evaluate expanded CLint models (XGBoost + LightGBM)

Usage:
    python3 scripts/chembl_clint_extraction.py

Requires:
    - ChEMBL SQLite database at data/chembl/chembl_36_sqlite/chembl_36.db
    - TDC clearance datasets (downloaded automatically)
    - Existing holdout.json and clinical_pk.json

Output:
    - data/chembl/chembl_clint_raw.csv           (raw extraction)
    - data/chembl/chembl_clint_analysis.json      (Phase 0 + 0b report)
    - data/training/clint_expanded_v2.csv         (deduplicated training set)
    - models/adme/xgboost_clint_v2.json           (best model if improved)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHEMBL_DB = ROOT / "data" / "chembl" / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"
RAW_OUTPUT = ROOT / "data" / "chembl" / "chembl_clint_raw.csv"
ANALYSIS_OUTPUT = ROOT / "data" / "chembl" / "chembl_clint_analysis.json"
TRAINING_OUTPUT = ROOT / "data" / "training" / "clint_expanded_v2.csv"
HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = ROOT / "data" / "reference" / "clinical_pk.json"
TDC_HEP_TAB = ROOT / "data" / "training" / "clearance_hepatocyte_az.tab"
TDC_MIC_TAB = ROOT / "data" / "clearance_microsome_az.tab"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Microsome → hepatocyte conversion:
# CLint_hep (µL/min/10^6 cells) = CLint_mic (µL/min/mg) × MPPGL / HPGL
# MPPGL = 40 mg microsomal protein / g liver
# HPGL = 120 × 10^6 hepatocytes / g liver
MPPGL = 40.0  # mg microsomal protein per g liver (Barter et al., 2007)
HPGL = 120.0  # 10^6 hepatocytes per g liver (Wilson et al., 2003)
MIC_TO_HEP_FACTOR = MPPGL / HPGL  # = 0.333


# ---------------------------------------------------------------------------
# RDKit helpers
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
# Phase 0: ChEMBL SQL Extraction
# ---------------------------------------------------------------------------
def extract_clint_from_chembl(db_path: Path) -> pd.DataFrame:
    """Extract CLint data from ChEMBL SQLite database.

    Returns DataFrame with columns:
        compound_id, canonical_smiles, standard_value, standard_units,
        standard_type, standard_relation, assay_description, assay_type,
        src_id, doc_id
    """
    log.info("=" * 60)
    log.info("PHASE 0: ChEMBL CLint Extraction")
    log.info("=" * 60)

    if not db_path.exists():
        log.error("ChEMBL database not found at %s", db_path)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    # First, check available standard_types related to clearance
    log.info("Scanning for CLint-related standard_types ...")
    type_query = """
    SELECT DISTINCT act.standard_type, COUNT(*) as n
    FROM activities act
    JOIN assays ass ON act.assay_id = ass.assay_id
    WHERE (
        act.standard_type LIKE '%CLint%'
        OR act.standard_type LIKE '%clearance%'
        OR act.standard_type LIKE '%Clearance%'
        OR act.standard_type LIKE '%intrinsic%'
        OR act.standard_type LIKE '%Intrinsic%'
        OR act.standard_type = 'CLint'
    )
    AND ass.assay_organism = 'Homo sapiens'
    AND act.standard_value IS NOT NULL
    GROUP BY act.standard_type
    ORDER BY n DESC
    """
    type_df = pd.read_sql_query(type_query, conn)
    log.info("CLint-related standard_types found:")
    for _, row in type_df.iterrows():
        log.info("  %s: %d records", row["standard_type"], row["n"])

    # Main extraction query
    log.info("Running main CLint extraction query ...")
    main_query = """
    SELECT DISTINCT
        md.chembl_id AS compound_id,
        cs.canonical_smiles,
        act.standard_value,
        act.standard_units,
        act.standard_type,
        act.standard_relation,
        ass.description AS assay_description,
        ass.assay_type,
        ass.src_id,
        docs.chembl_id AS doc_id,
        src.src_description AS source_name
    FROM activities act
    JOIN assays ass ON act.assay_id = ass.assay_id
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    LEFT JOIN docs ON ass.doc_id = docs.doc_id
    LEFT JOIN source src ON ass.src_id = src.src_id
    WHERE (
        act.standard_type IN ('CLint', 'Intrinsic clearance')
        OR (act.standard_type LIKE '%CLint%' AND act.standard_type NOT LIKE '%unbound%')
    )
    AND ass.assay_organism = 'Homo sapiens'
    AND act.standard_relation = '='
    AND act.standard_value IS NOT NULL
    AND act.data_validity_comment IS NULL
    AND cs.canonical_smiles IS NOT NULL
    """
    df = pd.read_sql_query(main_query, conn)
    log.info("Raw extraction: %d records", len(df))

    # Also get unbound CLint for reference (won't mix with bound CLint)
    unbound_query = """
    SELECT COUNT(*) as n
    FROM activities act
    JOIN assays ass ON act.assay_id = ass.assay_id
    WHERE act.standard_type LIKE '%unbound%CLint%'
    AND ass.assay_organism = 'Homo sapiens'
    AND act.standard_value IS NOT NULL
    """
    unbound_n = pd.read_sql_query(unbound_query, conn).iloc[0]["n"]
    log.info("Unbound CLint records (excluded): %d", unbound_n)

    conn.close()

    # Save raw extraction
    df.to_csv(str(RAW_OUTPUT), index=False)
    log.info("Raw extraction saved to %s", RAW_OUTPUT)

    return df


def classify_assay_type(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each record as hepatocyte or microsome based on assay_description + units."""
    def _classify(row):
        desc = str(row["assay_description"]).lower()
        units = str(row["standard_units"]).lower()

        # Unit-based classification (most reliable)
        if "ul/min/10^6 cells" in units or "ul/min/million cells" in units:
            return "hepatocyte"
        if "ul/min/mg" in units or "ul/min/mg protein" in units:
            return "microsome"

        # Description-based classification
        if any(kw in desc for kw in ["hepatocyte", "hepatocytes", "hep "]):
            return "hepatocyte"
        if any(kw in desc for kw in [
            "microsome", "microsomes", "microsomal", "hlm", "rlm",
            "liver microsome",
        ]):
            return "microsome"

        # Check for specific patterns
        if "10^6 cells" in units or "cells" in units:
            return "hepatocyte"
        if "mg" in units and "protein" in desc:
            return "microsome"

        return "unknown"

    df = df.copy()
    df["assay_class"] = df.apply(_classify, axis=1)
    return df


def analyze_sources(df: pd.DataFrame) -> dict:
    """Analyze data source distribution."""
    log.info("-" * 40)
    log.info("Source distribution:")

    source_counts = df.groupby(["src_id", "source_name"]).agg(
        n_records=("compound_id", "count"),
        n_compounds=("compound_id", "nunique"),
    ).reset_index().sort_values("n_records", ascending=False)

    for _, row in source_counts.iterrows():
        log.info(
            "  src_id=%s (%s): %d records, %d compounds",
            row["src_id"], row["source_name"], row["n_records"], row["n_compounds"],
        )

    log.info("-" * 40)
    log.info("Assay classification:")
    class_counts = df.groupby("assay_class").agg(
        n_records=("compound_id", "count"),
        n_compounds=("compound_id", "nunique"),
    ).reset_index()
    for _, row in class_counts.iterrows():
        log.info(
            "  %s: %d records, %d compounds",
            row["assay_class"], row["n_records"], row["n_compounds"],
        )

    log.info("-" * 40)
    log.info("Unit distribution:")
    unit_counts = df["standard_units"].value_counts()
    for unit, count in unit_counts.items():
        log.info("  %s: %d", unit, count)

    return {
        "source_distribution": source_counts.to_dict("records"),
        "assay_classification": class_counts.to_dict("records"),
        "unit_distribution": unit_counts.to_dict(),
    }


# ---------------------------------------------------------------------------
# Phase 0b: Homogeneity Assessment
# ---------------------------------------------------------------------------
def assess_homogeneity(df: pd.DataFrame) -> dict:
    """Compute interlaboratory CV and identify homogeneous subsets."""
    log.info("=" * 60)
    log.info("PHASE 0b: Homogeneity Assessment")
    log.info("=" * 60)

    # Canonicalize SMILES
    log.info("Canonicalizing SMILES ...")
    df = df.copy()
    df["canon_smiles"] = df["canonical_smiles"].apply(_canonical_smiles)
    df = df.dropna(subset=["canon_smiles"]).reset_index(drop=True)

    # For compounds measured by multiple sources: compute CV
    log.info("Computing interlaboratory variability ...")

    # Group by compound + assay class, compute stats per source
    multi_source = (
        df.groupby(["canon_smiles", "assay_class"])
        .agg(
            n_sources=("src_id", "nunique"),
            n_measurements=("standard_value", "count"),
            values=("standard_value", list),
            sources=("src_id", list),
        )
        .reset_index()
    )

    multi_source_compounds = multi_source[multi_source["n_sources"] > 1]
    log.info(
        "Compounds measured by multiple sources: %d (of %d total compound-class pairs)",
        len(multi_source_compounds),
        len(multi_source),
    )

    # Compute CV for multi-source compounds
    cvs = []
    for _, row in multi_source_compounds.iterrows():
        vals = np.array(row["values"], dtype=float)
        if len(vals) >= 2 and np.mean(vals) > 0:
            cv = np.std(vals) / np.mean(vals) * 100  # percent
            cvs.append(cv)

    if cvs:
        cvs = np.array(cvs)
        log.info("Interlaboratory CV distribution:")
        log.info("  Median: %.1f%%", np.median(cvs))
        log.info("  75th percentile: %.1f%%", np.percentile(cvs, 75))
        log.info("  90th percentile: %.1f%%", np.percentile(cvs, 90))
        log.info("  CV > 100%%: %d / %d (%.1f%%)",
                 np.sum(cvs > 100), len(cvs), np.sum(cvs > 100) / len(cvs) * 100)
    else:
        log.warning("No multi-source compounds found for CV calculation")

    # Identify single-protocol datasets (same doc_id)
    log.info("-" * 40)
    log.info("Single-protocol dataset identification:")
    doc_compounds = (
        df.groupby("doc_id")
        .agg(
            n_compounds=("canon_smiles", "nunique"),
            src_id=("src_id", "first"),
            source_name=("source_name", "first"),
        )
        .reset_index()
        .sort_values("n_compounds", ascending=False)
    )

    top_docs = doc_compounds.head(15)
    for _, row in top_docs.iterrows():
        log.info(
            "  doc=%s: %d compounds (src=%s, %s)",
            row["doc_id"], row["n_compounds"], row["src_id"], row["source_name"],
        )

    # Identify AZ deposit (src_id = 26 is typically AstraZeneca in ChEMBL)
    # Find AZ-like sources
    az_src_ids = set()
    for src_id, name in df[["src_id", "source_name"]].drop_duplicates().values:
        if name and "astra" in str(name).lower():
            az_src_ids.add(int(src_id))
            log.info("AstraZeneca source identified: src_id=%d (%s)", src_id, name)

    if not az_src_ids:
        log.info("No AstraZeneca-specific source found; checking src_id=26 ...")
        if 26 in df["src_id"].values:
            az_src_ids.add(26)

    az_compounds = set()
    non_az_compounds = set()
    for _, row in df.iterrows():
        if int(row["src_id"]) in az_src_ids:
            az_compounds.add(row["canon_smiles"])
        else:
            non_az_compounds.add(row["canon_smiles"])

    # Net new = non-AZ compounds that are NOT also in AZ
    net_new_non_az = non_az_compounds - az_compounds
    log.info("-" * 40)
    log.info("AZ vs non-AZ compound breakdown:")
    log.info("  AZ compounds: %d", len(az_compounds))
    log.info("  Non-AZ compounds: %d", len(non_az_compounds))
    log.info("  Overlap (in both AZ and non-AZ): %d", len(az_compounds & non_az_compounds))
    log.info("  Net new non-AZ (unique to non-AZ): %d", len(net_new_non_az))

    homogeneity_report = {
        "n_multi_source_compounds": int(len(multi_source_compounds)),
        "cv_median": float(np.median(cvs)) if cvs else None,
        "cv_75th": float(np.percentile(cvs, 75)) if cvs else None,
        "cv_90th": float(np.percentile(cvs, 90)) if cvs else None,
        "cv_gt_100pct_frac": float(np.sum(cvs > 100) / len(cvs)) if cvs else None,
        "az_src_ids": list(az_src_ids),
        "az_compounds": len(az_compounds),
        "non_az_compounds": len(non_az_compounds),
        "net_new_non_az": len(net_new_non_az),
        "top_docs": top_docs.to_dict("records"),
    }

    return homogeneity_report


# ---------------------------------------------------------------------------
# TDC data loading
# ---------------------------------------------------------------------------
def load_tdc_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TDC Hepatocyte_AZ and Microsome_AZ from local .tab files."""
    log.info("Loading TDC datasets from local files ...")

    hep_df = pd.read_csv(str(TDC_HEP_TAB), sep="\t")
    hep_df.columns = ["Drug_ID", "Drug", "Y"]
    log.info("  TDC Hepatocyte_AZ: %d records", len(hep_df))

    mic_df = pd.read_csv(str(TDC_MIC_TAB), sep="\t")
    mic_df.columns = ["Drug_ID", "Drug", "Y"]
    log.info("  TDC Microsome_AZ: %d records", len(mic_df))

    # Canonicalize
    hep_df["canon_smiles"] = hep_df["Drug"].apply(_canonical_smiles)
    mic_df["canon_smiles"] = mic_df["Drug"].apply(_canonical_smiles)
    hep_df = hep_df.dropna(subset=["canon_smiles"]).reset_index(drop=True)
    mic_df = mic_df.dropna(subset=["canon_smiles"]).reset_index(drop=True)

    return hep_df, mic_df


# ---------------------------------------------------------------------------
# Holdout key construction (from train_clint_expanded.py)
# ---------------------------------------------------------------------------
def build_holdout_keys() -> dict:
    """Build 3-key exclusion sets from holdout data."""
    with open(HOLDOUT_JSON) as f:
        holdout_data = json.load(f)
    with open(CLINICAL_PK_JSON) as f:
        clinical_pk = json.load(f)

    holdout_names = holdout_data.get("holdout", [])
    train_names = holdout_data.get("train", [])
    all_names = holdout_names + train_names

    canonical_smiles: set[str] = set()
    inchikey_prefixes: set[str] = set()
    names: set[str] = set()

    drugs = clinical_pk.get("drugs", {})
    for name in all_names:
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
        "Holdout keys: %d SMILES, %d InChIKey, %d names",
        len(canonical_smiles), len(inchikey_prefixes), len(names),
    )
    return {
        "canonical_smiles": canonical_smiles,
        "inchikey_prefixes": inchikey_prefixes,
        "names": names,
    }


def is_holdout(smiles: str, holdout_keys: dict) -> bool:
    """Return True if compound matches any holdout key."""
    csmi = _canonical_smiles(smiles)
    if csmi and csmi in holdout_keys["canonical_smiles"]:
        return True
    ik = _inchikey_prefix(smiles)
    if ik and ik in holdout_keys["inchikey_prefixes"]:
        return True
    return False


# ---------------------------------------------------------------------------
# Phase 1: Build deduplicated training set
# ---------------------------------------------------------------------------
def build_training_set(
    chembl_df: pd.DataFrame,
    tdc_hep: pd.DataFrame,
    tdc_mic: pd.DataFrame,
    holdout_keys: dict,
) -> pd.DataFrame:
    """Merge ChEMBL + TDC, deduplicate, exclude holdout, harmonize units."""
    log.info("=" * 60)
    log.info("PHASE 1: Build Deduplicated Training Set")
    log.info("=" * 60)

    # --- Step 1: Process ChEMBL data ---
    chembl = chembl_df.copy()
    chembl["canon_smiles"] = chembl["canonical_smiles"].apply(_canonical_smiles)
    chembl = chembl.dropna(subset=["canon_smiles"]).reset_index(drop=True)

    # Convert microsome to hepatocyte units
    # hepatocyte: µL/min/10^6 cells (target unit)
    # microsome: µL/min/mg → multiply by MIC_TO_HEP_FACTOR (0.333)
    def _harmonize_value(row):
        val = float(row["standard_value"])
        if row["assay_class"] == "microsome":
            return val * MIC_TO_HEP_FACTOR
        return val

    chembl["clint_hep"] = chembl.apply(_harmonize_value, axis=1)

    # Remove invalid/extreme values
    chembl = chembl[chembl["clint_hep"] > 0].reset_index(drop=True)
    log10_clint = np.log10(chembl["clint_hep"])
    extreme_mask = (log10_clint < -2) | (log10_clint > 4)
    n_extreme = extreme_mask.sum()
    if n_extreme > 0:
        log.info("Removing %d extreme CLint values (log10 < -2 or > 4)", n_extreme)
        chembl = chembl[~extreme_mask].reset_index(drop=True)

    # Aggregate by compound: geometric mean of all measurements
    chembl_agg = (
        chembl.groupby("canon_smiles")
        .agg(
            clint_hep=("clint_hep", lambda x: 10 ** np.mean(np.log10(x))),  # geomean
            n_measurements=("clint_hep", "count"),
            n_sources=("src_id", "nunique"),
            sources=("src_id", lambda x: list(set(x))),
        )
        .reset_index()
    )
    log.info("ChEMBL after aggregation: %d unique compounds", len(chembl_agg))

    # --- Step 2: Process TDC data ---
    # TDC Hepatocyte_AZ: already in µL/min/10^6 cells
    tdc_hep_agg = (
        tdc_hep.groupby("canon_smiles")
        .agg(clint_hep=("Y", "mean"))
        .reset_index()
    )
    tdc_hep_agg["source"] = "tdc_hepatocyte_az"

    # TDC Microsome_AZ: µL/min/mg → convert
    tdc_mic_agg = (
        tdc_mic.groupby("canon_smiles")
        .agg(clint_hep=("Y", lambda x: np.mean(x) * MIC_TO_HEP_FACTOR))
        .reset_index()
    )
    tdc_mic_agg["source"] = "tdc_microsome_az"

    tdc_smiles = set(tdc_hep_agg["canon_smiles"]) | set(tdc_mic_agg["canon_smiles"])
    log.info("TDC unique compounds: %d (Hep: %d, Mic: %d)",
             len(tdc_smiles), len(tdc_hep_agg), len(tdc_mic_agg))

    # --- Step 3: Compute overlap with TDC ---
    chembl_smiles = set(chembl_agg["canon_smiles"])
    overlap = chembl_smiles & tdc_smiles
    chembl_only = chembl_smiles - tdc_smiles
    tdc_only = tdc_smiles - chembl_smiles
    log.info("Overlap ChEMBL ∩ TDC: %d", len(overlap))
    log.info("ChEMBL-only (net new): %d", len(chembl_only))
    log.info("TDC-only: %d", len(tdc_only))

    # --- Step 4: Build merged dataset ---
    # Priority: TDC Hepatocyte > TDC Microsome > ChEMBL
    # (TDC is AZ single-lab, more homogeneous)

    # Start with TDC Hepatocyte
    merged_records = {}
    for _, row in tdc_hep_agg.iterrows():
        smi = row["canon_smiles"]
        merged_records[smi] = {
            "canon_smiles": smi,
            "clint_hep": float(row["clint_hep"]),
            "source": "tdc_hepatocyte_az",
        }

    # Add TDC Microsome (only if not already in Hepatocyte)
    for _, row in tdc_mic_agg.iterrows():
        smi = row["canon_smiles"]
        if smi not in merged_records:
            merged_records[smi] = {
                "canon_smiles": smi,
                "clint_hep": float(row["clint_hep"]),
                "source": "tdc_microsome_az",
            }

    # Add ChEMBL-only compounds
    n_chembl_added = 0
    for _, row in chembl_agg.iterrows():
        smi = row["canon_smiles"]
        if smi not in merged_records:
            merged_records[smi] = {
                "canon_smiles": smi,
                "clint_hep": float(row["clint_hep"]),
                "source": "chembl",
            }
            n_chembl_added += 1

    log.info("Merged before holdout exclusion: %d compounds (%d from ChEMBL)",
             len(merged_records), n_chembl_added)

    # --- Step 5: Holdout exclusion ---
    merged_df = pd.DataFrame(list(merged_records.values()))
    n_before = len(merged_df)
    holdout_mask = merged_df["canon_smiles"].apply(
        lambda s: is_holdout(s, holdout_keys)
    )
    merged_df = merged_df[~holdout_mask].reset_index(drop=True)
    n_excluded = n_before - len(merged_df)
    log.info("Holdout exclusion: %d excluded, %d remain", n_excluded, len(merged_df))

    # Remove CLint = 0 or negative (can't log-transform)
    merged_df = merged_df[merged_df["clint_hep"] > 0].reset_index(drop=True)

    # Floor at 0.1 (consistent with existing pipeline)
    merged_df["clint_hep"] = merged_df["clint_hep"].clip(lower=0.1)

    # --- Report ---
    source_dist = merged_df["source"].value_counts()
    log.info("Final training set: %d compounds", len(merged_df))
    for src, cnt in source_dist.items():
        log.info("  %s: %d", src, cnt)

    # Save
    merged_df.to_csv(str(TRAINING_OUTPUT), index=False)
    log.info("Training set saved to %s", TRAINING_OUTPUT)

    return merged_df


# ---------------------------------------------------------------------------
# Phase 2: Model Training
# ---------------------------------------------------------------------------
def train_models(
    training_df: pd.DataFrame,
    tdc_hep_baseline: pd.DataFrame,
    holdout_keys: dict,
) -> dict:
    """Train XGBoost + LightGBM with scaffold CV, compare with TDC-only baseline."""
    log.info("=" * 60)
    log.info("PHASE 2: Model Training & Evaluation")
    log.info("=" * 60)

    import xgboost as xgb
    from sisyphus.descriptors import compute_features
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    # --- Feature computation for expanded set ---
    log.info("Computing features for expanded set (%d compounds) ...", len(training_df))
    X_list, y_list, smiles_list, sources_list = [], [], [], []
    failed = 0
    for _, row in training_df.iterrows():
        try:
            feat = compute_features(row["canon_smiles"])
            X_list.append(feat)
            y_list.append(np.log10(max(float(row["clint_hep"]), 0.1)))
            smiles_list.append(row["canon_smiles"])
            sources_list.append(row["source"])
        except Exception:
            failed += 1

    X_exp = np.array(X_list, dtype=np.float64)
    y_exp = np.array(y_list, dtype=np.float64)
    log.info("Expanded features: %s (failed: %d)", X_exp.shape, failed)

    # --- Feature computation for TDC-only baseline ---
    # Exclude holdout from TDC
    tdc_base = tdc_hep_baseline.copy()
    tdc_base_mask = tdc_base["canon_smiles"].apply(
        lambda s: is_holdout(s, holdout_keys)
    )
    tdc_base = tdc_base[~tdc_base_mask].reset_index(drop=True)
    # Deduplicate
    tdc_base = (
        tdc_base.groupby("canon_smiles")
        .agg(Y=("Y", "mean"))
        .reset_index()
    )

    log.info("Computing features for TDC baseline (%d compounds) ...", len(tdc_base))
    X_base_list, y_base_list, smi_base_list = [], [], []
    for _, row in tdc_base.iterrows():
        try:
            feat = compute_features(row["canon_smiles"])
            X_base_list.append(feat)
            y_base_list.append(np.log10(max(float(row["Y"]), 0.1)))
            smi_base_list.append(row["canon_smiles"])
        except Exception:
            pass

    X_base = np.array(X_base_list, dtype=np.float64)
    y_base = np.array(y_base_list, dtype=np.float64)
    log.info("Baseline features: %s", X_base.shape)

    # --- Scaffold split helper ---
    def _scaffold_split(smiles: list[str], n_folds: int = 5, seed: int = 42):
        scaffold_to_idx: dict[str, list[int]] = {}
        for i, smi in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                scaf = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
            except Exception:
                scaf = ""
            scaffold_to_idx.setdefault(scaf, []).append(i)

        rng = np.random.default_rng(seed)
        scaffolds = list(scaffold_to_idx.keys())
        rng.shuffle(scaffolds)
        folds = [[] for _ in range(n_folds)]
        for i, scaf in enumerate(scaffolds):
            folds[i % n_folds].extend(scaffold_to_idx[scaf])
        return folds

    # --- CV evaluation helper ---
    def _cv_evaluate(X, y, smiles, label, model_cls, model_params, sample_weight=None):
        folds = _scaffold_split(smiles)
        all_preds = np.zeros(len(y))
        fold_r2s, fold_maes = [], []

        for fold_idx in range(5):
            test_idx = folds[fold_idx]
            train_idx = [j for f in range(5) if f != fold_idx for j in folds[f]]
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model = model_cls(**model_params)
            fit_kwargs = {"verbose": False} if hasattr(model, "verbose") else {}
            if sample_weight is not None:
                sw_tr = sample_weight[train_idx]
                model.fit(X_tr, y_tr, sample_weight=sw_tr, **fit_kwargs)
            else:
                model.fit(X_tr, y_tr, **fit_kwargs)

            preds = model.predict(X_te)
            all_preds[test_idx] = preds

            ss_res = np.sum((y_te - preds) ** 2)
            ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            mae = float(np.mean(np.abs(y_te - preds)))
            fold_r2s.append(r2)
            fold_maes.append(mae)

        # Overall
        ss_res = np.sum((y - all_preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        overall_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        overall_mae = float(np.mean(np.abs(y - all_preds)))

        # In-sample R² (overfitting check)
        model_full = model_cls(**model_params)
        if sample_weight is not None:
            model_full.fit(X, y, sample_weight=sample_weight,
                           **({} if not hasattr(model_full, "verbose") else {"verbose": False}))
        else:
            model_full.fit(X, y, **({} if not hasattr(model_full, "verbose") else {"verbose": False}))
        full_preds = model_full.predict(X)
        ss_res_full = np.sum((y - full_preds) ** 2)
        ss_tot_full = np.sum((y - np.mean(y)) ** 2)
        insample_r2 = float(1 - ss_res_full / ss_tot_full)

        result = {
            "label": label,
            "n": len(y),
            "cv_r2_overall": overall_r2,
            "cv_mae_overall": overall_mae,
            "cv_r2_mean": float(np.mean(fold_r2s)),
            "cv_r2_std": float(np.std(fold_r2s)),
            "cv_mae_mean": float(np.mean(fold_maes)),
            "cv_mae_std": float(np.std(fold_maes)),
            "insample_r2": insample_r2,
            "overfitting_flag": abs(insample_r2 - overall_r2) > 0.15,
        }

        log.info(
            "%s: N=%d, CV R²=%.3f (%.3f±%.3f), MAE=%.3f, InSample R²=%.3f %s",
            label, len(y), overall_r2, np.mean(fold_r2s), np.std(fold_r2s),
            overall_mae, insample_r2,
            "⚠ OVERFIT" if result["overfitting_flag"] else "",
        )

        return result, model_full

    # --- Random split CV helper (for comparison) ---
    def _random_cv_r2(X, y, model_cls, model_params, n_folds=5, seed=42):
        rng = np.random.default_rng(seed)
        indices = np.arange(len(y))
        rng.shuffle(indices)
        fold_size = len(y) // n_folds
        all_preds = np.zeros(len(y))
        for f in range(n_folds):
            start = f * fold_size
            end = start + fold_size if f < n_folds - 1 else len(y)
            test_idx = indices[start:end]
            train_idx = np.setdiff1d(indices, test_idx)
            model = model_cls(**model_params)
            model.fit(X[train_idx], y[train_idx],
                      **({} if not hasattr(model, "verbose") else {"verbose": False}))
            all_preds[test_idx] = model.predict(X[test_idx])
        ss_res = np.sum((y - all_preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)

    # --- XGBoost params ---
    xgb_params = dict(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4, verbosity=0,
    )

    results = {}

    # === 1. TDC-only Baseline (XGBoost) ===
    log.info("-" * 60)
    log.info("1. TDC-only Baseline (XGBoost, scaffold split)")
    r_base, _ = _cv_evaluate(X_base, y_base, smi_base_list,
                             "XGB_TDC_baseline", xgb.XGBRegressor, xgb_params)
    r_base["random_cv_r2"] = _random_cv_r2(X_base, y_base, xgb.XGBRegressor, xgb_params)
    log.info("  Random split R²: %.3f", r_base["random_cv_r2"])
    results["baseline_xgb"] = r_base

    # === 2. Expanded XGBoost (uniform weight) ===
    log.info("-" * 60)
    log.info("2. Expanded XGBoost (uniform weight, scaffold split)")
    r_exp_uniform, model_exp_uniform = _cv_evaluate(
        X_exp, y_exp, smiles_list,
        "XGB_expanded_uniform", xgb.XGBRegressor, xgb_params,
    )
    r_exp_uniform["random_cv_r2"] = _random_cv_r2(
        X_exp, y_exp, xgb.XGBRegressor, xgb_params
    )
    log.info("  Random split R²: %.3f", r_exp_uniform["random_cv_r2"])
    results["expanded_xgb_uniform"] = r_exp_uniform

    # === 3. Expanded XGBoost with source-based weighting (grid search) ===
    log.info("-" * 60)
    log.info("3. Expanded XGBoost with source weight optimization")

    # Build source array aligned with features
    source_arr = np.array(sources_list)
    is_tdc = np.array([s.startswith("tdc") for s in sources_list])

    best_w = None
    best_r2 = -999
    for w_non_tdc in np.arange(0.1, 1.05, 0.1):
        weights = np.where(is_tdc, 1.0, w_non_tdc)
        folds = _scaffold_split(smiles_list)
        all_preds = np.zeros(len(y_exp))
        for fold_idx in range(5):
            test_idx = folds[fold_idx]
            train_idx = [j for f in range(5) if f != fold_idx for j in folds[f]]
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_exp[train_idx], y_exp[train_idx],
                      sample_weight=weights[train_idx], verbose=False)
            all_preds[test_idx] = model.predict(X_exp[test_idx])
        ss_res = np.sum((y_exp - all_preds) ** 2)
        ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
        r2 = float(1 - ss_res / ss_tot)
        if r2 > best_r2:
            best_r2 = r2
            best_w = w_non_tdc

    log.info("  Best non-TDC weight: %.1f → R²=%.3f", best_w, best_r2)

    # Re-run with best weight to get full metrics
    best_weights = np.where(is_tdc, 1.0, best_w)
    r_exp_weighted, model_exp_weighted = _cv_evaluate(
        X_exp, y_exp, smiles_list,
        f"XGB_expanded_w={best_w:.1f}", xgb.XGBRegressor, xgb_params,
        sample_weight=best_weights,
    )
    results["expanded_xgb_weighted"] = r_exp_weighted
    results["expanded_xgb_weighted"]["best_non_tdc_weight"] = float(best_w)

    # === 4. LightGBM comparison ===
    try:
        import lightgbm as lgb

        lgb_params = dict(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=4, verbose=-1,
        )

        log.info("-" * 60)
        log.info("4. LightGBM expanded (uniform weight, scaffold split)")
        r_lgb, _ = _cv_evaluate(
            X_exp, y_exp, smiles_list,
            "LGB_expanded_uniform", lgb.LGBMRegressor, lgb_params,
        )
        results["expanded_lgb_uniform"] = r_lgb
    except ImportError:
        log.warning("LightGBM not installed, skipping")

    # === Summary ===
    log.info("=" * 60)
    log.info("MODEL COMPARISON SUMMARY")
    log.info("=" * 60)
    print()
    print(f"{'Model':<35s} {'N':>6s} {'Scaffold R²':>12s} {'Random R²':>10s} {'MAE':>6s}")
    print("-" * 75)
    for key, r in results.items():
        rand_r2 = r.get("random_cv_r2", "")
        rand_str = f"{rand_r2:.3f}" if isinstance(rand_r2, float) else "N/A"
        print(f"{r['label']:<35s} {r['n']:>6d} {r['cv_r2_overall']:>12.3f} "
              f"{rand_str:>10s} {r['cv_mae_overall']:>6.3f}")

    baseline_r2 = results["baseline_xgb"]["cv_r2_overall"]
    best_expanded_key = max(
        [k for k in results if k != "baseline_xgb"],
        key=lambda k: results[k]["cv_r2_overall"],
    )
    best_expanded_r2 = results[best_expanded_key]["cv_r2_overall"]
    delta_r2 = best_expanded_r2 - baseline_r2

    print()
    print(f"Baseline R² (scaffold):    {baseline_r2:.3f}")
    print(f"Best expanded R²:          {best_expanded_r2:.3f} ({best_expanded_key})")
    print(f"ΔR²:                       {delta_r2:+.3f}")
    print()

    if delta_r2 < 0.03:
        print("⚠ GATE FAILED: ΔR² < 0.03 — expansion adds noise, not signal.")
        print("  Consider: homogeneous subset only, or accept public data ceiling.")
    else:
        print("✓ GATE PASSED: ΔR² >= 0.03 — expansion provides signal.")

    # Save best expanded model
    best_model_path = ROOT / "models" / "adme" / "xgboost_clint_v2.json"
    if best_expanded_key.startswith("XGB"):
        # Retrain the corresponding model on full data
        if "weighted" in best_expanded_key:
            model_exp_weighted.save_model(str(best_model_path))
        else:
            model_exp_uniform.save_model(str(best_model_path))
        log.info("Best model saved to %s", best_model_path)

    return {
        "results": results,
        "baseline_r2": baseline_r2,
        "best_expanded_r2": best_expanded_r2,
        "delta_r2": delta_r2,
        "gate_passed": delta_r2 >= 0.03,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    log.info("=" * 60)
    log.info("ChEMBL CLint Extraction & Expansion Pipeline")
    log.info("=" * 60)

    # === Phase 0: Extract ===
    chembl_raw = extract_clint_from_chembl(CHEMBL_DB)
    chembl_raw = classify_assay_type(chembl_raw)
    source_report = analyze_sources(chembl_raw)

    # === Phase 0 reporting ===
    # Canonicalize for overlap analysis
    chembl_raw["canon_smiles"] = chembl_raw["canonical_smiles"].apply(_canonical_smiles)
    chembl_raw = chembl_raw.dropna(subset=["canon_smiles"]).reset_index(drop=True)

    total_records = len(chembl_raw)
    total_unique = chembl_raw["canon_smiles"].nunique()
    hep_only = chembl_raw[chembl_raw["assay_class"] == "hepatocyte"]
    mic_only = chembl_raw[chembl_raw["assay_class"] == "microsome"]

    log.info("PHASE 0 SUMMARY:")
    log.info("  Total records: %d", total_records)
    log.info("  Unique compounds (canonical SMILES): %d", total_unique)
    log.info("  Hepatocyte: %d records, %d compounds",
             len(hep_only), hep_only["canon_smiles"].nunique())
    log.info("  Microsome: %d records, %d compounds",
             len(mic_only), mic_only["canon_smiles"].nunique())

    # TDC overlap
    tdc_hep, tdc_mic = load_tdc_data()
    tdc_smiles = set(tdc_hep["canon_smiles"]) | set(tdc_mic["canon_smiles"])
    chembl_smiles = set(chembl_raw["canon_smiles"])
    overlap_tdc = chembl_smiles & tdc_smiles
    net_new = chembl_smiles - tdc_smiles
    log.info("  TDC overlap: %d compounds", len(overlap_tdc))
    log.info("  Net new (ChEMBL only): %d compounds", len(net_new))

    # === Phase 0 Gate ===
    if len(net_new) < 500:
        log.warning(
            "GATE CHECK: net new compounds (%d) < 500. "
            "Proceeding, but expansion may be insufficient.",
            len(net_new),
        )

    # === Phase 0b: Homogeneity ===
    homogeneity = assess_homogeneity(chembl_raw)

    # Save analysis
    analysis = {
        "phase0": {
            "total_records": total_records,
            "total_unique_compounds": total_unique,
            "hepatocyte_records": len(hep_only),
            "hepatocyte_compounds": int(hep_only["canon_smiles"].nunique()),
            "microsome_records": len(mic_only),
            "microsome_compounds": int(mic_only["canon_smiles"].nunique()),
            "tdc_overlap": len(overlap_tdc),
            "net_new_from_chembl": len(net_new),
            "sources": source_report,
        },
        "phase0b": homogeneity,
    }
    with open(ANALYSIS_OUTPUT, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    log.info("Analysis saved to %s", ANALYSIS_OUTPUT)

    # === Phase 1: Build training set ===
    holdout_keys = build_holdout_keys()
    training_df = build_training_set(chembl_raw, tdc_hep, tdc_mic, holdout_keys)

    # === Phase 2: Train models ===
    model_results = train_models(training_df, tdc_hep, holdout_keys)

    # Append to analysis
    analysis["phase2"] = model_results
    with open(ANALYSIS_OUTPUT, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
