#!/usr/bin/env python3
"""Rebuild VDss from the pinned public TDC Lombardo dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_clint_expanded import scaffold_split_indices  # noqa: E402
from scripts.train_peff import build_holdout_keys, is_holdout  # noqa: E402
from sisyphus.descriptors import compute_features  # noqa: E402

SOURCE = ROOT / "data/vdss_lombardo.tab"
SOURCE_SHA = "00bb7e0dea19f78c4c1887e27ecf476135d9de2ef5ecc26d7c645c37bd9cb7af"
SOURCE_URL = "https://dataverse.harvard.edu/api/access/datafile/4267387"
DATASET = ROOT / "data/training/vdss_tdc_public_clean.csv"
MODEL = ROOT / "models/adme/xgboost_vdss.json"
META = MODEL.with_suffix(".meta.json")
PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
              colsample_bytree=0.8, random_state=42, n_jobs=4, verbosity=0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_rows() -> pd.DataFrame:
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("VDss_Lombardo source SHA256 changed")
    raw = pd.read_csv(SOURCE, sep="\t").rename(columns={"ID": "drug_id", "X": "smiles"})
    raw["canonical_smiles"] = raw["smiles"].apply(
        lambda smiles: Chem.MolToSmiles(mol, isomericSmiles=True)
        if (mol := Chem.MolFromSmiles(smiles)) is not None else None
    )
    raw = raw.dropna(subset=["canonical_smiles"])
    holdout = json.loads((ROOT / "data/reference/holdout.json").read_text())["holdout"]
    clinical = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())
    keys = build_holdout_keys(holdout, clinical)
    matched = raw.apply(
        lambda row: is_holdout(row["canonical_smiles"], row["drug_id"], keys), axis=1
    )
    excluded_structures = set(raw.loc[matched, "canonical_smiles"])
    clean = raw[~raw["canonical_smiles"].isin(excluded_structures)]
    fitted = clean.groupby("canonical_smiles").agg({"Y": "mean", "drug_id": "first"}).reset_index()
    if (len(raw), raw["canonical_smiles"].nunique(), len(excluded_structures),
            len(fitted)) != (1130, 1111, 56, 1055):
        raise ValueError("Unexpected VDss source, duplicate, or holdout counts")
    if not np.isfinite(fitted["Y"]).all() or (fitted["Y"] <= 0).any():
        raise ValueError("VDss training targets must be positive and finite")
    return fitted[["canonical_smiles", "drug_id", "Y"]]


def main() -> None:
    rows = training_rows()
    rows.to_csv(DATASET, index=False, float_format="%.17g")
    fitted = pd.read_csv(DATASET)
    X = np.asarray([compute_features(s) for s in fitted["canonical_smiles"]])
    y = np.log10(np.maximum(fitted["Y"].to_numpy(dtype=float), 0.01))

    oof = np.empty_like(y)
    folds = scaffold_split_indices(fitted["canonical_smiles"].tolist())
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        model = xgb.XGBRegressor(**PARAMS).fit(X[train], y[train])
        oof[test] = model.predict(X[test])
    cv_r2 = float(1 - np.sum((oof - y) ** 2) / np.sum((y - y.mean()) ** 2))
    cv_spearman = float(spearmanr(y, oof).statistic)
    cv_aafe = float(10 ** np.mean(np.abs(oof - y)))

    model = xgb.XGBRegressor(**PARAMS).fit(X, y)
    model.save_model(MODEL)
    metadata = json.loads(META.read_text())
    metadata.update(
        version="v1_public_lombardo",
        artifact_sha256=sha256(MODEL),
        trained_at=datetime.now(timezone.utc).isoformat(),
        trained_on={"dataset_path": str(DATASET.relative_to(ROOT)),
                    "sha256": sha256(DATASET), "n_drugs_clean": len(fitted),
                    "source_sha256": SOURCE_SHA, "source_url": SOURCE_URL},
        n_drugs_original=1130,
        n_drugs_excluded=1130 - len(fitted),
        n_duplicate_rows=19,
        n_holdout_structures_excluded=56,
        holdout_version="N=107 (data/reference/holdout.json)",
        holdout_metric={"name": "five_fold_scaffold_cv_spearman", "value": cv_spearman,
                        "r2": cv_r2, "aafe": cv_aafe},
        hyperparameters=PARAMS,
        retrained_reason="Pin public TDC Lombardo fitted rows and replace unknown legacy source",
    )
    META.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"N={len(fitted)} scaffold_CV_Spearman={cv_spearman:.3f} "
        f"R2={cv_r2:.3f} AAFE={cv_aafe:.3f}"
    )
    print(f"dataset_sha256={sha256(DATASET)} model_sha256={sha256(MODEL)}")


if __name__ == "__main__":
    main()
