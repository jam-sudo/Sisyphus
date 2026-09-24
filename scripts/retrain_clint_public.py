#!/usr/bin/env python3
"""Rebuild hepatocyte-only CLint from the pinned public TDC snapshot."""

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_clint_expanded import scaffold_split_indices  # noqa: E402
from scripts.train_peff import build_holdout_keys, is_holdout  # noqa: E402
from sisyphus.descriptors import compute_features  # noqa: E402

SOURCE = ROOT / "data/clearance_hepatocyte_az.tab"
SOURCE_SHA = "2c217f46600c22e107b72faa2bfb1d9bf799213aee47b601f6e0d6d73b68f274"
DATASET = ROOT / "data/training/clint_tdc_public_clean.csv"
MODEL = ROOT / "models/adme/xgboost_clint.json"
META = MODEL.with_suffix(".meta.json")
PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
              colsample_bytree=0.8, random_state=42, n_jobs=4, verbosity=0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_rows() -> pd.DataFrame:
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("Hepatocyte_AZ source SHA256 changed")
    raw = pd.read_csv(SOURCE, sep="\t").rename(columns={"ID": "drug_id", "X": "smiles"})
    raw["canonical_smiles"] = raw["smiles"].apply(
        lambda smiles: Chem.MolToSmiles(mol, isomericSmiles=True)
        if (mol := Chem.MolFromSmiles(smiles)) is not None else None
    )
    raw = raw.dropna(subset=["canonical_smiles"])
    dedup = raw.groupby("canonical_smiles").agg({"Y": "mean", "drug_id": "first"}).reset_index()
    holdout = json.loads((ROOT / "data/reference/holdout.json").read_text())["holdout"]
    clinical = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())
    keys = build_holdout_keys(holdout, clinical)
    clean = dedup[~dedup.apply(
        lambda row: is_holdout(row["canonical_smiles"], row["drug_id"], keys), axis=1
    )].reset_index(drop=True)
    if (len(raw), len(dedup), len(clean)) != (1213, 1020, 996):
        raise ValueError(
            f"Unexpected hepatocyte raw/unique/clean counts: {len(raw)}/{len(dedup)}/{len(clean)}"
        )
    return clean[["canonical_smiles", "drug_id", "Y"]]


def main() -> None:
    rows = training_rows()
    rows.to_csv(DATASET, index=False, float_format="%.17g")
    fitted = pd.read_csv(DATASET)
    X = np.asarray([compute_features(s) for s in fitted["canonical_smiles"]])
    y = np.log10(np.maximum(fitted["Y"].to_numpy(dtype=float), 0.1))

    oof = np.empty_like(y)
    folds = scaffold_split_indices(fitted["canonical_smiles"].tolist())
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        model = xgb.XGBRegressor(**PARAMS).fit(X[train], y[train])
        oof[test] = model.predict(X[test])
    cv_r2 = float(1 - np.sum((oof - y) ** 2) / np.sum((y - y.mean()) ** 2))
    cv_aafe = float(10 ** np.mean(np.abs(oof - y)))

    model = xgb.XGBRegressor(**PARAMS).fit(X, y)
    model.save_model(MODEL)
    metadata = json.loads(META.read_text())
    metadata.update(
        version="v1_public_hepatocyte",
        artifact_sha256=sha256(MODEL),
        trained_at=datetime.now(timezone.utc).isoformat(),
        trained_on={"dataset_path": str(DATASET.relative_to(ROOT)),
                    "sha256": sha256(DATASET), "n_drugs_clean": len(fitted),
                    "source_sha256": SOURCE_SHA},
        n_drugs_original=1213,
        n_drugs_excluded=1213 - len(fitted),
        n_duplicate_rows=193,
        n_holdout_structures_excluded=24,
        holdout_version="N=107 (data/reference/holdout.json)",
        holdout_metric={"name": "five_fold_scaffold_cv_r2", "value": cv_r2,
                        "aafe": cv_aafe},
        hyperparameters=PARAMS,
        retrained_reason=(
            "Pin single-assay TDC hepatocyte fitted rows and replace unknown legacy source"
        ),
    )
    META.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"N={len(fitted)} scaffold_CV_R2={cv_r2:.3f} scaffold_CV_AAFE={cv_aafe:.3f}")
    print(f"dataset_sha256={sha256(DATASET)} model_sha256={sha256(MODEL)}")


if __name__ == "__main__":
    main()
