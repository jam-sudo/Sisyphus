#!/usr/bin/env python3
"""Rebuild the public-only fup model from the pinned human PPBR_AZ rows."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb
from rdkit import Chem
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sisyphus.descriptors import compute_features  # noqa: E402
from sisyphus.validation.identity import ik14  # noqa: E402

SOURCE = ROOT / "data/ppbr_az.tab"
DATASET = ROOT / "data/training/fup_tdc_public_clean.csv"
MODEL = ROOT / "models/adme/xgboost_fup_v2.json"
META = MODEL.with_suffix(".meta.json")
SOURCE_SHA = "54c9520f4b6e04419b18bab7335583c3865d86fc53db40f74c2cff1577c8ec1b"
PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, random_state=42, n_jobs=4, verbosity=0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_rows() -> tuple[list[tuple[str, str, float]], int]:
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("PPBR_AZ source SHA256 changed")
    holdout = json.loads((ROOT / "data/reference/holdout.json").read_text())["holdout"]
    clinical = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())["drugs"]
    names = {name.casefold() for name in holdout}
    structures = set()
    keys = set()
    for name in holdout:
        entry = clinical.get(name) or clinical.get(name.replace(" ", "_"))
        if not entry or not entry.get("smiles"):
            continue
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol:
            structures.add(Chem.MolToSmiles(mol, isomericSmiles=True))
        key = ik14(entry["smiles"])
        if key:
            keys.add(key)

    rows = []
    seen = set()
    human = 0
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["Species"] != "Homo sapiens":
                continue
            human += 1
            mol = Chem.MolFromSmiles(row["Drug"])
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            if (row["Drug_ID"].casefold() in names or canonical in structures
                    or ik14(row["Drug"]) in keys):
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            fup = (100.0 - float(row["Y"])) / 100.0
            if math.isfinite(fup) and 0.001 <= fup <= 0.999:
                rows.append((row["Drug_ID"], canonical, fup))
    if (human, len(rows)) != (1614, 1557):
        raise ValueError(f"Unexpected human/clean PPBR_AZ row counts: {human}/{len(rows)}")
    return rows, human


def main() -> None:
    rows, human = training_rows()
    with DATASET.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("name", "smiles", "fup"))
        writer.writerows((name, smiles, repr(fup)) for name, smiles, fup in rows)

    with DATASET.open(newline="") as handle:
        fitted = list(csv.DictReader(handle))
    X = np.asarray([compute_features(row["smiles"]) for row in fitted])
    y = np.asarray([math.log(float(row["fup"]) / (1 - float(row["fup"])))
                    for row in fitted])
    predictions = np.empty_like(y)
    for train, test in KFold(n_splits=5).split(X):
        model = xgb.XGBRegressor(**PARAMS).fit(X[train], y[train])
        predictions[test] = model.predict(X[test])
    observed = 1 / (1 + np.exp(-y))
    predicted = 1 / (1 + np.exp(-predictions))
    cv_aafe = float(np.exp(np.mean(np.abs(np.log(predicted / observed)))))
    cv_r2 = float(1 - np.sum((observed - predicted) ** 2)
                  / np.sum((observed - observed.mean()) ** 2))

    model = xgb.XGBRegressor(**PARAMS).fit(X, y)
    model.save_model(MODEL)
    metadata = json.loads(META.read_text())
    metadata.update(version="v2_public_tdc", artifact_sha256=sha256(MODEL),
                    trained_at=datetime.now(timezone.utc).isoformat(),
                    n_drugs_original=human, n_drugs_excluded=human - len(fitted),
                    holdout_version="N=107 (data/reference/holdout.json)",
                    holdout_metric={"name": "five_fold_cv_aafe", "value": cv_aafe,
                                    "r2": cv_r2}, hyperparameters=PARAMS,
                    retrained_reason=(
                        "Replace unpinned DrugBank fup targets with public-only TDC data"
                    ))
    metadata["trained_on"] = {"dataset_path": str(DATASET.relative_to(ROOT)),
                              "sha256": sha256(DATASET), "n_drugs_clean": len(fitted),
                              "source_sha256": SOURCE_SHA}
    META.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"N={len(fitted)} CV_AAFE={cv_aafe:.3f} CV_R2={cv_r2:.3f}")
    print(f"dataset_sha256={sha256(DATASET)} model_sha256={sha256(MODEL)}")


if __name__ == "__main__":
    main()
