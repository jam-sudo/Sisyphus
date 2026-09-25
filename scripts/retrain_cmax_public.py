#!/usr/bin/env python3
"""Rebuild the direct Cmax model from the pinned, holdout-excluded Omega rows."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_clint_expanded import scaffold_split_indices  # noqa: E402
from scripts.train_peff import build_holdout_keys, is_holdout  # noqa: E402
from sisyphus.descriptors import compute_features  # noqa: E402

SOURCE = ROOT / "data/training/omega_mmpk_clean.csv"
SOURCE_SHA = "e7228d14bdfdfc6c790177207779630c1e5655c19d451528c87b80e2e9de9c3d"
SOURCE_WORKBOOKS = [
    {"url": "https://github.com/jam-sudo/Omega/blob/08a45047a2b5dcdca8c9a8f36ff1fe3b50ed3d6d/data/external/mmpk/approved.xlsx",
     "sha256": "4521a2d89dde71c9e9381ab0a920e84dfc69c714860de377cd73b338572026d2"},
    {"url": "https://github.com/jam-sudo/Omega/blob/08a45047a2b5dcdca8c9a8f36ff1fe3b50ed3d6d/data/external/mmpk/approved_2024.xlsx",
     "sha256": "bfdc1bac564dcf3a29f2f8021f69c4f6e087a687bea329433ccd47d233b4d32f"},
]
MISMATCH = ROOT / "data/validation/cmax_administered_analyte_mismatch_v1.csv"
MISMATCH_SHA = "09dca8c11133045cb75d9351b940311064b6b36dbaa2f95c3d5bd1b9de236fff"
DATASET = ROOT / "data/training/cmax_omega_public_clean.csv"
MODEL = ROOT / "models/direct_pk/xgboost_cmax.json"
META = MODEL.with_suffix(".meta.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_rows() -> list[dict[str, str]]:
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("Omega Cmax source SHA256 changed")
    holdout = json.loads((ROOT / "data/reference/holdout.json").read_text())["holdout"]
    clinical = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())
    keys = build_holdout_keys(holdout, clinical)
    with SOURCE.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = [row for row in source if not is_holdout(row["smiles"], row["name"], keys)]
    if (len(source), len(rows)) != (1128, 1028):
        raise ValueError(f"Unexpected Omega source/clean row counts: {len(source)}/{len(rows)}")
    # These source aggregates measure whole blood, not the model's plasma target.
    blood_rows = {
        "indapamide": (5, 2, np.sqrt(263 * 231) / 1000),
        "cyclosporine": (372.5, 1, 1.22),
        "everolimus": (2, 3, (17.9 * 17.1 * 16.7) ** (1 / 3) / 1000),
        "tacrolimus": (5, 2, np.sqrt(27.23 * 40.62) / 1000),
        "pimecrolimus": (15, 1, 41.2 / 1000),
        "voclosporin": (18.07, 1, 32 / 1000),
    }
    for name, (dose, studies, cmax) in blood_rows.items():
        matched = [row for row in rows if row["name"] == name]
        if (len(matched) != 1 or float(matched[0]["dose_mg"]) != dose
                or int(matched[0]["n_studies"]) != studies
                or not np.isclose(float(matched[0]["cmax_mg_L"]), cmax)):
            raise ValueError(f"{name} source row changed; re-adjudicate its matrix")
        rows.remove(matched[0])
    if sha256(MISMATCH) != MISMATCH_SHA:
        raise ValueError("Administered/analyte mismatch ledger changed; re-adjudicate")
    with MISMATCH.open(newline="") as handle:
        mismatches = list(csv.DictReader(handle))
    if len(mismatches) != 114 or len({r["name"] for r in mismatches}) != 114:
        raise ValueError("Unexpected administered/analyte mismatch ledger")
    for mismatch in mismatches:
        matched = [row for row in rows if row["name"] == mismatch["name"]]
        if (len(matched) != 1
                or float(matched[0]["dose_mg"]) != float(mismatch["dose_mg"])
                or int(matched[0]["n_studies"]) != int(mismatch["n_studies"])
                or not np.isclose(float(matched[0]["cmax_mg_L"]), float(mismatch["cmax_mg_L"]))):
            raise ValueError(f"{mismatch['name']} source row changed; re-adjudicate administered identity")
        rows.remove(matched[0])
    # PMID 8453023: oral parent dolasetron was too sparse for PK analysis;
    # the reported concentration profile is reduced dolasetron (hydrodolasetron).
    dolasetron = [row for row in rows if row["name"] == "dolasetron"]
    if (len(dolasetron) != 1 or float(dolasetron[0]["dose_mg"]) != 200
            or int(dolasetron[0]["n_studies"]) != 1
            or not np.isclose(float(dolasetron[0]["cmax_mg_L"]), 0.5781)):
        raise ValueError("Dolasetron source row changed; re-adjudicate its analyte")
    rows.remove(dolasetron[0])
    # Richens et al. 1997, Table 1: 600 mg young low-dose Cmax is 8.9 µg/mL.
    # The archived Omega row divided by 1000 as if that value were ng/mL.
    felbamate = [row for row in rows if row["name"] == "felbamate"]
    if len(felbamate) != 1 or not np.isclose(float(felbamate[0]["cmax_mg_L"]), 0.0089):
        raise ValueError("Felbamate source row changed; re-adjudicate its units")
    felbamate[0]["cmax_mg_L"] = "8.9"
    felbamate[0]["log_cmax_per_dose"] = str(float(np.log10(8.9 / 600)))
    return rows


def main() -> None:
    rows = training_rows()
    with DATASET.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with DATASET.open(newline="") as handle:
        fitted = list(csv.DictReader(handle))
    X = np.asarray([compute_features(row["smiles"]) for row in fitted])
    y = np.asarray([float(row["log_cmax_per_dose"]) for row in fitted])
    metadata = json.loads(META.read_text())
    params = metadata["hyperparameters"]

    oof = np.empty_like(y)
    for test in scaffold_split_indices([row["smiles"] for row in fitted]):
        train = np.setdiff1d(np.arange(len(y)), test)
        model = xgb.XGBRegressor(**params).fit(X[train], y[train])
        oof[test] = model.predict(X[test])
    cv_aafe = float(10 ** np.mean(np.abs(oof - y)))
    cv_r2 = float(1 - np.sum((oof - y) ** 2) / np.sum((y - y.mean()) ** 2))

    model = xgb.XGBRegressor(**params).fit(X, y)
    model.save_model(MODEL)
    metadata.update(
        version="v3_public_omega",
        artifact_sha256=sha256(MODEL),
        trained_at=datetime.now(timezone.utc).isoformat(),
        trained_on={"dataset_path": str(DATASET.relative_to(ROOT)),
                    "sha256": sha256(DATASET), "n_drugs_clean": len(fitted),
                    "source_sha256": SOURCE_SHA, "source_workbooks": SOURCE_WORKBOOKS},
        n_drugs_original=1128,
        n_drugs_excluded=221,
        holdout_version="N=107 (data/reference/holdout.json)",
        holdout_metric={"name": "five_fold_scaffold_cv_aafe", "value": cv_aafe, "r2": cv_r2},
        retrained_reason="Correct felbamate units; quarantine six blood-matrix, 114 administered/analyte-mismatched, and one misidentified dolasetron-metabolite Cmax label",
    )
    META.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"N={len(fitted)} CV_AAFE={cv_aafe:.3f} CV_R2={cv_r2:.3f}")
    print(f"dataset_sha256={sha256(DATASET)} model_sha256={sha256(MODEL)}")


if __name__ == "__main__":
    main()
