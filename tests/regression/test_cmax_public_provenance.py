"""The shipped Cmax model is tied to the public holdout-excluded Omega rows."""

import csv
import json
import math

from scripts.retrain_cmax_public import (
    DATASET,
    META,
    MODEL,
    ROOT,
    SOURCE,
    SOURCE_SHA,
    sha256,
    training_rows,
)


def test_public_cmax_fitted_rows_and_artifact_are_pinned():
    with DATASET.open(newline="") as handle:
        fitted = list(csv.DictReader(handle))
    assert sha256(SOURCE) == SOURCE_SHA
    assert fitted == training_rows()
    assert len(fitted) == 908
    blood_names = {"indapamide", "cyclosporine", "everolimus", "tacrolimus", "pimecrolimus", "voclosporin"}
    assert not any(row["name"] in blood_names for row in fitted)
    with (ROOT / "data/validation/cmax_administered_analyte_mismatch_v1.csv").open(newline="") as handle:
        mismatched_names = {row["name"] for row in csv.DictReader(handle)}
    assert len(mismatched_names) == 114
    assert not any(row["name"] in mismatched_names for row in fitted)
    for name in ("mmpk_expanded_full.csv", "mmpk_expanded_v2.csv", "mmpk_pbpk_features.csv"):
        with (ROOT / "data/training" / name).open(newline="") as handle:
            assert not any(row["name"] in blood_names | mismatched_names for row in csv.DictReader(handle))
    felbamate = next(row for row in fitted if row["name"] == "felbamate")
    assert float(felbamate["cmax_mg_L"]) == 8.9
    assert math.isclose(float(felbamate["log_cmax_per_dose"]), math.log10(8.9 / 600))

    metadata = json.loads(META.read_text())
    assert metadata["trained_on"]["dataset_path"] == str(DATASET.relative_to(ROOT))
    assert metadata["trained_on"]["sha256"] == sha256(DATASET)
    assert sha256(DATASET) == "f8fbfde1c07fbc22f2cd6e3f72304f429ed14745261a65b29de1ac8545013694"
    assert metadata["trained_on"]["n_drugs_clean"] == 908
    assert metadata["artifact_sha256"] == sha256(MODEL)
    assert sha256(MODEL) == "c776768146b0746a70e31af4446e25ba1245139b09b2d5497fb09d93ecd2b0e1"
