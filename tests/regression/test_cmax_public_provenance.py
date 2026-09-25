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
    assert len(fitted) == 1024
    blood_names = {"indapamide", "cyclosporine", "everolimus", "tacrolimus"}
    assert not any(row["name"] in blood_names for row in fitted)
    for name in ("mmpk_expanded_full.csv", "mmpk_expanded_v2.csv", "mmpk_pbpk_features.csv"):
        with (ROOT / "data/training" / name).open(newline="") as handle:
            assert not any(row["name"] in blood_names for row in csv.DictReader(handle))
    felbamate = next(row for row in fitted if row["name"] == "felbamate")
    assert float(felbamate["cmax_mg_L"]) == 8.9
    assert math.isclose(float(felbamate["log_cmax_per_dose"]), math.log10(8.9 / 600))

    metadata = json.loads(META.read_text())
    assert metadata["trained_on"]["dataset_path"] == str(DATASET.relative_to(ROOT))
    assert metadata["trained_on"]["sha256"] == sha256(DATASET)
    assert sha256(DATASET) == "f4369df7ad8d7c7a46e1fb875881ef0aee4b0853db7b9c4163c0ddd44d416bb6"
    assert metadata["trained_on"]["n_drugs_clean"] == 1024
    assert metadata["artifact_sha256"] == sha256(MODEL)
    assert sha256(MODEL) == "078ad396add8241982be386fd534ea66a3415b9685508cde577fa6536f045a1b"
