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
    SOURCE_WORKBOOKS,
    sha256,
    training_rows,
)


def test_public_cmax_fitted_rows_and_artifact_are_pinned():
    with DATASET.open(newline="") as handle:
        fitted = list(csv.DictReader(handle))
    assert sha256(SOURCE) == SOURCE_SHA
    assert fitted == training_rows()
    assert len(fitted) == 907
    blood_names = {"indapamide", "cyclosporine", "everolimus", "tacrolimus", "pimecrolimus", "voclosporin"}
    assert not any(row["name"] in blood_names for row in fitted)
    with (ROOT / "data/validation/cmax_administered_analyte_mismatch_v1.csv").open(newline="") as handle:
        mismatched_names = {row["name"] for row in csv.DictReader(handle)}
    assert len(mismatched_names) == 114
    assert not any(row["name"] in mismatched_names for row in fitted)
    assert not any(row["name"] == "dolasetron" for row in fitted)
    for name in ("mmpk_expanded_full.csv", "mmpk_expanded_v2.csv", "mmpk_pbpk_features.csv"):
        with (ROOT / "data/training" / name).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
            assert not any(row["name"] in blood_names | mismatched_names | {"dolasetron"} for row in rows)
            if name == "mmpk_pbpk_features.csv":
                assert not any(row["name"] in {"lisdexamfetamine", "zofenopril"} for row in rows)
    with (ROOT / "data/training/clf_training.csv").open(newline="") as handle:
        clf_rows = list(csv.DictReader(handle))
    assert not any(row["name"] in blood_names | mismatched_names | {"dolasetron"} for row in clf_rows)
    for name, dose in (("lisdexamfetamine", 30 * 263.385 / 455.60), ("zofenopril", 57.3)):
        row = next(row for row in fitted if row["name"] == name)
        assert math.isclose(float(row["dose_mg"]), dose)
        assert math.isclose(float(row["log_cmax_per_dose"]), math.log10(float(row["cmax_mg_L"]) / dose))
    felbamate = next(row for row in fitted if row["name"] == "felbamate")
    assert float(felbamate["cmax_mg_L"]) == 8.9
    assert math.isclose(float(felbamate["log_cmax_per_dose"]), math.log10(8.9 / 600))

    metadata = json.loads(META.read_text())
    assert metadata["trained_on"]["dataset_path"] == str(DATASET.relative_to(ROOT))
    assert metadata["trained_on"]["sha256"] == sha256(DATASET)
    assert sha256(DATASET) == "d35d556443a8bc75ccf925c9e31a26d5d8644379e458fd1c0e265b19e43d3ebc"
    assert metadata["trained_on"]["n_drugs_clean"] == 907
    assert metadata["trained_on"]["source_workbooks"] == SOURCE_WORKBOOKS
    assert metadata["artifact_sha256"] == sha256(MODEL)
    assert sha256(MODEL) == "7430d602fb196a71c3c087a8cc3448d7feaa4d74e0175de9bb23084ab70bfe33"
    assert metadata["holdout_metric"]["name"] == "five_fold_scaffold_cv_aafe"
    assert math.isclose(metadata["holdout_metric"]["value"], 3.338214776718077)

    clf_dataset = ROOT / "data/training/clf_training.csv"
    for model_name, expected_n, expected_hash in (
        ("clf", 901, "c48ad93f5337658dfadea8923db29536c99faa7dcd8673372ee88d3423e8d636"),
        ("vdf", 832, "efdebbe4e0876927fb1fdb2c1b25b022eb1290221e81b5335fd3d4a3bd3e9761"),
    ):
        model = ROOT / f"models/direct_pk/xgboost_{model_name}.json"
        meta = json.loads(model.with_suffix(".meta.json").read_text())
        assert meta["trained_on"]["sha256"] == sha256(clf_dataset)
        assert meta["trained_on"]["n_drugs_clean"] == expected_n
        assert meta["artifact_sha256"] == sha256(model) == expected_hash
