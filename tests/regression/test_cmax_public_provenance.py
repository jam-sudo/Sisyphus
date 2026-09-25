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
                assert not any(row["name"] in {"lisdexamfetamine", "zofenopril", "metformin", "bupropion", "pyridostigmine", "trospium"} for row in rows)
    with (ROOT / "data/training/clf_training.csv").open(newline="") as handle:
        clf_rows = list(csv.DictReader(handle))
    assert not any(row["name"] in blood_names | mismatched_names | {"dolasetron"} for row in clf_rows)
    for name, dose in (
        ("lisdexamfetamine", 30 * 263.385 / 455.60), ("zofenopril", 57.3),
        ("metformin", 389.93), ("bupropion", 100 * 239.74 / 276.20),
        ("pyridostigmine", 120 * 181.21 / 261.12), ("trospium", 60 * 392.51 / 427.96),
    ):
        row = next(row for row in fitted if row["name"] == name)
        assert math.isclose(float(row["dose_mg"]), dose)
        assert math.isclose(float(row["log_cmax_per_dose"]), math.log10(float(row["cmax_mg_L"]) / dose))
    felbamate = next(row for row in fitted if row["name"] == "felbamate")
    assert float(felbamate["cmax_mg_L"]) == 8.9
    assert math.isclose(float(felbamate["log_cmax_per_dose"]), math.log10(8.9 / 600))

    metadata = json.loads(META.read_text())
    assert metadata["trained_on"]["dataset_path"] == str(DATASET.relative_to(ROOT))
    assert metadata["trained_on"]["sha256"] == sha256(DATASET)
    assert sha256(DATASET) == "c7f771ec43fab0ac001ec00ec9536134baafc36958c8b509346f0bdc50de338e"
    assert metadata["trained_on"]["n_drugs_clean"] == 907
    assert metadata["trained_on"]["source_workbooks"] == SOURCE_WORKBOOKS
    assert metadata["artifact_sha256"] == sha256(MODEL)
    assert sha256(MODEL) == "c084a39ad3f5fdaa694be10975e76ad91f1b1450738b542813f4a95cf5ab1551"
    assert metadata["holdout_metric"]["name"] == "five_fold_scaffold_cv_aafe"
    assert math.isclose(metadata["holdout_metric"]["value"], 3.343907668395042)

    clf_dataset = ROOT / "data/training/clf_training.csv"
    for model_name, expected_n, expected_hash in (
        ("clf", 901, "90452ac18696b96c4c41ccd9c66fbed325d2b9f8273f1fdcfb7329e1505e5c9a"),
        ("vdf", 832, "69cce14f803ffc3a84da38a0c3ebcfca088f9f95f7d784d4c2fa90d04d451a88"),
    ):
        model = ROOT / f"models/direct_pk/xgboost_{model_name}.json"
        meta = json.loads(model.with_suffix(".meta.json").read_text())
        assert meta["trained_on"]["sha256"] == sha256(clf_dataset)
        assert meta["trained_on"]["n_drugs_clean"] == expected_n
        assert meta["artifact_sha256"] == sha256(model) == expected_hash
