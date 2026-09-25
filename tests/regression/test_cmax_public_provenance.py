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
    assert len(fitted) == 906
    blood_names = {
        "indapamide", "cyclosporine", "everolimus", "tacrolimus", "pimecrolimus", "voclosporin",
    }
    assert not any(row["name"] in blood_names for row in fitted)
    mismatch_path = ROOT / "data/validation/cmax_administered_analyte_mismatch_v1.csv"
    with mismatch_path.open(newline="") as handle:
        mismatched_names = {row["name"] for row in csv.DictReader(handle)}
    assert len(mismatched_names) == 114
    assert not any(row["name"] in mismatched_names for row in fitted)
    assert not any(row["name"] == "dolasetron" for row in fitted)
    assert not any(row["name"] == "methenamine" for row in fitted)
    excluded = blood_names | mismatched_names | {"dolasetron", "methenamine"}
    for name in ("mmpk_expanded_full.csv", "mmpk_expanded_v2.csv", "mmpk_pbpk_features.csv"):
        with (ROOT / "data/training" / name).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
            assert not any(row["name"] in excluded for row in rows)
            if name == "mmpk_pbpk_features.csv":
                salt_names = {
                    "lisdexamfetamine", "zofenopril", "metformin", "bupropion",
                    "pyridostigmine", "trospium", "methenamine",
                    "Acoramidis", "Givinostat",
                }
                assert not any(row["name"] in salt_names for row in rows)
    with (ROOT / "data/training/clf_training.csv").open(newline="") as handle:
        clf_rows = list(csv.DictReader(handle))
    assert not any(row["name"] in excluded for row in clf_rows)
    for name, dose in (
        ("lisdexamfetamine", 30 * 263.385 / 455.60), ("zofenopril", 57.3),
        ("metformin", 389.93), ("bupropion", 100 * 239.74 / 276.20),
        ("pyridostigmine", 120 * 181.21 / 261.12), ("trospium", 60 * 392.51 / 427.96),
        ("Acoramidis", 50 * 292.13 / 328.77),
    ):
        row = next(row for row in fitted if row["name"] == name)
        assert math.isclose(float(row["dose_mg"]), dose)
        assert math.isclose(
            float(row["log_cmax_per_dose"]), math.log10(float(row["cmax_mg_L"]) / dose)
        )
    felbamate = next(row for row in fitted if row["name"] == "felbamate")
    assert float(felbamate["cmax_mg_L"]) == 8.9
    assert math.isclose(float(felbamate["log_cmax_per_dose"]), math.log10(8.9 / 600))
    givinostat = next(row for row in fitted if row["name"] == "Givinostat")
    assert math.isclose(float(givinostat["dose_mg"]), 50 * 8.86 / 10)
    assert math.isclose(float(givinostat["cmax_mg_L"]), 0.053 * 8.86 / 10)
    assert math.isclose(
        float(givinostat["log_cmax_per_dose"]),
        math.log10(float(givinostat["cmax_mg_L"]) / float(givinostat["dose_mg"])),
    )
    for name in ("mmpk_expanded_full.csv", "mmpk_expanded_v2.csv"):
        with (ROOT / "data/training" / name).open(newline="") as handle:
            arms = [row for row in csv.DictReader(handle) if row["name"] == "Givinostat"]
        assert len(arms) == 5
        for row, (dose, cmax, auc) in zip(
            arms, ((50, 53, 359), (100, 99, 671), (200, 236, 1346),
                   (400, 526.3, 2503), (600, 542, 3543)), strict=True
        ):
            assert math.isclose(float(row["dose_mg"]), dose * 0.886)
            assert math.isclose(float(row["cmax_mg_L"]), cmax * 0.886 / 1000)
            assert math.isclose(float(row["auc_ng_h_ml"]), auc * 0.886)

    metadata = json.loads(META.read_text())
    assert metadata["trained_on"]["dataset_path"] == str(DATASET.relative_to(ROOT))
    assert metadata["trained_on"]["sha256"] == sha256(DATASET)
    assert sha256(DATASET) == "a39e03c26298e2c3af41152e5025f2ad71cac30f879d858258abc96aa9054117"
    assert metadata["trained_on"]["n_drugs_clean"] == 906
    assert metadata["trained_on"]["source_workbooks"] == SOURCE_WORKBOOKS
    assert metadata["artifact_sha256"] == sha256(MODEL)
    assert sha256(MODEL) == "9f208986df039551e8170fb90a1af883bcea87bc2d359d6d9ac65362a54ab011"
    assert metadata["holdout_metric"]["name"] == "five_fold_scaffold_cv_aafe"
    assert math.isclose(metadata["holdout_metric"]["value"], 3.382722659199057)

    clf_dataset = ROOT / "data/training/clf_training.csv"
    for model_name, expected_n, expected_hash in (
        ("clf", 900, "11369f5d1c1944416424ff762d0e40d70da1a5b72c17d67ab8fab687635e95bd"),
        ("vdf", 831, "1e23113191dbf138c7f18a0622ca477be25158c730c665d1864dbbec75b68852"),
    ):
        model = ROOT / f"models/direct_pk/xgboost_{model_name}.json"
        meta = json.loads(model.with_suffix(".meta.json").read_text())
        assert meta["trained_on"]["sha256"] == sha256(clf_dataset)
        assert meta["trained_on"]["n_drugs_clean"] == expected_n
        assert meta["artifact_sha256"] == sha256(model) == expected_hash
