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
    assert len(fitted) == 1028
    felbamate = next(row for row in fitted if row["name"] == "felbamate")
    assert float(felbamate["cmax_mg_L"]) == 8.9
    assert math.isclose(float(felbamate["log_cmax_per_dose"]), math.log10(8.9 / 600))

    metadata = json.loads(META.read_text())
    assert metadata["trained_on"]["dataset_path"] == str(DATASET.relative_to(ROOT))
    assert metadata["trained_on"]["sha256"] == sha256(DATASET)
    assert sha256(DATASET) == "c81553eb73fb07856b66718badbc3084417a79804243f98b1ec21c79915cc9dc"
    assert metadata["trained_on"]["n_drugs_clean"] == 1028
    assert metadata["artifact_sha256"] == sha256(MODEL)
    assert sha256(MODEL) == "8e0134b56916db68e51bdf23b89e6dd37249a4c5d0fc93edb8d130008acd9c28"
