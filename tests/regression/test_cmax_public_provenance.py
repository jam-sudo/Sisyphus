"""The shipped Cmax model is tied to the public holdout-excluded Omega rows."""

import csv
import json

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

    metadata = json.loads(META.read_text())
    assert metadata["trained_on"]["dataset_path"] == str(DATASET.relative_to(ROOT))
    assert metadata["trained_on"]["sha256"] == sha256(DATASET)
    assert sha256(DATASET) == "5668d3747e11d03d8c3d57c03ff0842ed3aa330f4c938386e47c05e1e318c919"
    assert metadata["trained_on"]["n_drugs_clean"] == 1028
    assert metadata["artifact_sha256"] == sha256(MODEL)
    assert sha256(MODEL) == "14391eb0881cb3ec83ab2f81592c5fe75da7c949290ec438fddc2d7136f792a0"
