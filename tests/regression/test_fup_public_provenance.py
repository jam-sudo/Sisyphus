"""Public fup model membership is reproducible from the pinned TDC source."""

import csv
import json

import numpy as np
import pytest

from scripts.retrain_fup_public import DATASET, META, MODEL, ROOT, sha256, training_rows
from sisyphus.predict import adme


def test_public_fup_training_source_and_artifact_are_pinned():
    rows, human = training_rows()
    with DATASET.open(newline="") as handle:
        fitted = list(csv.DictReader(handle))
    assert human == 1614
    assert [(r["name"], r["smiles"], float(r["fup"])) for r in fitted] == rows
    metadata = json.loads(META.read_text())
    assert metadata["trained_on"]["dataset_path"] == str(DATASET.relative_to(ROOT))
    assert metadata["trained_on"]["sha256"] == sha256(DATASET)
    assert sha256(DATASET) == "e2c70c83707b2031904a351be76ce61899557a4b14316b9429c4a77fd2eac742"
    assert metadata["trained_on"]["n_drugs_clean"] == len(rows)
    assert metadata["artifact_sha256"] == sha256(MODEL)
    assert sha256(MODEL) == "b0731734730746866646a2628dded73d74254eb06ef1a6c7b457bf273d867096"


def test_missing_public_fup_model_does_not_load_legacy_artifact(tmp_path, monkeypatch):
    (tmp_path / "xgboost_fup.json").write_text("legacy model")
    monkeypatch.setattr(adme, "_MODEL_DIR", tmp_path)
    monkeypatch.setattr(adme, "_model_cache", {})
    with pytest.raises(FileNotFoundError, match="xgboost_fup_v2.json"):
        adme._predict_fup(np.zeros((1, 2057)))
