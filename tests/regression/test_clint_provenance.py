"""Guard: the production CLint model must stay single-assay (hepatocyte).

Mixing non-hepatocyte assays (TDC microsome, ChEMBL, Biogen/Fang) into the CLint
training label halves the hepatocyte-holdout R^2 (DE-52: ~0.24-0.26 -> ~0.11-0.15,
1.7-2.2x; microsome is the dominant culprit). The mixed-assay experiments
(`xgboost_clint_expanded.json`, `xgboost_clint_v3_biogen.json`) were reverted on
Meta-AAFE worsening (DE-11 / DE-16) and must never be promoted into the production
artifact `xgboost_clint.json` that `predict/adme.py` loads.

This test pins that invariant two ways: (1) the production model's meta declares
single-assay hepatocyte provenance; (2) the production model file is byte-distinct
from every known mixed-assay artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.retrain_clint_public import (
    DATASET,
    MODEL,
    ROOT,
    SOURCE,
    SOURCE_SHA,
    sha256,
    training_rows,
)
from sisyphus.validation.identity import ik14

_ADME = Path(__file__).resolve().parents[2] / "models" / "adme"
_PROD = _ADME / "xgboost_clint.json"
_PROD_META = _ADME / "xgboost_clint.meta.json"
_MIXED_ARTIFACTS = ("xgboost_clint_expanded.json", "xgboost_clint_v3_biogen.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_clint_meta_declares_hepatocyte_only() -> None:
    """The shipped CLint model names the exact single-assay fitted rows."""
    meta = json.loads(_PROD_META.read_text())
    dataset = meta["trained_on"]["dataset_path"]
    assert dataset == str(DATASET.relative_to(ROOT)), (
        f"production CLint provenance is {dataset!r}, not single-assay hepatocyte. "
        "If a mixed-assay model was promoted, hepatocyte-holdout R^2 is ~halved (DE-52)."
    )
    assert sha256(SOURCE) == SOURCE_SHA
    assert meta["trained_on"]["sha256"] == sha256(DATASET)
    assert sha256(DATASET) == "300f5be5d0ada5fad88d94f9dc60c0c13ab77b122fed2494e06d764b43f6cd07"
    assert meta["trained_on"]["n_drugs_clean"] == 996
    assert meta["artifact_sha256"] == sha256(MODEL)
    assert sha256(MODEL) == "1fdafa7c2e287dd9e4f44f68b97e5ace80a6837f97ef17bed3c01daeff610401"


def test_production_clint_fitted_rows_match_filtered_source() -> None:
    expected = training_rows()
    fitted = pd.read_csv(DATASET)
    assert len(expected) == len(fitted) == 996
    assert [ik14(s) for s in fitted["canonical_smiles"]] == [
        ik14(s) for s in expected["canonical_smiles"]
    ]
    assert fitted["drug_id"].tolist() == expected["drug_id"].tolist()
    np.testing.assert_allclose(fitted["Y"], expected["Y"], atol=3e-14, rtol=0)


def test_production_clint_is_not_a_known_mixed_assay_model() -> None:
    """The production artifact must not be byte-identical to any reverted mixed model."""
    prod_hash = _sha256(_PROD)
    for name in _MIXED_ARTIFACTS:
        artifact = _ADME / name
        if not artifact.exists():
            continue  # a missing experimental artifact cannot have been promoted
        assert prod_hash != _sha256(artifact), (
            f"xgboost_clint.json is byte-identical to the mixed-assay model {name} — "
            "this halves hepatocyte-holdout R^2 (DE-52). Revert to the hepatocyte model."
        )
