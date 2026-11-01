"""The shipped Peff model is tied to the public fitted-row snapshot."""

import hashlib
import json

import numpy as np
import pandas as pd

from scripts.train_peff import (
    CLINICAL_PK_JSON,
    DATASET,
    HOLDOUT_JSON,
    OUTPUT_MODEL,
    ROOT,
    SOURCE,
    SOURCE_SHA256,
    build_holdout_keys,
    load_caco2_wang,
)
from sisyphus.validation.identity import ik14


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_peff_fitted_rows_and_artifact_are_pinned():
    holdout = json.loads(HOLDOUT_JSON.read_text())["holdout"]
    clinical = json.loads(CLINICAL_PK_JSON.read_text())
    expected = load_caco2_wang(build_holdout_keys(holdout, clinical))
    fitted = pd.read_csv(DATASET)
    assert _sha(SOURCE) == SOURCE_SHA256
    assert len(expected) == len(fitted) == 874
    # RDKit can serialize one stereobond differently across macOS/Linux; the
    # normalized molecular identities and fitted targets must still agree.
    assert [ik14(s) for s in fitted["canonical_smiles"]] == [
        ik14(s) for s in expected["canonical_smiles"]
    ]
    assert fitted["drug_id"].tolist() == expected["drug_id"].tolist()
    np.testing.assert_allclose(fitted["log_peff"], expected["log_peff"], atol=1e-15, rtol=0)

    metadata = json.loads(OUTPUT_MODEL.with_suffix(".meta.json").read_text())
    assert metadata["trained_on"]["dataset_path"] == str(DATASET.relative_to(ROOT))
    assert metadata["trained_on"]["sha256"] == _sha(DATASET)
    assert _sha(DATASET) == "5159f816c73f90c7c6ef8921983c082cf066a6a2ea18a91c415a04e3eb49e57b"
    assert metadata["trained_on"]["n_drugs_clean"] == 874
    assert metadata["artifact_sha256"] == _sha(OUTPUT_MODEL)
    assert _sha(OUTPUT_MODEL) == "b3b62cb12f80785d44ffedfae6c50f202c47e06423f168591b7ee4877cfaf5c0"
