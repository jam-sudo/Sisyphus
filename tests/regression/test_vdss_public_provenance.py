"""The shipped VDss model is tied to the public Lombardo fitted rows."""

import json

import numpy as np
import pandas as pd

from scripts.retrain_vdss_public import (
    DATASET,
    META,
    MODEL,
    ROOT,
    SOURCE,
    SOURCE_SHA,
    SOURCE_URL,
    sha256,
    training_rows,
)
from sisyphus.validation.identity import ik14


def test_public_vdss_fitted_rows_and_artifact_are_pinned():
    expected = training_rows()
    fitted = pd.read_csv(DATASET)
    assert sha256(SOURCE) == SOURCE_SHA
    assert len(expected) == len(fitted) == 1055
    # Canonical SMILES string ordering can shift across RDKit versions.
    fitted = fitted.sort_values("drug_id").reset_index(drop=True)
    expected = expected.sort_values("drug_id").reset_index(drop=True)
    assert fitted["drug_id"].is_unique and expected["drug_id"].is_unique
    assert fitted["drug_id"].tolist() == expected["drug_id"].tolist()
    assert [ik14(s) for s in fitted["canonical_smiles"]] == [
        ik14(s) for s in expected["canonical_smiles"]
    ]
    np.testing.assert_allclose(fitted["Y"], expected["Y"], atol=2e-13, rtol=0)

    metadata = json.loads(META.read_text())
    assert metadata["trained_on"]["dataset_path"] == str(DATASET.relative_to(ROOT))
    assert metadata["trained_on"]["sha256"] == sha256(DATASET)
    assert metadata["trained_on"]["source_url"] == SOURCE_URL
    assert sha256(DATASET) == "778851b9dad82c2eb3d948b7ffb3ae829ce5fd529395b7d0599d4c86e02e5e54"
    assert metadata["trained_on"]["n_drugs_clean"] == 1055
    assert metadata["artifact_sha256"] == sha256(MODEL)
    assert sha256(MODEL) == "29f84cbff97da197ae516fccb2d91aff972f9bae661a8b2bde39786a984c1c35"
