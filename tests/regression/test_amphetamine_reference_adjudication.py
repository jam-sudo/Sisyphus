"""Keep amphetamine reference identities and observed-dose evidence aligned."""

import json
from pathlib import Path

import pytest
from rdkit import Chem

from sisyphus.validation.reference import load_reference

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data/reference/clinical_pk.json"
DEXTRO_KEY = "KWTSXDURSIMDCE-QMMMGPOBSA-N"  # PubChem CID 5826


def test_amphetamine_reference_adjudication():
    data = json.loads(REFERENCE.read_text())
    drugs = data["drugs"]
    for name in ("d_amphetamine", "dextroamphetamine"):
        assert Chem.MolToInchiKey(Chem.MolFromSmiles(drugs[name]["smiles"])) == DEXTRO_KEY

    assert "cmax_mg_L" not in drugs["d_amphetamine"]["pk_params"]
    assert "ct_curve" not in drugs["d_amphetamine"]
    assert data["metadata"]["n_with_cmax"] == sum(
        bool(drug.get("pk_params", {}).get("cmax_mg_L")) for drug in drugs.values()
    )

    references = {reference.name: reference for reference in load_reference(REFERENCE)}
    assert "d_amphetamine" not in references
    assert references["dextroamphetamine"].dose_mg == pytest.approx(15 * 270.42 / 368.49, rel=1e-4)
    assert references["dextroamphetamine"].cmax_obs == pytest.approx(0.0366)
