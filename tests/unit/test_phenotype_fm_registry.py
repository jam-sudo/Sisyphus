"""Phenotype fm registry matching and routing."""

import json
from pathlib import Path

from sisyphus.pipeline.predict import predict
from sisyphus.predict.phenotype import _fm_from_registry

_ROOT = Path(__file__).resolve().parents[2]


def test_registry_matches_structure_and_applied_gene():
    clinical = json.loads((_ROOT / "data/reference/clinical_pk.json").read_text())["drugs"]
    smiles = clinical["metoprolol"]["smiles"]
    assert _fm_from_registry(smiles, {"CYP2D6"}) == ("CYP2D6", 0.8)
    assert _fm_from_registry(smiles, {"CYP2C9"}) is None


def test_phenotype_uses_registry_only_when_requested():
    clinical = json.loads((_ROOT / "data/reference/clinical_pk.json").read_text())["drugs"]
    metoprolol = clinical["metoprolol"]
    result = predict(
        metoprolol["smiles"], metoprolol["dose_mg"],
        phenotypes={"CYP2D6": "PM"},
    )
    assert "fm:validation:pgx_fm_registry" in result.warnings
    assert "fm:compound-type default; phenotype effect is class-generic" not in result.warnings
