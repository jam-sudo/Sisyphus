"""Regression: measured_adme=None must leave the SMILES-only path bit-identical."""
import pytest

from sisyphus.pipeline.predict import predict
from sisyphus.predict.adme import MeasuredADMEInput

# Diverse, valid SMILES — identity-agnostic.
_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",                   # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",            # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",              # ibuprofen
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",   # warfarin
]


@pytest.mark.parametrize("smiles", _SMILES)
def test_none_is_bitidentical(smiles):
    a = predict(smiles, 100.0)
    b = predict(smiles, 100.0, measured_adme=None)
    assert a.pk.cmax.mean == b.pk.cmax.mean
    assert a.engine_pk is not None and b.engine_pk is not None
    assert a.engine_pk.cmax.mean == b.engine_pk.cmax.mean


def test_measured_changes_cmax():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    base = predict(smiles, 100.0)
    meas = predict(smiles, 100.0,
                   measured_adme=MeasuredADMEInput(fup=0.20, clint=200.0))
    assert base.engine_pk.cmax.mean != meas.engine_pk.cmax.mean


def test_warning_tag_present():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    r = predict(smiles, 100.0,
                measured_adme=MeasuredADMEInput(fup=0.20, clint=200.0))
    assert any("measured_adme" in w for w in r.warnings)
