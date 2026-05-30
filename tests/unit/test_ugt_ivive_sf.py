"""B-14 loader unit tests."""
from __future__ import annotations
from sisyphus.predict.non_cyp_substrates import get_ugt_ivive_sf

_CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # not a UGT substrate


def test_unlisted_returns_empty():
    assert get_ugt_ivive_sf(_CAFFEINE) == {}


def test_invalid_smiles_returns_empty_no_raise():
    assert get_ugt_ivive_sf("not_a_smiles") == {}
    assert get_ugt_ivive_sf("") == {}


def test_seed_returns_ugt_map():
    morphine = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
    sf = get_ugt_ivive_sf(morphine)
    assert "UGT2B7" in sf
    assert isinstance(sf["UGT2B7"], float)  # value set later by Phase 0; structure stable
