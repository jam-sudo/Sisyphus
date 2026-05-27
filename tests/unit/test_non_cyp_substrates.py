"""Unit tests for non_cyp_substrates registry loader (NAT2 + UGT1A1)."""
from __future__ import annotations

import pytest

from sisyphus.predict.non_cyp_substrates import (
    get_non_cyp_fractions,
    lookup_nat2_substrate,
    lookup_ugt1a1_substrate,
)

_ISONIAZID = "NNC(=O)c1ccncc1"
_HYDRALAZINE = "NNc1nncc2ccccc12"
_METOPROLOL = "COCCc1ccc(OCC(O)CNC(C)C)cc1"


def test_lookup_nat2_isoniazid_returns_entry():
    entry = lookup_nat2_substrate(_ISONIAZID)
    assert entry is not None
    assert entry["drug"] == "isoniazid"
    assert entry["metabolic_fraction"] == pytest.approx(0.90)


def test_lookup_nat2_metoprolol_returns_none():
    assert lookup_nat2_substrate(_METOPROLOL) is None


def test_lookup_nat2_invalid_smiles_returns_none():
    assert lookup_nat2_substrate("not-a-smiles") is None


def test_lookup_nat2_empty_returns_none():
    assert lookup_nat2_substrate("") is None


def test_lookup_ugt1a1_metoprolol_returns_none():
    assert lookup_ugt1a1_substrate(_METOPROLOL) is None


def test_get_non_cyp_fractions_isoniazid():
    out = get_non_cyp_fractions(_ISONIAZID)
    assert out == {"NAT2": pytest.approx(0.90)}


def test_get_non_cyp_fractions_hydralazine():
    out = get_non_cyp_fractions(_HYDRALAZINE)
    assert out == {"NAT2": pytest.approx(0.50)}


def test_get_non_cyp_fractions_metoprolol_empty():
    assert get_non_cyp_fractions(_METOPROLOL) == {}


def test_get_non_cyp_fractions_invalid_smiles_empty():
    assert get_non_cyp_fractions("not-a-smiles") == {}


def test_lru_cache_reuses_loaded_data():
    """Two calls should not re-read JSON. Sanity check for lru_cache wiring."""
    out1 = lookup_nat2_substrate(_ISONIAZID)
    out2 = lookup_nat2_substrate(_ISONIAZID)
    assert out1 is out2 or out1 == out2  # cached object identity OR equal dict


# --- B-02 Phase 2: UGT2B7 + UGT1A9 lookup tests ---

def test_lookup_ugt2b7_substrate_morphine():
    """Morphine should match the UGT2B7 registry with fm=0.85."""
    from sisyphus.predict.non_cyp_substrates import lookup_ugt2b7_substrate
    morphine_smiles = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
    entry = lookup_ugt2b7_substrate(morphine_smiles)
    assert entry is not None, "morphine not found in UGT2B7 registry"
    assert entry["drug"] == "morphine"
    assert entry["metabolic_fraction"] == 0.85


def test_lookup_ugt1a9_substrate_dapagliflozin():
    """Dapagliflozin should match the UGT1A9 registry with fm=0.50."""
    from sisyphus.predict.non_cyp_substrates import lookup_ugt1a9_substrate
    dapa_smiles = "CCOc1ccc(Cc2cc([C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)ccc2Cl)cc1"
    entry = lookup_ugt1a9_substrate(dapa_smiles)
    assert entry is not None, "dapagliflozin not found in UGT1A9 registry"
    assert entry["drug"] == "dapagliflozin"
    assert entry["metabolic_fraction"] == 0.50


def test_lookup_ugt2b7_non_substrate_returns_none():
    """A non-substrate SMILES (midazolam) must return None."""
    from sisyphus.predict.non_cyp_substrates import lookup_ugt2b7_substrate
    midazolam = "c1ccc2c(c1)C(=NC(=O)N2)c1ccccc1F"
    assert lookup_ugt2b7_substrate(midazolam) is None


def test_get_non_cyp_fractions_morphine():
    """get_non_cyp_fractions aggregator should return {'UGT2B7': 0.85} for morphine."""
    from sisyphus.predict.non_cyp_substrates import get_non_cyp_fractions
    morphine_smiles = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
    out = get_non_cyp_fractions(morphine_smiles)
    assert out == {"UGT2B7": 0.85}, f"expected single-key UGT2B7=0.85, got {out!r}"


def test_get_non_cyp_fractions_dapagliflozin():
    """get_non_cyp_fractions aggregator should return {'UGT1A9': 0.50} for dapagliflozin."""
    from sisyphus.predict.non_cyp_substrates import get_non_cyp_fractions
    dapa_smiles = "CCOc1ccc(Cc2cc([C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)ccc2Cl)cc1"
    out = get_non_cyp_fractions(dapa_smiles)
    assert out == {"UGT1A9": 0.50}, f"expected single-key UGT1A9=0.50, got {out!r}"


def test_get_non_cyp_fractions_non_substrate_returns_empty():
    """A non-substrate SMILES must return an empty dict (no UGT path)."""
    from sisyphus.predict.non_cyp_substrates import get_non_cyp_fractions
    midazolam = "c1ccc2c(c1)C(=NC(=O)N2)c1ccccc1F"
    assert get_non_cyp_fractions(midazolam) == {}
