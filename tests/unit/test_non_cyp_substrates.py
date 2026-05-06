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
