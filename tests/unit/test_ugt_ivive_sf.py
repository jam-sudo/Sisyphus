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


from sisyphus.predict.ivive import _decompose_clint
# match ivive.py's Distribution import from Step 1, e.g.:
from sisyphus.core import Distribution


def _aff(ugt_ivive_sf=None, fractions=None, ugt=None, ctype="base"):
    clint = Distribution(mean=50.0, cv=0.3)
    return _decompose_clint(
        clint, ctype, None,
        ugt_enzymes=ugt or {"UGT2B7"},
        non_cyp_fractions=fractions or {"UGT2B7": 0.85},
        ugt_ivive_sf=ugt_ivive_sf,
    )


def test_sf_none_and_empty_are_noop():
    base = _aff()
    assert _aff(ugt_ivive_sf=None)["UGT2B7"].mean == base["UGT2B7"].mean
    assert _aff(ugt_ivive_sf={})["UGT2B7"].mean == base["UGT2B7"].mean


def test_sf_scales_only_the_named_ugt():
    base = _aff()
    scaled = _aff(ugt_ivive_sf={"UGT2B7": 3.0})
    assert scaled["UGT2B7"].mean == base["UGT2B7"].mean * 3.0
    for tag in base:
        if tag != "UGT2B7":
            assert scaled[tag].mean == base[tag].mean, f"{tag} (non-UGT2B7) must be unchanged"


def test_multi_ugt_scales_each_tag_independently():
    fr = {"UGT2B7": 0.4, "UGT1A9": 0.4}
    base = _aff(fractions=fr, ugt={"UGT2B7", "UGT1A9"}, ctype="neutral")
    scaled = _aff(ugt_ivive_sf={"UGT2B7": 2.0, "UGT1A9": 5.0}, fractions=fr,
                  ugt={"UGT2B7", "UGT1A9"}, ctype="neutral")
    assert scaled["UGT2B7"].mean == base["UGT2B7"].mean * 2.0
    assert scaled["UGT1A9"].mean == base["UGT1A9"].mean * 5.0
