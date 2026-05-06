"""Tests for _get_fm_fractions non_cyp_fractions extension (issue #10)."""
from __future__ import annotations

import pytest

from sisyphus.predict.ivive import _get_fm_fractions


def test_non_cyp_fractions_none_preserves_existing():
    """non_cyp_fractions=None must produce identical output to no kwarg."""
    a = _get_fm_fractions("acid", substrate_enzymes=None, ugt_enzymes=None)
    b = _get_fm_fractions(
        "acid", substrate_enzymes=None, ugt_enzymes=None, non_cyp_fractions=None
    )
    assert a == b


def test_non_cyp_fractions_empty_preserves_existing():
    a = _get_fm_fractions("acid", substrate_enzymes=None, ugt_enzymes=None)
    b = _get_fm_fractions(
        "acid", substrate_enzymes=None, ugt_enzymes=None, non_cyp_fractions={}
    )
    assert a == b


def test_non_cyp_fractions_nat2_only_routes_to_nat2():
    out = _get_fm_fractions(
        "acid",
        substrate_enzymes=None,
        ugt_enzymes=None,
        non_cyp_fractions={"NAT2": 0.90},
    )
    assert out["NAT2"] == pytest.approx(0.90)
    cyp_total = sum(v for k, v in out.items() if k != "NAT2")
    assert cyp_total == pytest.approx(0.10)
    assert sum(out.values()) == pytest.approx(1.0)


def test_non_cyp_fractions_ugt1a1_only():
    out = _get_fm_fractions(
        "neutral",
        substrate_enzymes=None,
        ugt_enzymes=None,
        non_cyp_fractions={"UGT1A1": 0.70},
    )
    assert out["UGT1A1"] == pytest.approx(0.70)
    assert sum(v for k, v in out.items() if k != "UGT1A1") == pytest.approx(0.30)


def test_non_cyp_fractions_both_genes():
    out = _get_fm_fractions(
        "neutral",
        substrate_enzymes=None,
        ugt_enzymes=None,
        non_cyp_fractions={"NAT2": 0.40, "UGT1A1": 0.40},
    )
    assert out["NAT2"] == pytest.approx(0.40)
    assert out["UGT1A1"] == pytest.approx(0.40)
    assert sum(out.values()) == pytest.approx(1.0)


def test_non_cyp_fractions_value_out_of_range_raises():
    with pytest.raises(ValueError):
        _get_fm_fractions(
            "acid",
            substrate_enzymes=None,
            ugt_enzymes=None,
            non_cyp_fractions={"NAT2": 1.5},
        )


def test_non_cyp_fractions_negative_raises():
    with pytest.raises(ValueError):
        _get_fm_fractions(
            "acid",
            substrate_enzymes=None,
            ugt_enzymes=None,
            non_cyp_fractions={"NAT2": -0.1},
        )
