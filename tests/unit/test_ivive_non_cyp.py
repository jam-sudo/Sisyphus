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


def test_build_drug_on_graph_non_cyp_default_none_unchanged(monkeypatch):
    """build_drug_on_graph default kwarg path unchanged for non-NAT2/UGT1A1 drug."""
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.ivive import build_drug_on_graph

    smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # caffeine
    profile = compute_profile(smiles)
    adme = predict_adme(profile)

    drug_default = build_drug_on_graph(profile, adme, dose_mg=100.0, route="oral")
    drug_explicit_none = build_drug_on_graph(
        profile, adme, dose_mg=100.0, route="oral", non_cyp_fractions=None,
    )
    drug_explicit_empty = build_drug_on_graph(
        profile, adme, dose_mg=100.0, route="oral", non_cyp_fractions={},
    )
    # Same enzyme_affinity dict regardless of None vs {} vs unset
    assert set(drug_default.enzyme_affinity.keys()) == set(drug_explicit_none.enzyme_affinity.keys())
    assert set(drug_default.enzyme_affinity.keys()) == set(drug_explicit_empty.enzyme_affinity.keys())
    for tag in drug_default.enzyme_affinity:
        assert drug_default.enzyme_affinity[tag].mean == pytest.approx(
            drug_explicit_none.enzyme_affinity[tag].mean
        )


def test_build_drug_on_graph_isoniazid_with_non_cyp_fractions():
    """Isoniazid + non_cyp_fractions={'NAT2': 0.9} produces non-zero NAT2 affinity."""
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.ivive import build_drug_on_graph

    smiles = "NNC(=O)c1ccncc1"  # isoniazid
    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    drug = build_drug_on_graph(
        profile, adme, dose_mg=300.0, route="oral",
        non_cyp_fractions={"NAT2": 0.90},
    )
    assert "NAT2" in drug.enzyme_affinity
    assert drug.enzyme_affinity["NAT2"].mean > 0
