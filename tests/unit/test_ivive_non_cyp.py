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
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
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
    assert set(drug_default.enzyme_affinity.keys()) == set(drug_explicit_none.enzyme_affinity.keys())  # noqa: E501
    assert set(drug_default.enzyme_affinity.keys()) == set(drug_explicit_empty.enzyme_affinity.keys())  # noqa: E501
    for tag in drug_default.enzyme_affinity:
        assert drug_default.enzyme_affinity[tag].mean == pytest.approx(
            drug_explicit_none.enzyme_affinity[tag].mean
        )


def test_build_drug_on_graph_isoniazid_with_non_cyp_fractions():
    """Isoniazid + non_cyp_fractions={'NAT2': 0.9} produces non-zero NAT2 affinity."""
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
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


# ── detect_disposition: shared OATP-ECM + non-CYP detection (review #10) ──
MIDAZOLAM = "C[n+]1cnc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"
CODEINE = "COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H](C2)N(C)CC[C@@]341"


def test_detect_disposition_returns_non_cyp_for_ugt_substrate():
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import detect_disposition

    oatp, ecm, non_cyp = detect_disposition(compute_profile(CODEINE))
    assert oatp is None and ecm is None  # codeine is not an OATP1B1-ECM substrate
    assert non_cyp and "UGT2B7" in non_cyp  # but it IS a UGT2B7 substrate


def test_detect_disposition_empty_for_pure_cyp_substrate():
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import detect_disposition

    oatp, ecm, non_cyp = detect_disposition(compute_profile(MIDAZOLAM))
    assert oatp is None and ecm is None and non_cyp == {}


# ── UGT single-path routing (double-allocation regression) ──


def test_ugt_tag_double_allocation_suppresses_cyp_at_decomposition():
    """A UGT tag present in BOTH ugt_enzymes and non_cyp_fractions is
    double-allocated: it lands in the CYP+UGT block AND is overwritten by the
    registry value, so the CYP residual (1 - fm) leaks into a phantom UGT slot and
    CYP is ~8x suppressed. This pins the decomposition behavior the builder must
    avoid by routing each UGT tag through exactly one mechanism."""
    honest = _get_fm_fractions(
        "base", substrate_enzymes=None, ugt_enzymes=None,
        non_cyp_fractions={"UGT2B7": 0.85},
    )
    double = _get_fm_fractions(
        "base", substrate_enzymes=None, ugt_enzymes={"UGT2B7"},
        non_cyp_fractions={"UGT2B7": 0.85},
    )
    cyp_honest = sum(v for k, v in honest.items() if k.startswith("CYP"))
    cyp_double = sum(v for k, v in double.items() if k.startswith("CYP"))
    assert honest["UGT2B7"] == pytest.approx(0.85, abs=0.01)
    assert cyp_honest == pytest.approx(0.15, abs=0.01)
    assert cyp_double < 0.03  # the ~8x-suppressed regime the builder must not hit


def test_build_drug_on_graph_routes_ugt_registry_tag_single_path():
    """Regression guard for the builder fix: when non_cyp_fractions carries a UGT
    tag (as the pipeline supplies via detect_disposition), build_drug_on_graph's
    internal ugt_enzymes derivation must EXCLUDE it, so the tag is not
    double-allocated. Observed via enzyme_affinity: with the honest 0.85/0.15 split
    the CYP affinity keeps a meaningful share; the double-allocation bug crushes
    CYP fm ~9x (0.15 -> 0.017), which would blow up the UGT:CYP affinity ratio."""
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    profile = compute_profile(CODEINE)  # UGT2B7 substrate
    adme = predict_adme(profile)
    drug = build_drug_on_graph(
        profile, adme, dose_mg=30.0, route="oral",
        non_cyp_fractions={"UGT2B7": 0.85},
    )
    ea = drug.enzyme_affinity
    assert "UGT2B7" in ea
    cyp = sum(v.mean for k, v in ea.items() if k.startswith("CYP"))
    ugt = sum(v.mean for k, v in ea.items() if k.startswith("UGT"))
    assert cyp > 0, "CYP affinity fully suppressed — UGT tag double-allocated"
    assert ugt / cyp < 12.0, (
        f"UGT:CYP affinity ratio {ugt / cyp:.1f} implies CYP was ~8x suppressed — "
        "the UGT registry tag is being double-allocated (also entering ugt_enzymes)."
    )
