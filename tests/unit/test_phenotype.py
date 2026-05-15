"""Unit tests for CYP phenotype scaling."""

from __future__ import annotations

from pathlib import Path

import pytest

from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.phenotype import (
    NAT2_ACETYLATOR_ALIASES,
    PHENOTYPE_SCALES,
    TRANSPORTER_ALIASES,
    apply_phenotype_to_graph,
    parse_phenotype_spec,
)

ROOT = Path(__file__).resolve().parent.parent.parent
_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"


# ---------------------------------------------------------------------------
# parse_phenotype_spec
# ---------------------------------------------------------------------------


def test_parse_single_enzyme():
    assert parse_phenotype_spec("CYP2D6:PM") == {"CYP2D6": "PM"}


def test_parse_multiple_enzymes():
    out = parse_phenotype_spec("CYP2D6:PM,CYP3A5:EM")
    assert out == {"CYP2D6": "PM", "CYP3A5": "EM"}


def test_parse_accepts_short_form():
    """User can drop the 'CYP' prefix."""
    assert parse_phenotype_spec("2D6:pm") == {"CYP2D6": "PM"}


def test_parse_case_insensitive():
    assert parse_phenotype_spec("cyp2d6:pm") == {"CYP2D6": "PM"}


def test_parse_empty():
    assert parse_phenotype_spec("") == {}
    assert parse_phenotype_spec(None or "") == {}


def test_parse_invalid_missing_colon():
    with pytest.raises(ValueError, match="expected"):
        parse_phenotype_spec("CYP2D6 PM")


def test_parse_invalid_phenotype():
    with pytest.raises(ValueError, match="Unknown phenotype"):
        parse_phenotype_spec("CYP2D6:BOGUS")


def test_all_phenotype_codes_valid():
    """Every code in PHENOTYPE_SCALES must parse."""
    for code in PHENOTYPE_SCALES:
        out = parse_phenotype_spec(f"CYP2D6:{code}")
        assert out == {"CYP2D6": code}


# ---------------------------------------------------------------------------
# CPIC NAT2 acetylator alias (B-05)
# ---------------------------------------------------------------------------


def test_nat2_slow_acetylator_aliases_to_pm():
    """CPIC NAT2:SA (slow acetylator) → PM (0.10×)."""
    assert parse_phenotype_spec("NAT2:SA") == {"NAT2": "PM"}


def test_nat2_intermediate_acetylator_aliases_to_im():
    assert parse_phenotype_spec("NAT2:IA") == {"NAT2": "IM"}


def test_nat2_rapid_acetylator_aliases_to_em():
    """NAT2:RA (rapid acetylator) → EM (1.00×), NOT RM (1.50×).

    NAT2 "Rapid" is the wild-type/normal phenotype, distinct from the
    CYP "Rapid Metabolizer" category. Alias is gene-conditional.
    """
    assert parse_phenotype_spec("NAT2:RA") == {"NAT2": "EM"}


def test_nat2_acetylator_alias_case_insensitive():
    assert parse_phenotype_spec("nat2:sa") == {"NAT2": "PM"}


def test_nat2_pm_pass_through_unchanged():
    """Existing NAT2:PM input is untouched (no double-aliasing)."""
    assert parse_phenotype_spec("NAT2:PM") == {"NAT2": "PM"}


def test_acetylator_alias_only_applies_to_nat2():
    """SA is not a global alias — CYP2D6:SA must still fail."""
    with pytest.raises(ValueError, match="Unknown phenotype"):
        parse_phenotype_spec("CYP2D6:SA")


def test_nat2_acetylator_alias_targets_match_phenotype_scales():
    """Every acetylator alias target must be a real phenotype code."""
    for ace, target in NAT2_ACETYLATOR_ALIASES.items():
        assert target in PHENOTYPE_SCALES, f"{ace} → {target} not in PHENOTYPE_SCALES"


def test_nat2_sa_scales_enzyme_like_pm():
    """End-to-end: NAT2:SA on graph equals NAT2:PM (0.10× abundance)."""
    g = build_from_yaml(_PHYS)
    sa_graph = apply_phenotype_to_graph(g, parse_phenotype_spec("NAT2:SA"))
    pm_graph = apply_phenotype_to_graph(g, parse_phenotype_spec("NAT2:PM"))
    assert sa_graph.nodes["liver"].enzymes["NAT2"].mean == pytest.approx(
        pm_graph.nodes["liver"].enzymes["NAT2"].mean
    )


# ---------------------------------------------------------------------------
# apply_phenotype_to_graph
# ---------------------------------------------------------------------------


def test_apply_scales_liver_enzyme():
    g = build_from_yaml(_PHYS)
    original = g.nodes["liver"].enzymes["CYP2D6"].mean
    g2 = apply_phenotype_to_graph(g, {"CYP2D6": "PM"})
    new = g2.nodes["liver"].enzymes["CYP2D6"].mean
    assert new == pytest.approx(original * 0.10, rel=1e-6)


def test_apply_preserves_cv():
    g = build_from_yaml(_PHYS)
    original_cv = g.nodes["liver"].enzymes["CYP2D6"].cv
    g2 = apply_phenotype_to_graph(g, {"CYP2D6": "PM"})
    assert g2.nodes["liver"].enzymes["CYP2D6"].cv == original_cv


def test_apply_leaves_other_enzymes_unchanged():
    g = build_from_yaml(_PHYS)
    cyp3a4_before = g.nodes["liver"].enzymes["CYP3A4"].mean
    g2 = apply_phenotype_to_graph(g, {"CYP2D6": "PM"})
    cyp3a4_after = g2.nodes["liver"].enzymes["CYP3A4"].mean
    assert cyp3a4_after == cyp3a4_before


def test_apply_multi_enzyme():
    g = build_from_yaml(_PHYS)
    d6_before = g.nodes["liver"].enzymes["CYP2D6"].mean
    c9_before = g.nodes["liver"].enzymes["CYP2C9"].mean
    g2 = apply_phenotype_to_graph(g, {"CYP2D6": "PM", "CYP2C9": "IM"})
    assert g2.nodes["liver"].enzymes["CYP2D6"].mean == pytest.approx(d6_before * 0.10)
    assert g2.nodes["liver"].enzymes["CYP2C9"].mean == pytest.approx(c9_before * 0.50)


def test_apply_empty_phenotype_is_noop():
    g = build_from_yaml(_PHYS)
    g2 = apply_phenotype_to_graph(g, {})
    assert g2 is g  # identity — no copy


def test_apply_unknown_enzyme_warns_but_continues(caplog):
    g = build_from_yaml(_PHYS)
    with caplog.at_level("WARNING"):
        g2 = apply_phenotype_to_graph(g, {"CYP9Z9": "PM"})
    assert any("not found" in r.message for r in caplog.records)
    # Known enzymes still present and unchanged
    assert g2.nodes["liver"].enzymes["CYP2D6"].mean == g.nodes["liver"].enzymes["CYP2D6"].mean


def test_apply_does_not_mutate_input_graph():
    g = build_from_yaml(_PHYS)
    original = g.nodes["liver"].enzymes["CYP2D6"].mean
    _ = apply_phenotype_to_graph(g, {"CYP2D6": "PM"})
    assert g.nodes["liver"].enzymes["CYP2D6"].mean == original


def test_um_scales_up():
    g = build_from_yaml(_PHYS)
    original = g.nodes["liver"].enzymes["CYP2D6"].mean
    g2 = apply_phenotype_to_graph(g, {"CYP2D6": "UM"})
    assert g2.nodes["liver"].enzymes["CYP2D6"].mean == pytest.approx(original * 2.0)


def test_nm_em_equivalent():
    """NM (CPIC) and EM (legacy) should give the same scale."""
    g = build_from_yaml(_PHYS)
    g_nm = apply_phenotype_to_graph(g, {"CYP2D6": "NM"})
    g_em = apply_phenotype_to_graph(g, {"CYP2D6": "EM"})
    assert g_nm.nodes["liver"].enzymes["CYP2D6"].mean == g_em.nodes["liver"].enzymes["CYP2D6"].mean


# ---------------------------------------------------------------------------
# Transporter phenotype (SLCO1B1 → OATP1B1)
# ---------------------------------------------------------------------------


def test_parse_slco1b1():
    assert parse_phenotype_spec("SLCO1B1:PM") == {"SLCO1B1": "PM"}


def test_parse_mixed_cyp_and_transporter():
    out = parse_phenotype_spec("CYP2D6:PM,SLCO1B1:IM")
    assert out == {"CYP2D6": "PM", "SLCO1B1": "IM"}


def test_transporter_alias_mapping():
    assert TRANSPORTER_ALIASES["SLCO1B1"] == "OATP1B1"


def test_apply_slco1b1_scales_oatp1b1():
    g = build_from_yaml(_PHYS)
    original = g.nodes["liver"].transporters["OATP1B1"].mean
    g2 = apply_phenotype_to_graph(g, {"SLCO1B1": "PM"})
    new = g2.nodes["liver"].transporters["OATP1B1"].mean
    assert new == pytest.approx(original * 0.10, rel=1e-6)


def test_apply_slco1b1_preserves_cv():
    g = build_from_yaml(_PHYS)
    original_cv = g.nodes["liver"].transporters["OATP1B1"].cv
    g2 = apply_phenotype_to_graph(g, {"SLCO1B1": "IM"})
    assert g2.nodes["liver"].transporters["OATP1B1"].cv == original_cv


def test_apply_slco1b1_does_not_touch_enzymes():
    g = build_from_yaml(_PHYS)
    d6_before = g.nodes["liver"].enzymes["CYP2D6"].mean
    g2 = apply_phenotype_to_graph(g, {"SLCO1B1": "PM"})
    assert g2.nodes["liver"].enzymes["CYP2D6"].mean == d6_before


def test_apply_cyp_does_not_touch_transporters():
    g = build_from_yaml(_PHYS)
    oatp_before = g.nodes["liver"].transporters["OATP1B1"].mean
    g2 = apply_phenotype_to_graph(g, {"CYP2D6": "PM"})
    assert g2.nodes["liver"].transporters["OATP1B1"].mean == oatp_before


def test_apply_mixed_phenotypes():
    g = build_from_yaml(_PHYS)
    d6_before = g.nodes["liver"].enzymes["CYP2D6"].mean
    oatp_before = g.nodes["liver"].transporters["OATP1B1"].mean
    g2 = apply_phenotype_to_graph(g, {"CYP2D6": "PM", "SLCO1B1": "IM"})
    assert g2.nodes["liver"].enzymes["CYP2D6"].mean == pytest.approx(d6_before * 0.10)
    assert g2.nodes["liver"].transporters["OATP1B1"].mean == pytest.approx(oatp_before * 0.50)


def test_apply_unknown_transporter_warns(caplog):
    g = build_from_yaml(_PHYS)
    with caplog.at_level("WARNING"):
        # ABCB1 not aliased and not in enzyme dict — should be treated as
        # an unknown enzyme tag and warn.
        _ = apply_phenotype_to_graph(g, {"ABCB1": "PM"})
    assert any("not found" in r.message for r in caplog.records)


def test_apply_slco1b1_does_not_mutate_input_graph():
    g = build_from_yaml(_PHYS)
    original = g.nodes["liver"].transporters["OATP1B1"].mean
    _ = apply_phenotype_to_graph(g, {"SLCO1B1": "PM"})
    assert g.nodes["liver"].transporters["OATP1B1"].mean == original


def test_slco1b1_um_scales_up():
    g = build_from_yaml(_PHYS)
    original = g.nodes["liver"].transporters["OATP1B1"].mean
    g2 = apply_phenotype_to_graph(g, {"SLCO1B1": "UM"})
    assert g2.nodes["liver"].transporters["OATP1B1"].mean == pytest.approx(original * 2.0)
