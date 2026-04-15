"""Unit tests for CYP phenotype scaling."""

from __future__ import annotations

from pathlib import Path

import pytest

from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.phenotype import (
    PHENOTYPE_SCALES,
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
