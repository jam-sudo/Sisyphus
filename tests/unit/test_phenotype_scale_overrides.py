"""Unit tests for phenotype_scale_overrides kwarg on apply_phenotype_to_graph (issue #31)."""
from __future__ import annotations

import logging
import pathlib

import pytest

from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.phenotype import apply_phenotype_to_graph


def _fresh_graph():
    return build_from_yaml(pathlib.Path("data/physiology/reference_man.yaml"))


def test_overrides_none_preserves_existing():
    """phenotype_scale_overrides=None must produce identical output to no kwarg."""
    g = _fresh_graph()
    a = apply_phenotype_to_graph(g, {"CYP1A2": "PM"})
    b = apply_phenotype_to_graph(g, {"CYP1A2": "PM"}, phenotype_scale_overrides=None)
    assert a.nodes["liver"].enzymes["CYP1A2"].mean == pytest.approx(
        b.nodes["liver"].enzymes["CYP1A2"].mean
    )


def test_overrides_empty_preserves_existing():
    g = _fresh_graph()
    a = apply_phenotype_to_graph(g, {"CYP1A2": "PM"})
    b = apply_phenotype_to_graph(g, {"CYP1A2": "PM"}, phenotype_scale_overrides={})
    assert a.nodes["liver"].enzymes["CYP1A2"].mean == pytest.approx(
        b.nodes["liver"].enzymes["CYP1A2"].mean
    )


def test_override_replaces_default_scale_enzyme():
    """SLCO1B1:PM with override 0.30 scales OATP1B1 abundance to 30% (vs default 10%)."""
    g = _fresh_graph()
    original = g.nodes["liver"].transporters["OATP1B1"].mean

    default_pm = apply_phenotype_to_graph(g, {"SLCO1B1": "PM"})
    overridden = apply_phenotype_to_graph(
        g, {"SLCO1B1": "PM"},
        phenotype_scale_overrides={"SLCO1B1": 0.30},
    )

    assert default_pm.nodes["liver"].transporters["OATP1B1"].mean == pytest.approx(
        original * 0.10
    )
    assert overridden.nodes["liver"].transporters["OATP1B1"].mean == pytest.approx(
        original * 0.30
    )


def test_override_replaces_default_scale_cyp():
    """CYP1A2:PM with override 0.50 scales abundance to 50% (vs default 10%)."""
    g = _fresh_graph()
    original = g.nodes["liver"].enzymes["CYP1A2"].mean

    overridden = apply_phenotype_to_graph(
        g, {"CYP1A2": "PM"},
        phenotype_scale_overrides={"CYP1A2": 0.50},
    )
    assert overridden.nodes["liver"].enzymes["CYP1A2"].mean == pytest.approx(
        original * 0.50
    )


def test_negative_override_raises():
    g = _fresh_graph()
    with pytest.raises(ValueError, match="negative"):
        apply_phenotype_to_graph(
            g, {"CYP1A2": "PM"},
            phenotype_scale_overrides={"CYP1A2": -0.1},
        )


def test_override_for_gene_not_in_phenotypes_logs_info(caplog):
    """Override key for a gene not in the phenotypes dict is silently ignored."""
    g = _fresh_graph()
    caplog.set_level(logging.INFO)
    out = apply_phenotype_to_graph(
        g, {"CYP1A2": "PM"},
        phenotype_scale_overrides={"CYP2C9": 0.20},
    )
    # CYP2C9 abundance unchanged (override not applied because CYP2C9 not in phenotypes)
    assert out.nodes["liver"].enzymes["CYP2C9"].mean == pytest.approx(
        g.nodes["liver"].enzymes["CYP2C9"].mean
    )
    # logger.info note about ignored override
    info_records = [r for r in caplog.records if r.levelno == logging.INFO and "ignored" in r.getMessage()]  # noqa: E501
    assert info_records, "expected logger.info about ignored override key"


def test_multiple_genes_overridden_independently():
    g = _fresh_graph()
    cyp_orig = g.nodes["liver"].enzymes["CYP1A2"].mean
    nat2_orig = g.nodes["liver"].enzymes["NAT2"].mean

    out = apply_phenotype_to_graph(
        g,
        {"CYP1A2": "PM", "NAT2": "IM"},
        phenotype_scale_overrides={"CYP1A2": 0.40, "NAT2": 0.65},
    )
    assert out.nodes["liver"].enzymes["CYP1A2"].mean == pytest.approx(cyp_orig * 0.40)
    assert out.nodes["liver"].enzymes["NAT2"].mean == pytest.approx(nat2_orig * 0.65)
