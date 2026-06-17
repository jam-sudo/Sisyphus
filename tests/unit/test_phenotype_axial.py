"""Unit tests for apply_phenotype_to_graph under axial expansion (and its
non-axial bit-identity guard). Spec: 2026-06-16-phenotype-axial-node-fix-design.md
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from sisyphus.graph.axial import expand_axial
from sisyphus.graph.builder import build_from_yaml
from sisyphus.graph.types import ClearanceEdge
from sisyphus.predict.phenotype import apply_phenotype_to_graph

ROOT = Path(__file__).resolve().parent.parent.parent
_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"


def _ref_graph():
    return build_from_yaml(_PHYS)


def test_nonaxial_scaling_is_bit_identical():
    """On a normal (non-axial) graph, the liver CYP2D6 abundance is scaled by
    exactly the PM factor (0.10) — unchanged from pre-fix behaviour. Headline-safety guard."""
    g = _ref_graph()
    base = g.nodes["liver"].enzymes["CYP2D6"].mean
    out = apply_phenotype_to_graph(g, {"CYP2D6": "PM"})
    assert out.nodes["liver"].enzymes["CYP2D6"].mean == pytest.approx(base * 0.10, rel=0, abs=0)
    assert out.nodes["liver"].enzymes["CYP3A4"].mean == g.nodes["liver"].enzymes["CYP3A4"].mean
    assert out.nodes["liver"].enzymes["CYP2D6"].cv == g.nodes["liver"].enzymes["CYP2D6"].cv


def _axial_liver_graph(n_sub: int = 5):
    """Reference graph with the liver turned into a parallel_tube organ and
    axially expanded into ``n_sub`` serial well_stirred sub-tanks. After
    expand_axial there is NO node literally named 'liver'; the sub-tanks
    'liver__ax{i}' carry lookup_name='liver' and 1/n of the enzyme abundance."""
    g = build_from_yaml(_PHYS)
    g.nodes["liver"] = dataclasses.replace(g.nodes["liver"], axial_subcompartments=n_sub)
    g.edges = [
        dataclasses.replace(e, model="parallel_tube")
        if isinstance(e, ClearanceEdge) and e.source == "liver"
        else e
        for e in g.edges
    ]
    return expand_axial(g)


def _liver_subtanks(graph):
    return [n for n in graph.nodes.values() if (n.lookup_name or n.name) == "liver"]


def test_axial_expansion_produces_subtanks_no_literal_liver():
    g = _axial_liver_graph(5)
    assert "liver" not in g.nodes
    subs = _liver_subtanks(g)
    assert len(subs) == 5
    assert all("CYP2D6" in s.enzymes for s in subs)


def test_axial_scaling_applies_to_every_subtank():
    g = _axial_liver_graph(5)
    pre_total = sum(s.enzymes["CYP2D6"].mean for s in _liver_subtanks(g))
    out = apply_phenotype_to_graph(g, {"CYP2D6": "IM"})
    out_subs = _liver_subtanks(out)
    assert len(out_subs) == 5
    for s in out_subs:
        assert s.enzymes["CYP2D6"].mean == pytest.approx(pre_total / 5 * 0.50)
    assert sum(s.enzymes["CYP2D6"].mean for s in out_subs) == pytest.approx(pre_total * 0.50)


def test_axial_symptom_regression_abundance_changes():
    """The bug's observable: pre-fix this was a silent no-op (fold == 1.0)."""
    g = _axial_liver_graph(5)
    pre = {s.name: s.enzymes["CYP2D6"].mean for s in _liver_subtanks(g)}
    out = apply_phenotype_to_graph(g, {"CYP2D6": "PM"})
    post = {s.name: s.enzymes["CYP2D6"].mean for s in _liver_subtanks(out)}
    assert post.keys() == pre.keys()
    assert all(post[k] == pytest.approx(pre[k] * 0.10) for k in pre)
    assert all(post[k] != pytest.approx(pre[k]) for k in pre)


def test_axial_override_path_applies_per_subtank():
    g = _axial_liver_graph(4)
    pre_total = sum(s.enzymes["CYP2D6"].mean for s in _liver_subtanks(g))
    out = apply_phenotype_to_graph(
        g, {"CYP2D6": "PM"}, phenotype_scale_overrides={"CYP2D6": 0.33}
    )
    assert sum(s.enzymes["CYP2D6"].mean for s in _liver_subtanks(out)) == pytest.approx(
        pre_total * 0.33
    )


def test_truly_absent_node_warns_and_returns_unchanged(caplog):
    import logging

    g = _ref_graph()
    with caplog.at_level(logging.WARNING):
        out = apply_phenotype_to_graph(g, {"CYP2D6": "PM"}, node="nonexistent_organ")
    assert out is g
    assert any("nonexistent_organ" in r.getMessage() for r in caplog.records)
