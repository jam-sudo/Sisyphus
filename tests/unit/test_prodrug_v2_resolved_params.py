"""Unit tests for ResolvedParams v2 prodrug accessors."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    Node,
    ProdrugActivationEdge,
)
from tests.unit.test_prodrug_v2_drug import _minimal_active, _minimal_drug


def _minimal_graph() -> BodyGraph:
    g = BodyGraph()
    g.add_node(Node(name="liver", node_type="organ", volume=Distribution(1.5)))
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool", volume=Distribution(20.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink", volume=Distribution(0.0)))
    return g


def test_drug_enzyme_affinity_for_conversion_returns_mean():
    drug = _minimal_drug(
        enzyme_affinity_for_conversion={"SPR": Distribution(mean=42.0, cv=0.0)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    g = _minimal_graph()
    params = ResolvedParams(g, drug)
    assert params.drug_enzyme_affinity_for_conversion("SPR") == 42.0


def test_drug_enzyme_affinity_for_conversion_returns_zero_for_missing_tag():
    drug = _minimal_drug()
    g = _minimal_graph()
    params = ResolvedParams(g, drug)
    assert params.drug_enzyme_affinity_for_conversion("DOES_NOT_EXIST") == 0.0


def test_edge_param_for_prodrug_activation_no_conversion_rate():
    """v2 ProdrugActivationEdge no longer has conversion_rate; only conversion_yield."""
    g = _minimal_graph()
    g.add_edge(ProdrugActivationEdge(
        source="liver", target="venous_blood_active",
        enzyme_tags=frozenset({"SPR"}),
        conversion_yield=Distribution(mean=0.85, cv=0.0),
        mw_parent=237.0, mw_active=241.25,
    ))
    drug = _minimal_drug(
        enzyme_affinity_for_conversion={"SPR": Distribution(100.0)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    params = ResolvedParams(g, drug)
    assert params.edge_param(0, "conversion_yield") == 0.85
    with pytest.raises(KeyError):
        params.edge_param(0, "conversion_rate")
