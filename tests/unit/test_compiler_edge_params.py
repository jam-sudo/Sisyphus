"""Tests for ResolvedParams._build_edge_params extension for new edge types.

v2 (2026-04-27): ProdrugActivationEdge no longer has conversion_rate;
edge_params now caches conversion_yield only (CLint computed dynamically
from drug.enzyme_affinity_for_conversion × node enzyme abundance).
"""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)


def _make_graph_and_drug():
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool",
                    volume=Distribution(150.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    g.add_edge(ProdrugActivationEdge(
        source="gut_wall", target="venous_blood_active",
        enzyme_tags=frozenset({"SPR"}),
        conversion_yield=Distribution(mean=0.85),
        mw_parent=237.26, mw_active=241.25,
    ))
    g.add_edge(OneCompartmentEliminationEdge(
        source="venous_blood_active", target="metabolized_gut",
        cl_per_h=Distribution(mean=40.0), vd_l=Distribution(mean=150.0),
    ))
    # Use the shared _minimal_drug helper from test_active_metabolite (defined there)
    from tests.unit.test_active_metabolite import _minimal_drug
    return g, _minimal_drug()


def test_resolved_params_caches_prodrug_activation_edge():
    """v2 ProdrugActivationEdge → only conversion_yield cached in edge_params.

    conversion_rate removed in v2; CLint is computed dynamically inside
    ProdrugActivationFluxSpec from drug.enzyme_affinity_for_conversion[tag]
    × node.enzymes[tag] × ivive_scaling. See test_prodrug_v2_resolved_params.
    """
    g, drug = _make_graph_and_drug()
    rp = ResolvedParams(g, drug)
    # Edge 0 is ProdrugActivationEdge
    assert rp.edge_param(0, "conversion_yield") == 0.85
    with pytest.raises(KeyError):
        rp.edge_param(0, "conversion_rate")


def test_resolved_params_caches_one_compartment_elim_edge():
    """OneCompartmentEliminationEdge → cl_per_h, vd_l in edge_params."""
    g, drug = _make_graph_and_drug()
    rp = ResolvedParams(g, drug)
    # Edge 1 is OneCompartmentEliminationEdge
    assert rp.edge_param(1, "cl_per_h") == 40.0
    assert rp.edge_param(1, "vd_l") == 150.0
