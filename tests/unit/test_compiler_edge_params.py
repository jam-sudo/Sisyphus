"""Tests for ResolvedParams._build_edge_params extension for new edge types."""
from __future__ import annotations

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
        conversion_rate=Distribution(mean=12.0),
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
    """ProdrugActivationEdge → conversion_rate, conversion_yield in edge_params."""
    g, drug = _make_graph_and_drug()
    rp = ResolvedParams(g, drug)
    # Edge 0 is ProdrugActivationEdge
    assert rp.edge_param(0, "conversion_rate") == 12.0
    assert rp.edge_param(0, "conversion_yield") == 0.85


def test_resolved_params_caches_one_compartment_elim_edge():
    """OneCompartmentEliminationEdge → cl_per_h, vd_l in edge_params."""
    g, drug = _make_graph_and_drug()
    rp = ResolvedParams(g, drug)
    # Edge 1 is OneCompartmentEliminationEdge
    assert rp.edge_param(1, "cl_per_h") == 40.0
    assert rp.edge_param(1, "vd_l") == 150.0
