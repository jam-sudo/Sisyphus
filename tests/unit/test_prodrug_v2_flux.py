"""Unit tests for v2 ProdrugActivationFluxSpec well-stirred math."""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.core import ActiveMetabolite, Distribution
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.engine.flux import ProdrugActivationFluxSpec
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    FlowEdge,
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)
from tests.unit.test_prodrug_v2_drug import _minimal_active, _minimal_drug


def _build_state_index(graph: BodyGraph) -> dict[str, int]:
    """Build state_index for testing — node order is dict insertion order."""
    return {name: i for i, name in enumerate(graph.nodes.keys())}


def _build_flow_graph(
    abundance: float = 1e6,
    affinity: float = 10.0,
    ivive: float = 6e-5,
    fup: float = 0.5,
    q: float = 60.0,
    v_source: float = 10.0,
):
    """Build a 4-node flow-through graph for well-stirred testing.

    Topology:
        infusion_source --[FlowEdge Q]--> conversion_node --[FlowEdge Q]--> exit_sink
                                            └--[ProdrugActivationEdge]--> active_pool
                                                                            └--[1C elim]--> elim_sink
    """
    g = BodyGraph()
    g.add_node(Node(name="infusion_source", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_node(Node(
        name="conversion_node", node_type="organ", volume=Distribution(v_source),
        enzymes={"X": Distribution(mean=abundance, cv=0.0)},
        ivive_scaling=ivive,
    ))
    g.add_node(Node(name="active_pool", node_type="blood_pool", volume=Distribution(10.0)))
    g.add_node(Node(name="exit_sink", node_type="sink", volume=Distribution(0.0)))
    g.add_node(Node(name="elim_sink", node_type="sink", volume=Distribution(0.0)))

    g.add_edge(FlowEdge(source="infusion_source", target="conversion_node",
                       flow_rate=Distribution(q)))
    g.add_edge(FlowEdge(source="conversion_node", target="exit_sink",
                       flow_rate=Distribution(q)))
    g.add_edge(ProdrugActivationEdge(
        source="conversion_node", target="active_pool",
        enzyme_tags=frozenset({"X"}),
        conversion_yield=Distribution(mean=1.0, cv=0.0),
        mw_parent=200.0, mw_active=200.0,
    ))
    g.add_edge(OneCompartmentEliminationEdge(
        source="active_pool", target="elim_sink",
        cl_per_h=Distribution(10.0), vd_l=Distribution(10.0),
    ))

    drug = _minimal_drug(
        fup=Distribution(fup),
        enzyme_affinity_for_conversion={"X": Distribution(mean=affinity, cv=0.0)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    return g, drug


def test_flux_well_stirred_rate_matches_formula():
    """Rate should equal (Q × fup × CLint) / (Q + fup × CLint) × C_out × yield × mw_ratio."""
    abundance, affinity, ivive = 1e6, 10.0, 6e-5
    fup, q, v_source = 0.5, 60.0, 10.0
    g, drug = _build_flow_graph(
        abundance=abundance, affinity=affinity, ivive=ivive,
        fup=fup, q=q, v_source=v_source,
    )
    params = ResolvedParams(g, drug)

    edge_id = next(i for i, e in enumerate(g.edges) if isinstance(e, ProdrugActivationEdge))
    flux_spec = ProdrugActivationFluxSpec.from_edge(edge_id, g.edges[edge_id], _build_state_index(g))

    state_idx = _build_state_index(g)
    y = np.zeros(len(state_idx))
    a_parent = 100.0
    y[state_idx["conversion_node"]] = a_parent

    dydt = np.zeros_like(y)
    flux_spec.apply(t=0.0, y=y, dydt=dydt, params=params)

    clint = abundance * affinity * ivive
    cl_organ = (q * fup * clint) / (q + fup * clint)
    c_out = a_parent / v_source
    expected_rate_parent = cl_organ * c_out
    expected_rate_active = expected_rate_parent * 1.0 * 1.0  # mw_ratio=1, yield=1

    assert dydt[state_idx["conversion_node"]] == pytest.approx(-expected_rate_parent, rel=1e-6)
    assert dydt[state_idx["active_pool"]] == pytest.approx(expected_rate_active, rel=1e-6)


def test_flux_zero_when_clint_zero():
    """Affinity=0 should yield zero flux (no extraction)."""
    g, drug = _build_flow_graph(affinity=0.0)
    params = ResolvedParams(g, drug)

    edge_id = next(i for i, e in enumerate(g.edges) if isinstance(e, ProdrugActivationEdge))
    flux_spec = ProdrugActivationFluxSpec.from_edge(edge_id, g.edges[edge_id], _build_state_index(g))

    state_idx = _build_state_index(g)
    y = np.zeros(len(state_idx))
    y[state_idx["conversion_node"]] = 100.0
    dydt = np.zeros_like(y)
    flux_spec.apply(t=0.0, y=y, dydt=dydt, params=params)

    assert dydt[state_idx["conversion_node"]] == 0.0
    assert dydt[state_idx["active_pool"]] == 0.0


def test_flux_mw_ratio_scales_active_mass():
    """Active mass = parent loss × (mw_active/mw_parent) × yield."""
    g = BodyGraph()
    g.add_node(Node(name="src", node_type="organ", volume=Distribution(10.0),
                   enzymes={"X": Distribution(1e6)}, ivive_scaling=6e-5))
    g.add_node(Node(name="active", node_type="blood_pool", volume=Distribution(10.0)))
    g.add_node(Node(name="src_in", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source="src_in", target="src", flow_rate=Distribution(60.0)))
    g.add_edge(ProdrugActivationEdge(
        source="src", target="active",
        enzyme_tags=frozenset({"X"}),
        conversion_yield=Distribution(0.5),
        mw_parent=200.0, mw_active=400.0,  # mw_ratio = 2
    ))
    drug = _minimal_drug(
        fup=Distribution(0.5),
        enzyme_affinity_for_conversion={"X": Distribution(10.0)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    params = ResolvedParams(g, drug)

    edge_id = next(i for i, e in enumerate(g.edges) if isinstance(e, ProdrugActivationEdge))
    flux_spec = ProdrugActivationFluxSpec.from_edge(edge_id, g.edges[edge_id], _build_state_index(g))

    state_idx = _build_state_index(g)
    y = np.zeros(len(state_idx))
    y[state_idx["src"]] = 100.0
    dydt = np.zeros_like(y)
    flux_spec.apply(t=0.0, y=y, dydt=dydt, params=params)

    parent_loss = -dydt[state_idx["src"]]
    active_gain = dydt[state_idx["active"]]
    assert active_gain == pytest.approx(parent_loss * 2.0 * 0.5, rel=1e-6)
