"""Tests for BodyGraph.sample() handling of new edge types.

Updated 2026-04-28 (T18): v2 ProdrugActivationEdge has enzyme_tags +
conversion_yield (no conversion_rate). sample() must round-trip
Distribution fields to cv=0 and preserve enzyme_tags / mw constants.
"""
from __future__ import annotations

import numpy as np

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)


def _minimal_graph_with_active():
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool",
                    volume=Distribution(150.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    g.add_edge(ProdrugActivationEdge(
        source="gut_wall", target="venous_blood_active",
        enzyme_tags=frozenset({"SPR"}),
        conversion_yield=Distribution(mean=0.85, cv=0.1),
        mw_parent=237.26, mw_active=241.25,
    ))
    g.add_edge(OneCompartmentEliminationEdge(
        source="venous_blood_active", target="metabolized_gut",
        cl_per_h=Distribution(mean=40.0, cv=0.35),
        vd_l=Distribution(mean=150.0, cv=0.3),
    ))
    return g


def test_sample_resamples_prodrug_activation_distributions():
    """ProdrugActivationEdge Distribution fields collapse to cv=0 after sample().

    v2: only conversion_yield is a Distribution; enzyme_tags / mw scalars
    are preserved unchanged.
    """
    g = _minimal_graph_with_active()
    rng = np.random.default_rng(42)
    g2 = g.sample(rng)
    prodrug_edges = [e for e in g2.edges if isinstance(e, ProdrugActivationEdge)]
    assert len(prodrug_edges) == 1
    edge = prodrug_edges[0]
    assert edge.conversion_yield.cv == 0.0
    # enzyme_tags must be preserved exactly
    assert edge.enzyme_tags == frozenset({"SPR"})
    # MW must be preserved exactly (deterministic float)
    assert edge.mw_parent == 237.26
    assert edge.mw_active == 241.25
    # Source/target preserved
    assert edge.source == "gut_wall"
    assert edge.target == "venous_blood_active"


def test_sample_resamples_one_compartment_elim_distributions():
    """OneCompartmentEliminationEdge Distribution fields collapse to cv=0."""
    g = _minimal_graph_with_active()
    rng = np.random.default_rng(42)
    g2 = g.sample(rng)
    elim_edges = [e for e in g2.edges if isinstance(e, OneCompartmentEliminationEdge)]
    assert len(elim_edges) == 1
    edge = elim_edges[0]
    assert edge.cl_per_h.cv == 0.0
    assert edge.vd_l.cv == 0.0


def test_sample_preserves_edge_count_and_types():
    """Sampled graph has same edge count and types."""
    g = _minimal_graph_with_active()
    rng = np.random.default_rng(42)
    g2 = g.sample(rng)
    assert len(g2.edges) == 2
    edge_types = sorted(e.edge_type for e in g2.edges)
    assert edge_types == ["one_compartment_elimination", "prodrug_activation"]


def test_sample_with_cv_zero_inputs_preserves_means():
    """Edge fields with cv=0 should sample to exactly the input mean."""
    g = BodyGraph()
    g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
    g.add_node(Node(name="b", node_type="blood_pool", volume=Distribution(50.0)))
    g.add_node(Node(name="c", node_type="sink", volume=Distribution(0.0)))
    g.add_edge(ProdrugActivationEdge(
        source="a", target="b",
        enzyme_tags=frozenset({"X"}),
        conversion_yield=Distribution(mean=0.7, cv=0.0),
        mw_parent=200.0, mw_active=210.0,
    ))
    g.add_edge(OneCompartmentEliminationEdge(
        source="b", target="c",
        cl_per_h=Distribution(mean=20.0, cv=0.0),
        vd_l=Distribution(mean=80.0, cv=0.0),
    ))
    rng = np.random.default_rng(0)
    g2 = g.sample(rng)
    pa = [e for e in g2.edges if isinstance(e, ProdrugActivationEdge)][0]
    assert pa.conversion_yield.mean == 0.7
    assert pa.enzyme_tags == frozenset({"X"})
    elim = [e for e in g2.edges if isinstance(e, OneCompartmentEliminationEdge)][0]
    assert elim.cl_per_h.mean == 20.0
    assert elim.vd_l.mean == 80.0
