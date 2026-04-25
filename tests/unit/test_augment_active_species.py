"""Tests for graph.builder.augment_for_active_species."""
from __future__ import annotations

import dataclasses

import pytest

from sisyphus.core import ActiveMetabolite, Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import ACTIVE_SUFFIX, augment_for_active_species
from sisyphus.graph.types import (
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)


def _bare_parent_graph():
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood", node_type="blood_pool",
                    volume=Distribution(5.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    return g


def _drug_with(active_metabolite=None, obs_species="parent"):
    """Use the shared _minimal_drug helper from test_active_metabolite."""
    from tests.unit.test_active_metabolite import _minimal_drug
    return _minimal_drug(active=active_metabolite, obs_species=obs_species)


def _bh4_active():
    return ActiveMetabolite(
        name="BH4", mw=241.25,
        fup=Distribution(0.23), CL_per_h=Distribution(40.0),
        Vd_L=Distribution(150.0), conversion_rate_per_h=Distribution(12.0),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(0.85),
    )


def test_augment_no_active_returns_unchanged():
    """active_metabolite=None → graph returned unchanged."""
    g = _bare_parent_graph()
    drug = _drug_with(active_metabolite=None)
    g2 = augment_for_active_species(g, drug, observation_node="venous_blood")
    assert len(g2.nodes) == 3
    assert len(g2.edges) == 0


def test_augment_with_active_adds_node_and_two_edges():
    g = _bare_parent_graph()
    drug = _drug_with(active_metabolite=_bh4_active(), obs_species="active")
    g2 = augment_for_active_species(g, drug, observation_node="venous_blood")
    expected_active_node = "venous_blood" + ACTIVE_SUFFIX
    assert expected_active_node in g2.nodes
    assert len(g2.edges) == 2
    edge_types = sorted(e.edge_type for e in g2.edges)
    assert edge_types == ["one_compartment_elimination", "prodrug_activation"]


def test_augment_invalid_conversion_site_raises():
    g = _bare_parent_graph()
    am = ActiveMetabolite(
        name="X", mw=200.0,
        fup=Distribution(0.5), CL_per_h=Distribution(10.0),
        Vd_L=Distribution(50.0), conversion_rate_per_h=Distribution(5.0),
        conversion_site="nonexistent_node",
        conversion_yield_fraction=Distribution(1.0),
    )
    drug = _drug_with(active_metabolite=am, obs_species="active")
    with pytest.raises(ValueError, match="conversion_site"):
        augment_for_active_species(g, drug, observation_node="venous_blood")


def test_augment_collision_raises():
    """If '<obs>_active' already exists, raise."""
    g = _bare_parent_graph()
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool",
                    volume=Distribution(0.0)))
    drug = _drug_with(active_metabolite=_bh4_active(), obs_species="active")
    with pytest.raises(ValueError, match="active node name collision"):
        augment_for_active_species(g, drug, observation_node="venous_blood")


def test_augment_calls_twice_raises():
    """Second call raises (no idempotency — node already exists)."""
    g = _bare_parent_graph()
    drug = _drug_with(active_metabolite=_bh4_active(), obs_species="active")
    augment_for_active_species(g, drug, observation_node="venous_blood")
    with pytest.raises(ValueError, match="active node name collision"):
        augment_for_active_species(g, drug, observation_node="venous_blood")


def test_augment_uses_existing_sink_for_elimination():
    """OneCompartmentEliminationEdge target must be an existing sink node."""
    g = _bare_parent_graph()
    drug = _drug_with(active_metabolite=_bh4_active(), obs_species="active")
    g2 = augment_for_active_species(g, drug, observation_node="venous_blood")
    elim_edges = [e for e in g2.edges if isinstance(e, OneCompartmentEliminationEdge)]
    assert len(elim_edges) == 1
    assert elim_edges[0].target == "metabolized_gut"  # existing sink


def test_augment_no_existing_sink_raises():
    """If the chosen sink node doesn't exist, raise."""
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood", node_type="blood_pool",
                    volume=Distribution(5.0)))
    drug = _drug_with(active_metabolite=_bh4_active(), obs_species="active")
    with pytest.raises(ValueError, match="sink node"):
        augment_for_active_species(g, drug, observation_node="venous_blood")
