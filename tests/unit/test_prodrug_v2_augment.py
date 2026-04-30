"""Unit tests for v2 augment_for_active_species multi-site discovery."""
from __future__ import annotations

import pytest

from sisyphus.core import ActiveMetabolite, Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import ACTIVE_SUFFIX, augment_for_active_species
from sisyphus.graph.types import (
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)
from tests.unit.test_prodrug_v2_drug import _minimal_active, _minimal_drug


def _base_graph_with_two_spr_sites() -> BodyGraph:
    g = BodyGraph()
    g.add_node(Node(
        name="liver", node_type="organ", volume=Distribution(1.5),
        enzymes={"SPR": Distribution(mean=1e6, cv=0.5),
                 "CYP3A4": Distribution(mean=9e6, cv=0.7)},
        ivive_scaling=6e-5,
    ))
    g.add_node(Node(
        name="gut_wall", node_type="barrier_organ", volume=Distribution(1.0),
        enzymes={"SPR": Distribution(mean=3e5, cv=0.7)},
        ivive_scaling=6e-5,
    ))
    g.add_node(Node(name="venous_blood", node_type="blood_pool",
                    volume=Distribution(5.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    return g


def test_no_op_when_no_active_metabolite():
    g = _base_graph_with_two_spr_sites()
    drug = _minimal_drug()  # active_metabolite=None
    n_nodes_before = len(g.nodes)
    n_edges_before = len(g.edges)

    result = augment_for_active_species(g, drug)
    assert result is g  # same instance
    assert len(g.nodes) == n_nodes_before
    assert len(g.edges) == n_edges_before


def test_creates_active_node_and_edges_per_site():
    g = _base_graph_with_two_spr_sites()
    drug = _minimal_drug(
        active_metabolite=_minimal_active(),
        observation_species="parent",
        enzyme_affinity_for_conversion={"SPR": Distribution(100.0)},
    )
    augment_for_active_species(g, drug)

    assert "venous_blood_active" in g.nodes

    activation_edges = [e for e in g.edges if isinstance(e, ProdrugActivationEdge)]
    assert len(activation_edges) == 2
    sources = sorted(e.source for e in activation_edges)
    assert sources == ["gut_wall", "liver"]
    for e in activation_edges:
        assert e.target == "venous_blood_active"
        assert e.enzyme_tags == frozenset({"SPR"})

    elim_edges = [e for e in g.edges if isinstance(e, OneCompartmentEliminationEdge)]
    assert len(elim_edges) == 1
    assert elim_edges[0].source == "venous_blood_active"
    assert elim_edges[0].target == "metabolized_gut"


def test_raises_when_no_site_in_physiology():
    g = _base_graph_with_two_spr_sites()
    drug = _minimal_drug(
        active_metabolite=_minimal_active(),
        observation_species="parent",
        enzyme_affinity_for_conversion={"NONEXISTENT_ENZYME": Distribution(100.0)},
    )
    with pytest.raises(ValueError, match="No conversion site"):
        augment_for_active_species(g, drug)


def test_raises_when_active_metab_present_but_affinity_empty():
    g = _base_graph_with_two_spr_sites()
    drug = _minimal_drug(
        active_metabolite=_minimal_active(),
        observation_species="parent",
        enzyme_affinity_for_conversion={},
    )
    with pytest.raises(ValueError, match="enzyme_affinity_for_conversion"):
        augment_for_active_species(g, drug)


def test_augment_called_twice_raises_on_collision():
    """Calling augment twice on same graph should raise (active node collision)."""
    g = _base_graph_with_two_spr_sites()
    drug = _minimal_drug(
        active_metabolite=_minimal_active(),
        observation_species="parent",
        enzyme_affinity_for_conversion={"SPR": Distribution(100.0)},
    )
    augment_for_active_species(g, drug)
    with pytest.raises(ValueError, match="collision"):
        augment_for_active_species(g, drug)
