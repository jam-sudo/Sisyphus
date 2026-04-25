"""Tests for ProdrugActivationEdge and OneCompartmentEliminationEdge."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.graph.types import (
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)


def test_prodrug_activation_edge_construction():
    """Construct ProdrugActivationEdge with required fields."""
    edge = ProdrugActivationEdge(
        source="gut_wall",
        target="venous_blood_active",
        conversion_rate=Distribution(mean=12.0, cv=0.4),
        conversion_yield=Distribution(mean=0.85, cv=0.1),
        mw_parent=237.26,
        mw_active=241.25,
    )
    assert edge.edge_type == "prodrug_activation"
    assert edge.source == "gut_wall"
    assert edge.target == "venous_blood_active"
    assert edge.mw_parent == 237.26
    assert edge.mw_active == 241.25
    assert edge.conversion_rate.mean == 12.0
    assert edge.conversion_yield.mean == 0.85


def test_one_compartment_elimination_edge_construction():
    """Construct OneCompartmentEliminationEdge with required fields."""
    edge = OneCompartmentEliminationEdge(
        source="venous_blood_active",
        target="metabolized_gut",
        cl_per_h=Distribution(mean=40.0, cv=0.35),
        vd_l=Distribution(mean=150.0, cv=0.3),
    )
    assert edge.edge_type == "one_compartment_elimination"
    assert edge.cl_per_h.mean == 40.0
    assert edge.vd_l.mean == 150.0


def test_edges_are_frozen():
    """Both edge types are frozen dataclasses."""
    edge = ProdrugActivationEdge(
        source="a", target="b",
        conversion_rate=Distribution(1.0), conversion_yield=Distribution(1.0),
        mw_parent=100.0, mw_active=100.0,
    )
    with pytest.raises(AttributeError):
        edge.source = "c"  # type: ignore[misc]


def test_edge_type_is_init_false():
    """edge_type must be auto-set from class default (init=False)."""
    # ProdrugActivationEdge: cannot pass edge_type as kwarg
    with pytest.raises(TypeError):
        ProdrugActivationEdge(
            source="a", target="b", edge_type="wrong",  # type: ignore[call-arg]
            conversion_rate=Distribution(1.0), conversion_yield=Distribution(1.0),
            mw_parent=100.0, mw_active=100.0,
        )
