"""Unit tests for v2 ProdrugActivationEdge struct."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.graph.types import ProdrugActivationEdge


def test_edge_has_enzyme_tags_field():
    """v2 edge replaces conversion_rate with enzyme_tags frozenset."""
    edge = ProdrugActivationEdge(
        source="liver",
        target="venous_blood_active",
        enzyme_tags=frozenset({"SPR"}),
        conversion_yield=Distribution(mean=0.85, cv=0.1),
        mw_parent=237.0,
        mw_active=241.25,
    )
    assert edge.enzyme_tags == frozenset({"SPR"})
    assert edge.mw_parent == 237.0
    assert edge.mw_active == 241.25
    assert edge.conversion_yield.mean == 0.85


def test_edge_no_conversion_rate_field():
    """v1 conversion_rate field removed in v2."""
    edge = ProdrugActivationEdge(
        source="liver",
        target="venous_blood_active",
        enzyme_tags=frozenset({"SPR"}),
        mw_parent=237.0,
        mw_active=241.25,
    )
    assert not hasattr(edge, "conversion_rate")


def test_edge_default_enzyme_tags_empty():
    """Default enzyme_tags is empty frozenset (mirrors v1 default-zero pattern)."""
    edge = ProdrugActivationEdge(
        source="x",
        target="y",
        mw_parent=100.0,
        mw_active=100.0,
    )
    assert edge.enzyme_tags == frozenset()


def test_edge_is_frozen():
    """Edge dataclass remains frozen."""
    edge = ProdrugActivationEdge(
        source="x",
        target="y",
        enzyme_tags=frozenset({"X"}),
        mw_parent=100.0,
        mw_active=100.0,
    )
    with pytest.raises(AttributeError):
        edge.enzyme_tags = frozenset({"Y"})
