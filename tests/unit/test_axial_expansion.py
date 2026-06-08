"""WS-3: axial sub-compartment expansion."""
from __future__ import annotations

from sisyphus.core import Distribution
from sisyphus.graph.types import Node


def test_node_has_axial_field():
    n = Node(name="liver", node_type="organ", volume=Distribution(1.0), axial_subcompartments=5)
    assert n.axial_subcompartments == 5


def test_node_axial_defaults_to_one():
    n = Node(name="liver", node_type="organ", volume=Distribution(1.0))
    assert n.axial_subcompartments == 1
