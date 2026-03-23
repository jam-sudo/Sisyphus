import numpy as np
import pytest

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import FlowEdge, Node


class TestBodyGraph:
    def test_add_node(self):
        g = BodyGraph()
        n = Node(name="a", node_type="organ", volume=Distribution(1.0))
        g.add_node(n)
        assert "a" in g.nodes
        assert g.nodes["a"] is n

    def test_add_duplicate_node_raises(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
        with pytest.raises(ValueError, match="duplicate"):
            g.add_node(Node(name="a", node_type="organ", volume=Distribution(2.0)))

    def test_add_edge(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
        g.add_node(Node(name="b", node_type="organ", volume=Distribution(1.0)))
        e = FlowEdge(source="a", target="b", flow_rate=Distribution(10.0))
        g.add_edge(e)
        assert len(g.edges) == 1

    def test_add_edge_missing_node_raises(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
        with pytest.raises(ValueError, match="not found"):
            g.add_edge(FlowEdge(source="a", target="z", flow_rate=Distribution(10.0)))

    def test_remove_node(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
        g.add_node(Node(name="b", node_type="organ", volume=Distribution(1.0)))
        g.add_edge(FlowEdge(source="a", target="b", flow_rate=Distribution(10.0)))
        g.remove_node("a")
        assert "a" not in g.nodes
        assert len(g.edges) == 0  # edge removed too

    def test_validate_valid_graph(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
        g.add_node(Node(name="b", node_type="organ", volume=Distribution(1.0)))
        g.add_edge(FlowEdge(source="a", target="b", flow_rate=Distribution(10.0)))
        g.add_edge(FlowEdge(source="b", target="a", flow_rate=Distribution(10.0)))
        errors = g.validate()
        assert errors == []

    def test_validate_flow_imbalance(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
        g.add_node(Node(name="b", node_type="organ", volume=Distribution(1.0)))
        g.add_edge(FlowEdge(source="a", target="b", flow_rate=Distribution(10.0)))
        g.add_edge(FlowEdge(source="b", target="a", flow_rate=Distribution(5.0)))
        errors = g.validate()
        assert len(errors) > 0  # flow imbalance detected

    def test_sample(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0, cv=0.1)))
        g.add_node(Node(name="b", node_type="organ", volume=Distribution(2.0, cv=0.1)))
        g.add_edge(FlowEdge(source="a", target="b", flow_rate=Distribution(10.0, cv=0.1)))
        g.add_edge(FlowEdge(source="b", target="a", flow_rate=Distribution(10.0, cv=0.1)))
        g.global_params["co"] = Distribution(390.0, cv=0.1)
        rng = np.random.default_rng(42)
        g2 = g.sample(rng)
        assert g2.nodes["a"].volume.cv == 0.0  # sampled = deterministic
        assert g2.global_params["co"].cv == 0.0
        # Sampled values should differ from mean (with high probability for cv=0.1)
        assert g2.nodes["a"].volume.mean != 1.0 or g2.nodes["b"].volume.mean != 2.0
