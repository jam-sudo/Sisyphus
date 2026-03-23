"""Unit tests for engine/uncertainty.py -- Monte Carlo propagation."""

import numpy as np

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler
from sisyphus.engine.uncertainty import UncertaintyEngine, UncertaintyResult
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node


def _make_simple_graph() -> BodyGraph:
    """Minimal 3-node graph: venous_blood -> lung -> arterial_blood, with clearance."""
    g = BodyGraph()
    g.add_node(Node(name="venous_blood", node_type="blood_pool", volume=Distribution(3.7)))
    g.add_node(Node(name="arterial_blood", node_type="blood_pool", volume=Distribution(1.5)))
    g.add_node(Node(name="lung", node_type="organ", volume=Distribution(0.5)))
    g.add_node(
        Node(
            name="liver",
            node_type="organ",
            volume=Distribution(1.8),
            enzymes={"CYP3A4": Distribution(1000.0)},
            ivive_scaling=0.00006,
        )
    )
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1e10)))

    q = Distribution(390.0)
    g.add_edge(FlowEdge(source="venous_blood", target="lung", flow_rate=q))
    g.add_edge(FlowEdge(source="lung", target="arterial_blood", flow_rate=q))
    g.add_edge(FlowEdge(source="arterial_blood", target="liver", flow_rate=Distribution(100.0)))
    g.add_edge(FlowEdge(source="liver", target="venous_blood", flow_rate=Distribution(100.0)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model="well_stirred"))
    return g


def _make_simple_drug() -> DrugOnGraph:
    """Minimal IV drug for testing."""
    return DrugOnGraph(
        name="test_drug",
        smiles="C",
        dose_mg=100.0,
        route="iv",
        administration_node="venous_blood",
        mw=194.19,
        pka=None,
        compound_type="neutral",
        fup=Distribution(0.5, cv=0.2),
        rbp=Distribution(1.0),
        kp_method="provided",
        kp_overrides={"liver": Distribution(1.0), "lung": Distribution(1.0)},
        peff=Distribution(5.0),
        solubility=Distribution(10.0),
        enzyme_affinity={"CYP3A4": Distribution(10.0, cv=0.3)},
        renal_clearance=Distribution(0.0),
    )


class TestUncertaintyEngine:
    def test_propagate_returns_uncertainty_result(self):
        graph = _make_simple_graph()
        drug = _make_simple_drug()
        compiled = ODECompiler().compile(graph)
        ue = UncertaintyEngine()
        result = ue.propagate(compiled, graph, drug, n_samples=5, seed=42)
        assert isinstance(result, UncertaintyResult)
        assert result.n_samples + result.n_failures == 5
        assert result.n_samples > 0

    def test_propagate_produces_sim_results(self):
        graph = _make_simple_graph()
        drug = _make_simple_drug()
        compiled = ODECompiler().compile(graph)
        ue = UncertaintyEngine()
        result = ue.propagate(compiled, graph, drug, n_samples=3, seed=42)
        assert len(result.sim_results) == result.n_samples
        for sr in result.sim_results:
            assert sr.solver_success
            assert "venous_blood" in sr.concentrations

    def test_propagate_reproducible(self):
        """Same seed produces same results."""
        graph = _make_simple_graph()
        drug = _make_simple_drug()
        compiled = ODECompiler().compile(graph)
        ue = UncertaintyEngine()
        r1 = ue.propagate(compiled, graph, drug, n_samples=3, seed=42)
        r2 = ue.propagate(compiled, graph, drug, n_samples=3, seed=42)
        assert r1.n_samples == r2.n_samples
        for sr1, sr2 in zip(r1.sim_results, r2.sim_results):
            np.testing.assert_array_almost_equal(
                sr1.concentrations["venous_blood"],
                sr2.concentrations["venous_blood"],
            )

    def test_propagate_different_seeds_differ(self):
        """Different seeds produce different results (when cv > 0)."""
        graph = _make_simple_graph()
        drug = _make_simple_drug()
        compiled = ODECompiler().compile(graph)
        ue = UncertaintyEngine()
        r1 = ue.propagate(compiled, graph, drug, n_samples=1, seed=42)
        r2 = ue.propagate(compiled, graph, drug, n_samples=1, seed=99)
        assert r1.n_samples == 1 and r2.n_samples == 1
        # With different seeds and cv > 0, concentrations should differ
        c1 = r1.sim_results[0].concentrations["venous_blood"]
        c2 = r2.sim_results[0].concentrations["venous_blood"]
        assert not np.allclose(c1, c2)
