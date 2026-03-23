"""Tests for ODE compiler, CompiledODE, and ResolvedParams."""

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 — registers all FluxSpec implementations
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import FlowEdge, Node

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_two_node_graph():
    g = BodyGraph()
    g.add_node(Node(name="a", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_node(Node(name="b", node_type="organ", volume=Distribution(2.0)))
    g.add_edge(FlowEdge(source="a", target="b", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="b", target="a", flow_rate=Distribution(10.0)))
    g.global_params["cardiac_output"] = Distribution(390.0)
    return g


def make_minimal_drug():
    return DrugOnGraph(
        name="test",
        smiles="C",
        dose_mg=100.0,
        route="oral",
        administration_node="a",
        mw=100.0,
        pka=None,
        compound_type="neutral",
        fup=Distribution(0.5),
        rbp=Distribution(1.0),
        kp_method="provided",
        kp_overrides={"b": Distribution(2.0)},
        peff=Distribution(1.0),
        solubility=Distribution(10.0),
        enzyme_affinity={},
        renal_clearance=Distribution(0.0),
    )


# ---------------------------------------------------------------------------
# ODECompiler tests
# ---------------------------------------------------------------------------


class TestODECompiler:
    def test_compile_assigns_states(self):
        g = make_two_node_graph()
        compiled = ODECompiler().compile(g)
        assert compiled.n_states == 2
        assert "a" in compiled.state_index
        assert "b" in compiled.state_index

    def test_compile_state_indices_sorted(self):
        g = make_two_node_graph()
        compiled = ODECompiler().compile(g)
        assert compiled.state_index["a"] == 0
        assert compiled.state_index["b"] == 1

    def test_compile_creates_flux_specs(self):
        g = make_two_node_graph()
        compiled = ODECompiler().compile(g)
        assert len(compiled.flux_specs) == 2

    def test_make_rhs_returns_callable(self):
        g = make_two_node_graph()
        compiled = ODECompiler().compile(g)
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        rhs = compiled.make_rhs(params)
        assert callable(rhs)
        y = np.array([100.0, 0.0])
        dydt = rhs(0.0, y)
        assert dydt.shape == (2,)

    def test_compile_unknown_edge_type_raises(self):
        """Compiler should raise ValueError for unregistered edge types."""
        from sisyphus.graph.types import Edge

        g = BodyGraph()
        g.add_node(Node(name="x", node_type="organ", volume=Distribution(1.0)))
        g.add_node(Node(name="y", node_type="organ", volume=Distribution(1.0)))
        # Manually add an edge with an unregistered type
        g.edges.append(Edge(source="x", target="y", edge_type="wormhole"))
        with pytest.raises(ValueError, match="No FluxSpec registered"):
            ODECompiler().compile(g)


# ---------------------------------------------------------------------------
# ResolvedParams tests
# ---------------------------------------------------------------------------


class TestResolvedParams:
    def test_node_param_volume(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        assert params.node_param("a", "volume") == 1.0
        assert params.node_param("b", "volume") == 2.0

    def test_node_param_ivive_scaling(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        assert params.node_param("a", "ivive_scaling") == 0.0

    def test_node_param_unknown_raises(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        with pytest.raises(KeyError, match="Unknown node param"):
            params.node_param("a", "color")

    def test_drug_kp(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        assert params.drug_kp("b") == 2.0  # from kp_overrides
        assert params.drug_kp("a") == 1.0  # blood_pool default

    def test_drug_param(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        assert params.drug_param("fup") == 0.5
        assert params.drug_param("rbp") == 1.0
        assert params.drug_param("dose_mg") == 100.0

    def test_drug_param_unknown_raises(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        with pytest.raises(KeyError, match="Unknown drug param"):
            params.drug_param("magic")

    def test_total_inflow(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        assert params.total_inflow("a") == 10.0
        assert params.total_inflow("b") == 10.0

    def test_edge_param(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        assert params.edge_param(0, "flow_rate") == 10.0
        assert params.edge_param(1, "flow_rate") == 10.0

    def test_node_enzymes_empty(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        assert params.node_enzymes("a") == {}

    def test_drug_enzyme_affinity_missing(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        assert params.drug_enzyme_affinity("CYP3A4") == 0.0

    def test_drug_ps_default(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        assert params.drug_ps("a") == 0.0

    def test_global_param(self):
        g = make_two_node_graph()
        drug = make_minimal_drug()
        params = ResolvedParams(g, drug)
        assert params.global_param("cardiac_output") == 390.0
        assert params.global_param("nonexistent") == 0.0
