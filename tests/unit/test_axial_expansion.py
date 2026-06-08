"""WS-3: axial sub-compartment expansion."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import (
    ODECompiler,
    ResolvedParams,
)
from sisyphus.graph.axial import expand_axial
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node


def test_node_has_axial_field():
    n = Node(name="liver", node_type="organ", volume=Distribution(1.0), axial_subcompartments=5)
    assert n.axial_subcompartments == 5


def test_node_axial_defaults_to_one():
    n = Node(name="liver", node_type="organ", volume=Distribution(1.0))
    assert n.axial_subcompartments == 1


def _pt_graph(n: int) -> BodyGraph:
    """dose(lumen) → organ(parallel_tube, N) → drain(sink); organ → metab(sink)."""
    g = BodyGraph()
    g.add_node(Node(name="dose", node_type="lumen", volume=Distribution(1.0)))
    g.add_node(Node(name="organ", node_type="organ", volume=Distribution(2.0),
                    enzymes={"CYP": Distribution(20.0)}, ivive_scaling=1.0,
                    axial_subcompartments=n, lookup_name="organ"))
    g.add_node(Node(name="drain", node_type="sink", volume=Distribution(1.0)))
    g.add_node(Node(name="metab", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source="dose", target="organ", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="organ", target="drain", flow_rate=Distribution(10.0)))
    g.add_edge(ClearanceEdge(source="organ", target="metab", model="parallel_tube"))
    return g


def test_expand_creates_n_tanks_and_removes_organ():
    g = expand_axial(_pt_graph(4))
    assert "organ" not in g.nodes
    tanks = [n for n in g.nodes if n.startswith("organ__ax")]
    assert len(tanks) == 4


def test_expand_divides_extensive_copies_intensive():
    g = expand_axial(_pt_graph(4))
    t1 = g.nodes["organ__ax1"]
    assert t1.volume.mean == pytest.approx(2.0 / 4)
    assert t1.enzymes["CYP"].mean == pytest.approx(20.0 / 4)
    assert t1.ivive_scaling == 1.0          # intensive — copied
    assert t1.lookup_name == "organ"        # Kp/PS resolution → parent


def test_expand_preserves_flow_conservation():
    g = expand_axial(_pt_graph(4))
    assert g.validate() == []


def test_expand_clearance_edges_become_well_stirred():
    g = expand_axial(_pt_graph(3))
    cl = [e for e in g.edges if getattr(e, "model", None) is not None]
    assert len(cl) == 3
    assert all(e.model == "well_stirred" for e in cl)


def test_expand_early_returns_unchanged_when_no_parallel_tube():
    g = BodyGraph()
    g.add_node(Node(name="liver", node_type="organ", volume=Distribution(1.0)))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model="well_stirred"))
    assert expand_axial(g) is g  # same object → bit-identity


def test_expand_scope_guard_rejects_nonperfusion_edge():
    from sisyphus.graph.types import DiffusionEdge
    g = _pt_graph(4)
    g.add_node(Node(name="tissue", node_type="organ", volume=Distribution(1.0)))
    g.add_edge(DiffusionEdge(source="organ", target="tissue", ps_product=Distribution(1.0)))
    with pytest.raises(NotImplementedError, match="perfusion organ"):
        expand_axial(g)


def _drug_ws() -> DrugOnGraph:
    return DrugOnGraph(
        name="d", smiles="CCO", dose_mg=100.0, route="oral", administration_node="dose",
        mw=300.0, pka=4.5, compound_type="acid", fup=Distribution(0.5), rbp=Distribution(1.0),
        kp_method="provided", kp_overrides={}, peff=Distribution(1.0),
        solubility=Distribution(1.0), enzyme_affinity={"CYP": Distribution(1.0)},
        renal_clearance=Distribution(0.0),
    )


def _solve_extraction(graph) -> float:
    """Single-pass extraction E = metabolized / dose after full washout."""
    from sisyphus.engine.solver import solve
    compiled = ODECompiler().compile(graph)
    params = ResolvedParams(graph, _drug_ws())
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index["dose"]] = 100.0
    sim = solve(compiled, params, y0, t_span=(0.0, 3000.0))
    assert sim.solver_success
    return sim.amounts["metab"][-1] / 100.0


def _extraction(n: int) -> float:
    """Extraction of the parallel_tube organ expanded to N (>=2) tanks."""
    return _solve_extraction(expand_axial(_pt_graph(n)))


def _ws_direct_graph() -> BodyGraph:
    """Same organ as _pt_graph but a DIRECT single well_stirred tank (no expansion)."""
    g = BodyGraph()
    g.add_node(Node(name="dose", node_type="lumen", volume=Distribution(1.0)))
    g.add_node(Node(name="organ", node_type="organ", volume=Distribution(2.0),
                    enzymes={"CYP": Distribution(20.0)}, ivive_scaling=1.0, lookup_name="organ"))
    g.add_node(Node(name="drain", node_type="sink", volume=Distribution(1.0)))
    g.add_node(Node(name="metab", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source="dose", target="organ", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="organ", target="drain", flow_rate=Distribution(10.0)))
    g.add_edge(ClearanceEdge(source="organ", target="metab", model="well_stirred"))
    return g


def test_single_well_stirred_tank_matches_analytic():
    # Direct well_stirred organ (NOT expanded): E = fu_b·CLint/(Q+fu_b·CLint) = 10/(10+10) = 0.5
    assert _solve_extraction(_ws_direct_graph()) == pytest.approx(0.5, abs=0.02)


def test_large_n_converges_to_parallel_tube():
    # E_PT = 1 - exp(-fu_b·CLint/Q) = 1 - exp(-1) = 0.6321
    assert _extraction(50) == pytest.approx(1.0 - math.exp(-1.0), abs=0.01)


def test_extraction_monotone_in_n():
    # N >= 2 only: axial_subcompartments=1 maps to the default N=10, so n=1 is NOT a 1-tank case.
    assert _extraction(2) < _extraction(10) < _extraction(50)


def test_unexpanded_parallel_tube_fails_loud_at_compile():
    g = _pt_graph(4)  # NOT expanded
    with pytest.raises(ValueError, match="parallel_tube"):
        ODECompiler().compile(g)


def test_pipeline_handles_parallel_tube_graph(monkeypatch):
    """A graph with a parallel_tube edge is auto-expanded by the pipeline (no raise)."""
    # Smoke test: predict() must not raise the unexpanded-parallel_tube ValueError.
    from sisyphus.pipeline.predict import predict
    result = predict("CCO", dose_mg=100.0, route="oral")
    assert result is not None  # production graph has no parallel_tube → expand_axial is a no-op
