"""WS-3: axial sub-compartment expansion."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.engine.compiler import (  # noqa: F401  (used by later tasks)
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
