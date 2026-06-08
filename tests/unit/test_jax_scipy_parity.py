"""WS-4: JAX↔SciPy RHS-level parity per flux branch."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import sisyphus.engine.flux  # noqa: F401,E402
from sisyphus.core import Distribution, DrugOnGraph  # noqa: E402
from sisyphus.engine.compiler import ODECompiler, ResolvedParams  # noqa: E402
from sisyphus.engine.params_jax import resolve_to_jax  # noqa: E402
from sisyphus.engine.rhs_jax import make_jax_rhs  # noqa: E402
from sisyphus.graph.body import BodyGraph  # noqa: E402
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node  # noqa: E402


def _drug(fu_corr: float = 1.0) -> DrugOnGraph:
    return DrugOnGraph(
        name="d", smiles="CCO", dose_mg=100.0, route="oral",
        administration_node="blood", mw=300.0, pka=4.5, compound_type="acid",
        fup=Distribution(0.3), rbp=Distribution(1.0), kp_method="provided",
        kp_overrides={"organ": Distribution(1.0)}, peff=Distribution(1.0),
        solubility=Distribution(1.0), enzyme_affinity={"CYP3A4": Distribution(5.0)},
        renal_clearance=Distribution(0.0), fu_correction_liver=Distribution(fu_corr),
    )


def _ws_flagged_graph(flagged: bool = True) -> BodyGraph:
    g = BodyGraph()
    g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(5.0)))
    g.add_node(Node(name="organ", node_type="organ", volume=Distribution(1.0),
                    enzymes={"CYP3A4": Distribution(1.0e6)}, ivive_scaling=1.0e-6,
                    fu_correction_applicable=1.0 if flagged else 0.0, lookup_name="organ"))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source="blood", target="organ", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="organ", target="blood", flow_rate=Distribution(10.0)))
    g.add_edge(ClearanceEdge(source="organ", target="sink", model="well_stirred"))
    return g


def _rhs_pair(graph, drug):
    compiled = ODECompiler().compile(graph)
    params = ResolvedParams(graph, drug)
    y = np.zeros(compiled.n_states)
    y[compiled.state_index["organ"]] = 3.0
    y[compiled.state_index["blood"]] = 7.0
    dydt_scipy = compiled.make_rhs(params)(0.0, y)
    dydt_jax = np.asarray(
        make_jax_rhs(compiled)(0.0, jnp.asarray(y), resolve_to_jax(compiled, params))
    )
    return dydt_scipy, dydt_jax


def test_ws_fu_correction_parity():
    """Flagged well_stirred node with non-identity fu_correction: JAX == SciPy."""
    s, j = _rhs_pair(_ws_flagged_graph(), _drug(fu_corr=1.5))
    np.testing.assert_allclose(j, s, rtol=1e-9, atol=1e-12)
