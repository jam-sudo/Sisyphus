"""Unit tests for ActiveTransportFluxSpec (SciPy path).

Covers the bug fix where ivive_scaling and transporter abundance must
read from the TARGET node (where the transporter is expressed), not
the source.
"""

from __future__ import annotations

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 — registers all FluxSpec implementations
from sisyphus.core import Distribution, DrugOnGraph, TransporterKinetics
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    ActiveTransportEdge,
    FlowEdge,
    Node,
)


def _minimal_uptake_graph() -> BodyGraph:
    """Source (blood pool, ivive=0) → target (organ, ivive=1e-4, OATP1B1=1e10)."""
    g = BodyGraph()
    g.add_node(
        Node(
            name="src_blood",
            node_type="blood_pool",
            volume=Distribution(1.0),
            ivive_scaling=0.0,
        )
    )
    g.add_node(
        Node(
            name="tgt_organ",
            node_type="organ",
            volume=Distribution(1.5),
            transporters={"OATP1B1": Distribution(1.0e10)},
            ivive_scaling=1.0e-4,
        )
    )
    # Need at least one flow edge so flow conservation is satisfied.
    g.add_edge(FlowEdge(source="src_blood", target="tgt_organ", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="tgt_organ", target="src_blood", flow_rate=Distribution(10.0)))
    g.add_edge(ActiveTransportEdge(source="src_blood", target="tgt_organ"))
    g.global_params = {"cardiac_output": Distribution(10.0)}
    return g


def _drug_with_oatp(jmax: float = 228.0, km: float = 13.6) -> DrugOnGraph:
    return DrugOnGraph(
        name="test_drug",
        smiles="CCO",
        dose_mg=100.0,
        route="oral",
        administration_node="src_blood",
        mw=424.0,
        pka=4.5,
        compound_type="acid",
        fup=Distribution(0.5),
        rbp=Distribution(1.0),
        kp_method="rodgers_rowland",
        kp_overrides={},
        peff=Distribution(1.0),
        solubility=Distribution(1.0),
        enzyme_affinity={},
        renal_clearance=Distribution(0.0),
        transporter_kinetics={
            "OATP1B1": TransporterKinetics(
                jmax=Distribution(jmax),
                km=Distribution(km),
            ),
        },
    )


def _drug_no_transport() -> DrugOnGraph:
    return DrugOnGraph(
        name="no_transport",
        smiles="CCO",
        dose_mg=100.0,
        route="oral",
        administration_node="src_blood",
        mw=424.0,
        pka=4.5,
        compound_type="acid",
        fup=Distribution(0.5),
        rbp=Distribution(1.0),
        kp_method="rodgers_rowland",
        kp_overrides={},
        peff=Distribution(1.0),
        solubility=Distribution(1.0),
        enzyme_affinity={},
        renal_clearance=Distribution(0.0),
    )


def test_active_transport_uses_target_ivive_and_transporters():
    """Flux is non-zero when target has OATP1B1 and drug has kinetics,
    even though source is a blood pool with ivive=0 and no transporters."""
    graph = _minimal_uptake_graph()
    drug = _drug_with_oatp()

    compiled = ODECompiler().compile(graph)
    params = ResolvedParams(graph, drug)
    rhs = compiled.make_rhs(params)

    # Put 10 mg in source, 0 in target.
    y = np.zeros(compiled.n_states)
    y[compiled.state_index["src_blood"]] = 10.0
    dydt = rhs(0.0, y)

    src_idx = compiled.state_index["src_blood"]
    tgt_idx = compiled.state_index["tgt_organ"]

    # Mass leaves source, lands in target. The flow edge also contributes,
    # so we check the active-transport component by subtracting a no-transport run.
    drug_flow_only = _drug_no_transport()
    params_flow_only = ResolvedParams(graph, drug_flow_only)
    rhs_flow_only = compiled.make_rhs(params_flow_only)
    dydt_flow_only = rhs_flow_only(0.0, y)

    transport_contrib_src = dydt[src_idx] - dydt_flow_only[src_idx]
    transport_contrib_tgt = dydt[tgt_idx] - dydt_flow_only[tgt_idx]

    assert transport_contrib_src < 0, (
        f"expected mass leaving source via active transport, got {transport_contrib_src}"
    )
    assert transport_contrib_tgt > 0, (
        f"expected mass arriving at target via active transport, got {transport_contrib_tgt}"
    )
    # Mass conservation within the active transport path
    assert abs(transport_contrib_src + transport_contrib_tgt) < 1e-12


def test_active_transport_scipy_jax_parity():
    """SciPy and JAX backends produce the same dydt for an active-transport edge."""
    from sisyphus.engine.params_jax import resolve_to_jax
    from sisyphus.engine.rhs_jax import make_jax_rhs
    import jax.numpy as jnp

    graph = _minimal_uptake_graph()
    drug = _drug_with_oatp()

    compiled = ODECompiler().compile(graph)
    params_scipy = ResolvedParams(graph, drug)
    rhs_scipy = compiled.make_rhs(params_scipy)

    y = np.zeros(compiled.n_states)
    y[compiled.state_index["src_blood"]] = 10.0
    dydt_scipy = rhs_scipy(0.0, y)

    jax_params = resolve_to_jax(compiled, params_scipy)
    rhs_jax_fn = make_jax_rhs(compiled)
    dydt_jax = np.asarray(rhs_jax_fn(0.0, jnp.asarray(y), jax_params))

    np.testing.assert_allclose(dydt_jax, dydt_scipy, rtol=1e-6, atol=1e-9)
