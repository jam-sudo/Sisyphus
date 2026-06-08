"""WS-5: ActiveTransportFluxSpec honors edge direction (uptake vs efflux)."""
from __future__ import annotations

import numpy as np

import sisyphus.engine.flux  # noqa: F401 — registers FluxSpec implementations
from sisyphus.core import Distribution, DrugOnGraph, TransporterKinetics
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ActiveTransportEdge, FlowEdge, Node


def _drug() -> DrugOnGraph:
    return DrugOnGraph(
        name="d", smiles="CCO", dose_mg=100.0, route="oral",
        administration_node="gut_wall", mw=424.0, pka=4.5, compound_type="acid",
        fup=Distribution(0.5), rbp=Distribution(1.0), kp_method="rodgers_rowland",
        kp_overrides={}, peff=Distribution(1.0), solubility=Distribution(1.0),
        enzyme_affinity={}, renal_clearance=Distribution(0.0),
        transporter_kinetics={
            "PGP": TransporterKinetics(jmax=Distribution(228.0), km=Distribution(50.0))
        },
    )


def _efflux_graph() -> BodyGraph:
    """gut_wall (PGP, source) --efflux--> lumen (no transporters)."""
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(1.0),
                    transporters={"PGP": Distribution(1.0e10)}, ivive_scaling=1.0e-4))
    g.add_node(Node(name="lumen", node_type="lumen", volume=Distribution(1.0)))
    g.add_edge(ActiveTransportEdge(source="gut_wall", target="lumen", direction="efflux"))
    return g


def test_efflux_reads_source_transporters():
    """Efflux: transporter sits at the SOURCE (gut_wall); flux moves mass out of it."""
    g = _efflux_graph()
    compiled = ODECompiler().compile(g)
    rhs = compiled.make_rhs(ResolvedParams(g, _drug()))
    y = np.zeros(compiled.n_states)
    y[compiled.state_index["gut_wall"]] = 10.0
    dydt = rhs(0.0, y)
    assert dydt[compiled.state_index["gut_wall"]] < 0, "efflux should remove mass from source"
    assert dydt[compiled.state_index["lumen"]] > 0, "efflux should add mass to target"


def test_uptake_default_unchanged():
    """direction defaults to 'uptake' → transporter read at TARGET (legacy behavior)."""
    g = BodyGraph()
    g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_node(Node(name="liver", node_type="organ", volume=Distribution(1.5),
                    transporters={"PGP": Distribution(1.0e10)}, ivive_scaling=1.0e-4))
    g.add_edge(FlowEdge(source="blood", target="liver", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="liver", target="blood", flow_rate=Distribution(10.0)))
    g.add_edge(ActiveTransportEdge(source="blood", target="liver"))  # no direction → uptake
    compiled = ODECompiler().compile(g)
    rhs = compiled.make_rhs(ResolvedParams(g, _drug()))
    y = np.zeros(compiled.n_states)
    y[compiled.state_index["blood"]] = 10.0
    dydt = rhs(0.0, y)
    # Uptake reads target (liver) transporters; mass moves blood → liver via transport.
    assert dydt[compiled.state_index["liver"]] > 0


def test_efflux_scipy_jax_parity():
    import pytest
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from sisyphus.engine.params_jax import resolve_to_jax
    from sisyphus.engine.rhs_jax import make_jax_rhs

    g = _efflux_graph()
    compiled = ODECompiler().compile(g)
    params = ResolvedParams(g, _drug())
    rhs_scipy = compiled.make_rhs(params)
    y = np.zeros(compiled.n_states)
    y[compiled.state_index["gut_wall"]] = 10.0
    dydt_scipy = rhs_scipy(0.0, y)

    jax_params = resolve_to_jax(compiled, params)
    dydt_jax = np.asarray(make_jax_rhs(compiled)(0.0, jnp.asarray(y), jax_params))
    np.testing.assert_allclose(dydt_jax, dydt_scipy, rtol=1e-6, atol=1e-9)
