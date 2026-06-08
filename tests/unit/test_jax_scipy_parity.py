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
from sisyphus.graph.types import (  # noqa: E402
    AbsorptionEdge,
    ClearanceEdge,
    DiffusionEdge,
    FlowEdge,
    Node,
    TransitEdge,
)


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


def test_jax_fails_loud_on_distinct_km_multitransporter():
    """≥2 active transporters with distinct Km at one node: the JAX aggregate
    Vmax/weighted-Km approximation diverges from SciPy → fail loud, not silent."""
    from sisyphus.core import TransporterKinetics
    from sisyphus.graph.types import ActiveTransportEdge

    g = BodyGraph()
    g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_node(Node(name="organ", node_type="organ", volume=Distribution(1.0),
                    transporters={"A": Distribution(1e10), "B": Distribution(1e10)},
                    ivive_scaling=1e-4))
    g.add_edge(FlowEdge(source="blood", target="organ", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="organ", target="blood", flow_rate=Distribution(10.0)))
    g.add_edge(ActiveTransportEdge(source="blood", target="organ"))
    drug = DrugOnGraph(
        name="d", smiles="CCO", dose_mg=100.0, route="oral", administration_node="blood",
        mw=300.0, pka=4.5, compound_type="acid", fup=Distribution(0.3), rbp=Distribution(1.0),
        kp_method="provided", kp_overrides={"organ": Distribution(1.0)}, peff=Distribution(1.0),
        solubility=Distribution(1.0), enzyme_affinity={}, renal_clearance=Distribution(0.0),
        transporter_kinetics={
            "A": TransporterKinetics(jmax=Distribution(100.0), km=Distribution(10.0)),
            "B": TransporterKinetics(jmax=Distribution(100.0), km=Distribution(80.0)),
        },
    )
    compiled = ODECompiler().compile(g)
    with pytest.raises(NotImplementedError, match="distinct Km"):
        resolve_to_jax(compiled, ResolvedParams(g, drug))


# ---------------------------------------------------------------------------
# Per-branch RHS-level parity (SciPy make_rhs == make_jax_rhs at fixed t, y).
# Each test builds a minimal valid graph exercising ONLY one flux branch and
# asserts dydt agreement to rtol=1e-9, atol=1e-12. RHS-level (not integrated
# trajectories): isolates flux math from integrator differences.
# ---------------------------------------------------------------------------


def _assert_parity(graph, drug, fill):
    compiled = ODECompiler().compile(graph)
    params = ResolvedParams(graph, drug)
    y = np.zeros(compiled.n_states)
    for name, amt in fill.items():
        y[compiled.state_index[name]] = amt
    s = compiled.make_rhs(params)(0.0, y)
    j = np.asarray(make_jax_rhs(compiled)(0.0, jnp.asarray(y), resolve_to_jax(compiled, params)))
    np.testing.assert_allclose(j, s, rtol=1e-9, atol=1e-12)


def test_ws_no_fu_correction_parity():
    g = _ws_flagged_graph(flagged=False)  # well_stirred WITHOUT the fu_correction flag
    _assert_parity(g, _drug(fu_corr=1.0), {"organ": 3.0, "blood": 7.0})


def test_gfr_parity():
    import dataclasses
    g = BodyGraph()
    g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(5.0)))
    g.add_node(Node(name="kidney", node_type="organ", volume=Distribution(1.0),
                    lookup_name="kidney"))
    g.add_node(Node(name="urine", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source="blood", target="kidney", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="kidney", target="blood", flow_rate=Distribution(10.0)))
    g.add_edge(ClearanceEdge(source="kidney", target="urine", model="gfr_filtration"))
    drug = dataclasses.replace(_drug(), renal_clearance=Distribution(5.0))
    _assert_parity(g, drug, {"kidney": 2.0, "blood": 8.0})


def test_flow_parity():
    # FlowEdge between a blood pool (C_out = A/V) and an organ
    # (C_out = A*RBP/(V*Kp)); both convective forms exercised, balanced flow.
    g = BodyGraph()
    g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(5.0)))
    g.add_node(Node(name="organ", node_type="organ", volume=Distribution(1.0),
                    lookup_name="organ"))
    g.add_edge(FlowEdge(source="blood", target="organ", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="organ", target="blood", flow_rate=Distribution(10.0)))
    _assert_parity(g, _drug(), {"organ": 3.0, "blood": 7.0})


def test_transit_parity():
    # First-order transit (rate = k * A_source) between two flow-exempt
    # lumen nodes — no flow-conservation requirement.
    g = BodyGraph()
    g.add_node(Node(name="lumen1", node_type="lumen", volume=Distribution(1.0)))
    g.add_node(Node(name="lumen2", node_type="lumen", volume=Distribution(1.0)))
    g.add_edge(TransitEdge(source="lumen1", target="lumen2", transit_rate=Distribution(0.5)))
    _assert_parity(g, _drug(), {"lumen1": 4.0})


def test_absorption_parity():
    # ka = 2.88 * peff * ka_fraction / particle_radius_um; _drug() has
    # peff=1.0 and the default radius=25.0 → ka>0. Lumen source is flow-exempt;
    # the organ target has no flow edges, so conservation is satisfied trivially.
    g = BodyGraph()
    g.add_node(Node(name="lumen", node_type="lumen", volume=Distribution(1.0)))
    g.add_node(Node(name="gut", node_type="organ", volume=Distribution(1.0),
                    lookup_name="gut"))
    g.add_edge(AbsorptionEdge(source="lumen", target="gut", ka_fraction=Distribution(1.0)))
    _assert_parity(g, _drug(), {"lumen": 5.0})


def test_diffusion_parity():
    # PS-limited exchange: flux = PS * (fup*C_vasc/RBP - fup*C_tissue/Kp).
    # ps_product on the edge (no drug ps_override) drives the flux; mass in
    # vasc with the tissue defaulting Kp=1.0 yields a nonzero gradient.
    g = BodyGraph()
    g.add_node(Node(name="vasc", node_type="blood_pool", volume=Distribution(2.0)))
    g.add_node(Node(name="tissue", node_type="organ", volume=Distribution(1.0),
                    lookup_name="tissue"))
    g.add_edge(DiffusionEdge(source="vasc", target="tissue", ps_product=Distribution(20.0)))
    _assert_parity(g, _drug(), {"vasc": 6.0, "tissue": 1.0})
