"""Tests for FluxSpec implementations — flow, clearance, transit, absorption, diffusion."""

import numpy as np

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.flux import (
    FLUX_REGISTRY,
    AbsorptionFluxSpec,
    ClearanceFluxSpec,
    DiffusionFluxSpec,
    FlowFluxSpec,
    TransitFluxSpec,
)
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    AbsorptionEdge,
    ClearanceEdge,
    DiffusionEdge,
    FlowEdge,
    Node,
    TransitEdge,
)

# ---------------------------------------------------------------------------
# ECM Fields Test
# ---------------------------------------------------------------------------


def test_drug_on_graph_has_ecm_fields():
    """DrugOnGraph should have ps_passive, ps_eff, cl_int_bile with WS-limit defaults."""
    drug = DrugOnGraph(
        name="t", smiles="C", dose_mg=100.0, route="iv",
        administration_node="venous_blood",
        mw=100.0, pka=None, compound_type="neutral",
        fup=Distribution(0.5), rbp=Distribution(1.0),
        kp_method="provided", kp_overrides={},
        peff=Distribution(1.0), solubility=Distribution(10.0),
        enzyme_affinity={}, renal_clearance=Distribution(0.0),
    )
    assert drug.ps_passive.mean == 1e6
    assert drug.ps_eff.mean == 1e6
    assert drug.cl_int_bile.mean == 0.0
    assert drug.ps_passive.cv == 0.0
    assert drug.ps_eff.cv == 0.0
    assert drug.cl_int_bile.cv == 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_drug(**overrides) -> DrugOnGraph:
    """Create a minimal DrugOnGraph with sensible defaults."""
    defaults = dict(
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
        kp_overrides={},
        peff=Distribution(1.0),
        solubility=Distribution(10.0),
        enzyme_affinity={},
        renal_clearance=Distribution(0.0),
        particle_radius_um=25.0,
        ps_overrides={},
    )
    defaults.update(overrides)
    return DrugOnGraph(**defaults)


# ---------------------------------------------------------------------------
# FlowFluxSpec
# ---------------------------------------------------------------------------


class TestFlowFlux:
    def test_registered(self):
        assert "flow" in FLUX_REGISTRY
        assert FLUX_REGISTRY["flow"] is FlowFluxSpec

    def test_mass_conservation(self):
        """Flow from A to B: total dydt sums to zero."""
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="blood_pool", volume=Distribution(5.0)))
        g.add_node(Node(name="b", node_type="blood_pool", volume=Distribution(3.0)))
        g.add_edge(FlowEdge(source="a", target="b", flow_rate=Distribution(10.0)))

        drug = _make_drug()
        params = ResolvedParams(g, drug)

        state_index = {"a": 0, "b": 1}
        spec = FlowFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([50.0, 30.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        # Mass conservation: what leaves source arrives at target
        assert abs(dydt[0] + dydt[1]) < 1e-12
        # Flux should be positive (drug moves from a to b)
        assert dydt[0] < 0
        assert dydt[1] > 0

    def test_kp_correction(self):
        """Tissue node with Kp=2: outflow concentration is halved."""
        g = BodyGraph()
        g.add_node(Node(name="src", node_type="organ", volume=Distribution(2.0)))
        g.add_node(Node(name="dst", node_type="blood_pool", volume=Distribution(5.0)))
        g.add_edge(FlowEdge(source="src", target="dst", flow_rate=Distribution(10.0)))

        drug_kp2 = _make_drug(kp_overrides={"src": Distribution(2.0)})
        params_kp2 = ResolvedParams(g, drug_kp2)

        drug_kp1 = _make_drug(kp_overrides={"src": Distribution(1.0)})
        params_kp1 = ResolvedParams(g, drug_kp1)

        state_index = {"dst": 0, "src": 1}
        spec = FlowFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([0.0, 100.0])  # src has 100mg

        dydt_kp2 = np.zeros(2)
        spec.apply(0.0, y, dydt_kp2, params_kp2)

        dydt_kp1 = np.zeros(2)
        spec.apply(0.0, y, dydt_kp1, params_kp1)

        # With Kp=2, flux should be half of Kp=1 (same amount, same volume)
        assert abs(dydt_kp2[1]) > 0
        assert abs(dydt_kp2[1] / dydt_kp1[1] - 0.5) < 1e-10

    def test_zero_volume_no_crash(self):
        """Zero volume source should produce zero flux, not division error."""
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="blood_pool", volume=Distribution(0.0)))
        g.add_node(Node(name="b", node_type="blood_pool", volume=Distribution(5.0)))
        g.add_edge(FlowEdge(source="a", target="b", flow_rate=Distribution(10.0)))

        drug = _make_drug()
        params = ResolvedParams(g, drug)

        state_index = {"a": 0, "b": 1}
        spec = FlowFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([50.0, 30.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        assert dydt[0] == 0.0
        assert dydt[1] == 0.0


# ---------------------------------------------------------------------------
# ClearanceFluxSpec
# ---------------------------------------------------------------------------


class TestClearanceFlux:
    def test_registered(self):
        assert "clearance" in FLUX_REGISTRY
        assert FLUX_REGISTRY["clearance"] is ClearanceFluxSpec

    def test_well_stirred_removes_mass(self):
        """Clearance reduces drug in source, adds to sink."""
        g = BodyGraph()
        g.add_node(
            Node(
                name="liver",
                node_type="organ",
                volume=Distribution(1.5),
                enzymes={"CYP3A4": Distribution(100.0)},
                ivive_scaling=0.01,
            )
        )
        g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
        # Need a flow edge targeting liver so total_inflow is non-zero
        g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(5.0)))
        g.add_edge(FlowEdge(source="blood", target="liver", flow_rate=Distribution(20.0)))
        g.add_edge(ClearanceEdge(source="liver", target="sink", model="well_stirred"))

        drug = _make_drug(
            kp_overrides={"liver": Distribution(2.0)},
            enzyme_affinity={"CYP3A4": Distribution(50.0)},
        )
        params = ResolvedParams(g, drug)

        # state_index: blood=0, liver=1, sink=2 (alphabetical)
        state_index = {"blood": 0, "liver": 1, "sink": 2}
        edge = g.edges[1]  # ClearanceEdge
        spec = ClearanceFluxSpec.from_edge(1, edge, state_index)

        y = np.array([100.0, 50.0, 0.0])
        dydt = np.zeros(3)
        spec.apply(0.0, y, dydt, params)

        # Source should lose mass, sink should gain
        assert dydt[1] < 0  # liver loses drug
        assert dydt[2] > 0  # sink gains drug
        # Conservation: what liver loses, sink gains
        assert abs(dydt[1] + dydt[2]) < 1e-12

    def test_no_enzymes_no_clearance(self):
        """Node without matching enzymes: zero clearance rate."""
        g = BodyGraph()
        g.add_node(
            Node(
                name="liver",
                node_type="organ",
                volume=Distribution(1.5),
                enzymes={},  # no enzymes
                ivive_scaling=0.01,
            )
        )
        g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
        g.add_edge(ClearanceEdge(source="liver", target="sink", model="well_stirred"))

        drug = _make_drug(
            enzyme_affinity={"CYP3A4": Distribution(50.0)},
        )
        params = ResolvedParams(g, drug)

        state_index = {"liver": 0, "sink": 1}
        spec = ClearanceFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([50.0, 0.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        assert dydt[0] == 0.0
        assert dydt[1] == 0.0

    def test_gfr_filtration(self):
        """GFR filtration clearance removes mass proportional to renal_clearance."""
        g = BodyGraph()
        g.add_node(Node(name="kidney", node_type="organ", volume=Distribution(0.3)))
        g.add_node(Node(name="urine", node_type="sink", volume=Distribution(1.0)))
        g.add_edge(ClearanceEdge(source="kidney", target="urine", model="gfr_filtration"))

        drug = _make_drug(
            renal_clearance=Distribution(7.2),
            kp_overrides={"kidney": Distribution(1.0)},
            rbp=Distribution(1.0),
        )
        params = ResolvedParams(g, drug)

        state_index = {"kidney": 0, "urine": 1}
        spec = ClearanceFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([30.0, 0.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        # rate = renal_cl * c_plasma = 7.2 * (30 * 1.0 / (0.3 * 1.0)) = 7.2 * 100 = 720
        expected_rate = 7.2 * (30.0 / 0.3)
        assert abs(dydt[0] - (-expected_rate)) < 1e-10
        assert abs(dydt[1] - expected_rate) < 1e-10

    def test_gfr_zero_renal_clearance(self):
        """Zero renal clearance produces zero flux."""
        g = BodyGraph()
        g.add_node(Node(name="kidney", node_type="organ", volume=Distribution(0.3)))
        g.add_node(Node(name="urine", node_type="sink", volume=Distribution(1.0)))
        g.add_edge(ClearanceEdge(source="kidney", target="urine", model="gfr_filtration"))

        drug = _make_drug(renal_clearance=Distribution(0.0))
        params = ResolvedParams(g, drug)

        state_index = {"kidney": 0, "urine": 1}
        spec = ClearanceFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([30.0, 0.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        assert dydt[0] == 0.0
        assert dydt[1] == 0.0


# ---------------------------------------------------------------------------
# TransitFluxSpec
# ---------------------------------------------------------------------------


class TestTransitFlux:
    def test_registered(self):
        assert "transit" in FLUX_REGISTRY
        assert FLUX_REGISTRY["transit"] is TransitFluxSpec

    def test_first_order_transit(self):
        """Transit moves mass proportional to amount in source."""
        g = BodyGraph()
        g.add_node(Node(name="seg1", node_type="lumen", volume=Distribution(0.25)))
        g.add_node(Node(name="seg2", node_type="lumen", volume=Distribution(0.25)))
        g.add_edge(TransitEdge(source="seg1", target="seg2", transit_rate=Distribution(4.0)))

        drug = _make_drug()
        params = ResolvedParams(g, drug)

        state_index = {"seg1": 0, "seg2": 1}
        spec = TransitFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([80.0, 20.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        # rate = k_transit * A_source = 4.0 * 80.0 = 320.0
        assert abs(dydt[0] - (-320.0)) < 1e-10
        assert abs(dydt[1] - 320.0) < 1e-10

    def test_mass_conservation(self):
        """Transit conserves mass: dydt sums to zero."""
        g = BodyGraph()
        g.add_node(Node(name="s1", node_type="lumen", volume=Distribution(0.25)))
        g.add_node(Node(name="s2", node_type="lumen", volume=Distribution(0.25)))
        g.add_edge(TransitEdge(source="s1", target="s2", transit_rate=Distribution(2.5)))

        drug = _make_drug()
        params = ResolvedParams(g, drug)

        state_index = {"s1": 0, "s2": 1}
        spec = TransitFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([42.0, 58.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        assert abs(dydt[0] + dydt[1]) < 1e-12


# ---------------------------------------------------------------------------
# AbsorptionFluxSpec
# ---------------------------------------------------------------------------


class TestAbsorptionFlux:
    def test_registered(self):
        assert "absorption" in FLUX_REGISTRY
        assert FLUX_REGISTRY["absorption"] is AbsorptionFluxSpec

    def test_absorption_rate(self):
        """Absorption rate = 2.88 * peff * ka_frac / radius * amount."""
        g = BodyGraph()
        g.add_node(Node(name="lumen", node_type="lumen", volume=Distribution(0.25)))
        g.add_node(Node(name="gut", node_type="organ", volume=Distribution(1.0)))
        g.add_edge(
            AbsorptionEdge(
                source="lumen",
                target="gut",
                ka_fraction=Distribution(0.8),
            )
        )

        peff = 2.0
        radius = 25.0
        drug = _make_drug(peff=Distribution(peff), particle_radius_um=radius)
        params = ResolvedParams(g, drug)

        state_index = {"gut": 0, "lumen": 1}
        spec = AbsorptionFluxSpec.from_edge(0, g.edges[0], state_index)

        amount = 100.0
        y = np.array([0.0, amount])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        ka = 2.88 * peff * 0.8 / radius
        expected_rate = ka * amount
        assert abs(dydt[1] - (-expected_rate)) < 1e-10
        assert abs(dydt[0] - expected_rate) < 1e-10

    def test_zero_peff_no_absorption(self):
        """Zero effective permeability means no absorption."""
        g = BodyGraph()
        g.add_node(Node(name="lumen", node_type="lumen", volume=Distribution(0.25)))
        g.add_node(Node(name="gut", node_type="organ", volume=Distribution(1.0)))
        g.add_edge(
            AbsorptionEdge(
                source="lumen",
                target="gut",
                ka_fraction=Distribution(1.0),
            )
        )

        drug = _make_drug(peff=Distribution(0.0))
        params = ResolvedParams(g, drug)

        state_index = {"gut": 0, "lumen": 1}
        spec = AbsorptionFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([0.0, 100.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        assert dydt[0] == 0.0
        assert dydt[1] == 0.0

    def test_zero_ka_fraction_no_absorption(self):
        """Zero ka_fraction means no absorption at this segment."""
        g = BodyGraph()
        g.add_node(Node(name="lumen", node_type="lumen", volume=Distribution(0.25)))
        g.add_node(Node(name="gut", node_type="organ", volume=Distribution(1.0)))
        g.add_edge(
            AbsorptionEdge(
                source="lumen",
                target="gut",
                ka_fraction=Distribution(0.0),
            )
        )

        drug = _make_drug(peff=Distribution(2.0))
        params = ResolvedParams(g, drug)

        state_index = {"gut": 0, "lumen": 1}
        spec = AbsorptionFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([0.0, 100.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        assert dydt[0] == 0.0
        assert dydt[1] == 0.0


# ---------------------------------------------------------------------------
# DiffusionFluxSpec
# ---------------------------------------------------------------------------


class TestDiffusionFlux:
    def test_registered(self):
        assert "diffusion" in FLUX_REGISTRY
        assert FLUX_REGISTRY["diffusion"] is DiffusionFluxSpec

    def test_equilibrium_no_flux(self):
        """When unbound concentrations are equal, flux is zero."""
        g = BodyGraph()
        g.add_node(
            Node(
                name="adipose_vasc",
                node_type="organ",
                volume=Distribution(1.0),
                lookup_name="adipose",
            )
        )
        g.add_node(
            Node(
                name="adipose_tissue",
                node_type="organ",
                volume=Distribution(10.0),
                lookup_name="adipose",
            )
        )
        g.add_edge(
            DiffusionEdge(
                source="adipose_vasc",
                target="adipose_tissue",
                ps_product=Distribution(5.0),
            )
        )

        kp = 2.0
        fup = 0.5
        rbp = 1.0
        drug = _make_drug(
            fup=Distribution(fup),
            rbp=Distribution(rbp),
            kp_overrides={"adipose": Distribution(kp)},
        )
        params = ResolvedParams(g, drug)

        state_index = {"adipose_tissue": 0, "adipose_vasc": 1}
        spec = DiffusionFluxSpec.from_edge(0, g.edges[0], state_index)

        # Set amounts so unbound concentrations are equal:
        # cu_vasc = fup * (A_vasc / V_vasc) / rbp
        # cu_tissue = fup * (A_tissue / V_tissue) / kp
        # Equal when A_vasc / V_vasc / rbp = A_tissue / V_tissue / kp
        # i.e., A_vasc / (1.0 * 1.0) = A_tissue / (10.0 * 2.0)
        # So if A_vasc = 1.0, then A_tissue = 20.0
        y = np.array([20.0, 1.0])  # [adipose_tissue, adipose_vasc]
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        assert abs(dydt[0]) < 1e-12
        assert abs(dydt[1]) < 1e-12

    def test_net_flux_direction(self):
        """Flux flows from higher to lower unbound concentration."""
        g = BodyGraph()
        g.add_node(
            Node(
                name="adipose_vasc",
                node_type="organ",
                volume=Distribution(1.0),
                lookup_name="adipose",
            )
        )
        g.add_node(
            Node(
                name="adipose_tissue",
                node_type="organ",
                volume=Distribution(10.0),
                lookup_name="adipose",
            )
        )
        g.add_edge(
            DiffusionEdge(
                source="adipose_vasc",
                target="adipose_tissue",
                ps_product=Distribution(5.0),
            )
        )

        drug = _make_drug(
            fup=Distribution(0.5),
            rbp=Distribution(1.0),
            kp_overrides={"adipose": Distribution(2.0)},
        )
        params = ResolvedParams(g, drug)

        state_index = {"adipose_tissue": 0, "adipose_vasc": 1}
        spec = DiffusionFluxSpec.from_edge(0, g.edges[0], state_index)

        # High vasc concentration, low tissue concentration
        y = np.array([0.0, 10.0])  # [adipose_tissue, adipose_vasc]
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        # Flux should go from vasc to tissue
        assert dydt[1] < 0  # vasc loses
        assert dydt[0] > 0  # tissue gains
        assert abs(dydt[0] + dydt[1]) < 1e-12  # mass conservation

    def test_ps_override(self):
        """Drug-specific PS takes precedence over edge PS."""
        g = BodyGraph()
        g.add_node(
            Node(
                name="brain_vasc", node_type="organ", volume=Distribution(0.5), lookup_name="brain"
            )
        )
        g.add_node(
            Node(
                name="brain_tissue",
                node_type="organ",
                volume=Distribution(1.4),
                lookup_name="brain",
            )
        )
        g.add_edge(
            DiffusionEdge(
                source="brain_vasc",
                target="brain_tissue",
                ps_product=Distribution(100.0),  # edge PS - should be overridden
            )
        )

        # Drug with PS override for brain
        drug_with_override = _make_drug(
            fup=Distribution(0.5),
            rbp=Distribution(1.0),
            kp_overrides={"brain": Distribution(1.5)},
            ps_overrides={"brain": Distribution(2.0)},  # much smaller PS
        )
        params_override = ResolvedParams(g, drug_with_override)

        # Drug without PS override
        drug_no_override = _make_drug(
            fup=Distribution(0.5),
            rbp=Distribution(1.0),
            kp_overrides={"brain": Distribution(1.5)},
            ps_overrides={},
        )
        params_no_override = ResolvedParams(g, drug_no_override)

        state_index = {"brain_tissue": 0, "brain_vasc": 1}
        spec = DiffusionFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([0.0, 10.0])

        dydt_override = np.zeros(2)
        spec.apply(0.0, y, dydt_override, params_override)

        dydt_no_override = np.zeros(2)
        spec.apply(0.0, y, dydt_no_override, params_no_override)

        # With PS=2.0 override vs PS=100.0 edge, flux should differ by 50x
        ratio = abs(dydt_no_override[0] / dydt_override[0])
        assert abs(ratio - 50.0) < 1e-10

    def test_zero_ps_no_flux(self):
        """Zero PS (both override and edge) produces no flux."""
        g = BodyGraph()
        g.add_node(Node(name="test_vasc", node_type="organ", volume=Distribution(1.0)))
        g.add_node(Node(name="test_tissue", node_type="organ", volume=Distribution(5.0)))
        g.add_edge(
            DiffusionEdge(
                source="test_vasc",
                target="test_tissue",
                ps_product=Distribution(0.0),
            )
        )

        drug = _make_drug()
        params = ResolvedParams(g, drug)

        state_index = {"test_tissue": 0, "test_vasc": 1}
        spec = DiffusionFluxSpec.from_edge(0, g.edges[0], state_index)

        y = np.array([0.0, 10.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        assert dydt[0] == 0.0
        assert dydt[1] == 0.0


# ---------------------------------------------------------------------------
# Integration: full compile + RHS evaluation
# ---------------------------------------------------------------------------


class TestFluxIntegration:
    def test_compile_and_evaluate_with_real_fluxes(self):
        """Full cycle: build graph -> compile -> make_rhs -> evaluate."""
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="blood_pool", volume=Distribution(5.0)))
        g.add_node(Node(name="b", node_type="organ", volume=Distribution(2.0)))
        g.add_edge(FlowEdge(source="a", target="b", flow_rate=Distribution(10.0)))
        g.add_edge(FlowEdge(source="b", target="a", flow_rate=Distribution(10.0)))

        drug = _make_drug(kp_overrides={"b": Distribution(2.0)})
        compiled = ODECompiler().compile(g)
        params = ResolvedParams(g, drug)
        rhs = compiled.make_rhs(params)

        y = np.array([100.0, 0.0])
        dydt = rhs(0.0, y)

        # With flow in both directions and drug only in 'a':
        # Flow a->b: flux = 10 * (100 / 5) = 200  (blood pool, kp=1, rbp=1)
        # Flow b->a: flux = 10 * (0 * 1 / (2 * 2)) = 0
        assert dydt.shape == (2,)
        assert abs(dydt[0] - (-200.0)) < 1e-10
        assert abs(dydt[1] - 200.0) < 1e-10
