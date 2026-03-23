# tests/integration/test_extensibility.py
"""Phase 3 extensibility proof: 3 extensions with zero engine changes.

Proves that SC injection, pediatric physiology, and tumor compartments
can be added purely through YAML + graph operations, without modifying
any file in src/sisyphus/engine/.

Extensions:
    1. SC (subcutaneous) injection — depot node + absorption edge
    2. Pediatric 5-year-old — scaled volumes, flows, enzyme ontogeny
    3. Tumor compartment — new organ node with diverted blood flow
"""

from pathlib import Path

import numpy as np

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.compounds import load_compound
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import build_from_yaml
from sisyphus.graph.types import (
    AbsorptionEdge,
    FlowEdge,
    Node,
    TissueComposition,
)
from sisyphus.pk.endpoints import compute_endpoints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")


def _run_simulation(
    graph: BodyGraph,
    drug: DrugOnGraph,
    t_span: tuple[float, float] = (0.0, 24.0),
):
    """Compile graph, solve deterministically, return (pk, result)."""
    compiler = ODECompiler()
    compiled = compiler.compile(graph)

    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = drug.sample(rng)
    params = ResolvedParams(realized_graph, realized_drug)

    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg

    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    result = solve(compiled, params, y0, t_span=t_span, t_eval=t_eval)
    pk = compute_endpoints(result, observation_node="venous_blood")
    return pk, result


# ---------------------------------------------------------------------------
# Extension 1: SC (Subcutaneous) Injection
# ---------------------------------------------------------------------------


class TestSCInjection:
    """SC injection: add a depot node + absorption edge to existing graph."""

    def _build_sc_graph(self) -> BodyGraph:
        """Build reference_man graph with SC depot added."""
        graph = build_from_yaml(DATA_DIR / "physiology" / "reference_man.yaml")

        # Add SC depot node (lumen type — exempt from flow conservation)
        sc_node = Node(
            name="sc_depot",
            node_type="lumen",
            volume=Distribution(0.01),
        )
        graph.add_node(sc_node)

        # Add absorption edge: sc_depot -> venous_blood
        graph.add_edge(
            AbsorptionEdge(
                source="sc_depot",
                target="venous_blood",
                ka_fraction=Distribution(1.0),
            )
        )

        return graph

    def _make_sc_drug(self) -> DrugOnGraph:
        """Load midazolam and override for SC administration."""
        drug = load_compound(DATA_DIR / "compounds" / "midazolam.yaml")
        return DrugOnGraph(
            name="Midazolam_SC",
            smiles=drug.smiles,
            dose_mg=5.0,
            route="sc",
            administration_node="sc_depot",
            mw=drug.mw,
            pka=drug.pka,
            compound_type=drug.compound_type,
            fup=drug.fup,
            rbp=drug.rbp,
            kp_method=drug.kp_method,
            kp_overrides=drug.kp_overrides,
            peff=Distribution(5.37),  # same Peff, absorption formula still applies
            solubility=drug.solubility,
            enzyme_affinity=drug.enzyme_affinity,
            renal_clearance=drug.renal_clearance,
            particle_radius_um=5.0,  # smaller radius -> faster SC absorption
            ps_overrides=drug.ps_overrides,
        )

    def test_sc_solver_success(self):
        """SC simulation completes without solver failure."""
        graph = self._build_sc_graph()
        drug = self._make_sc_drug()
        pk, result = _run_simulation(graph, drug)
        assert result.solver_success, "SC injection solver failed"

    def test_sc_mass_balance(self):
        """SC injection conserves mass."""
        graph = self._build_sc_graph()
        drug = self._make_sc_drug()
        _, result = _run_simulation(graph, drug)
        assert result.mass_balance_error < 1e-6, (
            f"SC mass balance error: {result.mass_balance_error:.2e}"
        )

    def test_sc_positive_cmax(self):
        """SC injection produces measurable Cmax in venous blood."""
        graph = self._build_sc_graph()
        drug = self._make_sc_drug()
        pk, _ = _run_simulation(graph, drug)
        assert pk.cmax.mean > 0, "SC Cmax should be > 0"

    def test_sc_higher_cmax_than_oral(self):
        """SC bypasses gut first-pass: higher Cmax per mg than oral.

        Midazolam oral 2 mg has very low Cmax (~0.007 mg/L) because
        CYP3A4 in gut wall metabolizes ~60-70% before reaching systemic.
        SC 5 mg bypasses gut entirely -> much higher systemic exposure.
        """
        # Oral reference
        oral_graph = build_from_yaml(DATA_DIR / "physiology" / "reference_man.yaml")
        oral_drug = load_compound(DATA_DIR / "compounds" / "midazolam.yaml")
        oral_pk, _ = _run_simulation(oral_graph, oral_drug)

        # SC
        sc_graph = self._build_sc_graph()
        sc_drug = self._make_sc_drug()
        sc_pk, _ = _run_simulation(sc_graph, sc_drug)

        # SC 5mg vs oral 2mg: dose-normalized SC should be much higher
        # (bypasses gut CYP3A4 first-pass)
        sc_cmax_per_mg = sc_pk.cmax.mean / 5.0
        oral_cmax_per_mg = oral_pk.cmax.mean / 2.0

        print(f"\nSC Cmax = {sc_pk.cmax.mean:.6f} mg/L (5 mg SC)")
        print(f"Oral Cmax = {oral_pk.cmax.mean:.6f} mg/L (2 mg oral)")
        print(f"SC Cmax/mg = {sc_cmax_per_mg:.6f}, Oral Cmax/mg = {oral_cmax_per_mg:.6f}")
        print(f"SC/Oral ratio (dose-normalized) = {sc_cmax_per_mg / oral_cmax_per_mg:.2f}x")

        assert sc_cmax_per_mg > oral_cmax_per_mg, (
            f"SC should have higher dose-normalized Cmax than oral "
            f"(SC: {sc_cmax_per_mg:.6f}/mg, oral: {oral_cmax_per_mg:.6f}/mg)"
        )

    def test_sc_depot_empties(self):
        """SC depot should be nearly empty by 24h."""
        graph = self._build_sc_graph()
        drug = self._make_sc_drug()
        _, result = _run_simulation(graph, drug)
        final_depot = result.amounts["sc_depot"][-1]
        assert final_depot < 0.01 * 5.0, (
            f"SC depot should be <1% of dose at 24h, got {final_depot:.4f} mg"
        )


# ---------------------------------------------------------------------------
# Extension 2: Pediatric (5-year-old) Model
# ---------------------------------------------------------------------------


class TestPediatricModel:
    """Pediatric model: same graph topology, different parameter values."""

    def test_pediatric_graph_builds(self):
        """Pediatric YAML builds and validates successfully."""
        graph = build_from_yaml(DATA_DIR / "physiology" / "pediatric_5y.yaml")
        assert len(graph.nodes) == 34  # same node count as adult
        assert graph.edge_count() > 0

    def test_pediatric_solver_success(self):
        """Pediatric simulation completes without solver failure."""
        graph = build_from_yaml(DATA_DIR / "physiology" / "pediatric_5y.yaml")
        drug = load_compound(DATA_DIR / "compounds" / "caffeine.yaml")
        _, result = _run_simulation(graph, drug)
        assert result.solver_success, "Pediatric solver failed"

    def test_pediatric_mass_balance(self):
        """Pediatric model conserves mass."""
        graph = build_from_yaml(DATA_DIR / "physiology" / "pediatric_5y.yaml")
        drug = load_compound(DATA_DIR / "compounds" / "caffeine.yaml")
        _, result = _run_simulation(graph, drug)
        assert result.mass_balance_error < 1e-6, (
            f"Pediatric mass balance error: {result.mass_balance_error:.2e}"
        )

    def test_pediatric_different_from_adult(self):
        """Pediatric Cmax differs from adult (different physiology)."""
        adult_graph = build_from_yaml(DATA_DIR / "physiology" / "reference_man.yaml")
        ped_graph = build_from_yaml(DATA_DIR / "physiology" / "pediatric_5y.yaml")

        drug = load_compound(DATA_DIR / "compounds" / "caffeine.yaml")

        adult_pk, _ = _run_simulation(adult_graph, drug)
        ped_pk, _ = _run_simulation(ped_graph, drug)

        print(f"\nAdult Cmax = {adult_pk.cmax.mean:.4f} mg/L (caffeine 100 mg)")
        print(f"Pediatric Cmax = {ped_pk.cmax.mean:.4f} mg/L (caffeine 100 mg)")
        print(f"Ratio (ped/adult) = {ped_pk.cmax.mean / adult_pk.cmax.mean:.2f}")

        assert ped_pk.cmax.mean > 0, "Pediatric Cmax should be > 0"
        assert ped_pk.cmax.mean != adult_pk.cmax.mean, "Pediatric and adult Cmax should differ"

    def test_pediatric_higher_cmax_per_kg(self):
        """Pediatric should have higher Cmax (same dose, smaller body).

        100 mg caffeine in an 18 kg child vs 70 kg adult.
        Smaller Vd + lower clearance (enzyme ontogeny) -> higher Cmax.
        """
        adult_graph = build_from_yaml(DATA_DIR / "physiology" / "reference_man.yaml")
        ped_graph = build_from_yaml(DATA_DIR / "physiology" / "pediatric_5y.yaml")

        drug = load_compound(DATA_DIR / "compounds" / "caffeine.yaml")

        adult_pk, _ = _run_simulation(adult_graph, drug)
        ped_pk, _ = _run_simulation(ped_graph, drug)

        # Same absolute dose, smaller child -> higher concentration
        assert ped_pk.cmax.mean > adult_pk.cmax.mean, (
            f"Pediatric Cmax ({ped_pk.cmax.mean:.4f}) should exceed "
            f"adult Cmax ({adult_pk.cmax.mean:.4f}) for same dose"
        )

    def test_pediatric_scaled_cardiac_output(self):
        """Pediatric CO is allometrically scaled from adult."""
        ped_graph = build_from_yaml(DATA_DIR / "physiology" / "pediatric_5y.yaml")
        adult_graph = build_from_yaml(DATA_DIR / "physiology" / "reference_man.yaml")

        adult_co = adult_graph.global_params["cardiac_output"].mean
        ped_co = ped_graph.global_params["cardiac_output"].mean

        # (18/70)^0.75 ≈ 0.370
        expected_ratio = (18.0 / 70.0) ** 0.75
        actual_ratio = ped_co / adult_co

        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.02, (
            f"Pediatric CO scaling: expected ratio {expected_ratio:.3f}, got {actual_ratio:.3f}"
        )


# ---------------------------------------------------------------------------
# Extension 3: Tumor Compartment
# ---------------------------------------------------------------------------


class TestTumorCompartment:
    """Tumor compartment: add an organ node with flow to existing graph."""

    def _build_tumor_graph(self) -> BodyGraph:
        """Build reference_man graph with tumor added."""
        graph = build_from_yaml(DATA_DIR / "physiology" / "reference_man.yaml")

        co = graph.global_params["cardiac_output"].mean  # 390 L/h

        # Add tumor node
        tumor = Node(
            name="tumor",
            node_type="organ",
            volume=Distribution(0.05),
            composition=TissueComposition(
                fn=0.0132,
                fp=0.0100,
                fw=0.700,
                pH=6.8,
            ),
        )
        graph.add_node(tumor)

        # Steal 0.5% of CO from rest for tumor blood flow
        tumor_flow = 0.005 * co  # 1.95 L/h

        # Add flow edges for tumor
        graph.add_edge(
            FlowEdge(
                source="arterial_blood",
                target="tumor",
                flow_rate=Distribution(tumor_flow),
            )
        )
        graph.add_edge(
            FlowEdge(
                source="tumor",
                target="venous_blood",
                flow_rate=Distribution(tumor_flow),
            )
        )

        # Reduce rest flow to compensate (maintain flow conservation)
        # Current rest inflow = 0.069 * 390 = 26.91
        # New rest inflow = (0.069 - 0.005) * 390 = 24.96
        # Remove old rest edges and re-add with reduced flow
        rest_new_flow = (0.069 - 0.005) * co
        graph.edges = [
            e
            for e in graph.edges
            if not (
                isinstance(e, FlowEdge)
                and (
                    (e.source == "arterial_blood" and e.target == "rest")
                    or (e.source == "rest" and e.target == "venous_blood")
                )
            )
        ]
        graph.add_edge(
            FlowEdge(
                source="arterial_blood",
                target="rest",
                flow_rate=Distribution(rest_new_flow),
            )
        )
        graph.add_edge(
            FlowEdge(
                source="rest",
                target="venous_blood",
                flow_rate=Distribution(rest_new_flow),
            )
        )

        return graph

    def test_tumor_solver_success(self):
        """Tumor simulation completes without solver failure."""
        graph = self._build_tumor_graph()
        drug = load_compound(DATA_DIR / "compounds" / "midazolam.yaml")
        # Add tumor Kp override
        drug_with_tumor = DrugOnGraph(
            name=drug.name,
            smiles=drug.smiles,
            dose_mg=drug.dose_mg,
            route=drug.route,
            administration_node=drug.administration_node,
            mw=drug.mw,
            pka=drug.pka,
            compound_type=drug.compound_type,
            fup=drug.fup,
            rbp=drug.rbp,
            kp_method=drug.kp_method,
            kp_overrides={**drug.kp_overrides, "tumor": Distribution(8.0)},
            peff=drug.peff,
            solubility=drug.solubility,
            enzyme_affinity=drug.enzyme_affinity,
            renal_clearance=drug.renal_clearance,
            particle_radius_um=drug.particle_radius_um,
            ps_overrides=drug.ps_overrides,
        )
        _, result = _run_simulation(graph, drug_with_tumor)
        assert result.solver_success, "Tumor simulation solver failed"

    def test_tumor_mass_balance(self):
        """Tumor model conserves mass."""
        graph = self._build_tumor_graph()
        drug = load_compound(DATA_DIR / "compounds" / "midazolam.yaml")
        drug_with_tumor = DrugOnGraph(
            name=drug.name,
            smiles=drug.smiles,
            dose_mg=drug.dose_mg,
            route=drug.route,
            administration_node=drug.administration_node,
            mw=drug.mw,
            pka=drug.pka,
            compound_type=drug.compound_type,
            fup=drug.fup,
            rbp=drug.rbp,
            kp_method=drug.kp_method,
            kp_overrides={**drug.kp_overrides, "tumor": Distribution(8.0)},
            peff=drug.peff,
            solubility=drug.solubility,
            enzyme_affinity=drug.enzyme_affinity,
            renal_clearance=drug.renal_clearance,
            particle_radius_um=drug.particle_radius_um,
            ps_overrides=drug.ps_overrides,
        )
        _, result = _run_simulation(graph, drug_with_tumor)
        assert result.mass_balance_error < 1e-6, (
            f"Tumor mass balance error: {result.mass_balance_error:.2e}"
        )

    def test_tumor_accumulates_drug(self):
        """Tumor shows drug accumulation (Kp=8 for lipophilic base)."""
        graph = self._build_tumor_graph()
        drug = load_compound(DATA_DIR / "compounds" / "midazolam.yaml")
        drug_with_tumor = DrugOnGraph(
            name=drug.name,
            smiles=drug.smiles,
            dose_mg=drug.dose_mg,
            route=drug.route,
            administration_node=drug.administration_node,
            mw=drug.mw,
            pka=drug.pka,
            compound_type=drug.compound_type,
            fup=drug.fup,
            rbp=drug.rbp,
            kp_method=drug.kp_method,
            kp_overrides={**drug.kp_overrides, "tumor": Distribution(8.0)},
            peff=drug.peff,
            solubility=drug.solubility,
            enzyme_affinity=drug.enzyme_affinity,
            renal_clearance=drug.renal_clearance,
            particle_radius_um=drug.particle_radius_um,
            ps_overrides=drug.ps_overrides,
        )
        _, result = _run_simulation(graph, drug_with_tumor)

        tumor_conc = result.concentrations["tumor"]
        tumor_max = float(np.max(tumor_conc))
        print(f"\nTumor max concentration = {tumor_max:.6f} mg/L")
        assert tumor_max > 0, "Tumor should accumulate drug"

    def test_tumor_graph_has_35_nodes(self):
        """Tumor graph has 35 nodes (34 adult + 1 tumor)."""
        graph = self._build_tumor_graph()
        assert len(graph.nodes) == 35, f"Expected 35 nodes, got {len(graph.nodes)}"

    def test_tumor_flow_conservation(self):
        """Tumor graph passes flow conservation validation."""
        graph = self._build_tumor_graph()
        errors = graph.validate()
        assert len(errors) == 0, f"Flow conservation errors: {errors}"

    def test_tumor_does_not_affect_baseline(self):
        """Adding tumor does not change other compartments' Cmax significantly.

        The 0.5% flow diversion from rest is minimal; venous_blood Cmax
        should change by <5% vs baseline (without tumor).
        """
        # Baseline without tumor
        baseline_graph = build_from_yaml(DATA_DIR / "physiology" / "reference_man.yaml")
        drug = load_compound(DATA_DIR / "compounds" / "midazolam.yaml")
        drug_with_tumor = DrugOnGraph(
            name=drug.name,
            smiles=drug.smiles,
            dose_mg=drug.dose_mg,
            route=drug.route,
            administration_node=drug.administration_node,
            mw=drug.mw,
            pka=drug.pka,
            compound_type=drug.compound_type,
            fup=drug.fup,
            rbp=drug.rbp,
            kp_method=drug.kp_method,
            kp_overrides={**drug.kp_overrides, "tumor": Distribution(8.0)},
            peff=drug.peff,
            solubility=drug.solubility,
            enzyme_affinity=drug.enzyme_affinity,
            renal_clearance=drug.renal_clearance,
            particle_radius_um=drug.particle_radius_um,
            ps_overrides=drug.ps_overrides,
        )
        baseline_pk, _ = _run_simulation(baseline_graph, drug)

        # With tumor
        tumor_graph = self._build_tumor_graph()
        tumor_pk, _ = _run_simulation(tumor_graph, drug_with_tumor)

        rel_diff = abs(tumor_pk.cmax.mean - baseline_pk.cmax.mean) / baseline_pk.cmax.mean
        print(f"\nBaseline Cmax = {baseline_pk.cmax.mean:.6f}")
        print(f"Tumor Cmax = {tumor_pk.cmax.mean:.6f}")
        print(f"Relative difference = {rel_diff:.2%}")

        assert rel_diff < 0.05, f"Tumor addition changed Cmax by {rel_diff:.1%} (>5%)"
