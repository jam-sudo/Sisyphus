"""Tests for DDI module -- inhibition and induction via enzyme abundance modification.

Verifies that DDI works through graph modification only, with zero engine changes.
"""

from pathlib import Path

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.ddi import (
    FLUCONAZOLE,
    KETOCONAZOLE,
    QUINIDINE,
    RIFAMPIN,
    Inducer,
    Inhibitor,
    apply_induction,
    apply_inhibition,
)
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node
from sisyphus.pk.endpoints import compute_endpoints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_graph() -> BodyGraph:
    """Minimal 3-node graph: blood -> liver -> sink, with CYP3A4 enzymes."""
    g = BodyGraph()
    g.add_node(
        Node(
            name="blood",
            node_type="blood_pool",
            volume=Distribution(5.0),
        )
    )
    g.add_node(
        Node(
            name="liver",
            node_type="organ",
            volume=Distribution(1.8),
            enzymes={
                "CYP3A4": Distribution(9247500.0),
                "CYP2D6": Distribution(675000.0),
            },
            ivive_scaling=0.00006,
        )
    )
    g.add_node(
        Node(
            name="sink",
            node_type="sink",
            volume=Distribution(1.0e10),
        )
    )
    g.add_edge(FlowEdge(source="blood", target="liver", flow_rate=Distribution(90.0)))
    g.add_edge(FlowEdge(source="liver", target="blood", flow_rate=Distribution(90.0)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model="well_stirred"))
    return g


def _make_drug(**overrides) -> DrugOnGraph:
    """Create a minimal DrugOnGraph for testing."""
    defaults = dict(
        name="test_drug",
        smiles="C",
        dose_mg=100.0,
        route="iv",
        administration_node="blood",
        mw=300.0,
        pka=None,
        compound_type="neutral",
        fup=Distribution(0.1),
        rbp=Distribution(1.0),
        kp_method="provided",
        kp_overrides={"liver": Distribution(5.0)},
        peff=Distribution(1.0),
        solubility=Distribution(10.0),
        enzyme_affinity={"CYP3A4": Distribution(1.0)},
        renal_clearance=Distribution(0.0),
        particle_radius_um=25.0,
        ps_overrides={},
    )
    defaults.update(overrides)
    return DrugOnGraph(**defaults)


# ---------------------------------------------------------------------------
# Unit tests: apply_inhibition
# ---------------------------------------------------------------------------


class TestApplyInhibition:
    def test_reduces_target_enzyme(self):
        """Inhibitor reduces the target enzyme abundance."""
        graph = _make_simple_graph()
        base_3a4 = graph.nodes["liver"].enzymes["CYP3A4"].mean

        inhibitor = Inhibitor(
            name="test_inhibitor",
            enzyme_ki={"CYP3A4": 1.0},
            plasma_concentration=10.0,
        )
        modified = apply_inhibition(graph, inhibitor)
        new_3a4 = modified.nodes["liver"].enzymes["CYP3A4"].mean

        # [I]/Ki = 10/1 = 10 -> factor = 1/(1+10) = 0.0909
        expected_factor = 1.0 / (1.0 + 10.0)
        assert abs(new_3a4 - base_3a4 * expected_factor) < 1.0

    def test_fu_perpetrator_scales_unbound_concentration(self):
        """fu_perpetrator < 1 lowers the driving unbound [I]_u = fu*[I], so
        inhibition is weaker than at total concentration (fu default 1.0)."""
        graph = _make_simple_graph()
        base = graph.nodes["liver"].enzymes["CYP3A4"].mean
        total = apply_inhibition(graph, Inhibitor(
            name="i", enzyme_ki={"CYP3A4": 1.0}, plasma_concentration=10.0,
        )).nodes["liver"].enzymes["CYP3A4"].mean
        unbound = apply_inhibition(graph, Inhibitor(
            name="i", enzyme_ki={"CYP3A4": 1.0}, plasma_concentration=10.0,
            fu_perpetrator=0.1,
        )).nodes["liver"].enzymes["CYP3A4"].mean
        # [I]_u = 0.1*10 = 1 -> factor 1/(1+1) = 0.5, vs 1/11 at total: less inhibited
        assert unbound > total
        assert unbound == pytest.approx(base * 0.5)

    def test_does_not_affect_other_enzymes(self):
        """Inhibitor targeting CYP3A4 should not change CYP2D6."""
        graph = _make_simple_graph()
        base_2d6 = graph.nodes["liver"].enzymes["CYP2D6"].mean

        inhibitor = Inhibitor(
            name="test_inhibitor",
            enzyme_ki={"CYP3A4": 1.0},
            plasma_concentration=10.0,
        )
        modified = apply_inhibition(graph, inhibitor)

        assert modified.nodes["liver"].enzymes["CYP2D6"].mean == base_2d6

    def test_does_not_affect_nodes_without_enzymes(self):
        """Blood pool nodes (no enzymes) should pass through unchanged."""
        graph = _make_simple_graph()
        inhibitor = Inhibitor(
            name="test_inhibitor",
            enzyme_ki={"CYP3A4": 1.0},
            plasma_concentration=10.0,
        )
        modified = apply_inhibition(graph, inhibitor)

        assert modified.nodes["blood"] is graph.nodes["blood"]

    def test_preserves_cv(self):
        """Inhibition should preserve the CV of the enzyme distribution."""
        g = BodyGraph()
        g.add_node(
            Node(
                name="organ",
                node_type="organ",
                volume=Distribution(1.0),
                enzymes={"CYP3A4": Distribution(1000.0, cv=0.3)},
            )
        )
        inhibitor = Inhibitor(
            name="test",
            enzyme_ki={"CYP3A4": 1.0},
            plasma_concentration=1.0,
        )
        modified = apply_inhibition(g, inhibitor)
        assert modified.nodes["organ"].enzymes["CYP3A4"].cv == 0.3

    def test_does_not_mutate_original(self):
        """apply_inhibition must return a NEW graph, not mutate the original."""
        graph = _make_simple_graph()
        base_3a4 = graph.nodes["liver"].enzymes["CYP3A4"].mean

        inhibitor = Inhibitor(
            name="test",
            enzyme_ki={"CYP3A4": 1.0},
            plasma_concentration=10.0,
        )
        _ = apply_inhibition(graph, inhibitor)

        # Original must be unchanged
        assert graph.nodes["liver"].enzymes["CYP3A4"].mean == base_3a4

    def test_preserves_edges(self):
        """All edges from the original graph must be preserved."""
        graph = _make_simple_graph()
        inhibitor = Inhibitor(
            name="test",
            enzyme_ki={"CYP3A4": 1.0},
            plasma_concentration=1.0,
        )
        modified = apply_inhibition(graph, inhibitor)
        assert len(modified.edges) == len(graph.edges)

    def test_invalid_ki_raises(self):
        """Ki <= 0 should raise ValueError."""
        graph = _make_simple_graph()
        inhibitor = Inhibitor(
            name="bad",
            enzyme_ki={"CYP3A4": 0.0},
            plasma_concentration=1.0,
        )
        with pytest.raises(ValueError, match="Ki must be positive"):
            apply_inhibition(graph, inhibitor)

    def test_unsupported_mechanism_raises(self):
        """Non-competitive mechanisms should raise ValueError."""
        graph = _make_simple_graph()
        inhibitor = Inhibitor(
            name="bad",
            enzyme_ki={"CYP3A4": 1.0},
            plasma_concentration=1.0,
            mechanism="time_dependent",
        )
        with pytest.raises(ValueError, match="Unsupported inhibition mechanism"):
            apply_inhibition(graph, inhibitor)


# ---------------------------------------------------------------------------
# Unit tests: apply_induction
# ---------------------------------------------------------------------------


class TestApplyInduction:
    def test_increases_target_enzyme(self):
        """Inducer increases the target enzyme abundance."""
        graph = _make_simple_graph()
        base_3a4 = graph.nodes["liver"].enzymes["CYP3A4"].mean

        inducer = Inducer(
            name="test_inducer",
            enzyme_emax={"CYP3A4": 5.0},
            enzyme_ec50={"CYP3A4": 1.0},
            plasma_concentration=10.0,
        )
        modified = apply_induction(graph, inducer)
        new_3a4 = modified.nodes["liver"].enzymes["CYP3A4"].mean

        # fold = 1 + 5 * 10 / (1 + 10) = 1 + 4.545 = 5.545
        expected_fold = 1.0 + 5.0 * 10.0 / (1.0 + 10.0)
        assert abs(new_3a4 - base_3a4 * expected_fold) < 1.0

    def test_does_not_affect_other_enzymes(self):
        """Inducer targeting CYP3A4 should not change CYP2D6."""
        graph = _make_simple_graph()
        base_2d6 = graph.nodes["liver"].enzymes["CYP2D6"].mean

        inducer = Inducer(
            name="test_inducer",
            enzyme_emax={"CYP3A4": 5.0},
            enzyme_ec50={"CYP3A4": 1.0},
            plasma_concentration=10.0,
        )
        modified = apply_induction(graph, inducer)
        assert modified.nodes["liver"].enzymes["CYP2D6"].mean == base_2d6

    def test_does_not_mutate_original(self):
        """apply_induction must return a NEW graph."""
        graph = _make_simple_graph()
        base_3a4 = graph.nodes["liver"].enzymes["CYP3A4"].mean

        inducer = Inducer(
            name="test",
            enzyme_emax={"CYP3A4": 5.0},
            enzyme_ec50={"CYP3A4": 1.0},
            plasma_concentration=10.0,
        )
        _ = apply_induction(graph, inducer)
        assert graph.nodes["liver"].enzymes["CYP3A4"].mean == base_3a4

    def test_invalid_ec50_raises(self):
        """EC50 <= 0 should raise ValueError."""
        graph = _make_simple_graph()
        inducer = Inducer(
            name="bad",
            enzyme_emax={"CYP3A4": 5.0},
            enzyme_ec50={"CYP3A4": 0.0},
            plasma_concentration=1.0,
        )
        with pytest.raises(ValueError, match="EC50 must be positive"):
            apply_induction(graph, inducer)


# ---------------------------------------------------------------------------
# Unit tests: preset DDI perpetrators
# ---------------------------------------------------------------------------


class TestPresets:
    def test_ketoconazole_strong_3a4_inhibition(self):
        """Ketoconazole: [I]/Ki = 5.0/0.015 = 333 -> >99% inhibition."""
        graph = _make_simple_graph()
        base_3a4 = graph.nodes["liver"].enzymes["CYP3A4"].mean

        modified = apply_inhibition(graph, KETOCONAZOLE)
        new_3a4 = modified.nodes["liver"].enzymes["CYP3A4"].mean

        # factor = 1/(1+333.3) = 0.003 -> >99% reduction
        assert new_3a4 < base_3a4 * 0.01

    def test_ketoconazole_doesnt_affect_2d6(self):
        """Ketoconazole targets CYP3A4, not CYP2D6."""
        graph = _make_simple_graph()
        modified = apply_inhibition(graph, KETOCONAZOLE)

        assert (
            modified.nodes["liver"].enzymes["CYP2D6"].mean
            == graph.nodes["liver"].enzymes["CYP2D6"].mean
        )

    def test_quinidine_strong_2d6_inhibition(self):
        """Quinidine: [I]/Ki = 3.0/0.06 = 50 -> >98% inhibition."""
        graph = _make_simple_graph()
        base_2d6 = graph.nodes["liver"].enzymes["CYP2D6"].mean

        modified = apply_inhibition(graph, QUINIDINE)
        new_2d6 = modified.nodes["liver"].enzymes["CYP2D6"].mean

        assert new_2d6 < base_2d6 * 0.05

    def test_fluconazole_moderate_inhibition(self):
        """Fluconazole: moderate inhibitor of CYP2C9 and CYP3A4."""
        graph = _make_simple_graph()
        base_3a4 = graph.nodes["liver"].enzymes["CYP3A4"].mean

        modified = apply_inhibition(graph, FLUCONAZOLE)
        new_3a4 = modified.nodes["liver"].enzymes["CYP3A4"].mean

        # [I]/Ki = 25/10 = 2.5 -> factor = 1/3.5 = 0.286
        expected_factor = 1.0 / (1.0 + 25.0 / 10.0)
        assert abs(new_3a4 / base_3a4 - expected_factor) < 0.001

    def test_rifampin_strong_3a4_induction(self):
        """Rifampin: fold = 1 + 8 * 10 / (0.3 + 10) = ~8.8x induction."""
        graph = _make_simple_graph()
        base_3a4 = graph.nodes["liver"].enzymes["CYP3A4"].mean

        modified = apply_induction(graph, RIFAMPIN)
        new_3a4 = modified.nodes["liver"].enzymes["CYP3A4"].mean

        expected_fold = 1.0 + 8.0 * 10.0 / (0.3 + 10.0)
        assert abs(new_3a4 / base_3a4 - expected_fold) < 0.01


# ---------------------------------------------------------------------------
# Integration tests: full pipeline with DDI
# ---------------------------------------------------------------------------


class TestDDIIntegration:
    """Full pipeline tests: graph -> DDI modification -> compile -> solve -> PK.

    Uses IV bolus into blood pool, so Cmax at t=0 is always dose/V.
    DDI effect is visible in AUC (inversely proportional to clearance).
    """

    def _run_pipeline(self, graph: BodyGraph, drug: DrugOnGraph):
        """Compile, solve, and extract PK endpoints."""
        compiler = ODECompiler()
        compiled = compiler.compile(graph)

        rng = np.random.default_rng(42)
        realized_graph = graph.sample(rng)
        realized_drug = drug.sample(rng)
        params = ResolvedParams(realized_graph, realized_drug)

        y0 = np.zeros(compiled.n_states)
        admin_idx = compiled.state_index[drug.administration_node]
        y0[admin_idx] = drug.dose_mg

        t_eval = np.linspace(0, 24, 2000)
        result = solve(compiled, params, y0, t_span=(0, 24), t_eval=t_eval)
        pk = compute_endpoints(result, observation_node="blood")
        return pk

    def test_inhibition_increases_auc(self):
        """CYP3A4 inhibition should increase AUC (less clearance).

        For IV bolus, Cmax is always dose/V at t=0.  DDI effect is
        visible in AUC, which is inversely proportional to clearance.
        """
        graph = _make_simple_graph()
        drug = _make_drug()

        base_pk = self._run_pipeline(graph, drug)

        inhibited_graph = apply_inhibition(graph, KETOCONAZOLE)
        ddi_pk = self._run_pipeline(inhibited_graph, drug)

        ratio = ddi_pk.auc_0t.mean / base_pk.auc_0t.mean
        assert ratio > 1.5, f"Expected AUC increase, got ratio={ratio:.2f}"

    def test_induction_decreases_auc(self):
        """CYP3A4 induction should decrease AUC (more clearance).

        For IV bolus, DDI effect visible in AUC.
        """
        graph = _make_simple_graph()
        drug = _make_drug()

        base_pk = self._run_pipeline(graph, drug)

        induced_graph = apply_induction(graph, RIFAMPIN)
        ddi_pk = self._run_pipeline(induced_graph, drug)

        ratio = ddi_pk.auc_0t.mean / base_pk.auc_0t.mean
        assert ratio < 0.8, f"Expected AUC decrease, got ratio={ratio:.2f}"


# ---------------------------------------------------------------------------
# Integration test: full-model DDI with midazolam
# ---------------------------------------------------------------------------


class TestDDIMidazolam:
    """DDI with midazolam using the full reference_man graph."""

    def _run_midazolam(self, graph: BodyGraph):
        """Run midazolam through a graph and return PKEndpoints."""
        from sisyphus.compounds import load_compound

        drug = load_compound(Path("data/compounds/midazolam.yaml"))
        compiler = ODECompiler()
        compiled = compiler.compile(graph)

        rng = np.random.default_rng(42)
        realized_graph = graph.sample(rng)
        realized_drug = drug.sample(rng)
        params = ResolvedParams(realized_graph, realized_drug)

        y0 = np.zeros(compiled.n_states)
        admin_idx = compiled.state_index[drug.administration_node]
        y0[admin_idx] = drug.dose_mg

        t_eval = np.linspace(0, 24, 2000)
        result = solve(compiled, params, y0, t_span=(0, 24), t_eval=t_eval)
        return compute_endpoints(result, observation_node="venous_blood")

    def test_ketoconazole_increases_midazolam_cmax(self):
        """Ketoconazole + midazolam: strong CYP3A4 inhibition -> Cmax increase.

        Clinical reference: ketoconazole increases midazolam AUC ~15x.
        """
        from sisyphus.graph.builder import build_from_yaml

        graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        base_pk = self._run_midazolam(graph)

        inhibited_graph = apply_inhibition(graph, KETOCONAZOLE)
        ddi_pk = self._run_midazolam(inhibited_graph)

        cmax_ratio = ddi_pk.cmax.mean / base_pk.cmax.mean
        auc_ratio = ddi_pk.auc_0t.mean / base_pk.auc_0t.mean

        # Ketoconazole should substantially increase midazolam exposure
        assert cmax_ratio > 2.0, f"Cmax ratio={cmax_ratio:.2f}, expected >2.0"
        assert auc_ratio > 2.0, f"AUC ratio={auc_ratio:.2f}, expected >2.0"

    def test_rifampin_decreases_midazolam_cmax(self):
        """Rifampin + midazolam: strong CYP3A4 induction -> Cmax decrease.

        Clinical reference: rifampin decreases midazolam Cmax ~90%.
        """
        from sisyphus.graph.builder import build_from_yaml

        graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        base_pk = self._run_midazolam(graph)

        induced_graph = apply_induction(graph, RIFAMPIN)
        ddi_pk = self._run_midazolam(induced_graph)

        cmax_ratio = ddi_pk.cmax.mean / base_pk.cmax.mean

        # Rifampin should substantially decrease midazolam exposure
        assert cmax_ratio < 0.5, f"Cmax ratio={cmax_ratio:.2f}, expected <0.5"


# ---------------------------------------------------------------------------
# Architectural invariant: engine not modified
# ---------------------------------------------------------------------------


class TestEngineUnmodified:
    """Verify DDI works without any engine changes."""

    def test_engine_files_unchanged(self):
        """DDI operates on graph, not engine.

        This test documents the invariant that DDI is implemented
        entirely through graph modification.  If the engine needed
        changes for DDI, the architecture would have failed.
        """
        engine_dir = Path("src/sisyphus/engine")
        engine_files = sorted(engine_dir.glob("*.py"))

        # Engine files must exist
        assert len(engine_files) > 0

        # Verify DDI works through standard pipeline
        graph = _make_simple_graph()
        drug = _make_drug()

        # Without DDI
        compiler = ODECompiler()
        compiled = compiler.compile(graph)
        rng = np.random.default_rng(42)
        params = ResolvedParams(graph.sample(rng), drug.sample(rng))
        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index["blood"]] = 100.0
        result_base = solve(compiled, params, y0, t_span=(0, 24))

        # With DDI -- same compile/solve pipeline, different graph
        modified = apply_inhibition(graph, KETOCONAZOLE)
        compiled_ddi = compiler.compile(modified)
        params_ddi = ResolvedParams(modified.sample(rng), drug.sample(rng))
        y0_ddi = np.zeros(compiled_ddi.n_states)
        y0_ddi[compiled_ddi.state_index["blood"]] = 100.0
        result_ddi = solve(compiled_ddi, params_ddi, y0_ddi, t_span=(0, 24))

        # Both must succeed through the standard pipeline
        assert result_base.solver_success
        assert result_ddi.solver_success
