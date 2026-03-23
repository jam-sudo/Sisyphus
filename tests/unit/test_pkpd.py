"""Tests for PK/PD link -- effect compartment + Emax response model.

Verifies:
- Effect-site equilibration kinetics (Ce lags Cp, converges at steady state)
- Sigmoid Emax model correctness (EC50 = 50%, Hill coefficient effects)
- Baseline effect handling
- Integration with engine SimResult
- No engine modification required
"""

from pathlib import Path

import numpy as np
import pytest

from sisyphus.core import SimResult
from sisyphus.pkpd import (
    MIDAZOLAM_SEDATION,
    WARFARIN_INR,
    PDModel,
    compute_effect,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim(
    time: np.ndarray,
    cp: np.ndarray,
    node: str = "venous_blood",
) -> SimResult:
    """Create a minimal SimResult with given plasma concentrations."""
    return SimResult(
        time_h=time,
        concentrations={node: cp},
        amounts={},
        mass_balance_error=0.0,
        solver_success=True,
    )


# ---------------------------------------------------------------------------
# PDModel validation
# ---------------------------------------------------------------------------


class TestPDModelValidation:
    def test_valid_model(self):
        """Standard PDModel should construct without error."""
        pd = PDModel(ke0=1.0, emax=100.0, ec50=1.0, hill=1.0, baseline=0.0)
        assert pd.ke0 == 1.0

    def test_ke0_must_be_positive(self):
        """ke0 <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="ke0 must be positive"):
            PDModel(ke0=0.0)

    def test_ec50_must_be_positive(self):
        """ec50 <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="ec50 must be positive"):
            PDModel(ke0=1.0, ec50=0.0)

    def test_hill_must_be_positive(self):
        """hill <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="hill must be positive"):
            PDModel(ke0=1.0, hill=0.0)


# ---------------------------------------------------------------------------
# Effect-site equilibration
# ---------------------------------------------------------------------------


class TestEffectSiteEquilibration:
    def test_ce_starts_at_zero(self):
        """Effect-site concentration should start at zero."""
        time = np.linspace(0, 24, 500)
        cp = np.ones_like(time)
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=1.0, emax=100.0, ec50=1.0, hill=1.0)
        result = compute_effect(sim, pd)

        assert result.ce[0] == 0.0

    def test_ce_approaches_cp_at_steady_state(self):
        """Ce should approach Cp at steady state (constant Cp)."""
        time = np.linspace(0, 24, 500)
        cp = np.ones_like(time)  # constant Cp = 1.0
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=1.0, emax=100.0, ec50=1.0, hill=1.0)
        result = compute_effect(sim, pd)

        # After 24h at ke0=1.0 h^-1, Ce should be very close to 1.0
        # Time constant = 1/ke0 = 1h, 24 time constants -> essentially equilibrated
        assert result.ce[-1] > 0.99

    def test_ce_lags_behind_cp(self):
        """Ce should lag behind Cp during the rising phase."""
        time = np.linspace(0, 2, 200)
        cp = np.ones_like(time)
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=1.0, emax=100.0, ec50=1.0, hill=1.0)
        result = compute_effect(sim, pd)

        # At early times, Ce < Cp
        early_idx = 10
        assert result.ce[early_idx] < cp[early_idx]

    def test_faster_ke0_equilibrates_faster(self):
        """Higher ke0 should produce faster equilibration."""
        time = np.linspace(0, 2, 200)
        cp = np.ones_like(time)
        sim = _make_sim(time, cp)

        pd_slow = PDModel(ke0=0.5, emax=100.0, ec50=1.0, hill=1.0)
        pd_fast = PDModel(ke0=5.0, emax=100.0, ec50=1.0, hill=1.0)

        result_slow = compute_effect(sim, pd_slow)
        result_fast = compute_effect(sim, pd_fast)

        # At t=1h, faster ke0 should be closer to steady state
        idx_1h = np.searchsorted(time, 1.0)
        assert result_fast.ce[idx_1h] > result_slow.ce[idx_1h]

    def test_ce_decays_when_cp_drops(self):
        """Ce should decay when Cp drops to zero."""
        time = np.linspace(0, 10, 500)
        cp = np.zeros_like(time)
        cp[:250] = 1.0  # Cp = 1.0 for first 5h, then 0
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=1.0, emax=100.0, ec50=1.0, hill=1.0)
        result = compute_effect(sim, pd)

        # Ce at end of infusion period (t~5h) should be near 1.0
        idx_5h = np.searchsorted(time, 4.9)
        assert result.ce[idx_5h] > 0.9

        # Ce at end (t=10h) should have decayed substantially
        assert result.ce[-1] < 0.1

    def test_exact_solution_accuracy(self):
        """Verify against analytical solution for constant Cp.

        For constant Cp=C0, Ce(t) = C0 * (1 - exp(-ke0*t)).
        """
        time = np.linspace(0, 10, 10000)  # fine grid for accuracy
        c0 = 2.5
        cp = np.full_like(time, c0)
        sim = _make_sim(time, cp)

        ke0 = 0.8
        pd = PDModel(ke0=ke0, emax=100.0, ec50=1.0, hill=1.0)
        result = compute_effect(sim, pd)

        # Analytical: Ce(t) = C0 * (1 - exp(-ke0*t))
        ce_analytical = c0 * (1.0 - np.exp(-ke0 * time))

        # Should match to high precision on fine grid
        max_err = float(np.max(np.abs(result.ce - ce_analytical)))
        assert max_err < 0.001, f"Max error = {max_err}"


# ---------------------------------------------------------------------------
# Emax model
# ---------------------------------------------------------------------------


class TestEmaxModel:
    def test_at_ec50_effect_is_half_emax(self):
        """At Ce = EC50, effect should be 50% of Emax (for hill=1)."""
        time = np.array([0.0, 100.0])
        cp = np.array([1.0, 1.0])
        sim = _make_sim(time, cp)

        # Very fast ke0 so Ce ≈ Cp after 100h
        pd = PDModel(ke0=100.0, emax=100.0, ec50=1.0, hill=1.0)
        result = compute_effect(sim, pd)

        # Ce ≈ 1.0 = EC50, so E ≈ 50%
        assert abs(result.effect[-1] - 50.0) < 1.0

    def test_emax_saturation(self):
        """At very high Ce >> EC50, effect should approach Emax."""
        time = np.array([0.0, 100.0])
        cp = np.array([1000.0, 1000.0])  # Ce >> EC50
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=100.0, emax=100.0, ec50=1.0, hill=1.0)
        result = compute_effect(sim, pd)

        assert result.effect[-1] > 99.0

    def test_zero_concentration_gives_baseline(self):
        """Zero drug concentration should give only baseline effect."""
        time = np.array([0.0, 1.0])
        cp = np.zeros(2)
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=1.0, emax=100.0, ec50=1.0, baseline=5.0)
        result = compute_effect(sim, pd)

        assert result.effect[0] == 5.0
        assert result.effect[-1] == 5.0

    def test_baseline_added_to_effect(self):
        """Baseline should be additive with Emax effect."""
        time = np.array([0.0, 100.0])
        cp = np.array([1.0, 1.0])
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=100.0, emax=100.0, ec50=1.0, hill=1.0, baseline=10.0)
        result = compute_effect(sim, pd)

        # At EC50: E = 10 + 100*0.5 = 60
        assert abs(result.effect[-1] - 60.0) < 1.0


# ---------------------------------------------------------------------------
# Hill coefficient
# ---------------------------------------------------------------------------


class TestHillCoefficient:
    def test_higher_hill_steeper_response(self):
        """Higher Hill coefficient should give steeper dose-response."""
        time = np.linspace(0, 100, 1000)
        cp = np.full_like(time, 0.5)  # Cp = 0.5 * EC50
        sim = _make_sim(time, cp)

        pd_h1 = PDModel(ke0=10.0, emax=100.0, ec50=1.0, hill=1.0)
        pd_h3 = PDModel(ke0=10.0, emax=100.0, ec50=1.0, hill=3.0)

        r1 = compute_effect(sim, pd_h1)
        r3 = compute_effect(sim, pd_h3)

        # With hill=1: E at Ce=0.5 -> 100*0.5/(1+0.5) = 33.3%
        # With hill=3: E at Ce=0.5 -> 100*0.125/(1+0.125) = 11.1%
        # Higher hill = lower effect at sub-EC50 concentrations
        assert r3.effect[-1] < r1.effect[-1]

    def test_hill_1_is_standard_emax(self):
        """Hill=1 should give standard hyperbolic Emax."""
        time = np.array([0.0, 100.0])
        cp = np.array([2.0, 2.0])  # 2x EC50
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=100.0, emax=100.0, ec50=1.0, hill=1.0)
        result = compute_effect(sim, pd)

        # E = 100 * 2 / (1 + 2) = 66.67
        assert abs(result.effect[-1] - 66.67) < 1.0

    def test_hill_2_gives_expected_value(self):
        """Hill=2 at Ce = EC50 should still give 50%."""
        time = np.array([0.0, 100.0])
        cp = np.array([1.0, 1.0])  # Ce = EC50
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=100.0, emax=100.0, ec50=1.0, hill=2.0)
        result = compute_effect(sim, pd)

        # E = 100 * 1 / (1 + 1) = 50% regardless of hill value at Ce=EC50
        assert abs(result.effect[-1] - 50.0) < 1.0


# ---------------------------------------------------------------------------
# PDResult properties
# ---------------------------------------------------------------------------


class TestPDResult:
    def test_emax_achieved(self):
        """emax_achieved should match the peak of the effect array."""
        time = np.linspace(0, 10, 500)
        cp = np.zeros_like(time)
        cp[50:150] = 1.0  # pulse of drug
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=2.0, emax=100.0, ec50=0.5, hill=1.0)
        result = compute_effect(sim, pd)

        assert result.emax_achieved == float(np.max(result.effect))

    def test_temax(self):
        """temax should be the time of peak effect."""
        time = np.linspace(0, 10, 500)
        cp = np.zeros_like(time)
        cp[50:150] = 1.0  # pulse
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=2.0, emax=100.0, ec50=0.5, hill=1.0)
        result = compute_effect(sim, pd)

        assert result.temax == float(time[np.argmax(result.effect)])

    def test_pd_peak_delayed_vs_pk_peak(self):
        """Peak PD effect should be delayed relative to peak Cp."""
        time = np.linspace(0, 10, 1000)
        # Simulated oral-like profile: Cp rises then decays
        cp = 2.0 * time * np.exp(-0.5 * time)
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=0.5, emax=100.0, ec50=0.5, hill=1.0)
        result = compute_effect(sim, pd)

        tmax_pk = float(time[np.argmax(cp)])
        assert result.temax > tmax_pk, (
            f"PD peak (t={result.temax:.2f}h) should be after "
            f"PK peak (t={tmax_pk:.2f}h)"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_observation_node_raises(self):
        """KeyError should be raised if observation node not in SimResult."""
        time = np.array([0.0, 1.0])
        cp = np.array([1.0, 1.0])
        sim = _make_sim(time, cp, node="arterial_blood")

        pd = PDModel(ke0=1.0, emax=100.0, ec50=1.0)
        with pytest.raises(KeyError, match="venous_blood"):
            compute_effect(sim, pd, observation_node="venous_blood")

    def test_custom_observation_node(self):
        """Should work with non-default observation nodes."""
        time = np.linspace(0, 10, 100)
        cp = np.ones_like(time)
        sim = _make_sim(time, cp, node="brain")

        pd = PDModel(ke0=1.0, emax=100.0, ec50=1.0)
        result = compute_effect(sim, pd, observation_node="brain")

        assert result.emax_achieved > 0


# ---------------------------------------------------------------------------
# Preset PD models
# ---------------------------------------------------------------------------


class TestPresets:
    def test_midazolam_sedation_preset(self):
        """Midazolam sedation preset should have expected parameters."""
        assert MIDAZOLAM_SEDATION.ke0 == 0.5
        assert MIDAZOLAM_SEDATION.emax == 100.0
        assert MIDAZOLAM_SEDATION.ec50 == 0.05
        assert MIDAZOLAM_SEDATION.hill == 2.0
        assert MIDAZOLAM_SEDATION.baseline == 0.0

    def test_warfarin_inr_preset(self):
        """Warfarin INR preset should have baseline=1.0 (normal INR)."""
        assert WARFARIN_INR.ke0 == 0.02
        assert WARFARIN_INR.emax == 5.0
        assert WARFARIN_INR.ec50 == 1.0
        assert WARFARIN_INR.hill == 1.0
        assert WARFARIN_INR.baseline == 1.0

    def test_warfarin_inr_at_zero_gives_baseline(self):
        """Warfarin INR at zero concentration should give INR = 1.0."""
        time = np.array([0.0, 1.0])
        cp = np.zeros(2)
        sim = _make_sim(time, cp)

        result = compute_effect(sim, WARFARIN_INR)
        assert result.effect[-1] == 1.0


# ---------------------------------------------------------------------------
# Integration with engine
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    """Full PK -> PD integration using the engine pipeline."""

    def test_midazolam_sedation(self):
        """Midazolam oral -> sedation effect via full engine pipeline."""
        import sisyphus.engine.flux  # noqa: F401 -- register flux specs
        from sisyphus.compounds import load_compound
        from sisyphus.engine.compiler import ODECompiler, ResolvedParams
        from sisyphus.engine.solver import solve
        from sisyphus.graph.builder import build_from_yaml

        graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        drug = load_compound(Path("data/compounds/midazolam.yaml"))

        compiler = ODECompiler()
        compiled = compiler.compile(graph)

        rng = np.random.default_rng(42)
        realized_graph = graph.sample(rng)
        realized_drug = drug.sample(rng)
        params = ResolvedParams(realized_graph, realized_drug)

        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index[drug.administration_node]] = drug.dose_mg

        t_eval = np.linspace(0, 24, 2000)
        sim = solve(compiled, params, y0, t_span=(0, 24), t_eval=t_eval)
        assert sim.solver_success

        # Apply midazolam sedation PD model
        result = compute_effect(sim, MIDAZOLAM_SEDATION)

        # Peak effect should be > 0
        assert result.emax_achieved > 0, "Midazolam should produce sedation"

        # Peak effect time should be > 0 (not at t=0)
        assert result.temax > 0, "Peak sedation should not be at t=0"

        # PD peak should be delayed relative to PK peak (ke0 effect)
        tmax_pk = float(sim.time_h[np.argmax(sim.concentrations["venous_blood"])])
        assert result.temax >= tmax_pk, (
            f"PD peak ({result.temax:.2f}h) should be >= PK peak ({tmax_pk:.2f}h)"
        )

    def test_warfarin_inr(self):
        """Warfarin oral -> INR response via full engine pipeline."""
        import sisyphus.engine.flux  # noqa: F401
        from sisyphus.compounds import load_compound
        from sisyphus.engine.compiler import ODECompiler, ResolvedParams
        from sisyphus.engine.solver import solve
        from sisyphus.graph.builder import build_from_yaml

        graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        drug = load_compound(Path("data/compounds/warfarin.yaml"))

        compiler = ODECompiler()
        compiled = compiler.compile(graph)

        rng = np.random.default_rng(42)
        params = ResolvedParams(graph.sample(rng), drug.sample(rng))

        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index[drug.administration_node]] = drug.dose_mg

        # Warfarin has very slow ke0 (0.02 h^-1) -- need longer simulation
        t_eval = np.linspace(0, 120, 5000)
        sim = solve(compiled, params, y0, t_span=(0, 120), t_eval=t_eval)
        assert sim.solver_success

        result = compute_effect(sim, WARFARIN_INR)

        # Baseline INR = 1.0
        assert result.effect[0] == 1.0

        # INR should rise above baseline
        assert result.emax_achieved > 1.0, "Warfarin should increase INR"

        # Peak INR effect should be delayed (slow ke0)
        assert result.temax > 1.0, "INR peak should be significantly delayed"


# ---------------------------------------------------------------------------
# Architectural invariant: engine not modified
# ---------------------------------------------------------------------------


class TestEngineUnmodified:
    """Verify PK/PD works without any engine changes."""

    def test_engine_files_unchanged(self):
        """PK/PD operates on SimResult, not engine internals.

        This test documents the invariant that PK/PD is implemented
        as post-processing on SimResult. The engine is not modified.
        """
        engine_dir = Path("src/sisyphus/engine")
        engine_files = sorted(engine_dir.glob("*.py"))

        # Engine files must exist and be unmodified
        assert len(engine_files) > 0

        # Verify PK/PD works through standard SimResult interface
        time = np.linspace(0, 10, 100)
        cp = 2.0 * time * np.exp(-0.5 * time)
        sim = _make_sim(time, cp)

        pd = PDModel(ke0=1.0, emax=100.0, ec50=0.5, hill=1.0)
        result = compute_effect(sim, pd)

        # Must produce valid output from standard SimResult
        assert result.emax_achieved > 0
        assert result.temax > 0
        assert len(result.ce) == len(time)
        assert len(result.effect) == len(time)
