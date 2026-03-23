"""Tests for the ODE solver wrapper."""

from pathlib import Path

import numpy as np

import sisyphus.engine.flux  # noqa: F401 — register flux specs
from sisyphus.compounds import load_compound
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml


class TestSolver:
    def test_simple_system(self):
        """Build reference_man graph, solve with midazolam, verify solver success."""
        graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        drug = load_compound(Path("data/compounds/midazolam.yaml"))

        compiled = ODECompiler().compile(graph)
        rng = np.random.default_rng(42)
        realized_graph = graph.sample(rng)
        realized_drug = drug.sample(rng)
        params = ResolvedParams(realized_graph, realized_drug)

        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index[drug.administration_node]] = drug.dose_mg

        result = solve(compiled, params, y0, t_span=(0, 24))

        assert result.solver_success
        assert len(result.time_h) > 0
        assert "venous_blood" in result.concentrations
        assert result.concentrations["venous_blood"].shape == result.time_h.shape

    def test_mass_balance(self):
        """Mass balance error should be small."""
        graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        drug = load_compound(Path("data/compounds/caffeine.yaml"))

        compiled = ODECompiler().compile(graph)
        rng = np.random.default_rng(42)
        params = ResolvedParams(graph.sample(rng), drug.sample(rng))

        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index[drug.administration_node]] = drug.dose_mg

        result = solve(compiled, params, y0, t_span=(0, 24))
        assert result.mass_balance_error < 1e-4, f"MBE = {result.mass_balance_error}"

    def test_concentration_positive(self):
        """All concentrations should be non-negative."""
        graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        drug = load_compound(Path("data/compounds/caffeine.yaml"))

        compiled = ODECompiler().compile(graph)
        rng = np.random.default_rng(42)
        params = ResolvedParams(graph.sample(rng), drug.sample(rng))

        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index[drug.administration_node]] = drug.dose_mg

        result = solve(compiled, params, y0, t_span=(0, 24))
        for name, conc in result.concentrations.items():
            assert np.all(conc >= -1e-10), f"Negative concentration in {name}: min={conc.min()}"
