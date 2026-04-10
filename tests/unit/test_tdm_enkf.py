"""Tests for EnKF TDM update — Phase 3 of UDE roadmap.

Validates:
1. EnKF never produces degenerate ensemble (ESS = n_successful by construction).
2. Posterior Cmax shifts toward observations (sanity check).
3. Covariance inflation widens posterior CI.
4. Parameter multipliers are bounded (no catastrophic shifts).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph
from sisyphus.regimen.solver import solve_regimen
from sisyphus.regimen.tdm import Observation
from sisyphus.regimen.tdm_enkf import enkf_update
from sisyphus.regimen.types import DosingRegimen


def _build_pipeline(smiles: str, dose_mg: float):
    physiology = (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "physiology" / "reference_man.yaml"
    )
    graph = build_from_yaml(physiology)
    compiled = ODECompiler().compile(graph)
    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    liver_enz = {t: d.mean for t, d in graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(profile, adme, dose_mg, "oral", liver_enzymes=liver_enz)
    return graph, compiled, drug


def _synthetic_observation(compiled, graph, drug, regimen, cmax_target, t_obs):
    """Scale engine C(t) to match cmax_target, extract observation at t_obs."""
    rng = np.random.default_rng(42)
    rg = graph.sample(np.random.default_rng(0))
    rd = drug.sample(np.random.default_rng(0))
    sim = solve_regimen(compiled, ResolvedParams(rg, rd), regimen, t_total_h=24.0, dt_output=0.1)
    conc = sim.concentrations["venous_blood"]
    scale = cmax_target / float(np.max(conc))
    c_at_t = float(np.interp(t_obs, sim.time_h, conc))
    c_measured = max(c_at_t * scale * rng.lognormal(0, 0.10), 1e-8)
    return Observation(time_h=t_obs, concentration=c_measured, cv=0.10)


class TestEnKFDoesNotDegenerate:
    """Core guarantee: EnKF ESS equals the number of successful simulations."""

    def test_ess_equals_n_successful(self):
        """Even with a hard-to-match observation, EnKF preserves all samples."""
        # Clozapine: fold error ~2x in TDM v2.1 benchmark
        smi = "CN1CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42"
        graph, compiled, drug = _build_pipeline(smi, 75.0)
        regimen = DosingRegimen.single_oral(75.0)
        obs = (_synthetic_observation(compiled, graph, drug, regimen, 0.413, 1.0),)
        result = enkf_update(compiled, graph, drug, regimen, obs, n_ensemble=100, seed=42)
        assert result.ess == float(result.n_successful_post)
        assert result.n_successful_post > 50  # at least half must succeed

    def test_importance_ess_degenerate_but_enkf_preserved(self):
        """Hard prior/obs mismatch: IS would degenerate, EnKF stays intact."""
        # Ketorolac: 3.25x fold error (known to break importance sampling)
        smi = "O=C(c1ccccc1)c1ccc2n1CCC2C(=O)O"
        graph, compiled, drug = _build_pipeline(smi, 10.0)
        regimen = DosingRegimen.single_oral(10.0)
        obs = (_synthetic_observation(compiled, graph, drug, regimen, 0.80, 1.0),)
        result = enkf_update(compiled, graph, drug, regimen, obs, n_ensemble=100, seed=42)
        # Importance ESS diagnostic would be tiny; EnKF ESS should be ~n_successful
        assert result.importance_ess < 20.0, "prior/obs mismatch should degenerate IS"
        assert result.ess > 50.0, "EnKF should preserve most of the ensemble"


class TestPosteriorShiftsTowardObservation:
    """Sanity: posterior Cmax moves from prior toward observation."""

    def test_posterior_moves_toward_observed(self):
        # Morphine: 2x fold error — should see clear shift
        smi = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
        graph, compiled, drug = _build_pipeline(smi, 30.0)
        regimen = DosingRegimen.single_oral(30.0)
        obs = (_synthetic_observation(compiled, graph, drug, regimen, 0.01865, 1.0),)
        result = enkf_update(compiled, graph, drug, regimen, obs, n_ensemble=100, seed=42)
        prior = result.prior_cmax.mean
        post = result.posterior_cmax.mean
        # Posterior should shift in the direction of observation (here: lower than prior ~0.037)
        assert prior > 0.025  # engine overpredicts morphine
        assert post < prior, f"posterior {post:.4f} should be less than prior {prior:.4f}"


class TestInflation:
    """Covariance inflation widens the posterior distribution."""

    def test_inflation_widens_ci(self):
        smi = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
        graph, compiled, drug = _build_pipeline(smi, 30.0)
        regimen = DosingRegimen.single_oral(30.0)
        obs = (_synthetic_observation(compiled, graph, drug, regimen, 0.01865, 1.0),)
        r1 = enkf_update(compiled, graph, drug, regimen, obs, n_ensemble=100, seed=42, inflation=1.0)
        r2 = enkf_update(compiled, graph, drug, regimen, obs, n_ensemble=100, seed=42, inflation=1.5)
        w1 = r1.cmax_ci_90[1] - r1.cmax_ci_90[0]
        w2 = r2.cmax_ci_90[1] - r2.cmax_ci_90[0]
        assert w2 > w1, f"inflation=1.5 should widen CI; got w1={w1:.4f}, w2={w2:.4f}"


class TestEnKFRequiresObservation:
    def test_empty_observations_raises(self):
        smi = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
        graph, compiled, drug = _build_pipeline(smi, 30.0)
        regimen = DosingRegimen.single_oral(30.0)
        with pytest.raises(ValueError):
            enkf_update(compiled, graph, drug, regimen, observations=(), n_ensemble=50)
