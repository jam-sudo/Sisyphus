"""Tests for TDM Bayesian update (regimen/tdm.py)."""

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import build_from_yaml
from sisyphus.graph.types import FlowEdge, Node
from sisyphus.regimen.tdm import (
    Observation,
    TDMResult,
    _interpolate_concentration,
    _log_likelihood,
    bayesian_update,
)
from sisyphus.regimen.types import DosingRegimen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def physiology():
    """Load reference_man graph and compile ODE."""
    import pathlib
    yaml_path = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "physiology" / "reference_man.yaml"
    graph = build_from_yaml(yaml_path)
    compiled = ODECompiler().compile(graph)
    return graph, compiled


# ---------------------------------------------------------------------------
# Observation tests
# ---------------------------------------------------------------------------

class TestObservation:
    def test_basic_creation(self):
        obs = Observation(time_h=12.0, concentration=2.5)
        assert obs.time_h == 12.0
        assert obs.concentration == 2.5
        assert obs.cv == 0.10
        assert obs.node == "venous_blood"

    def test_sigma(self):
        obs = Observation(time_h=12.0, concentration=2.5, cv=0.10)
        assert abs(obs.sigma - 0.25) < 1e-10

    def test_sigma_zero_concentration(self):
        obs = Observation(time_h=12.0, concentration=0.0, cv=0.10)
        assert obs.sigma > 0  # should not be zero


# ---------------------------------------------------------------------------
# Log-likelihood tests
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def test_perfect_prediction_has_highest_likelihood(self, physiology):
        """When pred == obs, log-likelihood should be maximized."""
        graph, compiled = physiology
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.chemistry import compute_profile
        from sisyphus.predict.ivive import build_drug_on_graph
        from sisyphus.regimen.solver import solve_regimen

        smiles = "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"  # ciprofloxacin
        profile = compute_profile(smiles)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, 500.0, "oral")
        regimen = DosingRegimen.single_oral(500.0)

        rng = np.random.default_rng(42)
        sim = solve_regimen(
            compiled, ResolvedParams(graph.sample(rng), drug.sample(rng)),
            regimen, t_total_h=24.0,
        )

        # Observation exactly matching prediction
        t_obs = 4.0
        c_pred = float(np.interp(t_obs, sim.time_h, sim.concentrations["venous_blood"]))

        obs_perfect = (Observation(time_h=t_obs, concentration=c_pred, cv=0.10),)
        obs_far = (Observation(time_h=t_obs, concentration=c_pred * 10.0, cv=0.10),)

        ll_perfect = _log_likelihood(sim, obs_perfect)
        ll_far = _log_likelihood(sim, obs_far)

        assert ll_perfect > ll_far


# ---------------------------------------------------------------------------
# Bayesian update tests
# ---------------------------------------------------------------------------

class TestBayesianUpdate:
    def test_requires_observations(self, physiology):
        graph, compiled = physiology
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.chemistry import compute_profile
        from sisyphus.predict.ivive import build_drug_on_graph

        smiles = "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"
        profile = compute_profile(smiles)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, 500.0, "oral")
        regimen = DosingRegimen.single_oral(500.0)

        with pytest.raises(ValueError, match="observation"):
            bayesian_update(compiled, graph, drug, regimen, observations=[])

    def test_posterior_cv_less_than_prior(self, physiology):
        """Core TDM property: observations should reduce uncertainty."""
        graph, compiled = physiology
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.chemistry import compute_profile
        from sisyphus.predict.ivive import build_drug_on_graph
        from sisyphus.regimen.solver import solve_regimen

        smiles = "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"
        profile = compute_profile(smiles)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, 500.0, "oral")
        regimen = DosingRegimen.single_oral(500.0)

        # Generate a "true" observation
        rng = np.random.default_rng(999)
        sim = solve_regimen(
            compiled, ResolvedParams(graph.sample(rng), drug.sample(rng)),
            regimen, t_total_h=24.0,
        )
        c_at_6h = float(np.interp(6.0, sim.time_h, sim.concentrations["venous_blood"]))

        result = bayesian_update(
            compiled, graph, drug, regimen,
            observations=[Observation(time_h=6.0, concentration=c_at_6h, cv=0.10)],
            n_prior=200,
            seed=42,
        )

        assert result.n_successful > 0
        assert result.ess > 1.0
        # Core assertion: posterior CV < prior CV
        assert result.posterior_cmax.cv < result.prior_cmax.cv

    def test_returns_tdm_result(self, physiology):
        graph, compiled = physiology
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.chemistry import compute_profile
        from sisyphus.predict.ivive import build_drug_on_graph

        smiles = "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"
        profile = compute_profile(smiles)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, 500.0, "oral")
        regimen = DosingRegimen.single_oral(500.0)

        result = bayesian_update(
            compiled, graph, drug, regimen,
            observations=[Observation(time_h=6.0, concentration=1.0, cv=0.10)],
            n_prior=50,
            seed=42,
        )

        assert isinstance(result, TDMResult)
        assert result.n_prior == 50
        assert len(result.weights) == result.n_successful
        assert len(result.log_likelihoods) == result.n_successful
        assert result.prior_cmax.mean > 0
        assert result.posterior_cmax.mean > 0
