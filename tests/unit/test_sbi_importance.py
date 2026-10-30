"""Checks for the pseudo-marginal SBI correction."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sisyphus.sbi import importance  # noqa: E402
from sisyphus.sbi.importance import (  # noqa: E402
    CorrectedSBCPosterior,
    LowImportanceESS,
    sample_corrected,
)
from sisyphus.sbi.priors import PRIOR_HIGH, PRIOR_LOW  # noqa: E402


class UniformPosterior:
    def sample(self, shape, *, x, show_progress_bars):
        rng = np.random.default_rng(1)
        return torch.as_tensor(
            rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(shape[0], 3)), dtype=torch.float32,
        )

    def log_prob(self, theta, *, x):
        return torch.full((len(theta),), -np.log(np.prod(PRIOR_HIGH - PRIOR_LOW)))


def test_corrected_sampler_moves_toward_likelihood_peak():
    posterior = UniformPosterior()

    def log_draw(theta, rng):
        return float(-0.5 * ((theta[0] - 0.7) / 0.2) ** 2)

    samples, ess = sample_corrected(
        posterior, np.zeros(1), log_draw,
        n_draws=1000, n_proposals=2000, n_nuisance=2, seed=7,
    )
    assert 0.55 < samples[:, 0].mean() < 0.8
    assert ess > 32


def test_corrected_sampler_rejects_degenerate_weights():
    with pytest.raises(LowImportanceESS):
        sample_corrected(
            UniformPosterior(), np.zeros(1),
            lambda theta, rng: -1000.0 * (theta[0] - 0.7) ** 2,
            n_draws=10, n_proposals=32, n_nuisance=1, seed=7,
        )


def test_flow_numerical_failure_uses_existing_fallback():
    class BrokenPosterior(UniformPosterior):
        def sample(self, shape, *, x, show_progress_bars):
            raise AssertionError("spline discriminant")

    with pytest.raises(LowImportanceESS, match="numerical failure"):
        sample_corrected(
            BrokenPosterior(), np.zeros(1), lambda theta, rng: 0.0,
            n_draws=10, n_proposals=64, n_nuisance=1,
        )


def test_direct_posterior_uses_raw_flow_proposal_density():
    class DirectPosteriorStub(UniformPosterior):
        posterior_estimator = object()

        def sample(self, shape, *, x, show_progress_bars, reject_outside_prior):
            assert reject_outside_prior is False
            return super().sample(shape, x=x, show_progress_bars=show_progress_bars)

        def log_prob(self, theta, *, x, norm_posterior):
            assert norm_posterior is False
            return super().log_prob(theta, x=x)

    draws, ess = sample_corrected(
        DirectPosteriorStub(), np.zeros(1), lambda theta, rng: 0.0,
        n_draws=10, n_proposals=100, n_nuisance=1,
    )
    assert draws.shape == (10, 3)
    assert ess == pytest.approx(100)


def test_corrected_sampler_excludes_flow_draws_outside_box_prior():
    class OutsidePosterior(UniformPosterior):
        def sample(self, shape, *, x, show_progress_bars):
            return torch.as_tensor(
                np.tile(PRIOR_HIGH + 1.0, (shape[0], 1)), dtype=torch.float32,
            )

    draws, ess = sample_corrected(
        OutsidePosterior(), np.zeros(1), lambda theta, rng: 0.0,
        n_draws=100, n_proposals=100, n_nuisance=1, seed=7, ess_floor=1,
    )
    assert np.all((draws >= PRIOR_LOW) & (draws <= PRIOR_HIGH))
    assert ess == pytest.approx(5)


def test_sbc_sampler_counts_exact_rejection_fallback(monkeypatch):
    class Simulator:
        obs_sigma_log10 = 0.2

        def simulate_single(self, theta, seed, add_noise=True):
            assert not add_noise
            return theta[0]

    def low_ess(*args, **kwargs):
        raise LowImportanceESS("forced")

    monkeypatch.setattr(importance, "sample_corrected", low_ess)
    sampler = CorrectedSBCPosterior(UniformPosterior(), Simulator(), seed=7)
    draws = sampler.sample((100,), x=torch.as_tensor([[0.7]]))
    assert sampler.rejection_count == 1
    assert 0.5 < draws[:, 0].mean() < 0.9


def test_sbc_forced_exact_rejection_is_recorded(monkeypatch):
    class Simulator:
        obs_sigma_log10 = 0.2

    def exact(log_draw, *, n_draws, seed, max_log_likelihood, workers):
        assert workers == 2
        return np.tile((PRIOR_LOW + PRIOR_HIGH) / 2, (n_draws, 1)), n_draws

    monkeypatch.setattr(importance, "sample_exact_rejection_parallel", exact)
    sampler = CorrectedSBCPosterior(
        UniformPosterior(), Simulator(), seed=7, exact_rejection_workers=2
    )
    draws = sampler.sample((5,), x=torch.as_tensor([[0.0]]))
    assert draws.shape == (5, 3)
    assert sampler.rejection_count == 1
    assert sampler.forced_exact_rejection_count == 1


def test_sbc_case_streams_are_independent_of_block_boundaries(monkeypatch):
    class Simulator:
        obs_sigma_log10 = 0.2

    seen = []

    def corrected(*args, seed, n_draws, **kwargs):
        seen.append((seed, torch.initial_seed()))
        return np.zeros((n_draws, 3)), 64.0

    monkeypatch.setattr(importance, "sample_corrected", corrected)
    full = CorrectedSBCPosterior(UniformPosterior(), Simulator(), seed=7)
    for _ in range(6):
        full.sample((1,), x=torch.as_tensor([[0.0]]))
    unsplit = seen[:]
    seen.clear()
    block = CorrectedSBCPosterior(UniformPosterior(), Simulator(), seed=7, case_start=5)
    block.sample((1,), x=torch.as_tensor([[0.0]]))
    assert seen == [unsplit[5]]
    assert len(set(unsplit)) == 6


def test_time_aware_sampler_accepts_single_dose_context(monkeypatch):
    class Simulator:
        nominal_drug = type("Drug", (), {"dose_mg": 10.0})()

    def corrected(*args, **kwargs):
        return np.zeros((2, 3)), 64.0

    monkeypatch.setattr(importance, "sample_corrected", corrected)
    context = np.zeros(24, dtype=np.float32)
    context[:3] = [1 / 24, -1.0, 1.0]
    context[11] = 1 / 3
    sampler = CorrectedSBCPosterior(
        UniformPosterior(), Simulator(), seed=7, multi_observation=True,
    )
    draws = sampler.sample((2,), x=torch.as_tensor(context[None, :]))
    assert draws.shape == (2, 3)


def test_time_aware_likelihood_uses_joint_observations_per_nuisance_draw(monkeypatch):
    from types import SimpleNamespace

    from sisyphus.sbi import simulator as simulator_module
    from sisyphus.sbi.multi_observation import ASSAY_SIGMA_LOG10

    class Simulator:
        nominal_drug = type("Drug", (), {"dose_mg": 10.0})()
        nominal_peff = 1.0
        compiled = object()
        obs_node = "venous_blood"

        def _sample_graph(self, rng):
            return object()

    observed = []
    solves = []

    def corrected(posterior, x, log_draw, **kwargs):
        observed.append(log_draw(np.zeros(3), np.random.default_rng(1)))
        return np.zeros((2, 3)), 64.0

    def solve(*args, **kwargs):
        solves.append(1)
        return SimpleNamespace(
            solver_success=True, time_h=np.array([0.0, 1.0, 2.0]),
            concentrations={"venous_blood": np.array([0.1, 0.2, 0.4])},
        )

    monkeypatch.setattr(importance, "sample_corrected", corrected)
    monkeypatch.setattr(importance, "solve_regimen", solve)
    monkeypatch.setattr(importance, "ResolvedParams", lambda graph, drug: object())
    monkeypatch.setattr(simulator_module, "apply_theta_to_drug", lambda *args: SimpleNamespace(
        sample=lambda rng: object(),
    ))
    context = np.zeros(24, dtype=np.float32)
    context[:6] = [1 / 24, np.log10(0.2), 1.0, 2 / 24, np.log10(0.4), 1.0]
    context[11] = 1 / 3
    sampler = CorrectedSBCPosterior(
        UniformPosterior(), Simulator(), seed=7, multi_observation=True,
    )
    sampler.sample((2,), x=torch.as_tensor(context[None, :]))
    expected = 2 * (-np.log(ASSAY_SIGMA_LOG10) - 0.5 * np.log(2 * np.pi))
    assert len(solves) == 1
    assert observed[0] == pytest.approx(expected, abs=1e-5)
