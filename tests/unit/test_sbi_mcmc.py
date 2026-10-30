"""Pseudo-marginal SBC fallback checks."""

import numpy as np
import pytest

from sisyphus.sbi.mcmc import mcmc_diagnostics, sample_pseudo_marginal_mcmc
from sisyphus.sbi.priors import PRIOR_HIGH, PRIOR_LOW


def test_mcmc_diagnostics_separate_mixed_and_stuck_chains():
    rng = np.random.default_rng(7)
    mixed = rng.normal(size=(4, 2000, 3))
    good = mcmc_diagnostics(mixed)
    assert max(good["rhat"]) < 1.03
    assert min(good["bulk_ess"]) > 500
    stuck = mixed * 0.01 + np.arange(4)[:, None, None]
    bad = mcmc_diagnostics(stuck)
    assert max(bad["rhat"]) > 1.1
    assert min(bad["bulk_ess"]) < 500


def test_pseudo_marginal_sampler_targets_likelihood_and_reports_diagnostics():
    torch = pytest.importorskip("torch")

    class UniformPosterior:
        def sample(self, shape, *, x, show_progress_bars):
            return torch.as_tensor(
                np.random.default_rng(int(torch.initial_seed())).uniform(
                    PRIOR_LOW, PRIOR_HIGH, size=(shape[0], 3),
                ), dtype=torch.float32,
            )

    samples, diagnostics = sample_pseudo_marginal_mcmc(
        UniformPosterior(), np.zeros(1),
        lambda theta, rng: float(-0.5 * ((theta[0] - 0.6) / 0.3) ** 2),
        n_draws=40, n_nuisance=1, seed=7,
        max_retained_steps=1000, min_bulk_ess=20, max_rhat=1.2,
    )
    assert samples.shape == (40, 3)
    assert 0.3 < samples[:, 0].mean() < 0.7
    assert diagnostics["chains"] == 4
    assert max(diagnostics["rhat"]) <= 1.2
    assert min(diagnostics["bulk_ess"]) >= 20
