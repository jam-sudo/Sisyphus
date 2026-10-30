"""Exact SBC rejection remains deterministic when proposals run in parallel."""

import numpy as np

from sisyphus.sbi.importance import sample_exact_rejection_parallel
from sisyphus.sbi.priors import PRIOR_HIGH, PRIOR_LOW


def test_parallel_exact_rejection_preserves_conditional_box_prior():
    midpoint = (PRIOR_LOW[0] + PRIOR_HIGH[0]) / 2

    def log_draw(theta: np.ndarray, rng: np.random.Generator) -> float:
        return 0.0 if theta[0] > midpoint else -np.inf

    kwargs = {"n_draws": 200, "seed": 17, "max_log_likelihood": 0.0,
              "workers": 2, "max_attempts": 4096}
    samples, attempts = sample_exact_rejection_parallel(log_draw, **kwargs)
    repeat, repeat_attempts = sample_exact_rejection_parallel(log_draw, **kwargs)
    assert np.array_equal(samples, repeat)
    assert attempts == repeat_attempts
    assert np.all(samples[:, 0] > midpoint)
    assert np.all(samples <= PRIOR_HIGH)
    assert np.all(samples >= PRIOR_LOW)
    conditional_mean = (midpoint + PRIOR_HIGH[0]) / 2
    assert abs(samples[:, 0].mean() - conditional_mean) < 0.1
