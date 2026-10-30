"""Joint nuisance-stream and tempered SMC checks."""

import numpy as np

from sisyphus.sbi.joint_smc import _GaussianStream, sample_joint_smc


def test_gaussian_stream_preserves_nuisance_transforms():
    stream = _GaussianStream(np.zeros(5))
    assert stream.normal(2.0, 3.0) == 2.0
    assert stream.lognormal(0.0, 0.5) == 1.0
    assert stream.uniform(0.0, 4.0) == 2.0
    np.testing.assert_allclose(
        stream.multivariate_normal(np.array([1.0, 2.0]), np.eye(2)),
        [1.0, 2.0],
    )
    assert stream.offset == 5


def test_joint_smc_matches_toy_marginal_likelihood():
    # Integrating z analytically gives N(0.3; theta[0], sqrt(0.5**2 + 0.2**2)).
    def log_likelihood(theta, stream):
        prediction = theta[0] + stream.normal(0.0, 0.5)
        return float(-0.5 * ((0.3 - prediction) / 0.2) ** 2)

    draws, diagnostics = sample_joint_smc(
        log_likelihood, n_draws=200, seed=12, n_particles=300,
        workers=2, rejuvenation_sweeps=2,
    )
    grid = np.linspace(-1, 1, 2001)
    density = np.exp(-0.5 * ((0.3 - grid) / np.hypot(0.5, 0.2)) ** 2)
    expected_mean = np.trapezoid(grid * density, grid) / np.trapezoid(density, grid)
    assert abs(draws[:, 0].mean() - expected_mean) < 0.15
    assert diagnostics["nuisance_dimensions"] == 1
    assert diagnostics["stages"][-1]["beta"] == 1.0
    assert diagnostics["simulator_evaluations"] > 300
