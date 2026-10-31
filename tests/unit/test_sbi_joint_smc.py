"""Joint nuisance-stream and tempered SMC checks."""

import numpy as np

from sisyphus.sbi import joint_smc
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


def test_low_ancestor_guard_repeats_with_four_times_the_particles(monkeypatch):
    calls = []

    def run_once(*args, n_particles, rejuvenation_sweeps, **kwargs):
        calls.append((n_particles, rejuvenation_sweeps))
        return np.zeros((500, 3)), {
            "particles": n_particles,
            "unique_initial_ancestors": 36 if len(calls) == 1 else 140,
            "stages": [{"beta": 1.0}],
            "simulator_evaluations": 10 * n_particles,
            "elapsed_s": float(n_particles),
        }

    monkeypatch.setattr(joint_smc, "_sample_joint_smc_once", run_once)
    _, diagnostics = sample_joint_smc(lambda theta, stream: 0.0, n_draws=500, seed=1)
    assert calls == [(1000, 2), (4000, 3)]
    assert diagnostics["ancestor_guard_passed"] is True
    assert diagnostics["first_attempt"]["unique_initial_ancestors"] == 36
    assert diagnostics["simulator_evaluations"] == 50000
    calls.clear()
    _, diagnostics = sample_joint_smc(
        lambda theta, stream: 0.0, n_draws=500, seed=1, n_particles=4000,
    )
    assert calls == [(4000, 2)]
    assert diagnostics["ancestor_guard_passed"] is False
