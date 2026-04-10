"""Tests for SBI amortizer training wrapper.

These tests use small synthetic data to verify the training API works end
to end. Full validation on engine-simulated data lives in the POC
scripts (scripts/sbi_*.py) and docs/sbi_poc_results.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sbi")
pytest.importorskip("torch")

from sisyphus.sbi.amortizer import load_result, save_result, train_npe  # noqa: E402
from sisyphus.sbi.priors import PriorSpec, build_box_prior  # noqa: E402
from sisyphus.sbi.sbc import run_sbc  # noqa: E402


def _make_linear_synthetic(
    n: int = 2000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear generative model: x = 0.8*θ0 − 0.3*θ1 + 0.05*θ2 + N(0, 0.03).

    Priors: θ0 ~ U(−1,1), θ1 ~ U(0.01,1), θ2 ~ U(−0.5,0.5).
    Used as a cheap identifiability test for the amortizer.
    """
    rng = np.random.default_rng(seed)
    low = np.array([-1.0, 0.01, -0.5])
    high = np.array([1.0, 1.00, 0.5])
    theta = rng.uniform(low, high, size=(n, 3)).astype(np.float32)
    x = (
        0.8 * theta[:, 0]
        - 0.3 * theta[:, 1]
        + 0.05 * theta[:, 2]
        + rng.normal(0, 0.03, n)
    ).reshape(-1, 1).astype(np.float32)
    return theta, x, low, high


def test_train_npe_returns_working_posterior():
    theta, x, low, high = _make_linear_synthetic(n=2000, seed=1)
    result = train_npe(
        theta=theta,
        x=x,
        prior_low=low,
        prior_high=high,
        max_num_epochs=100,
        stop_after_epochs=20,
        seed=1,
    )
    assert result.n_train_samples == 2000

    # For x=+0.5, theta[0] should be pulled toward ~0.625-0.8 (with prior
    # mass + likelihood).  Check that posterior is clearly shifted off prior.
    x_obs = np.array([0.5], dtype=np.float32)
    samples = result.sample(x_obs, n=2000)
    assert samples.shape == (2000, 3)
    # θ0 posterior mean should be clearly positive for x=+0.5 (not at 0 prior mean)
    assert samples[:, 0].mean() > 0.3
    # Respect prior bounds
    assert samples[:, 0].min() >= low[0] - 0.05
    assert samples[:, 0].max() <= high[0] + 0.05
    assert samples[:, 1].min() >= low[1] - 0.05
    assert samples[:, 1].max() <= high[1] + 0.05


def test_npe_save_load_roundtrip(tmp_path: Path):
    theta, x, low, high = _make_linear_synthetic(n=1500, seed=2)
    result = train_npe(
        theta=theta,
        x=x,
        prior_low=low,
        prior_high=high,
        max_num_epochs=50,
        stop_after_epochs=15,
        seed=2,
    )

    out_path = tmp_path / "posterior.pt"
    save_result(result, out_path)
    assert out_path.exists()

    loaded = load_result(out_path)
    assert loaded.n_train_samples == result.n_train_samples
    np.testing.assert_array_equal(loaded.prior_low, result.prior_low)
    np.testing.assert_array_equal(loaded.prior_high, result.prior_high)

    # Sampling with the same seed should give deterministic results
    import torch
    torch.manual_seed(42)
    s1 = result.sample(np.array([0.3], dtype=np.float32), n=500)
    torch.manual_seed(42)
    s2 = loaded.sample(np.array([0.3], dtype=np.float32), n=500)
    # Samples may differ slightly if sbi internally randomizes, so check means/CIs are close.
    assert abs(s1[:, 0].mean() - s2[:, 0].mean()) < 0.1


def test_sbc_runs_on_linear_model():
    theta, x, low, high = _make_linear_synthetic(n=3000, seed=3)
    result = train_npe(
        theta=theta,
        x=x,
        prior_low=low,
        prior_high=high,
        max_num_epochs=100,
        stop_after_epochs=20,
        seed=3,
    )

    rng = np.random.default_rng(999)

    def prior_sample_fn(n: int) -> np.ndarray:
        return rng.uniform(low, high, size=(n, 3)).astype(np.float64)

    def simulator_fn(theta_vec: np.ndarray) -> float:
        noise = float(np.random.default_rng().normal(0, 0.03))
        return float(
            0.8 * theta_vec[0] - 0.3 * theta_vec[1] + 0.05 * theta_vec[2] + noise
        )

    sbc = run_sbc(
        posterior=result.posterior,
        simulator_fn=simulator_fn,
        prior_sample_fn=prior_sample_fn,
        n_calibration=100,
        n_posterior_samples=200,
        seed=4,
    )

    assert sbc.ranks.shape == (100, 3)
    assert sbc.ks_pvalues.shape == (3,)
    for level in (50, 80, 90, 95):
        assert level in sbc.coverage
        assert sbc.coverage[level].shape == (3,)
    # For a well-identified linear model, coverage should be roughly correct.
    # This is a sanity bound, not a tight calibration test.
    for level in (50, 90):
        nominal = level / 100.0
        assert np.all(np.abs(sbc.coverage[level] - nominal) < 0.25)


def test_prior_spec_as_torch_matches_numpy():
    prior = build_box_prior()
    import torch

    torch_prior = prior.as_torch()
    # sbi BoxUniform should have matching support
    assert torch.allclose(
        torch_prior.support.base_constraint.lower_bound,
        torch.tensor(prior.low, dtype=torch.float32),
    )
