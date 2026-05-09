"""Tests for the SBI engine simulator wrapper."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from sisyphus.sbi.priors import (  # noqa: E402
    N_THETA,
    THETA_NAMES,
    _logit,
    _sigmoid,
    build_box_prior,
)
from sisyphus.sbi.simulator import EngineSimulator, apply_theta_to_drug  # noqa: E402

# Small, well-characterized drug for fast tests (morphine, holdout member).
MORPHINE_SMILES = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
MORPHINE_DOSE_MG = 30.0


@pytest.fixture(scope="module")
def simulator() -> EngineSimulator:
    return EngineSimulator.for_drug(MORPHINE_SMILES, MORPHINE_DOSE_MG, route="oral")


def test_prior_shape():
    prior = build_box_prior()
    assert prior.n_dim == N_THETA
    assert prior.names == THETA_NAMES
    assert prior.low.shape == (N_THETA,)
    assert prior.high.shape == (N_THETA,)
    assert np.all(prior.low < prior.high)


def test_prior_sampling_inside_box():
    prior = build_box_prior()
    rng = np.random.default_rng(0)
    theta = prior.sample_numpy(1000, rng)
    assert theta.shape == (1000, N_THETA)
    assert np.all(theta >= prior.low - 1e-12)
    assert np.all(theta <= prior.high + 1e-12)


def test_apply_theta_preserves_drug_identity(simulator: EngineSimulator):
    """Overriding theta must not touch non-ADME drug identity fields."""
    # theta[1] is now logit(fup); logit(0.42) ≈ -0.323
    theta = np.array([0.3, _logit(0.42), -0.1])
    overridden = apply_theta_to_drug(
        simulator.nominal_drug, theta, simulator.nominal_peff
    )
    assert overridden.name == simulator.nominal_drug.name
    assert overridden.smiles == simulator.nominal_drug.smiles
    assert overridden.dose_mg == simulator.nominal_drug.dose_mg
    assert overridden.route == simulator.nominal_drug.route
    assert overridden.compound_type == simulator.nominal_drug.compound_type
    assert overridden.mw == simulator.nominal_drug.mw


def test_apply_theta_changes_adme_correctly(simulator: EngineSimulator):
    # theta[1] is logit(fup); logit(0.25) ≈ -1.099
    logit_025 = _logit(0.25)
    theta = np.array([0.5, logit_025, 0.2])
    overridden = apply_theta_to_drug(
        simulator.nominal_drug, theta, simulator.nominal_peff
    )
    # fup should be sigmoid(logit(0.25)) ≈ 0.25
    assert abs(overridden.fup.mean - 0.25) < 1e-6
    # Peff should be 10^0.2 * nominal
    expected_peff = (10**0.2) * simulator.nominal_peff
    assert abs(overridden.peff.mean - expected_peff) < 1e-6
    # Each enzyme_affinity should scale by 10^0.5 vs nominal
    for tag, dist in overridden.enzyme_affinity.items():
        nom = simulator.nominal_drug.enzyme_affinity[tag]
        ratio = dist.mean / nom.mean if nom.mean > 0 else 1.0
        assert abs(ratio - 10**0.5) < 1e-6


def test_simulate_single_deterministic(simulator: EngineSimulator):
    """Same (theta, seed) must yield the same log10_cmax."""
    # logit(0.65) ≈ 0.619
    theta = np.array([0.0, _logit(0.65), 0.0])
    a = simulator.simulate_single(theta, seed=7)
    b = simulator.simulate_single(theta, seed=7)
    assert np.isfinite(a)
    assert a == b


def test_simulate_single_seed_changes_result(simulator: EngineSimulator):
    """Different seeds must yield different results (variance via Distribution CVs)."""
    theta = np.array([0.0, _logit(0.65), 0.0])
    a = simulator.simulate_single(theta, seed=1)
    b = simulator.simulate_single(theta, seed=2)
    assert np.isfinite(a) and np.isfinite(b)
    assert a != b


def test_simulate_batch_shape(simulator: EngineSimulator):
    prior = build_box_prior()
    rng = np.random.default_rng(0)
    thetas = prior.sample_numpy(5, rng)
    out = simulator.simulate_batch(thetas, seed=100)
    assert out.shape == (5, 1)
    assert np.all(np.isfinite(out))  # morphine range shouldn't fail


def test_torch_simulator_interface(simulator: EngineSimulator):
    import torch

    sim_fn = simulator.torch_simulator()
    # theta[1] is now logit(fup): logit(0.5)=0.0, logit(0.2)≈-1.386
    theta_t = torch.tensor(
        [[0.0, _logit(0.5), 0.0], [0.3, _logit(0.2), -0.2]],
        dtype=torch.float32,
    )
    out = sim_fn(theta_t)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 1)
    assert out.dtype == torch.float32


def test_theta_affects_output(simulator: EngineSimulator):
    """Large CLint increase must push Cmax down (more clearance)."""
    low_clint = np.array([-1.0, _logit(0.65), 0.0])
    high_clint = np.array([1.0, _logit(0.65), 0.0])
    lo = simulator.simulate_single(low_clint, seed=50)
    hi = simulator.simulate_single(high_clint, seed=50)
    assert np.isfinite(lo) and np.isfinite(hi)
    # High CLint should produce lower Cmax (log space: hi < lo)
    assert hi < lo


# --- Phase 2.0.5: logit(fup) reparameterization tests ---


def test_logit_sigmoid_roundtrip():
    """logit and sigmoid must be exact inverses over the physical fup range."""
    fup_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    for fup in fup_values:
        assert abs(_sigmoid(_logit(fup)) - fup) < 1e-9, f"roundtrip failed for fup={fup}"


def test_logit_sigmoid_known_values():
    """Check specific known logit/sigmoid values."""
    # logit(0.5) = 0
    assert abs(_logit(0.5)) < 1e-9
    # sigmoid(0) = 0.5
    assert abs(_sigmoid(0.0) - 0.5) < 1e-9
    # logit(0.01) ≈ -4.595
    assert abs(_logit(0.01) - (-4.595)) < 0.01
    # logit(0.99) ≈ +4.595
    assert abs(_logit(0.99) - 4.595) < 0.01


def test_prior_logit_fup_bounds():
    """Prior bounds for theta[1] should span logit(0.01) to logit(0.99)."""
    prior = build_box_prior()
    # theta[1] bounds should be logit(0.01) to logit(0.99)
    assert abs(prior.low[1] - _logit(0.01)) < 1e-3
    assert abs(prior.high[1] - _logit(0.99)) < 1e-3
    # theta_names should say "logit_fup"
    assert prior.names[1] == "logit_fup"


def test_prior_samples_recover_physical_fup():
    """Samples from the prior, mapped through sigmoid, should span [0.01, 0.99]."""
    prior = build_box_prior()
    rng = np.random.default_rng(42)
    theta = prior.sample_numpy(10000, rng)
    logit_fup = theta[:, 1]
    fup = _sigmoid(logit_fup)
    assert fup.min() < 0.02, "min fup should be near 0.01"
    assert fup.max() > 0.98, "max fup should be near 0.99"
    # Median should be near 0.5 (uniform in logit space, sigmoid is symmetric)
    assert abs(np.median(fup) - 0.5) < 0.05


def test_apply_theta_logit_fup_extreme_values(simulator: EngineSimulator):
    """Very low and very high logit(fup) values must produce physical fup."""
    # Very low fup: logit(0.01) ≈ -4.595
    theta_low = np.array([0.0, _logit(0.01), 0.0])
    drug_low = apply_theta_to_drug(
        simulator.nominal_drug, theta_low, simulator.nominal_peff
    )
    assert 0.009 < drug_low.fup.mean < 0.015

    # Very high fup: logit(0.99) ≈ 4.595
    theta_high = np.array([0.0, _logit(0.99), 0.0])
    drug_high = apply_theta_to_drug(
        simulator.nominal_drug, theta_high, simulator.nominal_peff
    )
    assert 0.98 < drug_high.fup.mean <= 1.0
