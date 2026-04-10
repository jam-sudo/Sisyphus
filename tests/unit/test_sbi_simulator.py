"""Tests for the SBI engine simulator wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.sbi.priors import build_box_prior, N_THETA, THETA_NAMES
from sisyphus.sbi.simulator import EngineSimulator, apply_theta_to_drug

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
    theta = np.array([0.3, 0.42, -0.1])
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
    theta = np.array([0.5, 0.25, 0.2])
    overridden = apply_theta_to_drug(
        simulator.nominal_drug, theta, simulator.nominal_peff
    )
    # fup should be theta[1]
    assert abs(overridden.fup.mean - 0.25) < 1e-9
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
    theta = np.array([0.0, 0.65, 0.0])
    a = simulator.simulate_single(theta, seed=7)
    b = simulator.simulate_single(theta, seed=7)
    assert np.isfinite(a)
    assert a == b


def test_simulate_single_seed_changes_result(simulator: EngineSimulator):
    """Different seeds must yield different results (variance via Distribution CVs)."""
    theta = np.array([0.0, 0.65, 0.0])
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
    theta_t = torch.tensor([[0.0, 0.5, 0.0], [0.3, 0.2, -0.2]], dtype=torch.float32)
    out = sim_fn(theta_t)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 1)
    assert out.dtype == torch.float32


def test_theta_affects_output(simulator: EngineSimulator):
    """Large CLint increase must push Cmax down (more clearance)."""
    low_clint = np.array([-1.0, 0.65, 0.0])
    high_clint = np.array([1.0, 0.65, 0.0])
    lo = simulator.simulate_single(low_clint, seed=50)
    hi = simulator.simulate_single(high_clint, seed=50)
    assert np.isfinite(lo) and np.isfinite(hi)
    # High CLint should produce lower Cmax (log space: hi < lo)
    assert hi < lo
