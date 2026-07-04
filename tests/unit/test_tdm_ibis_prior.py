"""Tests for the IBIS MCMC-rejuvenation prior term (tdm_ibis).

The rejuvenation kernel must target the partial posterior ``L * pi_0``, not the
likelihood alone. These tests pin the prior-density helpers that supply the
missing ``pi_0`` term: consistency with the prior the particles were sampled
from, the Fenton-Wilkinson sum-of-lognormals approximation for CLint_total, and
that the prior actually penalizes proposals that drift away from it.
"""

from __future__ import annotations

import math
import types

import numpy as np

from sisyphus.core import Distribution
from sisyphus.regimen import tdm_ibis as m


def _drug(fup=0.1, fup_cv=0.3, peff=5e-4, peff_cv=0.5, enzymes=None):
    """Minimal distributional drug-like object for the prior helpers."""
    if enzymes is None:
        enzymes = {"CYP3A4": Distribution(mean=2.0, cv=0.8)}
    return types.SimpleNamespace(
        fup=Distribution(fup, fup_cv),
        peff=Distribution(peff, peff_cv),
        enzyme_affinity=enzymes,
    )


def test_lognormal_log_params_matches_distribution_sampling():
    """(mu_ln, sigma_ln) must match the moments of log(X) for X ~ the same
    Distribution the IBIS particles are drawn from, else the MCMC targets a
    different prior than the importance weights."""
    d = Distribution(mean=0.3, cv=0.4)
    mu, sigma = m._lognormal_log_params(d.mean, d.cv)
    rng = np.random.default_rng(0)
    logs = np.log([d.sample(rng) for _ in range(200_000)])
    assert abs(logs.mean() - mu) < 5e-3
    assert abs(logs.std() - sigma) < 5e-3


def test_lognormal_log_params_undefined_for_nonpositive_mean():
    assert m._lognormal_log_params(0.0, 0.5) is None
    assert m._lognormal_log_params(-1.0, 0.5) is None


def test_lognormal_log_params_floors_sigma_for_deterministic_prior():
    """cv=0 (deterministic) must not produce sigma_ln=0 (a delta the density
    code cannot represent); it is floored so the coordinate is pinned."""
    mu, sigma = m._lognormal_log_params(2.0, 0.0)
    assert sigma >= m._PRIOR_SIGMA_LN_FLOOR
    assert math.isfinite(mu)


def test_prior_params_single_enzyme_is_exact_lognormal():
    """Fenton-Wilkinson of a single lognormal is that lognormal exactly."""
    drug = _drug(enzymes={"CYP3A4": Distribution(mean=2.0, cv=0.8)})
    _, clint, _ = m._prior_log_params_3d(drug)
    assert np.allclose(clint, m._lognormal_log_params(2.0, 0.8))


def test_prior_params_fenton_wilkinson_two_enzymes():
    """Sum of two independent lognormals is moment-matched: E[S]=sum of means,
    Var[S]=sum of variances."""
    drug = _drug(enzymes={
        "A": Distribution(2.0, 0.8),
        "B": Distribution(3.0, 1.0),
    })
    _, clint, _ = m._prior_log_params_3d(drug)
    e_sum = 5.0
    var_sum = (0.8 * 2.0) ** 2 + (1.0 * 3.0) ** 2
    cv_sum = math.sqrt(var_sum) / e_sum
    assert np.allclose(clint, m._lognormal_log_params(e_sum, cv_sum))


def test_prior_params_no_enzymes_gives_flat_clint():
    drug = _drug(enzymes={})
    fup, clint, peff = m._prior_log_params_3d(drug)
    assert clint is None          # CLint_total has no prior when no enzymes
    assert fup is not None
    assert peff is not None


def test_log_prior_peaks_at_the_prior_mode():
    drug = _drug()
    prior = m._prior_log_params_3d(drug)
    mode = np.array([prior[0][0], prior[1][0], prior[2][0]])
    lp_mode = m._log_prior_3d(mode, prior)
    for shift in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0])):
        assert m._log_prior_3d(mode + shift, prior) < lp_mode


def test_log_prior_penalizes_outward_moves_symmetrically():
    """The term the fix adds to log_alpha (proposed_lp - current_lp): a move
    away from the prior mode lowers the acceptance ratio; a move back raises it."""
    drug = _drug()
    prior = m._prior_log_params_3d(drug)
    mode = np.array([prior[0][0], prior[1][0], prior[2][0]])
    outward = mode + np.array([0.8, 0.8, 0.8])
    farther = mode + np.array([1.6, 1.6, 1.6])
    # farther from the mode => even lower prior density
    assert m._log_prior_3d(farther, prior) < m._log_prior_3d(outward, prior)
    # the prior contribution to log_alpha for an outward step is negative
    assert m._log_prior_3d(farther, prior) - m._log_prior_3d(outward, prior) < 0.0


def test_log_prior_flat_coords_contribute_zero():
    assert m._log_prior_3d(np.array([1.0, 2.0, 3.0]), (None, None, None)) == 0.0
