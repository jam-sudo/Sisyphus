"""Tests for the MIPD inference core (engine-as-prior likelihood + SIR posterior).

Pure numpy — no engine dependency, so these are fast unit tests of the
Bayesian-update mechanism that turns measured observations into a posterior
over the engine's bioavailability latent F and the resulting PK.
"""
import numpy as np
import pytest

from sisyphus.mipd.core import (
    AnalyticForward,
    APrioriPK,
    FPrior,
    MeasuredAUC,
    MeasuredCmax,
    MeasuredF,
    Posterior,
    sir_posterior,
)

# a-priori engine state: Cmax0=1.0 at F_engine=0.5, AUC0=10.0
AP = APrioriPK(cmax0=1.0, auc0=10.0, f_engine=0.5)


def test_analytic_forward_scales_cmax_linearly_with_F():
    fwd = AnalyticForward(AP)
    state = fwd(np.array([0.5, 1.0]))  # F == f_engine, then 2x f_engine
    assert state["cmax"][0] == pytest.approx(1.0)  # F=f_engine reproduces Cmax0
    assert state["cmax"][1] == pytest.approx(2.0)  # doubling F doubles Cmax
    assert state["auc"][1] == pytest.approx(20.0)


def test_measured_F_loglik_peaks_at_observed_value():
    obs = MeasuredF(0.3, cv=0.1)
    ll = obs.log_likelihood({"f": np.array([0.1, 0.3, 0.9])})
    assert np.argmax(ll) == 1


def test_measured_cmax_loglik_peaks_where_forward_cmax_matches():
    state = AnalyticForward(AP)(np.array([0.25, 0.5, 1.0]))  # cmax -> 0.5,1.0,2.0
    ll = MeasuredCmax(2.0, cv=0.1).log_likelihood(state)
    assert np.argmax(ll) == 2


def test_measured_auc_loglik_peaks_where_forward_auc_matches():
    state = AnalyticForward(AP)(np.array([0.25, 0.5, 1.0]))  # auc -> 5,10,20
    ll = MeasuredAUC(5.0, cv=0.1).log_likelihood(state)
    assert np.argmax(ll) == 0


def test_fprior_samples_stay_in_unit_interval_and_center_near_f_engine():
    s = FPrior(f_engine=0.5, cv=1.0).sample(20000, np.random.default_rng(0))
    assert s.min() > 0.0
    assert s.max() <= 1.0
    assert np.median(s) == pytest.approx(0.5, abs=0.1)


def test_sir_with_tight_measured_F_concentrates_posterior_at_observed():
    rng = np.random.default_rng(0)
    post = sir_posterior(
        FPrior(0.5, 1.0), AnalyticForward(AP), [MeasuredF(0.2, cv=0.05)],
        n_samples=20000, rng=rng,
    )
    assert post.f.point == pytest.approx(0.2, abs=0.03)
    # Cmax follows the forward map: Cmax0 * F/F_engine = 1.0 * 0.2/0.5 = 0.4
    assert post.cmax.point == pytest.approx(0.4, rel=0.15)


def test_sir_empty_observations_returns_prior_centered_on_apriori():
    rng = np.random.default_rng(0)
    post = sir_posterior(
        FPrior(0.5, 0.5), AnalyticForward(AP), [], n_samples=20000, rng=rng,
    )
    assert post.cmax.point == pytest.approx(1.0, rel=0.15)  # ~ a-priori Cmax0
    lo, hi = post.cmax.ci90
    assert lo < post.cmax.point < hi  # interval brackets the point


def test_sir_reports_effective_sample_size_drop_under_informative_obs():
    rng = np.random.default_rng(0)
    diffuse = sir_posterior(FPrior(0.5, 1.0), AnalyticForward(AP), [], n_samples=20000, rng=rng)
    tight = sir_posterior(
        FPrior(0.5, 1.0), AnalyticForward(AP), [MeasuredF(0.05, cv=0.05)],
        n_samples=20000, rng=rng,
    )
    # a far, tight observation puts weight on few prior samples -> lower n_eff
    assert tight.n_eff < diffuse.n_eff


def test_posterior_ci90_brackets_point_estimate():
    p = Posterior(np.arange(1.0, 101.0))
    lo, hi = p.ci90
    assert lo < p.point < hi
    assert p.point == pytest.approx(np.median(np.arange(1.0, 101.0)))
