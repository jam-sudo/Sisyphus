"""Small contract tests for the optional MIPD neural backend."""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.mipd.amortizer import (
    NeuralAmortizer,
    _clipped_log_normal_logprob,
    _neural_context,
)
from sisyphus.mipd.clgrid import CLGrid, CLGridForward, CLPrior, MeasuredConc, sir_posterior_2d
from sisyphus.mipd.core import FPrior, MeasuredF
from sisyphus.regimen.types import DosingRegimen


def _case():
    grid = CLGrid(
        s_grid=np.array([0.05, 1.0, 20.0]), t_grid=np.array([0.0, 4.0, 16.0, 24.0]),
        conc=np.array([
            [0.01, 2.0, 1.5, 1.0], [0.01, 1.0, 0.4, 0.2], [0.01, 0.5, 0.05, 0.01],
        ]),
        cmax=np.array([2.0, 1.0, 0.5]), auc=np.array([25.0, 10.0, 4.0]),
        f_engine=np.array([0.5, 0.5, 0.5]),
    )
    return FPrior(0.5), CLPrior(), CLGridForward(grid), DosingRegimen.single_oral(10)


def test_zero_observations_returns_engine_prior():
    f_prior, cl_prior, forward, regimen = _case()
    post = NeuralAmortizer(None).posterior(
        f_prior, cl_prior, forward, (),
        regimen=regimen, n_samples=1000, rng=np.random.default_rng(0),
    )
    assert post.n_eff == 1000
    assert post.cl_scale is not None
    assert np.median(post.f.samples) == pytest.approx(0.5, rel=0.1)
    assert np.median(post.cl_scale.samples) == pytest.approx(1.0, rel=0.1)
    assert not post.warnings
    assert not NeuralAmortizer(None).posterior(
        f_prior, cl_prior, forward, (), regimen=regimen,
        n_samples=5, raw_flow=True,
    ).warnings


def test_clipped_prior_density_includes_endpoint_mass():
    from scipy.stats import lognorm

    values = np.array([1e-6, 0.5, 1.0])
    sigma = np.sqrt(np.log(2.0))
    logp = _clipped_log_normal_logprob(values, 0.5, 1.0, 1e-6, 1.0)
    assert np.exp(logp[0]) == pytest.approx(lognorm.cdf(1e-6, sigma, scale=0.5))
    assert np.exp(logp[1]) == pytest.approx(lognorm.pdf(0.5, sigma, scale=0.5))
    assert np.exp(logp[2]) == pytest.approx(lognorm.sf(1.0, sigma, scale=0.5))


def test_context_validates_regimen_and_observations():
    f_prior, cl_prior, forward, regimen = _case()
    context = _neural_context(
        f_prior, cl_prior, forward, (MeasuredF(0.4), MeasuredConc(0.2, 16.0)),
        regimen,
    )
    assert context.shape == (20 + 4 + 14,)
    changed_grid = _case()[2].grid
    changed_grid.cmax[0] *= 2
    changed = _neural_context(f_prior, cl_prior, CLGridForward(changed_grid), (), regimen)
    assert changed[0] != _neural_context(f_prior, cl_prior, forward, (), regimen)[0]
    with pytest.raises(ValueError, match="one oral dose"):
        _neural_context(
            f_prior, cl_prior, forward, (MeasuredF(0.4),),
            DosingRegimen.oral_repeated(10, 12, 2),
        )
    with pytest.raises(ValueError, match="at most two"):
        _neural_context(
            f_prior, cl_prior, forward, (MeasuredF(0.4),) * 3, regimen
        )
    with pytest.raises(ValueError, match="prior CV 1"):
        _neural_context(FPrior(0.5, 0.5), cl_prior, forward, (), regimen)
    with pytest.raises(ValueError, match="trained CLint prior bounds"):
        _neural_context(f_prior, CLPrior(s_min=0.1), forward, (), regimen)
    _neural_context(
        f_prior, CLPrior(s_max=np.nextafter(20.0, 0.0)), forward, (), regimen
    )


def test_training_pairs_use_engine_priors_and_observation_schema():
    from scripts.train_mipd_neural_amortizer import _draw_pair

    f_prior, cl_prior, forward, regimen = _case()
    rng = np.random.default_rng(2)
    kinds = set()
    for _ in range(40):
        theta, context, observations = _draw_pair(
            forward, f_prior, cl_prior, regimen, rng
        )
        assert 0 < theta[0] <= 1
        assert 0.05 <= theta[1] <= 20
        assert context.shape == (38,)
        assert np.isfinite(context).all()
        kinds.add(type(observations[0]))
    assert len(kinds) == 4


def test_cached_grid_checks_compound_identity(tmp_path):
    from scripts.train_mipd_neural_amortizer import _setup

    grid = _case()[2].grid
    np.savez_compressed(
        tmp_path / "0.npz", smiles="CCO", dose_mg=10.0,
        s_grid=grid.s_grid, t_grid=grid.t_grid, conc=grid.conc,
        cmax=grid.cmax, auc=grid.auc, f_engine=grid.f_engine,
    )
    setup = _setup({"smiles": "CCO", "dose_mg": 10, "_index": 0}, 3, tmp_path)
    assert setup[0].grid.s_grid.size == 3
    with pytest.raises(ValueError, match="input mismatch"):
        _setup({"smiles": "CCC", "dose_mg": 10, "_index": 0}, 3, tmp_path)


def test_sir_reference_raises_particle_count_until_ess_is_usable(monkeypatch):
    from types import SimpleNamespace

    from scripts import train_mipd_neural_amortizer as trainer

    calls = []

    def fake_sir(*args, n_samples, **kwargs):
        calls.append(n_samples)
        return SimpleNamespace(n_eff=50 if n_samples == 20000 else 300)

    monkeypatch.setattr(trainer, "sir_posterior_2d", fake_sir)
    post, n = trainer._sir_reference(None, None, None, (), np.random.default_rng(0))
    assert calls == [20000, 40000]
    assert n == 40000 and post.n_eff == 300


def test_agreement_uses_sir_ess_not_resample_count():
    from scripts.train_mipd_neural_amortizer import _mean_agrees_with_mc

    neural = np.tile([-0.8, 1.2], 250)
    sir = np.tile([-1.0, 1.0], 10000)
    assert _mean_agrees_with_mc(neural, sir, sir_ess=200)
    assert not _mean_agrees_with_mc(neural, sir, sir_ess=20000)


def test_neural_posterior_maps_latents_through_grid():
    pytest.importorskip("torch")
    f_prior, cl_prior, forward, regimen = _case()

    class Model:
        def sample(self, context, n):
            assert len(context) == 20 + 4 + 14
            return np.tile([0.4, 2.0], (n, 1))

    post = NeuralAmortizer(Model()).posterior(
        f_prior, cl_prior, forward, (MeasuredConc(0.2, 16.0),),
        regimen=regimen, n_samples=20, raw_flow=True,
    )
    assert post.f.point == pytest.approx(0.4)
    assert post.cl_scale.point == pytest.approx(2.0)
    assert post.cmax.point == pytest.approx(forward(np.array([0.4]), np.array([2.0]))["cmax"][0])
    assert any("does not match exact SIR" in warning for warning in post.warnings)


def test_neural_is_retains_f_boundary_mass_and_agrees_with_sir(monkeypatch):
    torch = pytest.importorskip("torch")
    f_prior, cl_prior, forward, regimen = _case()

    class UniformFlow:
        density_estimator = torch.nn.Linear(1, 1)

        def sample(self, context, n, *, reject_outside_prior=True):
            rng = np.random.default_rng(3)
            return np.column_stack((rng.uniform(0, 1.2, n), rng.uniform(0, 20.5, n)))

        class posterior:
            @staticmethod
            def log_prob(theta, x, norm_posterior=True):
                return torch.full((len(theta),), -np.log(1.2 * 20.5))

    obs = (MeasuredF(0.8, cv=0.3),)
    post = NeuralAmortizer(UniformFlow()).posterior(
        f_prior, cl_prior, forward, obs, regimen=regimen,
        n_samples=2000, rng=np.random.default_rng(4),
    )
    sir = sir_posterior_2d(
        f_prior, cl_prior, forward, obs, n_samples=50000, rng=np.random.default_rng(5),
    )
    assert post.proposal_n_eff is not None and post.proposal_n_eff >= 200
    assert np.all((post.f.samples >= 1e-6) & (post.f.samples <= 1))
    assert np.mean(post.f.samples == 1.0) > 0.1
    assert abs(post.f.mean - sir.f.mean) < 0.05

    monkeypatch.setattr("sisyphus.mipd.amortizer._MIN_NEURAL_ESS_COUNT", 2001)
    fallback = NeuralAmortizer(UniformFlow()).posterior(
        f_prior, cl_prior, forward, obs, regimen=regimen,
        n_samples=50, rng=np.random.default_rng(4),
    )
    assert len(fallback.f.samples) == 50
    assert fallback.proposal_n_eff is not None and fallback.proposal_n_eff < 2001
    assert any("used grid SIR" in warning for warning in fallback.warnings)


def test_npe_sampler_can_draw_untruncated_flow():
    torch = pytest.importorskip("torch")
    from sisyphus.sbi.amortizer import NPETrainingResult

    class DensityEstimator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.condition = None

        def sample(self, shape, *, condition):
            self.condition = condition
            return torch.ones((shape[0], 1, 2))

    estimator = DensityEstimator()
    result = NPETrainingResult(
        posterior=None, density_estimator=estimator,
        training_losses=[], validation_losses=[], n_train_samples=1,
        prior_low=np.zeros(2), prior_high=np.ones(2),
    )
    assert result.sample(np.zeros(3), n=4, reject_outside_prior=False).shape == (4, 2)
    assert estimator.condition.shape == (1, 3)


def test_load_requires_optional_dependencies(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def missing_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_torch)
    with pytest.raises(ImportError, match="optional sbi extra"):
        NeuralAmortizer.load(tmp_path / "missing.pt")
