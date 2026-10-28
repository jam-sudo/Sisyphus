"""Small contract tests for the optional MIPD neural backend."""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.mipd.amortizer import NeuralAmortizer, _neural_context
from sisyphus.mipd.clgrid import CLGrid, CLGridForward, CLPrior, MeasuredConc
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
    return FPrior(0.5), CLPrior(), CLGridForward(grid), np.ones(9), DosingRegimen.single_oral(10)


def test_zero_observations_returns_engine_prior():
    f_prior, cl_prior, forward, features, regimen = _case()
    post = NeuralAmortizer(None).posterior(
        f_prior, cl_prior, forward, (), molecular_features=features,
        regimen=regimen, n_samples=1000, rng=np.random.default_rng(0),
    )
    assert post.n_eff == 1000
    assert post.cl_scale is not None
    assert np.median(post.f.samples) == pytest.approx(0.5, rel=0.1)
    assert np.median(post.cl_scale.samples) == pytest.approx(1.0, rel=0.1)


def test_context_validates_regimen_and_observations():
    f_prior, cl_prior, forward, features, regimen = _case()
    context = _neural_context(
        f_prior, cl_prior, forward, (MeasuredF(0.4), MeasuredConc(0.2, 16.0)),
        features, regimen,
    )
    assert context.shape == (9 + 6 + 14,)
    with pytest.raises(ValueError, match="one oral dose"):
        _neural_context(
            f_prior, cl_prior, forward, (MeasuredF(0.4),), features,
            DosingRegimen.oral_repeated(10, 12, 2),
        )
    with pytest.raises(ValueError, match="at most two"):
        _neural_context(
            f_prior, cl_prior, forward, (MeasuredF(0.4),) * 3, features, regimen
        )
    with pytest.raises(ValueError, match="prior CV 1"):
        _neural_context(FPrior(0.5, 0.5), cl_prior, forward, (), features, regimen)


def test_training_pairs_use_engine_priors_and_observation_schema():
    from scripts.train_mipd_neural_amortizer import _draw_pair

    f_prior, cl_prior, forward, features, regimen = _case()
    rng = np.random.default_rng(2)
    kinds = set()
    for _ in range(40):
        theta, context, observations = _draw_pair(
            forward, f_prior, cl_prior, features, regimen, rng
        )
        assert 0 < theta[0] <= 1
        assert 0.05 <= theta[1] <= 20
        assert context.shape == (29,)
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


def test_neural_posterior_maps_latents_through_grid():
    pytest.importorskip("torch")
    f_prior, cl_prior, forward, features, regimen = _case()

    class Model:
        def sample(self, context, n):
            assert len(context) == 9 + 6 + 14
            return np.tile([0.4, 2.0], (n, 1))

    post = NeuralAmortizer(Model()).posterior(
        f_prior, cl_prior, forward, (MeasuredConc(0.2, 16.0),),
        molecular_features=features, regimen=regimen, n_samples=20,
    )
    assert post.f.point == pytest.approx(0.4)
    assert post.cl_scale.point == pytest.approx(2.0)
    assert post.cmax.point == pytest.approx(forward(np.array([0.4]), np.array([2.0]))["cmax"][0])


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
