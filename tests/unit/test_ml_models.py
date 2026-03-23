"""Tests for ml/models.py (PKPredictor) and ml/ensemble.py (MetaLearner)."""

from __future__ import annotations

import pytest

from sisyphus.core import Distribution, PKEndpoints
from sisyphus.ml.ensemble import MetaLearner
from sisyphus.ml.models import PKPredictor


class TestPKPredictor:
    def test_predict_cmax_positive(self):
        predictor = PKPredictor()
        cmax = predictor.predict_cmax("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)  # caffeine
        assert cmax.mean > 0
        assert cmax.cv == 0.5

    def test_predict_midazolam(self):
        predictor = PKPredictor()
        cmax = predictor.predict_cmax("Clc1ccc2c(c1)C(=NCc3nccn3C)c1cc(F)ccc1N2", dose_mg=2.0)
        assert cmax.mean > 0
        # Midazolam 2mg Cmax should be in the range 0.0001-1.0 mg/L
        assert 0.0001 < cmax.mean < 1.0

    def test_predict_returns_distribution(self):
        predictor = PKPredictor()
        cmax = predictor.predict_cmax("c1ccccc1", dose_mg=10.0)  # benzene
        assert isinstance(cmax, Distribution)
        assert cmax.cv == 0.5

    def test_dose_scaling(self):
        """Higher dose should produce higher Cmax (linear in dose)."""
        predictor = PKPredictor()
        smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # caffeine
        cmax_low = predictor.predict_cmax(smiles, dose_mg=10.0)
        cmax_high = predictor.predict_cmax(smiles, dose_mg=100.0)
        # Model predicts log10(Cmax/dose), so Cmax scales linearly with dose
        assert cmax_high.mean == pytest.approx(cmax_low.mean * 10.0, rel=0.01)

    def test_invalid_smiles_raises(self):
        predictor = PKPredictor()
        with pytest.raises(ValueError):
            predictor.predict_cmax("INVALID", dose_mg=10.0)

    def test_lazy_loading(self):
        """Model is not loaded until first prediction."""
        predictor = PKPredictor()
        assert predictor._model is None
        predictor.predict_cmax("c1ccccc1", dose_mg=1.0)
        assert predictor._model is not None

    def test_cmax_floor(self):
        """Cmax should never be negative or zero."""
        predictor = PKPredictor()
        # Even with a tiny dose, result should be positive
        cmax = predictor.predict_cmax("c1ccccc1", dose_mg=1e-10)
        assert cmax.mean >= 1e-10


class TestMetaLearner:
    def test_combine_both_available(self):
        ml = MetaLearner()
        engine_pk = PKEndpoints(
            cmax=Distribution(0.005), tmax=Distribution(1.5), auc_0t=Distribution(0.05)
        )
        ml_pk = PKEndpoints(
            cmax=Distribution(0.01), tmax=Distribution(2.0), auc_0t=Distribution(0.08)
        )
        result = ml.combine(engine_pk, ml_pk, dose_mg=2.0, logp=3.0)
        assert result.cmax.mean > 0
        assert isinstance(result.cmax, Distribution)
        assert result.cmax.cv == 0.3

    def test_combine_engine_only(self):
        ml = MetaLearner()
        engine_pk = PKEndpoints(
            cmax=Distribution(0.005), tmax=Distribution(1.5), auc_0t=Distribution(0.05)
        )
        result = ml.combine(engine_pk, None)
        assert result.cmax.mean == pytest.approx(0.005, rel=0.01)

    def test_combine_ml_only(self):
        ml = MetaLearner()
        ml_pk = PKEndpoints(
            cmax=Distribution(0.01), tmax=Distribution(2.0), auc_0t=Distribution(0.08)
        )
        result = ml.combine(None, ml_pk)
        assert result.cmax.mean == pytest.approx(0.01, rel=0.01)

    def test_combine_neither(self):
        ml = MetaLearner()
        result = ml.combine(None, None)
        assert result.cmax.mean == pytest.approx(0.0, abs=1e-9)

    def test_tmax_prefers_engine(self):
        """Tmax should come from engine when available."""
        ml = MetaLearner()
        engine_pk = PKEndpoints(
            cmax=Distribution(0.005), tmax=Distribution(1.5), auc_0t=Distribution(0.05)
        )
        ml_pk = PKEndpoints(
            cmax=Distribution(0.01), tmax=Distribution(2.0), auc_0t=Distribution(0.08)
        )
        result = ml.combine(engine_pk, ml_pk)
        assert result.tmax.mean == pytest.approx(1.5)

    def test_auc_prefers_engine(self):
        """AUC should come from engine when available."""
        ml = MetaLearner()
        engine_pk = PKEndpoints(
            cmax=Distribution(0.005), tmax=Distribution(1.5), auc_0t=Distribution(0.05)
        )
        ml_pk = PKEndpoints(
            cmax=Distribution(0.01), tmax=Distribution(2.0), auc_0t=Distribution(0.08)
        )
        result = ml.combine(engine_pk, ml_pk)
        assert result.auc_0t.mean == pytest.approx(0.05)

    def test_result_is_pk_endpoints(self):
        ml = MetaLearner()
        engine_pk = PKEndpoints(
            cmax=Distribution(0.005), tmax=Distribution(1.5), auc_0t=Distribution(0.05)
        )
        result = ml.combine(engine_pk, None)
        assert isinstance(result, PKEndpoints)

    def test_lazy_loading(self):
        ml = MetaLearner()
        assert not ml._loaded
        ml.combine(None, None)
        assert ml._loaded
