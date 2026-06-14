"""Unit tests for pipeline/predict.py — end-to-end SMILES -> PredictionResult."""

import pytest

from sisyphus.core import PredictionResult


class TestPipeline:
    def test_end_to_end_caffeine(self):
        """Full pipeline: caffeine 100mg oral -> PredictionResult."""
        from sisyphus.pipeline.predict import predict

        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        assert isinstance(result, PredictionResult)
        assert result.pk.cmax.mean > 0
        assert result.method in ("hybrid", "engine", "ml")
        assert result.dose_mg == 100.0

    def test_end_to_end_midazolam(self):
        """Full pipeline: midazolam 2mg oral."""
        from sisyphus.pipeline.predict import predict

        result = predict("Clc1ccc2c(c1)C(=NCc3nccn3C)c1cc(F)ccc1N2", dose_mg=2.0)
        assert result.pk.cmax.mean > 0
        assert result.dose_mg == 2.0

    def test_invalid_smiles_raises(self):
        """Invalid SMILES should raise ValueError."""
        from sisyphus.pipeline.predict import predict

        with pytest.raises(ValueError):
            predict("INVALID_SMILES", dose_mg=10.0)

    def test_iv_route(self):
        """IV route should set route correctly."""
        from sisyphus.pipeline.predict import predict

        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0, route="iv")
        assert result.route == "iv"

    def test_result_has_warnings_tuple(self):
        """PredictionResult always has a warnings tuple."""
        from sisyphus.pipeline.predict import predict

        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        assert isinstance(result.warnings, tuple)

    def test_result_has_ad_flags(self):
        """PredictionResult carries applicability domain flags."""
        from sisyphus.pipeline.predict import predict

        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        assert isinstance(result.ad_flags, tuple)
        assert isinstance(result.in_applicability_domain, bool)


# ── engine_f surfacing (review #10): opt-in F_engine on PredictionResult ──
_MDZ = "C[n+]1cnc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"


def test_predict_engine_f_is_none_by_default():
    from sisyphus.pipeline.predict import predict

    assert predict(_MDZ, 7.5).engine_f is None


def test_predict_compute_f_engine_surfaces_oral_bioavailability():
    from sisyphus.pipeline.predict import predict

    r = predict(_MDZ, 7.5, compute_f_engine=True)
    assert r.engine_f is not None
    assert 0.0 < r.engine_f <= 1.0


def test_predict_compute_f_engine_leaves_engine_pk_and_meta_bit_identical():
    from sisyphus.pipeline.predict import predict

    a = predict(_MDZ, 7.5)
    b = predict(_MDZ, 7.5, compute_f_engine=True)
    assert b.engine_pk.cmax.mean == a.engine_pk.cmax.mean
    assert b.engine_pk.auc_0t.mean == a.engine_pk.auc_0t.mean
    assert b.pk.cmax.mean == a.pk.cmax.mean  # meta untouched
