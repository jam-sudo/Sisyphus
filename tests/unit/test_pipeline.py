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

    def test_result_has_warnings_list(self):
        """PredictionResult always has a warnings list."""
        from sisyphus.pipeline.predict import predict

        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        assert isinstance(result.warnings, list)

    def test_result_has_ad_flags(self):
        """PredictionResult carries applicability domain flags."""
        from sisyphus.pipeline.predict import predict

        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        assert isinstance(result.ad_flags, list)
        assert isinstance(result.in_applicability_domain, bool)
