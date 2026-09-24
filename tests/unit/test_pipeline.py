"""Unit tests for pipeline/predict.py — end-to-end SMILES -> PredictionResult."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from sisyphus.core import Distribution, PredictionResult


class TestPipeline:
    @pytest.mark.parametrize("dose", [0, -1, float("nan"), float("inf")])
    def test_rejects_nonpositive_or_nonfinite_dose(self, dose):
        from sisyphus.pipeline.predict import predict

        with pytest.raises(ValueError, match="positive and finite"):
            predict("CCO", dose)

    @pytest.mark.parametrize("route", ["sc", "IV", "", None])
    def test_rejects_unknown_route_instead_of_defaulting_to_oral(self, route):
        from sisyphus.pipeline.predict import predict

        with pytest.raises(ValueError, match="route must be"):
            predict("CCO", 10.0, route=route)

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
        assert result.method == "engine"
        assert result.ml_pk is None
        assert result.cmax_prediction.residual_interval_90 is None

    def test_result_has_warnings_tuple(self):
        """PredictionResult always has a warnings tuple."""
        from sisyphus.pipeline.predict import predict

        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        assert isinstance(result.warnings, tuple)

    def test_strict_solver_nonconvergence_raises(self, monkeypatch):
        import sisyphus.engine.solver as solver
        from sisyphus.core import SimResult
        from sisyphus.pipeline.predict import predict

        def failed_solve(*args, **kwargs):
            return SimResult(
                time_h=np.array([0.0]), concentrations={}, amounts={},
                mass_balance_error=0.0, solver_success=False,
            )

        monkeypatch.setattr(solver, "solve", failed_solve)
        with pytest.raises(RuntimeError, match="ODE solver did not converge"):
            predict("CCO", 10.0, strict=True)

    def test_strict_vdss_failure_raises_instead_of_dropping_track(self, monkeypatch):
        pipeline = import_module("sisyphus.pipeline.predict")
        adme_module = import_module("sisyphus.predict.adme")
        predict_adme = adme_module.predict_adme

        def invalid_vdss(profile):
            return replace(predict_adme(profile), vdss=Distribution(mean=0.0))

        monkeypatch.setattr(adme_module, "predict_adme", invalid_vdss)
        with pytest.raises(ZeroDivisionError):
            pipeline.predict("CCO", 10.0, strict=True)
        result = pipeline.predict("CCO", 10.0)
        assert any("VDss analytical failed" in warning for warning in result.warnings)

    def test_result_has_ad_flags(self):
        """PredictionResult carries applicability domain flags."""
        from sisyphus.pipeline.predict import predict

        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        assert isinstance(result.ad_flags, tuple)
        assert isinstance(result.in_applicability_domain, bool)

    def test_final_cmax_and_engine_simulation_are_separate_contracts(self):
        from sisyphus.pipeline.predict import predict

        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        assert result.cmax_prediction is not None
        assert result.engine_simulation is not None
        assert result.cmax_prediction.cmax.mean == result.pk.cmax.mean
        assert result.engine_simulation.endpoints == result.engine_pk
        assert len(result.engine_simulation.time_h) == len(
            result.engine_simulation.concentration_mg_l
        )
        assert dict(result.cmax_prediction.tracks)["engine"] == pytest.approx(
            result.engine_pk.cmax.mean
        )

    def test_ad_membership_never_claims_high_confidence(self):
        from sisyphus.pipeline.predict import predict

        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        assert result.confidence in {"medium", "low"}
        assert result.resource_profile == "public"


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
