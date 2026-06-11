"""Integration tests for mipd.tdm.predict_tdm (IV steady-state TDM)."""
import pytest

from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.covariates import Covariates
from sisyphus.mipd.tdm import predict_tdm
from sisyphus.regimen.types import DosingRegimen

ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"  # high-fup, renally-cleared


def _iv_regimen():
    return DosingRegimen.iv_infusion(dose_mg=50.0, duration_h=0.5, interval_h=8.0, n_doses=5)


def test_predict_tdm_rejects_oral_regimen():
    oral = DosingRegimen.oral_repeated(dose_mg=50.0, interval_h=8.0, n_doses=3)
    with pytest.raises(ValueError, match="IV"):
        predict_tdm(ATENOLOL, oral, [], seed=0)


def test_predict_tdm_output_is_honest_for_iv():
    post = predict_tdm(ATENOLOL, _iv_regimen(), [], n_grid=5, seed=0)
    assert post.meta_cmax is None      # no oral-calibrated population blend for IV
    assert post.cmax_90ci is None      # no oral-calibrated conformal band for IV
    assert post.renal_scale is not None
    assert post.cmax.point > 0


def test_predict_tdm_low_trough_means_faster_clearance_lower_exposure():
    reg = _iv_regimen()
    base = predict_tdm(ATENOLOL, reg, [], n_grid=9, seed=0)
    base_cmax = float(base.cmax.point)
    # a trough far BELOW baseline -> faster clearance -> renal_scale > 1 -> lower exposure
    low = predict_tdm(
        ATENOLOL, reg, [MeasuredConc(value=base_cmax * 0.1, t=39.0, cv=0.2)],
        n_grid=9, seed=0,
    )
    assert low.renal_scale.point > 1.0
    assert low.auc.point < base.auc.point


def test_predict_tdm_extreme_crcl_warns():
    post = predict_tdm(
        ATENOLOL, _iv_regimen(), [], covariates=Covariates(crcl_ml_min=3), n_grid=5, seed=0
    )
    assert any("crcl" in w.lower() for w in post.warnings)
