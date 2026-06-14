"""Oral steady-state TDM: mipd.tdm.predict_tdm oral branch (F-only / free-both).

Task 3 adds an ORAL route to ``predict_tdm``: a single-phase observation frees only
F (internal clint-scale pinned at s=1); a multi-phase (or measured-F) observation
frees both F and the metabolic clint-scale. The IV path is untouched. The oral
output never attaches the oral-population ``meta_cmax``/``cmax_90ci`` or the
``renal_scale`` (that is the IV-only latent).
"""
import numpy as np

from sisyphus.mipd.clgrid import CLGridForward, MeasuredConc
from sisyphus.mipd.oral_grid import build_oral_cl_grid
from sisyphus.mipd.tdm import predict_tdm
from sisyphus.regimen.types import DosingRegimen

SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def _reg(dose=100.0, tau=12.0, n=6):
    return DosingRegimen.oral_repeated(dose_mg=dose, interval_h=tau, n_doses=n)


def _engine_conc(grid, t):
    """Engine a-priori (mid-grid F, s=1) venous conc at time ``t`` (mg/L)."""
    f = np.array([grid.f_engine[grid.s_grid.size // 2]])
    return float(CLGridForward(grid)(f, np.ones(1))["conc_at"](t)[0])


def _engine_trough(grid, last, tau):
    return _engine_conc(grid, last + tau)


def test_single_trough_is_f_only():
    reg = _reg()
    post = predict_tdm(SMILES, reg, [MeasuredConc(value=1.0, t=72.0)], n_grid=5)
    assert post.cl_scale is None
    assert post.f is not None
    assert post.meta_cmax is None and post.cmax_90ci is None
    assert post.cmax.samples.size > 0
    _ = post.cmax.ci90


def test_peak_plus_trough_is_free_both():
    reg = _reg()
    obs = [MeasuredConc(value=5.0, t=62.0), MeasuredConc(value=1.0, t=72.0)]
    post = predict_tdm(SMILES, reg, obs, n_grid=7)
    assert post.cl_scale is not None


def test_inference_direction_low_trough_lowers_F():
    reg = _reg()
    grid, _, _ = build_oral_cl_grid(SMILES, reg, n_grid=7)
    last, tau = float(reg.last_dose_time_h), 12.0
    eng = _engine_trough(grid, last, tau)
    post = predict_tdm(SMILES, reg, [MeasuredConc(value=0.4 * eng, t=last + tau)], n_grid=7)
    assert post.f.mean < grid.f_engine[grid.s_grid.size // 2]


def test_iv_path_unchanged():
    iv = DosingRegimen.iv_infusion(dose_mg=1000.0, duration_h=1.0, interval_h=8.0, n_doses=5)
    post = predict_tdm("CCO", iv, [MeasuredConc(value=1.0, t=36.0)], n_grid=5)
    assert post.renal_scale is not None


def test_renal_prior_cv_warning_on_oral_only_when_set():
    reg = _reg()
    obs = [MeasuredConc(value=1.0, t=72.0)]
    default = predict_tdm(SMILES, reg, obs, n_grid=5)
    assert not any("renal_prior_cv" in w for w in default.warnings)
    explicit = predict_tdm(SMILES, reg, obs, n_grid=5, renal_prior_cv=0.3)
    assert any("renal_prior_cv" in w for w in explicit.warnings)


def test_attribution_honesty_trough_agrees_cmax_auc_diverge():
    # Spec §7.5 / Bx2: only the observed-trough conc is attribution-independent;
    # free-both's AUC band is wider than free-F-only's (NOT equal). The peak is set
    # to the engine's own a-priori peak (engine-consistent, so the free-both posterior
    # is well-conditioned, not collapsed) — a peak far off the engine curve would
    # force a degenerate corner and an artificially narrow band, defeating the test.
    reg = _reg()
    grid, _, _ = build_oral_cl_grid(SMILES, reg, n_grid=9)
    last, tau = float(reg.last_dose_time_h), 12.0
    trough_obs = MeasuredConc(value=_engine_conc(grid, last + tau), t=last + tau)
    peak = MeasuredConc(value=_engine_conc(grid, last + 2.0), t=last + 2.0)
    f_only = predict_tdm(SMILES, reg, [trough_obs], n_grid=9)
    both = predict_tdm(SMILES, reg, [trough_obs, peak], n_grid=9)
    assert both.cl_scale is not None and f_only.cl_scale is None

    def _w(p):
        lo, hi = p.auc.ci90
        return hi - lo

    assert _w(both) > _w(f_only)


# --- spec §7 acceptance tests #7 / #15 / #18 ---


def test_acceptance_7_free_f_only_internal_s_is_one():
    # #7: a single-trough oral predict_tdm frees only F (cl_scale is None) and the
    # F posterior is non-degenerate.
    reg = _reg()
    post = predict_tdm(SMILES, reg, [MeasuredConc(value=1.0, t=72.0)], n_grid=5)
    assert post.cl_scale is None
    assert post.f.samples.std() > 0


def test_acceptance_15_flat_tail_ridge_warning():
    # #15: two MeasuredConc at DISTINCT within-interval phases but both on the flat
    # elimination tail (closely-spaced late troughs) route free-both yet emit the
    # ridge-wide / clint-axis warning (the phases do not identify the clint axis).
    reg = _reg()
    last = float(reg.last_dose_time_h)
    obs = [
        MeasuredConc(value=1.0, t=last + 10.0),
        MeasuredConc(value=0.95, t=last + 11.5),
    ]
    post = predict_tdm(SMILES, reg, obs, n_grid=9)
    assert post.cl_scale is not None
    assert any(("ridge-wide" in w) or ("clint axis" in w) for w in post.warnings)


def test_acceptance_18_cmax_ci90_finite_fallback():
    # #18: a single-trough oral predict_tdm returns a cmax with samples and a finite
    # 2-tuple ci90 (do NOT assert two-sidedness).
    reg = _reg()
    post = predict_tdm(SMILES, reg, [MeasuredConc(value=1.0, t=72.0)], n_grid=5)
    assert post.cmax.samples.size > 0
    lo, hi = post.cmax.ci90
    assert np.isfinite(lo) and np.isfinite(hi)
    assert len((lo, hi)) == 2
