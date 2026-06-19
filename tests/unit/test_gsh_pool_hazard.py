"""Unit tests for the GSH-pool hazard post-processor and the transition-width metric."""
import numpy as np

from sisyphus.validation.pgx_metrics import gsh_pool_hazard, mm_rate, transition_width


def _const_profile(c, t_end=24.0, n=2401):
    t = np.linspace(0.0, t_end, n)
    return [np.full_like(t, float(c))], t


def test_zero_formation_gives_zero_hazard_and_full_pool():
    # Vmax_bio=0 => no NAPQI => pool stays at GSH0, hazard == 0.
    c_by_zone, t = _const_profile(1.0)
    h = gsh_pool_hazard(c_by_zone, vmax_bio_by_zone=[0.0], km_bio=1.0,
                        gsh0_by_zone=[10.0], k_syn=0.3, kg=1.0, time=t)
    assert h[0] == 0.0


def test_constant_cu_reaches_analytic_steady_state():
    # Long constant exposure: numeric GSH(end) matches the algebraic steady state
    # k_syn*(GSH0-g) = R*g/(Kg+g), and hazard rate -> R*(1 - g/(Kg+g)).
    c = 2.0
    vmax_bio, km_bio, gsh0, k_syn, kg = 5.0, 1.0, 10.0, 0.3, 1.0
    r = mm_rate(c, vmax_bio, km_bio)
    # solve quadratic q*g^2 - g*(q*GSH0 - q*Kg - r) - q*GSH0*Kg = 0 for g (positive root)
    q = k_syn
    b = q * gsh0 - q * kg - r
    g_star = (b + np.sqrt(b * b + 4 * q * q * gsh0 * kg)) / (2 * q)
    haz_rate_star = r * (1.0 - g_star / (kg + g_star))
    # Read the late-window rate off ONE continuous trajectory (equilibrated by ~150h) via
    # the cumulative-hazard difference H[0,200]-H[0,150]; both runs share the same dt and
    # the same 0->150 trajectory, so the difference is the equilibrated [150,200] segment.
    c200, t200 = _const_profile(c, t_end=200.0, n=20001)
    c150, t150 = _const_profile(c, t_end=150.0, n=15001)
    h200 = gsh_pool_hazard(c200, [vmax_bio], km_bio, [gsh0], k_syn, kg, t200)[0]
    h150 = gsh_pool_hazard(c150, [vmax_bio], km_bio, [gsh0], k_syn, kg, t150)[0]
    late_rate = (h200 - h150) / 50.0
    assert abs(late_rate - haz_rate_star) / haz_rate_star < 0.02


def test_saturating_formation_drains_pool_hazard_to_full_escape():
    # Huge formation vs tiny synthesis: pool ~0, escape factor ~1, hazard -> ∫R_form.
    c = 50.0
    vmax_bio, km_bio = 100.0, 1.0
    c_by_zone, t = _const_profile(c, t_end=24.0, n=4801)
    r = mm_rate(c, vmax_bio, km_bio)
    h = gsh_pool_hazard(c_by_zone, [vmax_bio], km_bio, gsh0_by_zone=[1.0],
                        k_syn=0.01, kg=1.0, time=t)
    integral_rform = r * float(t[-1] - t[0])
    assert h[0] > 0.9 * integral_rform


def test_hazard_monotone_decreasing_in_gsh0():
    # G-NAC analytic lever: raising GSH0 can only lower the escaped-flux hazard.
    c_by_zone, t = _const_profile(5.0, t_end=24.0, n=4801)
    kw = dict(vmax_bio_by_zone=[20.0], km_bio=1.0, k_syn=0.3, kg=1.0, time=t)
    h_lo = gsh_pool_hazard(c_by_zone, gsh0_by_zone=[5.0], **kw)[0]
    h_hi = gsh_pool_hazard(c_by_zone, gsh0_by_zone=[50.0], **kw)[0]
    assert h_hi < h_lo


def test_deterministic():
    c_by_zone, t = _const_profile(5.0)
    a = gsh_pool_hazard(c_by_zone, [20.0], 1.0, [5.0], 0.3, 1.0, t)
    b = gsh_pool_hazard(c_by_zone, [20.0], 1.0, [5.0], 0.3, 1.0, t)
    assert a == b


def test_transition_width_sharp_lt_broad():
    doses = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    # broad: gradual ramp; sharp: near-step at one dose
    broad = [0.0, 0.1, 0.3, 0.6, 0.85, 1.0]
    sharp = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    w_broad = transition_width(doses, broad)
    w_sharp = transition_width(doses, sharp)
    assert w_sharp < w_broad


def test_transition_width_zero_curve_is_nan_safe():
    # All-zero (or flat) curve has no defined 10->90 transition; return inf (not a crash).
    doses = [1.0, 2.0, 3.0]
    assert transition_width(doses, [0.0, 0.0, 0.0]) == float("inf")
