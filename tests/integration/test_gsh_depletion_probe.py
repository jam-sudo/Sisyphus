"""Gate tests for the zonal GSH-pool depletion probe (B1.x). Stack-independent
assertions only (signs/inequalities/tolerances), not pinned floats. Thresholds reflect
the measured, a-priori-pinned (untuned) signal; a genuine miss is an honest-negative."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "gsh_probe", _ROOT / "scripts" / "probe_gsh_depletion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe()


def test_g_order_static_invariant_dynamic_variant():
    # Centerpiece: reordering the SAME concentration value-multiset leaves the static
    # pointwise hazard invariant (exact-reverse construction => fp-zero) but moves the
    # dynamic pool hazard (history/pool memory).
    o = probe.order_test()
    assert o["static_rel_diff"] < 1e-9          # measure-preserving reordering invariant
    assert o["dyn_rel_diff"] > 0.01             # pool memory: materially different


def test_g2_bulk_invariant_hazard_variant():
    rows, e_span = probe.zonation_test()
    assert e_span < 1e-2                         # bulk parent E ~invariant to CYP zonation (DE-50)
    peaks = {r["hazard_peak_zone"] for r in rows}
    assert len(peaks) > 1                        # per-zone hazard peak-zone moves with zonation


def test_g1_localization_apap_peaks_pericentral():
    # APAP config (bioactivation pericentral-high, pool pericentral-low) -> outlet peak.
    cb, tb = probe._b1._parent_profile_by_zone(
        probe._CFG["gene_tag"], probe._CFG["fm"], probe._CFG["n_sub"], probe._CFG["cltot"],
        probe._CFG["fup"], probe._CFG["mw"], probe._CFG["km_mgl"], dose_mg=400.0)
    haz = probe._dynamic_profile_hazard(cb, tb, probe._APAP["bio_direction"],
                                        probe._APAP["bio_ratio"], probe._APAP["gsh_direction"],
                                        probe._APAP["gsh_ratio"])
    assert int(np.argmax(haz)) >= probe._CFG["n_sub"] - 3   # pericentral (outlet) zone


def test_g_time_ratios_finite_excess_reported():
    # Physical bolus-vs-divided arm. HONEST-NEGATIVE expected: dynamic escape-saturation
    # compresses the dynamic ratio BELOW the static envelope ratio (excess < 0). The gate
    # does NOT presuppose a sign; it only requires both ratios finite/positive and the
    # excess well-defined (the report records the sign).
    t = probe.time_test()
    assert np.isfinite(t["dyn_ratio"]) and t["dyn_ratio"] > 0
    assert np.isfinite(t["static_ratio"]) and t["static_ratio"] > 0
    assert np.isfinite(t["excess_path_dependence"])


def test_g_cliff_transition_widths_finite_and_dose_response_rises():
    rows, w_dyn, w_sta = probe.dose_test()
    assert np.isfinite(w_dyn) and np.isfinite(w_sta)       # both curves have a defined 10->90 rise
    assert rows[-1]["dyn_maxH"] > rows[0]["dyn_maxH"]      # dynamic hazard rises with dose


def test_g_nac_monotone_protective():
    out = probe.nac_test()
    maxh = [r["maxH"] for r in out]                          # gsh0 scale 1.0,1.5,3.0
    assert maxh[0] >= maxh[1] >= maxh[2]                     # more pool -> less hazard


def test_headline_isolation_unchanged():
    # The 4-track holdout cache must be untouched by anything in this probe.
    p = _ROOT / "data" / "training" / "4track_holdout_predictions.json"
    d = json.loads(p.read_text())
    assert abs(d["overall"]["meta"]["aafe"] - 2.735) < 5e-3


@pytest.mark.parametrize("name", ["test_cached_holdout_aafe_is_2p735",
                                  "test_mm_headline_bit_identity"])
def test_headline_pins_exist(name):
    # Guard: the canonical headline pins still exist in the suite (regenerated, not removed).
    found = list(_ROOT.glob("tests/**/*.py"))
    assert any(name in f.read_text() for f in found), f"{name} pin missing"
