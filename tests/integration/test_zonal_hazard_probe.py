"""Zonal reactive-metabolite hazard probe — harness-isolated, synthetic engine only.
Spec: 2026-06-18-zonal-reactive-metabolite-hazard-design.md. No predict()/holdout."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = ROOT / "scripts" / "probe_zonal_hazard.py"


def _probe():
    spec = importlib.util.spec_from_file_location("zonal_hazard_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parent_profile_by_zone_decreasing_inlet_to_outlet():
    """Sanity: the parent is extracted as it flows, so per-zone peak C_u decreases
    inlet(ax1)->outlet(axN) for a meaningfully-extracted drug."""
    p = _probe()
    c_by_zone, time = p._parent_profile_by_zone(
        "CYP3A4", fm=0.9, n_sub=10, cltot=1.0e6, fup=0.3, mw=300.0, km_mgl=0.5,
        dose_mg=100.0,
    )
    assert len(c_by_zone) == 10 and len(time) == len(c_by_zone[0])
    peaks = [float(np.max(c)) for c in c_by_zone]
    assert peaks[0] > peaks[-1] > 0          # extracted along the tube


def _aceta_cfg():
    """Acetaminophen-like config (controller-calibrated): bioactivation pericentral-high,
    detox pericentral-LOW (periportal-high), Km_bio above low-dose C_u so a clean dose-
    threshold exists. Synthetic-param selection for mechanism visibility, NOT clinical fit."""
    return dict(gene_tag="CYP3A4", fm=0.9, n_sub=10, cltot=1.0e6, fup=0.3, mw=300.0,
                km_mgl=0.5, vmax_bio_total=300.0, vmax_detox_total=15.0, km_bio=1.0)


def test_G1_hazard_localizes_pericentral_for_aceta_config():
    """G1 (sanity): bio pericentral-high + detox pericentral-low -> hazard peaks at the
    OUTLET zone (zone 3)."""
    p = _probe()
    cfg = _aceta_cfg()
    haz = p.zone_hazard_profile(**cfg, dose_mg=200.0, bio_direction="pericentral",
                                bio_ratio=3.0, detox_direction="periportal", detox_ratio=3.0)
    assert int(np.argmax(haz)) >= cfg["n_sub"] - 3


def test_G2_bulk_E_invariant_while_hazard_profile_varies():
    """G2 (centerpiece, DE-50 closure): varying bioactivation zonation leaves bulk parent
    E ~invariant while the per-zone hazard peak-zone moves materially."""
    p = _probe()
    cfg = _aceta_cfg()
    e_peri = p.bulk_E(cfg["gene_tag"], cfg["fm"], cfg["n_sub"], cfg["cltot"], cfg["fup"],
                      cfg["mw"], cfg["km_mgl"], "pericentral", 3.0)
    e_port = p.bulk_E(cfg["gene_tag"], cfg["fm"], cfg["n_sub"], cfg["cltot"], cfg["fup"],
                      cfg["mw"], cfg["km_mgl"], "periportal", 3.0)
    assert abs(e_peri - e_port) < 0.01                       # bulk ~invariant (DE-50)
    haz_peri = p.zone_hazard_profile(**cfg, dose_mg=200.0, bio_direction="pericentral",
                                     bio_ratio=3.0, detox_direction="uniform", detox_ratio=1.0)
    haz_port = p.zone_hazard_profile(**cfg, dose_mg=200.0, bio_direction="periportal",
                                     bio_ratio=3.0, detox_direction="uniform", detox_ratio=1.0)
    assert int(np.argmax(haz_peri)) != int(np.argmax(haz_port))   # peak moves with zonation


def test_G3_dose_threshold_and_zone_specificity():
    """G3 (the mechanism): below the threshold dose NO zone has hazard; above it the
    pericentral (high-bio/low-detox) zone crosses FIRST; raising detox protects."""
    p = _probe()
    cfg = _aceta_cfg()
    kw = dict(bio_direction="pericentral", bio_ratio=3.0,
              detox_direction="periportal", detox_ratio=3.0)   # detox pericentral-low
    haz_lo = p.zone_hazard_profile(**cfg, dose_mg=50.0, **kw)
    haz_hi = p.zone_hazard_profile(**cfg, dose_mg=200.0, **kw)
    assert max(haz_lo) == pytest.approx(0.0)                  # below threshold: no hazard
    assert max(haz_hi) > 0.0                                  # above threshold: hazard
    assert int(np.argmax(haz_hi)) >= cfg["n_sub"] - 3         # ...pericentral zone
    cfg_protected = dict(cfg, vmax_detox_total=cfg["vmax_detox_total"] * 3.0)
    haz_protected = p.zone_hazard_profile(**cfg_protected, dose_mg=200.0, **kw)
    assert max(haz_protected) < max(haz_hi)                   # detoxification protects


def test_headline_isolation_holdout_cache_untouched():
    """Running the probe leaves the holdout cache byte-identical and the v2.2a + cached
    2.731 pins passing. Headline untouched by construction."""
    import subprocess
    import sys

    cache = ROOT / "data" / "training" / "4track_holdout_predictions.json"
    before = cache.read_bytes()
    p = _probe()
    p._parent_profile_by_zone("CYP3A4", 0.9, 8, 1.0e6, 0.3, 300.0, 0.5, dose_mg=100.0)
    assert cache.read_bytes() == before
    r = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/regression/test_mm_headline_bit_identity.py",
         "tests/integration/test_holdout_regression.py::test_cached_holdout_aafe_is_2p731",
         "-q"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
