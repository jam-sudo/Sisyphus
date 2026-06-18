"""Zonal reactive-metabolite hazard probe — harness-isolated, synthetic engine only.
Spec: 2026-06-18-zonal-reactive-metabolite-hazard-design.md. No predict()/holdout."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

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
