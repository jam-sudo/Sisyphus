"""Data-independent mechanism tests for the two-arm genotype-nonlinearity harness.
Spec: 2026-06-16-...-two-arm-design.md §5, §8. Runs the synthetic engine only —
no clinical data, no predict()/holdout. Imports the harness script by path."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = ROOT / "scripts" / "validate_pgx_cmax_v2b.py"


def _harness():
    spec = importlib.util.spec_from_file_location("v2b_harness", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_axial_graph_has_subtanks_no_literal_liver():
    h = _harness()
    g = h._axial_graph("CYP2D6", n_sub=8)
    assert "liver" not in g.nodes
    subs = [n for n in g.nodes.values() if (n.lookup_name or n.name) == "liver"]
    assert len(subs) == 8 and all("CYP2D6" in s.enzymes for s in subs)


def test_steady_state_exposure_accumulates():
    h = _harness()
    g = h._well_stirred_graph("CYP2C9")
    abund = g.nodes["liver"].enzymes["CYP2C9"].mean
    drug = h._drug("CYP2C9", 0.9, 5.0, abund, peff=20.0, kp=3.0, fup=0.10,
                   dose_mg=100.0, mw=252.3)
    e1 = h._steady_state_exposure(g, drug, interval_h=24.0, n_doses=1, metric="css_avg")
    e30 = h._steady_state_exposure(g, drug, interval_h=24.0, n_doses=20, metric="css_avg")
    assert e30 > e1 > 0  # accumulation to steady state


def test_single_dose_exposure_positive():
    h = _harness()
    g = h._well_stirred_graph("CYP2D6")
    abund = g.nodes["liver"].enzymes["CYP2D6"].mean
    drug = h._sat_drug("CYP2D6", 0.8, 1.0e7, abund, peff=20.0, kp=3.0, km_mgl=0.9,
                       fup=0.3, dose_mg=150.0, mw=341.4)
    assert h._single_dose_exposure(g, drug, metric="auc") > 0
