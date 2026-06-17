"""Liver-zonation invariance probe — harness-isolated, synthetic engine only.
Spec: 2026-06-17-liver-zonation-phase0-design.md §4. No predict()/holdout."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = ROOT / "scripts" / "probe_liver_zonation.py"


def _probe():
    spec = importlib.util.spec_from_file_location("zonation_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_apply_zonation_preserves_total_abundance():
    p = _probe()
    g = p.h._axial_graph("CYP3A4", n_sub=10)
    subs = p._liver_subtanks(g)
    total_before = sum(n.enzymes["CYP3A4"].mean for n in subs)
    w = p.zonation_weights(10, 3.0, "pericentral")
    gz = p.apply_zonation(g, "CYP3A4", w)
    subs_z = p._liver_subtanks(gz)
    total_after = sum(n.enzymes["CYP3A4"].mean for n in subs_z)
    assert total_after == pytest.approx(total_before, rel=1e-12)
    # pericentral => abundance increases toward the outlet sub-tank
    means = [n.enzymes["CYP3A4"].mean for n in subs_z]
    assert all(means[i] < means[i + 1] for i in range(len(means) - 1))
