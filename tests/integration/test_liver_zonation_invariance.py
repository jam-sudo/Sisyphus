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


def test_G1_delta_E_decays_to_zero_with_N_linear():
    """G1 (linear): |ΔE(N)| shrinks toward 0 as N grows — first-pass is invariant to
    zonation in the plug-flow limit."""
    p = _probe()
    d10, _, _ = p.delta_E("CYP3A4", 0.9, 10, 3.0, "pericentral", cltot=100000.0, fup=0.3, mw=300.0)
    d80, _, _ = p.delta_E("CYP3A4", 0.9, 80, 3.0, "pericentral", cltot=100000.0, fup=0.3, mw=300.0)
    assert abs(d80) < abs(d10)
    assert abs(d80) < 0.005


def test_G1_delta_E_decays_to_zero_with_N_saturable():
    """G1 (saturable): same invariance with the v2.2a MM flux engaged (high extraction)."""
    p = _probe()
    d10, _, _ = p.delta_E("CYP3A4", 0.9, 10, 3.0, "pericentral", cltot=1.0e6, fup=0.3,
                          mw=300.0, km_mgl=0.5)
    d80, _, _ = p.delta_E("CYP3A4", 0.9, 80, 3.0, "pericentral", cltot=1.0e6, fup=0.3,
                          mw=300.0, km_mgl=0.5)
    assert abs(d80) < abs(d10)
    assert abs(d80) < 0.005


def test_G3_finite_N_artifact_is_saturation_asymmetric():
    """G3: linear is direction-SYMMETRIC (pericentral≈periportal — convexity is symmetric);
    saturable is direction-ASYMMETRIC. Assert the asymmetry EXISTS; REPORT the sign (the §2
    derivation expects periportal>pericentral; reported, not gated — PGx DE-49 discipline)."""
    p = _probe()
    n = 8
    _, _, ez_peri_lin = p.delta_E("CYP3A4", 0.9, n, 3.0, "pericentral", cltot=100000.0,
                                  fup=0.3, mw=300.0)
    _, _, ez_port_lin = p.delta_E("CYP3A4", 0.9, n, 3.0, "periportal", cltot=100000.0,
                                  fup=0.3, mw=300.0)
    _, _, ez_peri_sat = p.delta_E("CYP3A4", 0.9, n, 3.0, "pericentral", cltot=1.0e6,
                                  fup=0.3, mw=300.0, km_mgl=0.5)
    _, _, ez_port_sat = p.delta_E("CYP3A4", 0.9, n, 3.0, "periportal", cltot=1.0e6,
                                  fup=0.3, mw=300.0, km_mgl=0.5)
    lin_asym = abs(ez_peri_lin - ez_port_lin)
    sat_asym = abs(ez_peri_sat - ez_port_sat)
    assert lin_asym < 1e-3
    assert sat_asym > lin_asym
    print(f"G3 saturable direction sign: periportal-pericentral = "
          f"{ez_port_sat - ez_peri_sat:+.4f} (>0 => periportal extracts more, §2 expectation)")


def test_G2_E_stabilizes_with_N():
    """G2: the axial cascade converges — E(N) stabilizes as N grows (successive
    differences shrink), validating the PR-#79 discretization. Saturable config (the
    linear E converges by N=5, so it cannot demonstrate stabilization)."""
    p = _probe()
    es = p.e_curve("CYP3A4", 0.9, [10, 20, 40, 80], cltot=1.0e6, fup=0.3, mw=300.0, km_mgl=0.5)
    assert abs(es[-1] - es[-2]) < abs(es[1] - es[0])
    assert abs(es[-1] - es[-2]) < 0.003


def test_G2_uniform_and_zonated_share_the_limit():
    """G2: uniform and zonated E converge to the SAME limit (distribution-invariance)."""
    p = _probe()
    eu = p.e_curve("CYP3A4", 0.9, [80], cltot=1.0e6, fup=0.3, mw=300.0, km_mgl=0.5)[0]
    ez = p.e_curve("CYP3A4", 0.9, [80], cltot=1.0e6, fup=0.3, mw=300.0, km_mgl=0.5,
                   direction="pericentral", ratio=3.0)[0]
    assert abs(eu - ez) < 0.005


def test_ratio1_oracle_is_noop():
    """ratio=1 zonation reproduces the unmodified axial E bit-identically."""
    p = _probe()
    g = p.h._axial_graph("CYP3A4", n_sub=10)
    e0 = p.h._engine_e_h(g, "CYP3A4", 0.9, 100000.0, p.h._SYNTHETIC_GENE_ABUND, 20.0, 3.0,
                         0.3, 100.0, 300.0, None)
    gz = p.apply_zonation(g, "CYP3A4", p.zonation_weights(10, 1.0, "pericentral"))
    ez = p.h._engine_e_h(gz, "CYP3A4", 0.9, 100000.0, p.h._SYNTHETIC_GENE_ABUND, 20.0, 3.0,
                         0.3, 100.0, 300.0, None)
    assert ez == pytest.approx(e0, rel=1e-12)
