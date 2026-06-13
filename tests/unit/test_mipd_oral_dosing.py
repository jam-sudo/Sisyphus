"""Unit tests for the ORAL branch of mipd.dosing (F-scaled interval reference)."""
import numpy as np
import pytest

from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.dosing import (
    Constraint,
    DoseTarget,
    _interval_reference_oral,
    recommend_dose,
)
from sisyphus.mipd.oral_grid import build_oral_cl_grid
from sisyphus.regimen.types import DosingRegimen

SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def _reg(dose=100.0, tau=12.0, n=6):
    return DosingRegimen.oral_repeated(dose_mg=dose, interval_h=tau, n_doses=n)


def test_oral_recommend_returns_no_renal_latent():
    rec = recommend_dose(
        SMILES, _reg(), [MeasuredConc(value=1.0, t=72.0)],
        DoseTarget((Constraint("trough", low=0.5, high=2.0),)),
        candidate_intervals=(12.0, 24.0), n_grid=5, n_samples=4000, seed=0,
    )
    assert rec.renal_scale is None
    assert rec.f is not None
    assert rec.dose_mg > 0


def test_oral_b1_trough_carries_F_scale():
    reg = _reg()
    grid, _, _ = build_oral_cl_grid(SMILES, reg, n_grid=5)
    s1 = grid.s_grid.size // 2
    f = np.array([2.0 * grid.f_engine[s1]])
    s = np.array([grid.s_grid[s1]])
    last, tau = float(reg.last_dose_time_h), 12.0
    raw = float(grid.conc_at(s, last + tau)[0])
    q, _ = _interval_reference_oral(SMILES, reg, tau, f, s, n_grid=5)
    assert q["trough"][0] == pytest.approx(2.0 * raw, rel=0.05)


def test_oral_auc24_factor():
    reg = _reg(tau=8.0)
    grid, _, _ = build_oral_cl_grid(SMILES, reg, n_grid=5)
    s1 = grid.s_grid.size // 2
    f = np.array([grid.f_engine[s1]])
    s = np.array([grid.s_grid[s1]])
    q, _ = _interval_reference_oral(SMILES, reg, 8.0, f, s, n_grid=5)
    assert q["auc24"][0] == pytest.approx(q["auc"][0] * 3.0, rel=1e-6)


def test_oral_tau_change_emits_shape_caveat():
    rec = recommend_dose(
        SMILES, _reg(tau=12.0), [MeasuredConc(value=1.0, t=72.0)],
        DoseTarget((Constraint("trough", low=0.5, high=2.0),)),
        candidate_intervals=(8.0, 24.0), n_grid=5, n_samples=4000, seed=0,
    )
    assert any("shape" in w.lower() for w in rec.warnings)
