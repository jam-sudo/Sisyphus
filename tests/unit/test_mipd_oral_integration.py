"""Integration guards for the oral steady-state TDM path.

Regression locks that the Task-1/3 route/uniformity guards reject malformed
regimens before any engine work: a node-mixed regimen and a non-uniform oral
regimen must both raise ``ValueError`` from ``predict_tdm``.
"""
import dataclasses

import pytest

from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.tdm import predict_tdm
from sisyphus.regimen.types import DosingRegimen

SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def test_mixed_route_regimen_raises():
    oral = DosingRegimen.oral_repeated(dose_mg=100.0, interval_h=12.0, n_doses=3)
    iv = DosingRegimen.iv_infusion(dose_mg=100.0, duration_h=1.0, interval_h=12.0, n_doses=2)
    # Node-mixed but time-ascending so the regimen constructor accepts it (it
    # validates strictly-ascending times BEFORE route classification runs):
    # oral at t=0, IV at t=12. The route classifier must reject the mixed nodes.
    assert iv.events[-1].time_h > oral.events[0].time_h
    mixed = DosingRegimen(events=(oral.events[0], iv.events[-1]))
    with pytest.raises(ValueError):
        predict_tdm(SMILES, mixed, [MeasuredConc(value=1.0, t=12.0)])


def test_nonuniform_oral_raises():
    reg = DosingRegimen.oral_repeated(dose_mg=100.0, interval_h=12.0, n_doses=4)
    ev = list(reg.events)
    ev[2] = dataclasses.replace(ev[2], time_h=ev[2].time_h + 6.0)
    with pytest.raises(ValueError):
        predict_tdm(SMILES, DosingRegimen(events=tuple(ev)), [MeasuredConc(value=1.0, t=48.0)])
