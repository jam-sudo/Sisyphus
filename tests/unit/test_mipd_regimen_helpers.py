import pytest

from sisyphus.mipd._regimen import (
    _distinct_phases,
    _regimen_interval_h,
    _regimen_route,
    _require_uniform_regimen,
)
from sisyphus.regimen.types import DosingRegimen


def _oral(dose=100.0, tau=12.0, n=5):
    return DosingRegimen.oral_repeated(dose_mg=dose, interval_h=tau, n_doses=n)


def _iv(dose=100.0, dur=1.0, tau=12.0, n=5):
    return DosingRegimen.iv_infusion(dose_mg=dose, duration_h=dur, interval_h=tau, n_doses=n)


def test_route_oral_and_iv():
    assert _regimen_route(_oral()) == "oral"
    assert _regimen_route(_iv()) == "iv"


def test_route_mixed_raises():
    oral = _oral()
    iv = _iv()
    # Node-mixed but time-sorted: oral at t=0, IV at t=12 (the regimen constructor
    # requires ascending event times; route classification is what must reject the mix).
    mixed = DosingRegimen(events=(oral.events[0], iv.events[1]))
    with pytest.raises(ValueError):
        _regimen_route(mixed)


def test_uniform_ok_nonuniform_raises():
    import dataclasses
    _require_uniform_regimen(_oral(tau=12.0))  # no raise
    reg = _oral(tau=12.0)
    ev = list(reg.events)
    ev[1] = dataclasses.replace(ev[1], time_h=ev[1].time_h + 5.0)
    with pytest.raises(ValueError):
        _require_uniform_regimen(DosingRegimen(events=tuple(ev)))


def test_interval_is_final_interval():
    assert _regimen_interval_h(_oral(tau=8.0)) == pytest.approx(8.0)
    single = DosingRegimen.oral_repeated(dose_mg=100.0, interval_h=12.0, n_doses=1)
    assert _regimen_interval_h(single) == pytest.approx(24.0)


def test_distinct_phases_circular():
    from sisyphus.mipd.clgrid import MeasuredConc
    tau = 12.0
    same = [MeasuredConc(value=1.0, t=12.0), MeasuredConc(value=1.0, t=24.0)]
    assert _distinct_phases(same, tau) is False
    shape = [MeasuredConc(value=5.0, t=2.0), MeasuredConc(value=1.0, t=12.0)]
    assert _distinct_phases(shape, tau) is True
    wrap = [MeasuredConc(value=1.0, t=0.1), MeasuredConc(value=1.0, t=11.9)]
    assert _distinct_phases(wrap, tau) is False
