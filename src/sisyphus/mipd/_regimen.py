"""Shared regimen helpers for the MIPD TDM stack (route, uniformity, interval, shape).

Co-locating these keeps ``tdm.py``/``dosing.py``/``oral_grid.py`` on one contract and
gives ``renal_grid`` a single ``_regimen_interval_h`` to import. All operate on a
``regimen.types.DosingRegimen`` and are identity-blind.
"""
from __future__ import annotations

import numpy as np

from sisyphus.regimen.types import DEFAULT_IV_NODE, DEFAULT_ORAL_NODE

# Phase-distinctness tolerance as a fraction of tau (spec §6).
_SHAPE_PHASE_TOL_FRAC: float = 0.1


def _regimen_route(regimen) -> str:
    """'iv' if every event targets the IV node, 'oral' if every event the oral node."""
    nodes = {ev.node for ev in regimen.events}
    if nodes == {DEFAULT_IV_NODE}:
        return "iv"
    if nodes == {DEFAULT_ORAL_NODE}:
        return "oral"
    raise ValueError(
        f"regimen mixes/uses unsupported administration nodes {sorted(nodes)!r}; "
        f"TDM supports a pure IV ({DEFAULT_IV_NODE!r}) or pure oral "
        f"({DEFAULT_ORAL_NODE!r}) regimen."
    )


def _require_uniform_regimen(regimen) -> None:
    """Raise ``ValueError`` if dosing intervals are non-uniform (>~1% spread)."""
    times = np.array([ev.time_h for ev in regimen.events], dtype=float)
    if times.size < 3:
        return
    gaps = np.diff(times)
    median = float(np.median(gaps))
    if median <= 0:
        raise ValueError("regimen event times are non-increasing")
    if float(np.max(np.abs(gaps - median))) > 0.01 * median:
        raise ValueError(
            "non-uniform dosing interval detected; oral/IV steady-state TDM assumes "
            "a uniform interval (non-uniform regimens are out of scope)"
        )


def _regimen_interval_h(regimen) -> float:
    """The dosing interval tau (h): the FINAL interval, or 24.0 for a single dose."""
    events = regimen.events
    if len(events) < 2:
        return 24.0
    return float(events[-1].time_h - events[-2].time_h)


def _distinct_phases(observations, tau: float) -> bool:
    """True if the MeasuredConc phases span distinct within-interval positions.

    Phase ``phi = t mod tau``; distinctness uses the maximum pairwise CIRCULAR
    distance ``min(|dphi|, tau-|dphi|)`` so a 0/tau pair reads as same-phase.
    Only obs with a ``.t`` (MeasuredConc) are considered.
    """
    phis = [float(o.t) % tau for o in observations if hasattr(o, "t")]
    if len(phis) < 2:
        return False
    tol = _SHAPE_PHASE_TOL_FRAC * tau
    max_d = 0.0
    for i in range(len(phis)):
        for j in range(i + 1, len(phis)):
            d = abs(phis[i] - phis[j])
            d = min(d, tau - d)
            max_d = max(max_d, d)
    return max_d > tol
