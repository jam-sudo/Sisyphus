"""Dose recommendation (target attainment) over the IV steady-state TDM posterior.

``predict_tdm`` infers the patient's renal-clearance posterior from a measured
steady-state trough. This module closes the clinical loop: given a target (a set of
constraints on the steady-state trough, peak Cmax, and/or AUC24), recommend the
(dose, interval) that maximizes the probability the target is met under that
posterior.

The engine is **linear in dose** (concentration-independent clearance — no saturable
Michaelis-Menten in this path), so at a fixed disposition every steady-state exposure
scales linearly with dose. The dose knob is therefore inverted *analytically* (one
solve per interval, then a max-interval-overlap sweep over per-sample feasible
dose-multiplier ranges); only the interval knob (nonlinear accumulation) costs an
engine re-solve. See docs/superpowers/specs/2026-06-12-mipd-dose-recommendation-design.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sisyphus.mipd.core import Posterior

_QUANTITIES = ("trough", "cmax", "auc24")


@dataclass(frozen=True)
class Constraint:
    """A bound on one steady-state PK quantity, evaluated under the posterior.

    ``quantity`` is one of ``"trough"`` / ``"cmax"`` / ``"auc24"``. At least one of
    ``low`` / ``high`` must be set (mg/L for trough/cmax, mg*h/L for auc24).
    """

    quantity: str
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if self.quantity not in _QUANTITIES:
            raise ValueError(f"quantity must be one of {_QUANTITIES}, got {self.quantity!r}")
        if self.low is None and self.high is None:
            raise ValueError("Constraint needs at least one of low/high")
        if self.low is not None and self.low < 0:
            raise ValueError(f"low must be >= 0, got {self.low}")
        if self.high is not None and self.high <= 0:
            raise ValueError(f"high must be > 0, got {self.high}")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"low {self.low} > high {self.high}")


@dataclass(frozen=True)
class DoseTarget:
    """A set of constraints. Attainment = P(ALL constraints satisfied) under the posterior."""

    constraints: tuple[Constraint, ...]

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError("DoseTarget needs at least one constraint")


@dataclass(frozen=True)
class CandidateEval:
    """One (dose, interval) row of the recommendation search — for transparency."""

    dose_mg: float
    interval_h: float
    attainment_prob: float
    trough_median: float
    cmax_median: float
    auc24_median: float


@dataclass(frozen=True)
class DoseRecommendation:
    """The recommended regimen and the exposure posteriors it produces."""

    dose_mg: float
    interval_h: float
    attainment_prob: float
    cmax: Posterior
    trough: Posterior
    auc24: Posterior
    target: DoseTarget
    candidates: tuple[CandidateEval, ...]
    renal_scale: Posterior
    n_eff: float
    warnings: tuple[str, ...]


def _sample_m_intervals(
    q_ref: dict[str, np.ndarray], target: DoseTarget
) -> tuple[np.ndarray, np.ndarray]:
    """Per posterior-sample feasible dose-multiplier interval ``[m_lo, m_hi]``.

    Each exposure scales linearly with the dose multiplier ``m`` (LTI). A constraint
    ``low <= m*q_ref <= high`` becomes ``m in [low/q_ref, high/q_ref]``; intersecting
    across constraints gives ``[m_lo[i], m_hi[i]]`` per sample (0 / +inf when a side
    is unbounded).

    A sample whose reference exposure is ~0 (floored to 1e-300) gets m_lo -> inf, which
    correctly marks it infeasible.
    """
    n = len(next(iter(q_ref.values())))
    m_lo = np.zeros(n)
    m_hi = np.full(n, np.inf)
    for c in target.constraints:
        q = np.maximum(np.asarray(q_ref[c.quantity], dtype=float), 1e-300)
        if c.low is not None:
            m_lo = np.maximum(m_lo, c.low / q)
        if c.high is not None:
            m_hi = np.minimum(m_hi, c.high / q)
    return m_lo, m_hi


def _attainment(m: float, m_lo: np.ndarray, m_hi: np.ndarray) -> float:
    """Fraction of posterior samples whose feasible interval covers dose-multiplier ``m``."""
    return float(np.mean((m_lo <= m) & (m <= m_hi)))


def _max_overlap_region(
    m_lo: np.ndarray, m_hi: np.ndarray
) -> tuple[float, float, int]:
    """Dose-multiplier segment ``[a, b]`` where the most sample-intervals overlap.

    Classic max-interval-overlap sweep. At a tie, a start is ordered before an end so
    a point shared by ``[., x]`` and ``[x, .]`` counts both (closed intervals). Returns
    ``(a, b, count)``; ``b`` may be ``inf`` (only-floor constraints). Empty sample
    intervals (``m_lo > m_hi``) are dropped.
    """
    feasible = m_lo <= m_hi
    los = m_lo[feasible]
    his = m_hi[feasible]
    if los.size == 0:
        return (0.0, 0.0, 0)
    pts = np.concatenate([los, his])
    kinds = np.concatenate([np.ones(los.size), -np.ones(his.size)])  # +1 start, -1 end
    order = np.lexsort((-kinds, pts))  # by point asc; starts (+1) before ends (-1) at ties
    pts = pts[order]
    cov = np.cumsum(kinds[order])
    best_i = int(np.argmax(cov))
    a = float(pts[best_i])
    b = float(pts[best_i + 1]) if best_i + 1 < pts.size else a
    return (a, b, int(cov[best_i]))


def _center_m(a: float, b: float) -> float:
    """Pick the dose multiplier within the max-overlap region ``[a, b]``.

    Bounded window -> geometric midpoint (max margin). Only floors (b == inf) ->
    smallest dose meeting them (``a``). Only ceilings (a == 0) -> largest dose under
    them (``b``).

    If the region is unbounded above with a==0 (a floor that binds no sample), keep the
    current dose (1.0).
    """
    if not math.isfinite(b):
        return a if a > 0.0 else 1.0
    if a <= 0.0:
        return b
    return math.sqrt(a * b)


def _interval_reference(
    smiles: str,
    regimen,
    tau: float,
    r_samples: np.ndarray,
    *,
    renal_factor: float,
    body_weight_kg: float | None,
    age_years: float | None,
    n_grid: int,
    kp_method: str,
) -> tuple[dict[str, np.ndarray], float]:
    """Per posterior-sample steady-state exposures at the regimen's reference dose.

    Builds one renal-CL grid at this interval (one engine re-solve), then reads each
    quantity at the posterior's renal-scale samples: ``trough`` = the curve at the end
    of the final dosing interval; ``cmax`` / per-interval ``auc`` from the forward;
    ``auc24`` = per-interval AUC * (24/tau). Returns the quantity dict and the
    reference dose ``D_ref`` (= the regimen's per-dose amount).
    """
    from sisyphus.mipd.renal_grid import RenalCLForward, build_renal_cl_grid

    grid = build_renal_cl_grid(
        smiles, regimen, n_grid=n_grid, renal_factor=renal_factor,
        body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
    )
    state = RenalCLForward(grid)(r_samples)
    trough = grid.conc_at(r_samples, float(regimen.last_dose_time_h) + tau)
    q_ref = {
        "trough": np.asarray(trough, dtype=float),
        "cmax": np.asarray(state["cmax"], dtype=float),
        "auc24": np.asarray(state["auc"], dtype=float) * (24.0 / tau),
    }
    d_ref = float(regimen.events[0].dose_mg)
    return q_ref, d_ref


def recommend_dose(*args, **kwargs):  # implemented in Task 4
    raise NotImplementedError("recommend_dose is implemented in a later task")
