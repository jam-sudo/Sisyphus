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

    Precondition: ``tau`` must equal the regimen's own dosing interval (the orchestrator
    builds the regimen and tau together), so the trough time stays within the grid horizon.
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


_INFEASIBLE_ATTAINMENT = 0.5  # soft-warn when the best candidate falls below this
_TIE_EPS = 1e-3               # attainment ties within this prefer the longer interval


def _tie_tolerance(n_samples: int) -> float:
    """Attainment-tie band: ~2x the binomial SE of a proportion at ``n_samples``.

    Attainment is ``mean(boolean)`` over the posterior particles, so its sampling SE
    is ``sqrt(p(1-p)/n) <= sqrt(0.25/n)``. Two intervals whose attainment differs by
    less than this are statistically tied and the longer-interval preference should
    decide between them; a fixed ``_TIE_EPS`` below the SE would let MC noise defeat
    that rule. Floored at ``_TIE_EPS``.
    """
    return max(_TIE_EPS, 2.0 * math.sqrt(0.25 / max(n_samples, 1)))


def recommend_dose(
    smiles: str,
    regimen,
    observations,
    target: DoseTarget,
    *,
    covariates=None,
    candidate_intervals: tuple[float, ...] | None = None,
    dose_step_mg: float | None = None,
    dose_bounds_mg: tuple[float, float] | None = None,
    renal_prior_cv: float = 1.0,
    n_samples: int = 20000,
    n_grid: int = 13,
    seed: int = 0,
    kp_method: str = "rodgers_rowland",
) -> DoseRecommendation:
    """Recommend the (dose, interval) that best attains ``target`` under the posterior.

    ``regimen`` is the CURRENT IV regimen the ``observations`` (steady-state troughs)
    were measured under. The renal-CL posterior is inferred once (a patient property),
    then propagated to each candidate interval; the dose is solved analytically per
    interval (LTI). The winner maximizes attainment, breaking ties toward the longer
    interval. IV-only (oral steady-state TDM is a future layer).

    Callers should check ``attainment_prob`` (and ``warnings``) before acting: an
    infeasible target still returns the best-effort longest-interval candidate.
    Steady state for a candidate interval is approximated by ``n_doses =
    max(2, round(cur_last/tau)+1)`` doses; a slowly-accumulating (long-half-life) drug
    at a long candidate interval may be modeled slightly below true steady state.
    The current regimen is assumed uniform (dose/duration/interval read from its first
    events); a non-uniform input regimen is reconstructed as uniform.
    """
    from sisyphus.mipd.tdm import predict_tdm
    from sisyphus.regimen.types import DEFAULT_IV_NODE, DosingRegimen

    if any(ev.node != DEFAULT_IV_NODE for ev in regimen.events):
        raise ValueError(
            "recommend_dose supports IV regimens only (every event must target the IV "
            f"node {DEFAULT_IV_NODE!r}); oral steady-state TDM is a future extension."
        )

    if float(regimen.events[0].dose_mg) <= 0.0:
        raise ValueError("recommend_dose requires the current regimen's dose to be > 0")

    observations = list(observations)
    post = predict_tdm(
        smiles, regimen, observations, covariates=covariates,
        renal_prior_cv=renal_prior_cv, n_samples=n_samples, n_grid=n_grid,
        seed=seed, kp_method=kp_method,
    )
    r_samples = post.renal_scale.samples
    warnings_list = list(post.warnings)

    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    body_weight_kg = covariates.body_weight_kg if covariates is not None else None
    age_years = covariates.age_years if covariates is not None else None

    cur_dose = float(regimen.events[0].dose_mg)
    cur_dur = float(regimen.events[0].duration_h)
    cur_last = float(regimen.last_dose_time_h)
    cur_tau = (
        float(regimen.events[1].time_h - regimen.events[0].time_h)
        if regimen.n_doses >= 2 else 24.0
    )

    base = tuple(candidate_intervals) if candidate_intervals is not None else (8.0, 12.0, 24.0)
    taus = sorted(set(base + (cur_tau,)))

    rows: list[tuple[CandidateEval, dict[str, np.ndarray], float]] = []
    for tau in taus:
        n_doses = max(2, int(round(cur_last / tau)) + 1)
        reg_tau = DosingRegimen.iv_infusion(
            dose_mg=cur_dose, duration_h=cur_dur, interval_h=tau, n_doses=n_doses
        )
        q_ref, d_ref = _interval_reference(
            smiles, reg_tau, tau, r_samples, renal_factor=renal_factor,
            body_weight_kg=body_weight_kg, age_years=age_years, n_grid=n_grid,
            kp_method=kp_method,
        )
        m_lo, m_hi = _sample_m_intervals(q_ref, target)
        a, b, _ = _max_overlap_region(m_lo, m_hi)
        dose = _center_m(a, b) * d_ref
        if dose_step_mg:
            dose = round(dose / dose_step_mg) * dose_step_mg
        if dose_bounds_mg is not None:
            dose = min(max(dose, dose_bounds_mg[0]), dose_bounds_mg[1])
        m_actual = dose / d_ref
        attain = _attainment(m_actual, m_lo, m_hi)
        rows.append((
            CandidateEval(
                dose_mg=float(dose), interval_h=float(tau), attainment_prob=attain,
                trough_median=float(np.median(q_ref["trough"] * m_actual)),
                cmax_median=float(np.median(q_ref["cmax"] * m_actual)),
                auc24_median=float(np.median(q_ref["auc24"] * m_actual)),
            ),
            q_ref, m_actual,
        ))

    best_attain = max(row[0].attainment_prob for row in rows)
    tie_eps = _tie_tolerance(r_samples.size)
    winners = [row for row in rows if row[0].attainment_prob >= best_attain - tie_eps]
    win_cand, win_q, win_m = max(winners, key=lambda row: row[0].interval_h)

    if win_cand.dose_mg <= 0.0:
        warnings_list.append(
            f"recommended dose rounded to {win_cand.dose_mg:g} mg under the dose "
            f"granularity (dose_step_mg={dose_step_mg}); the optimal dose is below one step"
        )

    if win_cand.attainment_prob < _INFEASIBLE_ATTAINMENT:
        warnings_list.append(
            f"best attainment {win_cand.attainment_prob:.2f} < "
            f"{_INFEASIBLE_ATTAINMENT:.2f}; target may be infeasible for this patient"
        )

    return DoseRecommendation(
        dose_mg=win_cand.dose_mg,
        interval_h=win_cand.interval_h,
        attainment_prob=win_cand.attainment_prob,
        cmax=Posterior(win_q["cmax"] * win_m),
        trough=Posterior(win_q["trough"] * win_m),
        auc24=Posterior(win_q["auc24"] * win_m),
        target=target,
        candidates=tuple(row[0] for row in rows),
        renal_scale=post.renal_scale,
        n_eff=post.n_eff,
        warnings=tuple(warnings_list),
    )
