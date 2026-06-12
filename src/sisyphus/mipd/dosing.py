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

import math  # noqa: F401  (used by Task 4 solver)
from dataclasses import dataclass

import numpy as np  # noqa: F401  (used by Task 4 solver)

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


def recommend_dose(*args, **kwargs):  # implemented in Task 4
    raise NotImplementedError("recommend_dose is implemented in a later task")
