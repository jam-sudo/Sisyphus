"""ECM generalization test classifier.

Pure functions for pre-registered pass/fail logic per spec
docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md.

Separation from the execution script enables fast unit tests and keeps
the classification logic frozen independently of orchestration changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


_FE_GATE_LOG10 = 0.48  # |log10 FE| <= 0.48 iff FE <= 3.02
_MODE_B_MAGNITUDE = 0.5  # |median log10 FE of failures| > 0.5 for Mode B


class Mode(str, Enum):
    """Aggregate outcome taxonomy."""

    A = "A"  # all-pass
    B = "B"  # systematic bias (same-direction fail, magnitude > 0.5)
    C = "C"  # inconclusive (default)
    D = "D"  # all-fail mixed direction


@dataclass(frozen=True)
class DrugOutcome:
    """Per-drug classification result."""

    drug: str
    observed: float
    point_estimate: float
    pi_low: float
    pi_high: float
    log10_fe: float
    passed: bool


def classify_drug(
    drug: str,
    observed: float,
    point_estimate: float,
    pi_low: float,
    pi_high: float,
) -> DrugOutcome:
    """Classify one drug as pass/fail per spec §Per-drug criterion.

    Passes iff:
    1. 90% PI contains observed.
    2. |log10 FE| <= 0.48 (= FE <= 3.02x).
    """
    if observed <= 0 or point_estimate <= 0:
        raise ValueError("observed and point_estimate must be positive")
    log10_fe = math.log10(point_estimate / observed)
    pi_contains = pi_low <= observed <= pi_high
    fe_ok = abs(log10_fe) <= _FE_GATE_LOG10
    return DrugOutcome(
        drug=drug,
        observed=observed,
        point_estimate=point_estimate,
        pi_low=pi_low,
        pi_high=pi_high,
        log10_fe=log10_fe,
        passed=pi_contains and fe_ok,
    )


def classify_aggregate(outcomes: list[DrugOutcome]) -> Mode:
    """Classify the set of per-drug outcomes into Mode A/B/C/D.

    Precedence: A → B → D → C (C is the fallback).

    - A: all pass.
    - B: (>=2 failures with same-direction log10 FE) AND
         |median log10 FE of failures| > 0.5. Includes 3/3 same-direction fail.
    - D: 3/3 fail AND failures are NOT same-direction (mixed signs).
    - C: everything else.
    """
    n = len(outcomes)
    failures = [o for o in outcomes if not o.passed]
    n_fail = len(failures)

    if n_fail == 0:
        return Mode.A

    fail_signs = {1 if o.log10_fe > 0 else -1 for o in failures}
    same_direction = len(fail_signs) == 1

    if n_fail >= 2 and same_direction:
        median_log10_fe = _median([abs(o.log10_fe) for o in failures])
        if median_log10_fe > _MODE_B_MAGNITUDE:
            return Mode.B

    if n_fail == n and not same_direction:
        return Mode.D

    return Mode.C


def _median(values: list[float]) -> float:
    s = sorted(values)
    k = len(s)
    if k == 0:
        return 0.0
    if k % 2 == 1:
        return s[k // 2]
    return 0.5 * (s[k // 2 - 1] + s[k // 2])
