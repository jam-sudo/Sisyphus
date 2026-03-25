"""Concentration profile analysis and steady-state detection.

Provides post-processing of multi-dose SimResult to extract
ConcentrationProfile (with uncertainty bounds from MC) and
SteadyStateMetrics (Css, accumulation ratio, SS detection).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sisyphus.core import SimResult
from sisyphus.regimen.types import DosingRegimen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Steady-state convergence threshold: consecutive Cmax values
# must differ by less than this fraction to be considered at SS
SS_CONVERGENCE_THRESHOLD: float = 0.05

# Percentile bounds for prediction intervals
CI_LOWER_PERCENTILE: float = 5.0
CI_UPPER_PERCENTILE: float = 95.0


# ---------------------------------------------------------------------------
# ConcentrationProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcentrationProfile:
    """Concentration time-course with uncertainty bounds.

    Produced by aggregating multiple MC SimResult replicates at a
    single observation node.

    Attributes:
        time_h: Time points (hours).
        median: Median concentration at each time point (mg/L).
        ci_lower: 5th percentile concentration (mg/L).
        ci_upper: 95th percentile concentration (mg/L).
        node: Name of the observation node.
    """

    time_h: NDArray[np.float64]
    median: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    node: str


# ---------------------------------------------------------------------------
# SteadyStateMetrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SteadyStateMetrics:
    """Steady-state pharmacokinetic metrics from a multi-dose simulation.

    Attributes:
        css_max: Steady-state peak concentration (mg/L).
        css_min: Steady-state trough concentration (mg/L).
        accumulation_ratio: Css_max / C1_max — how much drug accumulates.
        is_steady_state: True if consecutive Cmax values differ < 5%.
        dose_to_steady_state: Dose number at which SS was first reached
            (1-indexed). 0 if SS was not reached within the simulation.
    """

    css_max: float
    css_min: float
    accumulation_ratio: float
    is_steady_state: bool
    dose_to_steady_state: int


# ---------------------------------------------------------------------------
# Public API: build ConcentrationProfile from MC replicates
# ---------------------------------------------------------------------------


def build_concentration_profile(
    results: list[SimResult],
    node: str,
) -> ConcentrationProfile:
    """Build a ConcentrationProfile from multiple MC SimResult replicates.

    All results must share the same time grid (same t_eval was used).

    Args:
        results: List of SimResult instances from MC sampling.
        node: Observation node name (e.g. ``"venous_blood"``).

    Returns:
        ConcentrationProfile with median and 90% PI.

    Raises:
        ValueError: If results is empty or node not found.
    """
    if not results:
        raise ValueError("results list must not be empty")
    if node not in results[0].concentrations:
        raise ValueError(f"Node {node!r} not found in SimResult concentrations")

    time_h = results[0].time_h

    # Stack all MC replicates: shape (n_mc, n_timepoints)
    conc_matrix = np.array([r.concentrations[node] for r in results])

    median = np.percentile(conc_matrix, 50.0, axis=0)
    ci_lower = np.percentile(conc_matrix, CI_LOWER_PERCENTILE, axis=0)
    ci_upper = np.percentile(conc_matrix, CI_UPPER_PERCENTILE, axis=0)

    return ConcentrationProfile(
        time_h=time_h,
        median=median,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        node=node,
    )


# ---------------------------------------------------------------------------
# Public API: compute steady-state metrics
# ---------------------------------------------------------------------------


def compute_steady_state_metrics(
    result: SimResult,
    regimen: DosingRegimen,
    node: str = "venous_blood",
) -> SteadyStateMetrics:
    """Compute steady-state metrics from a multi-dose SimResult.

    Identifies peak (Cmax) and trough (Cmin) concentrations within
    each dosing interval, then checks for convergence.

    Args:
        result: SimResult from a multi-dose ``solve_regimen()`` call.
        regimen: The DosingRegimen used to generate the result.
        node: Observation node for concentration extraction.

    Returns:
        SteadyStateMetrics with accumulation ratio and SS detection.

    Raises:
        ValueError: If node not found or regimen has < 1 dose.
    """
    if node not in result.concentrations:
        raise ValueError(f"Node {node!r} not found in SimResult concentrations")

    conc = result.concentrations[node]
    time = result.time_h

    events = regimen.events

    # Extract per-dose Cmax and Cmin values
    dose_cmax: list[float] = []
    dose_cmin: list[float] = []

    for i, ev in enumerate(events):
        t_start = ev.time_h
        # Interval ends at next dose time or end of simulation
        if i + 1 < len(events):
            t_end = events[i + 1].time_h
        else:
            t_end = time[-1]

        # Find time points within this dosing interval
        mask = (time >= t_start - 1e-12) & (time <= t_end + 1e-12)
        interval_conc = conc[mask]

        if len(interval_conc) == 0:
            continue

        dose_cmax.append(float(np.max(interval_conc)))
        dose_cmin.append(float(interval_conc[-1]))  # trough = end of interval

    if not dose_cmax:
        return SteadyStateMetrics(
            css_max=0.0,
            css_min=0.0,
            accumulation_ratio=1.0,
            is_steady_state=False,
            dose_to_steady_state=0,
        )

    # Steady-state detection: consecutive Cmax values differ < threshold
    is_ss = False
    ss_dose = 0  # 0 = not reached

    for i in range(1, len(dose_cmax)):
        if dose_cmax[i - 1] > 0:
            rel_change = abs(dose_cmax[i] - dose_cmax[i - 1]) / dose_cmax[i - 1]
            if rel_change < SS_CONVERGENCE_THRESHOLD:
                is_ss = True
                ss_dose = i + 1  # 1-indexed
                break

    # Css = last dosing interval values (or last available)
    css_max = dose_cmax[-1]
    css_min = dose_cmin[-1] if dose_cmin else 0.0

    # Accumulation ratio: Css_max / C1_max
    c1_max = dose_cmax[0] if dose_cmax[0] > 0 else 1e-12
    accumulation_ratio = css_max / c1_max

    return SteadyStateMetrics(
        css_max=css_max,
        css_min=css_min,
        accumulation_ratio=accumulation_ratio,
        is_steady_state=is_ss,
        dose_to_steady_state=ss_dose,
    )
