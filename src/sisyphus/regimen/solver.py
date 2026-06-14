"""Event-driven multi-dose solver.

Wraps ``sisyphus.engine.solver.solve()`` to support dosing regimens.
Does NOT modify the engine RHS — doses are injected into the state
vector between integration segments.

Infusions are approximated as sequences of small boluses (every
INFUSION_STEP_H) to avoid modifying the engine RHS.

Strategy:
    1. Expand infusion events into discrete micro-boluses.
    2. Sort all bolus events by time.
    3. Integrate segment-by-segment: inject dose → solve → capture.
    4. Concatenate segments into a single SimResult.
"""

from __future__ import annotations

import logging

import numpy as np

from sisyphus.core import SimResult
from sisyphus.engine.compiler import CompiledODE, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.regimen.types import INFUSION_STEP_H, DosingEvent, DosingRegimen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Short infusions (< this threshold) are treated as a single bolus
SHORT_INFUSION_THRESHOLD_H: float = 0.5

# Default observation window after last dose (hours)
DEFAULT_TAIL_H: float = 24.0

# Minimum segment duration to avoid degenerate solver calls (hours)
MIN_SEGMENT_H: float = 1e-6


# ---------------------------------------------------------------------------
# Internal: expand infusions into micro-boluses
# ---------------------------------------------------------------------------


def _expand_infusions(events: tuple[DosingEvent, ...]) -> list[DosingEvent]:
    """Expand infusion events into sequences of small boluses.

    Short infusions (duration_h < SHORT_INFUSION_THRESHOLD_H) are kept
    as a single bolus at the start time. Longer infusions are split
    into micro-boluses every INFUSION_STEP_H.

    Args:
        events: Original dosing events (may contain infusions).

    Returns:
        List of bolus-only DosingEvents, sorted by time.
    """
    boluses: list[DosingEvent] = []

    for ev in events:
        if ev.is_bolus or ev.duration_h < SHORT_INFUSION_THRESHOLD_H:
            # Treat as single bolus at start time
            boluses.append(
                DosingEvent(
                    time_h=ev.time_h,
                    dose_mg=ev.dose_mg,
                    node=ev.node,
                    duration_h=0.0,
                )
            )
        else:
            # Split infusion into micro-boluses
            n_steps = max(1, int(round(ev.duration_h / INFUSION_STEP_H)))
            dose_per_step = ev.dose_mg / n_steps
            for i in range(n_steps):
                t = ev.time_h + i * (ev.duration_h / n_steps)
                boluses.append(
                    DosingEvent(
                        time_h=t,
                        dose_mg=dose_per_step,
                        node=ev.node,
                        duration_h=0.0,
                    )
                )

    # Sort by time (stable sort preserves order of simultaneous events)
    boluses.sort(key=lambda e: e.time_h)
    return boluses


# ---------------------------------------------------------------------------
# Internal: group simultaneous boluses
# ---------------------------------------------------------------------------


def _group_simultaneous(
    boluses: list[DosingEvent],
    tol: float = 1e-10,
) -> list[tuple[float, list[DosingEvent]]]:
    """Group boluses that occur at the same time.

    Args:
        boluses: Sorted list of bolus events.
        tol: Time tolerance for grouping (hours).

    Returns:
        List of (time, [events_at_time]) pairs, ordered by time.
    """
    if not boluses:
        return []

    groups: list[tuple[float, list[DosingEvent]]] = []
    current_time = boluses[0].time_h
    current_group: list[DosingEvent] = [boluses[0]]

    for b in boluses[1:]:
        if abs(b.time_h - current_time) <= tol:
            current_group.append(b)
        else:
            groups.append((current_time, current_group))
            current_time = b.time_h
            current_group = [b]

    groups.append((current_time, current_group))
    return groups


# ---------------------------------------------------------------------------
# Public API: solve_regimen
# ---------------------------------------------------------------------------


def solve_regimen(
    compiled: CompiledODE,
    params: ResolvedParams,
    regimen: DosingRegimen,
    t_total_h: float | None = None,
    dt_output: float = 0.1,
) -> SimResult:
    """Solve the ODE system for a multi-dose regimen.

    Wraps ``engine.solver.solve()`` in a segment-by-segment loop:
    each dosing event triggers a new integration segment. The final
    state of one segment becomes the initial state of the next,
    with the new dose injected into the appropriate state variable.

    Args:
        compiled: Compiled ODE skeleton (from ``ODECompiler.compile()``).
        params: Resolved point-value parameters.
        regimen: Dosing regimen specifying all dose events.
        t_total_h: Total simulation time (hours). If ``None``, defaults
            to ``last_dose_time + DEFAULT_TAIL_H``.
        dt_output: Output time resolution (hours). Default 0.1h (6 min).

    Returns:
        A single SimResult spanning the entire simulation window,
        with concatenated time/concentration/amount arrays.
    """
    if t_total_h is None:
        t_total_h = regimen.last_dose_time_h + DEFAULT_TAIL_H

    # Expand infusions into micro-boluses
    boluses = _expand_infusions(regimen.events)
    dose_groups = _group_simultaneous(boluses)

    # Build the uniform output time grid
    n_points = max(2, int(round(t_total_h / dt_output)) + 1)
    t_grid = np.linspace(0.0, t_total_h, n_points)

    # Collect segment results
    all_time: list[np.ndarray] = []
    all_amounts: dict[str, list[np.ndarray]] = {
        name: [] for name in compiled.state_index
    }
    all_concentrations: dict[str, list[np.ndarray]] = {
        name: [] for name in compiled.state_index
    }

    # State vector — starts at zero
    y_current = np.zeros(compiled.n_states)

    # Track overall solver success and mass balance
    overall_success = True
    max_mbe = 0.0

    # Build the list of segment boundaries:
    # Each dose group triggers a segment boundary
    segment_starts: list[float] = []
    dose_map: dict[int, list[DosingEvent]] = {}  # segment_index -> doses to inject

    for i, (t_dose, events) in enumerate(dose_groups):
        segment_starts.append(t_dose)
        dose_map[i] = events

    # Add final time if not already a dose time
    if not segment_starts or segment_starts[-1] < t_total_h:
        segment_starts.append(t_total_h)

    logger.debug(
        "solve_regimen: %d dose groups, %d segments, t_total=%.1fh",
        len(dose_groups),
        len(segment_starts) - 1,
        t_total_h,
    )

    for seg_idx in range(len(segment_starts) - 1):
        t_start = segment_starts[seg_idx]
        t_end = segment_starts[seg_idx + 1]

        # Inject doses at this segment's start time
        if seg_idx in dose_map:
            for ev in dose_map[seg_idx]:
                node_idx = compiled.state_index.get(ev.node)
                if node_idx is None:
                    logger.warning(
                        "Dose target node %r not found in compiled graph; skipping",
                        ev.node,
                    )
                    continue
                y_current[node_idx] += ev.dose_mg
                logger.debug(
                    "Injected %.2f mg into %r at t=%.2fh",
                    ev.dose_mg,
                    ev.node,
                    t_start,
                )

        # Skip degenerate segments
        if t_end - t_start < MIN_SEGMENT_H:
            continue

        # Determine output points for this segment from the global grid
        seg_t_eval = t_grid[(t_grid >= t_start - 1e-12) & (t_grid <= t_end + 1e-12)]
        if len(seg_t_eval) < 2:
            seg_t_eval = np.array([t_start, t_end])

        # Ensure endpoints are included
        if seg_t_eval[0] > t_start + 1e-12:
            seg_t_eval = np.concatenate(([t_start], seg_t_eval))
        if seg_t_eval[-1] < t_end - 1e-12:
            seg_t_eval = np.concatenate((seg_t_eval, [t_end]))

        # Clamp to the segment span and de-duplicate: a global-grid point can sit
        # ~1e-15 past t_end from float rounding (first surfaced by IV-infusion
        # micro-bolus segments where dt_output is commensurate with the infusion
        # step), and solve_ivp strictly rejects any t_eval outside [t_start, t_end].
        seg_t_eval = np.unique(np.clip(seg_t_eval, t_start, t_end))

        # Solve this segment
        result = solve(
            compiled,
            params,
            y_current,
            t_span=(t_start, t_end),
            t_eval=seg_t_eval,
        )

        if not result.solver_success:
            logger.warning(
                "Solver failed for segment [%.2f, %.2f]h",
                t_start,
                t_end,
            )
            overall_success = False

        max_mbe = max(max_mbe, result.mass_balance_error)

        # Determine which time points to keep (avoid duplicating boundary
        # points from adjacent segments)
        if all_time:
            # Skip first point if it duplicates the last point of the
            # previous segment
            start_idx = 1 if len(result.time_h) > 1 else 0
        else:
            start_idx = 0

        all_time.append(result.time_h[start_idx:])
        for name in compiled.state_index:
            all_amounts[name].append(result.amounts[name][start_idx:])
            all_concentrations[name].append(result.concentrations[name][start_idx:])

        # Update state vector to final state of this segment
        for name, idx in compiled.state_index.items():
            y_current[idx] = result.amounts[name][-1]

    # Handle edge case: if all segments were degenerate, return minimal result
    if not all_time:
        return SimResult(
            time_h=np.array([0.0]),
            concentrations={name: np.array([0.0]) for name in compiled.state_index},
            amounts={name: np.array([0.0]) for name in compiled.state_index},
            mass_balance_error=0.0,
            solver_success=False,
        )

    # Concatenate all segments
    time_concat = np.concatenate(all_time)
    conc_concat = {
        name: np.concatenate(arrs) for name, arrs in all_concentrations.items()
    }
    amt_concat = {
        name: np.concatenate(arrs) for name, arrs in all_amounts.items()
    }

    # Recompute mass balance over full time course using cumulative dose
    # at each time point (fixes spurious MBE for multi-dose regimens where
    # early time points only have the first dose administered).
    total = np.zeros_like(time_concat)
    for name in compiled.state_index:
        total += amt_concat[name]

    cumulative_dose = np.zeros_like(time_concat)
    for ev in regimen.events:
        cumulative_dose[time_concat >= ev.time_h - 1e-12] += ev.dose_mg

    valid = cumulative_dose > 0
    if np.any(valid):
        full_mbe = float(np.max(np.abs(total[valid] - cumulative_dose[valid]) / cumulative_dose[valid]))  # noqa: E501
    else:
        full_mbe = 0.0

    return SimResult(
        time_h=time_concat,
        concentrations=conc_concat,
        amounts=amt_concat,
        mass_balance_error=full_mbe,
        solver_success=overall_success,
    )
