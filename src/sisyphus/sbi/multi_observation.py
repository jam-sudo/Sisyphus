"""Fixed-width context for the time-aware SBI posterior."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sisyphus.regimen.types import DEFAULT_ORAL_NODE, DosingRegimen
from sisyphus.sbi.multi_drug import N_DRUG_FEATURES

# Source: scripts/run_tdm_benchmark.py evaluates up to three observation times.
MAX_OBSERVATION_COUNT = 3
# Source: 24 hours per day, used to scale regimen and sample times.
HOURS_PER_DAY_H = 24.0
# Source: scripts/run_tdm_benchmark.py evaluates 1–3 oral doses.
MAX_DOSE_COUNT = 3
# Source: TDM benchmark observes at 1, 4, and 8 h; 2 and 12 h extend that grid.
OBSERVATION_OFFSETS_H = (1.0, 2.0, 4.0, 8.0, 12.0)
# Source: common once, twice, or three-times-daily oral dosing schedules.
DOSING_INTERVALS_H = (8.0, 12.0, 24.0)
# Source: sisyphus/sbi/simulator.py 10% assay CV in log10 concentration.
ASSAY_SIGMA_LOG10 = 0.0414


def pack_observations(
    observations: Sequence[tuple[float, float]],
    regimen: DosingRegimen,
    drug_features: np.ndarray,
) -> np.ndarray:
    """Pack observed (time_h, concentration_mg_l) pairs, regimen, and features."""
    if not 1 <= len(observations) <= MAX_OBSERVATION_COUNT:
        raise ValueError(f"Expected 1–{MAX_OBSERVATION_COUNT} observations")
    if len(drug_features) != N_DRUG_FEATURES:
        raise ValueError(f"Expected {N_DRUG_FEATURES} drug features")
    events = regimen.events
    dose = events[0].dose_mg
    if len(events) > MAX_DOSE_COUNT or events[0].time_h != 0 or dose <= 0 or any(
        event.dose_mg != dose or event.duration_h != 0 or event.node != DEFAULT_ORAL_NODE
        for event in events
    ):
        raise ValueError("Time-aware SBI requires equal bolus doses starting at t=0")
    interval = events[1].time_h if len(events) > 1 else 0.0
    if len(events) > 1 and (interval <= 0 or any(
        event.time_h != i * interval for i, event in enumerate(events)
    )):
        raise ValueError("Time-aware SBI requires a fixed dosing interval")

    obs = np.zeros((MAX_OBSERVATION_COUNT, 3), dtype=np.float32)
    for i, (time_h, concentration) in enumerate(observations):
        if (time_h < 0 or not np.isfinite(time_h)
                or concentration <= 0 or not np.isfinite(concentration)):
            raise ValueError("Observation time and concentration must be finite and positive")
        if not any(np.isclose(time_h - regimen.last_dose_time_h, offset)
                   for offset in OBSERVATION_OFFSETS_H):
            raise ValueError("Observation time is outside the time-aware training grid")
        obs[i] = (time_h / HOURS_PER_DAY_H, np.log10(concentration), 1.0)
    return np.concatenate([
        obs.ravel(),
        np.array([
            np.log10(dose), interval / HOURS_PER_DAY_H, len(events) / MAX_DOSE_COUNT
        ], dtype=np.float32),
        np.asarray(drug_features, dtype=np.float32),
    ])


def _simulate_context(
    sim,
    theta: np.ndarray,
    rng: np.random.Generator,
    regimen: DosingRegimen,
    times_h: Sequence[float],
    features: np.ndarray,
) -> np.ndarray | None:
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.regimen.solver import solve_regimen
    from sisyphus.sbi.simulator import apply_theta_to_drug

    overridden = apply_theta_to_drug(sim.nominal_drug, theta, sim.nominal_peff)
    realized_graph = sim.graph.sample(rng)
    realized_drug = overridden.sample(rng)
    try:
        result = solve_regimen(
            sim.compiled, ResolvedParams(realized_graph, realized_drug), regimen,
            t_total_h=max(times_h) + 1.0, dt_output=0.25,
        )
        if not result.solver_success:
            return None
        conc = result.concentrations[sim.obs_node]
        measured = [
            (t, float(np.interp(t, result.time_h, conc))
             * 10 ** rng.normal(0, ASSAY_SIGMA_LOG10))
            for t in times_h
        ]
        if any(c <= 0 or not np.isfinite(c) for _, c in measured):
            return None
        return pack_observations(measured, regimen, features)
    except Exception:
        return None
