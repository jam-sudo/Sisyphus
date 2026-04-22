"""ODE solver wrapper.

Wraps SciPy's ``solve_ivp`` with PBPK-appropriate defaults
(LSODA method, mass-balance checking, event detection).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from sisyphus.core import SimResult
from sisyphus.engine.compiler import CompiledODE, ResolvedParams

# Clinical first-draw convention for IV bolus Cmax (5 min post-injection).
# Used by route-aware Cmax extraction to skip the deterministic t=0 spike
# (see docs/superpowers/specs/2026-04-22-iv-cmax-observation-design.md §5).
_IV_CMAX_DELAY_H = 5.0 / 60.0


def solve(
    compiled: CompiledODE,
    params: ResolvedParams,
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
    t_min_h: float = 0.0,
) -> SimResult:
    """Solve the ODE system for a single parameter realization.

    Args:
        compiled: Compiled ODE skeleton.
        params: Resolved point-value parameters.
        y0: Initial state vector (amounts in mg).
        t_span: Integration interval ``(t_start, t_end)`` in hours.
        t_eval: Optional time points for output.  If ``None``, 500
            evenly-spaced points are used.
        t_min_h: Minimum observation time in hours.  When > 0 and
            ``t_eval`` is ``None``, injects ``t_min_h`` as a guaranteed
            anchor in ``t_eval`` so windowed Cmax extraction always has
            a sample at this boundary.  Default ``0.0`` (V2-compatible).
            Ignored when ``t_eval`` is supplied explicitly.

    Returns:
        A SimResult with concentration and amount time series.
    """
    rhs = compiled.make_rhs(params)

    if t_eval is None:
        if t_min_h > 0.0:
            # linspace starts exactly at t_min_h (IEEE 754 exact), so the
            # anchor is guaranteed without needing a separate prepended copy.
            t_eval = np.concatenate(
                [[0.0], np.linspace(t_min_h, t_span[1], 499)]
            )
        else:
            t_eval = np.linspace(t_span[0], t_span[1], 500)

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    # Build named concentration and amount dicts
    # Blood pool: C = A / V (blood concentration, mg/L)
    # Tissue:     C = A / (V * Kp) (plasma-equivalent concentration)
    # Sinks/lumen: raw amount
    concentrations = {}
    amounts = {}
    for name, idx in compiled.state_index.items():
        amounts[name] = sol.y[idx]
        v = params.node_param(name, "volume")
        if v > 0:
            if params.is_blood_pool(name):
                concentrations[name] = sol.y[idx] / v
            else:
                kp = params.drug_kp(name)
                concentrations[name] = sol.y[idx] / (v * kp) if kp > 0 else sol.y[idx] / v
        else:
            concentrations[name] = sol.y[idx]  # sinks, etc.

    # Mass balance check: sum of all states should equal dose at all times
    total = np.zeros_like(sol.t)
    for idx in range(compiled.n_states):
        total += sol.y[idx]
    dose = params.drug_param("dose_mg")
    if dose > 0:
        mbe = float(np.max(np.abs(total - dose) / dose))
    else:
        mbe = 0.0

    return SimResult(
        time_h=sol.t,
        concentrations=concentrations,
        amounts=amounts,
        mass_balance_error=mbe,
        solver_success=bool(sol.success),
    )


def solve_mc(
    compiled: CompiledODE,
    params: ResolvedParams,
    y0: np.ndarray,
    t_span: tuple[float, float],
    observation_node: str = "venous_blood",
    t_min_h: float = 0.0,
) -> tuple[float, float, float, bool]:
    """Fast MC solve -- returns only (cmax, tmax, auc, success).

    Optimized for Monte Carlo sampling:
    - rtol=1e-4, atol=1e-6 (sufficient for Cmax/AUC accuracy across N samples)
    - No t_eval by default (solver chooses its own adaptive grid)
    - Returns scalars, not full SimResult (avoids dict/array allocation)

    Args:
        compiled: Compiled ODE skeleton.
        params: Resolved point-value parameters.
        y0: Initial state vector (amounts in mg).
        t_span: Integration interval ``(t_start, t_end)`` in hours.
        observation_node: Node to extract PK from (default: venous_blood).
        t_min_h: Minimum observation time in hours.  When > 0, injects
            ``t_min_h`` as a guaranteed anchor in ``t_eval`` and extracts
            Cmax/Tmax only from the window ``t >= t_min_h``.  AUC remains
            full-interval.  Default ``0.0`` (V2-compatible).

    Returns:
        Tuple of (cmax, tmax, auc, success).
    """
    rhs = compiled.make_rhs(params)

    if t_min_h > 0.0:
        # Same construction as solve(), but coarser grid (100 vs 500 points)
        # because MC is speed-critical.
        t_eval = np.concatenate([[0.0], np.linspace(t_min_h, t_span[1], 99)])
    else:
        t_eval = None  # adaptive grid (V2 behavior)

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-6,
    )

    if not sol.success:
        return 0.0, 0.0, 0.0, False

    # Extract concentration from the observation node
    obs_idx = compiled.state_index.get(observation_node)
    if obs_idx is None:
        return 0.0, 0.0, 0.0, False

    v = params.node_param(observation_node, "volume")
    if v > 0 and params.is_blood_pool(observation_node):
        conc = sol.y[obs_idx] / v
    elif v > 0:
        kp = params.drug_kp(observation_node)
        conc = sol.y[obs_idx] / (v * kp) if kp > 0 else sol.y[obs_idx] / v
    else:
        conc = sol.y[obs_idx]

    if t_min_h > 0.0:
        mask = sol.t >= t_min_h
        if not np.any(mask):
            return 0.0, 0.0, 0.0, False
        conc_window = conc[mask]
        t_window = sol.t[mask]
        cmax = float(np.max(conc_window))
        tmax = float(t_window[np.argmax(conc_window)])
    else:
        cmax = float(np.max(conc))
        tmax = float(sol.t[np.argmax(conc)])

    _trapz = getattr(np, "trapezoid", np.trapz)  # numpy 2.0+ vs 1.x
    # AUC is full-interval by design: total drug exposure is independent of
    # the Cmax observation window; masking AUC would be clinically incorrect.
    auc = float(_trapz(conc, sol.t))

    return cmax, tmax, auc, True
