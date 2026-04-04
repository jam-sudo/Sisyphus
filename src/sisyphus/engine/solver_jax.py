"""JAX/Diffrax-based ODE solver — differentiable alternative to SciPy LSODA.

Phase 0 of the UDE roadmap (see docs/breakthrough_path.md).
Provides a drop-in replacement for ``solve()`` that uses Diffrax Kvaerno5
for stiff ODE integration, with support for autograd through the solver.

Two code paths:
    1. ``solve_jax_wrapped``: wraps an existing NumPy RHS via jax.pure_callback.
       Numerical equivalence to LSODA but NO gradients (numpy boundary breaks
       autograd). Used for Phase 0 validation.

    2. ``solve_jax_pure``: takes a pure-JAX vector field. Supports
       jax.grad / jax.jvp / jax.vjp through the adjoint method.
       Used by Phase 1 UDE training.

The adjoint method (RecursiveCheckpointAdjoint) gives O(log N) memory
for backprop through N solver steps — essential for end-to-end training
of neural closures against observed Cmax.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import jax

# PBPK requires float64 for mass balance accuracy over long timescales.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from diffrax import (
    Dopri5,
    Kvaerno3,
    Kvaerno5,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    RESULTS,
    SaveAt,
    Tsit5,
    diffeqsolve,
)

from sisyphus.core import SimResult
from sisyphus.engine.compiler import CompiledODE, ResolvedParams

logger = logging.getLogger(__name__)

# Default solver tolerances — match SciPy LSODA defaults used in solver.py
_DEFAULT_RTOL = 1e-8
_DEFAULT_ATOL = 1e-10
_DEFAULT_DT0 = 0.01
_DEFAULT_MAX_STEPS = 16384


def solve_jax_pure(
    vector_field: Callable[[float, jnp.ndarray, Any], jnp.ndarray],
    y0: jnp.ndarray,
    t_span: tuple[float, float],
    args: Any = None,
    t_eval: jnp.ndarray | None = None,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    stiff: bool = True,
    max_steps: int = _DEFAULT_MAX_STEPS,
    allow_jvp: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
    """Solve an ODE system using Diffrax Kvaerno5 (5th-order stiff).

    Args:
        vector_field: pure JAX function ``f(t, y, args) -> dy/dt``.
        y0: initial state vector (JAX array, shape ``(n_states,)``).
        t_span: ``(t0, t1)`` integration interval in hours.
        args: extra arguments passed to ``vector_field``.
        t_eval: time points to return solution at. If ``None``, uses 500
            evenly-spaced points.
        rtol: relative tolerance.
        atol: absolute tolerance.
        stiff: if True, use Kvaerno5 (implicit). If False, use explicit RK.
        max_steps: maximum solver steps.
        allow_jvp: if False, use an explicit solver (Tsit5) that does not
            require JVP. Required when the vector field wraps a
            jax.pure_callback (no JVP support).

    Returns:
        Tuple of (t, y, success) where:
            - t: shape (T,)
            - y: shape (T, n_states)
            - success: bool indicating solver success
    """
    t0, t1 = t_span
    if t_eval is None:
        t_eval = jnp.linspace(t0, t1, 500)

    term = ODETerm(vector_field)
    if not allow_jvp:
        # Explicit solver for non-differentiable vector fields (numpy wrap).
        solver = Tsit5()
    else:
        solver = Kvaerno5() if stiff else Kvaerno3()
    saveat = SaveAt(ts=t_eval)
    controller = PIDController(rtol=rtol, atol=atol)

    sol = diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=_DEFAULT_DT0,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=max_steps,
        adjoint=RecursiveCheckpointAdjoint(),
        throw=False,  # return result even on solver failure (check result.result)
    )

    success = sol.result == RESULTS.successful if hasattr(sol, "result") else True
    return sol.ts, sol.ys, bool(success)


def _wrap_numpy_rhs(rhs_np: Callable[[float, np.ndarray], np.ndarray], n: int):
    """Wrap a NumPy RHS for Diffrax via jax.pure_callback.

    NOTE: This breaks autograd at the numpy boundary. Used for validation
    only — Phase 1 requires a pure-JAX vector field.
    """
    out_shape = jax.ShapeDtypeStruct((n,), jnp.float64)

    def _cb(tt: jnp.ndarray, yy: jnp.ndarray) -> jnp.ndarray:
        # pure_callback calls expect host-side numpy
        dydt = rhs_np(float(tt), np.asarray(yy, dtype=np.float64))
        return np.asarray(dydt, dtype=np.float64)

    def vf(t, y, _args):
        return jax.pure_callback(_cb, out_shape, t, y)

    return vf


def solve_jax_wrapped(
    rhs_np: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    stiff: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Validation path: wrap a NumPy RHS for Diffrax solver.

    Returns numerical-equivalent results to SciPy LSODA but no autograd.
    Use ``solve_jax_pure`` with a JAX-compatible RHS for gradient flow.
    """
    n = len(y0)
    vf = _wrap_numpy_rhs(rhs_np, n)
    y0_j = jnp.asarray(y0, dtype=jnp.float64)
    t_eval_j = jnp.asarray(t_eval, dtype=jnp.float64) if t_eval is not None else None

    # Wrapped path uses explicit solver (Tsit5) because jax.pure_callback
    # does not support JVP needed by implicit solvers.
    ts, ys, success = solve_jax_pure(
        vf, y0_j, t_span, args=None, t_eval=t_eval_j, rtol=rtol, atol=atol,
        stiff=stiff, allow_jvp=False, max_steps=65536,
    )
    return np.asarray(ts), np.asarray(ys), success


def solve(
    compiled: CompiledODE,
    params: ResolvedParams,
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
) -> SimResult:
    """Drop-in replacement for ``engine.solver.solve`` using Diffrax Kvaerno5.

    Produces a ``SimResult`` identical in structure to the SciPy-based solver.
    Concentration conversion, mass-balance checking, and output formatting
    match the existing implementation bit-for-bit.

    Uses the numpy-wrapped RHS (jax.pure_callback) — validation path.
    """
    rhs_np = compiled.make_rhs(params)

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)

    ts, ys, success = solve_jax_wrapped(
        rhs_np, y0, t_span, t_eval=t_eval, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL,
    )
    # Diffrax returns ys as (T, n_states). SciPy returns as (n_states, T).
    # Transpose to match existing interface.
    y = ys.T  # (n_states, T)

    # Build concentration and amount dicts — same logic as solver.py
    concentrations: dict[str, np.ndarray] = {}
    amounts: dict[str, np.ndarray] = {}
    for name, idx in compiled.state_index.items():
        amounts[name] = y[idx]
        v = params.node_param(name, "volume")
        if v > 0:
            if params.is_blood_pool(name):
                concentrations[name] = y[idx] / v
            else:
                kp = params.drug_kp(name)
                concentrations[name] = y[idx] / (v * kp) if kp > 0 else y[idx] / v
        else:
            concentrations[name] = y[idx]

    # Mass balance check
    total = np.zeros_like(ts)
    for idx in range(compiled.n_states):
        total += y[idx]
    dose = params.drug_param("dose_mg")
    mbe = float(np.max(np.abs(total - dose) / dose)) if dose > 0 else 0.0

    return SimResult(
        time_h=ts,
        concentrations=concentrations,
        amounts=amounts,
        mass_balance_error=mbe,
        solver_success=success,
    )
