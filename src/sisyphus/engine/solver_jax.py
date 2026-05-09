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
from diffrax import (  # noqa: E402
    RESULTS,
    Kvaerno3,
    Kvaerno5,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Tsit5,
    diffeqsolve,
)

from sisyphus.core import SimResult  # noqa: E402
from sisyphus.engine.compiler import CompiledODE, ResolvedParams  # noqa: E402
from sisyphus.engine.params_jax import JaxParams, resolve_to_jax  # noqa: E402
from sisyphus.engine.rhs_jax import make_jax_rhs  # noqa: E402

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


# ---------------------------------------------------------------------------
# vmap MC — batch ODE solve over N parameter sets
# ---------------------------------------------------------------------------

_MC_RTOL = 1e-4
_MC_ATOL = 1e-6
_MC_MAX_STEPS = 8192


def _solve_single_mc(
    rhs_fn,
    y0: jnp.ndarray,
    t_span: tuple[float, float],
    params: JaxParams,
    obs_idx: int,
    obs_volume: float,
    obs_kp: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve one MC sample, return (cmax, tmax, auc, success)."""
    t0, t1 = t_span
    # Adaptive grid for speed — use SaveAt with dense output
    t_eval = jnp.linspace(t0, t1, 200)
    term = ODETerm(lambda t, y, args: rhs_fn(t, y, args))
    solver = Kvaerno5()
    saveat = SaveAt(ts=t_eval)
    controller = PIDController(rtol=_MC_RTOL, atol=_MC_ATOL)

    sol = diffeqsolve(
        term, solver,
        t0=t0, t1=t1, dt0=0.01,
        y0=y0, args=params,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=_MC_MAX_STEPS,
        adjoint=RecursiveCheckpointAdjoint(),
        throw=False,
    )

    success = jnp.where(sol.result == RESULTS.successful, 1.0, 0.0)

    # Extract observation node concentration: C = A / (V * Kp)
    amounts_obs = sol.ys[:, obs_idx]
    conc = amounts_obs / (obs_volume * obs_kp)
    conc = jnp.maximum(conc, 0.0)

    cmax = jnp.max(conc)
    tmax = sol.ts[jnp.argmax(conc)]
    auc = jnp.trapezoid(conc, sol.ts)

    return cmax, tmax, auc, success


def solve_mc_jax(
    compiled: CompiledODE,
    params_list: list[ResolvedParams],
    y0_template: np.ndarray,
    t_span: tuple[float, float],
    observation_node: str = "venous_blood",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch MC solve over N parameter sets using JAX vmap.

    Args:
        compiled: Compiled ODE skeleton.
        params_list: List of N ResolvedParams (one per MC sample).
        y0_template: Initial state vector (same for all samples).
        t_span: Integration interval (t0, t1).
        observation_node: Node to extract Cmax/AUC from.

    Returns:
        (cmax[N], tmax[N], auc[N], success[N]) as numpy arrays.
    """
    n_samples = len(params_list)
    if n_samples == 0:
        empty = np.array([])
        return empty, empty, empty, empty

    # Build the pure-JAX RHS (topology is static, shared across samples)
    rhs_fn = make_jax_rhs(compiled)

    # Convert all ResolvedParams to JaxParams and stack into batched arrays
    jax_params_all = [resolve_to_jax(compiled, p) for p in params_list]

    # Stack into a single JaxParams with leading N dimension
    stacked = JaxParams(
        node_volumes=jnp.stack([p.node_volumes for p in jax_params_all]),
        node_kp=jnp.stack([p.node_kp for p in jax_params_all]),
        node_is_blood=jnp.stack([p.node_is_blood for p in jax_params_all]),
        drug_fup=jnp.array([p.drug_fup for p in jax_params_all]),
        drug_rbp=jnp.array([p.drug_rbp for p in jax_params_all]),
        drug_mw=jnp.array([p.drug_mw for p in jax_params_all]),
        drug_renal_cl=jnp.array([p.drug_renal_cl for p in jax_params_all]),
        drug_peff=jnp.array([p.drug_peff for p in jax_params_all]),
        drug_particle_radius=jnp.array([p.drug_particle_radius for p in jax_params_all]),
        edge_flow_rates=jnp.stack([p.edge_flow_rates for p in jax_params_all]),
        edge_transit_rates=jnp.stack([p.edge_transit_rates for p in jax_params_all]),
        edge_ka_fractions=jnp.stack([p.edge_ka_fractions for p in jax_params_all]),
        edge_ps_products=jnp.stack([p.edge_ps_products for p in jax_params_all]),
        node_clint_organ=jnp.stack([p.node_clint_organ for p in jax_params_all]),
        node_total_inflow=jnp.stack([p.node_total_inflow for p in jax_params_all]),
        node_ivive_scaling=jnp.stack([p.node_ivive_scaling for p in jax_params_all]),
        node_transport_vmax=jnp.stack([p.node_transport_vmax for p in jax_params_all]),
        node_transport_km=jnp.stack([p.node_transport_km for p in jax_params_all]),
    )

    obs_idx = compiled.state_index[observation_node]
    # Use first sample's volume and kp for observation node
    # (these vary per sample, but we pass them as arrays for vmap)
    obs_volumes = stacked.node_volumes[:, obs_idx]
    obs_kps = stacked.node_kp[:, obs_idx]

    y0_jax = jnp.asarray(y0_template, dtype=jnp.float64)

    # vmap over samples: each sample gets its own JaxParams slice
    @jax.jit
    def _batch_solve(stacked_params, obs_vols, obs_kps):
        def _single(params_i, vol_i, kp_i):
            return _solve_single_mc(
                rhs_fn, y0_jax, t_span, params_i, obs_idx, vol_i, kp_i,
            )
        return jax.vmap(_single)(stacked_params, obs_vols, obs_kps)

    logger.info("solve_mc_jax: vmapping %d samples...", n_samples)
    cmax_arr, tmax_arr, auc_arr, success_arr = _batch_solve(
        stacked, obs_volumes, obs_kps,
    )

    return (
        np.asarray(cmax_arr),
        np.asarray(tmax_arr),
        np.asarray(auc_arr),
        np.asarray(success_arr),
    )
