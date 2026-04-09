"""Monte Carlo uncertainty propagation.

Compile once, parameterize many: the compiled ODE skeleton is reused
across N MC samples.  Each sample draws from the parameter distributions,
resolves to point values, and solves the ODE.

Two modes:
- ``propagate`` — full MC, stores SimResult per sample (for downstream NCA).
- ``propagate_fast`` — fast MC, returns aggregated PKEndpoints only
  (no SimResults stored, uses reduced-tolerance solver).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from sisyphus.core import Distribution, DrugOnGraph, PKEndpoints, SimResult
from sisyphus.engine.compiler import CompiledODE, ResolvedParams
from sisyphus.engine.solver import solve, solve_mc
from sisyphus.graph.body import BodyGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UncertaintyResult:
    """Aggregated results from MC sampling.

    Contains raw SimResult samples.  PK endpoint extraction is handled
    by the pk layer (not by engine) to respect layer boundaries.

    Attributes:
        sim_results: List of SimResult, one per successful MC sample.
        n_samples: Number of successful samples.
        n_failures: Number of solver failures.
    """

    sim_results: list[SimResult]
    n_samples: int
    n_failures: int


@dataclass(frozen=True)
class MCResult:
    """Aggregated results from fast MC propagation.

    Contains aggregated PK endpoints (as Distributions with CV), 90%
    prediction intervals, and raw sample arrays for downstream analysis.
    No SimResults are stored -- this is the lightweight fast-path output.

    Attributes:
        pk: Aggregated PKEndpoints (median values, CV from samples).
        n_samples: Number of successful samples.
        n_failures: Number of solver failures.
        cmax_90ci: 90% prediction interval for Cmax (5th, 95th percentile).
        auc_90ci: 90% prediction interval for AUC (5th, 95th percentile).
        cmax_samples: Raw Cmax samples array.
        tmax_samples: Raw Tmax samples array.
        auc_samples: Raw AUC samples array.
    """

    pk: PKEndpoints
    n_samples: int
    n_failures: int
    cmax_90ci: tuple[float, float]
    auc_90ci: tuple[float, float]
    cmax_samples: np.ndarray
    tmax_samples: np.ndarray
    auc_samples: np.ndarray


class UncertaintyEngine:
    """Monte Carlo propagation engine.

    Two modes:

    - ``propagate`` — full MC, stores SimResult per sample.
    - ``propagate_fast`` — fast MC, returns aggregated PKEndpoints only.

    Usage (full)::

        compiled = ODECompiler().compile(graph)
        ue = UncertaintyEngine()
        result = ue.propagate(compiled, graph, drug, n_samples=1000)
        # Then in pipeline: pk_list = [compute_endpoints(sr) for sr in result.sim_results]

    Usage (fast)::

        mc = ue.propagate_fast(compiled, graph, drug, n_samples=1000)
        print(mc.pk.cmax, mc.cmax_90ci)
    """

    def propagate(
        self,
        compiled: CompiledODE,
        graph: BodyGraph,
        drug: DrugOnGraph,
        n_samples: int = 1000,
        seed: int = 42,
        t_span: tuple[float, float] = (0.0, 24.0),
    ) -> UncertaintyResult:
        """Run MC sampling and return aggregated SimResults.

        For each sample:
        1. Sample graph distributions -> realized graph.
        2. Sample drug distributions -> realized drug.
        3. Build ResolvedParams.
        4. Solve ODE -> SimResult.

        PK endpoint computation is NOT done here -- that is the pk
        layer's responsibility (called by pipeline).

        Args:
            compiled: Pre-compiled ODE skeleton (reused across samples).
            graph: BodyGraph with distributional parameters.
            drug: DrugOnGraph with distributional parameters.
            n_samples: Number of MC iterations.
            seed: Base RNG seed (sample i uses seed + i).
            t_span: Integration interval (t_start, t_end) in hours.

        Returns:
            UncertaintyResult with raw SimResult samples.
        """
        results: list[SimResult] = []
        n_failures = 0

        for i in range(n_samples):
            rng = np.random.default_rng(seed + i)

            # Sample parameter distributions
            realized_graph = graph.sample(rng)
            realized_drug = drug.sample(rng)
            params = ResolvedParams(realized_graph, realized_drug)

            # Build initial state: all drug in the administration node
            y0 = np.zeros(compiled.n_states)
            admin_idx = compiled.state_index[drug.administration_node]
            y0[admin_idx] = drug.dose_mg

            try:
                result = solve(compiled, params, y0, t_span=t_span)
                if result.solver_success:
                    results.append(result)
                else:
                    n_failures += 1
                    logger.debug("MC sample %d: solver did not converge", i)
            except Exception:
                n_failures += 1
                logger.debug("MC sample %d: solver exception", i, exc_info=True)

        logger.info(
            "MC propagation: %d/%d successful, %d failures",
            len(results),
            n_samples,
            n_failures,
        )

        return UncertaintyResult(
            sim_results=results,
            n_samples=len(results),
            n_failures=n_failures,
        )

    def propagate_fast(
        self,
        compiled: CompiledODE,
        graph: BodyGraph,
        drug: DrugOnGraph,
        n_samples: int = 1000,
        seed: int = 42,
        t_span: tuple[float, float] = (0.0, 24.0),
        observation_node: str = "venous_blood",
        backend: str = "scipy",
    ) -> MCResult:
        """Fast MC propagation -- returns aggregated PK only (no SimResults).

        Uses ``solve_mc`` with reduced tolerances and no ``t_eval`` to
        minimize per-iteration cost.  Returns scalar PK samples aggregated
        into a ``MCResult`` with median, CV, and 90% prediction intervals.

        Args:
            compiled: Pre-compiled ODE skeleton (reused across samples).
            graph: BodyGraph with distributional parameters.
            drug: DrugOnGraph with distributional parameters.
            n_samples: Number of MC iterations.
            seed: Base RNG seed (sample i uses seed + i).
            t_span: Integration interval (t_start, t_end) in hours.
            observation_node: Node to extract PK from.

        Returns:
            MCResult with aggregated PKEndpoints and 90% PI.
        """
        # Pre-compute initial conditions template (reused each iteration)
        y0_template = np.zeros(compiled.n_states)
        admin_idx = compiled.state_index[drug.administration_node]
        y0_template[admin_idx] = drug.dose_mg

        # ── Surrogate backend: MLP forward pass (no ODE) ──
        if backend == "surrogate":
            from sisyphus.engine.surrogate import (
                load_surrogate_ensemble,
                solve_surrogate_batch,
            )

            models = load_surrogate_ensemble()
            scaler = np.load("models/surrogate/scaler.npz")

            params_list = []
            for i in range(n_samples):
                rng = np.random.default_rng(seed + i)
                realized_graph = graph.sample(rng)
                realized_drug = drug.sample(rng)
                params_list.append(ResolvedParams(realized_graph, realized_drug))

            cmax_arr, uncertainty = solve_surrogate_batch(
                models, params_list, scaler["mean"], scaler["std"],
            )

            mask = cmax_arr > 0
            cmax_arr = cmax_arr[mask]
            n_ok = int(np.sum(mask))
            n_failures = n_samples - n_ok

            # Surrogate doesn't produce tmax/auc — fill with zeros
            tmax_arr = np.zeros(n_ok)
            auc_arr = np.zeros(n_ok)

            logger.info("MC-fast surrogate: %d/%d successful", n_ok, n_samples)
            return self._build_mc_result(cmax_arr, tmax_arr, auc_arr, n_ok, n_failures)

        # ── JAX backend: vmap over all samples in one JIT call ──
        if backend == "jax":
            from sisyphus.engine.solver_jax import solve_mc_jax

            params_list = []
            for i in range(n_samples):
                rng = np.random.default_rng(seed + i)
                realized_graph = graph.sample(rng)
                realized_drug = drug.sample(rng)
                params_list.append(ResolvedParams(realized_graph, realized_drug))

            cmax_arr, tmax_arr, auc_arr, success_arr = solve_mc_jax(
                compiled, params_list, y0_template, t_span, observation_node,
            )

            mask = (success_arr > 0.5) & (cmax_arr > 0)
            cmax_arr = cmax_arr[mask]
            tmax_arr = tmax_arr[mask]
            auc_arr = auc_arr[mask]
            n_ok = int(np.sum(mask))
            n_failures = n_samples - n_ok

            logger.info("MC-fast JAX: %d/%d successful", n_ok, n_samples)
            return self._build_mc_result(cmax_arr, tmax_arr, auc_arr, n_ok, n_failures)

        # ── SciPy backend: sequential loop (default) ──
        cmax_samples: list[float] = []
        tmax_samples: list[float] = []
        auc_samples: list[float] = []
        n_failures = 0

        for i in range(n_samples):
            rng = np.random.default_rng(seed + i)

            # Sample parameter distributions
            realized_graph = graph.sample(rng)
            realized_drug = drug.sample(rng)
            params = ResolvedParams(realized_graph, realized_drug)

            try:
                cmax, tmax, auc, success = solve_mc(
                    compiled,
                    params,
                    y0_template,
                    t_span,
                    observation_node,
                )
                if success and cmax > 0:
                    cmax_samples.append(cmax)
                    tmax_samples.append(tmax)
                    auc_samples.append(auc)
                else:
                    n_failures += 1
                    logger.debug("MC-fast sample %d: solver failure or zero Cmax", i)
            except Exception:
                n_failures += 1
                logger.debug("MC-fast sample %d: solver exception", i, exc_info=True)

        n_ok = len(cmax_samples)
        logger.info(
            "MC-fast propagation: %d/%d successful, %d failures",
            n_ok,
            n_samples,
            n_failures,
        )

        cmax_arr = np.array(cmax_samples)
        tmax_arr = np.array(tmax_samples)
        auc_arr = np.array(auc_samples)

        return self._build_mc_result(cmax_arr, tmax_arr, auc_arr, n_ok, n_failures)

    @staticmethod
    def _build_mc_result(
        cmax_arr: np.ndarray,
        tmax_arr: np.ndarray,
        auc_arr: np.ndarray,
        n_ok: int,
        n_failures: int,
    ) -> MCResult:
        """Aggregate MC sample arrays into an MCResult."""
        if n_ok == 0:
            return MCResult(
                pk=PKEndpoints(
                    cmax=Distribution(0.0),
                    tmax=Distribution(0.0),
                    auc_0t=Distribution(0.0),
                ),
                n_samples=0,
                n_failures=n_failures,
                cmax_90ci=(0.0, 0.0),
                auc_90ci=(0.0, 0.0),
                cmax_samples=np.array([]),
                tmax_samples=np.array([]),
                auc_samples=np.array([]),
            )

        cmax_med = float(np.median(cmax_arr))
        tmax_med = float(np.median(tmax_arr))
        auc_med = float(np.median(auc_arr))

        cmax_cv = float(np.std(cmax_arr) / cmax_med) if cmax_med > 0 else 0.0
        auc_cv = float(np.std(auc_arr) / auc_med) if auc_med > 0 else 0.0
        tmax_cv = float(np.std(tmax_arr) / tmax_med) if tmax_med > 0 else 0.0

        cmax_lo, cmax_hi = float(np.percentile(cmax_arr, 5)), float(np.percentile(cmax_arr, 95))
        auc_lo, auc_hi = float(np.percentile(auc_arr, 5)), float(np.percentile(auc_arr, 95))

        pk = PKEndpoints(
            cmax=Distribution(cmax_med, cv=cmax_cv),
            tmax=Distribution(tmax_med, cv=tmax_cv),
            auc_0t=Distribution(auc_med, cv=auc_cv),
        )

        return MCResult(
            pk=pk,
            n_samples=n_ok,
            n_failures=n_failures,
            cmax_90ci=(cmax_lo, cmax_hi),
            auc_90ci=(auc_lo, auc_hi),
            cmax_samples=cmax_arr,
            tmax_samples=tmax_arr,
            auc_samples=auc_arr,
        )
