"""Monte Carlo uncertainty propagation.

Compile once, parameterize many: the compiled ODE skeleton is reused
across N MC samples.  Each sample draws from the parameter distributions,
resolves to point values, and solves the ODE.

Returns SimResult samples — PK endpoint computation is the pk layer's
responsibility, orchestrated by pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from sisyphus.core import DrugOnGraph, SimResult
from sisyphus.engine.compiler import CompiledODE, ResolvedParams
from sisyphus.engine.solver import solve
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


class UncertaintyEngine:
    """Monte Carlo propagation engine.

    Usage::

        compiled = ODECompiler().compile(graph)
        ue = UncertaintyEngine()
        result = ue.propagate(compiled, graph, drug, n_samples=1000)
        # Then in pipeline: pk_list = [compute_endpoints(sr) for sr in result.sim_results]
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
