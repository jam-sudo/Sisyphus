"""Monte Carlo propagation through an event-driven dosing regimen."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from sisyphus.core import DrugOnGraph
from sisyphus.engine.compiler import CompiledODE, ResolvedParams
from sisyphus.engine.contracts import assert_fu_correction_honored
from sisyphus.engine.uncertainty import MCResult, UncertaintyEngine
from sisyphus.graph.body import BodyGraph
from sisyphus.pk.nca import auc_trapezoidal
from sisyphus.regimen.solver import solve_regimen
from sisyphus.regimen.types import DosingRegimen

logger = logging.getLogger(__name__)


def propagate_regimen(
    compiled: CompiledODE,
    graph: BodyGraph,
    drug: DrugOnGraph,
    regimen: DosingRegimen,
    n_samples: int = 1000,
    seed: int = 42,
    observation_node: str = "venous_blood",
    t_min_h: float = 0.0,
    t_total_h: float = 24.0,
    graph_sampler: Callable[[BodyGraph, np.random.Generator], BodyGraph] | None = None,
) -> MCResult:
    """Propagate parameter draws through the same regimen as the point solve."""
    assert_fu_correction_honored(graph, drug.fu_correction_liver.mean)
    cmax_samples: list[float] = []
    tmax_samples: list[float] = []
    auc_samples: list[float] = []
    n_failures = 0
    for i in range(n_samples):
        rng = np.random.default_rng(seed + i)
        realized_graph = graph_sampler(graph, rng) if graph_sampler else graph.sample(rng)
        realized_drug = drug.sample(rng)
        try:
            result = solve_regimen(
                compiled, ResolvedParams(realized_graph, realized_drug), regimen,
                t_total_h=t_total_h,
            )
            if not result.solver_success:
                n_failures += 1
                continue
            time = result.time_h
            conc = result.concentrations[observation_node]
            mask = time >= t_min_h
            if not np.any(mask) or not np.any(conc[mask] > 0):
                n_failures += 1
                continue
            peak = int(np.argmax(conc[mask]))
            cmax_samples.append(float(conc[mask][peak]))
            tmax_samples.append(float(time[mask][peak]))
            auc_samples.append(auc_trapezoidal(time, conc))
        except Exception:
            n_failures += 1
            logger.debug("MC-regimen sample %d: solver exception", i, exc_info=True)
    n_ok = len(cmax_samples)
    logger.info("MC-regimen propagation: %d/%d successful", n_ok, n_samples)
    return UncertaintyEngine._build_mc_result(
        np.asarray(cmax_samples), np.asarray(tmax_samples), np.asarray(auc_samples),
        n_ok, n_failures,
    )
