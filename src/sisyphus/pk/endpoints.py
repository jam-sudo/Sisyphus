"""PK endpoint extraction from simulation results.

Computes Cmax, Tmax, AUC, t½ from SimResult concentration-time data.
"""

from __future__ import annotations

import numpy as np

from sisyphus.core import Distribution, PKEndpoints, SimResult
from sisyphus.pk.nca import auc_trapezoidal, terminal_half_life


def compute_endpoints(
    result: SimResult,
    observation_node: str = "venous_blood",
    t_min_h: float = 0.0,
) -> PKEndpoints:
    """Extract PK endpoints from a SimResult.

    Args:
        result: Raw ODE simulation output.
        observation_node: Node to use for plasma concentrations.
        t_min_h: Minimum time for Cmax extraction (skips t < t_min_h). Used
            for IV bolus to avoid the deterministic t=0 spike; default 0.0
            (V2-compatible). AUC and terminal half-life remain full-interval.
            See docs/superpowers/specs/2026-04-22-iv-cmax-observation-design.md.

    Returns:
        PKEndpoints with Cmax, Tmax, AUC, t½.
    """
    conc = result.concentrations[observation_node]
    time = result.time_h

    if t_min_h > 0.0:
        mask = time >= t_min_h
        if not np.any(mask):
            cmax = 0.0
            tmax = 0.0
        else:
            conc_window = conc[mask]
            time_window = time[mask]
            cmax = float(np.max(conc_window))
            tmax = float(time_window[np.argmax(conc_window)])
    else:
        cmax = float(np.max(conc))
        tmax = float(time[np.argmax(conc)])

    # AUC is full-interval by design: total drug exposure is independent of
    # the Cmax observation window; masking AUC would be clinically incorrect.
    auc = auc_trapezoidal(time, conc)
    t_half = terminal_half_life(time, conc)

    return PKEndpoints(
        cmax=Distribution(cmax),
        tmax=Distribution(tmax),
        auc_0t=Distribution(auc),
        t_half=Distribution(t_half) if t_half is not None else None,
    )
