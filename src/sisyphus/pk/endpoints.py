"""PK endpoint extraction from simulation results.

Computes Cmax, Tmax, AUC, t½ from SimResult concentration-time data.
"""

from __future__ import annotations

import numpy as np

from sisyphus.core import Distribution, PKEndpoints, SimResult
from sisyphus.pk.nca import auc_trapezoidal, terminal_half_life


def compute_endpoints(result: SimResult, observation_node: str = "venous_blood") -> PKEndpoints:
    """Extract PK endpoints from a SimResult.

    Args:
        result: Raw ODE simulation output.
        observation_node: Node to use for plasma concentrations.

    Returns:
        PKEndpoints with Cmax, Tmax, AUC, t½.
    """
    conc = result.concentrations[observation_node]
    time = result.time_h

    cmax = float(np.max(conc))
    tmax = float(time[np.argmax(conc)])
    auc = auc_trapezoidal(time, conc)
    t_half = terminal_half_life(time, conc)

    return PKEndpoints(
        cmax=Distribution(cmax),
        tmax=Distribution(tmax),
        auc_0t=Distribution(auc),
        t_half=Distribution(t_half) if t_half is not None else None,
    )
