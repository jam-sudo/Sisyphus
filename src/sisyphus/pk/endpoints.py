"""PK endpoint extraction from simulation results.

Computes Cmax, Tmax, AUC, t½ from SimResult concentration-time data.
"""

from __future__ import annotations

import logging

import numpy as np

from sisyphus.core import Distribution, PKEndpoints, SimResult
from sisyphus.pk.nca import auc_trapezoidal, terminal_half_life

logger = logging.getLogger(__name__)

# Source: c3-auc0inf-guard audit; conservative terminal-return criterion.
_MAX_TERMINAL_CMAX_FRACTION_UNITLESS = 0.05


def compute_endpoints(
    result: SimResult,
    observation_node: str = "venous_blood",
    t_min_h: float = 0.0,
    dose_mg: float | None = None,
    route: str | None = None,
) -> PKEndpoints:
    """Extract PK endpoints from a SimResult.

    Args:
        result: Raw ODE simulation output.
        observation_node: Node to use for plasma concentrations.
        t_min_h: Minimum time for Cmax extraction (skips t < t_min_h). Used
            for IV bolus to avoid the deterministic t=0 spike; default 0.0
            (V2-compatible). AUC and terminal half-life remain full-interval.
            See docs/_internal/specs/2026-04-22-iv-cmax-observation-design.md.

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
    auc_inf = None
    cl = None
    vss = None
    if t_half is not None:
        if (conc[-1] > _MAX_TERMINAL_CMAX_FRACTION_UNITLESS * float(np.max(conc))
                or conc[-1] >= conc[-2]):
            logger.warning("AUC(0-inf) unavailable: concentration has not returned toward baseline")
        else:
            elimination_rate = np.log(2) / t_half
            auc_inf = auc + float(conc[-1]) / elimination_rate
            if route == "iv" and dose_mg is not None and auc_inf > 0:
                cl = dose_mg / auc_inf
                aumc = auc_trapezoidal(time, time * conc)
                aumc += float(conc[-1]) * (
                    float(time[-1]) / elimination_rate + 1 / elimination_rate**2
                )
                vss = dose_mg * aumc / auc_inf**2

    return PKEndpoints(
        cmax=Distribution(cmax),
        tmax=Distribution(tmax),
        auc_0t=Distribution(auc),
        auc_0inf=Distribution(auc_inf) if auc_inf is not None else None,
        t_half=Distribution(t_half) if t_half is not None else None,
        cl=Distribution(cl) if cl is not None else None,
        vss=Distribution(vss) if vss is not None else None,
    )
