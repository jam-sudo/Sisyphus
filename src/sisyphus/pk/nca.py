"""Non-compartmental analysis (NCA).

Model-independent PK parameter estimation from concentration-time data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def auc_trapezoidal(time: NDArray[np.float64], conc: NDArray[np.float64]) -> float:
    """Compute AUC using the linear trapezoidal rule.

    Args:
        time: Time points (h).
        conc: Concentration values (mg/L).

    Returns:
        AUC in mg·h/L.
    """
    return float(np.trapezoid(conc, time))


def terminal_half_life(
    time: NDArray[np.float64],
    conc: NDArray[np.float64],
) -> float | None:
    """Estimate terminal elimination half-life.

    Fits a log-linear regression to the terminal phase.

    Args:
        time: Time points (h).
        conc: Concentration values (mg/L).

    Returns:
        t½ in hours, or ``None`` if estimation fails.
    """
    # Find index of Cmax
    i_max = int(np.argmax(conc))

    # Terminal phase: after Cmax, concentration > 0
    terminal_mask = np.arange(len(conc)) > i_max
    terminal_mask &= conc > 0

    t_term = time[terminal_mask]
    c_term = conc[terminal_mask]

    if len(t_term) < 3:
        return None

    # Log-linear regression on terminal phase
    log_c = np.log(c_term)
    slope, _ = np.polyfit(t_term, log_c, 1)

    if slope >= 0:
        return None  # Not declining

    return float(np.log(2) / abs(slope))
