"""Analytical PK solutions (1-compartment, 2-compartment).

Closed-form solutions used as fallback when the ODE solver fails,
and for rapid sanity-checking of engine outputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def one_compartment_oral(
    time: NDArray[np.float64],
    dose_mg: float,
    ka: float,
    ke: float,
    vd: float,
    f: float,
) -> NDArray[np.float64]:
    """Analytical 1-compartment oral PK model.

    C(t) = (F * Dose * ka) / (Vd * (ka - ke)) * (e^(-ke*t) - e^(-ka*t))

    Args:
        time: Time points (h).
        dose_mg: Dose (mg).
        ka: Absorption rate constant (h^-1).
        ke: Elimination rate constant (h^-1).
        vd: Volume of distribution (L).
        f: Bioavailability (0-1).

    Returns:
        Concentration array (mg/L).
    """
    if abs(ka - ke) < 1e-12:
        # Degenerate case: ka ≈ ke
        return (f * dose_mg * ka / vd) * time * np.exp(-ke * time)
    coeff = (f * dose_mg * ka) / (vd * (ka - ke))
    return coeff * (np.exp(-ke * time) - np.exp(-ka * time))


def two_compartment_iv(
    time: NDArray[np.float64],
    dose_mg: float,
    a: float,
    b: float,
    alpha: float,
    beta: float,
) -> NDArray[np.float64]:
    """Analytical 2-compartment IV bolus model.

    C(t) = A * e^(-alpha*t) + B * e^(-beta*t)

    Args:
        time: Time points (h).
        dose_mg: Dose (mg).
        a: Coefficient A (mg/L).
        b: Coefficient B (mg/L).
        alpha: Distribution rate constant (h^-1).
        beta: Elimination rate constant (h^-1).

    Returns:
        Concentration array (mg/L).
    """
    return a * np.exp(-alpha * time) + b * np.exp(-beta * time)
