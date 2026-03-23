"""PK prediction accuracy metrics.

AAFE, fold error, and prediction interval coverage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def aafe(predicted: NDArray[np.float64], observed: NDArray[np.float64]) -> float:
    """Absolute Average Fold Error.

    AAFE = 10^(mean(|log10(predicted / observed)|))

    Args:
        predicted: Predicted values.
        observed: Observed values.

    Returns:
        AAFE (>= 1.0, lower is better).  Returns inf if no valid pairs.
    """
    mask = (predicted > 0) & (observed > 0)
    if not np.any(mask):
        return float("inf")
    log_ratios = np.abs(np.log10(predicted[mask] / observed[mask]))
    return float(10 ** np.mean(log_ratios))


def pct_within_n_fold(
    predicted: NDArray[np.float64],
    observed: NDArray[np.float64],
    n: float = 2.0,
) -> float:
    """Percentage of predictions within n-fold of observed.

    Args:
        predicted: Predicted values.
        observed: Observed values.
        n: Fold threshold (default 2.0).

    Returns:
        Percentage (0-100).
    """
    mask = (predicted > 0) & (observed > 0)
    if not np.any(mask):
        return 0.0
    ratios = predicted[mask] / observed[mask]
    within = np.sum((ratios >= 1.0 / n) & (ratios <= n))
    return float(within / len(ratios) * 100)


def pi_coverage(
    observed: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Prediction interval coverage.

    Args:
        observed: Observed values.
        lower: Lower bound of prediction interval.
        upper: Upper bound of prediction interval.

    Returns:
        Coverage fraction (0-1).
    """
    if len(observed) == 0:
        return 0.0
    within = np.sum((observed >= lower) & (observed <= upper))
    return float(within / len(observed))
