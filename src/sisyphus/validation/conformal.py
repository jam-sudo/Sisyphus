"""Split-conformal prediction intervals on log-Cmax.

The production Monte-Carlo PI propagates *parameter* uncertainty only and covers
~30% at nominal 90% — ~60 percentage points of the residual spread is structural
model-form error not in the MC. Split-conformal prediction gives valid *marginal*
coverage regardless of base-model structural error (the bias is absorbed into a
wider interval; the guarantee needs only exchangeability, not a correct model).

Convention: the nonconformity score is the absolute log10 fold-error
``s = |log10(pred) - log10(obs)|`` and the interval is multiplicative,
``pred /÷ 10**q``, symmetric in log space.

⚠ Invariant #5: the deployed quantile ``q`` must be calibrated on a set the base
models did NOT train on (train / cross-conformal), never on the holdout — fitting
``q`` on holdout residuals is tuning on the holdout. The holdout is the honest
out-of-sample coverage *validation*, not the calibration source.

Spec: docs/_internal/specs/2026-06-04-conformal-prediction-interval-design.md
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "nonconformity_scores",
    "conformal_quantile",
    "conformal_interval",
    "empirical_coverage",
]


def nonconformity_scores(pred, obs) -> np.ndarray:
    """Absolute log10 fold-error scores for split conformal on log-Cmax.

    Args:
        pred: predicted Cmax (array-like, > 0).
        obs: observed Cmax (array-like, > 0).

    Returns:
        ``|log10(pred) - log10(obs)|`` elementwise.
    """
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    return np.abs(np.log10(pred) - np.log10(obs))


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample split-conformal quantile of calibration ``scores``.

    Returns the ``ceil((n+1)(1-alpha)) / n`` empirical quantile (the rank that
    guarantees marginal coverage >= 1-alpha on a fresh exchangeable point). If the
    required rank exceeds ``n`` (too few calibration points for the level), returns
    ``inf`` — the interval is unbounded and coverage cannot be guaranteed.

    Args:
        scores: nonconformity scores (array-like).
        alpha: miscoverage level (e.g. 0.1 for a 90% interval).

    Returns:
        The conformal quantile ``q`` (>= 0), or ``inf``.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(s[rank - 1])  # rank is 1-indexed


def conformal_interval(pred: float, q: float) -> tuple[float, float]:
    """Multiplicative log-space interval ``(pred / 10**q, pred * 10**q)``.

    ``q = inf`` yields the unbounded ``(0.0, inf)``.
    """
    if not np.isfinite(q):
        return (0.0, float("inf"))
    factor = 10.0 ** q
    return (pred / factor, pred * factor)


def empirical_coverage(preds, obs, q: float) -> float:
    """Fraction of ``obs`` falling inside the conformal interval ``pred /÷ 10**q``."""
    preds = np.asarray(preds, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if not np.isfinite(q):
        return 1.0
    factor = 10.0 ** q
    lo = preds / factor
    hi = preds * factor
    return float(np.mean((obs >= lo) & (obs <= hi)))
