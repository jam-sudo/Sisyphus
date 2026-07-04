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
    _trapz = getattr(np, "trapezoid", np.trapz)  # numpy 2.0+ vs 1.x
    return float(_trapz(conc, time))


def terminal_half_life(
    time: NDArray[np.float64],
    conc: NDArray[np.float64],
) -> float | None:
    """Estimate terminal elimination half-life via terminal-window selection.

    λz is fit only on the true terminal log-linear phase, selected by the
    adjusted-R² "best fit" rule (Phoenix/WinNonlin): among all candidate windows
    made of the last k post-Cmax points (k ≥ 3), pick the one that maximizes the
    adjusted R² of the log-linear regression, preferring more points on a tie.
    Regressing *all* post-Cmax points instead would blend the steep distribution
    (α) slope into λz and bias t½ short for multi-compartment disposition; on a
    mono-exponential curve every window fits equally and the whole tail is used.

    Args:
        time: Time points (h).
        conc: Concentration values (mg/L).

    Returns:
        t½ in hours, or ``None`` if estimation fails (fewer than 3 declining
        terminal points, or no declining window).
    """
    time = np.asarray(time, dtype=float)
    conc = np.asarray(conc, dtype=float)

    # Candidate terminal points: strictly after Cmax with positive concentration.
    i_max = int(np.argmax(conc))
    idx = np.where((np.arange(len(conc)) > i_max) & (conc > 0))[0]
    if len(idx) < 3:
        return None

    t_all = time[idx]
    log_c_all = np.log(conc[idx])
    n = len(idx)

    # (adj_r2, n_points, slope) for each declining last-k-points window.
    candidates: list[tuple[float, int, float]] = []
    for k in range(3, n + 1):
        t_w = t_all[n - k:]
        y_w = log_c_all[n - k:]
        slope, intercept = np.polyfit(t_w, y_w, 1)
        if slope >= 0:
            continue  # not declining over this window
        y_hat = slope * t_w + intercept
        ss_res = float(np.sum((y_w - y_hat) ** 2))
        ss_tot = float(np.sum((y_w - y_w.mean()) ** 2))
        r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
        # adjusted R² with one predictor (slope); k > 2 guaranteed here
        adj_r2 = 1.0 - (1.0 - r2) * (k - 1) / (k - 2)
        candidates.append((adj_r2, k, float(slope)))

    if not candidates:
        return None

    # Best fit: maximize adjusted R²; on a near-tie prefer the window with more
    # points (WinNonlin ARS convention).
    max_adj = max(c[0] for c in candidates)
    _, _, slope = max(
        (c for c in candidates if c[0] >= max_adj - 1e-4), key=lambda c: c[1]
    )
    return float(np.log(2) / abs(slope))
