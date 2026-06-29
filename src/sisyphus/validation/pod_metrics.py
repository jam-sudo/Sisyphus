"""Point-of-departure (PoD) feasibility metrics — pure functions for Bridge B / SP0.

The hazard surface produces a per-drug dose->hazard curve; these helpers turn that
into the single-global-H* test: implied threshold extraction, monotone PoD inversion,
leave-one-out PoD prediction, and the cluster-span metric. No engine, no I/O.
"""
from __future__ import annotations

import numpy as np


def geomean_span(values: list[float]) -> float:
    """max/min over strictly-positive values (the G-cluster spread). NaN if <2 positive."""
    v = [float(x) for x in values if x is not None and x > 0.0]
    if len(v) < 2:
        return float("nan")
    return max(v) / min(v)


def _sorted_by_dose(doses: list[float], hazards: list[float]):
    d = np.asarray(doses, dtype=float)
    h = np.asarray(hazards, dtype=float)
    order = np.argsort(d)
    return d[order], h[order]


def implied_threshold(doses: list[float], hazards: list[float], pod_obs: float) -> float:
    """Hazard at the observed PoD dose, linear-in-dose interpolation on the swept curve."""
    d, h = _sorted_by_dose(doses, hazards)
    return float(np.interp(float(pod_obs), d, h))


def invert_pod(doses: list[float], hazards: list[float], h_star: float) -> float:
    """Smallest dose whose hazard reaches h_star (curve is monotone non-decreasing).

    Clamps to the swept dose range when h_star is below/above the observed hazard span.
    """
    d, h = _sorted_by_dose(doses, hazards)
    if h_star <= h[0]:
        return float(d[0])
    if h_star >= h[-1]:
        return float(d[-1])
    return float(np.interp(float(h_star), h, d))


def loo_predict_pod(per_drug: dict[str, dict]) -> dict[str, dict]:
    """Leave-one-out PoD prediction.

    Parameters
    ----------
    per_drug:
        Mapping of drug name to a dict with keys ``h_star``, ``doses``,
        ``hazards``, and ``pod_obs``.

    Returns
    -------
    dict[str, dict]
        For each drug: ``pod_pred`` (inverted PoD), ``h_star_used`` (geometric
        mean of all other drugs' H*), and ``fold_error`` (max(pred/obs, obs/pred)).

    For each held-out drug, H* is the geometric mean of the OTHER drugs' h_star;
    the drug's own curve is then inverted at that H* to produce the PoD prediction.
    """
    names = list(per_drug)
    out: dict[str, dict] = {}
    for n in names:
        others = [
            per_drug[m]["h_star"]
            for m in names
            if m != n and per_drug[m]["h_star"] is not None and per_drug[m]["h_star"] > 0
        ]
        if not others:
            out[n] = {"pod_pred": float("nan"), "h_star_used": float("nan"),
                      "fold_error": float("nan")}
            continue
        hstar = float(np.exp(np.mean(np.log(others))))
        pod_pred = invert_pod(per_drug[n]["doses"], per_drug[n]["hazards"], hstar)
        pod_obs = float(per_drug[n]["pod_obs"])
        fe = (
            max(pod_pred / pod_obs, pod_obs / pod_pred)
            if pod_pred > 0 and pod_obs > 0
            else float("nan")
        )
        out[n] = {"pod_pred": pod_pred, "h_star_used": hstar, "fold_error": fe}
    return out
