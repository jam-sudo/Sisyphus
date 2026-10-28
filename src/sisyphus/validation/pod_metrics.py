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


def auc_null_hazard(c_u_liver, time, vmax_bio: float, km_bio: float) -> float:
    """Well-mixed, non-zonal attribution null: bioactivation intrinsic clearance times
    unbound hepatic AUC = (Vmax_bio/Km_bio) * integral(C_u dt). No zonation, no detox
    threshold — the covalent-burden-at-toxic-dose baseline the zonal model must beat."""
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(trapz(np.asarray(c_u_liver, dtype=float), np.asarray(time, dtype=float)))
    return (float(vmax_bio) / float(km_bio)) * auc


def cluster_pass(h_stars: list[float], max_span: float = 10.0) -> tuple[bool, float]:
    """G-cluster: implied {H*_d} cluster within max_span (necessary-not-sufficient)."""
    span = geomean_span(h_stars)
    return (bool(np.isfinite(span) and span <= max_span), span)


def attribution_pass(zonal_span: float, null_span: float) -> bool:
    """G-attribution: zonal model clusters no WORSE than the AUC null."""
    return bool(np.isfinite(zonal_span) and zonal_span <= null_span)


def negative_pass(neg_hazard: float, toxicant_h_stars: list[float],
                  margin: float = 1.0) -> bool:
    """G-negative: a benign drug's predicted hazard sits below the toxicants' H* band
    (by at least `margin` in ratio)."""
    floor = min(float(x) for x in toxicant_h_stars)
    return bool(neg_hazard < floor / float(margin))


def zone_pass(peak_zone_idx: int, observed_zone: str, n_sub: int) -> bool:
    """G-zone: 0=inlet=periportal, n_sub-1=outlet=pericentral. Pass if the predicted
    peak-hazard zone is in the correct half for the documented injury zone."""
    half = n_sub // 2
    if observed_zone == "pericentral":
        return peak_zone_idx >= half
    if observed_zone == "periportal":
        return peak_zone_idx < half
    raise ValueError(f"unknown observed_zone {observed_zone!r}")


def exposure_pass(pred: float, obs: float, fold: float = 3.0) -> bool:
    """G-exposure precondition: predicted rat exposure within `fold` of observed."""
    if pred <= 0 or obs <= 0:
        return False
    return max(pred / obs, obs / pred) <= float(fold)


def predict_pass(loo_results: dict[str, dict], brackets: dict[str, tuple[float, float]],
                 fold_band: float = 5.0) -> tuple[bool, int]:
    """G-predict: LOO PoD within the observed [NOEL,LOEL] bracket OR within fold_band of
    its geomean, for >= N-1 of N drugs. Returns (pass, n_within)."""
    n_ok = 0
    names = list(loo_results)
    for n in names:
        pod = loo_results[n]["pod_pred"]
        lo, hi = brackets[n]
        in_bracket = lo <= pod <= hi
        gm = (lo * hi) ** 0.5
        in_band = pod > 0 and gm > 0 and max(pod / gm, gm / pod) <= fold_band
        if in_bracket or in_band:
            n_ok += 1
    return (n_ok >= len(names) - 1, n_ok)
