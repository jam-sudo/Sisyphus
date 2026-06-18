# src/sisyphus/validation/pgx_metrics.py
"""Pure metric functions for the PGx genotype-fold validation.

No engine / no I/O — operates on plain numbers so the science is unit-testable
in isolation. See docs/_internal/specs/2026-06-14-pgx-genotype-fold-validation
-design.md (sec 4).
"""
from __future__ import annotations

import math

import numpy as np


def analytical_fold(fm: float, activity: float) -> float:
    """Closed-form genotype AUC fold-ratio: 1 / (1 - fm + fm*activity).

    fm = fraction of total clearance via the gene; activity = variant multiplier
    relative to EM/NM (PM=0, IM=0.5, UM=2.0). Flow-independent for oral,
    hepatically-cleared drugs (see spec sec 2).
    """
    denom = 1.0 - fm + fm * activity
    if denom <= 0:
        raise ValueError(f"non-physical denom {denom} (fm={fm}, activity={activity})")
    return 1.0 / denom


def fm_invivo(obs_fold_pm: float) -> float:
    """In-vivo-implied fm from a PM fold (PM activity = 0): fm = 1 - 1/fold."""
    if obs_fold_pm <= 0:
        raise ValueError(f"obs_fold_pm must be > 0, got {obs_fold_pm}")
    return 1.0 - 1.0 / obs_fold_pm


def a_emp(obs_fold: float, fm: float) -> float:
    """Back-calculated empirical activity from an observed fold and fm.

    a = (1/fold - (1 - fm)) / fm. Well-conditioned only for high fm (>= 0.6);
    callers restrict to that regime (spec sec 4.3).
    """
    if fm <= 0:
        raise ValueError("fm must be > 0")
    return (1.0 / obs_fold - (1.0 - fm)) / fm


def fm_invivo_ci(obs_fold_ci: tuple[float, float]) -> tuple[float, float]:
    """Propagate a fold CI to an fm_invivo CI (monotone increasing in fold)."""
    lo, hi = obs_fold_ci
    return (fm_invivo(lo), fm_invivo(hi))


def fm_agreement(fm_vitro: list[float], fm_vivo: list[float], tol: float = 0.15) -> dict:
    """Agreement between independent in-vitro fm and in-vivo-derived fm.

    Returns n, fraction within absolute tolerance, mean absolute deviation, and
    the OLS slope of fm_vivo ~ fm_vitro (≈ 1 expected).
    """
    if len(fm_vitro) != len(fm_vivo) or not fm_vitro:
        raise ValueError("fm_vitro and fm_vivo must be equal, non-empty")
    n = len(fm_vitro)
    devs = [abs(a - b) for a, b in zip(fm_vitro, fm_vivo)]
    within = sum(d <= tol for d in devs)
    mx = sum(fm_vitro) / n
    my = sum(fm_vivo) / n
    sxx = sum((x - mx) ** 2 for x in fm_vitro)
    sxy = sum((x - mx) * (y - my) for x, y in zip(fm_vitro, fm_vivo))
    slope = sxy / sxx if sxx > 0 else float("nan")
    return {
        "n": n,
        "frac_within_tol": within / n,
        "mad": sum(devs) / n,
        "slope": slope,
        "tol": tol,
    }


def km_uM_to_unbound_mgL(km_uM: float, mw: float, fu_mic: float) -> float:
    """Literature Km (µM, total-microsomal) → engine unbound mg/L.

    Km_unbound[mg/L] = km_uM × fu_mic × MW / 1000   (basis of C_u = fup·c_plasma).
    """
    if km_uM <= 0 or mw <= 0:
        raise ValueError(f"km_uM and mw must be positive, got {km_uM}, {mw}")
    if not 0 < fu_mic <= 1:
        raise ValueError(f"fu_mic must be in (0, 1], got {fu_mic}")
    return km_uM * fu_mic * mw / 1000.0


def loglog_beta(doses: list[float], exposures: list[float]) -> float:
    """Log–log slope β = d log(exposure) / d log(dose) (least squares).

    β = 1 ⇒ dose-proportional (linear); β > 1 ⇒ supra-proportional (saturation).
    """
    if len(doses) != len(exposures) or len(doses) < 2:
        raise ValueError("need >=2 matched (dose, exposure) points")
    ld = np.log(np.asarray(doses, dtype=float))
    le = np.log(np.asarray(exposures, dtype=float))
    return float(np.polyfit(ld, le, 1)[0])


def delta_beta(beta_pm: float, beta_em: float) -> float:
    """Genotype cross-term Δβ = β_PM − β_EM.

    >0 ⇒ systemic divergence (PM's low Vmax saturates first);
    <0 ⇒ first-pass convergence (EM's high extraction saturates).
    """
    return float(beta_pm - beta_em)


def box_robustness_pass(deltas: list[float], threshold: float = 0.10) -> bool:
    """True iff |Δlog fold| exceeds `threshold` at EVERY Km×fu_mic box corner.

    Engagement only at the favorable corner is a FAIL (the v2.2b cherry-pick
    foreclosure). `deltas` are the per-corner |Δlog AUC-fold (sat − linear-null)|.
    """
    if not deltas:
        raise ValueError("deltas must be non-empty")
    return all(d > threshold for d in deltas)


def zonation_weights(n: int, ratio: float, direction: str, shape: str = "linear") -> list[float]:
    """Per-sub-tank abundance weights (sum=1) for an axial zonation gradient.

    direction: 'pericentral' (increasing toward the OUTLET tank N), 'periportal'
    (decreasing toward the outlet), or 'uniform'. ratio = w_max/w_min (>=1; 1 => uniform).
    shape='linear' is an evenly-spaced ramp.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if ratio < 1.0:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    if direction == "uniform" or ratio == 1.0:
        return [1.0 / n] * n
    if direction not in ("pericentral", "periportal"):
        raise ValueError(f"unknown direction {direction!r}")
    if shape != "linear":
        raise ValueError(f"unknown shape {shape!r}")
    if n == 1:
        return [1.0]
    raw = [1.0 + (ratio - 1.0) * i / (n - 1) for i in range(n)]  # raw[0]=1 ... raw[-1]=ratio
    if direction == "periportal":
        raw = raw[::-1]
    s = sum(raw)
    return [r / s for r in raw]


def plugflow_E_linear(fu: float, clint_total: float, q: float) -> float:
    """Plug-flow (N->inf) hepatic extraction for LINEAR clearance: 1 - exp(-fu*CLint/Q).

    Reference value for the axial-cascade convergence (G2). NOTE: the engine applies an
    internal ivive_scaling, so the engine's measured E converges to this SHAPE with an
    effective CLint, not necessarily this exact constant.
    """
    if q <= 0:
        raise ValueError(f"q must be > 0, got {q}")
    return 1.0 - math.exp(-fu * clint_total / q)


def mm_rate(c: float, vmax: float, km: float) -> float:
    """Michaelis-Menten rate Vmax*c/(Km+c)."""
    return vmax * c / (km + c)


def zonal_hazard(c_u_by_zone, vmax_bio_by_zone, km_bio, vmax_detox_by_zone, time):
    """Per-zone reactive-metabolite hazard = time-integral of bioactivation rate that
    EXCEEDS local saturable detox capacity: H_i = ∫ max(0, MM(C_u,i; Vmax_bio,i, Km_bio)
    − Vmax_detox,i) dt. The covalent-binding / toxicity proxy (spec §2).

    c_u_by_zone: list of per-zone unbound-conc arrays (each len == len(time)).
    vmax_bio_by_zone / vmax_detox_by_zone: per-zone scalars. Returns a per-zone list.
    """
    t = np.asarray(time, dtype=float)
    trapz = getattr(np, "trapezoid", np.trapz)
    out = []
    for c_u, vmax_bio, vmax_detox in zip(c_u_by_zone, vmax_bio_by_zone, vmax_detox_by_zone):
        c_arr = np.asarray(c_u, dtype=float)
        form = vmax_bio * c_arr / (km_bio + c_arr)          # MM formation rate
        excess = np.maximum(0.0, form - vmax_detox)         # reactive escaping detox
        out.append(float(trapz(excess, t)))
    return out
