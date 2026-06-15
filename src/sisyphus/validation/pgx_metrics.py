# src/sisyphus/validation/pgx_metrics.py
"""Pure metric functions for the PGx genotype-fold validation.

No engine / no I/O — operates on plain numbers so the science is unit-testable
in isolation. See docs/superpowers/specs/2026-06-14-pgx-genotype-fold-validation
-design.md (sec 4).
"""
from __future__ import annotations


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
