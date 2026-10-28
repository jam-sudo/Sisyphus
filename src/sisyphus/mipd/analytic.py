"""Opt-in one-compartment multiple-dose forward and structural MIPD posterior.

The prior centres come from the mechanistic engine; this compact forward gives
separate CL, central V, and oral absorption-rate axes for sparse TDM updates.
It is an approximation to the multi-compartment engine, so reported intervals
remain parameter uncertainty only.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sisyphus.mipd._regimen import _distinct_phases, _regimen_interval_h, _regimen_route
from sisyphus.mipd.clgrid import CLPrior
from sisyphus.mipd.core import FPrior, Posterior, PosteriorPK, _softmax_resample
from sisyphus.regimen.types import DosingRegimen

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


@dataclass(frozen=True)
class StructuralPrior:
    """Engine-centred one-compartment prior for CL (L/h), V (L), ka (1/h), and F."""

    cl_l_h: float
    volume_l: float
    f: float = 1.0
    ka_per_h: float | None = None
    renal_cl_l_h: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) and x > 0 for x in (self.cl_l_h, self.volume_l, self.f)):
            raise ValueError("CL, V, and F prior centres must be finite and positive")
        if self.ka_per_h is not None and (
            not math.isfinite(self.ka_per_h) or self.ka_per_h <= 0
        ):
            raise ValueError("ka prior centre must be finite and positive")
        if not math.isfinite(self.renal_cl_l_h) or self.renal_cl_l_h < 0:
            raise ValueError("renal CL prior centre must be finite and non-negative")


class AnalyticRegimenForward:
    """Vectorized finite-dose one-compartment IV bolus/infusion or oral forward."""

    # shortcut: one compartment omits tissue kinetics, upgrade with a calibrated PBPK latent grid.

    def __init__(self, regimen: DosingRegimen) -> None:
        self.regimen = regimen
        self.route = _regimen_route(regimen)
        self.tau_h = _regimen_interval_h(regimen)

    def concentration(
        self, t_h: float, cl_l_h: np.ndarray, volume_l: np.ndarray,
        f: np.ndarray, ka_per_h: np.ndarray | None = None,
    ) -> np.ndarray:
        """Plasma concentration (mg/L) at time ``t_h`` for posterior draws."""
        cl = np.asarray(cl_l_h, dtype=float)
        v = np.asarray(volume_l, dtype=float)
        bio = np.asarray(f, dtype=float)
        ke = cl / v
        c = np.zeros_like(cl)
        if self.route == "oral" and ka_per_h is None:
            raise ValueError("oral forward requires ka_per_h")
        for event in self.regimen.events:
            u = t_h - event.time_h
            if u < 0:
                continue
            if self.route == "oral":
                ka = np.asarray(ka_per_h, dtype=float)
                delta = ka - ke
                regular = (np.exp(-ke * u) - np.exp(-ka * u)) / np.where(
                    np.abs(delta) > 1e-8, delta, 1.0
                )
                equal = u * np.exp(-ke * u)
                c += bio * event.dose_mg / v * ka * np.where(
                    np.abs(delta) > 1e-8, regular, equal
                )
            elif event.duration_h == 0:
                c += event.dose_mg / v * np.exp(-ke * u)
            else:
                infusion = -np.expm1(-ke * min(u, event.duration_h))
                tail = np.exp(-ke * max(u - event.duration_h, 0.0))
                c += event.dose_mg / (event.duration_h * cl) * infusion * tail
        return c

    def state(
        self, cl_l_h: np.ndarray, volume_l: np.ndarray,
        f: np.ndarray, ka_per_h: np.ndarray | None = None,
    ) -> dict:
        """Final-interval Cmax/AUC and an observation-time concentration closure."""
        last = self.regimen.last_dose_time_h
        times = np.linspace(last, last + self.tau_h, 121)
        curve = np.stack([
            self.concentration(t, cl_l_h, volume_l, f, ka_per_h) for t in times
        ])

        def conc_at(t: float) -> np.ndarray:
            if t < 0 or t > last + self.tau_h:
                raise ValueError("observation time is outside the regimen horizon")
            return self.concentration(t, cl_l_h, volume_l, f, ka_per_h)

        return {
            "f": f,
            "cmax": np.max(curve, axis=0),
            "auc": _trapz(curve, times, axis=0),
            "conc_at": conc_at,
        }

    def steady_state_concentration(
        self, phase_h: float, cl_l_h: np.ndarray, volume_l: np.ndarray,
        f: np.ndarray, ka_per_h: np.ndarray | None = None,
    ) -> np.ndarray:
        """Exact uniform-regimen one-compartment concentration at a dose phase."""
        if phase_h < 0 or phase_h > self.tau_h:
            raise ValueError("phase_h must lie in [0, interval_h]")
        cl = np.asarray(cl_l_h, dtype=float)
        v = np.asarray(volume_l, dtype=float)
        bio = np.asarray(f, dtype=float)
        ke = cl / v
        tau = self.tau_h
        dose = self.regimen.events[0].dose_mg
        if self.route == "oral":
            if ka_per_h is None:
                raise ValueError("oral forward requires ka_per_h")
            ka = np.asarray(ka_per_h, dtype=float)
            ek = np.exp(-ke * tau)
            gk = np.exp(-ke * phase_h) / (-np.expm1(-ke * tau))
            ga = np.exp(-ka * phase_h) / (-np.expm1(-ka * tau))
            delta = ka - ke
            regular = (gk - ga) / np.where(np.abs(delta) > 1e-8, delta, 1.0)
            equal = np.exp(-ke * phase_h) * (
                phase_h / (1.0 - ek) + tau * ek / (1.0 - ek) ** 2
            )
            return bio * dose / v * ka * np.where(np.abs(delta) > 1e-8, regular, equal)
        duration = self.regimen.events[0].duration_h
        if duration >= tau:
            raise ValueError("steady-state forward requires infusion duration < interval")
        if duration == 0:
            return dose / v * np.exp(-ke * phase_h) / (-np.expm1(-ke * tau))
        completed = -np.expm1(-ke * duration)
        if phase_h <= duration:
            current = -np.expm1(-ke * phase_h)
            previous = completed * np.exp(-ke * (tau + phase_h - duration))
            previous /= -np.expm1(-ke * tau)
            return dose / (duration * cl) * (current + previous)
        return (
            dose / (duration * cl) * completed
            * np.exp(-ke * (phase_h - duration)) / (-np.expm1(-ke * tau))
        )

    def steady_state_state(
        self, cl_l_h: np.ndarray, volume_l: np.ndarray,
        f: np.ndarray, ka_per_h: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Steady-state trough, Cmax, and AUC24 per posterior draw."""
        phases = np.linspace(0.0, self.tau_h, 121)
        curve = np.stack([
            self.steady_state_concentration(t, cl_l_h, volume_l, f, ka_per_h)
            for t in phases
        ])
        dose = self.regimen.events[0].dose_mg
        return {
            "trough": curve[-1],
            "cmax": np.max(curve, axis=0),
            "auc24": np.asarray(f) * dose / np.asarray(cl_l_h) * 24.0 / self.tau_h,
        }


def sir_structural_posterior(
    prior: StructuralPrior,
    regimen: DosingRegimen,
    observations: Sequence,
    *,
    n_samples: int = 20000,
    rng: np.random.Generator | None = None,
    crcl_cv: float = 0.0,
) -> PosteriorPK:
    """SIR posterior over engine-centred CL, V, ka, F, and optional CrCl error."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not math.isfinite(crcl_cv) or crcl_cv < 0:
        raise ValueError("crcl_cv must be finite and >= 0")
    if rng is None:
        rng = np.random.default_rng()
    route = _regimen_route(regimen)
    if route == "oral" and prior.ka_per_h is None:
        raise ValueError("oral prior requires ka_per_h")
    forward = AnalyticRegimenForward(regimen)
    obs = tuple(observations)
    if not obs:
        cl = np.full(n_samples, prior.cl_l_h)
        v = np.full(n_samples, prior.volume_l)
        f = np.full(n_samples, prior.f)
        ka = np.full(n_samples, prior.ka_per_h) if route == "oral" else None
        state = forward.state(cl, v, f, ka)
        return PosteriorPK(
            f=Posterior(f), cmax=Posterior(state["cmax"]),
            auc=Posterior(state["auc"]), n_eff=float(n_samples),
            volume_l=Posterior(v), total_clearance_l_h=Posterior(cl),
            absorption_rate_per_h=Posterior(ka) if ka is not None else None,
        )

    # Reuse the metabolic CLPrior's CV and clipping for all structural scales.
    scale_prior = CLPrior()
    metabolic = scale_prior.sample(n_samples, rng)
    renal = scale_prior.sample(n_samples, rng)
    v_draw = prior.volume_l * scale_prior.sample(n_samples, rng)
    ka_draw = (
        prior.ka_per_h * scale_prior.sample(n_samples, rng) if route == "oral" else None
    )
    crcl = (
        rng.lognormal(0.0, math.sqrt(math.log1p(crcl_cv * crcl_cv)), n_samples)
        if crcl_cv else np.ones(n_samples)
    )
    renal_cl = min(prior.renal_cl_l_h, prior.cl_l_h)
    metabolic_cl = prior.cl_l_h - renal_cl
    # Only one CL source is freed when there is a single concentration phase.
    if route == "iv" and renal_cl >= metabolic_cl:
        metabolic = np.ones(n_samples)
    elif route == "iv":
        renal = np.ones(n_samples)
    cl = metabolic_cl * metabolic + renal_cl * renal * crcl
    conc_times = [o.t for o in obs if hasattr(o, "t")]
    shape = _distinct_phases(obs, forward.tau_h) or (
        route == "iv" and len(conc_times) >= 2
        and max(conc_times) - min(conc_times) > 0.5 * forward.tau_h
    )
    v_used = v_draw if shape else np.full(n_samples, prior.volume_l)
    ka_used = ka_draw if shape else (
        np.full(n_samples, prior.ka_per_h) if route == "oral" else None
    )
    f = (
        FPrior(prior.f, CLPrior().cv).sample(n_samples, rng)
        if route == "oral" else np.ones(n_samples)
    )
    state = forward.state(cl, v_used, f, ka_used)
    loglik = np.zeros(n_samples)
    for observation in obs:
        loglik += observation.log_likelihood(state)
    idx, n_eff = _softmax_resample(loglik, rng)
    reported = forward.state(
        cl[idx], v_draw[idx], f[idx], ka_draw[idx] if ka_draw is not None else None
    )
    return PosteriorPK(
        f=Posterior(f[idx]), cmax=Posterior(reported["cmax"]),
        auc=Posterior(reported["auc"]), n_eff=n_eff,
        cl_scale=Posterior(metabolic[idx]),
        renal_scale=Posterior(renal[idx]) if route == "iv" else None,
        volume_l=Posterior(v_draw[idx]),
        absorption_rate_per_h=Posterior(ka_draw[idx]) if ka_draw is not None else None,
        total_clearance_l_h=Posterior(cl[idx]),
        crcl_scale=Posterior(crcl[idx]) if crcl_cv else None,
    )


def engine_structural_prior(
    smiles: str, regimen: DosingRegimen, *, renal_factor: float = 1.0,
    body_weight_kg: float | None = None, age_years: float | None = None,
    kp_method: str = "rodgers_rowland",
) -> tuple[StructuralPrior, bool]:
    """Estimate one-compartment prior centres from the current engine regimen."""
    dose = float(regimen.events[0].dose_mg)
    route = _regimen_route(regimen)
    if route == "iv":
        from sisyphus.mipd.renal_grid import build_renal_cl_grid

        grid = build_renal_cl_grid(
            smiles, regimen, n_grid=1, r_range=(1.0, 1.0), renal_factor=renal_factor,
            body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
        )
        f0 = 1.0
        ka0 = None
        t_half = grid.t_half_h
        is_ss = grid.is_steady_state
    else:
        from sisyphus.mipd.oral_grid import build_oral_cl_grid

        grid, is_ss, t_half = build_oral_cl_grid(
            smiles, regimen, n_grid=1, s_range=(1.0, 1.0), renal_factor=renal_factor,
            body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
        )
        f0 = float(grid.f_engine[0])
        first_interval = grid.t_grid <= _regimen_interval_h(regimen)
        first_curve = grid.conc[0, first_interval]
        first_times = grid.t_grid[first_interval]
        t_peak = float(first_times[np.argmax(first_curve)])
        ka0 = 1.0 / max(
            t_peak, float(first_times[1]) if first_times.size > 1
            else _regimen_interval_h(regimen)
        )
    cl0 = f0 * dose / float(grid.auc[0])
    v0 = (
        cl0 * t_half / math.log(2.0) if t_half is not None
        else dose * f0 / float(grid.cmax[0])
    )
    prior = StructuralPrior(
        cl_l_h=cl0, volume_l=v0, f=f0, ka_per_h=ka0,
        renal_cl_l_h=grid.renal_cl_l_h or 0.0,
    )
    return prior, is_ss
