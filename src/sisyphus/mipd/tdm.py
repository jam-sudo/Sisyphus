"""Public IV steady-state TDM: SMILES + IV regimen + trough -> renal-CL posterior.

``predict_tdm`` conditions the engine-as-prior on a steady-state IV trough to
individualize renal clearance. The latent is a free renal-CL scale (prior centered
on the CrCl-implied value); F == 1 (IV). The oral-train-calibrated ``meta_cmax`` /
``cmax_90ci`` are NOT attached — they are oral-Cmax artifacts, invalid for IV. The
primary output is the conditioned engine posterior ``cmax``/``auc`` with the
``renal_scale`` posterior; ``cmax.ci90`` is a parameter-uncertainty band (does not
carry calibrated structural coverage). See the design spec.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from sisyphus.mipd.core import PosteriorPK
from sisyphus.mipd.covariates import Covariates
from sisyphus.regimen.types import DosingRegimen


def predict_tdm(
    smiles: str,
    regimen: DosingRegimen,
    observations,
    *,
    covariates: Covariates | None = None,
    renal_prior_cv: float | None = None,
    n_samples: int = 20000,
    n_grid: int = 13,
    seed: int = 0,
    kp_method: str = "rodgers_rowland",
) -> PosteriorPK:
    """Posterior PK for an IV ``regimen`` given steady-state trough ``observations``.

    Args:
        regimen: an IV ``DosingRegimen`` (e.g. ``DosingRegimen.iv_infusion(...)``).
            Every event must target the IV node; an oral regimen is rejected.
        observations: ``MeasuredConc`` troughs at times within the regimen horizon.
        covariates: v1 supports a measured CrCl, which sets the renal-CL prior
            center (``CrCl/125``); the trough updates the latent around it.
    """
    from sisyphus.mipd._regimen import _regimen_route, _require_uniform_regimen
    from sisyphus.mipd.renal_grid import (
        RenalCLForward,
        RenalCLPrior,
        build_renal_cl_grid,
        sir_posterior_renal,
    )

    route = _regimen_route(regimen)
    _require_uniform_regimen(regimen)
    if route == "oral":
        return _predict_tdm_oral(
            smiles, regimen, list(observations), covariates=covariates,
            renal_prior_cv=renal_prior_cv, n_samples=n_samples, n_grid=n_grid,
            seed=seed, kp_method=kp_method,
        )

    # ---- IV branch (unchanged) ----
    # None -> 1.0 coalesce keeps the IV path byte-identical: a default IV call
    # (renal_prior_cv unset) still draws the renal-CL prior at cv=1.0.
    renal_cv = 1.0 if renal_prior_cv is None else renal_prior_cv

    observations = list(observations)
    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    body_weight_kg = covariates.body_weight_kg if covariates is not None else None
    age_years = covariates.age_years if covariates is not None else None
    rng = np.random.default_rng(seed)

    warnings_list: list[str] = list(covariates.warnings()) if covariates is not None else []

    grid = build_renal_cl_grid(
        smiles, regimen, n_grid=n_grid, renal_factor=renal_factor,
        body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
    )
    post = sir_posterior_renal(
        RenalCLPrior(cv=renal_cv, r_min=float(grid.r_grid[0]), r_max=float(grid.r_grid[-1])),
        RenalCLForward(grid),
        observations,
        n_samples=n_samples,
        rng=rng,
    )
    return dataclasses.replace(post, warnings=tuple(warnings_list))


def _predict_tdm_oral(
    smiles,
    regimen,
    observations,
    *,
    covariates,
    renal_prior_cv,
    n_samples,
    n_grid,
    seed,
    kp_method,
) -> PosteriorPK:
    """Oral steady-state TDM posterior.

    A single-phase observation frees only F (internal metabolic clint-scale pinned
    at ``s=1``, a 1-point grid); a multi-phase (or measured-F) observation frees
    both F and the metabolic clint-scale via the full CL grid. ``renal_prior_cv``
    is an IV-only control — on an oral regimen the freed latent is F, not renal CL,
    so it is ignored (and warned when explicitly set). The oral output never carries
    the oral-population ``meta_cmax``/``cmax_90ci`` (API-attached) nor the IV-only
    ``renal_scale``.
    """
    from sisyphus.mipd._regimen import _distinct_phases, _regimen_interval_h
    from sisyphus.mipd.clgrid import CLGridForward, CLPrior, FPrior, sir_posterior_2d
    from sisyphus.mipd.oral_grid import build_oral_cl_grid

    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    body_weight_kg = covariates.body_weight_kg if covariates is not None else None
    age_years = covariates.age_years if covariates is not None else None
    warnings_list = list(covariates.warnings()) if covariates is not None else []
    if renal_prior_cv is not None:
        warnings_list.append(
            "renal_prior_cv is ignored on an oral regimen (oral frees F, not renal CL)"
        )
    rng = np.random.default_rng(seed)
    tau = _regimen_interval_h(regimen)
    has_f = any(not hasattr(o, "t") for o in observations)
    free_both = has_f or _distinct_phases(observations, tau)

    if free_both:
        grid, is_ss, _ = build_oral_cl_grid(
            smiles, regimen, n_grid=n_grid, renal_factor=renal_factor,
            body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
        )
        f0 = float(np.clip(grid.f_engine[int(np.argmin(np.abs(np.log(grid.s_grid))))], 1e-4, 1.0))
        post = sir_posterior_2d(
            FPrior(f0, 1.0),
            CLPrior(cv=1.0, s_min=float(grid.s_grid[0]), s_max=float(grid.s_grid[-1])),
            CLGridForward(grid), observations, n_samples=n_samples, rng=rng,
        )
        if _flat_tail(grid, observations, tau):
            warnings_list.append(
                "cl_scale marginal may be prior-ridge-wide; supplied phases do not "
                "identify the metabolic clint axis"
            )
    else:
        grid, is_ss, _ = build_oral_cl_grid(
            smiles, regimen, n_grid=1, s_range=(1.0, 1.0), renal_factor=renal_factor,
            body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
        )
        f0 = float(np.clip(grid.f_engine[0], 1e-4, 1.0))
        post = sir_posterior_2d(
            FPrior(f0, 1.0), CLPrior(cv=1.0, s_min=1.0, s_max=1.0),
            CLGridForward(grid), observations, n_samples=n_samples, rng=rng,
        )
        post = dataclasses.replace(post, cl_scale=None)

    if not is_ss:
        warnings_list.append(
            "observed regimen has not reached steady state at the grid horizon; "
            "the SS trough/posterior may be biased"
        )
    return dataclasses.replace(
        post, renal_scale=None, meta_cmax=None, cmax_90ci=None,
        warnings=tuple(warnings_list),
    )


def _flat_tail(grid, observations, tau, *, tol=0.5) -> bool:
    """True if the earliest/latest MeasuredConc straddle a near-flat elimination tail.

    A small ``|Δlog c|`` between the first and last observed times (at the engine's
    a-priori scale ``s=1``) means the supplied phases are on the flat tail and do
    NOT identify the metabolic clint (ke = CL/V) axis — the freed ``cl_scale``
    marginal stays prior-ridge-wide. Distinct-but-flat phases trip this; a true
    peak/trough pair (large ``|Δlog c|``) does not.
    """
    ts = sorted(float(o.t) for o in observations if hasattr(o, "t"))
    if len(ts) < 2:
        return False
    s1 = np.ones(1)
    c = [float(grid.conc_at(s1, t)[0]) for t in (ts[0], ts[-1])]
    if min(c) <= 0:
        return False
    return abs(np.log(c[1]) - np.log(c[0])) < tol
