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
from sisyphus.regimen.types import DEFAULT_IV_NODE, DosingRegimen


def predict_tdm(
    smiles: str,
    regimen: DosingRegimen,
    observations,
    *,
    covariates: Covariates | None = None,
    renal_prior_cv: float = 1.0,
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
    from sisyphus.mipd.renal_grid import (
        RenalCLForward,
        RenalCLPrior,
        build_renal_cl_grid,
        sir_posterior_renal,
    )

    if any(ev.node != DEFAULT_IV_NODE for ev in regimen.events):
        raise ValueError(
            "predict_tdm supports IV regimens only (every event must target the IV "
            f"node {DEFAULT_IV_NODE!r}); oral steady-state TDM is a future extension."
        )

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
        RenalCLPrior(cv=renal_prior_cv, r_min=float(grid.r_grid[0]), r_max=float(grid.r_grid[-1])),
        RenalCLForward(grid),
        observations,
        n_samples=n_samples,
        rng=rng,
    )
    return dataclasses.replace(post, warnings=tuple(warnings_list))
