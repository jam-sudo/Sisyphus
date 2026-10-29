"""Public MIPD API: SMILES + dose + sparse measured data -> posterior PK.

``predict_posterior`` updates the engine-as-prior from sparse measured data:

- No observations / measured F/Cmax/AUC: the fast F-only analytic path (the
  engine is linear in dose, so the F forward is exact). With no observations it
  returns the a-priori engine prediction, so the SMILES-only path is unchanged.
- A ``MeasuredConc`` observation (or ``cl_latent=True``): the CL-grid path, a
  2-latent (F, clint-scale) posterior. A clearance latent changes the curve
  shape, so the engine is solved once on a clint-scale grid and the forward
  interpolates it (see ``mipd.grid`` / ``mipd.clgrid``).

Either way the engine posterior is routed through the production meta blend
(``meta_cmax``, the product output). Its interval reflects posterior parameter
uncertainty; no residual interval is validated for this path.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from sisyphus.mipd.amortizer import SIRAmortizer
from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.core import APrioriPK, Posterior, PosteriorPK
from sisyphus.mipd.covariates import Covariates
from sisyphus.mipd.meta import build_meta_tracks, meta_blend_cmax

# Source: train_mipd_neural_amortizer._setup trains with s_range=(0.05, 20.0).
_NEURAL_CLINT_RANGE_RATIO = (0.05, 20.0)
# Source: train_mipd_neural_amortizer._main trains and validates on 13 grid points.
_NEURAL_GRID_POINTS_COUNT = 13


def _attach_meta(post: PosteriorPK, smiles: str, dose_mg: float, ap) -> PosteriorPK:
    """Route the engine posterior through the production meta blend.

    The non-engine tracks (ML/CLF/VDss) are F/CL-independent, so they are fixed
    across the posterior. ``meta_cmax`` is the product posterior (parameter
    uncertainty). No predictive residual interval is attached because the
    development residuals were not validated for conditioned posteriors.
    """
    tracks = build_meta_tracks(smiles, dose_mg, ap)
    meta_samples = meta_blend_cmax(post.cmax.samples, tracks)
    return dataclasses.replace(post, meta_cmax=Posterior(meta_samples))


def predict_posterior(
    smiles: str,
    dose_mg: float,
    observations=(),
    *,
    route: str = "oral",
    prior_cv: float = 1.0,
    cl_prior_cv: float = 1.0,
    n_samples: int = 20000,
    seed: int = 0,
    cl_latent: bool = False,
    method: Literal["sir", "neural"] = "sir",
    covariates: Covariates | None = None,
    phenotypes: dict[str, str] | None = None,
    n_grid: int = 13,
    **predict_kwargs,
) -> PosteriorPK:
    """Posterior PK for ``smiles`` at ``dose_mg`` given ``observations``.

    Output fields (all on the returned PosteriorPK):
      - ``cmax`` (+ ``cmax.ci90``): the engine-track posterior, fully conditioned
        and CrCl-individualized — the PRIMARY estimate for TDM / patient
        individualization. Its ci90 is a parameter-uncertainty band (under-covers
        structural error).
      - ``meta_cmax``: the production population blend (covariate-blind ML/CLF/VDss
        mixed in; damped under conditioning, DE-43) — the SMILES-anchor product.
      - ``cmax_90ci``: None; no residual band has been validated for this path.
      - ``warnings``: structured non-fatal flags (e.g. extreme CrCl).

    Args:
        observations: MeasuredF / MeasuredCmax / MeasuredAUC / MeasuredConc. Empty
            (default) returns the a-priori engine prediction as a prior posterior.
        prior_cv: width of the bioavailability (F) prior.
        cl_prior_cv: width of the metabolic clint-scale prior (CL-grid path only;
            scales enzyme CL — CYP/UGT/NAT — not renal/biliary clearance).
        cl_latent: force the 2-latent (F, metabolic clint-scale) CL-grid path.
            Auto-enabled when a MeasuredConc observation is present. NOTE: F and the
            clint-scale are only jointly identified by a curve-SHAPE observation
            (MeasuredConc on the elimination tail); with magnitude-only data
            (MeasuredCmax/AUC) they trade off along an F/CL ridge — prefer the F-only
            path or anchor F with a MeasuredF (review finding #7).
        method: ``sir`` (default) uses the reference posterior; ``neural`` explicitly
            selects the calibrated but SIR-disagreeing CL-grid approximation.
        covariates: patient covariates (v1: measured CrCl) that deterministically
            individualize the engine's renal clearance (CrCl/125). Routes through a
            single renal-scaled engine solve when no MeasuredConc is present, so the
            metabolic clint latent stays fixed.
        phenotypes: CPIC phenotype assignments such as ``{"CYP2D6": "PM"}``.
            The engine prior is conditioned on these via hepatic enzyme or
            transporter abundance before measured observations update it.
        n_grid: clint-scale grid resolution for the CL-grid path.
        seed: RNG seed for reproducible SIR.
        predict_kwargs: forwarded to ``pipeline.predict.predict`` (e.g. kp_method).
    """
    from sisyphus.pipeline.predict import predict

    if route != "oral":
        # The engine-as-prior latent is oral bioavailability F (IV has F≡1, so the
        # F prior degenerates at the 1.0 ceiling and F_engine≈1 carries no signal).
        raise ValueError(
            f"predict_posterior supports oral route only — the latent is oral "
            f"bioavailability F (IV has F≡1). Got route={route!r}."
        )
    if method not in ("sir", "neural"):
        raise ValueError(f"unknown MIPD posterior method: {method!r}")
    if method == "neural" and n_grid != _NEURAL_GRID_POINTS_COUNT:
        raise ValueError("neural MIPD posterior was validated on a 13-point CL grid")
    if phenotypes:
        from sisyphus.predict.phenotype import parse_phenotype_spec

        phenotypes = parse_phenotype_spec(
            ",".join(f"{tag}:{code}" for tag, code in phenotypes.items())
        )

    observations = list(observations)
    if covariates is not None and covariates.crcl_cv > 0:
        raise ValueError("CrCl measurement CV requires repeated-dose structural TDM")
    needs_grid = method == "neural" or cl_latent or any(
        isinstance(o, MeasuredConc) for o in observations
    )
    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    has_phys = covariates.has_physiology() if covariates is not None else False
    body_weight_kg = covariates.body_weight_kg if covariates is not None else None
    age_years = covariates.age_years if covariates is not None else None
    individualized = renal_factor != 1.0 or has_phys or bool(phenotypes)
    rng = np.random.default_rng(seed)

    warnings_list: list[str] = list(covariates.warnings()) if covariates is not None else []

    # ap is needed for the (covariate-blind) meta tracks regardless. engine_f is
    # only consumed by the reference F-only branch, so the IV-reference solve is
    # requested only there.
    ap = predict(
        smiles, dose_mg, route=route,
        compute_f_engine=(not needs_grid and not individualized),
        phenotypes=phenotypes,
        **predict_kwargs,
    )
    if ap.engine_pk is None or ap.engine_pk.cmax is None:
        raise ValueError("engine produced no Cmax for this input; cannot build a posterior")

    # Used by both the grid path and the single renal-scaled solve below.
    from sisyphus.mipd.grid import build_cl_grid

    if needs_grid:
        # CL-grid 2-latent (F, clint-scale) path — handles MeasuredConc. CrCl
        # individualizes renal CL via renal_factor; the clint latent is freed.
        from sisyphus.mipd.clgrid import CLGridForward, CLPrior, sir_posterior_2d
        from sisyphus.mipd.core import FPrior

        grid = build_cl_grid(
            smiles, dose_mg, route=route, n_grid=n_grid,
            kp_method=predict_kwargs.get("kp_method", "rodgers_rowland"),
            renal_factor=renal_factor,
            body_weight_kg=body_weight_kg,
            age_years=age_years,
            phenotypes=phenotypes,
            **({"s_range": _NEURAL_CLINT_RANGE_RATIO} if method == "neural" else {}),
        )
        i1 = int(np.argmin(np.abs(np.log(grid.s_grid))))
        f_engine0 = float(min(max(grid.f_engine[i1], 1e-4), 1.0))
        f_prior = FPrior(f_engine0, prior_cv)
        cl_prior = CLPrior(
            cv=cl_prior_cv, s_min=float(grid.s_grid[0]), s_max=float(grid.s_grid[-1])
        )
        forward = CLGridForward(grid)
        if method == "neural":
            from sisyphus.mipd.amortizer import NeuralAmortizer
            from sisyphus.regimen.types import DosingRegimen
            from sisyphus.resources import get_resource_config

            path = get_resource_config().model("mipd", "neural_clgrid_v2.pt")
            post = NeuralAmortizer.load(path).posterior(
                f_prior, cl_prior, forward, observations,
                regimen=DosingRegimen.single_oral(dose_mg), n_samples=n_samples, rng=rng,
            )
        else:
            post = sir_posterior_2d(
                f_prior, cl_prior, forward, observations,
                n_samples=n_samples, rng=rng,
            )
    elif individualized:
        # CrCl-only (no curve-shape obs): a single renal-scaled engine solve at
        # clint-scale s=1 (clint latent stays FIXED). Reuses build_cl_grid with a
        # 1-point grid; conc_at fragility does not bite (no MeasuredConc here).
        g1 = build_cl_grid(
            smiles, dose_mg, route=route, n_grid=1, s_range=(1.0, 1.0),
            kp_method=predict_kwargs.get("kp_method", "rodgers_rowland"),
            renal_factor=renal_factor,
            body_weight_kg=body_weight_kg,
            age_years=age_years,
            phenotypes=phenotypes,
        )
        cmax0 = float(g1.cmax[0])
        auc0 = float(g1.auc[0])
        f_engine = float(min(max(g1.f_engine[0], 1e-4), 1.0))
        apriori = APrioriPK(cmax0=cmax0, auc0=auc0, f_engine=f_engine)
        post = SIRAmortizer(prior_cv=prior_cv, n_samples=n_samples).posterior(
            apriori, observations, rng=rng
        )
    else:
        # Reference F-only analytic path (unchanged): cmax0 off the a-priori call.
        cmax0 = ap.engine_pk.cmax.mean
        auc0 = ap.engine_pk.auc_0t.mean if ap.engine_pk.auc_0t is not None else 0.0
        if ap.engine_f is None:
            raise ValueError(
                "engine F-reference solve unavailable; cannot build the F posterior "
                "(use cl_latent=True for the CL-grid path, which derives F_engine "
                "from the engine grid)"
            )
        f_engine = min(max(ap.engine_f, 1e-4), 1.0)
        apriori = APrioriPK(cmax0=cmax0, auc0=auc0, f_engine=f_engine)
        post = SIRAmortizer(prior_cv=prior_cv, n_samples=n_samples).posterior(
            apriori, observations, rng=rng
        )

    post = _attach_meta(post, smiles, dose_mg, ap)
    warnings_list.extend(post.warnings)
    return dataclasses.replace(post, warnings=tuple(warnings_list))
