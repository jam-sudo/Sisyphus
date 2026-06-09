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
(``meta_cmax``, the product output) and a calibrated split-conformal predictive
interval (``cmax_90ci``) is attached.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from sisyphus.mipd.amortizer import SIRAmortizer
from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.core import APrioriPK, Posterior, PosteriorPK
from sisyphus.mipd.meta import build_meta_tracks, meta_blend_cmax


def _attach_meta_and_interval(post: PosteriorPK, smiles: str, dose_mg: float, ap) -> PosteriorPK:
    """Route the engine posterior through the meta blend + attach the conformal PI.

    The non-engine tracks (ML/CLF/VDss) are F/CL-independent, so they are fixed
    across the posterior. ``meta_cmax`` is the product posterior (parameter
    uncertainty); ``cmax_90ci`` is the train-calibrated split-conformal predictive
    interval around the posterior meta point (the user-facing 90% band). The q90 is
    the **a-priori** (unconditioned) conformal quantile and is not re-calibrated for
    the conditioned posterior, so it is conservative when an informative Cmax obs is
    supplied (review finding #6; see ``PosteriorPK``).
    """
    from sisyphus.pipeline.predict import _conformal_q90_meta

    tracks = build_meta_tracks(smiles, dose_mg, ap)
    meta_samples = meta_blend_cmax(post.cmax.samples, tracks)
    meta_point = float(np.median(meta_samples))

    cmax_90ci: tuple[float, float] | None = None
    q90 = _conformal_q90_meta()
    if q90 is not None and meta_point > 0:
        factor = 10.0**q90
        cmax_90ci = (meta_point / factor, meta_point * factor)

    return dataclasses.replace(post, meta_cmax=Posterior(meta_samples), cmax_90ci=cmax_90ci)


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
    n_grid: int = 13,
    **predict_kwargs,
) -> PosteriorPK:
    """Posterior PK for ``smiles`` at ``dose_mg`` given ``observations``.

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

    observations = list(observations)
    needs_grid = cl_latent or any(isinstance(o, MeasuredConc) for o in observations)
    rng = np.random.default_rng(seed)

    # Read F_engine straight off the a-priori call (compute_f_engine) on the F-only
    # path — no second 'probe' predict + warning regex. The CL-grid path derives
    # F_engine from the grid, so it does not need the extra IV-reference solve.
    ap = predict(
        smiles, dose_mg, route=route,
        compute_f_engine=not needs_grid,
        **predict_kwargs,
    )
    if ap.engine_pk is None or ap.engine_pk.cmax is None:
        raise ValueError("engine produced no Cmax for this input; cannot build a posterior")

    if not needs_grid:
        # F-only analytic path (engine linear in dose -> exact vertical scaling).
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
    else:
        # CL-grid 2-latent (F, clint-scale) path — handles MeasuredConc.
        from sisyphus.mipd.clgrid import CLGridForward, CLPrior, sir_posterior_2d
        from sisyphus.mipd.core import FPrior
        from sisyphus.mipd.grid import build_cl_grid

        grid = build_cl_grid(
            smiles, dose_mg, route=route, n_grid=n_grid,
            kp_method=predict_kwargs.get("kp_method", "rodgers_rowland"),
        )
        i1 = int(np.argmin(np.abs(np.log(grid.s_grid))))  # the s=1 (a-priori) point
        f_engine0 = float(min(max(grid.f_engine[i1], 1e-4), 1.0))
        f_prior = FPrior(f_engine0, prior_cv)
        cl_prior = CLPrior(
            cv=cl_prior_cv, s_min=float(grid.s_grid[0]), s_max=float(grid.s_grid[-1])
        )
        post = sir_posterior_2d(
            f_prior, cl_prior, CLGridForward(grid), observations,
            n_samples=n_samples, rng=rng,
        )

    return _attach_meta_and_interval(post, smiles, dose_mg, ap)
