"""Model-Informed Precision Dosing (MIPD): engine-as-prior posterior PK.

Reposition the mechanistic engine from a one-shot SMILES->Cmax oracle into a
*structural prior* that any sparse measured observation sharply updates. The
dominant structural error is bioavailability F; this package treats true F as a
latent, places a wide prior on it centered at the engine's emergent F_engine,
and updates it from measured observations (measured F, Cmax, AUC) via SIR.

Charter: docs/_internal/specs/2026-06-09-engine-as-prior-mipd-charter.md
(Gate 0b/0c PASSED: one measured anchor materially improves Cmax, ~3x more OOD.)
"""
from sisyphus.mipd.clgrid import (
    CLGrid,
    CLGridForward,
    CLPrior,
    MeasuredConc,
    sir_posterior_2d,
)
from sisyphus.mipd.core import (
    AnalyticForward,
    APrioriPK,
    FPrior,
    MeasuredAUC,
    MeasuredCmax,
    MeasuredF,
    Posterior,
    PosteriorPK,
    sir_posterior,
)
from sisyphus.mipd.dosing import (
    CandidateEval,
    Constraint,
    DoseRecommendation,
    DoseTarget,
    recommend_dose,
)
from sisyphus.mipd.grid import build_cl_grid
from sisyphus.mipd.meta import MetaTracks, build_meta_tracks, meta_blend_cmax
from sisyphus.mipd.oral_grid import build_oral_cl_grid

__all__ = [
    "APrioriPK",
    "AnalyticForward",
    "FPrior",
    "MeasuredF",
    "MeasuredCmax",
    "MeasuredAUC",
    "MeasuredConc",
    "Posterior",
    "PosteriorPK",
    "sir_posterior",
    "MetaTracks",
    "build_meta_tracks",
    "meta_blend_cmax",
    "CLGrid",
    "CLGridForward",
    "CLPrior",
    "sir_posterior_2d",
    "build_cl_grid",
    "build_oral_cl_grid",
    "CandidateEval",
    "Constraint",
    "DoseRecommendation",
    "DoseTarget",
    "recommend_dose",
]
