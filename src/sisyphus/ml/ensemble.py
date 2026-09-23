"""Ensemble and meta-learner for combining predictions.

The meta-learner combines engine PK, ML PK, and CL/F analytical PK
predictions into a final point estimate using a geometric-weighted
combination in log space.

3-track adaptive weighting by compound_type (selected on development N=107):
- Base drugs: w_engine, w_ml, w_clf (sum=1)
- Other drugs: w_engine, w_ml, w_clf (sum=1)

When engine and ML disagree by >10-fold, the engine prediction is
down-weighted as it is more likely to be wrong at extreme values.
"""

from __future__ import annotations

import logging

import numpy as np

from sisyphus.core import CmaxPrediction, Distribution, PKEndpoints

logger = logging.getLogger(__name__)

# --- 3-track weights (LOOCV-selected on the repeatedly accessed N=107 development set) ---
# After fixing holdout leakage in ML Cmax/fup/peff/CLint models (2026-04-04),
# CL/F track now contributes for non-base drugs (was masked by contamination).
# Base drugs: engine mechanistic advantage strengthened (93.3% LOOCV stability)
_W_ENGINE_BASE = 0.60
_W_ML_BASE = 0.40
_W_CLF_BASE = 0.00

# Other drugs: 3-track blend (84.4% LOOCV stability)
_W_ENGINE_OTHER = 0.35
_W_ML_OTHER = 0.50
_W_CLF_OTHER = 0.15

# VDss analytical track weight (development result: alpha=0.20, Δ=-0.113 AAFE on N=107)
# Applied uniformly across compound types. Scales existing 3-track weights by (1-_W_VDSS).
_W_VDSS = 0.20

# When engine and ML disagree by more than this factor (in log10 units),
# reduce engine weight to prevent engine outliers from dominating.
_DISAGREEMENT_THRESHOLD_LOG10 = 1.0  # 10-fold disagreement


class MetaLearner:
    """Combines engine, ML, and CL/F Cmax predictions via adaptive geometric weighting.

    Uses a geometric-weighted mean in log space:
        log10(Cmax_final) = w_eng * log10(Cmax_engine)
                          + w_ml * log10(Cmax_ml)
                          + w_clf * log10(Cmax_clf)

    Weights are adaptive by compound_type and subject to disagreement penalty.
    """

    def __init__(
        self,
        w_engine_base: float | None = None,
        w_ml_base: float | None = None,
        w_clf_base: float | None = None,
        w_engine_other: float | None = None,
        w_ml_other: float | None = None,
        w_clf_other: float | None = None,
    ) -> None:
        """Initialize with optional weight overrides (for LOOCV optimization)."""
        self.w_engine_base = w_engine_base if w_engine_base is not None else _W_ENGINE_BASE
        self.w_ml_base = w_ml_base if w_ml_base is not None else _W_ML_BASE
        self.w_clf_base = w_clf_base if w_clf_base is not None else _W_CLF_BASE
        self.w_engine_other = w_engine_other if w_engine_other is not None else _W_ENGINE_OTHER
        self.w_ml_other = w_ml_other if w_ml_other is not None else _W_ML_OTHER
        self.w_clf_other = w_clf_other if w_clf_other is not None else _W_CLF_OTHER

    def combine_cmax(
        self,
        engine_pk: PKEndpoints | None,
        ml_pk: PKEndpoints | None,
        dose_mg: float = 1.0,
        logp: float = 2.0,
        tpsa: float = 60.0,
        mw: float = 300.0,
        fup: float = 0.5,
        clint: float = 10.0,
        compound_type: str = "neutral",
        pgp_flag: bool = False,
        clf_pk: PKEndpoints | None = None,
        vdss_cmax: float | None = None,
    ) -> CmaxPrediction:
        """Produce the authoritative combined Cmax prediction.

        Uses adaptive geometric weighting in log space across available tracks.
        The result intentionally contains no Tmax/AUC/half-life: the meta learner
        does not produce a coherent concentration-time curve.

        Args:
            engine_pk: PK endpoints from the PBPK engine (may be None).
            ml_pk: PK endpoints from ML direct prediction (may be None).
            dose_mg: Compatibility-only metadata; weights do not use it.
            logp: Compatibility-only metadata; weights do not use it.
            tpsa: Compatibility-only metadata; weights do not use it.
            mw: Compatibility-only metadata; weights do not use it.
            fup: Compatibility-only metadata; weights do not use it.
            clint: Compatibility-only metadata; weights do not use it.
            compound_type: One of "neutral", "acid", "base", "zwitterion".
            pgp_flag: Compatibility-only metadata; weights do not use it.
            clf_pk: PK endpoints from CL/F analytical track (may be None).
            vdss_cmax: VDss-analytical Cmax (dose/(VDss*70)) in mg/L (may be None).

        Returns:
            CmaxPrediction with track values and the effective normalized weights.
        """
        cmax_pbpk = engine_pk.cmax.mean if engine_pk is not None else None
        cmax_ml = ml_pk.cmax.mean if ml_pk is not None else None
        cmax_clf = clf_pk.cmax.mean if clf_pk is not None else None

        # Collect available log-Cmax values and their base weights
        is_base = compound_type == "base"
        w_eng_base = self.w_engine_base if is_base else self.w_engine_other
        w_ml_base = self.w_ml_base if is_base else self.w_ml_other
        w_clf_base = self.w_clf_base if is_base else self.w_clf_other

        tracks: list[tuple[str, float, float]] = []  # (name, log_cmax, weight)

        # Scale engine/ml/clf weights by (1-_W_VDSS) when VDss track is available
        vdss_available = vdss_cmax is not None and vdss_cmax > 0
        scale = (1.0 - _W_VDSS) if vdss_available else 1.0

        if cmax_pbpk is not None and cmax_pbpk > 0:
            tracks.append(("engine", np.log10(max(cmax_pbpk, 1e-10)), w_eng_base * scale))
        if cmax_ml is not None and cmax_ml > 0:
            tracks.append(("ml", np.log10(max(cmax_ml, 1e-10)), w_ml_base * scale))
        if cmax_clf is not None and cmax_clf > 0:
            tracks.append(("clf", np.log10(max(cmax_clf, 1e-10)), w_clf_base * scale))
        if vdss_available:
            tracks.append(("vdss", np.log10(max(vdss_cmax, 1e-10)), _W_VDSS))

        if len(tracks) >= 2:
            # Apply disagreement penalty to engine track
            if cmax_pbpk is not None and cmax_pbpk > 0 and cmax_ml is not None and cmax_ml > 0:
                log_eng = np.log10(max(cmax_pbpk, 1e-10))
                log_ml = np.log10(max(cmax_ml, 1e-10))
                disagreement = abs(log_eng - log_ml)
                if disagreement > _DISAGREEMENT_THRESHOLD_LOG10:
                    # Distinct name from the VDss `scale` above (already consumed
                    # when the track weights were built) — this only down-weights
                    # the engine track on engine/ML disagreement.
                    penalty = _DISAGREEMENT_THRESHOLD_LOG10 / disagreement
                    # Update engine weight in tracks
                    tracks = [
                        (name, lc, w * penalty) if name == "engine" else (name, lc, w)
                        for name, lc, w in tracks
                    ]

            # Renormalize weights to sum=1
            total_w = sum(w for _, _, w in tracks)
            if total_w > 0:
                tracks = [(name, lc, w / total_w) for name, lc, w in tracks]
            else:
                # Equal weighting fallback
                n = len(tracks)
                tracks = [(name, lc, 1.0 / n) for name, lc, _ in tracks]

            log_cmax = sum(w * lc for _, lc, w in tracks)
            cmax_final = float(10**log_cmax)
        elif len(tracks) == 1:
            name, lc, _ = tracks[0]
            cmax_final = float(10**lc)
            tracks = [(name, lc, 1.0)]
        else:
            cmax_final = 0.0

        track_values = (
            ("engine", float(cmax_pbpk)) if cmax_pbpk is not None and cmax_pbpk > 0 else None,
            ("ml", float(cmax_ml)) if cmax_ml is not None and cmax_ml > 0 else None,
            ("clf", float(cmax_clf)) if cmax_clf is not None and cmax_clf > 0 else None,
            ("vdss", float(vdss_cmax)) if vdss_available else None,
        )
        return CmaxPrediction(
            # The meta learner supplies a point estimate. Residual uncertainty
            # is carried separately by the development residual band, not a fixed CV.
            cmax=Distribution(mean=max(cmax_final, 1e-10), cv=0.0),
            method="geometric_meta" if len(tracks) >= 2 else "single_track_fallback",
            tracks=tuple(v for v in track_values if v is not None),
            weights=tuple((name, float(w)) for name, _, w in tracks),
        )

    def combine(
        self,
        engine_pk: PKEndpoints | None,
        ml_pk: PKEndpoints | None,
        dose_mg: float = 1.0,
        logp: float = 2.0,
        tpsa: float = 60.0,
        mw: float = 300.0,
        fup: float = 0.5,
        clint: float = 10.0,
        compound_type: str = "neutral",
        pgp_flag: bool = False,
        clf_pk: PKEndpoints | None = None,
        vdss_cmax: float | None = None,
    ) -> PKEndpoints:
        """Backward-compatible PKEndpoints adapter.

        New callers should use :meth:`combine_cmax`.  Non-Cmax endpoints in
        this adapter are explicitly copied from the engine and are not meta
        predictions.
        """
        cmax_prediction = self.combine_cmax(
            engine_pk,
            ml_pk,
            dose_mg=dose_mg,
            logp=logp,
            tpsa=tpsa,
            mw=mw,
            fup=fup,
            clint=clint,
            compound_type=compound_type,
            pgp_flag=pgp_flag,
            clf_pk=clf_pk,
            vdss_cmax=vdss_cmax,
        )

        # Legacy adapter only: these endpoints come from the engine/ML track.
        tmax = engine_pk.tmax if engine_pk else (ml_pk.tmax if ml_pk else Distribution(1.0))
        auc = engine_pk.auc_0t if engine_pk else (ml_pk.auc_0t if ml_pk else Distribution(0.0))
        t_half = engine_pk.t_half if engine_pk else None

        return PKEndpoints(
            cmax=cmax_prediction.cmax,
            tmax=tmax,
            auc_0t=auc,
            t_half=t_half,
        )
