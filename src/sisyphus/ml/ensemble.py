"""Ensemble and meta-learner for combining predictions.

The meta-learner combines engine PK, ML PK, and CL/F analytical PK
predictions into a final calibrated output using a geometric-weighted
combination in log space.

3-track adaptive weighting by compound_type (LOOCV-validated, N=107):
- Base drugs: w_engine, w_ml, w_clf (sum=1)
- Other drugs: w_engine, w_ml, w_clf (sum=1)

When engine and ML disagree by >10-fold, the engine prediction is
down-weighted as it is more likely to be wrong at extreme values.
"""

from __future__ import annotations

import logging

import numpy as np

from sisyphus.core import Distribution, PKEndpoints

logger = logging.getLogger(__name__)

# --- 3-track weights (LOOCV-optimized on N=107 clean predictions) ---
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

# VDss analytical track weight (LOOCV-validated: alpha=0.20, Δ=-0.113 AAFE on N=107 holdout)
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
        """Produce combined PK endpoints from engine, ML, and CL/F results.

        Uses adaptive geometric weighting in log space across 3 tracks.
        Falls back to available tracks when some are missing.

        Args:
            engine_pk: PK endpoints from the PBPK engine (may be None).
            ml_pk: PK endpoints from ML direct prediction (may be None).
            dose_mg: Dose in mg.
            logp: Crippen LogP.
            tpsa: Topological polar surface area.
            mw: Molecular weight.
            fup: Fraction unbound in plasma.
            clint: Intrinsic clearance (uL/min/pmol).
            compound_type: One of "neutral", "acid", "base", "zwitterion".
            pgp_flag: Whether the compound is a P-gp substrate.
            clf_pk: PK endpoints from CL/F analytical track (may be None).
            vdss_cmax: VDss-analytical Cmax (dose/(VDss*70)) in mg/L (may be None).

        Returns:
            Combined PKEndpoints with cv=0.3 on Cmax.
        """
        cmax_pbpk = engine_pk.cmax.mean if engine_pk is not None else None
        cmax_ml = ml_pk.cmax.mean if ml_pk is not None else None
        cmax_clf = clf_pk.cmax.mean if clf_pk is not None else None

        # Collect available log-Cmax values and their base weights
        is_base = compound_type == "base"
        w_eng_base = self.w_engine_base if is_base else self.w_engine_other
        w_ml_base = self.w_ml_base if is_base else self.w_ml_other
        w_clf_base = self.w_clf_base if is_base else self.w_clf_other

        tracks: list[tuple[float, float]] = []  # (log_cmax, weight)

        # Scale engine/ml/clf weights by (1-_W_VDSS) when VDss track is available
        vdss_available = vdss_cmax is not None and vdss_cmax > 0
        scale = (1.0 - _W_VDSS) if vdss_available else 1.0

        if cmax_pbpk is not None and cmax_pbpk > 0:
            tracks.append((np.log10(max(cmax_pbpk, 1e-10)), w_eng_base * scale))
        if cmax_ml is not None and cmax_ml > 0:
            tracks.append((np.log10(max(cmax_ml, 1e-10)), w_ml_base * scale))
        if cmax_clf is not None and cmax_clf > 0:
            tracks.append((np.log10(max(cmax_clf, 1e-10)), w_clf_base * scale))
        if vdss_available:
            tracks.append((np.log10(max(vdss_cmax, 1e-10)), _W_VDSS))

        if len(tracks) >= 2:
            # Apply disagreement penalty to engine track
            if cmax_pbpk is not None and cmax_pbpk > 0 and cmax_ml is not None and cmax_ml > 0:
                log_eng = np.log10(max(cmax_pbpk, 1e-10))
                log_ml = np.log10(max(cmax_ml, 1e-10))
                disagreement = abs(log_eng - log_ml)
                if disagreement > _DISAGREEMENT_THRESHOLD_LOG10:
                    scale = _DISAGREEMENT_THRESHOLD_LOG10 / disagreement
                    # Update engine weight in tracks
                    tracks = [
                        (lc, w * scale) if lc == log_eng else (lc, w)
                        for lc, w in tracks
                    ]

            # Renormalize weights to sum=1
            total_w = sum(w for _, w in tracks)
            if total_w > 0:
                tracks = [(lc, w / total_w) for lc, w in tracks]
            else:
                # Equal weighting fallback
                n = len(tracks)
                tracks = [(lc, 1.0 / n) for lc, _ in tracks]

            log_cmax = sum(w * lc for lc, w in tracks)
            cmax_final = float(10**log_cmax)
        elif len(tracks) == 1:
            cmax_final = float(10**tracks[0][0])
        else:
            cmax_final = 0.0

        # For Tmax, AUC, t_half: prefer engine values (more physiologically grounded)
        tmax = engine_pk.tmax if engine_pk else (ml_pk.tmax if ml_pk else Distribution(1.0))
        auc = engine_pk.auc_0t if engine_pk else (ml_pk.auc_0t if ml_pk else Distribution(0.0))
        t_half = engine_pk.t_half if engine_pk else None

        return PKEndpoints(
            cmax=Distribution(mean=max(cmax_final, 1e-10), cv=0.3),
            tmax=tmax,
            auc_0t=auc,
            t_half=t_half,
        )
