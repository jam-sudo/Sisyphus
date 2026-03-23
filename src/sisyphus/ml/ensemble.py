"""Ensemble and meta-learner for combining predictions.

The meta-learner combines engine PK predictions and ML PK predictions
into a final calibrated output using a geometric-weighted combination
in log space.

Engine weight is calibrated at 0.17 based on holdout validation:
- Engine provides physiologically-grounded structure (absorption, distribution)
- ML provides better overall accuracy (AAFE 2.4 vs engine AAFE 4.6)
- Combining with w_engine=0.17 yields AAFE ~2.35, %2-fold ~58%

When engine and ML disagree by >10-fold, the engine prediction is
down-weighted as it is more likely to be wrong at extreme values.
"""

from __future__ import annotations

import logging

import numpy as np

from sisyphus.core import Distribution, PKEndpoints

logger = logging.getLogger(__name__)

# Engine weight in log-space geometric combination.
# Calibrated on training set; validated on holdout.
# w_engine=0.17 balances engine's structural information with ML's accuracy.
_W_ENGINE = 0.17
_W_ML = 1.0 - _W_ENGINE

# When engine and ML disagree by more than this factor (in log10 units),
# reduce engine weight to prevent engine outliers from dominating.
_DISAGREEMENT_THRESHOLD_LOG10 = 1.0  # 10-fold disagreement


class MetaLearner:
    """Combines engine and ML Cmax predictions via calibrated geometric weighting.

    Uses a geometric-weighted mean in log space:
        log10(Cmax_final) = w_engine * log10(Cmax_engine) + w_ml * log10(Cmax_ml)

    When engine and ML disagree by >10-fold, engine weight is reduced
    to prevent extreme engine errors from dominating the final prediction.
    """

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
    ) -> PKEndpoints:
        """Produce combined PK endpoints from engine and ML results.

        Uses calibrated geometric weighting in log space.  Falls back to
        ML-only or engine-only if only one source is available.

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

        Returns:
            Combined PKEndpoints with cv=0.3 on Cmax.
        """
        cmax_pbpk = engine_pk.cmax.mean if engine_pk else None
        cmax_ml = ml_pk.cmax.mean if ml_pk else None

        if cmax_pbpk and cmax_ml and cmax_pbpk > 0 and cmax_ml > 0:
            log_eng = np.log10(max(cmax_pbpk, 1e-10))
            log_ml = np.log10(max(cmax_ml, 1e-10))

            # Adaptive weighting: reduce engine weight when disagreement is large
            disagreement = abs(log_eng - log_ml)
            if disagreement > _DISAGREEMENT_THRESHOLD_LOG10:
                # Scale engine weight down proportionally to disagreement
                scale = _DISAGREEMENT_THRESHOLD_LOG10 / disagreement
                w_eng = _W_ENGINE * scale
                w_ml = 1.0 - w_eng
            else:
                w_eng = _W_ENGINE
                w_ml = _W_ML

            log_cmax = w_eng * log_eng + w_ml * log_ml
            cmax_final = float(10**log_cmax)
        elif cmax_pbpk and cmax_pbpk > 0:
            cmax_final = cmax_pbpk
        elif cmax_ml and cmax_ml > 0:
            cmax_final = cmax_ml
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
