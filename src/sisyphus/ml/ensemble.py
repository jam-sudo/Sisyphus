"""Ensemble and meta-learner for combining predictions.

The meta-learner combines engine PK predictions and ML PK predictions
into a final calibrated output.  ML Cmax importance ~50%, engine Cmax ~26%
(from Omega empirical findings).

Meta-learner input (12 features):
    0. log10(cmax_pbpk)
    1. log10(cmax_ml)
    2. log10(dose_mg)
    3. log10(cmax_pbpk) - log10(cmax_ml)   # disagreement
    4. logP
    5. TPSA / 200
    6. MW / 600
    7. log10(fup)
    8. log10(clint)
    9. is_acid
    10. is_base
    11. pgp_flag

Output: log10(Cmax_mg_L) -> Cmax = 10^prediction
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from sisyphus.core import Distribution, PKEndpoints

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"


class MetaLearner:
    """Combines engine and ML Cmax predictions via XGBoost meta-learner.

    Uses a 12-feature XGBoost model trained to optimally weight engine
    and ML predictions based on molecular properties.

    Fallback: geometric mean if model not loaded.
    """

    def __init__(self) -> None:
        self._model: xgb.XGBRegressor | None = None
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            path = _MODEL_DIR / "meta_learner" / "xgboost_meta.json"
            if path.exists():
                self._model = xgb.XGBRegressor()
                self._model.load_model(str(path))
                logger.info("Meta-learner loaded from %s", path)
            else:
                logger.warning("Meta-learner not found at %s, using geometric mean fallback", path)
            self._loaded = True

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

        Uses XGBoost meta-learner with 12 molecular/prediction features.
        Falls back to geometric mean if model unavailable or inputs missing.

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
        self._ensure_loaded()

        cmax_pbpk = engine_pk.cmax.mean if engine_pk else None
        cmax_ml = ml_pk.cmax.mean if ml_pk else None

        # Need both predictions for meta-learner
        if cmax_pbpk and cmax_ml and cmax_pbpk > 0 and cmax_ml > 0 and self._model is not None:
            features = np.array(
                [
                    [
                        np.log10(max(cmax_pbpk, 1e-10)),
                        np.log10(max(cmax_ml, 1e-10)),
                        np.log10(max(dose_mg, 0.1)),
                        np.log10(max(cmax_pbpk, 1e-10)) - np.log10(max(cmax_ml, 1e-10)),
                        logp,
                        tpsa / 200.0,
                        mw / 600.0,
                        np.log10(max(fup, 1e-4)),
                        np.log10(max(clint, 0.1)),
                        1.0 if compound_type == "acid" else 0.0,
                        1.0 if compound_type == "base" else 0.0,
                        1.0 if pgp_flag else 0.0,
                    ]
                ]
            )

            log_cmax = float(self._model.predict(features)[0])
            cmax_final = 10**log_cmax
        elif cmax_pbpk and cmax_ml and cmax_pbpk > 0 and cmax_ml > 0:
            # Geometric mean fallback
            cmax_final = float(np.sqrt(cmax_pbpk * cmax_ml))
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
