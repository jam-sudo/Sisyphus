"""ML model wrappers for direct PK prediction.

Wraps the pre-trained XGBoost Cmax v2 model (1,128 MMPK drugs, 2057 features).
The model predicts log10(Cmax_ug_mL / dose_mg).
Cmax (mg/L) = 10^prediction * dose_mg (since ug/mL == mg/L).
"""

from __future__ import annotations

import logging
from pathlib import Path

import xgboost as xgb

from sisyphus.core import Distribution
from sisyphus.descriptors import compute_features

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"


class PKPredictor:
    """XGBoost-based direct Cmax predictor.

    Uses the v2 model trained on 1,128 MMPK drugs.
    Input: SMILES string + dose_mg
    Output: Cmax Distribution

    The model predicts log10(Cmax_ug_mL / dose_mg).
    Cmax (mg/L) = 10^prediction * dose_mg (since ug/mL == mg/L).
    """

    def __init__(self) -> None:
        self._model: xgb.XGBRegressor | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            path = _MODEL_DIR / "direct_pk" / "xgboost_cmax.json"
            self._model = xgb.XGBRegressor()
            self._model.load_model(str(path))
            logger.info("XGBoost Cmax model loaded from %s", path)

    def predict_cmax(self, smiles: str, dose_mg: float) -> Distribution:
        """Predict Cmax from SMILES and dose.

        Args:
            smiles: Input SMILES string.
            dose_mg: Dose in mg.

        Returns:
            Distribution with cv=0.5 (50% prediction uncertainty,
            reflecting ~0.65 RMSE in log space from cross-validation).

        Raises:
            ValueError: If the SMILES string is invalid.
        """
        self._ensure_loaded()
        features = compute_features(smiles).reshape(1, -1)
        log_cmax_per_dose = float(self._model.predict(features)[0])  # type: ignore[union-attr]
        cmax = 10**log_cmax_per_dose * dose_mg  # mg/L
        cmax = max(cmax, 1e-10)  # floor to avoid zero/negative
        return Distribution(mean=cmax, cv=0.5)
