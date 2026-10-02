"""ML model wrappers for direct PK prediction.

Wraps the pinned public Omega XGBoost Cmax model (1,028 drugs, 2057 features).
The model predicts log10(Cmax_ug_mL / dose_mg).
Cmax (mg/L) = 10^prediction * dose_mg (since ug/mL == mg/L).
"""

from __future__ import annotations

import logging

import xgboost as xgb

from sisyphus.core import Distribution
from sisyphus.descriptors import compute_features
from sisyphus.ml.registry import verify_model_artifact
from sisyphus.resources import get_resource_config

logger = logging.getLogger(__name__)

_MODEL_DIR = get_resource_config().models_dir


class PKPredictor:
    """XGBoost-based direct Cmax predictor.

    Uses the public Omega model trained on 1,028 holdout-excluded drugs.
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
            verify_model_artifact(path)
            self._model = xgb.XGBRegressor()
            self._model.load_model(str(path))
            logger.info("XGBoost Cmax model loaded from %s", path)

    def predict_cmax(self, smiles: str, dose_mg: float) -> Distribution:
        """Predict Cmax from SMILES and dose.

        Args:
            smiles: Input SMILES string.
            dose_mg: Dose in mg.

        Returns:
            Distribution with a heuristic cv=0.5; this is not a calibrated
            predictive interval. The final meta interval is separate.

        Raises:
            ValueError: If the SMILES string is invalid.
        """
        self._ensure_loaded()
        features = compute_features(smiles).reshape(1, -1)
        log_cmax_per_dose = float(self._model.predict(features)[0])  # type: ignore[union-attr]
        cmax = 10**log_cmax_per_dose * dose_mg  # mg/L
        cmax = max(cmax, 1e-10)  # floor to avoid zero/negative
        return Distribution(mean=cmax, cv=0.5)
