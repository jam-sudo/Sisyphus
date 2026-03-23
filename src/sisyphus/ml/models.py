"""ML model wrappers for direct PK prediction.

Wraps XGBoost and LightGBM with a uniform predict interface
that returns Distributions (point estimate + prediction interval).
"""

from __future__ import annotations

from sisyphus.core import Distribution


class PKPredictor:
    """Base interface for ML PK predictors."""

    def predict_cmax(self, features) -> Distribution:
        """Predict Cmax as a Distribution."""
        raise NotImplementedError

    def predict_auc(self, features) -> Distribution:
        """Predict AUC as a Distribution."""
        raise NotImplementedError
