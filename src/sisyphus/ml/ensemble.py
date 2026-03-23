"""Ensemble and meta-learner for combining predictions.

The meta-learner combines engine PK predictions and ML PK predictions
into a final calibrated output.  ML Cmax importance ~50%, engine Cmax ~26%
(from Omega empirical findings).
"""

from __future__ import annotations

from sisyphus.core import PKEndpoints


class MetaLearner:
    """Combines engine and ML predictions into final PK endpoints."""

    def combine(
        self,
        engine_pk: PKEndpoints | None,
        ml_pk: PKEndpoints | None,
    ) -> PKEndpoints:
        """Produce combined PK endpoints from engine and ML results.

        Args:
            engine_pk: PK endpoints from the PBPK engine.
            ml_pk: PK endpoints from ML direct prediction.

        Returns:
            Combined PKEndpoints with calibrated uncertainty.
        """
        raise NotImplementedError
