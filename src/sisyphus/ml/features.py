"""Feature engineering for ML-based PK prediction.

Re-exports ``compute_features`` from ``sisyphus.descriptors`` (shared
utility at package root) for backward compatibility.  The ml layer
and predict layer both import from ``sisyphus.descriptors`` to avoid
cross-layer dependencies.
"""

from __future__ import annotations

from sisyphus.descriptors import compute_features

__all__ = ["compute_features"]
