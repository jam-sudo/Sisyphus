"""Model versioning and registry.

Tracks trained model artifacts, their holdout metrics, and version
history.  Enforces the invariant that deployed models must have
recorded holdout AAFE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRecord:
    """Metadata for a trained model artifact.

    Attributes:
        name: Model identifier.
        version: Version string.
        path: Path to the serialized model.
        holdout_aafe: AAFE on the holdout set.  Must not be ``None``
            for deployed models.
    """

    name: str
    version: str
    path: Path
    holdout_aafe: float | None


class ModelRegistry:
    """Registry for trained model artifacts."""

    def register(self, record: ModelRecord) -> None:
        """Register a trained model.  Validates holdout metric exists."""
        raise NotImplementedError

    def get(self, name: str, version: str | None = None) -> ModelRecord:
        """Retrieve a model record by name and optional version."""
        raise NotImplementedError
