"""Pipeline configuration.

Central configuration for the prediction pipeline, including
model paths, MC sample count, and physiology preset selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the Sisyphus prediction pipeline.

    Attributes:
        physiology_preset: Name of the physiology preset
            (``"reference_man"``, ``"reference_woman"``).
        n_mc_samples: Number of Monte Carlo samples for uncertainty.
        observation_node: Node for plasma concentration readout.
        model_dir: Directory containing trained ML model artifacts.
        seed: Base random seed for reproducibility.
    """

    physiology_preset: str = "reference_man"
    n_mc_samples: int = 1000
    observation_node: str = "venous_blood"
    model_dir: Path = Path("models")
    seed: int = 42
