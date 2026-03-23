"""Preset BodyGraph configurations.

Convenience functions that load standard physiology YAML files
and return ready-to-use BodyGraph instances.
"""

from __future__ import annotations

from pathlib import Path

from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import build_from_yaml

_PHYSIOLOGY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "physiology"


def reference_man() -> BodyGraph:
    """ICRP Reference Man (70 kg, 30 y, healthy).

    Loads ``data/physiology/reference_man.yaml`` and returns a
    validated BodyGraph.
    """
    return build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")


def reference_woman() -> BodyGraph:
    """ICRP Reference Woman (58 kg, 30 y, healthy).

    Loads ``data/physiology/reference_woman.yaml`` and returns a
    validated BodyGraph.
    """
    return build_from_yaml(_PHYSIOLOGY_DIR / "reference_woman.yaml")
