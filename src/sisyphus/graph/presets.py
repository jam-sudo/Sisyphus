"""Preset BodyGraph configurations.

Convenience functions that load standard physiology YAML files
and return ready-to-use BodyGraph instances.
"""

from __future__ import annotations

from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import build_from_yaml
from sisyphus.resources import get_resource_config

_PHYSIOLOGY_DIR = get_resource_config().data("physiology")


def reference_man() -> BodyGraph:
    """ICRP Reference Man (70 kg, 30 y, healthy).

    Loads ``data/physiology/reference_man.yaml`` and returns a
    validated BodyGraph.
    """
    return build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")


def reference_woman() -> BodyGraph:
    """ICRP Publication 89 adult Reference Woman (60 kg, healthy)."""
    return build_from_yaml(_PHYSIOLOGY_DIR / "reference_woman.yaml")
