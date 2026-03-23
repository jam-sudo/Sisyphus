"""Preset BodyGraph configurations.

Convenience functions that load standard physiology YAML files
and return ready-to-use BodyGraph instances.
"""

from __future__ import annotations

from sisyphus.graph.body import BodyGraph


def reference_man() -> BodyGraph:
    """ICRP Reference Man (70 kg, 30 y, healthy).

    Loads ``data/physiology/reference_man.yaml`` and returns a
    validated BodyGraph.
    """
    raise NotImplementedError


def reference_woman() -> BodyGraph:
    """ICRP Reference Woman (58 kg, 30 y, healthy).

    Loads ``data/physiology/reference_woman.yaml`` and returns a
    validated BodyGraph.
    """
    raise NotImplementedError
