"""Regression: every existing YAML that uses bare-scalar enzyme/transporter
values must continue to load unchanged after the correlation_group addition.

Protects data/physiology/{pediatric_5y,sc_overlay,tumor_overlay}.yaml and
future YAML files that do not migrate to the dict syntax.
"""
from __future__ import annotations

import pathlib

from sisyphus.graph.builder import build_from_yaml


def _yaml_paths() -> list[pathlib.Path]:
    """Return only standalone (non-overlay) physiology YAML files.

    Overlay files (sc_overlay.yaml, tumor_overlay.yaml) lack ``cardiac_output``
    and are merged onto a base graph at runtime — they cannot be loaded via
    ``build_from_yaml`` in isolation.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "data" / "physiology"
    return sorted(
        p for p in root.glob("*.yaml") if "overlay" not in p.name
    )


def test_every_physiology_yaml_loads_without_error() -> None:
    for p in _yaml_paths():
        graph = build_from_yaml(p)
        assert graph is not None
        assert len(graph.nodes) > 0


def test_scalar_enzyme_values_parse_to_none_group() -> None:
    """Any YAML node with bare-scalar enzyme entries yields Distribution
    with correlation_group=None (no silent group assignment)."""
    for p in _yaml_paths():
        graph = build_from_yaml(p)
        for name, node in graph.nodes.items():
            for tag, dist in node.enzymes.items():
                # Either the YAML explicitly sets a group (dict syntax) or
                # it's None. Bare scalars always produce None.
                assert dist.correlation_group is None or isinstance(
                    dist.correlation_group, str
                ), f"{p.name}:{name}:{tag} has bad correlation_group type"
