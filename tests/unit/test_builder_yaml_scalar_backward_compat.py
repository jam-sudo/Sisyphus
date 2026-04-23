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


# Post-Task-9: reference_man.yaml liver has correlation_group on 5 CYPs.
# Every other (node, enzyme/transporter) pair must still be un-grouped.
MIGRATED_GROUPED = {
    # yaml_file_name: {(node_name, 'enzymes'|'transporters', tag), ...}
    "reference_man.yaml": {
        ("liver", "enzymes", "CYP3A4"),
        ("liver", "enzymes", "CYP2D6"),
        ("liver", "enzymes", "CYP1A2"),
        ("liver", "enzymes", "CYP2C9"),
        ("liver", "enzymes", "CYP2E1"),
    },
}


def test_scalar_enzyme_values_parse_to_none_group() -> None:
    """Bare-scalar entries → Distribution with correlation_group=None.
    Explicitly-migrated entries (Task 9 reference_man.yaml liver enzymes)
    are allowed to have a string correlation_group; everything else must
    be None. OATP1B1 on liver in reference_man.yaml is independent (Task 4
    empirical decision), so it's NOT in MIGRATED_GROUPED.
    """
    for p in _yaml_paths():
        graph = build_from_yaml(p)
        allowed_groups = MIGRATED_GROUPED.get(p.name, set())
        for name, node in graph.nodes.items():
            for tag, dist in node.enzymes.items():
                if (name, "enzymes", tag) in allowed_groups:
                    assert isinstance(dist.correlation_group, str), (
                        f"{p.name}:{name}:enzyme:{tag} expected migrated "
                        f"(non-None correlation_group), got None"
                    )
                else:
                    assert dist.correlation_group is None, (
                        f"{p.name}:{name}:enzyme:{tag} should have "
                        f"correlation_group=None, got {dist.correlation_group!r}"
                    )
            for tag, dist in node.transporters.items():
                if (name, "transporters", tag) in allowed_groups:
                    assert isinstance(dist.correlation_group, str)
                else:
                    assert dist.correlation_group is None, (
                        f"{p.name}:{name}:transporter:{tag} should have "
                        f"correlation_group=None, got {dist.correlation_group!r}"
                    )
