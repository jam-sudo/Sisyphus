"""Unit tests for apply_phenotype_to_graph under axial expansion (and its
non-axial bit-identity guard). Spec: 2026-06-16-phenotype-axial-node-fix-design.md
"""

from __future__ import annotations

import dataclasses  # noqa: F401  (used by Task 2 axial tests)
from pathlib import Path

import pytest

from sisyphus.graph.axial import expand_axial  # noqa: F401  (used by Task 2 axial tests)
from sisyphus.graph.builder import build_from_yaml
from sisyphus.graph.types import ClearanceEdge  # noqa: F401  (used by Task 2 axial tests)
from sisyphus.predict.phenotype import apply_phenotype_to_graph

ROOT = Path(__file__).resolve().parent.parent.parent
_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"


def _ref_graph():
    return build_from_yaml(_PHYS)


def test_nonaxial_scaling_is_bit_identical():
    """On a normal (non-axial) graph, the liver CYP2D6 abundance is scaled by
    exactly the PM factor (0.10) — unchanged from pre-fix behaviour. Headline-safety guard."""
    g = _ref_graph()
    base = g.nodes["liver"].enzymes["CYP2D6"].mean
    out = apply_phenotype_to_graph(g, {"CYP2D6": "PM"})
    assert out.nodes["liver"].enzymes["CYP2D6"].mean == pytest.approx(base * 0.10, rel=0, abs=0)
    assert out.nodes["liver"].enzymes["CYP3A4"].mean == g.nodes["liver"].enzymes["CYP3A4"].mean
    assert out.nodes["liver"].enzymes["CYP2D6"].cv == g.nodes["liver"].enzymes["CYP2D6"].cv
