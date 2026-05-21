"""Liver node carries fu_correction_applicable flag (B-11)."""
from __future__ import annotations

from sisyphus.graph.presets import reference_man


def test_liver_has_fu_correction_applicable_flag():
    body = reference_man()
    assert "liver" in body.nodes, "reference_man must have a liver node"
    liver = body.nodes["liver"]
    flag = liver.fu_correction_applicable
    assert flag == 1.0, (
        f"liver.fu_correction_applicable expected 1.0, got {flag!r}"
    )


def test_other_nodes_lack_flag():
    """No other node carries the flag (Phase A scope: liver-only)."""
    body = reference_man()
    flagged = [
        n for n, node in body.nodes.items()
        if node.fu_correction_applicable != 0.0 and n != "liver"
    ]
    assert flagged == [], (
        f"Unexpected non-liver nodes flagged for fu_correction: {flagged}. "
        "B-11 Phase A scope is liver-only. To extend to gut_wall etc., "
        "update the spec first."
    )
