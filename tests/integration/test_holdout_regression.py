"""Regression: 107-holdout Meta AAFE must not drift after physiology
infrastructure changes. Enforces spec Gate A (mean-path equivalence).
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HOLDOUT_JSON = ROOT / "data" / "training" / "4track_holdout_predictions.json"


def _aafe(preds: list[dict]) -> float:
    folds = []
    for p in preds:
        # Field names in 4track_holdout_predictions.json: obs, meta
        obs = p.get("obs") or p.get("observed_cmax_mg_l")
        pred = p.get("meta") or p.get("meta_cmax_mg_l") or p.get("meta_pred_mg_l")
        if obs and pred and obs > 0 and pred > 0:
            folds.append(abs(math.log10(pred / obs)))
    return 10 ** (sum(folds) / len(folds)) if folds else float("nan")


@pytest.mark.skipif(
    not HOLDOUT_JSON.exists(),
    reason=f"{HOLDOUT_JSON.name} not present — regeneration required",
)
def test_cached_holdout_aafe_is_2p679() -> None:
    """Cached predictions file: Meta AAFE is the headline 2.679 (±0.001).

    Baseline updated 2026-05-03 (v0.3 ECM auto-activation regen) — cache
    regenerated post-Task 5 ECM gating. Meta AAFE 2.679 reflects the
    2026-05-02 digoxin+pravastatin SMILES correction baseline. The prior pin
    of 2.695 was stale (Hardening-era pin not updated after the SMILES-fix
    regen in commits 193573a/7042a96).

    pitavastatin/fluvastatin/rosuvastatin/atorvastatin are not in the
    107-drug holdout, so the ECM gating fix (Task 5) has no holdout AAFE
    footprint — AAFE is bit-identical to the 2026-05-02 cache.

    If this fails, the holdout prediction cache has been regenerated with a
    behavior change. Investigate."""
    with HOLDOUT_JSON.open() as f:
        data = json.load(f)
    # Primary path: use pre-computed AAFE stored in the file
    if isinstance(data, dict) and "overall" in data and "meta" in data["overall"]:
        aafe = data["overall"]["meta"]["aafe"]
    else:
        # Fallback: recompute from per-drug predictions
        preds = data.get("predictions", data) if isinstance(data, dict) else data
        if isinstance(data, dict) and "drugs" in data:
            preds = data["drugs"]
        aafe = _aafe(preds)
    assert abs(aafe - 2.679) < 0.001, f"AAFE drifted: {aafe:.4f}"
