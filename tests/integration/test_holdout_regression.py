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
def test_cached_holdout_aafe_is_2p769() -> None:
    """Cached predictions file: Meta AAFE is the public-clone headline 2.769 (±0.005).

    Baseline updated 2026-05-25 (B-03.x literature-IVIVE shift on
    clopidogrel: CES1 / CYP3A4 / CYP2C9-surrogate affinities flipped
    from B-03 placeholders 0.030 each to literature-derived values
    per Subash 2025 PMC12673578 rCES1 Vmax/Km + Boberg 2017
    PMC5267516 CES1 abundance + Kazui 2010 DMD 38:92-99 85/15
    inactive/active fate split). Disposition state advanced
    ceiling_accepted -> literature_applied; affinity_source advanced
    literature -> literature_ivive (predict/registry.py enum
    extended in T13.5).

    Prior pin (2.772, 2026-05-20) reflected B-03 placeholders calibrated
    to the 85/15 fate split but not the absolute parent extraction.
    B-03.x shifts clopidogrel Meta FE 5.15x -> 4.67x (predicted Cmax
    1.402 mg/L vs observed 0.300 mg/L) — small Meta-track improvement
    (-0.003 AAFE) because the ML track (FE 1.37x) dominates the
    clopidogrel meta-learner weighting.

    Public-clone reproducible state (no DrugBank, no logp_correction;
    InChIKey-connectivity fallback active in both registry and
    cyp_clearance_overrides lookups):
      Meta AAFE  2.769  (overall N=107, %2-fold 44.9, %3-fold 63.6)
      Engine     4.057  (-0.008 vs B-03; CYP partition reshape)
      ML         3.010  (bit-identical — ML model artifacts unchanged)
      In-domain  2.859  (N=81)

    Tolerance kept at 0.005 (carried over from the 2026-05-09 baseline;
    accommodates the same in-domain composition drift around AD-flag
    thresholds).

    If this fails, the holdout prediction cache has been regenerated with
    a behavior change. Investigate."""
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
    assert abs(aafe - 2.769) < 0.005, f"AAFE drifted: {aafe:.4f}"
