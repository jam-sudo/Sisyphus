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
def test_cached_holdout_aafe_is_2p772() -> None:
    """Cached predictions file: Meta AAFE is the public-clone headline 2.772 (±0.005).

    Baseline updated 2026-05-20 (B-03 clopidogrel prodrug registry +
    metabolic_fraction InChIKey-connectivity fix). The cached value was
    previously pinned to 2.751 (2026-05-09 public-clone regen, pre-B-03)
    and briefly to 2.7535 (2026-05-19 B-03 with a silent double-counting
    bug in cyp_clearance_overrides.lookup_metabolic_fraction: the stereo
    registry InChIKey did not match the non-isomeric clinical_pk.json
    SMILES, so the XGBoost-derived hepatic CL ran in parallel with the
    explicit ProdrugActivationEdges).

    Public-clone reproducible state (no DrugBank, no logp_correction;
    InChIKey-connectivity fallback active in both registry and
    cyp_clearance_overrides lookups):
      Meta AAFE  2.772  (overall N=107, %2-fold 44.9, %3-fold 63.6)
      Engine     4.065
      ML         3.010  (bit-identical — ML model artifacts unchanged)
      In-domain  2.862  (N=81)

    The +0.7% Meta drift relative to the 2.751 cache reflects the genuine
    removal of the parent-CL double-count for clopidogrel: predicted
    Engine Cmax 0.670 -> 2.873 mg/L (fold 2.23x -> 9.58x). The B-03
    spec's CES1 / CYP3A4 / CYP2C9-surrogate affinities (0.030 each) were
    calibrated to preserve the literature ~85/15 inactive/active fate
    split, not the absolute parent extraction ratio; the absolute total
    CL is now visible as uncalibrated and is tracked as a separate
    follow-up (literature IVIVE-scaled CES1 affinities, e.g. from
    Tang 2006 Vmax/Km, are out of scope for B-03).

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
    assert abs(aafe - 2.772) < 0.005, f"AAFE drifted: {aafe:.4f}"
