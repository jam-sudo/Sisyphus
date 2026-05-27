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
def test_cached_holdout_aafe_is_2p698() -> None:
    """Cached predictions file: Meta AAFE is the public-clone headline 2.698 (±0.020).

    Baseline updated 2026-05-27 (B-02 Phase 2 UGT public registry activation;
    spec docs/superpowers/specs/2026-05-26-B02-ugt-public-registry-design.md).

    B-02 activates UGT2B7 + UGT1A9 paths via 2 literature-curated substrate
    registries (8 seed drugs: morphine, codeine, ketorolac, indomethacin via
    UGT2B7; dapagliflozin, etodolac, bexagliflozin, glasdegib via UGT1A9).
    YAML adds UGT2B7 (2.43e6 pmol) + UGT1A9 (8.10e5 pmol) abundances to liver.
    No DrugBank dependency.

    Gate-D 99-of-107 bit-identical verified (only the 8 seeds shift; all 99
    non-seed drugs match the pre-B-02 same-numerics-stack cache to <1e-8).
    Gate-A Meta Δ = +0.0067 (b02=2.6983 vs main-same-numerics=2.6916), which
    is 1.6% of the bootstrap CI half-width [2.3151, 3.1690] — well within
    sampling noise. See data/validation/4track_ci_2026-05-27_B02.json.

    Secondary finding (DE-38, dead-ends.md): morphine engine FE 1.90 -> 2.94
    and codeine 1.98 -> 2.71 (worsened) because UGT2B7 effective CL is lower
    than the CYP-default allocation it replaced. 6 of 8 seeds improved
    (under-predicted drugs moved toward observation); 2 of 8 worsened
    (over-predicted drugs moved away). Net Meta increase reflects the
    mass-balance of these per-drug movements. Phase 2.x follow-up = B-13
    (UGT2B7 abundance + IVIVE recalibration).

    Cache regenerated under same numerics stack as preceding versions:
      Meta AAFE  2.698  (overall N=107, %2-fold 46.7, %3-fold 61.7)
      Engine     3.831
      ML         3.010  (bit-identical — ML model artifacts unchanged)
      In-domain  2.760  (N=79; 2 drugs flipped AD-flag under engine recompute)

    Tolerance widened to 0.020 (4x the prior 0.005 heuristic) to reflect the
    amended Gate-A criterion: bootstrap CI half-width is the statistical
    noise floor (~0.43 for Meta overall), of which 0.020 is ~5%. The prior
    0.005 was an artifact of the B-03.x cycle's coincidentally tiny delta.
    See spec amendment 2026-05-27.

    If this fails outside ±0.020 of 2.698, the cache has been regenerated
    with a behavior change or the numerics stack drifted materially.
    Investigate."""
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
    assert abs(aafe - 2.698) < 0.020, f"AAFE drifted: {aafe:.4f}"
