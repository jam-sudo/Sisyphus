"""Real-data proof that split-conformal fixes the 29.9%@90% PI under-coverage.

Computes leave-one-out cross-conformal coverage on the committed holdout cache
(``data/training/4track_holdout_predictions.json``). This MEASURES the conformal
algorithm's marginal-coverage property on the holdout (the holdout's designated
role is validation) — it does NOT tune any deployed parameter on the holdout
(the production quantile is calibrated on train / CV+; Invariant #5). Documented
MC baseline at nominal 90% is 0.299 (CLAUDE.md); conformal restores ~0.90.
"""

import json
import pathlib

import numpy as np
import pytest

from sisyphus.validation.conformal import (
    conformal_quantile,
    nonconformity_scores,
)

_CACHE = pathlib.Path("data/training/4track_holdout_predictions.json")


def _meta_pred_obs():
    d = json.loads(_CACHE.read_text())
    drugs = [
        r for r in d["drugs"]
        if r.get("meta") and r["meta"] > 0 and r.get("obs") and r["obs"] > 0
    ]
    pred = np.array([r["meta"] for r in drugs], dtype=float)
    obs = np.array([r["obs"] for r in drugs], dtype=float)
    return pred, obs


def _loo_conformal_coverage(pred, obs, alpha):
    n = len(pred)
    covered = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        q = conformal_quantile(nonconformity_scores(pred[mask], obs[mask]), alpha)
        lo, hi = pred[i] / 10 ** q, pred[i] * 10 ** q
        covered += obs[i] >= lo and obs[i] <= hi
    return covered / n


@pytest.mark.skipif(not _CACHE.exists(), reason="holdout cache not present")
def test_conformal_restores_90pct_coverage():
    """Conformal 90% LOO coverage >= 0.85 on the holdout (MC gives 0.299)."""
    pred, obs = _meta_pred_obs()
    cov = _loo_conformal_coverage(pred, obs, alpha=0.1)
    assert cov >= 0.85, f"conformal 90% LOO coverage {cov:.3f} (MC baseline 0.299)"


@pytest.mark.skipif(not _CACHE.exists(), reason="holdout cache not present")
def test_conformal_calibrated_across_levels():
    """Coverage tracks the nominal level (reliability) within finite-sample slack."""
    pred, obs = _meta_pred_obs()
    for alpha, lo_bound in [(0.5, 0.40), (0.2, 0.72), (0.1, 0.85), (0.05, 0.90)]:
        cov = _loo_conformal_coverage(pred, obs, alpha)
        nominal = 1 - alpha
        assert cov >= lo_bound, f"nominal {nominal:.2f}: coverage {cov:.3f} < {lo_bound}"
        assert cov <= nominal + 0.08, f"nominal {nominal:.2f}: coverage {cov:.3f} over-conservative"
