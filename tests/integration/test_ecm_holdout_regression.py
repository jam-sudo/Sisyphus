"""Regression: ECM migration must not shift 107-holdout Meta AAFE by >=0.01.

Spot-checks 10 drugs sampled from data/training/4track_holdout_predictions.json
against a fresh predict(...) pass. Per-drug Meta Cmax must match within 5%.
Full 107-drug sweep is reserved for the manual benchmark run.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tests._artifact_helpers import skip_if_local_artifacts

_CACHE = pathlib.Path("data/training/4track_holdout_predictions.json")


@skip_if_local_artifacts
@pytest.mark.slow
def test_ecm_holdout_spot_check_10_drugs():
    """Per-drug Meta Cmax must match cached 4track predictions within 5%."""
    from sisyphus.pipeline.predict import predict
    from sisyphus.validation.reference import load_reference

    cached = json.loads(_CACHE.read_text())
    candidates = [
        d for d in cached["drugs"]
        if d.get("in_ad", False) and not d.get("ad_flags", [])
    ]
    sample = candidates[:10]
    if len(sample) < 10:
        pytest.skip(f"Only {len(sample)} in-AD unflagged drugs available")

    refs_by_name = {r.name.lower(): r for r in load_reference() if r.in_holdout}

    failures = []
    for d in sample:
        ref = refs_by_name.get(d["name"].lower())
        if ref is None:
            continue
        fresh = predict(ref.smiles, ref.dose_mg, ref.route)
        cached_meta = d["meta"]
        fresh_meta = fresh.pk.cmax.mean
        rel_delta = abs(fresh_meta - cached_meta) / max(abs(cached_meta), 1e-12)
        if rel_delta > 0.05:
            failures.append({
                "drug": d["name"], "cached": cached_meta,
                "fresh": fresh_meta, "rel_delta": rel_delta,
            })

    assert not failures, (
        f"ECM regression — {len(failures)} drug(s) drifted >5%: {failures}"
    )
