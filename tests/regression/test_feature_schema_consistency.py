"""Guard: shipped XGBoost models' recorded feature schema must match the current
``compute_features`` pipeline.

The production predictors load their models with a bare ``load_model()``; if
``descriptors.compute_features`` ever drifts (feature count, or the 2048-Morgan +
9-descriptor byte layout / order), the models would silently predict on the wrong
vector with no error. Each shipped model records its training-time feature schema
(``n_features`` + a sha256 of ``compute_features(caffeine)``); this test recomputes
both under the current code and asserts they still match, catching drift at CI
time. The feature bytes are RDKit-only (Morgan bit vector + deterministic RDKit
descriptors), so the sha256 is stable across BLAS/numpy stacks — a mismatch means
a real pipeline change, not numerics drift.

Complements the *runtime* warn-only check wired via
``registry.warn_on_feature_schema_drift`` in the model loaders.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

# Shipped models that declare a v1 feature_schema in their sibling meta.
_META_PATHS = (
    "models/adme/xgboost_clint.meta.json",
    "models/direct_pk/xgboost_clf.meta.json",
    "models/direct_pk/xgboost_vdf.meta.json",
)

_CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def test_shipped_models_feature_schema_matches_compute_features():
    pytest.importorskip("rdkit")
    from sisyphus.descriptors import compute_features
    from sisyphus.ml.registry import compute_feature_hash_v1

    current_hash = compute_feature_hash_v1()
    current_n = len(compute_features(_CAFFEINE))

    checked = 0
    problems: list[str] = []
    for rel in _META_PATHS:
        meta_path = _ROOT / rel
        if not meta_path.exists():
            continue
        schema = json.loads(meta_path.read_text()).get("feature_schema", {})
        rec_n = schema.get("n_features")
        rec_sha = schema.get("sha256")
        if rec_n is None and not rec_sha:
            continue  # nothing to check against
        checked += 1
        if rec_n is not None and rec_n != current_n:
            problems.append(f"{rel}: n_features {rec_n} != current {current_n}")
        if rec_sha and rec_sha != current_hash:
            problems.append(
                f"{rel}: feature sha256 drift (recorded {rec_sha[:12]}…, "
                f"current {current_hash[:12]}…)"
            )

    assert checked >= 2, (
        f"expected to check >=2 shipped model feature schemas, checked {checked} — "
        "model metas missing/renamed?"
    )
    assert not problems, (
        "compute_features drifted from shipped model feature schema; the models "
        "would predict on the wrong vector:\n  " + "\n  ".join(problems)
    )
