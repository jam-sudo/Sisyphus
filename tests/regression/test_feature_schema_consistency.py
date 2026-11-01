"""Guard: shipped XGBoost models' recorded feature schema must match the
feature pipeline their own manifest names.

Each shipped model records its training-time feature schema
(``name`` + ``n_features`` + a sha256 of that pipeline applied to caffeine);
this test recomputes both under the current code and asserts they still match,
catching drift at CI time before the runtime verifier rejects the model. The
feature bytes are RDKit-only (Morgan bit vector + deterministic RDKit
descriptors, plus the deterministic SMARTS ionization block for v2).

Complements the runtime fail-closed ``registry.verify_model_artifact`` check.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

# Shipped models that declare a compute_features feature_schema in their meta.
_META_PATHS = (
    "models/adme/xgboost_clint.meta.json",
    "models/adme/xgboost_vdss.meta.json",
    "models/adme/xgboost_fup_v2.meta.json",
    "models/adme/xgboost_peff.meta.json",
    "models/direct_pk/xgboost_cmax.meta.json",
    "models/direct_pk/xgboost_clf.meta.json",
    "models/direct_pk/xgboost_vdf.meta.json",
)

_CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def test_shipped_models_feature_schema_matches_its_named_pipeline():
    pytest.importorskip("rdkit")
    from sisyphus.descriptors import compute_features, compute_features_v2
    from sisyphus.ml.registry import compute_feature_hash_v1, compute_feature_hash_v2

    pipelines = {
        "compute_features_v1": (compute_feature_hash_v1(), len(compute_features(_CAFFEINE))),
        "compute_features_v2": (compute_feature_hash_v2(), len(compute_features_v2(_CAFFEINE))),
    }

    checked = 0
    problems: list[str] = []
    for rel in _META_PATHS:
        meta_path = _ROOT / rel
        if not meta_path.exists():
            continue
        schema = json.loads(meta_path.read_text()).get("feature_schema", {})
        name = schema.get("name")
        if name not in pipelines:
            continue  # artifact_only / logp_corr_6 / nothing recorded
        current_hash, current_n = pipelines[name]
        rec_n = schema.get("n_features")
        rec_sha = schema.get("sha256")
        if rec_n is None and not rec_sha:
            continue
        checked += 1
        if rec_n is not None and rec_n != current_n:
            problems.append(f"{rel}: {name} n_features {rec_n} != current {current_n}")
        if rec_sha and rec_sha != current_hash:
            problems.append(
                f"{rel}: {name} feature sha256 drift (recorded {rec_sha[:12]}…, "
                f"current {current_hash[:12]}…)"
            )

    assert checked >= 2, (
        f"expected to check >=2 shipped model feature schemas, checked {checked} — "
        "model metas missing/renamed?"
    )
    assert not problems, (
        "the feature pipeline drifted from a shipped model's feature schema; the "
        "models would predict on the wrong vector:\n  " + "\n  ".join(problems)
    )


def test_v2_prefix_is_bit_identical_to_v1():
    """v1 and v2 models share one featurization — v2 must extend, not reorder."""
    pytest.importorskip("rdkit")
    import numpy as np

    from sisyphus.descriptors import compute_features, compute_features_v2

    v1 = compute_features(_CAFFEINE)
    v2 = compute_features_v2(_CAFFEINE)
    assert np.array_equal(v1, v2[: len(v1)]), (
        "compute_features_v2 no longer starts with the exact v1 vector; mixed "
        "v1/v2 serving trims v2 down to 2057 and would feed v1 models garbage"
    )
