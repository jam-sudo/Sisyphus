"""Tests for sisyphus.ml.registry — manifest validation and feature hashing."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from sisyphus.ml.registry import (
    CANONICAL_SMILES,
    ModelRecord,
    ModelRegistry,
    check_feature_hash,
    compute_feature_hash_v1,
    load_manifest,
    manifest_path_for,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_ADME = ROOT / "models" / "adme"
MODELS_DIRECT_PK = ROOT / "models" / "direct_pk"


# The 12 actively-loaded models whose manifests must ship with the repo.
ACTIVE_MODELS = [
    MODELS_DIRECT_PK / "xgboost_cmax.json",
    MODELS_DIRECT_PK / "xgboost_clf.json",
    MODELS_DIRECT_PK / "xgboost_vdf.json",
    MODELS_ADME / "xgboost_fup.json",
    MODELS_ADME / "xgboost_fup_v2.json",
    MODELS_ADME / "xgboost_clint.json",
    MODELS_ADME / "xgboost_rbp.json",
    MODELS_ADME / "xgboost_vdss.json",
    MODELS_ADME / "xgboost_peff.json",
    MODELS_ADME / "xgboost_pka_acidic.json",
    MODELS_ADME / "xgboost_pka_basic.json",
    MODELS_ADME / "logp_correction.json",
]


# ---------------------------------------------------------------------------
# Manifest path resolution
# ---------------------------------------------------------------------------


def test_manifest_path_for_adme():
    p = MODELS_ADME / "xgboost_fup_v2.json"
    assert manifest_path_for(p) == MODELS_ADME / "xgboost_fup_v2.meta.json"


def test_manifest_path_for_direct_pk():
    p = MODELS_DIRECT_PK / "xgboost_cmax.json"
    assert manifest_path_for(p) == MODELS_DIRECT_PK / "xgboost_cmax.meta.json"


# ---------------------------------------------------------------------------
# Shipped manifests: all 12 must exist, parse, and validate clean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_path", ACTIVE_MODELS, ids=lambda p: p.name)
def test_shipped_manifest_exists(model_path):
    mpath = manifest_path_for(model_path)
    assert mpath.exists(), f"manifest missing for {model_path.name}: {mpath}"


@pytest.mark.parametrize("model_path", ACTIVE_MODELS, ids=lambda p: p.name)
def test_shipped_manifest_validates(model_path):
    manifest = load_manifest(model_path)
    assert manifest is not None
    warnings = validate_manifest(manifest)
    assert warnings == [], f"{model_path.name}: {warnings}"


# Canonical hash computed under the lockfile env (rdkit 2026.3.1, numpy 2.2.6).
# Shipped manifest sha256 values are pinned to this env; CI installs the
# lockfile so CI and the shipped hashes agree by construction. Developers
# running outside the lockfile env (e.g. older RDKit) will see a mismatch
# and the hash-match tests skip — manifest hashes are NOT meant to track
# arbitrary dev environments, only the pinned production env.
LOCKFILE_V1_HASH = "b41dddd78533661b8f4aed86bdb7c5f55d15ed3cd2ad20b39e9100557eda631e"
LOCKFILE_LOGP6_HASH = "a8da0c08b6425b5d5967e38d60fe0130119ad66282a1fac2016bcbf3e9ab0e6d"


@pytest.mark.parametrize(
    "model_path",
    [p for p in ACTIVE_MODELS if p.name != "logp_correction.json"],
    ids=lambda p: p.name,
)
def test_shipped_manifest_feature_hash_matches_v1(model_path):
    """Shipped manifest hash matches compute_features_v1 under the lockfile env.

    Skips when the runtime env differs from the lockfile env (e.g. local
    dev on older RDKit). CI installs the lockfile so this runs there.
    """
    current = compute_feature_hash_v1()
    if current != LOCKFILE_V1_HASH:
        pytest.skip(
            f"compute_features_v1 hash {current} differs from the "
            f"lockfile-pinned {LOCKFILE_V1_HASH}; run under "
            f"requirements-lock.txt to exercise this test"
        )
    manifest = load_manifest(model_path)
    assert manifest is not None
    mismatch = check_feature_hash(manifest, current)
    assert mismatch is None, f"{model_path.name}: {mismatch}"


def test_logp_correction_uses_separate_feature_schema():
    """logp_correction.json uses the 6-feature pipeline, not compute_features_v1."""
    manifest = load_manifest(MODELS_ADME / "logp_correction.json")
    assert manifest is not None
    feat = manifest["feature_schema"]
    assert feat["name"] == "logp_corr_6"
    assert feat["n_features"] == 6
    # Distinct from compute_features_v1 hash
    v1 = compute_feature_hash_v1()
    assert feat["sha256"] != v1


# ---------------------------------------------------------------------------
# Registry.get() returns a ModelRecord
# ---------------------------------------------------------------------------


def test_registry_get_returns_model_record():
    reg = ModelRegistry()
    record = reg.get(MODELS_DIRECT_PK / "xgboost_cmax.json")
    assert isinstance(record, ModelRecord)
    assert record.version == "v3_clean"
    assert record.n_drugs_original == 1128
    assert record.target.startswith("log10(cmax")


def test_registry_get_missing_manifest_returns_none(tmp_path, caplog):
    """Missing manifest -> warn, return None (warn-only policy)."""
    fake_model = tmp_path / "nonexistent.json"
    reg = ModelRegistry()
    with caplog.at_level(logging.WARNING):
        result = reg.get(fake_model)
    assert result is None
    assert any("manifest missing" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# register() is strict on missing fields
# ---------------------------------------------------------------------------


def test_registry_register_rejects_incomplete_manifest(tmp_path):
    reg = ModelRegistry()
    incomplete = {"version": "v1", "target": "log10(cmax)"}  # missing most fields
    with pytest.raises(ValueError, match="manifest incomplete"):
        reg.register(tmp_path / "fake.json", incomplete)


def test_registry_register_writes_complete_manifest(tmp_path):
    reg = ModelRegistry()
    manifest = {
        "version": "v1",
        "target": "log10(cmax)",
        "trained_on": {"dataset_path": "test.csv", "sha256": "unknown_legacy"},
        "feature_schema": {
            "name": "compute_features_v1",
            "n_features": 2057,
            "sha256": "dd014cd8",
            "description": "test",
        },
        "trained_at": "2026-04-24T00:00:00Z",
        "n_drugs_original": 100,
        "n_drugs_excluded": 5,
        "holdout_version": "v1",
        "holdout_metric": {"name": "AAFE", "value": 2.5},
        "hyperparameters": {"n_estimators": 100},
        "retrained_reason": "test",
    }
    model_path = tmp_path / "fake_model.json"
    reg.register(model_path, manifest)

    written = manifest_path_for(model_path)
    assert written.exists()
    loaded = json.loads(written.read_text())
    assert loaded == manifest


# ---------------------------------------------------------------------------
# Feature-hash mismatch produces a warning string
# ---------------------------------------------------------------------------


def test_check_feature_hash_mismatch():
    manifest = {"feature_schema": {"sha256": "abc"}}
    msg = check_feature_hash(manifest, "def")
    assert msg is not None
    assert "mismatch" in msg


def test_check_feature_hash_missing_sha():
    manifest = {"feature_schema": {}}
    msg = check_feature_hash(manifest, "def")
    assert msg is not None
    assert "not recorded" in msg


def test_check_feature_hash_match():
    manifest = {"feature_schema": {"sha256": "abc"}}
    msg = check_feature_hash(manifest, "abc")
    assert msg is None


# ---------------------------------------------------------------------------
# Canonical hash recomputation is stable
# ---------------------------------------------------------------------------


def test_compute_feature_hash_v1_stable():
    """compute_features_v1 hash must not drift between calls."""
    a = compute_feature_hash_v1()
    b = compute_feature_hash_v1()
    assert a == b
    # sha256 hex string length
    assert len(a) == 64


def test_canonical_smiles_is_caffeine():
    """Guard against accidental canonical drug change."""
    assert CANONICAL_SMILES == "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
