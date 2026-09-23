"""Validate model manifests and fail closed on artifact or feature drift."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


REQUIRED_TOP_LEVEL = (
    "version",
    "artifact_sha256",
    "target",
    "trained_on",
    "feature_schema",
    "trained_at",
    "n_drugs_original",
    "n_drugs_excluded",
    "holdout_version",
    "holdout_metric",
    "hyperparameters",
    "retrained_reason",
)

REQUIRED_FEATURE_SCHEMA = ("name", "n_features", "sha256", "description")

# Canonical SMILES used to compute the feature-vector fingerprint.
# See docs/science/model_manifest_schema.md.
CANONICAL_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine


# ---------------------------------------------------------------------------
# Manifest path resolution
# ---------------------------------------------------------------------------


def manifest_path_for(model_path: Path) -> Path:
    """Return the canonical manifest path next to a model artifact.

    Two conventions are supported:

    - ``models/adme/xgboost_fup_v2.json``    → ``models/adme/xgboost_fup_v2.meta.json``
    - ``models/direct_pk/xgboost_cmax.json`` → ``models/direct_pk/xgboost_cmax.meta.json``
    """
    return model_path.with_suffix(".meta.json")


# ---------------------------------------------------------------------------
# Manifest load / validate
# ---------------------------------------------------------------------------


def load_manifest(model_path: Path) -> dict[str, Any] | None:
    """Load the manifest accompanying a model artifact, or None.

    Emits a logger warning if the manifest file is missing.  Does not
    validate schema — see :func:`validate_manifest`.
    """
    mpath = manifest_path_for(model_path)
    if not mpath.exists():
        logger.warning("model manifest missing: %s", mpath)
        return None
    try:
        with open(mpath) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("model manifest unreadable (%s): %s", mpath, e)
        return None


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Validate a manifest against the schema.

    Returns a list of warning strings (empty list = clean).  Does not
    raise — caller decides whether warnings are fatal.
    """
    warnings: list[str] = []
    for field_name in REQUIRED_TOP_LEVEL:
        if field_name not in manifest:
            warnings.append(f"manifest field missing: {field_name}")

    feat = manifest.get("feature_schema")
    if isinstance(feat, dict):
        for key in REQUIRED_FEATURE_SCHEMA:
            if key not in feat:
                warnings.append(f"feature_schema field missing: {key}")

    return warnings


# ---------------------------------------------------------------------------
# Feature hash
# ---------------------------------------------------------------------------


def compute_feature_hash_v1() -> str:
    """Compute sha256 of compute_features(caffeine) under the v1 pipeline.

    Uses the 2048-Morgan + 9-descriptor pipeline shared by the majority
    of the pipeline's XGBoost models.
    """
    from sisyphus.descriptors import compute_features

    feats = compute_features(CANONICAL_SMILES)
    return hashlib.sha256(feats.tobytes()).hexdigest()


def check_feature_hash(manifest: dict[str, Any], expected_sha256: str) -> str | None:
    """Return a warning string if the manifest's feature hash disagrees.

    ``expected_sha256`` is the hash recomputed from the current code
    path.  A mismatch means the feature pipeline has drifted since the
    model was trained and its predictions may be meaningless.  Returns
    ``None`` on match.
    """
    feat = manifest.get("feature_schema", {})
    recorded = feat.get("sha256") if isinstance(feat, dict) else None
    if recorded is None:
        return "feature_schema.sha256 not recorded"
    if recorded != expected_sha256:
        return (
            f"feature schema hash mismatch: expected {expected_sha256}, "
            f"got {recorded}"
        )
    return None


@functools.lru_cache(maxsize=1)
def _current_feature_hash_v1() -> str:
    """Current-pipeline feature hash, cached (one caffeine featurization/process)."""
    return compute_feature_hash_v1()


def verify_model_artifact(model_json_path: Path | str) -> dict[str, Any]:
    """Fail closed on manifest, artifact-integrity, or feature-schema drift."""

    path = Path(model_json_path)
    manifest = load_manifest(path)
    if manifest is None:
        raise ValueError(f"Model manifest unavailable for {path}")
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError(f"Model manifest incomplete for {path}: {'; '.join(problems)}")
    actual_artifact_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest["artifact_sha256"] != actual_artifact_sha:
        raise ValueError(
            f"Model artifact hash mismatch for {path}: "
            f"manifest={manifest['artifact_sha256']}, actual={actual_artifact_sha}"
        )
    feature_schema = manifest.get("feature_schema", {})
    if feature_schema.get("name") == "compute_features_v1":
        warning = check_feature_hash(manifest, _current_feature_hash_v1())
        if warning:
            raise ValueError(f"Feature-schema drift for {path}: {warning}")
    return manifest
