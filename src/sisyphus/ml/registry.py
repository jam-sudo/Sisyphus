"""Model manifest registry.

Provides structured access to the `<model>.meta.json` manifest that
accompanies every XGBoost artifact loaded by the pipeline. Enforces the
warn-only provenance policy defined in
``docs/science/model_manifest_schema.md``.

Behaviour is intentionally non-blocking for legacy artifacts:

    - Missing manifest        → warning, returns ``None``
    - Incomplete manifest     → warning per missing field
    - Feature-hash mismatch   → warning, does not prevent load

``register_model()`` is stricter: it refuses to write an incomplete
manifest so newly trained models cannot ship without provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


REQUIRED_TOP_LEVEL = (
    "version",
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
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRecord:
    """Metadata for a trained model artifact.

    Mirrors the manifest JSON shape.  Keys not recovered from historical
    context carry the literal string ``"unknown_legacy"`` (or ``None``
    for numeric fields) per the schema.
    """

    model_path: Path
    version: str
    target: str
    trained_on: dict[str, Any]
    feature_schema: dict[str, Any]
    trained_at: str
    n_drugs_original: int | None
    n_drugs_excluded: int | None
    holdout_version: str
    holdout_metric: dict[str, Any]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    retrained_reason: str = "unknown_legacy"

    @classmethod
    def from_manifest(cls, model_path: Path, manifest: dict[str, Any]) -> ModelRecord:
        return cls(
            model_path=model_path,
            version=manifest.get("version", "unknown_legacy"),
            target=manifest.get("target", "unknown_legacy"),
            trained_on=manifest.get("trained_on", {}),
            feature_schema=manifest.get("feature_schema", {}),
            trained_at=manifest.get("trained_at", "unknown_legacy"),
            n_drugs_original=manifest.get("n_drugs_original"),
            n_drugs_excluded=manifest.get("n_drugs_excluded"),
            holdout_version=manifest.get("holdout_version", "unknown_legacy"),
            holdout_metric=manifest.get("holdout_metric", {}),
            hyperparameters=manifest.get("hyperparameters", {}),
            retrained_reason=manifest.get("retrained_reason", "unknown_legacy"),
        )


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


# ---------------------------------------------------------------------------
# Register (strict)
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Facade over ``load_manifest`` / ``validate_manifest``.

    Retained as a class for API symmetry with the previous stub.  The
    free functions above are the primary entry points.
    """

    def get(self, model_path: Path) -> ModelRecord | None:
        """Load and structure a manifest.  Returns ``None`` if missing."""
        manifest = load_manifest(model_path)
        if manifest is None:
            return None
        for w in validate_manifest(manifest):
            logger.warning("%s: %s", model_path.name, w)
        return ModelRecord.from_manifest(model_path, manifest)

    def register(self, model_path: Path, manifest: dict[str, Any]) -> None:
        """Write a manifest for a newly trained model.

        Strict: raises ``ValueError`` if any required field is missing
        or if ``feature_schema.sha256`` is absent.  This guards new
        artifacts — legacy manifests are handled via ``get`` (warn only).
        """
        warnings = validate_manifest(manifest)
        if warnings:
            raise ValueError(
                f"cannot register {model_path}: manifest incomplete "
                f"({'; '.join(warnings)})"
            )
        mpath = manifest_path_for(model_path)
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("wrote model manifest: %s", mpath)
