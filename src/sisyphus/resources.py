"""Explicit resource resolution for data and model artifacts.

Runtime code must not depend on the caller's current working directory.  A
deployment may set ``SISYPHUS_ROOT`` to a model bundle; source checkouts are
detected from this module's location.  Missing resources fail with a path-rich
error instead of silently selecting a different local profile.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ResourceConfig:
    """Root directories used by one explicit Sisyphus execution profile."""

    root: Path
    profile: str = "public"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def data(self, *parts: str, required: bool = True) -> Path:
        path = self.data_dir.joinpath(*parts)
        if required and not path.exists():
            raise FileNotFoundError(f"Sisyphus data resource not found: {path}")
        return path

    def model(self, *parts: str, required: bool = True) -> Path:
        path = self.models_dir.joinpath(*parts)
        if required and not path.exists():
            raise FileNotFoundError(f"Sisyphus model resource not found: {path}")
        return path


def _default_root() -> Path:
    configured = os.environ.get("SISYPHUS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    # src/sisyphus/resources.py -> repository root
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "data").is_dir() and (candidate / "models").is_dir():
        return candidate

    raise FileNotFoundError(
        "Unable to locate Sisyphus data/model bundle. Set SISYPHUS_ROOT to "
        "the directory containing data/ and models/."
    )


def get_resource_config(profile: str | None = None) -> ResourceConfig:
    """Return the active resource configuration.

    ``SISYPHUS_PROFILE`` is descriptive today (``public`` is the supported
    default) and is surfaced for provenance.  Licensed enrichment must be
    explicitly selected by deployments rather than discovered by cwd.
    """

    active_profile = profile or os.environ.get("SISYPHUS_PROFILE", "public")
    if active_profile not in {"public", "licensed_research"}:
        raise ValueError(
            "SISYPHUS_PROFILE must be 'public' or 'licensed_research', got "
            f"{active_profile!r}"
        )
    return ResourceConfig(root=_default_root(), profile=active_profile)


@lru_cache(maxsize=8)
def _artifact_provenance_cached(
    root_str: str, profile: str
) -> tuple[tuple[str, str], ...]:
    root = Path(root_str)
    fixed_paths = (
        "data/model_card.json",
        "data/physiology/reference_man.yaml",
        "data/validation/development_residual_interval.json",
        "data/validation/feature_schema_v1.json",
        "data/validation/solver_settings_v1.json",
        "models/direct_pk/xgboost_cmax.json",
        "models/direct_pk/xgboost_cmax.meta.json",
        "models/direct_pk/xgboost_clf.json",
        "models/direct_pk/xgboost_clf.meta.json",
        "models/direct_pk/xgboost_vdf.json",
        "models/direct_pk/xgboost_vdf.meta.json",
        "models/adme/xgboost_fup.json",
        "models/adme/xgboost_fup.meta.json",
        "models/adme/xgboost_fup_v2.json",
        "models/adme/xgboost_fup_v2.meta.json",
        "models/adme/xgboost_clint.json",
        "models/adme/xgboost_clint.meta.json",
        "models/adme/xgboost_peff.json",
        "models/adme/xgboost_peff.meta.json",
        "models/adme/xgboost_vdss.json",
        "models/adme/xgboost_vdss.meta.json",
    )
    discovered: set[str] = set(fixed_paths)
    for pattern in (
        "data/enzymes/*.json",
        "data/transporters/*.json",
        "data/sbi/prodrug_activation_registry.json",
    ):
        discovered.update(str(path.relative_to(root)) for path in root.glob(pattern))
    if profile == "licensed_research":
        discovered.update(
            {
                "models/adme/logp_correction.json",
                "models/adme/logp_correction.meta.json",
            }
        )
        discovered.update(str(path.relative_to(root)) for path in root.glob("data/drugbank/*"))

    values: list[tuple[str, str]] = [("resource_profile", profile)]
    for relative in sorted(discovered):
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        values.append((relative, digest))
    return tuple(values)


def artifact_provenance(profile: str | None = None) -> tuple[tuple[str, str], ...]:
    """SHA256 inventory of every external artifact on the primary Cmax path.

    The cache key includes the resolved bundle root, preventing provenance from
    one ``SISYPHUS_ROOT`` being reused after a deployment switches bundles.
    """

    resources = get_resource_config(profile)
    return _artifact_provenance_cached(str(resources.root), resources.profile)
