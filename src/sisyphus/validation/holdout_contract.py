"""Shared executable contracts for the frozen external-holdout workflow."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = ROOT / "data" / "reference"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_payload(payload: Any, schema_filename: str) -> None:
    """Validate a payload and report every JSON-path error deterministically."""

    try:
        json.dumps(payload, allow_nan=False)
    except ValueError as exc:
        raise ValueError("JSON payload contains a non-finite number") from exc
    schema = json.loads((SCHEMA_DIR / schema_filename).read_text())
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        rendered = []
        for error in errors:
            path = "$" + "".join(
                f"[{part}]" if isinstance(part, int) else f".{part}"
                for part in error.absolute_path
            )
            rendered.append(f"{path}: {error.message}")
        raise ValueError("JSON schema validation failed: " + "; ".join(rendered))


_SOURCE_HASH_FIELDS = (
    "dose_mg",
    "route",
    "dosage_form",
    "release_type",
    "food_state",
    "salt_form",
    "analyte",
    "matrix",
    "dose_regimen",
    "population",
    "co_medications",
    "study_n",
    "source",
)


def source_record_hash(label_arm: dict[str, Any]) -> str:
    """Bind the label metadata to the blinded manifest without hashing outcome."""

    return canonical_sha256({key: label_arm.get(key) for key in _SOURCE_HASH_FIELDS})


def primary_ineligibility_reasons(label_arm: dict[str, Any]) -> tuple[str, ...]:
    """Derive primary eligibility; a supplied boolean is never trusted."""

    population = label_arm.get("population") or {}
    checks = {
        "route_not_oral": label_arm.get("route") != "oral",
        "release_not_ir": label_arm.get("release_type") != "IR",
        "not_fasted": label_arm.get("food_state") != "fasted",
        "not_parent_analyte": label_arm.get("analyte") != "parent",
        "matrix_not_plasma": label_arm.get("matrix") != "plasma",
        "not_single_dose": label_arm.get("dose_regimen") != "single",
        "not_adult": population.get("age_group") != "adult",
        "not_healthy": population.get("health_status") != "healthy",
        "co_medication_present": bool(label_arm.get("co_medications")),
    }
    return tuple(reason for reason, failed in checks.items() if failed)


def is_primary_eligible(label_arm: dict[str, Any]) -> bool:
    return not primary_ineligibility_reasons(label_arm)


def validate_source_quotas(manifest: dict[str, Any]) -> dict[str, float]:
    """Enforce preregistered primary-compound source quotas from blinded fields."""

    compound_sources: list[tuple[str, str | None]] = []
    for compound in manifest["compounds"]:
        primary = [arm for arm in compound["arms"] if arm["primary_eligible"]]
        if not primary:
            continue
        sources = {(arm["source_category"], arm.get("source_agency")) for arm in primary}
        if len(sources) != 1:
            raise ValueError(
                f"Primary arms for {compound['candidate_id']} must share one source category/agency"
            )
        compound_sources.append(next(iter(sources)))

    if not compound_sources:
        raise ValueError("No primary compounds available for source-quota validation")
    n = len(compound_sources)
    regulatory = sum(category == "regulatory" for category, _ in compound_sources)
    regulatory_fraction = regulatory / n
    if regulatory_fraction < 0.70:
        raise ValueError(
            f"Regulatory primary-compound fraction {regulatory_fraction:.3f} is below 0.70"
        )
    agency_counts = Counter(
        agency for category, agency in compound_sources
        if category == "regulatory" and agency is not None
    )
    max_agency_fraction = max(agency_counts.values(), default=0) / n
    if max_agency_fraction > 0.30:
        agency, count = agency_counts.most_common(1)[0]
        raise ValueError(
            f"Agency {agency!r} supplies {count}/{n}={count / n:.3f}, above 0.30"
        )
    return {
        "regulatory_fraction": regulatory_fraction,
        "max_agency_fraction": max_agency_fraction,
    }


def resolve_frozen_path(root: Path, relative: str) -> Path:
    """Resolve a frozen repository path without permitting path traversal."""

    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Frozen path escapes repository root: {relative!r}") from exc
    if not path.is_file():
        raise ValueError(f"Frozen file does not exist: {relative!r}")
    return path


def verify_frozen_file(root: Path, freeze: dict[str, Any], stem: str) -> str:
    path = resolve_frozen_path(root, str(freeze[f"{stem}_path"]))
    actual = sha256_file(path)
    expected = str(freeze[f"{stem}_sha256"])
    if actual != expected:
        raise ValueError(f"{stem} SHA256 mismatch: expected={expected}, actual={actual}")
    return actual
