"""Shared executable contracts for the frozen external-holdout workflow."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = ROOT / "data" / "reference"

_PRODUCTION_FITTED_MODELS = (
    "models/direct_pk/xgboost_cmax.meta.json",
    "models/direct_pk/xgboost_clf.meta.json",
    "models/direct_pk/xgboost_vdf.meta.json",
    "models/adme/xgboost_fup_v2.meta.json",
    "models/adme/xgboost_clint.meta.json",
    "models/adme/xgboost_vdss.meta.json",
    "models/adme/xgboost_peff.meta.json",
)

# This historical artifact was fitted with licensed DrugBank fup targets as well as TDC.
# A public-only membership inventory cannot certify it by relabeling its metadata.
_DRUGBANK_FUP_ARTIFACT_SHA256 = "3ac08bf5dee4a7b5f7ebf9e058882f6101f965959f88a094de4db76c4b47e1f1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def label_content_sha256(labels: dict[str, Any]) -> str:
    """Commit blinded label content before the manifest hash is known."""

    return canonical_sha256({"cycle_id": labels["cycle_id"], "records": labels["records"]})


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
    "postdose_fast_h",
    "salt_form",
    "dose_basis",
    "dose_basis_evidence",
    "analyte",
    "matrix",
    "dose_regimen",
    "population",
    "co_medications",
    "cmax_statistic",
    "study_n",
    "source",
    "verified_by",
)


def source_record_hash(label_arm: dict[str, Any]) -> str:
    """Bind the label metadata to the blinded manifest without hashing outcome."""

    return canonical_sha256({key: label_arm.get(key) for key in _SOURCE_HASH_FIELDS})


def primary_ineligibility_reasons(label_arm: dict[str, Any]) -> tuple[str, ...]:
    """Derive primary eligibility; a supplied boolean is never trusted."""

    population = label_arm.get("population") or {}
    postdose_fast_h = label_arm.get("postdose_fast_h")
    checks = {
        "route_not_oral": label_arm.get("route") != "oral",
        "release_not_ir": label_arm.get("release_type") != "IR",
        "not_fasted": label_arm.get("food_state") != "fasted",
        "postdose_fast_under_4h": (
            isinstance(postdose_fast_h, bool)
            or not isinstance(postdose_fast_h, (int, float))
            or not math.isfinite(postdose_fast_h)
            or postdose_fast_h < 4
        ),
        "dose_basis_unverified": label_arm.get("dose_basis") != "parent_active_moiety",
        "dose_basis_evidence_missing": not label_arm.get("dose_basis_evidence"),
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


def verify_parent_prediction(smiles: str, candidate_id: str) -> None:
    """Reject compounds whose runtime Cmax refers to an active metabolite."""

    from sisyphus.predict.registry import lookup_active_metabolite

    routed = lookup_active_metabolite(smiles)
    if routed is not None and routed[1] != "parent":
        raise ValueError(
            f"{candidate_id}: prediction observes an active metabolite, "
            "but the primary label requires parent Cmax"
        )


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


def verify_audit_report(path: Path, expected_sha: str, manifest_sha: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError("Audit-report SHA256 does not match the committed value")
    report = json.loads(path.read_text())
    if report.get("pass") is not True:
        raise ValueError("External holdout audit report did not pass")
    if report.get("manifest_sha256") != manifest_sha:
        raise ValueError("Audit report was not produced from this exact manifest")
    return actual


def verify_training_membership(
    root: Path, freeze: dict[str, Any], expected_paths: set[str] | None = None
) -> str:
    """Require the public fitted-target inventory and every source it names."""

    inventory_path = "data/validation/training_membership_sources_v1.json"
    if freeze["training_membership_path"] != inventory_path:
        raise ValueError("Training membership must use the pinned public inventory")
    inventory_sha = verify_frozen_file(root, freeze, "training_membership")
    inventory = json.loads((root / inventory_path).read_text())
    if inventory.get("profile") != "public":
        raise ValueError("Training membership profile must be public")
    sources = inventory.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Training membership sources are missing")
    paths = [row["path"] for row in sources]
    if len(paths) != len(set(paths)) or (
        expected_paths is not None and set(paths) != expected_paths
    ):
        raise ValueError("Training membership sources differ from fitted-target corpora")
    for row in sources:
        path = resolve_frozen_path(root, row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Training membership source SHA256 mismatch: {row['path']}")
    for model_path in _PRODUCTION_FITTED_MODELS:
        metadata = json.loads(resolve_frozen_path(root, model_path).read_text())
        artifact_path = model_path.replace(".meta.json", ".json")
        if sha256_file(resolve_frozen_path(root, artifact_path)) != metadata.get("artifact_sha256"):
            raise ValueError(f"Production model artifact SHA256 mismatch: {artifact_path}")
        if (
            model_path == "models/adme/xgboost_fup_v2.meta.json"
            and sha256_file(resolve_frozen_path(root, "models/adme/xgboost_fup_v2.json"))
            == _DRUGBANK_FUP_ARTIFACT_SHA256
        ):
            raise ValueError("Shipped fup v2 uses DrugBank targets outside the public profile")
        trained_on = metadata.get("trained_on") or {}
        dataset = trained_on.get("dataset_path")
        digest = trained_on.get("sha256")
        if (
            not isinstance(dataset, str)
            or dataset not in paths
            or not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
        ):
            raise ValueError(f"Unverifiable production model training source: {model_path}")
        if sha256_file(resolve_frozen_path(root, dataset)) != digest:
            raise ValueError(f"Production model training source SHA256 mismatch: {model_path}")
    return inventory_sha


def verify_development_residual_interval(root: Path) -> float:
    """Require the primary residual band to belong to the frozen model stack."""

    path = root / "data/validation/development_residual_interval.json"
    artifact = json.loads(path.read_text())
    if (
        artifact.get("method") != "development_empirical_residual_quantile"
        or artifact.get("calibration_set") != "partially_in_sample_development"
        or artifact.get("source_cache_sha256") != sha256_file(
            root / "data/training/4track_holdout_predictions.json"
        )
    ):
        raise ValueError("Development residual interval source is not current")
    expected_models = {
        model_path.replace(".meta.json", ".json") for model_path in _PRODUCTION_FITTED_MODELS
    }
    recorded_models = artifact.get("model_artifact_sha256")
    if not isinstance(recorded_models, dict) or set(recorded_models) != expected_models:
        raise ValueError("Development residual interval model inventory is incomplete")
    for model_path, digest in recorded_models.items():
        if digest != sha256_file(root / model_path):
            raise ValueError(f"Development residual interval model hash mismatch: {model_path}")
    try:
        q = float(artifact["tracks"]["meta"]["0.1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Development residual interval lacks a 90% Meta quantile") from exc
    if not math.isfinite(q) or q <= 0:
        raise ValueError("Development residual interval 90% Meta quantile is invalid")
    return q


def verify_source_plan(
    manifest_path: Path,
    manifest: dict[str, Any],
    structure_key: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Bind the declared acquisition counts to the custodian's actual ID files."""

    root = manifest_path.parent
    plan_path = resolve_frozen_path(root, manifest["source_plan_path"])
    if sha256_file(plan_path) != manifest["source_plan_sha256"]:
        raise ValueError("source_plan_sha256 does not match source_plan_path")
    plan = json.loads(plan_path.read_text())
    validate_payload(plan, "external_holdout_v1_source_plan.schema.json")
    if plan["cycle_id"] != manifest["cycle_id"] or plan["final_test_n"] != manifest["n_target"]:
        raise ValueError("Source plan cycle or final-test size does not match manifest")

    files = {}
    for stem in ("inventory", "verified_shortlist", "allocation", "exclusion_flow"):
        path = resolve_frozen_path(root, plan[f"{stem}_path"])
        if sha256_file(path) != plan[f"{stem}_sha256"]:
            raise ValueError(f"{stem} SHA256 mismatch")
        files[stem] = json.loads(path.read_text())

    def indexed(rows: Any, label: str, count: int) -> dict[str, dict[str, Any]]:
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"{label} count does not match source plan")
        result = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
                raise ValueError(f"{label} contains an invalid candidate ID")
            cid = row["candidate_id"]
            if not cid or cid in result:
                raise ValueError(f"{label} contains a blank or duplicate candidate ID")
            result[cid] = row
        return result

    inventory = indexed(files["inventory"], "inventory", plan["inventory_n"])
    verified = indexed(files["verified_shortlist"], "verified shortlist", plan["verified_n"])
    if not verified.keys() <= inventory.keys():
        raise ValueError("Verified shortlist contains candidates absent from inventory")
    windows = plan["source_windows"]
    for cid, row in inventory.items():
        required = ("name", "source_family", "source_date", "source_ref")
        if not all(isinstance(row.get(key), str) and row[key] for key in required):
            raise ValueError(f"Inventory identity or source is incomplete: {cid}")
        try:
            source_date = date.fromisoformat(row["source_date"])
        except ValueError as exc:
            raise ValueError(f"Invalid inventory source date: {cid}") from exc
        if not any(
            window["source_family"] == row["source_family"]
            and date.fromisoformat(window["start_date"]) <= source_date
            <= date.fromisoformat(window["end_date"])
            for window in windows
        ):
            raise ValueError(f"Inventory source is outside frozen windows: {cid}")
    for cid, row in verified.items():
        if (
            row.get("name") != inventory[cid]["name"]
            or not isinstance(row.get("smiles"), str)
            or not row["smiles"]
        ):
            raise ValueError(f"Verified identity or structure is incomplete: {cid}")
        synonyms = row.get("synonyms")
        relations = row.get("related_structures")
        if (
            not isinstance(synonyms, list)
            or not all(isinstance(name, str) and name.strip() for name in synonyms)
            or len(synonyms) != len(set(synonyms))
            or not isinstance(relations, list)
            or any(
                not isinstance(relation, dict)
                or set(relation) != {"relationship", "smiles", "source_ref"}
                or relation["relationship"] not in {"parent", "prodrug", "active_metabolite"}
                or not isinstance(relation["smiles"], str)
                or not relation["smiles"]
                or not isinstance(relation["source_ref"], str)
                or not relation["source_ref"]
                for relation in relations
            )
        ):
            raise ValueError(f"Verified synonyms or related structures are invalid: {cid}")
    if structure_key is not None:
        seen: dict[str, str] = {}
        for cid, row in verified.items():
            key = structure_key(row["smiles"])
            if not key or key in seen:
                raise ValueError(
                    f"Verified structures are invalid or share a salt/stereo family: {cid}"
                )
            seen[key] = cid
            if any(not structure_key(relation["smiles"]) for relation in row["related_structures"]):
                raise ValueError(f"Verified related structure is invalid: {cid}")
        for compound in manifest["compounds"]:
            cid = compound["candidate_id"]
            if cid in verified and structure_key(compound["smiles"]) != structure_key(
                verified[cid]["smiles"]
            ):
                raise ValueError(f"Manifest structure differs from verified shortlist: {cid}")

    allocation = files["allocation"]
    roles = ("calibration", "final_test", "reserve")
    if not isinstance(allocation, dict) or set(allocation) != set(roles):
        raise ValueError("Allocation must contain calibration, final_test, and reserve")
    assigned = []
    for role in roles:
        ids = allocation[role]
        if (
            not isinstance(ids, list)
            or len(ids) != plan[f"{role}_n"]
            or not all(isinstance(cid, str) for cid in ids)
        ):
            raise ValueError(f"Allocation {role} count or IDs are invalid")
        assigned.extend(ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != verified.keys():
        raise ValueError("Allocation must partition the verified shortlist exactly once")

    flow = indexed(files["exclusion_flow"], "exclusion flow", plan["inventory_n"])
    if flow.keys() != inventory.keys():
        raise ValueError("Exclusion flow must cover the full inventory")
    for cid, row in flow.items():
        decision = "verified" if cid in verified else "excluded"
        if row.get("decision") != decision or (decision == "excluded" and not row.get("reason")):
            raise ValueError(f"Exclusion flow decision or reason is invalid: {cid}")

    primary_ids = {
        compound["candidate_id"] for compound in manifest["compounds"]
        if any(arm["primary_eligible"] for arm in compound["arms"])
    }
    if primary_ids != set(allocation["final_test"]):
        raise ValueError("Manifest primary cohort does not match frozen final-test allocation")
    if {compound["candidate_id"] for compound in manifest["compounds"]} != set(
        allocation["final_test"]
    ):
        raise ValueError("Manifest contains a compound outside the frozen final-test allocation")
    for compound in manifest["compounds"]:
        verify_parent_prediction(compound["smiles"], compound["candidate_id"])
    return plan
