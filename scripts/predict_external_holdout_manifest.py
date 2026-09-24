#!/usr/bin/env python3
"""Run one frozen, label-free external-holdout prediction pass.

The command accepts only the label-free manifest, enforces the public profile,
checks the manifest hash and frozen git SHA, and writes predictions without
reading observed Cmax. Run it inside the preregistered clean container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from sisyphus.validation.holdout_contract import (
    resolve_frozen_path,
    sha256_file,
    validate_payload,
    verify_audit_report,
    verify_development_residual_interval,
    verify_frozen_file,
    verify_source_plan,
    verify_training_membership,
)

ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.STDOUT, cwd=ROOT
    ).strip()


def _inventory_sha(values: tuple[tuple[str, str], ...]) -> str:
    return hashlib.sha256(
        json.dumps(dict(values), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_tree_sha256() -> str:
    paths = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).split(b"\0")
    digest = hashlib.sha256()
    for raw_path in paths:
        if not raw_path:
            continue
        path = ROOT / raw_path.decode()
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _verify_resource_root() -> None:
    from sisyphus.resources import get_resource_config

    if get_resource_config("public").root.resolve() != ROOT.resolve():
        raise ValueError("External holdout resources must come from the frozen checkout")


def _verify_runtime_settings(path: Path) -> None:
    """Ensure the hashed protocol settings describe the actual prediction path."""

    from sisyphus.engine import solver
    from sisyphus.pipeline.predict import (
        PRIMARY_OBSERVATION_NODE,
        PRIMARY_SIMULATION_HORIZON_H,
    )

    expected = {
        "profile": "external_holdout_v1_primary",
        "route": "oral",
        "dose_regimen": "single",
        "simulation_horizon_h": PRIMARY_SIMULATION_HORIZON_H,
        "observation_node": PRIMARY_OBSERVATION_NODE,
        "deterministic_solver": {
            "method": solver.DETERMINISTIC_SOLVER_METHOD,
            "rtol": solver.DETERMINISTIC_RTOL,
            "atol": solver.DETERMINISTIC_ATOL,
            "output_points": solver.DETERMINISTIC_OUTPUT_POINTS,
            "t_min_h": 0.0,
        },
        "monte_carlo_samples": 0,
        "measured_adme": False,
        "phenotypes": False,
        "strict": True,
    }
    if json.loads(path.read_text()) != expected:
        raise ValueError("Frozen solver settings do not match the runtime prediction path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--audit-report-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("SISYPHUS_PROFILE", "public") != "public":
        raise ValueError("External holdout execution requires SISYPHUS_PROFILE=public")

    actual_manifest_sha = _sha256(args.manifest)
    if actual_manifest_sha != args.manifest_sha256:
        raise ValueError("Manifest SHA256 does not match the frozen value")
    manifest = json.loads(args.manifest.read_text())
    validate_payload(manifest, "external_holdout_v1_manifest.schema.json")
    if manifest.get("labels_blinded") is not True:
        raise ValueError("Manifest must declare labels_blinded=true")
    verify_source_plan(args.manifest, manifest)

    actual_audit_sha = verify_audit_report(
        args.audit_report, args.audit_report_sha256, args.manifest
    )

    freeze = manifest["freeze"]
    if freeze["resource_profile"] != "public":
        raise ValueError("Frozen manifest resource profile must be public")
    head = _git(["rev-parse", "HEAD"])
    if head != freeze["git_sha"]:
        raise ValueError(f"Git SHA mismatch: manifest={freeze['git_sha']}, current={head}")
    if _git(["status", "--porcelain"]):
        raise ValueError("External holdout execution requires a clean worktree")
    source_tree_sha = _source_tree_sha256()
    if source_tree_sha != freeze["source_tree_sha256"]:
        raise ValueError("Tracked source-tree SHA256 does not match the manifest")
    dependency_sha = _sha256(ROOT / "requirements-lock.txt")
    if dependency_sha != freeze["dependency_lock_sha256"]:
        raise ValueError("Dependency-lock SHA256 does not match the manifest")
    training_membership_sha = verify_training_membership(ROOT, freeze)
    feature_schema_sha = verify_frozen_file(ROOT, freeze, "feature_schema")
    solver_settings_sha = verify_frozen_file(ROOT, freeze, "solver_settings")
    container_digest = os.environ.get("SISYPHUS_CONTAINER_DIGEST")
    if not container_digest:
        raise ValueError("SISYPHUS_CONTAINER_DIGEST must be set by the frozen container")
    if container_digest != freeze["container_digest"]:
        raise ValueError(
            f"Container digest mismatch: manifest={freeze['container_digest']}, "
            f"runtime={container_digest}"
        )

    _verify_resource_root()
    _verify_runtime_settings(resolve_frozen_path(ROOT, freeze["solver_settings_path"]))
    verify_development_residual_interval(ROOT)

    # Imports happen after profile/freeze checks so runtime resources cannot be
    # initialized under a different profile first.
    from sisyphus.pipeline.predict import predict
    from sisyphus.resources import artifact_provenance

    provenance = artifact_provenance("public")
    inventory_sha = _inventory_sha(provenance)
    if inventory_sha != freeze["artifact_inventory_sha256"]:
        raise ValueError(
            "Artifact inventory hash mismatch: "
            f"manifest={freeze['artifact_inventory_sha256']}, current={inventory_sha}"
        )

    rows: list[dict] = []
    for compound in manifest["compounds"]:
        for arm in compound["arms"]:
            result = predict(
                compound["smiles"],
                float(arm["dose_mg"]),
                route=arm["route"],
                n_mc_samples=0,
                strict=True,
            )
            cmax = result.cmax_prediction
            if cmax is None or result.ml_pk is None or result.execution_status != "ok":
                raise RuntimeError(
                    f"Incomplete strict prediction for {compound['candidate_id']}::{arm['arm_id']}"
                )
            tracks = dict(cmax.tracks)
            rows.append(
                {
                    "candidate_id": compound["candidate_id"],
                    "arm_id": arm["arm_id"],
                    "dose_mg": float(arm["dose_mg"]),
                    "route": arm["route"],
                    "primary_eligible": arm["primary_eligible"],
                    "meta_cmax_mg_l": cmax.cmax.mean,
                    "ml_cmax_mg_l": result.ml_pk.cmax.mean,
                    "engine_cmax_mg_l": tracks.get("engine"),
                    "clf_cmax_mg_l": tracks.get("clf"),
                    "vdss_cmax_mg_l": tracks.get("vdss"),
                    "meta_pi90_low_mg_l": (
                        cmax.residual_interval_90[0] if cmax.residual_interval_90 else None
                    ),
                    "meta_pi90_high_mg_l": (
                        cmax.residual_interval_90[1] if cmax.residual_interval_90 else None
                    ),
                    "interval_source": cmax.residual_interval_source,
                    "execution_status": result.execution_status,
                }
            )

    payload = {
        "cycle_id": manifest["cycle_id"],
        "manifest_sha256": actual_manifest_sha,
        "audit_report_sha256": actual_audit_sha,
        "git_sha": head,
        "source_tree_sha256": source_tree_sha,
        "dependency_lock_sha256": dependency_sha,
        "artifact_inventory_sha256": inventory_sha,
        "training_membership_sha256": training_membership_sha,
        "feature_schema_sha256": feature_schema_sha,
        "solver_settings_sha256": solver_settings_sha,
        "container_digest": container_digest,
        "artifact_provenance": dict(provenance),
        "rows": rows,
    }
    validate_payload(payload, "external_holdout_v1_predictions.schema.json")
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(rows)} blinded predictions to {args.out}")


if __name__ == "__main__":
    main()
