#!/usr/bin/env python3
"""Audit a label-free external-holdout manifest before model freeze.

The command never reads observed Cmax. It rejects structure/name collisions with
fitted corpora, runtime clinical registries, prior validation data, and the
development benchmark, then emits a hashable freeze report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import yaml

from sisyphus.validation.holdout_contract import (
    resolve_frozen_path,
    sha256_file,
    validate_payload,
    validate_source_quotas,
    verify_training_membership,
)

ROOT = Path(__file__).resolve().parent.parent


def _load_exclusion_module():
    path = ROOT / "scripts" / "build_n50_exclusion.py"
    spec = importlib.util.spec_from_file_location("sisyphus_exclusion", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load exclusion helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXCLUSION = _load_exclusion_module()


def _norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _walk_json(value, source: str, structures: dict[str, set[str]], names: dict[str, set[str]]):
    if isinstance(value, dict):
        for key in ("name", "drug_name", "compound", "drug"):
            name = value.get(key)
            if isinstance(name, str) and name.strip():
                names.setdefault(_norm_name(name), set()).add(source)
        for key in ("smiles", "canonical_smiles", "canon_smiles"):
            smiles = value.get(key)
            if isinstance(smiles, str):
                ik = EXCLUSION.ik14(smiles)
                if ik:
                    structures.setdefault(ik, set()).add(source)
        for child in value.values():
            _walk_json(child, source, structures, names)
    elif isinstance(value, list):
        for child in value:
            _walk_json(child, source, structures, names)


def _repository_exclusions(
    manifest_path: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    structures = EXCLUSION._ingest_hard(ROOT)
    names: dict[str, set[str]] = {}
    for rel, _, name_col, delimiter in EXCLUSION.HARD_SOURCES:
        if name_col is None:
            continue
        path = ROOT / rel
        if not path.exists():
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                name = row.get(name_col)
                if name:
                    names.setdefault(_norm_name(name), set()).add(rel)

    paths = sorted((ROOT / "data/reference").glob("*.json")) + [
        *sorted((ROOT / "data/validation").glob("*.json")),
        *sorted((ROOT / "data/enzymes").glob("*.json")),
        *sorted((ROOT / "data/transporters").glob("*.json")),
        *sorted((ROOT / "data/sbi").glob("*.json")),
        *sorted((ROOT / "data/training").glob("*.json")),
    ]
    for path in paths:
        if (
            not path.exists()
            or path.resolve() == manifest_path.resolve()
            or path.name.endswith(".schema.json")
            or path.name == "n50_exclusion_ik14.json"
        ):
            continue
        try:
            value = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        _walk_json(value, str(path.relative_to(ROOT)), structures, names)
        if path.name == "holdout.json" and isinstance(value, dict):
            for split in ("train", "holdout"):
                for name in value.get(split, []):
                    if isinstance(name, str):
                        names.setdefault(_norm_name(name), set()).add(
                            f"data/reference/holdout.json::{split}"
                        )
    for path in sorted((ROOT / "data/compounds").glob("*.yaml")):
        try:
            value = yaml.safe_load(path.read_text())
        except (yaml.YAMLError, UnicodeDecodeError):
            continue
        _walk_json(value, str(path.relative_to(ROOT)), structures, names)
    return structures, names


def audit(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    validate_payload(manifest, "external_holdout_v1_manifest.schema.json")
    source_plan_path = resolve_frozen_path(
        manifest_path.parent, str(manifest["source_plan_path"])
    )
    source_plan = json.loads(source_plan_path.read_text())
    validate_payload(source_plan, "external_holdout_v1_source_plan.schema.json")
    if sha256_file(source_plan_path) != manifest["source_plan_sha256"]:
        raise ValueError("source_plan_sha256 does not match source_plan_path")
    if source_plan["cycle_id"] != manifest["cycle_id"]:
        raise ValueError("Source plan cycle_id does not match manifest")
    if source_plan["final_test_n"] != manifest["n_target"]:
        raise ValueError("Source plan final_test_n does not match manifest n_target")
    compounds = manifest.get("compounds")
    if not isinstance(compounds, list):
        raise ValueError("manifest.compounds must be a list")
    if manifest.get("labels_blinded") is not True:
        raise ValueError("manifest.labels_blinded must be true")

    verify_training_membership(
        ROOT,
        manifest["freeze"],
        {rel for rel, *_ in EXCLUSION.HARD_SOURCES} | {EXCLUSION.TDC_HEP},
    )

    structures, names = _repository_exclusions(manifest_path)
    ids: set[str] = set()
    hits: list[dict] = []
    unparseable: list[str] = []
    duplicate_ids: list[str] = []
    duplicate_arm_ids: list[str] = []
    duplicate_structures: dict[str, list[str]] = {}
    candidate_iks: dict[str, list[str]] = {}
    arm_ids: set[str] = set()
    primary_compounds = 0

    for row in compounds:
        cid = str(row.get("candidate_id", ""))
        name = str(row.get("name", ""))
        if cid in ids:
            duplicate_ids.append(cid)
        ids.add(cid)
        ik = EXCLUSION.ik14(row.get("smiles"))
        if not ik:
            unparseable.append(cid)
            continue
        candidate_iks.setdefault(ik, []).append(cid)
        arms = row.get("arms")
        if not isinstance(arms, list) or not arms:
            unparseable.append(f"{cid}:missing_arms")
            continue
        if any(arm.get("primary_eligible") is True for arm in arms):
            primary_compounds += 1
        for arm in arms:
            arm_id = str(arm.get("arm_id", ""))
            joined = f"{cid}::{arm_id}"
            if not arm_id or joined in arm_ids:
                duplicate_arm_ids.append(joined)
            arm_ids.add(joined)
        reasons = sorted(structures.get(ik, set()) | names.get(_norm_name(name), set()))
        if reasons:
            hits.append({"candidate_id": cid, "name": name, "ik14": ik, "sources": reasons})

    duplicate_structures = {
        ik: cids for ik, cids in candidate_iks.items() if len(cids) > 1
    }
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    exclusion_union = {
        "structures": {k: sorted(v) for k, v in sorted(structures.items())},
        "names": {k: sorted(v) for k, v in sorted(names.items())},
    }
    exclusion_union_sha = hashlib.sha256(
        json.dumps(exclusion_union, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    target_mismatch = primary_compounds != manifest.get("n_target")
    exclusion_hash_mismatch = (
        manifest.get("exclusion_union_sha256") != exclusion_union_sha
    )
    source_quota_error = None
    source_quota_metrics: dict[str, float] = {}
    try:
        source_quota_metrics = validate_source_quotas(manifest)
    except ValueError as exc:
        source_quota_error = str(exc)
    report = {
        "protocol": "external_holdout_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "n_candidates": len(compounds),
        "n_primary_compounds": primary_compounds,
        "n_target": manifest.get("n_target"),
        "target_mismatch": target_mismatch,
        "exclusion_union_sha256": exclusion_union_sha,
        "exclusion_hash_mismatch": exclusion_hash_mismatch,
        "source_quota_metrics": source_quota_metrics,
        "source_quota_error": source_quota_error,
        "hard_collision_count": len(hits),
        "hard_collisions": hits,
        "unparseable": unparseable,
        "duplicate_ids": duplicate_ids,
        "duplicate_arm_ids": duplicate_arm_ids,
        "duplicate_structures": duplicate_structures,
        "pass": not (
            hits
            or unparseable
            or duplicate_ids
            or duplicate_arm_ids
            or duplicate_structures
            or target_mismatch
            or exclusion_hash_mismatch
            or source_quota_error
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.manifest.resolve())
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
