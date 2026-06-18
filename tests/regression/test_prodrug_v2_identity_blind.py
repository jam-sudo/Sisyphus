"""Regression: engine identity-blind invariant -- random tag rename -> identical result.

Replace SPR/CES1/CES2/ALPI with random strings everywhere
(physiology YAML + registry JSON); numerical Cmax must be byte-identical.

Verifies invariant 1 ("Engine is identity-blind") against
v2 prodrug enzyme infrastructure.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from sisyphus.pipeline.predict import predict


def _rename_in_yaml(yaml_text: str, mapping: dict[str, str]) -> str:
    data = yaml.safe_load(yaml_text)
    for node in data.get("nodes", []):
        if "enzymes" in node:
            new_enzymes = {}
            for tag, val in node["enzymes"].items():
                new_tag = mapping.get(tag, tag)
                new_enzymes[new_tag] = val
            node["enzymes"] = new_enzymes
    return yaml.safe_dump(data, sort_keys=False)


def _rename_in_registry(reg: dict, mapping: dict[str, str]) -> dict:
    out = {}
    for smi, entry in reg.items():
        new_entry = dict(entry)
        if "enzyme_affinity_for_conversion" in new_entry:
            new_aff = {}
            for tag, val in new_entry["enzyme_affinity_for_conversion"].items():
                new_aff[mapping.get(tag, tag)] = val
            new_entry["enzyme_affinity_for_conversion"] = new_aff
        out[smi] = new_entry
    return out


def test_random_rename_invariant(tmp_path, monkeypatch):
    """Replace SPR/CES1/CES2/ALPI -> random tags. Cmax must be byte-identical."""
    smiles = "C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1"  # sepiapterin
    dose_mg = 4200.0

    repo_root = Path(__file__).resolve().parent.parent.parent

    # ---- Baseline: original physiology + registry ----
    result_orig = predict(smiles, dose_mg=dose_mg, route="oral", n_mc_samples=0)
    cmax_orig = result_orig.pk.cmax.mean

    rename = {"SPR": "Z1Q9K", "CES1": "Q4M2X", "CES2": "P5N7T", "ALPI": "K8B3Y"}

    # ---- Rename inputs in tmp_path ----
    src_yaml_path = repo_root / "data" / "physiology" / "reference_man.yaml"
    src_yaml = src_yaml_path.read_text()
    renamed_yaml = _rename_in_yaml(src_yaml, rename)
    new_phys_dir = tmp_path / "physiology"
    new_phys_dir.mkdir()
    (new_phys_dir / "reference_man.yaml").write_text(renamed_yaml)

    # Copy any other files referenced by the YAML (e.g. correlation registry)
    # that might be loaded by build_from_yaml relative to physiology dir.
    src_phys_dir = src_yaml_path.parent
    for sibling in src_phys_dir.iterdir():
        if sibling.name == "reference_man.yaml":
            continue
        if sibling.is_file():
            (new_phys_dir / sibling.name).write_text(sibling.read_text())

    src_reg_path = repo_root / "data" / "sbi" / "prodrug_activation_registry.json"
    src_reg = json.loads(src_reg_path.read_text())
    renamed_reg = _rename_in_registry(src_reg, rename)
    new_reg_path = tmp_path / "registry.json"
    new_reg_path.write_text(json.dumps(renamed_reg))

    # ---- Monkeypatch: redirect predict + registry to renamed copies ----
    import sisyphus.pipeline.predict as pp_mod
    import sisyphus.predict.registry as reg_mod

    monkeypatch.setattr(pp_mod, "_PHYSIOLOGY_DIR", new_phys_dir, raising=False)
    monkeypatch.setattr(reg_mod, "_DEFAULT_REGISTRY_PATH", new_reg_path, raising=False)
    reg_mod._load_registry_cached.cache_clear()

    try:
        result_renamed = predict(smiles, dose_mg=dose_mg, route="oral", n_mc_samples=0)
        cmax_renamed = result_renamed.pk.cmax.mean
    finally:
        # Restore registry cache to original path on exit
        reg_mod._load_registry_cached.cache_clear()

    assert cmax_renamed == cmax_orig, (
        f"Identity-blind violated: orig={cmax_orig}, renamed={cmax_renamed}"
    )
