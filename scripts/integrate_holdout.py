#!/usr/bin/env python3
"""
Integration script — merges OSP observed data + curated PK data into clinical_pk_v2.json.
Run after expand_holdout.py (OSP) and agents (curated PK) have produced their outputs.

Inputs:
  data/reference/osp_observed.json          (from expand_holdout.py)
  data/reference/curated_pk_data.json       (from pk-researcher agent or manual)
  data/reference/fda_extraction_results.json (from fda-extractor agent, optional)
  data/reference/clinical_pk.json           (existing)
  data/reference/holdout.json               (existing split)

Outputs:
  data/reference/clinical_pk_v2.json        (merged)
  data/reference/holdout_v2.json            (updated split if needed)
"""

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_REF = ROOT / "data" / "reference"
DATA_DRUGBANK = ROOT / "data" / "drugbank"


def load_drugbank_lookup() -> dict[str, dict]:
    """name (lower) → {smiles, inchikey_14, mw}"""
    lookup = {}
    with open(DATA_DRUGBANK / "drugs.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().lower()
            smiles = row.get("canonical_smiles", row.get("smiles", "")).strip()
            ik14 = row.get("inchikey_14", "").strip()
            if not ik14 and row.get("inchikey", ""):
                ik14 = row["inchikey"].split("-")[0]
            mw = row.get("mw", "")
            if smiles and ik14:
                lookup[name] = {
                    "smiles": smiles,
                    "inchikey_14": ik14,
                    "mw": float(mw) if mw else None,
                }
    return lookup


def resolve_smiles(name: str, drugbank: dict) -> dict | None:
    """Resolve drug name to {smiles, inchikey_14}."""
    key = name.strip().lower()
    if key in drugbank:
        return drugbank[key]
    # Partial match
    for db_name, info in drugbank.items():
        if key in db_name or db_name in key:
            return info
    return None


def main():
    drugbank = load_drugbank_lookup()

    # Load existing data
    with open(DATA_REF / "holdout.json") as f:
        holdout_data = json.load(f)
    holdout_names = set(holdout_data["holdout"])
    train_names = set(holdout_data["train"])

    with open(DATA_REF / "clinical_pk.json") as f:
        clinical = json.load(f)
    drugs = dict(clinical["drugs"])  # mutable copy

    # Load MMPK training drug names
    mmpk_drugs = set()
    mmpk_path = ROOT / "data" / "training" / "mmpk_expanded_full.csv"
    with open(mmpk_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mmpk_drugs.add(row["name"].strip().lower())

    # Existing InChIKeys with Cmax
    existing_iks_with_cmax = set()
    for name, drug in drugs.items():
        if drug.get("pk_params", {}).get("cmax_mg_L"):
            info = resolve_smiles(name, drugbank)
            if info:
                existing_iks_with_cmax.add(info["inchikey_14"])

    # Count initial state
    initial_holdout_cmax = sum(
        1 for n in holdout_names
        if n in drugs and drugs[n].get("pk_params", {}).get("cmax_mg_L")
    )
    log.info(f"Initial holdout with Cmax: {initial_holdout_cmax}/100")

    # ── Source 1: OSP (improved dose selection) ──
    osp_path = DATA_REF / "osp_observed.json"
    osp_added = 0
    osp_updated = 0
    if osp_path.exists():
        with open(osp_path) as f:
            osp_data = json.load(f)
        log.info(f"OSP drugs: {len(osp_data)}")

        for obs in osp_data:
            name = obs["drug_name"].lower()
            cmax = obs["cmax_obs"]
            dose = obs["dose_mg"]
            smiles = obs["smiles"]
            ik14 = obs["inchikey_14"]

            # Skip if already has Cmax
            if name in drugs and drugs[name].get("pk_params", {}).get("cmax_mg_L"):
                continue

            # Existing drug without Cmax → update
            if name in drugs:
                drugs[name].setdefault("pk_params", {})["cmax_mg_L"] = cmax
                drugs[name]["dose_mg"] = dose
                if smiles:
                    drugs[name]["smiles"] = smiles
                drugs[name]["source"] = drugs[name].get("source", "") + f" + OSP"
                osp_updated += 1
                continue

            # Genuinely new drug — InChIKey dedup
            if ik14 in existing_iks_with_cmax:
                continue

            drugs[name] = {
                "name": name,
                "tier": "silver",
                "source": f"OSP observed data",
                "dose_mg": dose,
                "route": "oral",
                "pk_params": {"cmax_mg_L": cmax},
                "smiles": smiles,
            }
            existing_iks_with_cmax.add(ik14)
            osp_added += 1

        log.info(f"OSP: {osp_added} new drugs, {osp_updated} existing updated")

    # ── Source 2: Curated PK data ──
    curated_path = DATA_REF / "curated_pk_data.json"
    curated_added = 0
    curated_updated = 0
    if curated_path.exists():
        with open(curated_path) as f:
            curated_data = json.load(f)
        log.info(f"Curated drugs: {len(curated_data)}")

        for entry in curated_data:
            if entry.get("status") == "not_found":
                continue
            name = entry["drug_name"].lower()
            cmax = entry.get("cmax_mg_L")
            dose = entry.get("dose_mg")
            if not cmax or cmax <= 0:
                continue

            # Sanity check
            if cmax < 0.0001 or cmax > 500:
                log.warning(f"Curated {name}: Cmax={cmax} out of range, skipping")
                continue

            info = resolve_smiles(name, drugbank)
            if not info:
                log.warning(f"Curated {name}: no SMILES, skipping")
                continue

            smiles = info["smiles"]
            ik14 = info["inchikey_14"]

            # Skip if already has Cmax
            if name in drugs and drugs[name].get("pk_params", {}).get("cmax_mg_L"):
                continue

            # Existing drug without Cmax → update
            if name in drugs:
                drugs[name].setdefault("pk_params", {})["cmax_mg_L"] = cmax
                if dose:
                    drugs[name]["dose_mg"] = dose
                drugs[name]["smiles"] = smiles
                source_str = entry.get("source", "curated")
                drugs[name]["source"] = drugs[name].get("source", "") + f" + {source_str}"
                curated_updated += 1
                continue

            # Genuinely new drug
            if ik14 in existing_iks_with_cmax:
                continue

            source_str = entry.get("source", "curated reference")
            drugs[name] = {
                "name": name,
                "tier": "gold" if "FDA" in source_str else "silver",
                "source": source_str,
                "dose_mg": dose,
                "route": "oral",
                "pk_params": {"cmax_mg_L": cmax},
                "smiles": smiles,
            }
            existing_iks_with_cmax.add(ik14)
            curated_added += 1

        log.info(f"Curated: {curated_added} new drugs, {curated_updated} existing updated")

    # ── Source 2b: Manual PK curation ──
    manual_path = DATA_REF / "manual_pk_curation.json"
    manual_added = 0
    manual_updated = 0
    if manual_path.exists():
        with open(manual_path) as f:
            manual_data = json.load(f)
        log.info(f"Manual curation entries: {len(manual_data)}")

        for entry in manual_data:
            if entry.get("status") == "not_found":
                continue
            name = entry["drug_name"].lower()
            cmax = entry.get("cmax_mg_L")
            dose = entry.get("dose_mg")
            if not cmax or cmax <= 0:
                continue
            if cmax < 0.0001 or cmax > 500:
                log.warning(f"Manual {name}: Cmax={cmax} out of range, skipping")
                continue

            info = resolve_smiles(name, drugbank)
            if not info:
                log.warning(f"Manual {name}: no SMILES, skipping")
                continue

            smiles = info["smiles"]
            ik14 = info["inchikey_14"]

            if name in drugs and drugs[name].get("pk_params", {}).get("cmax_mg_L"):
                continue

            if name in drugs:
                drugs[name].setdefault("pk_params", {})["cmax_mg_L"] = cmax
                if dose:
                    drugs[name]["dose_mg"] = dose
                drugs[name]["smiles"] = smiles
                source_str = entry.get("source", "manual curation")
                drugs[name]["source"] = drugs[name].get("source", "") + f" + {source_str}"
                manual_updated += 1
                continue

            if ik14 in existing_iks_with_cmax:
                continue

            source_str = entry.get("source", "manual curation")
            drugs[name] = {
                "name": name,
                "tier": "gold" if "FDA" in source_str else "silver",
                "source": source_str,
                "dose_mg": dose,
                "route": "oral",
                "pk_params": {"cmax_mg_L": cmax},
                "smiles": smiles,
            }
            existing_iks_with_cmax.add(ik14)
            manual_added += 1

        log.info(f"Manual: {manual_added} new drugs, {manual_updated} existing updated")

    # ── Source 3: FDA extraction results (optional) ──
    fda_path = DATA_REF / "fda_extraction_results.json"
    fda_added = 0
    fda_updated = 0
    if fda_path.exists():
        with open(fda_path) as f:
            fda_data = json.load(f)
        log.info(f"FDA extraction results: {len(fda_data)}")

        for entry in fda_data:
            if entry.get("status") != "extracted":
                continue
            name = entry["drug_name"].lower()
            cmax = entry.get("cmax_mg_L")
            dose = entry.get("dose_mg")
            if not cmax or cmax <= 0:
                continue

            if cmax < 0.0001 or cmax > 500:
                continue

            info = resolve_smiles(name, drugbank)
            if not info:
                continue

            smiles = info["smiles"]
            ik14 = info["inchikey_14"]

            if name in drugs and drugs[name].get("pk_params", {}).get("cmax_mg_L"):
                continue

            if name in drugs:
                drugs[name].setdefault("pk_params", {})["cmax_mg_L"] = cmax
                if dose:
                    drugs[name]["dose_mg"] = dose
                drugs[name]["smiles"] = smiles
                drugs[name]["source"] = drugs[name].get("source", "") + " + FDA DailyMed"
                fda_updated += 1
                continue

            if ik14 in existing_iks_with_cmax:
                continue

            drugs[name] = {
                "name": name,
                "tier": "gold",
                "source": "FDA label via DailyMed",
                "dose_mg": dose,
                "route": "oral",
                "pk_params": {"cmax_mg_L": cmax},
                "smiles": smiles,
            }
            existing_iks_with_cmax.add(ik14)
            fda_added += 1

        log.info(f"FDA: {fda_added} new drugs, {fda_updated} existing updated")

    # ── Determine truly new drugs (added by our sources, not in original data) ──
    original_names = set(clinical["drugs"].keys())
    known_names = holdout_names | train_names
    newly_added = set(drugs.keys()) - original_names  # Only drugs we actually added
    new_not_in_split = set()
    for name in newly_added:
        if name not in known_names:
            pk = drugs[name].get("pk_params", {})
            if pk.get("cmax_mg_L"):
                if name in mmpk_drugs:
                    log.info(f"  {name} is in MMPK training → will add to holdout BUT exclude from MMPK training")
                new_not_in_split.add(name)

    # ── Write clinical_pk_v2.json ──
    final_holdout_cmax = sum(
        1 for n in holdout_names
        if n in drugs and drugs[n].get("pk_params", {}).get("cmax_mg_L")
    )

    total_with_cmax = sum(
        1 for d in drugs.values()
        if d.get("pk_params", {}).get("cmax_mg_L")
    )

    result = {
        "metadata": {
            "created": "2026-03-26",
            "n_drugs": len(drugs),
            "n_with_cmax": total_with_cmax,
            "holdout_with_cmax": final_holdout_cmax,
            "sources": {
                "original": len(clinical["drugs"]),
                "osp_new": osp_added,
                "osp_updated": osp_updated,
                "curated_new": curated_added,
                "curated_updated": curated_updated,
                "manual_new": manual_added,
                "manual_updated": manual_updated,
                "fda_new": fda_added,
                "fda_updated": fda_updated,
            },
        },
        "drugs": dict(sorted(drugs.items())),
    }

    with open(DATA_REF / "clinical_pk_v2.json", "w") as f:
        json.dump(result, f, indent=2)

    # ── Update holdout.json if needed ──
    if new_not_in_split:
        new_holdout = sorted(holdout_data["holdout"] + sorted(new_not_in_split))
        holdout_v2 = {
            "train": holdout_data["train"],
            "holdout": new_holdout,
            "metadata": {
                **holdout_data["metadata"],
                "n_holdout": len(new_holdout),
                "expansion_date": "2026-03-26",
                "expansion_note": "Added drugs from OSP/FDA/curated sources",
            },
        }
        with open(DATA_REF / "holdout_v2.json", "w") as f:
            json.dump(holdout_v2, f, indent=2)
        log.info(f"holdout_v2.json: {len(new_holdout)} holdout drugs")

    # ── Report ──
    log.info("\n" + "=" * 60)
    log.info("INTEGRATION REPORT")
    log.info("=" * 60)
    log.info(f"Holdout with Cmax: {initial_holdout_cmax} → {final_holdout_cmax}")
    log.info(f"Total drugs with Cmax: {total_with_cmax}")
    log.info(f"New drugs added to split: {len(new_not_in_split)}")
    if new_not_in_split:
        log.info(f"  {sorted(new_not_in_split)}")

    still_missing = sorted(
        n for n in holdout_names
        if n not in drugs or not drugs[n].get("pk_params", {}).get("cmax_mg_L")
    )
    log.info(f"Holdout STILL missing Cmax ({len(still_missing)}): {still_missing}")

    return final_holdout_cmax


if __name__ == "__main__":
    main()
