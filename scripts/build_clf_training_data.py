#!/usr/bin/env python3
"""Phase 0: Extract CL/F and Vd/F training data from MMPK.

CL/F = Dose(mg) / AUC(mg·h/L)
     = Dose(mg) / (AUC_ng_h_mL / 1000)
     = Dose(mg) × 1000 / AUC_ng_h_mL   [L/h]

CL/F (mL/min/kg) = CL/F (L/h) × 1000 / 60 / BW  [BW=70 kg]

Vd/F = CL/F(L/h) × t½(h) / 0.693  [L]
Vd/F (L/kg) = Vd/F(L) / BW  [BW=70 kg]

Unit conversion note:
  ng/mL = 10^-3 mg/L  (NOT 10^-6)
  1 ng = 10^-6 mg, 1 mL = 10^-3 L
  ng/mL = 10^-6 / 10^-3 mg/L = 10^-3 mg/L
  So: AUC(mg·h/L) = AUC(ng·h/mL) / 1000
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
BW_KG = 70.0  # reference body weight

# Sanity bounds for CL/F (mL/min/kg)
CLF_MIN = 0.1
CLF_MAX = 200.0

# Sanity bounds for Vd/F (L/kg)
VDF_MIN = 0.03
VDF_MAX = 50.0


def load_holdout_drugs() -> set[str]:
    """Load holdout drug names (lowercased) from holdout.json."""
    path = ROOT / "data" / "reference" / "holdout.json"
    with open(path) as f:
        data = json.load(f)
    holdout = {name.strip().lower() for name in data["holdout"]}
    logger.info("Holdout drugs loaded: %d", len(holdout))
    return holdout


def load_holdout_exclusions() -> set[str]:
    """Load MMPK holdout exclusion names (lowercased)."""
    path = ROOT / "data" / "training" / "mmpk_sisyphus_holdout_exclusions.json"
    with open(path) as f:
        data = json.load(f)
    names = {name.strip().lower() for name in data.keys()}
    logger.info("MMPK holdout exclusions loaded: %d", len(names))
    return names


def _inchikey14(smiles: str) -> str | None:
    """First 14 chars of the InChIKey (connectivity block), or None on failure."""
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        inchi = MolToInchi(mol)
        if not inchi:
            return None
        key = InchiToInchiKey(inchi)
        return key[:14] if key else None
    except Exception:
        return None


def load_holdout_inchikeys() -> set[str]:
    """Structural (InChIKey-14) keys for every holdout drug.

    The name/flag filters miss holdout compounds whose training-row name differs
    by spelling, salt form, or stereo descriptor (e.g. valaciclovir vs
    valacyclovir, darunavir vs darunavir ethanolate). InChIKey-14 matches on the
    connectivity block, closing that gap. SMILES come from clinical_pk.json via
    the reference loader.
    """
    from sisyphus.validation.reference import load_reference

    keys: set[str] = set()
    for r in load_reference():
        if r.in_holdout and r.smiles:
            k = _inchikey14(r.smiles)
            if k:
                keys.add(k)
    logger.info("Holdout InChIKey-14 keys loaded: %d", len(keys))
    return keys


def main() -> None:
    holdout_names = load_holdout_drugs()
    exclusion_names = load_holdout_exclusions()
    holdout_inchikeys = load_holdout_inchikeys()

    mmpk_path = ROOT / "data" / "training" / "mmpk_expanded_full.csv"
    logger.info("Loading MMPK data from %s", mmpk_path)

    # --- Pass 1: compute CL/F per entry ---
    entries: list[dict] = []
    n_total = 0
    n_no_auc = 0
    n_holdout_flag = 0
    n_holdout_name = 0
    n_holdout_excl = 0
    n_holdout_ik = 0
    n_no_smiles = 0
    n_zero_dose = 0
    n_clf_out_of_range = 0

    with open(mmpk_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            name = row["name"].strip()
            smiles = row["canon_smiles"].strip()
            name_lower = name.lower()

            # Skip holdout drugs (four keys: flag, name, exclusion list, structure)
            if row.get("in_holdout", "").strip().lower() == "true":
                n_holdout_flag += 1
                continue
            if name_lower in holdout_names:
                n_holdout_name += 1
                continue
            if name_lower in exclusion_names:
                n_holdout_excl += 1
                continue

            if not smiles:
                n_no_smiles += 1
                continue

            # Structural (InChIKey-14) holdout check — catches spelling / salt /
            # stereo variants the name filters miss (DE: CLF-track leak audit).
            if _inchikey14(smiles) in holdout_inchikeys:
                n_holdout_ik += 1
                continue

            dose_mg_str = row.get("dose_mg", "").strip()
            auc_str = row.get("auc_ng_h_ml", "").strip()

            if not auc_str:
                n_no_auc += 1
                continue

            try:
                dose_mg = float(dose_mg_str)
                auc_ng_h_ml = float(auc_str)
            except (ValueError, TypeError):
                n_no_auc += 1
                continue

            if dose_mg <= 0:
                n_zero_dose += 1
                continue
            if auc_ng_h_ml <= 0:
                n_no_auc += 1
                continue

            # --- CL/F calculation ---
            # AUC(mg·h/L) = AUC(ng·h/mL) / 1000
            auc_mg_h_L = auc_ng_h_ml / 1000.0

            # CL/F (L/h) = Dose(mg) / AUC(mg·h/L)
            clf_L_h = dose_mg / auc_mg_h_L

            # CL/F (mL/min/kg) = CL/F(L/h) × 1000 / 60 / BW
            clf_mL_min_kg = clf_L_h * 1000.0 / 60.0 / BW_KG

            # Sanity check
            if clf_mL_min_kg < CLF_MIN or clf_mL_min_kg > CLF_MAX:
                n_clf_out_of_range += 1
                # Still include but flag
                flag = "OUT_OF_RANGE"
            else:
                flag = ""

            # --- Vd/F calculation (if t½ available) ---
            thalf_str = row.get("thalf_h", "").strip()
            vdf_L_kg = None
            vdf_flag = ""
            if thalf_str:
                try:
                    thalf_h = float(thalf_str)
                    if thalf_h > 0:
                        # Vd/F (L) = CL/F(L/h) × t½(h) / 0.693
                        vdf_L = clf_L_h * thalf_h / 0.693
                        vdf_L_kg = vdf_L / BW_KG
                        if vdf_L_kg < VDF_MIN or vdf_L_kg > VDF_MAX:
                            vdf_flag = "OUT_OF_RANGE"
                except (ValueError, TypeError):
                    pass

            entries.append({
                "name": name,
                "smiles": smiles,
                "dose_mg": dose_mg,
                "auc_ng_h_ml": auc_ng_h_ml,
                "auc_mg_h_L": auc_mg_h_L,
                "clf_L_h": clf_L_h,
                "clf_mL_min_kg": clf_mL_min_kg,
                "log10_clf": np.log10(clf_mL_min_kg),
                "clf_flag": flag,
                "thalf_h": float(thalf_str) if thalf_str else None,
                "vdf_L_kg": vdf_L_kg,
                "log10_vdf": np.log10(vdf_L_kg) if vdf_L_kg and vdf_L_kg > 0 else None,
                "vdf_flag": vdf_flag,
            })

    logger.info("\n=== MMPK Entry-Level Statistics ===")
    logger.info("Total rows:              %d", n_total)
    logger.info("Holdout (in_holdout):    %d", n_holdout_flag)
    logger.info("Holdout (name match):    %d", n_holdout_name)
    logger.info("Holdout (InChIKey-14):   %d", n_holdout_ik)
    logger.info("Holdout (exclusion):     %d", n_holdout_excl)
    logger.info("No SMILES:               %d", n_no_smiles)
    logger.info("Zero/missing dose:       %d", n_zero_dose)
    logger.info("No/invalid AUC:          %d", n_no_auc)
    logger.info("CL/F out of range:       %d", n_clf_out_of_range)
    logger.info("Valid entries:           %d", len(entries))

    if not entries:
        logger.error("No valid entries! Aborting.")
        sys.exit(1)

    # --- Pass 2: aggregate per unique drug (by SMILES) ---
    from collections import defaultdict

    drug_entries: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        drug_entries[e["smiles"]].append(e)

    # Per-drug aggregation: geometric mean of CL/F across doses
    drugs: list[dict] = []
    n_nonlinear_flag = 0

    for smiles, elist in drug_entries.items():
        name = elist[0]["name"]
        clf_values = [e["clf_mL_min_kg"] for e in elist]
        clf_geomean = float(np.exp(np.mean(np.log(clf_values))))
        log10_clf = np.log10(clf_geomean)

        # Flag non-linear PK: CL/F varies >3-fold across doses
        clf_ratio = max(clf_values) / min(clf_values) if min(clf_values) > 0 else float("inf")
        nonlinear_flag = clf_ratio > 3.0
        if nonlinear_flag:
            n_nonlinear_flag += 1

        # Vd/F: geometric mean of non-None values
        vdf_values = [e["vdf_L_kg"] for e in elist if e["vdf_L_kg"] is not None and e["vdf_L_kg"] > 0]
        if vdf_values:
            vdf_geomean = float(np.exp(np.mean(np.log(vdf_values))))
            log10_vdf = np.log10(vdf_geomean)
        else:
            vdf_geomean = None
            log10_vdf = None

        # Check if any entry had out-of-range CL/F
        any_clf_oor = any(e["clf_flag"] == "OUT_OF_RANGE" for e in elist)
        any_vdf_oor = any(e["vdf_flag"] == "OUT_OF_RANGE" for e in elist)

        drugs.append({
            "name": name,
            "smiles": smiles,
            "n_entries": len(elist),
            "clf_mL_min_kg": clf_geomean,
            "log10_clf": log10_clf,
            "clf_ratio": clf_ratio,
            "clf_nonlinear": nonlinear_flag,
            "clf_oor": any_clf_oor,
            "vdf_L_kg": vdf_geomean,
            "log10_vdf": log10_vdf,
            "vdf_oor": any_vdf_oor,
            "has_vdf": vdf_geomean is not None,
        })

    # Count stats
    n_drugs = len(drugs)
    n_with_vdf = sum(1 for d in drugs if d["has_vdf"])
    n_clf_oor_drugs = sum(1 for d in drugs if d["clf_oor"])
    n_vdf_oor_drugs = sum(1 for d in drugs if d["vdf_oor"])

    # Subset: in-range CL/F only
    clf_in_range = [d for d in drugs if not d["clf_oor"]]
    vdf_in_range = [d for d in drugs if d["has_vdf"] and not d["vdf_oor"]]

    logger.info("\n=== Per-Drug Statistics ===")
    logger.info("Unique drugs (by SMILES): %d", n_drugs)
    logger.info("  with Vd/F:              %d", n_with_vdf)
    logger.info("  CL/F out of range:      %d (%.1f%%)", n_clf_oor_drugs, 100 * n_clf_oor_drugs / n_drugs)
    logger.info("  Non-linear PK flag:     %d (%.1f%%)", n_nonlinear_flag, 100 * n_nonlinear_flag / n_drugs)
    logger.info("  CL/F in range:          %d", len(clf_in_range))
    logger.info("  Vd/F in range:          %d", len(vdf_in_range))

    # Distribution summary
    clf_arr = np.array([d["log10_clf"] for d in drugs])
    logger.info("\n=== CL/F Distribution (all drugs, log10 mL/min/kg) ===")
    logger.info("  N:       %d", len(clf_arr))
    logger.info("  Mean:    %.3f (= %.1f mL/min/kg)", np.mean(clf_arr), 10**np.mean(clf_arr))
    logger.info("  Median:  %.3f (= %.1f mL/min/kg)", np.median(clf_arr), 10**np.median(clf_arr))
    logger.info("  Std:     %.3f", np.std(clf_arr))
    logger.info("  Min:     %.3f (= %.2f mL/min/kg)", np.min(clf_arr), 10**np.min(clf_arr))
    logger.info("  Max:     %.3f (= %.1f mL/min/kg)", np.max(clf_arr), 10**np.max(clf_arr))
    logger.info("  P5/P95:  %.3f / %.3f", np.percentile(clf_arr, 5), np.percentile(clf_arr, 95))

    if n_with_vdf > 0:
        vdf_arr = np.array([d["log10_vdf"] for d in drugs if d["log10_vdf"] is not None])
        logger.info("\n=== Vd/F Distribution (log10 L/kg) ===")
        logger.info("  N:       %d", len(vdf_arr))
        logger.info("  Mean:    %.3f (= %.1f L/kg)", np.mean(vdf_arr), 10**np.mean(vdf_arr))
        logger.info("  Median:  %.3f (= %.1f L/kg)", np.median(vdf_arr), 10**np.median(vdf_arr))
        logger.info("  Std:     %.3f", np.std(vdf_arr))
        logger.info("  Min:     %.3f (= %.2f L/kg)", np.min(vdf_arr), 10**np.min(vdf_arr))
        logger.info("  Max:     %.3f (= %.1f L/kg)", np.max(vdf_arr), 10**np.max(vdf_arr))

    # --- Gate check ---
    logger.info("\n=== GATE CHECK ===")
    if n_drugs >= 150:
        logger.info("PASS: %d unique drugs >= 150 threshold", n_drugs)
    else:
        logger.error("FAIL: %d unique drugs < 150 threshold", n_drugs)
        logger.error("Consider PKSmart (Lombardo, 399 drugs IV CL) as supplement")
        sys.exit(1)

    if n_with_vdf >= 150:
        logger.info("PASS: %d drugs with Vd/F >= 150 threshold", n_with_vdf)
    else:
        logger.warning("WARNING: %d drugs with Vd/F < 150 threshold", n_with_vdf)
        logger.warning("Fallback: use engine Vd for Phase 1b")

    # --- Save training data ---
    out_dir = ROOT / "data" / "training"
    clf_path = out_dir / "clf_training.csv"
    logger.info("\nSaving CL/F training data to %s", clf_path)

    with open(clf_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "smiles", "n_entries", "clf_mL_min_kg", "log10_clf",
            "clf_ratio", "clf_nonlinear", "clf_oor",
            "vdf_L_kg", "log10_vdf", "vdf_oor",
        ])
        writer.writeheader()
        for d in drugs:
            writer.writerow({
                "name": d["name"],
                "smiles": d["smiles"],
                "n_entries": d["n_entries"],
                "clf_mL_min_kg": f"{d['clf_mL_min_kg']:.4f}",
                "log10_clf": f"{d['log10_clf']:.4f}",
                "clf_ratio": f"{d['clf_ratio']:.2f}",
                "clf_nonlinear": d["clf_nonlinear"],
                "clf_oor": d["clf_oor"],
                "vdf_L_kg": f"{d['vdf_L_kg']:.4f}" if d["vdf_L_kg"] else "",
                "log10_vdf": f"{d['log10_vdf']:.4f}" if d["log10_vdf"] else "",
                "vdf_oor": d["vdf_oor"],
            })

    logger.info("Done. %d drugs written.", len(drugs))

    # --- Spot checks ---
    logger.info("\n=== Spot Checks (first 10 drugs) ===")
    logger.info("%-20s %8s %8s %8s %8s", "Drug", "Dose(mg)", "AUC(ng)", "CL/F", "log10")
    for e in entries[:10]:
        logger.info(
            "%-20s %8.1f %8.1f %8.2f %8.3f",
            e["name"][:20], e["dose_mg"], e["auc_ng_h_ml"],
            e["clf_mL_min_kg"], e["log10_clf"],
        )

    # Known drug sanity check: midazolam CL/F ≈ 6.4 mL/min/kg (literature)
    mida = [d for d in drugs if "midazolam" in d["name"].lower()]
    if mida:
        logger.info("\nMidazolam CL/F: %.2f mL/min/kg (literature ≈ 6-8)", mida[0]["clf_mL_min_kg"])
    # Metformin CL/F ≈ 8.0 mL/min/kg (renal dominant)
    metf = [d for d in drugs if "metformin" in d["name"].lower()]
    if metf:
        logger.info("Metformin CL/F: %.2f mL/min/kg (literature ≈ 7-9)", metf[0]["clf_mL_min_kg"])
    # Warfarin CL/F ≈ 0.05 mL/min/kg (very low clearance)
    warf = [d for d in drugs if "warfarin" in d["name"].lower()]
    if warf:
        logger.info("Warfarin CL/F: %.2f mL/min/kg (literature ≈ 0.03-0.06)", warf[0]["clf_mL_min_kg"])


if __name__ == "__main__":
    main()
