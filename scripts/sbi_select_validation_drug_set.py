#!/usr/bin/env python3
"""Select the ≥10 holdout validation drugs for Track A multi-drug SBI.

Protocol (docs/next_steps_plan.md Track A Pre-resolved Decisions):
  - 5 anchors (existing TDM benchmark continuity):
      morphine, clozapine, amantadine, ketorolac, rivaroxaban
  - ≥5 diversity additions from the 107-holdout, stratified by
    compound_type (neutral/acid/base/zwitter) and clearance mechanism
    (CYP-dominant, renal-dominant, transporter-dominant).

Output: data/sbi/validation_drug_set.json

SMILES + dose are pulled from data/training/mmpk_expanded_v2.csv (which
contains both holdout and non-holdout drugs with their Cmax observations).
Clearance mechanism is inferred from a small curated table of well-known
holdout drugs. The goal is *diversity* not completeness — any drug where
the mechanism is unknown is still eligible but will be tagged "unknown".
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
POOL_CSV = ROOT / "data" / "training" / "mmpk_expanded_v2.csv"
OUT_JSON = ROOT / "data" / "sbi" / "validation_drug_set.json"

ANCHORS = ("morphine", "clozapine", "amantadine", "ketorolac", "rivaroxaban")

# Curated clearance-mechanism labels for a subset of 107-holdout drugs.
# Sources: FDA labels, DrugBank clinical pharmacology sections.
CLEARANCE_MECHANISM = {
    "morphine": "ugt",
    "clozapine": "cyp1a2",
    "amantadine": "renal",
    "ketorolac": "renal_and_ugt",
    "rivaroxaban": "cyp3a4_transporter",
    "digoxin": "renal_pgp",
    "diclofenac": "cyp2c9",
    "losartan": "cyp2c9_prodrug",
    "pravastatin": "oatp1b1",
    "tamoxifen": "cyp3a4_cyp2d6",
    "sildenafil": "cyp3a4",
    "indomethacin": "cyp2c9",
    "phenytoin": "cyp2c9_narrow_ti",
    "apixaban": "cyp3a4_direct",
    "sirolimus": "cyp3a4",
    "posaconazole": "ugt_transporter",
    "paroxetine": "cyp2d6",
    "atovaquone": "biliary",
    "cabozantinib": "cyp3a4",
    "lenacapavir": "cyp3a4",
    "upadacitinib": "cyp3a4",
    "selegiline": "mao",
    "methylphenidate": "ces",
    "levofloxacin": "renal",
    "cimetidine": "renal",
    "pindolol": "renal",
    "ciprofloxacin": "renal",
    "famotidine": "renal",
    "metronidazole": "cyp",
    "mercaptopurine": "xo",
    "sumatriptan": "mao",
    "sonidegib": "cyp3a4",
    "dasatinib": "cyp3a4",
    "ruxolitinib": "cyp3a4",
    "darunavir ethanolate": "cyp3a4",
    "nilotinib": "cyp3a4",
    "leflunomide": "prodrug",
}

# Diversity additions: cover distinct clearance mechanisms not in anchors.
# This is the target set; we select at least 5 that successfully load.
DIVERSITY_CANDIDATES = (
    "diclofenac",       # acid, CYP2C9
    "digoxin",           # neutral, renal + Pgp
    "pravastatin",       # acid, OATP1B1 uptake
    "sildenafil",        # base, CYP3A4 heavy
    "phenytoin",         # neutral, CYP2C9 narrow TI
    "tamoxifen",         # base, CYP3A4/2D6 active metabolite
    "indomethacin",      # acid, CYP2C9 + renal
    "posaconazole",      # neutral, UGT + transporter
)


def load_holdout_and_pool() -> tuple[set[str], dict[str, dict]]:
    with open(HOLDOUT_JSON) as f:
        holdout = {h.lower().strip() for h in json.load(f)["holdout"]}
    pool: dict[str, dict] = {}
    with open(POOL_CSV) as f:
        for row in csv.DictReader(f):
            name = row["name"].lower().strip()
            if name not in holdout:
                continue
            smi = (row.get("canon_smiles") or "").strip()
            if not smi:
                continue
            try:
                dose = float(row.get("dose_mg") or 0)
            except ValueError:
                continue
            if dose <= 0 or dose > 5000:
                continue
            # Prefer first encountered row per drug
            if name not in pool:
                pool[name] = {"smiles": smi, "dose_mg": dose, "row": row}
    return holdout, pool


def enrich(name: str, info: dict) -> dict | None:
    from sisyphus.predict.chemistry import compute_profile
    try:
        p = compute_profile(info["smiles"])
    except Exception as exc:
        logging.warning("profile failed for %s: %s", name, exc)
        return None
    return {
        "name": name,
        "smiles": info["smiles"],
        "dose_mg": info["dose_mg"],
        "route": "oral",
        "compound_type": p.compound_type,
        "mw": float(p.mw),
        "logp": float(p.logp),
        "tpsa": float(p.tpsa),
        "hbd": int(p.hbd),
        "hba": int(p.hba),
        "rotatable_bonds": int(p.rotatable_bonds),
        "pka": float(p.pka) if p.pka is not None else None,
        "clearance_mechanism": CLEARANCE_MECHANISM.get(name, "unknown"),
    }


def main() -> None:
    print(f"[select-val] loading holdout + pool")
    holdout, pool = load_holdout_and_pool()
    print(f"[select-val] holdout size: {len(holdout)}, pool (holdout∩mmpk): {len(pool)}")

    selected: list[dict] = []

    # Anchors first
    print("[select-val] enriching anchors")
    for name in ANCHORS:
        if name not in pool:
            raise RuntimeError(
                f"anchor drug {name!r} is not in mmpk_expanded_v2.csv — "
                f"cannot proceed without SMILES+dose"
            )
        rec = enrich(name, pool[name])
        if rec is None:
            raise RuntimeError(f"anchor drug {name!r} failed profile computation")
        rec["role"] = "anchor"
        selected.append(rec)

    # Diversity additions
    print("[select-val] enriching diversity candidates")
    for name in DIVERSITY_CANDIDATES:
        if name not in pool:
            print(f"  skip {name}: not in pool")
            continue
        rec = enrich(name, pool[name])
        if rec is None:
            print(f"  skip {name}: profile failed")
            continue
        rec["role"] = "diversity"
        selected.append(rec)

    print(f"[select-val] selected {len(selected)} drugs total")
    if len(selected) < 10:
        raise RuntimeError(
            f"only {len(selected)} drugs selected, user required ≥10"
        )

    payload = {
        "source_holdout": str(HOLDOUT_JSON.relative_to(ROOT)),
        "source_pool": str(POOL_CSV.relative_to(ROOT)),
        "anchors": list(ANCHORS),
        "diversity_candidates": list(DIVERSITY_CANDIDATES),
        "min_required": 10,
        "drugs": selected,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[select-val] wrote {OUT_JSON}")

    print("\nFinal validation set:")
    for d in selected:
        print(
            f"  [{d['role']:>9s}] {d['name']:>30s}  {d['compound_type']:>11s}  "
            f"dose={d['dose_mg']:>6.1f}mg  mech={d['clearance_mechanism']}"
        )


if __name__ == "__main__":
    main()
