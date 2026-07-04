#!/usr/bin/env python3
"""Build the N50 exclusion inventory and audit a curated N50 by InChIKey-14.

The 2026Q2 curation used a name-only inventory over THREE sources (107-holdout,
MMPK, TDC hepatocyte). That missed (a) five further training corpora that feed
the Cmax pipeline (VDss, CLF, bioavailability, the two expanded-CLint sets) and
the DrugBank enrichment pool, and (b) every synonym / salt / stereo variant a
name string cannot see (rifampin vs rifampicin, torsemide vs torasemide,
paclitaxel protein-bound, ...). The retrospective InChIKey-14 audit found 21/50
of the 2026Q2 set inside hard training corpora and 47/50 inside DrugBank — the
set was not "never-touched" and the cycle was invalidated. See
docs/research/n50_2026q2_invalidation.md.

This tool keys exclusion on the **InChIKey-14 connectivity block** (stereo- and
salt-insensitive), which is what catches those variants. It ingests SMILES from
every shipped training/enrichment artifact and:

  build (default) -- write an IK14 -> [sources] inventory to
      data/reference/n50_exclusion_ik14.json, plus the legacy name list, and
      report per-source counts.

  --audit N50_FILE -- cross-check a curated N50 file's SMILES against the
      inventory and print a contamination report (hard training corpora vs the
      softer DrugBank-enrichment pool). Exits non-zero if any drug is in a hard
      corpus, so it can gate N50' curation in CI or a pre-freeze check.

Hard corpora (a hit = the drug's Cmax / CLint / F / VDss was in a model's
training set = real leakage): MMPK Cmax x3, CLF, bioavailability, expanded
CLint x2, VDss, TDC hepatocyte. DrugBank is reported separately: presence there
means the drug COULD have been an ADME-enrichment source (spec E4 is
conservative -- treat as seen), but it is not itself a fitted-target leak.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pathlib
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent
csv.field_size_limit(10**7)

# (relative path, SMILES column, name column or None, delimiter). Every artifact
# whose molecules were seen while fitting a track that feeds the Cmax pipeline.
HARD_SOURCES: list[tuple[str, str, str | None, str]] = [
    ("data/training/mmpk_expanded_full.csv", "canon_smiles", "name", ","),
    ("data/training/mmpk_expanded_v2.csv", "canon_smiles", "name", ","),
    ("data/training/mmpk_pbpk_features.csv", "smiles", "name", ","),
    ("data/training/clf_training.csv", "smiles", "name", ","),
    ("data/training/bioavailability_v1.csv", "smiles", "name", ","),
    ("data/training/clint_expanded_v2.csv", "canon_smiles", None, ","),
    ("data/training/clint_merged_v3_biogen.csv", "smiles", None, ","),
    ("data/training/vdss_v2_training.csv", "canonical_smiles", "name", ","),
]
# TDC hepatocyte is positional (col0 = ChEMBL id, col1 = SMILES, tab-delimited).
TDC_HEP = "data/training/clearance_hepatocyte_az.tab"
# DrugBank enrichment pool (soft E4); carries a precomputed inchikey_14 column.
DRUGBANK = "data/drugbank/drugs.csv"

EXCLUSION_OUT = "data/reference/n50_exclusion_ik14.json"


def ik14(smiles: str | None) -> str | None:
    """InChIKey-14 (connectivity block) for a SMILES, or None if unparseable."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key[:14] if key else None


def _ingest_hard(root: pathlib.Path) -> dict[str, set[str]]:
    """IK14 -> {"artifact::name", ...} across all hard training corpora."""
    hard: dict[str, set[str]] = {}

    def add(key: str | None, tag: str) -> None:
        if key:
            hard.setdefault(key, set()).add(tag)

    for rel, scol, ncol, delim in HARD_SOURCES:
        fp = root / rel
        if not fp.exists():
            logger.warning("  (skip missing %s)", rel)
            continue
        n = 0
        with fp.open() as f:
            for row in csv.DictReader(f, delimiter=delim):
                key = ik14(row.get(scol, ""))
                if key:
                    name = row.get(ncol, "?") if ncol else "?"
                    add(key, f"{fp.name}::{name}")
                    n += 1
        logger.info("  ingested %5d ik14 from %s", n, rel)

    tdc = root / TDC_HEP
    if tdc.exists():
        n = 0
        with tdc.open() as f:
            rr = csv.reader(f, delimiter="\t")
            next(rr, None)  # header
            for row in rr:
                if len(row) >= 2:
                    key = ik14(row[1])
                    if key:
                        add(key, f"{tdc.name}::{row[0]}")
                        n += 1
        logger.info("  ingested %5d ik14 from %s", n, TDC_HEP)
    return hard


def _ingest_drugbank(root: pathlib.Path) -> dict[str, str]:
    """IK14 -> drug name for the DrugBank enrichment pool (uses precomputed col)."""
    db: dict[str, str] = {}
    fp = root / DRUGBANK
    if not fp.exists():
        logger.warning("  (skip missing %s)", DRUGBANK)
        return db
    n = 0
    with fp.open() as f:
        for row in csv.DictReader(f):
            key = (row.get("inchikey_14") or "").strip()
            if not key:
                key = ik14(row.get("smiles") or row.get("canonical_smiles") or "")
            if key:
                db.setdefault(key, row.get("name", "?"))
                n += 1
    logger.info("  ingested %5d ik14 from %s", n, DRUGBANK)
    return db


def build(root: pathlib.Path) -> int:
    logger.info("Building N50 exclusion inventory (InChIKey-14 keyed)...")
    hard = _ingest_hard(root)
    db = _ingest_drugbank(root)

    inventory = {
        "description": (
            "N50 exclusion inventory keyed on InChIKey-14 (connectivity block, "
            "stereo/salt-insensitive). hard_corpora = fitted-target leakage; "
            "drugbank = softer E4 enrichment presence. Built by "
            "scripts/build_n50_exclusion.py."
        ),
        "hard_corpora": {k: sorted(v) for k, v in sorted(hard.items())},
        "drugbank": dict(sorted(db.items())),
        "counts": {"hard_ik14": len(hard), "drugbank_ik14": len(db)},
    }
    out = root / EXCLUSION_OUT
    out.write_text(json.dumps(inventory, indent=2))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    logger.info(
        "\nwrote %s  hard_ik14=%d  drugbank_ik14=%d  sha256=%s",
        EXCLUSION_OUT, len(hard), len(db), digest,
    )
    logger.info(
        "Gate a candidate for N50' with: it must be absent from BOTH maps by "
        "InChIKey-14 (genuinely-novel, not merely a name the old inventory missed)."
    )
    return 0


def audit(root: pathlib.Path, n50_path: pathlib.Path) -> int:
    """Cross-check a curated N50 file. Returns 1 if any hard-corpus hit."""
    logger.info("Ingesting corpora for audit of %s ...", n50_path)
    hard = _ingest_hard(root)
    db = _ingest_drugbank(root)

    drugs = json.loads(n50_path.read_text()).get("drugs", {})
    hard_hits: list[tuple[str, str, list[str]]] = []
    db_hits: list[tuple[str, str]] = []
    unparseable: list[str] = []

    for name, entry in sorted(drugs.items()):
        key = ik14(entry.get("smiles", ""))
        if key is None:
            unparseable.append(name)
            continue
        if key in hard:
            hard_hits.append((name, key, sorted(hard[key])))
        if key in db:
            db_hits.append((name, key))

    print("=" * 78)
    print(f"N50 InChIKey-14 EXCLUSION AUDIT — {n50_path.name}")
    print("=" * 78)
    print(f"drugs audited: {len(drugs)}   RDKit-unparseable: {len(unparseable)}")
    if unparseable:
        print(f"  ** unparseable SMILES: {unparseable}")

    print(f"\n--- HARD training-corpus hits (fitted-target leakage): {len(hard_hits)} ---")
    if not hard_hits:
        print("  NONE — clean of every ML/engine training corpus by IK14. ✓")
    for name, key, tags in hard_hits:
        print(f"  ** {name} ({key})")
        for tag in tags[:8]:
            print(f"       {tag}")

    print(f"\n--- DrugBank-enrichment presence (soft E4): {len(db_hits)} ---")
    for name, key in db_hits:
        print(f"  ~ {name} ({key})")

    print("\n--- VERDICT ---")
    if hard_hits:
        print(
            f"  FAIL: {len(hard_hits)}/{len(drugs)} drugs are in hard training "
            f"corpora. This N50 is contaminated and is NOT a valid never-touch "
            f"generalization instrument."
        )
        return 1
    print(
        f"  PASS on hard corpora. {len(db_hits)}/{len(drugs)} touch DrugBank "
        f"(spec E4 is conservative — review each before freeze)."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=pathlib.Path,
        metavar="N50_FILE",
        help="audit a curated N50 JSON for IK14 contamination (non-zero exit on "
        "any hard-corpus hit) instead of building the inventory",
    )
    args = parser.parse_args()
    if args.audit is not None:
        return audit(ROOT, args.audit)
    return build(ROOT)


if __name__ == "__main__":
    sys.exit(main())
