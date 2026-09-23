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

This tool strips counterions to the largest organic fragment, then keys exclusion
on the **InChIKey-14 connectivity block** (stereo-insensitive), which catches salt
and stereochemical variants. It ingests SMILES from
every shipped training/enrichment artifact and:

  build (default) -- write an IK14 -> [sources] inventory to
      data/reference/n50_exclusion_ik14.json, plus the legacy name list, and
      report per-source counts.

  --audit N50_FILE -- cross-check a curated N50 file's SMILES against the
      inventory and print a contamination report (repository training corpora
      and conservative DrugBank membership). Exits non-zero for missing
      sources, unparseable candidates, or either corpus hit.

Hard corpora (a hit = disqualifying, including conservative pre-exclusion
sources whose exact fitted rows are not proven): Omega MMPK Cmax source,
MMPK Cmax x3, CLF, bioavailability, expanded CLint x2, VDss, TDC hepatocyte.
DrugBank is reported separately because its catalog membership remains a
conservative E4 exclusion for the never-seen N50 design. The historical fup v2
artifact also used DrugBank protein-binding targets; the current public-only
fup artifact does not. Any DrugBank identity still disqualifies N50 under E4.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pathlib
import sys

from rdkit import RDLogger

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sisyphus.validation.identity import (  # noqa: E402
    _largest_organic_fragment as _largest_organic_fragment,
)
from sisyphus.validation.identity import (  # noqa: E402
    ik14,
)

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

csv.field_size_limit(10**7)

# (relative path, SMILES column, name column or None, delimiter). Every artifact
# whose molecules were seen while fitting a track that feeds the Cmax pipeline.
HARD_SOURCES: list[tuple[str, str, str | None, str]] = [
    # Raw ADME memberships used by the production fup and Peff artifacts.
    ("data/ppbr_az.tab", "Drug", "Drug_ID", "\t"),
    ("data/training/fup_tdc_public_clean.csv", "smiles", "name", ","),
    ("data/caco2_wang.tab", "Drug", "Drug_ID", "\t"),
    ("data/training/peff_tdc_public_clean.csv", "canonical_smiles", "drug_id", ","),
    # Upstream Omega source for the shipped Cmax model; includes pre-exclusion rows.
    ("data/training/omega_mmpk_clean.csv", "smiles", "name", ","),
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
# DrugBank identity superset retained for the conservative N50 E4 rule.
DRUGBANK = "data/drugbank/drugs.csv"

EXCLUSION_OUT = "data/reference/n50_exclusion_ik14.json"


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
    """IK14 -> drug name for the conservative DrugBank identity superset."""
    db: dict[str, str] = {}
    fp = root / DRUGBANK
    if not fp.exists():
        logger.warning("  (skip missing %s)", DRUGBANK)
        return db
    n = 0
    with fp.open() as f:
        for row in csv.DictReader(f):
            # Keep the published key and the salt-stripped active fragment:
            # precomputed DrugBank keys can include counterions.
            keys = {
                (row.get("inchikey_14") or "").strip(),
                ik14(row.get("canonical_smiles") or row.get("smiles") or ""),
            }
            for key in keys:
                if key:
                    db.setdefault(key, row.get("name", "?"))
            if any(keys):
                n += 1
    logger.info("  ingested %5d ik14 from %s", n, DRUGBANK)
    return db


def _require_sources(root: pathlib.Path) -> None:
    required = [rel for rel, *_ in HARD_SOURCES] + [TDC_HEP, DRUGBANK]
    missing = [rel for rel in required if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"N50 exclusion sources missing: {', '.join(missing)}")


def build(root: pathlib.Path) -> int:
    _require_sources(root)
    logger.info("Building N50 exclusion inventory (InChIKey-14 keyed)...")
    hard = _ingest_hard(root)
    db = _ingest_drugbank(root)

    inventory = {
        "description": (
            "N50 exclusion inventory keyed on InChIKey-14 (connectivity block, "
            "stereo/salt-insensitive). hard_corpora = training-source overlap; "
            "drugbank = potential fup fitted-target overlap (also excluding). Built by "
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
    """Cross-check a curated N50 file. Returns 1 for hits or bad structures."""
    _require_sources(root)
    logger.info("Ingesting corpora for audit of %s ...", n50_path)
    hard = _ingest_hard(root)
    db = _ingest_drugbank(root)

    drugs = json.loads(n50_path.read_text()).get("drugs", {})
    if not isinstance(drugs, dict) or not drugs:
        raise ValueError("N50 audit requires a non-empty drugs object")
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

    print(f"\n--- Repository training-source hits: {len(hard_hits)} ---")
    if not hard_hits and not unparseable:
        print("  NONE — no repository source hits by IK14.")
    elif not hard_hits:
        print("  No hits among parsed structures; unparseable candidates remain unresolved.")
    for name, key, tags in hard_hits:
        print(f"  ** {name} ({key})")
        for tag in tags[:8]:
            print(f"       {tag}")

    print(f"\n--- DrugBank potential fup-training hits: {len(db_hits)} ---")
    for name, key in db_hits:
        print(f"  ~ {name} ({key})")

    print("\n--- VERDICT ---")
    if hard_hits or db_hits or unparseable:
        print(
            f"  FAIL: {len(hard_hits)} repository hits, {len(db_hits)} DrugBank hits, "
            f"and {len(unparseable)} "
            f"unparseable structures among {len(drugs)} drugs. This N50 is NOT "
            f"a valid never-touch generalization instrument."
        )
        return 1
    print("  PASS: no repository or DrugBank identity hits.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=pathlib.Path,
        metavar="N50_FILE",
        help="audit a curated N50 JSON for IK14 contamination (non-zero exit on "
        "any repository or DrugBank hit) instead of building the inventory",
    )
    args = parser.parse_args()
    if args.audit is not None:
        return audit(ROOT, args.audit)
    return build(ROOT)


if __name__ == "__main__":
    sys.exit(main())
