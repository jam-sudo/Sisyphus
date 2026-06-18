#!/usr/bin/env python3
"""Build exclusion inventory for N50 curation.

Per docs/_internal/specs/2026-04-22-n50-secondary-holdout-design.md §3,
a drug is eligible for N50 only if NOT present in:
  E1: 107-drug primary holdout (data/reference/holdout.json::holdout[])
  E2: MMPK training corpus (data/training/mmpk_*.csv)
  E3: TDC CLint training (data/training/clearance_hepatocyte_az.tab)

Writes /tmp/n50_exclusion.txt (one lowercase drug name per line) and
reports its SHA-256 for inclusion in the N50 JSON provenance field.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    names: set[str] = set()

    # E1 + training
    with (ROOT / "data/reference/holdout.json").open() as f:
        ho = json.load(f)
    names |= {n.lower().strip() for n in ho.get("holdout", [])}
    names |= {n.lower().strip() for n in ho.get("train", [])}

    # E2: MMPK
    for p in [
        "data/training/mmpk_expanded_full.csv",
        "data/training/mmpk_expanded_v2.csv",
        "data/training/mmpk_pbpk_features.csv",
    ]:
        fp = ROOT / p
        if not fp.exists():
            continue
        with fp.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in ("drug", "name", "drug_name", "Drug", "Name"):
                    if key in row:
                        names.add(row[key].lower().strip())
                        break

    # E3: TDC CLint (tab-separated)
    tdc = ROOT / "data/training/clearance_hepatocyte_az.tab"
    if tdc.exists():
        with tdc.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                for key in ("Drug", "drug", "name", "Drug_ID", "ID"):
                    if key in row:
                        names.add(str(row[key]).lower().strip())
                        break

    # Strip empty
    names.discard("")

    out = pathlib.Path("/tmp/n50_exclusion.txt")
    with out.open("w") as f:
        for n in sorted(names):
            f.write(n + "\n")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out}  n={len(names)}  sha256={digest}")


if __name__ == "__main__":
    main()
