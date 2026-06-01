#!/usr/bin/env python3
"""Prospective-eligibility contamination gate (production-aware).

A drug is eligible for the prospective benchmark ONLY if it is genuinely
unseen by every corpus that feeds a SHIPPED production model or the engine's
reference anchoring. Membership in a non-production data file (an intermediate
or a model that the pipeline never loads) is reported but is NOT disqualifying.

This distinction matters: the 2026-05-31 audit found that the prospective set's
drugs appear in several local CSVs (mmpk_expanded_*, vdss_v2_training,
bioavailability_v1) whose models the pipeline does NOT load — so those are not
contamination. The genuine leaks are production-only:
  - vorasidenib  -> clinical_pk.json (gold reference)  [removed 2026-05-31]
  - aficamten    -> clf_training.csv  (xgboost_clf, CLF track)
  - gepotidacin  -> clf_training.csv  (xgboost_clf, CLF track)

PRODUCTION corpora (a hit here = INELIGIBLE):
  - data/reference/clinical_pk.json            (engine reference / anchoring)
  - data/training/clf_training.csv             (-> xgboost_clf.json, CLF track;
                                                 NO holdout/prospective exclusion)
  - data/reference/holdout.json                (train + holdout name lists)
  - data/training/mmpk_sisyphus_holdout_exclusions.json
  NOTE: the Cmax ML model trains on mmpk_clean.csv (Omega, ~1028 drugs, NOT in
  this repo, pre-2024) and the ADME/VDss models on TDC public sets (pre-2024);
  2024-2025 NMEs cannot be in those by construction, so the in-repo gate above
  is sufficient when combined with an FDA approval-date >= 2024 precondition.

NON-PRODUCTION corpora (hit = informational only, NOT disqualifying):
  mmpk_expanded_full/v2, vdss_v2_training, bioavailability_v1, clint_*,
  mmpk_pbpk_features  — their models (xgboost_vdss_v2, xgboost_bioavailability,
  ...) are not loaded by src/sisyphus, and the Cmax model uses mmpk_clean, not
  these.

Usage:
    python scripts/check_prospective_eligibility.py candidates.json [out.json]
        candidates.json = [{"name": "...", "smiles": "..."}, ...]
    python scripts/check_prospective_eligibility.py --selftest
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "data" / "training"
REFERENCE = ROOT / "data" / "reference"

# (csv filename, name column or None, smiles column or None, is_production)
_CSV_CORPORA = [
    ("clf_training.csv", "name", "smiles", True),          # -> xgboost_clf (CLF track)
    ("mmpk_expanded_full.csv", "name", "canon_smiles", False),
    ("mmpk_expanded_v2.csv", "name", "canon_smiles", False),
    ("clint_expanded_v2.csv", None, "canon_smiles", False),
    ("clint_merged_v3_biogen.csv", None, "smiles", False),
    ("vdss_v2_training.csv", "name", "canonical_smiles", False),  # xgboost_vdss_v2 (unused)
    ("bioavailability_v1.csv", "name", "smiles", False),  # xgboost_bioavailability (unused)
    ("mmpk_pbpk_features.csv", "name", "smiles", False),
]

# JSON corpora are all production (reference / holdout / exclusion lists).
_PRODUCTION_JSON = {"clinical_pk", "holdout.train", "holdout.holdout", "mmpk_exclusions"}


def _canon(smiles: str, isomeric: bool) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric) if mol else None


def _ik14(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        inchi = MolToInchi(mol)
        if not inchi:
            return None
        ik = InchiToInchiKey(inchi)
        return ik[:14] if ik else None
    except Exception:
        return None


def build_index() -> dict:
    names: dict[str, set[str]] = {}
    iso: dict[str, set[str]] = {}
    flat: dict[str, set[str]] = {}
    ik: dict[str, set[str]] = {}
    production: dict[str, bool] = {}

    def add(src: str, prod: bool, *, name: str | None = None, smiles: str | None = None) -> None:
        production[src] = prod
        if name and isinstance(name, str):
            names.setdefault(src, set()).add(name.strip().lower())
        if smiles and isinstance(smiles, str):
            ci, cf, k = _canon(smiles, True), _canon(smiles, False), _ik14(smiles)
            if ci:
                iso.setdefault(src, set()).add(ci)
            if cf:
                flat.setdefault(src, set()).add(cf)
            if k:
                ik.setdefault(src, set()).add(k)

    hj = json.loads((REFERENCE / "holdout.json").read_text())
    for n in hj.get("train", []):
        add("holdout.train", True, name=n)
    for n in hj.get("holdout", []):
        add("holdout.holdout", True, name=n)

    cpk = json.loads((REFERENCE / "clinical_pk.json").read_text())
    for n, rec in cpk.get("drugs", {}).items():
        add("clinical_pk", True, name=n, smiles=(rec or {}).get("smiles"))

    mex = json.loads((TRAINING / "mmpk_sisyphus_holdout_exclusions.json").read_text())
    for n, rec in mex.items():
        add("mmpk_exclusions", True, name=n, smiles=(rec or {}).get("smiles"))

    for fn, ncol, scol, prod in _CSV_CORPORA:
        p = TRAINING / fn
        if not p.exists():
            continue
        src = f"train:{fn}"
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                add(src, prod, name=row.get(ncol) if ncol else None,
                    smiles=row.get(scol) if scol else None)

    return {"names": names, "iso": iso, "flat": flat, "ik": ik, "production": production}


def check(name: str, smiles: str | None, idx: dict) -> tuple[bool, list[dict]]:
    """Return (eligible, hits). eligible=True iff no PRODUCTION corpus matches."""
    n = (name or "").strip().lower()
    ci = _canon(smiles, True) if smiles else None
    cf = _canon(smiles, False) if smiles else None
    k = _ik14(smiles) if smiles else None

    hits: list[dict] = []
    sources = set(idx["names"]) | set(idx["iso"]) | set(idx["flat"]) | set(idx["ik"])
    for src in sorted(sources):
        matched = []
        if n and n in idx["names"].get(src, set()):
            matched.append("name")
        if ci and ci in idx["iso"].get(src, set()):
            matched.append("smiles_iso")
        if cf and cf in idx["flat"].get(src, set()):
            matched.append("smiles_flat")
        if k and k in idx["ik"].get(src, set()):
            matched.append("ik14")
        if matched:
            prod = idx["production"].get(src, False)
            hits.append({"source": src, "match": matched, "production": prod})
    eligible = not any(h["production"] for h in hits)
    return eligible, hits


def _fmt(hits: list[dict], prod: bool) -> str:
    sel = [h for h in hits if h["production"] == prod]
    return ", ".join(f"{h['source']}({'+'.join(h['match'])})" for h in sel) or "—"


def _selftest(idx: dict) -> int:
    cases = [
        ("vorasidenib", "C[C@@H](Nc1nc(N[C@H](C)C(F)(F)F)nc(-c2cccc(Cl)n2)n1)C(F)(F)F", False),
        ("aficamten", "CCc1nc(-c2ccc3c(c2)CC[C@H]3NC(=O)c2cnn(C)c2)no1", False),
        ("gepotidacin", "O=c1ccc2ncc(=O)n3c2n1C[C@H]3CN1CCC(NCc2cc3c(cn2)OCCC3)CC1", False),
        ("resmetirom", "CC(C)c1cc(Oc2c(Cl)cc(-n3nc(C#N)c(=O)[nH]c3=O)cc2Cl)n[nH]c1=O", True),
        ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", False),
    ]
    ok = True
    for name, smi, expect in cases:
        eligible, hits = check(name, smi, idx)
        if eligible != expect:
            ok = False
        print(f"  [{'PASS' if eligible == expect else 'FAIL'}] {name:14s} eligible={eligible!s:5} "
              f"(exp {expect})  PROD: {_fmt(hits, True)}  | non-prod: {_fmt(hits, False)}")
    return 0 if ok else 1


def main() -> int:
    args = sys.argv[1:]
    print("Building corpus index...", file=sys.stderr)
    idx = build_index()
    if not args or args[0] == "--selftest":
        print("Self-test (eligible iff no PRODUCTION hit):")
        return _selftest(idx)

    candidates = json.loads(Path(args[0]).read_text())
    out = []
    n_elig = 0
    for c in candidates:
        eligible, hits = check(c.get("name", ""), c.get("smiles"), idx)
        n_elig += eligible
        print(f"{c.get('name',''):22s} eligible={eligible!s:5}  PROD: {_fmt(hits, True)}")
        out.append({**c, "eligible": eligible, "hits": hits})
    print(f"\n{n_elig}/{len(candidates)} eligible (production-clean / genuinely prospective)")
    if len(args) > 1:
        Path(args[1]).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
