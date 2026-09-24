"""Invariant #5 guard: no holdout drug may enter the ML Cmax (MMPK) training set.

A holdout drug leaks into the MMPK corpus only if it bypasses the holdout flag,
name, and InChIKey-14 filters in ``load_mmpk_data``. Salt normalization is
required: clopidogrel bisulfate and
sumatriptan (Onzetra Xsail) otherwise have different IK14s from their free
forms. Pravastatin previously had a wrong reference structure
(``GOSGZXISMCZCDW`` versus the MMPK ``TUZYXOIXSAXUGO``); the flag and name
exclusion were corrected then, and the reference structure is corrected now.
These tests pin the invariant against future identity drift.
"""
from __future__ import annotations

import csv
import json
import pathlib

from sisyphus.validation.identity import ik14

ROOT = pathlib.Path(__file__).resolve().parents[2]
_MMPK = [
    ROOT / "data/training/mmpk_expanded_full.csv",
    ROOT / "data/training/mmpk_expanded_v2.csv",
]
_HOLDOUT = json.loads((ROOT / "data/reference/holdout.json").read_text())
_HOLDOUT_NAMES = {n.lower().strip() for n in _HOLDOUT.get("holdout", [])}


def _rows(path: pathlib.Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_pravastatin_in_holdout_flag_is_true():
    """Direct guard on the corrected data defect (rdkit-free, always runs)."""
    for path in _MMPK:
        for r in _rows(path):
            if r["name"].strip().lower() == "pravastatin":
                assert r["in_holdout"].strip().lower() == "true", (
                    f"{path.name}: pravastatin is a holdout drug; its in_holdout "
                    f"flag must be True (got {r['in_holdout']!r})"
                )


def test_salt_form_holdout_flags_are_true():
    for path in _MMPK:
        for row in _rows(path):
            if row["name"].strip().lower() in {
                "clopidogrel bisulfate", "sumatriptan (onzetra xsail)",
            }:
                assert row["in_holdout"].strip().lower() == "true", (
                    f"{path.name}: {row['name']} must be excluded from training"
                )


def test_salt_form_ik14_matches_holdout_parent():
    clinical = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())["drugs"]
    salts = {
        "clopidogrel": "COC(=O)C(c1ccccc1Cl)N1CCc2sccc2C1.O=S(=O)(O)O",
        "sumatriptan": "CNS(=O)(=O)CC1=CC2=C(C=C1)NC=C2CCN(C)C.C(CC(=O)O)C(=O)O",
    }
    for name, salt in salts.items():
        assert ik14(salt) == ik14(clinical[name]["smiles"])


def test_no_holdout_drug_survives_mmpk_filters():
    """Replicate load_mmpk_data's effective (in_holdout OR InChIKey-14) filter
    and assert no holdout drug reaches the ML Cmax training set."""
    clinical = json.loads(
        (ROOT / "data/reference/clinical_pk.json").read_text()
    ).get("drugs", {})
    ref_names = _HOLDOUT.get("holdout", []) + _HOLDOUT.get("train", [])
    ho_ik = set()
    for n in ref_names:
        e = clinical.get(n) or clinical.get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            k = ik14(e["smiles"])
            if k:
                ho_ik.add(k)

    leaks: dict[str, set[str]] = {}
    for path in _MMPK:
        for r in _rows(path):
            if str(r.get("in_holdout", "")).strip().lower() == "true":
                continue
            try:
                if float(r.get("cmax_mg_L", 0)) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            k = ik14(r.get("canon_smiles"))
            if k is None or k in ho_ik:
                continue
            name = str(r.get("name", "")).strip().lower()
            if name in _HOLDOUT_NAMES:
                leaks.setdefault(path.name, set()).add(name)

    assert not leaks, f"holdout drugs leaking into MMPK training: {leaks}"
