"""Invariant #5 guard: no holdout drug may enter the ML Cmax (MMPK) training set.

A holdout drug leaks into the MMPK corpus only if it survives BOTH filters in
``scripts/ml_cmax_improvement.py::load_mmpk_data`` — an ``in_holdout=False`` row
AND a ``canon_smiles`` InChIKey-14 absent from the holdout key set (built from
clinical_pk SMILES). Pravastatin slipped both (connectivity-level SMILES
mismatch: clinical_pk ``GOSGZXISMCZCDW`` vs MMPK ``TUZYXOIXSAXUGO``) until its
``in_holdout`` flag was corrected and a name-based exclusion added. These tests
pin the invariant and catch any future SMILES-representation drift.
"""
from __future__ import annotations

import csv
import json
import pathlib

import pytest

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


def test_no_holdout_drug_survives_mmpk_filters():
    """Replicate load_mmpk_data's effective (in_holdout OR InChIKey-14) filter
    and assert no holdout drug reaches the ML Cmax training set."""
    pytest.importorskip("rdkit")
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi

    RDLogger.DisableLog("rdApp.*")

    def ik14(smiles: str | None) -> str | None:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi = MolToInchi(mol)
        if not inchi:
            return None
        key = InchiToInchiKey(inchi)
        return key[:14] if key else None

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
