"""Guard: the CL/F-track training builder must exclude holdout compounds by
structure (InChIKey-14), not name alone.

`scripts/build_clf_training_data.py` originally filtered holdout drugs with three
name/flag keys (in_holdout flag, holdout name list, MMPK exclusion list). Those
miss holdout compounds whose training-row name differs by spelling, salt form, or
stereo descriptor — e.g. valaciclovir vs valacyclovir, darunavir vs darunavir
ethanolate. A structural InChIKey-14 key closes that gap.

This test pins two invariants:
  1. The builder exposes a structural holdout key set (`load_holdout_inchikeys`).
  2. That set catches the known name-evading collisions found in the CLF audit.

Note (documented limitation): InChIKey-14 matches on the connectivity block and
ignores stereochemistry, so it also flags true diastereomers that share
connectivity (quinidine, present in clf_training.csv, collides with the holdout
drug quinine). This is a conservative false-positive: it removes a legitimate
training row rather than admitting a leak, which is the safe direction. The AAFE
impact of this exclusion is negligible (a leak-free retrain moves the 107-holdout
meta AAFE by Delta ~= 0.007, within bootstrap-CI noise).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BUILD = _ROOT / "scripts" / "build_clf_training_data.py"


def _load_builder():
    """Import the build script as a module (it guards main() under __main__)."""
    import sys

    sys.path.insert(0, str(_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_clf_training_data", _BUILD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_builder_exposes_structural_holdout_keys():
    """The builder must load a non-empty InChIKey-14 holdout set."""
    mod = _load_builder()
    assert hasattr(mod, "load_holdout_inchikeys"), (
        "build_clf_training_data.py must expose load_holdout_inchikeys() — the "
        "structural (InChIKey-14) holdout filter that name matching cannot replace."
    )
    keys = mod.load_holdout_inchikeys()
    assert isinstance(keys, set) and len(keys) >= 100, (
        f"expected ~106 holdout InChIKey-14 keys, got {len(keys)}"
    )


def test_structural_filter_catches_name_evading_leaks():
    """valacyclovir (spelling) and darunavir ethanolate (salt) must be caught by
    the InChIKey-14 filter even though their CLF-row names differ from holdout."""
    mod = _load_builder()
    keys = mod.load_holdout_inchikeys()

    # SMILES as they appear in clf_training.csv (name-evading forms)
    cases = {
        "valaciclovir": "CC(C)[C@H](N)C(=O)OCCOCn1cnc2c(=O)nc(N)[nH]c21",
        "darunavir": "CC(C)CN(C[C@@H](O)[C@H](Cc1ccccc1)NC(=O)O[C@H]1CO[C@H]2OCC[C@H]21)S(=O)(=O)c1ccc(N)cc1",  # noqa: E501
    }
    for name, smi in cases.items():
        ik = mod._inchikey14(smi)
        assert ik is not None, f"{name}: InChIKey-14 computation failed"
        assert ik in keys, (
            f"{name} (IK14 {ik}) is not caught by the structural holdout filter — "
            "a name-evading CLF-track leak would slip through."
        )


def test_shipped_clf_training_csv_is_holdout_clean():
    """Artifact-level guard (Invariant #5): the *committed* clf_training.csv must
    contain ZERO holdout drugs by InChIKey-14.

    The two tests above verify the builder's filter *code*; this one verifies the
    shipped *artifact* the CLF/VDF models are trained from. A stale regeneration
    or hand-edit that reintroduced a holdout drug would pass the code tests but
    fail here. Mirrors tests/regression/test_mmpk_holdout_leak.py's artifact scan.
    """
    import csv

    mod = _load_builder()
    holdout_ik = mod.load_holdout_inchikeys()
    csv_path = _ROOT / "data" / "training" / "clf_training.csv"
    leaks = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            ik = mod._inchikey14(row["smiles"].strip())
            if ik and ik in holdout_ik:
                leaks.append((row["name"], ik))
    assert not leaks, (
        f"clf_training.csv contains {len(leaks)} holdout drug(s) by InChIKey-14: "
        f"{leaks[:5]} — the structural filter was not applied to the shipped "
        "artifact. Regenerate via scripts/build_clf_training_data.py."
    )
