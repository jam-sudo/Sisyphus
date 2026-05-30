# v0.3 ECM Auto-Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `pipeline.predict.predict()` to automatically activate the Extended Clearance Model (ECM) for drugs registered as OATP-rate-limited substrates, and add a `phenotypes=` parameter for PGx-aware predictions.

**Architecture:** Three-layer extension of PR #22's registry pattern with no engine code changes. (1) New `ecm_applicable: bool` field in `oatp1b1.json` per drug, default false. (2) `predict()` looks up the flag via SMILES → InChIKey (RDKit), conditionally loading transporter kinetics + ECM params before `build_drug_on_graph()`. (3) Optional `phenotypes` dict triggers `apply_phenotype_to_graph` before drug binding. Initial seed list: `pravastatin` only (pitavastatin/rosuvastatin/atorvastatin deferred pending `metabolic_fraction` curation — empirical validation showed pitavastatin auto-ECM without `metabolic_fraction` triple-counts hepatic clearance).

**Tech Stack:** Python 3.10, RDKit (already a dependency), JSON registries, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-03-ecm-auto-activation-design.md`

---

## Task 1: Add `is_oatp_ecm_applicable` helper to transporter_db.py

**Files:**
- Modify: `src/sisyphus/predict/transporter_db.py`
- Test: `tests/unit/test_transporter_db_applicability.py` (NEW)

- [ ] **Step 1: Read existing transporter_db.py to understand the current load pattern**

Run: `cat src/sisyphus/predict/transporter_db.py | head -60`
Note the existing `load_oatp1b1_kinetics(name: str)` signature and the `_load_registry()` cache pattern.

- [ ] **Step 2: Write the failing unit test**

Create `tests/unit/test_transporter_db_applicability.py`:

```python
"""Unit tests for OATP ECM applicability lookup."""
from __future__ import annotations

from sisyphus.predict.transporter_db import is_oatp_ecm_applicable


_PRAVA_SMILES = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)
_FLUVA_SMILES = (
    "CC(C)N1C2=CC=CC=C2C(=C1/C=C/[C@H](O)C[C@H](O)CC(=O)O)"
    "C3=CC=C(F)C=C3"
)
_MIDAZOLAM_SMILES = "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2"


def test_pravastatin_is_applicable():
    """Pravastatin is the canonical OATP-rate-limited substrate (Niemi 2009 PM/EM ~2.6x)."""
    assert is_oatp_ecm_applicable(_PRAVA_SMILES) is True


def test_fluvastatin_not_applicable():
    """Fluvastatin is CYP2C9-dominant (Niemi 2009 PM/EM ~1.0x)."""
    assert is_oatp_ecm_applicable(_FLUVA_SMILES) is False


def test_non_oatp_drug_not_applicable():
    """Midazolam is not in the OATP1B1 registry at all."""
    assert is_oatp_ecm_applicable(_MIDAZOLAM_SMILES) is False


def test_invalid_smiles_returns_false():
    """Bad SMILES → False (no exception, fail-safe)."""
    assert is_oatp_ecm_applicable("not_a_valid_smiles") is False


def test_empty_smiles_returns_false():
    """Empty SMILES → False (no exception)."""
    assert is_oatp_ecm_applicable("") is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_transporter_db_applicability.py -v`
Expected: FAIL with `ImportError: cannot import name 'is_oatp_ecm_applicable' from 'sisyphus.predict.transporter_db'`

- [ ] **Step 4: Implement the helper**

Add to the bottom of `src/sisyphus/predict/transporter_db.py`:

```python
@functools.lru_cache(maxsize=1)
def _load_oatp_applicability_index() -> dict[str, bool]:
    """InChIKey → ecm_applicable flag, indexed once.

    Returns empty dict if registry missing. Drugs without an explicit
    `ecm_applicable` field default to False (key absent).
    """
    if not _OATP1B1_PATH.exists():
        return {}
    with _OATP1B1_PATH.open() as f:
        data = json.load(f)
    index: dict[str, bool] = {}
    for _name, entry in data.get("drugs", {}).items():
        ikey = entry.get("inchikey")
        flag = entry.get("ecm_applicable", False)
        if ikey is None:
            continue
        index[ikey] = bool(flag)
    return index


def is_oatp_ecm_applicable(smiles: str) -> bool:
    """Return True if SMILES's InChIKey is registered with ecm_applicable=true.

    Uses RDKit-canonical InChIKey to be robust against SMILES annotation
    differences. Returns False on any error (RDKit unavailable, invalid
    SMILES, InChIKey not registered, or registered with explicit false).

    Mirrors the cyp_clearance_overrides.lookup_metabolic_fraction pattern
    introduced in PR #22.
    """
    if not smiles:
        return False
    try:
        from rdkit import Chem
    except ImportError:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ikey = Chem.MolToInchiKey(mol)
    return _load_oatp_applicability_index().get(ikey, False)
```

You will likely need to add `import functools` at the top of the module if not already present, and verify that `_OATP1B1_PATH` (the path constant for `data/transporters/oatp1b1.json`) is accessible. If the existing module uses a different name (e.g. `_OATP_REGISTRY_PATH`), match that.

- [ ] **Step 5: Run test to verify it passes — note the registry doesn't yet have ecm_applicable=true**

Run: `python3 -m pytest tests/unit/test_transporter_db_applicability.py -v`
Expected: 4 PASS, 1 FAIL — `test_pravastatin_is_applicable` will fail because the registry doesn't yet have `ecm_applicable: true` set on pravastatin. This is expected; Task 2 fixes it.

- [ ] **Step 6: Mark the pravastatin test xfail temporarily**

Edit `tests/unit/test_transporter_db_applicability.py`:

```python
import pytest


@pytest.mark.xfail(reason="oatp1b1.json pravastatin entry not yet flagged ecm_applicable=true; Task 2 sets it")
def test_pravastatin_is_applicable():
    """Pravastatin is the canonical OATP-rate-limited substrate (Niemi 2009 PM/EM ~2.6x)."""
    assert is_oatp_ecm_applicable(_PRAVA_SMILES) is True
```

Run: `python3 -m pytest tests/unit/test_transporter_db_applicability.py -v`
Expected: 4 PASS, 1 XFAIL.

- [ ] **Step 7: Commit**

```bash
git add src/sisyphus/predict/transporter_db.py tests/unit/test_transporter_db_applicability.py
git commit -m "feat(transporter_db): add is_oatp_ecm_applicable helper

Mirrors PR #22 lookup_metabolic_fraction pattern for the parallel
ecm_applicable lookup. Pravastatin test xfail until Task 2 sets the
registry flag.

Part of v0.3 ECM auto-activation."
```

---

## Task 2: Set `ecm_applicable: true` on pravastatin in oatp1b1.json

**Files:**
- Modify: `data/transporters/oatp1b1.json:7-22` (pravastatin entry)
- Test: `tests/unit/test_transporter_db_applicability.py` (un-xfail)

- [ ] **Step 1: Read pravastatin's current entry**

Run: `python3 -c "import json; d=json.load(open('data/transporters/oatp1b1.json')); import json; print(json.dumps(d['drugs']['pravastatin'], indent=2))"`
Expected output: pravastatin entry with `jmax_pmol_per_min_per_mg`, `km_uM`, `source`, `smiles`, `inchikey` — but no `ecm_applicable` field yet.

- [ ] **Step 2: Add the flag**

Edit `data/transporters/oatp1b1.json` — find the pravastatin entry (under `"drugs": { "pravastatin": ... }`) and add `"ecm_applicable": true` as a new field. The entry will look like:

```json
"pravastatin": {
  "jmax_pmol_per_min_per_mg": {"mean": 228.0, "cv": 0.3},
  "km_uM": {"mean": 13.6, "cv": 0.25},
  "source": "Varma 2014 JPET Table 2",
  "smiles": "CC[C@@H](C)C(=O)O[C@@H]1C[C@@H](O)C=C2[C@@H](CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@H](C)CC[C@@H]21",
  "inchikey": "GOSGZXISMCZCDW-LYANWTNHSA-N",
  "ecm_applicable": true
}
```

- [ ] **Step 3: Verify JSON is valid**

Run: `python3 -c "import json; json.load(open('data/transporters/oatp1b1.json')); print('valid')"`
Expected: `valid`

- [ ] **Step 4: Remove the xfail marker from the test**

Edit `tests/unit/test_transporter_db_applicability.py` — remove `@pytest.mark.xfail(...)` from `test_pravastatin_is_applicable`. Keep the import of pytest (still used elsewhere or remove if unused).

- [ ] **Step 5: Run unit tests to verify all pass**

Run: `python3 -m pytest tests/unit/test_transporter_db_applicability.py -v`
Expected: 5 PASS.

- [ ] **Step 6: Verify no regression in existing OATP tests**

Run: `python3 -m pytest tests/integration/test_oatp_pravastatin.py tests/integration/test_oatp_ecm_statins.py -v`
Expected: same pass/xfail counts as before this task (4 passed, 3 xfailed). The `ecm_applicable` flag doesn't affect direct manual `build_drug_on_graph` calls.

- [ ] **Step 7: Commit**

```bash
git add data/transporters/oatp1b1.json tests/unit/test_transporter_db_applicability.py
git commit -m "feat(oatp1b1): flag pravastatin ecm_applicable=true

Initial seed list for v0.3 ECM auto-activation: pravastatin only.
Pravastatin is the canonical OATP-rate-limited substrate (Niemi 2009
PM/EM ~2.6x) and has metabolic_fraction=0 already registered in
cyp_clearance_overrides.json (PR #22), preventing the
double-counting that would occur for a drug without that pairing.

pitavastatin/rosuvastatin/atorvastatin remain unflagged (deferred
pending metabolic_fraction curation per spec §1.3)."
```

---

## Task 3: Add SMILES-keyed kinetics + ECM loaders to transporter_db.py

**Files:**
- Modify: `src/sisyphus/predict/transporter_db.py`
- Test: `tests/unit/test_transporter_db_applicability.py` (extend)

- [ ] **Step 1: Write failing tests for the SMILES-keyed loaders**

Append to `tests/unit/test_transporter_db_applicability.py`:

```python
from sisyphus.predict.transporter_db import (
    load_oatp1b1_kinetics_for_smiles,
    load_hepatic_ecm_params_for_smiles,
)


def test_load_oatp1b1_kinetics_for_smiles_pravastatin():
    """SMILES-keyed loader returns the same kinetics as name-keyed."""
    kin = load_oatp1b1_kinetics_for_smiles(_PRAVA_SMILES)
    assert kin is not None
    assert "OATP1B1" in kin
    assert kin["OATP1B1"].jmax.mean == 228.0
    assert kin["OATP1B1"].km.mean == 13.6


def test_load_oatp1b1_kinetics_for_smiles_unknown_returns_none():
    """Unregistered SMILES → None (caller can fall through to no-ECM path)."""
    assert load_oatp1b1_kinetics_for_smiles(_MIDAZOLAM_SMILES) is None


def test_load_hepatic_ecm_params_for_smiles_pravastatin():
    """SMILES-keyed ECM loader returns the registered params."""
    ecm = load_hepatic_ecm_params_for_smiles(_PRAVA_SMILES)
    assert ecm is not None
    assert ecm["ps_passive"].mean == 0.8
    assert ecm["ps_eff"].mean == 0.8
    assert ecm["cl_int_bile"].mean == 45.0


def test_load_hepatic_ecm_params_for_smiles_unknown_returns_none():
    assert load_hepatic_ecm_params_for_smiles(_MIDAZOLAM_SMILES) is None


def test_load_for_invalid_smiles_returns_none():
    assert load_oatp1b1_kinetics_for_smiles("not_smiles") is None
    assert load_hepatic_ecm_params_for_smiles("not_smiles") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_transporter_db_applicability.py::test_load_oatp1b1_kinetics_for_smiles_pravastatin -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement the SMILES-keyed loaders**

Add to `src/sisyphus/predict/transporter_db.py`:

```python
@functools.lru_cache(maxsize=1)
def _load_inchikey_to_name() -> dict[str, str]:
    """InChIKey → drug name reverse index from oatp1b1.json."""
    if not _OATP1B1_PATH.exists():
        return {}
    with _OATP1B1_PATH.open() as f:
        data = json.load(f)
    index: dict[str, str] = {}
    for name, entry in data.get("drugs", {}).items():
        ikey = entry.get("inchikey")
        if ikey is None:
            continue
        index[ikey] = name
    return index


def _smiles_to_drug_name(smiles: str) -> str | None:
    """Resolve SMILES → registered drug name via RDKit InChIKey, or None."""
    if not smiles:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ikey = Chem.MolToInchiKey(mol)
    return _load_inchikey_to_name().get(ikey)


def load_oatp1b1_kinetics_for_smiles(smiles: str):
    """SMILES-keyed wrapper around load_oatp1b1_kinetics. Returns None if unregistered."""
    name = _smiles_to_drug_name(smiles)
    if name is None:
        return None
    return load_oatp1b1_kinetics(name)


def load_hepatic_ecm_params_for_smiles(smiles: str):
    """SMILES-keyed wrapper around load_hepatic_ecm_params. Returns None if unregistered."""
    name = _smiles_to_drug_name(smiles)
    if name is None:
        return None
    return load_hepatic_ecm_params(name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_transporter_db_applicability.py -v`
Expected: 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/predict/transporter_db.py tests/unit/test_transporter_db_applicability.py
git commit -m "feat(transporter_db): SMILES-keyed kinetics + ECM loaders

Wraps the existing name-keyed load_oatp1b1_kinetics and
load_hepatic_ecm_params with InChIKey reverse-indexed SMILES lookups,
mirroring the cyp_clearance_overrides PR #22 pattern. Returns None
for unregistered SMILES so callers (predict()) can fall through to
the no-ECM path cleanly."
```

---

## Task 4: Schema regression test (oatp1b1.json + cyp_clearance_overrides.json pairing)

**Files:**
- Test: `tests/regression/test_oatp_registry_schema.py` (NEW)

- [ ] **Step 1: Write the schema regression test**

Create `tests/regression/test_oatp_registry_schema.py`:

```python
"""Schema gates for oatp1b1.json + cyp_clearance_overrides.json pairing.

Prevents the pitavastatin-class double-counting bug from recurring:
every drug flagged ecm_applicable=true MUST have a matching
metabolic_fraction entry in cyp_clearance_overrides.json, otherwise
auto-activating ECM in predict() triple-counts hepatic clearance.
"""
from __future__ import annotations

import json
import pathlib

import pytest
from rdkit import Chem


_OATP1B1_PATH = pathlib.Path("data/transporters/oatp1b1.json")
_OVERRIDES_PATH = pathlib.Path("data/transporters/cyp_clearance_overrides.json")


# Pin the v0.3 seed list. If a drug is added/removed here, this test must
# fail loud — the spec change requires an explicit decision.
_EXPECTED_ECM_APPLICABLE = frozenset({"pravastatin"})


def _load_oatp() -> dict:
    return json.loads(_OATP1B1_PATH.read_text())


def _load_overrides() -> dict:
    return json.loads(_OVERRIDES_PATH.read_text())


def test_seed_list_pinned():
    """Catch silent flag flips: only the expected drugs are flagged true."""
    data = _load_oatp()
    actual_true = {
        name for name, entry in data["drugs"].items()
        if entry.get("ecm_applicable") is True
    }
    assert actual_true == _EXPECTED_ECM_APPLICABLE, (
        f"ecm_applicable=true set drift: expected {_EXPECTED_ECM_APPLICABLE}, "
        f"got {actual_true}. If intentional, update _EXPECTED_ECM_APPLICABLE "
        f"and ensure cyp_clearance_overrides.json has paired metabolic_fraction "
        f"entries for the new drugs."
    )


def test_inchikey_matches_smiles():
    """For each ecm_applicable=true entry, the registered InChIKey must
    match the RDKit canonicalization of the registered SMILES."""
    data = _load_oatp()
    for name, entry in data["drugs"].items():
        if entry.get("ecm_applicable") is not True:
            continue
        smiles = entry["smiles"]
        registered = entry["inchikey"]
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"{name}: SMILES failed to parse"
        derived = Chem.MolToInchiKey(mol)
        assert derived == registered, (
            f"{name}: registered InChIKey {registered} does not match "
            f"RDKit-derived {derived} from SMILES. Either correct the "
            f"SMILES or update the InChIKey field."
        )


def test_metabolic_fraction_paired():
    """Every ecm_applicable=true drug has a paired entry in
    cyp_clearance_overrides.json — otherwise auto-activating ECM
    triple-counts hepatic clearance.

    This gate prevents the empirically-confirmed pitavastatin bug
    (auto-ECM with default metabolic_fraction=1.0 → FE 0.450 → 2.120,
    direction-flip with magnitude unchanged).
    """
    oatp = _load_oatp()
    overrides = _load_overrides()
    override_inchikeys = {entry["inchikey"] for entry in overrides["overrides"]}

    for name, entry in oatp["drugs"].items():
        if entry.get("ecm_applicable") is not True:
            continue
        ikey = entry["inchikey"]
        assert ikey in override_inchikeys, (
            f"{name} (InChIKey {ikey}) is flagged ecm_applicable=true "
            f"but has no metabolic_fraction entry in "
            f"cyp_clearance_overrides.json. This will triple-count "
            f"hepatic clearance when predict() auto-activates ECM. "
            f"Add a literature-justified metabolic_fraction entry "
            f"OR remove the ecm_applicable flag."
        )
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `python3 -m pytest tests/regression/test_oatp_registry_schema.py -v`
Expected: 3 PASS. (Pravastatin satisfies all three gates: in seed list, InChIKey matches SMILES, has metabolic_fraction=0 entry in overrides.)

- [ ] **Step 3: Verify the pairing gate catches the pitavastatin bug**

Temporarily edit `data/transporters/oatp1b1.json` — set pitavastatin's `ecm_applicable: true`. Run:

```bash
python3 -m pytest tests/regression/test_oatp_registry_schema.py::test_metabolic_fraction_paired -v
```

Expected: FAIL with message naming pitavastatin's InChIKey as missing from overrides.

Now revert the change:
```bash
git checkout data/transporters/oatp1b1.json
```

Re-run the test:
```bash
python3 -m pytest tests/regression/test_oatp_registry_schema.py -v
```
Expected: 3 PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/test_oatp_registry_schema.py
git commit -m "test(regression): pin OATP registry schema + metabolic_fraction pairing

Three-gate schema regression test:
  1. ecm_applicable=true set is pinned to {pravastatin} (catches silent
     flag flips)
  2. registered InChIKey matches RDKit-canonicalization of SMILES
     (catches SMILES drift like the issue #25 connectivity error)
  3. every ecm_applicable=true drug has a paired metabolic_fraction
     entry in cyp_clearance_overrides.json (prevents the empirically
     confirmed pitavastatin double-counting recurrence)

Promoting a drug to ecm_applicable=true requires updating both the
registry entry AND the seed list constant in this test, surfacing the
decision in code review."
```

---

## Task 5: predict() auto-ECM hook (no phenotypes yet)

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py`
- Test: `tests/integration/test_predict_auto_ecm.py` (NEW; 2 tests this task)

- [ ] **Step 1: Read existing predict.py to understand the build_drug_on_graph call site**

Run: `grep -n "build_drug_on_graph\|compute_profile\|predict_adme" src/sisyphus/pipeline/predict.py | head -20`
Note the line numbers and existing argument structure.

- [ ] **Step 2: Write the failing integration tests**

Create `tests/integration/test_predict_auto_ecm.py`:

```python
"""Integration tests for predict() auto-ECM activation (v0.3).

The first two tests gate the core wiring: pravastatin must auto-activate,
fluvastatin must NOT. Phenotype tests added in Task 7.
"""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.pipeline.predict import predict


_PRAVA_SMILES = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)
_FLUVA_SMILES = (
    "CC(C)N1C2=CC=CC=C2C(=C1/C=C/[C@H](O)C[C@H](O)CC(=O)O)"
    "C3=CC=C(F)C=C3"
)


@pytest.mark.slow
def test_pravastatin_auto_ecm_activates():
    """predict(pravastatin) auto-activates ECM → engine Cmax matches the
    ECM-active manual build.

    Reference Cmax (40 mg, realize_means(), ECM-active, metabolic_fraction=0):
    0.0422 mg/L per scripts/diagnose_pravastatin_ecm.py.
    """
    result = predict(_PRAVA_SMILES, dose_mg=40.0, route="oral", n_mc_samples=0)
    assert result.engine_pk is not None
    cmax = result.engine_pk.cmax.mean
    expected = 0.0422
    rel_err = abs(cmax - expected) / expected
    assert rel_err < 0.05, (
        f"pravastatin auto-ECM Cmax drift: actual={cmax:.4f}, expected={expected:.4f}, "
        f"rel_err={rel_err:.3f} (tolerance 5%). Auto-activation may be misfiring."
    )


@pytest.mark.slow
def test_fluvastatin_no_auto_ecm():
    """predict(fluvastatin) does NOT auto-activate ECM → engine Cmax matches
    the no-ECM manual build (~0.058 mg/L at 40 mg per the issue #21 closure
    diagnostic).

    Fluvastatin is not OATP-rate-limited (Niemi 2009 PM/EM ~1.0x).
    Activating ECM would triple-count hepatic clearance.
    """
    result = predict(_FLUVA_SMILES, dose_mg=40.0, route="oral", n_mc_samples=0)
    assert result.engine_pk is not None
    cmax = result.engine_pk.cmax.mean
    # No-ECM manual build gave 0.0583 mg/L. Allow 5% tolerance.
    expected = 0.0583
    rel_err = abs(cmax - expected) / expected
    assert rel_err < 0.05, (
        f"fluvastatin Cmax unexpectedly shifted: actual={cmax:.4f}, "
        f"expected={expected:.4f} (no-ECM path). Auto-activation may be "
        f"firing for a drug it shouldn't."
    )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/integration/test_predict_auto_ecm.py -v`
Expected: pravastatin FAIL (predict() returns the no-ECM Cmax 0.0500 not 0.0422); fluvastatin PASS by accident (no-ECM is the current default for both).

- [ ] **Step 4: Implement the auto-ECM hook in predict()**

Open `src/sisyphus/pipeline/predict.py`. Find the section that calls `build_drug_on_graph(...)` (typically right after `predict_adme(profile)`). Add the auto-ECM lookup before the call:

```python
from sisyphus.predict.transporter_db import (
    is_oatp_ecm_applicable,
    load_oatp1b1_kinetics_for_smiles,
    load_hepatic_ecm_params_for_smiles,
)

# ... inside predict(), after compute_profile + predict_adme, before build_drug_on_graph:

transporter_kinetics = None
hepatic_ecm_params = None
if is_oatp_ecm_applicable(smiles):
    transporter_kinetics = load_oatp1b1_kinetics_for_smiles(smiles)
    hepatic_ecm_params = load_hepatic_ecm_params_for_smiles(smiles)
    logger.info("Auto-ECM active for SMILES (substrate detected via InChIKey)")
```

Then pass these into the `build_drug_on_graph(...)` call as keyword arguments (the function already accepts both — see PR #22). The exact line might look like:

```python
drug = build_drug_on_graph(
    profile,
    adme,
    dose_mg=dose_mg,
    route=route,
    liver_enzymes=liver_enzymes,
    transporter_kinetics=transporter_kinetics,   # NEW
    hepatic_ecm_params=hepatic_ecm_params,       # NEW
)
```

If `build_drug_on_graph` is called with only positional args, switch to keyword args first to make the addition unambiguous.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/integration/test_predict_auto_ecm.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Verify no regression on the OATP suite**

Run: `python3 -m pytest tests/integration/test_oatp_pravastatin.py tests/integration/test_oatp_ecm_statins.py -v`
Expected: same pass/xfail pattern as before (4 passed, 3 xfailed). The manual-build tests are independent of predict()'s wiring.

- [ ] **Step 7: Run mass balance on auto-ECM path**

Quick smoke test the engine still conserves mass when auto-ECM is active. Append to `tests/integration/test_predict_auto_ecm.py`:

```python
@pytest.mark.slow
def test_pravastatin_auto_ecm_mass_balance():
    """Auto-ECM adds OATP1B1 saturable + ECM passive + biliary CL_int paths.
    Mass balance must still close (engine invariant)."""
    # predict() does not currently expose mass balance directly. Run via
    # the same engine path manually to verify the ECM additions don't
    # break conservation.
    import pathlib
    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph
    from sisyphus.predict.transporter_db import (
        load_hepatic_ecm_params_for_smiles,
        load_oatp1b1_kinetics_for_smiles,
    )

    graph = build_from_yaml(pathlib.Path("data/physiology/reference_man.yaml"))
    profile = compute_profile(_PRAVA_SMILES)
    adme = predict_adme(profile)
    liver_enz = {tag: d.mean for tag, d in graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        liver_enzymes=liver_enz,
        transporter_kinetics=load_oatp1b1_kinetics_for_smiles(_PRAVA_SMILES),
        hepatic_ecm_params=load_hepatic_ecm_params_for_smiles(_PRAVA_SMILES),
    )
    rg, rd = graph.realize_means(), drug.realize_means()
    compiler = ODECompiler()
    compiled = compiler.compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    result = solve(compiled, params, y0, t_span=(0, 24.0))
    assert result.solver_success
    assert result.mass_balance_error < 1e-6, (
        f"auto-ECM mass balance broken: error={result.mass_balance_error:.2e}"
    )
```

Run: `python3 -m pytest tests/integration/test_predict_auto_ecm.py::test_pravastatin_auto_ecm_mass_balance -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/sisyphus/pipeline/predict.py tests/integration/test_predict_auto_ecm.py
git commit -m "feat(predict): auto-activate ECM for OATP-rate-limited substrates

When predict() detects (via InChIKey lookup) that the SMILES is
flagged ecm_applicable=true in oatp1b1.json, it loads the substrate's
OATP1B1 kinetics and ECM passive/biliary params and passes them to
build_drug_on_graph. This fires for pravastatin only in the v0.3 seed
list. metabolic_fraction is already wired through ivive (PR #22).

Pravastatin engine Cmax shifts 0.0500 -> 0.0422 mg/L at 40 mg
(matches FDA 0.045 at FE 1.066 versus the prior 1.11x over-pred).
Fluvastatin (not flagged) unchanged.

Mass balance verified post-activation."
```

---

## Task 6: predict() phenotypes parameter

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py` (add `phenotypes=` kwarg)
- Test: `tests/integration/test_predict_auto_ecm.py` (extend)

- [ ] **Step 1: Write failing tests for phenotype API**

Append to `tests/integration/test_predict_auto_ecm.py`:

```python
_MIDAZOLAM_SMILES = "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2"


@pytest.mark.slow
def test_phenotype_changes_pravastatin_cmax():
    """SLCO1B1 PM raises pravastatin Cmax (less hepatic uptake).

    Replicates the existing test_oatp_pravastatin invariant via the new
    predict(phenotypes=) parameter.
    """
    cmax_em = predict(_PRAVA_SMILES, dose_mg=40.0, route="oral").engine_pk.cmax.mean
    cmax_pm = predict(
        _PRAVA_SMILES, dose_mg=40.0, route="oral",
        phenotypes={"SLCO1B1": "PM"},
    ).engine_pk.cmax.mean

    print(f"\npravastatin auto-ECM via predict: EM={cmax_em:.4f} PM={cmax_pm:.4f}")
    assert cmax_pm > cmax_em, (
        f"PM should raise Cmax (less hepatic uptake): EM={cmax_em:.4f} PM={cmax_pm:.4f}"
    )
    assert cmax_pm / cmax_em > 1.10, (
        f"expected ≥10% Cmax uplift at PM, got ratio {cmax_pm/cmax_em:.3f}"
    )


@pytest.mark.slow
def test_phenotype_orthogonal_for_non_substrate():
    """Phenotype scaling on a non-substrate has no clearance path
    consuming it → Cmax invariant. Documents the orthogonality guarantee.
    """
    cmax_default = predict(_MIDAZOLAM_SMILES, dose_mg=15.0, route="oral").engine_pk.cmax.mean
    cmax_pm = predict(
        _MIDAZOLAM_SMILES, dose_mg=15.0, route="oral",
        phenotypes={"SLCO1B1": "PM"},
    ).engine_pk.cmax.mean
    rel_diff = abs(cmax_pm - cmax_default) / cmax_default
    assert rel_diff < 0.001, (
        f"midazolam Cmax shifted under SLCO1B1 PM (should be invariant — "
        f"midazolam is not OATP1B1 substrate): default={cmax_default:.4f}, "
        f"PM={cmax_pm:.4f}, rel_diff={rel_diff:.4f}"
    )


@pytest.mark.slow
def test_smiles_variant_robustness():
    """Stereo-stripped pravastatin SMILES (different InChIKey) does NOT
    auto-activate ECM. Documents the full-InChIKey matching contract.
    """
    stripped = "CCC(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC(CC(CC(=O)O)O)O)O"
    # InChIKey for this connectivity-only SMILES is TUZYXOIXSAXUGO-... (issue #25
    # connectivity error case). It is NOT registered ecm_applicable=true.
    result = predict(stripped, dose_mg=40.0, route="oral")
    cmax = result.engine_pk.cmax.mean
    # If auto-ECM did NOT fire, Cmax matches the no-ECM path (~0.05 at 40 mg).
    # If it did fire (bug), Cmax would be ~0.042. Gate at 0.045 to distinguish.
    assert cmax > 0.045, (
        f"stereo-stripped pravastatin should NOT trigger auto-ECM (different "
        f"InChIKey TUZYXOIXSAXUGO vs registered GOSGZXISMCZCDW), but Cmax {cmax:.4f} "
        f"is below the no-ECM threshold."
    )


@pytest.mark.slow
def test_mc_sampling_with_auto_ecm():
    """n_mc_samples > 0 produces a non-degenerate prediction interval with
    auto-ECM active.
    """
    result = predict(_PRAVA_SMILES, dose_mg=40.0, route="oral", n_mc_samples=10)
    assert result.cmax_90ci is not None
    lo, hi = result.cmax_90ci
    assert lo > 0
    assert hi > lo, f"degenerate PI: lo={lo}, hi={hi}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/integration/test_predict_auto_ecm.py -v -k phenotype`
Expected: FAIL with `TypeError: predict() got an unexpected keyword argument 'phenotypes'`.

- [ ] **Step 3: Add phenotypes parameter to predict()**

Edit `src/sisyphus/pipeline/predict.py`. Update the signature:

```python
def predict(
    smiles: str,
    *,
    dose_mg: float,
    route: str = "oral",
    n_mc_samples: int = 0,
    phenotypes: dict[str, str] | None = None,  # NEW
) -> PredictionResult:
```

After `graph = build_from_yaml(...)` (or wherever the BodyGraph is constructed), insert:

```python
if phenotypes is not None:
    from sisyphus.predict.phenotype import apply_phenotype_to_graph
    graph = apply_phenotype_to_graph(graph, phenotypes)
    logger.info("Phenotypes applied: %s", phenotypes)
```

Update the docstring (find the existing `"""..."""` block and add):

```
        phenotypes: Optional dict of {gene: phenotype_label}. Supported
            keys: SLCO1B1, CYP2D6, CYP2C9, CYP2C19, CYP3A5, CYP1A2,
            CYP2B6 (whatever apply_phenotype_to_graph accepts). When
            provided, the BodyGraph is rebuilt with phenotype-scaled
            transporter / enzyme abundances before drug binding. Has no
            effect on drugs whose flagged abundance is not consumed by
            any active clearance path.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/integration/test_predict_auto_ecm.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Verify the existing OATP/Engine test suite still green**

Run: `python3 -m pytest tests/integration/test_oatp_pravastatin.py tests/integration/test_oatp_ecm_statins.py tests/integration/test_engine_validation.py -v`
Expected: same pass/xfail counts as before this PR. The signature change is additive (default `None` preserves prior behavior).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/pipeline/predict.py tests/integration/test_predict_auto_ecm.py
git commit -m "feat(predict): add phenotypes= parameter for PGx-aware predictions

Additive kwarg accepts a dict like {\"SLCO1B1\": \"PM\"} or
{\"CYP2D6\": \"PM\"}. When provided, apply_phenotype_to_graph scales
the relevant transporter/enzyme abundance before drug binding.
Default None preserves prior behavior. Compatible with auto-ECM
activation (Task 5) and with n_mc_samples > 0.

Supports all keys apply_phenotype_to_graph accepts today (SLCO1B1 +
six CYPs); future additions (NAT2, UGT1A1 per issue #10) inherit
automatically."
```

---

## Task 7: test_oatp_pravastatin consistency check (predict vs manual)

**Files:**
- Modify: `tests/integration/test_oatp_pravastatin.py`

- [ ] **Step 1: Read the existing manual-build test for the reference Cmax**

Run: `python3 -m pytest tests/integration/test_oatp_pravastatin.py::test_pravastatin_pm_higher_cmax_than_em -v -s`
Note the EM Cmax value printed (~0.0422 mg/L). This is the reference for the consistency check.

- [ ] **Step 2: Add the consistency test**

Append to `tests/integration/test_oatp_pravastatin.py`:

```python
@pytest.mark.slow
def test_predict_auto_ecm_matches_manual():
    """predict(pravastatin) auto-activates ECM and produces the same Cmax
    as the manual build_drug_on_graph(...transporter_kinetics, ...) path
    within 1% relative tolerance.

    Gates the v0.3 wiring: any drift between the two paths means
    predict() is loading different params or skipping a layer.
    """
    from sisyphus.pipeline.predict import predict as pipeline_predict

    graph_em = build_from_yaml(_PHYS)
    drug_em = _build_pravastatin(graph_em)
    cmax_manual = _simulate_cmax(drug_em, graph_em)

    result = pipeline_predict(_PRAVASTATIN, dose_mg=40.0, route="oral")
    cmax_predict = result.engine_pk.cmax.mean

    rel_err = abs(cmax_predict - cmax_manual) / cmax_manual
    assert rel_err < 0.01, (
        f"predict() vs manual ECM build drift: predict={cmax_predict:.5f}, "
        f"manual={cmax_manual:.5f}, rel_err={rel_err:.4f} (>1% tol). The "
        f"two paths should match exactly modulo MC noise (n_mc_samples=0)."
    )
```

- [ ] **Step 3: Run the test to verify it passes**

Run: `python3 -m pytest tests/integration/test_oatp_pravastatin.py::test_predict_auto_ecm_matches_manual -v`
Expected: PASS.

- [ ] **Step 4: Run the full pravastatin suite**

Run: `python3 -m pytest tests/integration/test_oatp_pravastatin.py -v`
Expected: 3 PASS (the original 2 + this new one), 0 fail.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_oatp_pravastatin.py
git commit -m "test(oatp): pin predict() vs manual ECM build consistency

Ensures any future drift between the production predict() auto-ECM
path and the direct build_drug_on_graph manual path surfaces as a
1%-tolerance failure, not a silent divergence."
```

---

## Task 8: Holdout regen + bootstrap CIs

**Files:**
- Regenerate: `data/training/4track_holdout_predictions.json`
- Create: `data/validation/4track_ci_2026-05-03.json` (or `_v2.json` if collision)
- Modify: `tests/regression/test_holdout_regression.py` (pin update)

- [ ] **Step 1: Run the engine benchmark to regenerate the cache**

Run: `python3 scripts/run_engine_benchmark.py 2>&1 | tail -30`
Expected: completes in 5-15 minutes, produces `data/training/4track_holdout_predictions.json` with 107 entries. The pravastatin entry should reflect the new auto-ECM Cmax.

- [ ] **Step 2: Read the new headline AAFE values from the regen output**

Run: `python3 -c "
import json
with open('data/training/4track_holdout_predictions.json') as f:
    data = json.load(f)
print('regen complete; headline metrics:')
for track, metrics in data.get('summary', {}).items():
    print(f'  {track}: {metrics}')
"`
Note the values. If `summary` is not in the JSON, run the bootstrap script directly (next step) and read its output.

- [ ] **Step 3: Refresh bootstrap CIs**

Run: `python3 -c "
import json, numpy as np
np.random.seed(20260422)
with open('data/training/4track_holdout_predictions.json') as f:
    data = json.load(f)
# Compute AAFE per track + bootstrap 10k
# (use the existing CI script if one exists; otherwise inline)
" 2>&1 | tail -20`

If a dedicated script exists (look for `scripts/refresh_bootstrap_ci.py` or similar):
```bash
ls scripts/ | grep -i 'bootstrap\|ci'
```
Run that script. If not, the engine benchmark itself usually produces CI artifacts. Check:
```bash
ls -lt data/validation/4track_ci_*.json | head -5
```
The newest file should be today's. If a `4track_ci_2026-05-03.json` already exists from earlier work today (the OATP triage), the regen will overwrite it — back up first:
```bash
cp data/validation/4track_ci_2026-05-03.json data/validation/4track_ci_2026-05-03_pre_v0.3.json
```
Then regenerate.

- [ ] **Step 4: Verify the AAFE gate (Engine ≤ 2% AND improving)**

Compute the relative change. With pre-v0.3 Engine 3.791 (from CLAUDE.md headline):

```bash
python3 -c "
import json
old = 3.791
with open('data/validation/4track_ci_2026-05-03.json') as f:
    new_data = json.load(f)
# Extract Engine AAFE point estimate; structure may vary
print(json.dumps(new_data, indent=2)[:1000])
"
```
Read the new Engine AAFE. Compute `rel_change = (new - 3.791) / 3.791 * 100`.

**Gate**: `rel_change ≤ 0` (improving) AND `abs(rel_change) ≤ 2%`.

If the gate fails, halt and investigate:
- Engine track worsened → unexpected; check that pravastatin's new Cmax appears correctly in the cache.
- Engine track shifted by > 2% → unrelated drug must have moved (regression bug). Diff `data/training/4track_holdout_predictions.json` against the pre-v0.3 version.

If the gate passes, proceed.

- [ ] **Step 5: Update the holdout regression test pin**

Edit `tests/regression/test_holdout_regression.py`. Find the pinned Meta AAFE value (likely a constant near the top, e.g. `_PINNED_AAFE = 2.679`). Update it to the new Meta AAFE point estimate from Step 4.

- [ ] **Step 6: Run the holdout regression test**

Run: `python3 -m pytest tests/regression/test_holdout_regression.py -v`
Expected: PASS with the new pin.

- [ ] **Step 7: Commit**

```bash
git add data/training/4track_holdout_predictions.json data/validation/4track_ci_2026-05-03.json tests/regression/test_holdout_regression.py
# also include the backup if you made one:
# git add data/validation/4track_ci_2026-05-03_pre_v0.3.json
git commit -m "data(holdout): regen post v0.3 ECM auto-activation

Pravastatin engine Cmax under predict() shifts from XGBoost-CYP-only
(~0.0500) to ECM-active (0.0422), matching FDA 0.045 (FE 1.066).
107 other holdout drugs unchanged (auto-ECM seed list = pravastatin
only).

Headline metrics: <fill in from Step 4 output>
Bootstrap CI artifact: data/validation/4track_ci_2026-05-03.json"
```

(Replace `<fill in from Step 4 output>` with the actual numeric deltas before committing.)

---

## Task 9: Update CLAUDE.md + README.md headline AAFE table

**Files:**
- Modify: `CLAUDE.md` (headline AAFE table near top)
- Modify: `README.md` (matching headline table)

- [ ] **Step 1: Read the new metric values you just committed**

```bash
grep -A 12 "Track | AAFE" CLAUDE.md | head -15
```
Note the current Meta / Engine / ML / In-domain Meta values you need to update.

- [ ] **Step 2: Edit CLAUDE.md headline table**

Open `CLAUDE.md`. Find the table starting with `| Track | AAFE | 95% CI | %2-fold | %3-fold | N |`. Update each cell to the new bootstrap CI values. The Meta row, Engine row, and In-domain Meta row will move; ML row should be invariant.

Also append a new "Current Performance" sub-bullet immediately above the existing 2026-05-02 entry:

```markdown
**2026-05-03 v0.3 ECM auto-activation** (per `docs/claude/experiment-log.md`):
- `pipeline.predict.predict()` now auto-activates the ECM machinery (OATP1B1 saturable + ECM passive + biliary CL_int) for drugs flagged `ecm_applicable: true` in `data/transporters/oatp1b1.json`. Initial seed list: pravastatin only.
- Pravastatin Engine track Cmax: 0.0500 → 0.0422 mg/L (40 mg) → matches FDA Pravachol 0.045 (FE 1.066). 106 other holdout drugs unchanged.
- Engine AAFE shift: 3.791 → <new value> (improving, within bootstrap CI).
- Meta AAFE: 2.679 → <new value> (statistically indistinguishable; ML/blending absorbs the engine improvement).
- Pitavastatin / rosuvastatin / atorvastatin remain unflagged pending literature-curated `metabolic_fraction` entries (per spec §1.3 — empirical validation showed activating ECM without paired `metabolic_fraction` triple-counts hepatic clearance).
- New `predict(phenotypes={"SLCO1B1": "PM"})` parameter exposes PGx-aware predictions to GenoADME and other downstream consumers without requiring manual `apply_phenotype_to_graph` calls.
```

(Fill in `<new value>` placeholders with actual numbers from Task 8 Step 4.)

- [ ] **Step 3: Edit README.md headline table**

Open `README.md`. Find the equivalent metrics block (search for `Meta` or `AAFE` in the file). Update the same cells.

- [ ] **Step 4: Run the full test suite to confirm nothing breaks from doc edits**

Run: `python3 -m pytest -q --ignore=tests/manual 2>&1 | tail -10`
Expected: pass count unchanged from prior tasks; no new failures from doc edits (these don't affect tests but a regression here would suggest you accidentally edited code).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs(claude/readme): v0.3 ECM auto-activation headline update

Pravastatin Engine track Cmax 0.0500 -> 0.0422 mg/L (matches FDA
Pravachol 0.045 at FE 1.066). 106 other holdout drugs unchanged.
Headline AAFE updated to bootstrap CI values from
data/validation/4track_ci_2026-05-03.json.

Pitavastatin / rosuvastatin / atorvastatin defer to v0.3.x follow-up
commits pending literature-curated metabolic_fraction (per spec §1.3
empirical validation: pitavastatin auto-ECM without metabolic_fraction
triple-counts hepatic clearance, FE direction-flips at unchanged
magnitude)."
```

---

## Task 10: Add experiment-log entry + final verification

**Files:**
- Modify: `docs/claude/experiment-log.md` (prepend entry)

- [ ] **Step 1: Append v0.3 entry to the experiment log**

Open `docs/claude/experiment-log.md`. Find the existing 2026-05-02 OATP1B1/ECM reconciliation entry. Add a new entry ABOVE it:

```markdown
## 2026-05-03 — v0.3 ECM auto-activation

**Commit**: <fill in HEAD SHA after Task 10 commit>
**Branch**: main (executed from main per session pattern)
**Spec**: `docs/superpowers/specs/2026-05-03-ecm-auto-activation-design.md`
**Plan**: `docs/superpowers/plans/2026-05-03-ecm-auto-activation.md`

### What shipped

`pipeline.predict.predict()` auto-activates ECM (OATP1B1 saturable + ECM passive + biliary CL_int) for drugs flagged `ecm_applicable: true` in `data/transporters/oatp1b1.json`. Initial seed list: pravastatin only.

Three-layer registry pattern (no engine code changes):
1. `oatp1b1.json` schema extension (`ecm_applicable: bool` per drug, default false).
2. New `is_oatp_ecm_applicable(smiles)`, `load_oatp1b1_kinetics_for_smiles(smiles)`, `load_hepatic_ecm_params_for_smiles(smiles)` helpers in `src/sisyphus/predict/transporter_db.py` (mirrors PR #22 `lookup_metabolic_fraction` pattern).
3. `predict()` reads the flag, conditionally loads kinetics/ECM, passes to `build_drug_on_graph`.
4. Additive `phenotypes: dict[str, str] | None = None` parameter on `predict()` triggers `apply_phenotype_to_graph` before drug binding.

Schema regression test (`tests/regression/test_oatp_registry_schema.py`) gates:
- Seed list pinned (catches silent flag flips)
- Registered InChIKey matches RDKit-canonicalization of SMILES
- Every `ecm_applicable=true` drug has a paired `metabolic_fraction` entry in `cyp_clearance_overrides.json` (prevents pitavastatin-class double-counting bug)

### Numbers

| Metric | Pre-v0.3 (2026-05-02) | Post-v0.3 (2026-05-03) | Δ |
|---|---|---|---|
| Meta AAFE | 2.679 | <fill in> | <fill in>% |
| Engine AAFE | 3.791 | <fill in> | <fill in>% |
| ML AAFE | 3.012 | 3.012 | 0% (invariant) |
| In-domain Meta | 2.733 | <fill in> | <fill in>% |
| Pravastatin engine Cmax (40 mg) | 0.0500 mg/L | 0.0422 mg/L | matches FDA 0.045 at FE 1.066 |

107-holdout drugs other than pravastatin: unchanged.

### Why the seed list is pravastatin only

Empirical pre-spec verification: pitavastatin auto-ECM without a paired `metabolic_fraction` entry in `cyp_clearance_overrides.json` (currently default 1.0) triple-counts hepatic clearance. Cmax 0.00777 → 0.00165 (FE direction flips 2.22× over → 2.12× under, magnitude unchanged). Promotion of pitavastatin/rosuvastatin/atorvastatin requires per-drug literature curation of metabolic_fraction first; tracked as v0.3.x follow-up commits.

### Open follow-ups

- pitavastatin metabolic_fraction curation (~0.15-0.25 estimate; UGT1A3/2B7 + minor CYP2C9; needs primary literature)
- rosuvastatin / atorvastatin: blocked on Peff over-prediction xfail (separate engine work)
- Method routing reassessment via `data/sbi/method_routing.json` (not auto-affected; offline-determined)
- v0.3.x follow-up: PredictionResult metadata fields (`ecm_activated: bool`, `phenotypes_applied: dict`) for GenoADME debugging

### Closes

- (No issues directly closed by this PR; v0.3 is a forward-looking feature.)
- Unblocks GenoADME Tier 1 PGx integration via the new `phenotypes=` API.
```

(Fill in `<fill in>` placeholders with actual numbers.)

- [ ] **Step 2: Run the full test suite**

Run: `python3 -m pytest -q --ignore=tests/manual 2>&1 | tail -15`
Expected: all green except pre-existing xfails (the 3 ECM-statin xfails for rosuvastatin/atorvastatin/fluvastatin, plus any other pre-existing xfails). New tests added across Tasks 1-7 should all pass.

- [ ] **Step 3: Run mass balance specifically**

Run: `python3 -m pytest tests/integration/test_engine_validation.py -v -k mass_balance`
Expected: 4 PASS (midazolam, caffeine, warfarin, propranolol).

- [ ] **Step 4: Commit experiment log**

```bash
git add docs/claude/experiment-log.md
git commit -m "docs(experiment-log): v0.3 ECM auto-activation entry

Documents the three-layer registry pattern, the seed-list rationale,
the empirical pitavastatin verification that justified single-drug
seed, and the bootstrap CI deltas."
```

- [ ] **Step 5: Push to main**

```bash
git push origin main
```
Expected: clean push, CI runs.

- [ ] **Step 6: Verify the schema gate works in CI by inspecting the test**

```bash
gh run list --limit 1
gh run view --log | grep -i "test_oatp_registry_schema\|test_predict_auto_ecm" | head -10
```
Expected: schema test and predict_auto_ecm test both shown as passing in the latest CI run.

---

## Self-Review Checklist (run before declaring complete)

- [ ] **Spec coverage**:
  - §1.1 schema extension → Task 2
  - §1.2 InChIKey policy → Tasks 1, 3, 4 + schema test
  - §1.3 seed list (pravastatin only) → Task 2 + schema test
  - §1.4 expected impact → Task 8 verifies numerically
  - §2 predict() change → Task 5
  - §3 phenotypes parameter → Task 6
  - §4.1 pre-merge gate → Task 8
  - §4.2 AAFE gate (≤2% improving) → Task 8 Step 4
  - §4.3 method routing follow-up → noted in Task 10 follow-ups
  - §4.4 mass balance → Task 5 Step 7 + Task 10 Step 3
  - §6.1 existing tests → Tasks 5, 7
  - §6.2 new test_predict_auto_ecm tests → Tasks 5, 6 (6 of 6 implemented)
  - §6.3 schema regression test → Task 4
  - Logging → Tasks 5, 6
  - CLAUDE.md / README.md update → Task 9
  - Experiment log → Task 10

- [ ] **Type consistency**: helper names match across tasks: `is_oatp_ecm_applicable`, `load_oatp1b1_kinetics_for_smiles`, `load_hepatic_ecm_params_for_smiles`. Constants `_PRAVA_SMILES`, `_FLUVA_SMILES`, `_MIDAZOLAM_SMILES` consistent across test files (they are duplicated rather than imported — acceptable for test isolation, intentional).

- [ ] **No placeholders**: Tasks 8-10 contain `<fill in>` markers for values that the engineer fills in DURING execution from script output. These are NOT placeholder gaps — they are explicit "read the actual number and substitute" instructions, similar to commit-SHA fill-ins. Acceptable.
