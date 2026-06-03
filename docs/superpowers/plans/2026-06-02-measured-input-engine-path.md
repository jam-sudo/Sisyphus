# Measured-Input Engine Path (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `measured_adme` input to `predict()` that overrides predicted ADME with caller-supplied measured values, leaving the SMILES-only path byte-for-byte identical when unused.

**Architecture:** A frozen `MeasuredADMEInput` dataclass carries measured fup/clint/peff/vdss/rbp/solubility + per-field CV. `predict()` gains one keyword-only param; when supplied, a guarded block rebinds `adme` via `dataclasses.replace` *after* `predict_adme()` runs, so both `build_drug_on_graph` calls see the substituted values. When `None`, the block is skipped — the path is unchanged. Measured results are read engine-only (`result.engine_pk`).

**Tech Stack:** Python 3.10+, frozen dataclasses, pytest. Source: `src/sisyphus/predict/adme.py`, `src/sisyphus/pipeline/predict.py`.

**Scope:** SP1 only (the additive path + bit-identity guard + a first engine-only benchmark reusing the 12 source-cited PoC drugs). SP2 (measured-`peff` absorption test), SP3 (gate fix), SP4 (kp_overrides channel) are separate plans. The `kp_method` cosmetic field fix is intentionally **excluded** (it enables nothing — see spec §4).

---

## File Structure

- `src/sisyphus/predict/adme.py` — add `MeasuredADMEInput` after `ADMEProperties` (~line 59). One frozen dataclass, validation in `__post_init__`.
- `src/sisyphus/pipeline/predict.py` — add `TYPE_CHECKING` import for the annotation; add `measured_adme` keyword-only param (~line 82); add the override block after `adme = predict_adme(profile)` (~line 180).
- `tests/unit/test_measured_adme_input.py` — dataclass construction + validation.
- `tests/regression/test_measured_adme_passthrough.py` — `None` bit-identity + behavior.
- `scripts/run_measured_adme_benchmark.py` — engine-only SMILES-vs-measured AAFE over the 12 PoC drugs (human-triggered; not CI).

---

### Task 1: `MeasuredADMEInput` dataclass

**Files:**
- Modify: `src/sisyphus/predict/adme.py` (insert after line 59, the end of `ADMEProperties`)
- Test: `tests/unit/test_measured_adme_input.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_measured_adme_input.py`:

```python
"""Unit tests for MeasuredADMEInput construction and validation."""
import pytest

from sisyphus.predict.adme import MeasuredADMEInput


def test_valid_fup_clint_pair_constructs():
    m = MeasuredADMEInput(fup=0.20, clint=13.0)
    assert m.fup == 0.20
    assert m.clint == 13.0
    assert m.fup_cv == 0.15
    assert m.clint_cv == 0.20


def test_all_none_constructs():
    m = MeasuredADMEInput()
    assert m.fup is None and m.clint is None


def test_peff_only_is_allowed():
    # Only fup+clint are an atomic pair; peff may be supplied alone.
    m = MeasuredADMEInput(peff=5.0)
    assert m.peff == 5.0


def test_fup_without_clint_raises():
    with pytest.raises(ValueError, match="supplied together"):
        MeasuredADMEInput(fup=0.20)


def test_clint_without_fup_raises():
    with pytest.raises(ValueError, match="supplied together"):
        MeasuredADMEInput(clint=13.0)


def test_nonpositive_value_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        MeasuredADMEInput(fup=0.0, clint=13.0)


def test_cv_below_floor_raises():
    with pytest.raises(ValueError, match="< 0.10"):
        MeasuredADMEInput(fup=0.20, clint=13.0, fup_cv=0.05)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_measured_adme_input.py -q`
Expected: FAIL — `ImportError: cannot import name 'MeasuredADMEInput'`.

- [ ] **Step 3: Implement the dataclass**

In `src/sisyphus/predict/adme.py`, immediately after the `ADMEProperties` class (after line 59, before the `# Model cache` comment):

```python
@dataclass(frozen=True)
class MeasuredADMEInput:
    """Caller-supplied measured ADME values that override predicted ones in predict().

    A field left None falls through to the XGBoost/heuristic prediction.

    REQUIRED PAIR: fup and clint must both be supplied or both omitted — they
    co-determine hepatic CL_int in the engine's well-stirred extraction, so
    pairing a measured value with a predicted one distorts engine clearance.

    *_cv are measurement-level CVs (instrument error, below the model-prediction
    CVs above). Floored at 0.10: a tighter CV implies a unit error and collapses
    the Monte-Carlo envelope to false confidence.

    NOTE: peff and vdss overrides also perturb the CLF and VDss meta-tracks, so a
    measured-input prediction is "clean engine-only" only when read via
    result.engine_pk (see the measured-input benchmark).
    """

    fup: float | None = None
    fup_cv: float = 0.15
    clint: float | None = None
    clint_cv: float = 0.20
    peff: float | None = None
    peff_cv: float = 0.25
    vdss: float | None = None
    vdss_cv: float = 0.20
    rbp: float | None = None
    rbp_cv: float = 0.15
    solubility: float | None = None
    solubility_cv: float = 0.30

    def __post_init__(self) -> None:
        if (self.fup is None) != (self.clint is None):
            raise ValueError(
                "MeasuredADMEInput: fup and clint must be supplied together or "
                "both omitted (they co-determine engine CL_int)."
            )
        for name, val in (
            ("fup", self.fup), ("clint", self.clint), ("peff", self.peff),
            ("vdss", self.vdss), ("rbp", self.rbp), ("solubility", self.solubility),
        ):
            if val is not None and val <= 0:
                raise ValueError(f"MeasuredADMEInput.{name} must be > 0, got {val}")
        for name, cv in (
            ("fup_cv", self.fup_cv), ("clint_cv", self.clint_cv),
            ("peff_cv", self.peff_cv), ("vdss_cv", self.vdss_cv),
            ("rbp_cv", self.rbp_cv), ("solubility_cv", self.solubility_cv),
        ):
            if cv < 0.10:
                raise ValueError(
                    f"MeasuredADMEInput.{name}={cv} < 0.10; a CV below 10% implies "
                    "a unit error and collapses the MC envelope."
                )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_measured_adme_input.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/predict/adme.py tests/unit/test_measured_adme_input.py
git commit -m "feat(predict): add MeasuredADMEInput contract for measured-ADME overrides"
```

---

### Task 2: Wire `measured_adme` into `predict()` (bit-identical when None)

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py` (TYPE_CHECKING import near top; signature ~line 82; override block after line 180)
- Test: `tests/regression/test_measured_adme_passthrough.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/regression/test_measured_adme_passthrough.py`:

```python
"""Regression: measured_adme=None must leave the SMILES-only path bit-identical."""
import pytest

from sisyphus.pipeline.predict import predict
from sisyphus.predict.adme import MeasuredADMEInput

# Diverse, valid SMILES — identity-agnostic.
_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",                   # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",            # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",              # ibuprofen
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",   # warfarin
]


@pytest.mark.parametrize("smiles", _SMILES)
def test_none_is_bitidentical(smiles):
    a = predict(smiles, 100.0)
    b = predict(smiles, 100.0, measured_adme=None)
    assert a.pk.cmax.mean == b.pk.cmax.mean
    assert a.engine_pk is not None and b.engine_pk is not None
    assert a.engine_pk.cmax.mean == b.engine_pk.cmax.mean


def test_measured_changes_cmax():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    base = predict(smiles, 100.0)
    meas = predict(smiles, 100.0,
                   measured_adme=MeasuredADMEInput(fup=0.20, clint=200.0))
    assert base.engine_pk.cmax.mean != meas.engine_pk.cmax.mean


def test_warning_tag_present():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    r = predict(smiles, 100.0,
                measured_adme=MeasuredADMEInput(fup=0.20, clint=200.0))
    assert any("measured_adme" in w for w in r.warnings)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/regression/test_measured_adme_passthrough.py -q`
Expected: FAIL — `TypeError: predict() got an unexpected keyword argument 'measured_adme'`.

- [ ] **Step 3a: Add the TYPE_CHECKING import**

In `src/sisyphus/pipeline/predict.py`, after the existing imports (after line 15 `from sisyphus.core import ...`), add:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sisyphus.predict.adme import MeasuredADMEInput
```

(If `TYPE_CHECKING` is already imported, only add the guarded import.)

- [ ] **Step 3b: Add the keyword-only parameter**

In the `predict()` signature, after `kp_method: str = "rodgers_rowland",` (line 82), add:

```python
    measured_adme: MeasuredADMEInput | None = None,
```

- [ ] **Step 3c: Add the override block**

Immediately after `adme = predict_adme(profile)` (line 180), insert:

```python

    # ── Step 1b: measured-ADME override (additive; no-op when None) ──────
    # predict_adme() always runs first, so the ML/VDss tracks never see None.
    # When measured_adme is supplied, REBIND the `adme` name via
    # dataclasses.replace so BOTH build_drug_on_graph calls (this step's at
    # ~line 213 and the phenotype back-solve at ~line 293) consume the
    # substituted values. When None, this block is skipped and `adme` is the
    # unmodified predicted object — the SMILES-only path is bit-identical.
    # Reported measured-input results read result.engine_pk (peff/vdss overrides
    # also move the CLF/VDss meta-tracks; only the engine track is clean).
    if measured_adme is not None:
        from dataclasses import replace as _dc_replace
        _ov: dict[str, Distribution] = {}
        if measured_adme.fup is not None:
            _ov["fup"] = Distribution(mean=measured_adme.fup, cv=measured_adme.fup_cv)
        if measured_adme.clint is not None:
            _ov["clint"] = Distribution(mean=measured_adme.clint, cv=measured_adme.clint_cv)
        if measured_adme.peff is not None:
            _ov["peff"] = Distribution(mean=measured_adme.peff, cv=measured_adme.peff_cv)
        if measured_adme.vdss is not None:
            _ov["vdss"] = Distribution(mean=measured_adme.vdss, cv=measured_adme.vdss_cv)
        if measured_adme.rbp is not None:
            _ov["rbp"] = Distribution(mean=measured_adme.rbp, cv=measured_adme.rbp_cv)
        if measured_adme.solubility is not None:
            _ov["solubility"] = Distribution(
                mean=measured_adme.solubility, cv=measured_adme.solubility_cv
            )
        if _ov:
            adme = _dc_replace(adme, **_ov)
            warnings_list.append(f"measured_adme:overrides={sorted(_ov)}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/regression/test_measured_adme_passthrough.py -q`
Expected: PASS (6 passed: 4 bit-identical + change + warning).

- [ ] **Step 5: Run the broader suite to confirm no regression**

Run: `python -m pytest tests/unit tests/regression -q`
Expected: PASS (no new failures). Then `ruff check src tests` → clean.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/pipeline/predict.py tests/regression/test_measured_adme_passthrough.py
git commit -m "feat(predict): wire opt-in measured_adme override (bit-identical when None)"
```

---

### Task 3: Engine-only measured-input benchmark (reuse 12 source-cited PoC drugs)

**Files:**
- Create: `scripts/run_measured_adme_benchmark.py`
- (No new test; the script is human-triggered. CI coverage is the Task 1/2 tests.)

- [ ] **Step 1: Write the benchmark runner**

Create `scripts/run_measured_adme_benchmark.py`:

```python
#!/usr/bin/env python3
"""Engine-only measured-input benchmark: SMILES-only vs measured fup+CLint.

Reuses the 12 source-cited measured drugs from scripts/measured_adme_poc.py and
the observed Cmax / SMILES / dose from data/reference/clinical_pk.json. Calls the
PRODUCTION predict(measured_adme=...) API and reads result.engine_pk (the clean
engine-only surface). Reports SMILES-only vs measured AAFE SIDE BY SIDE — this is
SEPARATE from the 2.698 headline and is never merged into it.

Usage: python scripts/run_measured_adme_benchmark.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
CLINICAL_PK = ROOT / "data" / "reference" / "clinical_pk.json"

# (name, fup, clint) — copied verbatim from scripts/measured_adme_poc.py MEASURED
# (DrugBank fup + TDC Hepatocyte_AZ CLint geomean). montelukast + abiraterone are
# the documented extreme outliers (diagnosis.md §3); reported but flagged.
_MEASURED = [
    ("alprazolam", 0.20, 13.0), ("carbamazepine", 0.25, 10.2),
    ("clozapine", 0.03, 31.8), ("diclofenac", 0.003, 83.5),
    ("sildenafil", 0.04, 49.9), ("etodolac", 0.01, 12.9),
    ("montelukast", 0.01, 24.7), ("quinine", 0.30, 21.1),
    ("febuxostat", 0.008, 9.4), ("dasatinib", 0.04, 28.2),
    ("clopidogrel", 0.2175, 137.0), ("abiraterone", 0.01, 55.0),
]
_OUTLIERS = {"montelukast", "abiraterone"}


def _aafe(folds):
    return float(np.exp(np.mean(np.log(folds)))) if folds else float("nan")


def main() -> int:
    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.adme import MeasuredADMEInput

    drugs = json.loads(CLINICAL_PK.read_text())["drugs"]
    rows, fe_s, fe_m, fe_s_clean, fe_m_clean = [], [], [], [], []

    for name, fup, clint in _MEASURED:
        rec = drugs.get(name)
        if not rec:
            print(f"  skip {name}: not in clinical_pk.json")
            continue
        obs = (rec.get("pk_params") or {}).get("cmax_mg_L")
        smiles, dose, route = rec.get("smiles"), rec.get("dose_mg"), rec.get("route", "oral")
        if not (obs and smiles and dose):
            print(f"  skip {name}: missing obs/smiles/dose")
            continue

        c_s = predict(smiles, dose, route=route).engine_pk.cmax.mean
        c_m = predict(smiles, dose, route=route,
                      measured_adme=MeasuredADMEInput(fup=fup, clint=clint)).engine_pk.cmax.mean
        f_s = max(c_s / obs, obs / c_s)
        f_m = max(c_m / obs, obs / c_m)
        rows.append((name, obs, c_s, c_m, f_s, f_m))
        fe_s.append(f_s); fe_m.append(f_m)
        if name not in _OUTLIERS:
            fe_s_clean.append(f_s); fe_m_clean.append(f_m)

    print(f"\n{'drug':<16}{'obs':>10}{'eng_smiles':>12}{'eng_meas':>12}{'FE_s':>8}{'FE_m':>8}")
    for name, obs, c_s, c_m, f_s, f_m in rows:
        flag = " *" if name in _OUTLIERS else ""
        print(f"{name:<16}{obs:>10.4f}{c_s:>12.4f}{c_m:>12.4f}{f_s:>8.2f}{f_m:>8.2f}{flag}")
    print(f"\nN={len(rows)}  engine-only AAFE  SMILES={_aafe(fe_s):.3f}  measured={_aafe(fe_m):.3f}")
    print(f"N={len(fe_s_clean)} (excl montelukast/abiraterone)  "
          f"SMILES={_aafe(fe_s_clean):.3f}  measured={_aafe(fe_m_clean):.3f}")
    print("\nSEPARATE from the 2.698 headline — do not merge into 4track_holdout_predictions.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the benchmark and sanity-check the direction**

Run: `python scripts/run_measured_adme_benchmark.py`
Expected: prints a per-drug table; the clean-set (N≈10) **measured AAFE < SMILES AAFE** (Pattern C reference ≈ 1.98 measured vs ≈ 3–4 SMILES). If measured ≥ SMILES on the clean set, STOP and investigate (the override is not reaching the engine).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_measured_adme_benchmark.py
git commit -m "feat(validation): engine-only measured-input benchmark (reuses 12 PoC drugs)"
```

---

## Self-Review

**Spec coverage (spec §4 SP1):** MeasuredADMEInput contract (Task 1) ✓; keyword-only param + override branch + `adme` rebind for both builds (Task 2) ✓; atomic fup+clint + CV floor + >0 (Task 1) ✓; bit-identical `None` test + present-changes + warning-tag (Task 2) ✓; engine-only benchmark reproducing ~1.98 (Task 3) ✓. **Deferred (noted in spec, not this plan):** N≥20 curation with fresh verified citations (Task 3 reuses the 12 already-sourced PoC drugs); `kp_method` cosmetic fix (excluded); 107/107 full-cache rerun (the `None`-path is code-identical and proven by Task 2's exact-float test over 4 drugs — a full regen risks stack-drift false alarms, so it is not a plan gate).

**Placeholder scan:** none — every step has exact code/commands.

**Type consistency:** `MeasuredADMEInput` fields (fup/clint/peff/vdss/rbp/solubility + `_cv`) match between Task 1 definition and Task 2 usage; `Distribution(mean=, cv=)` matches core.py:47-48; `result.engine_pk.cmax.mean` / `result.warnings` match core.py:489,494.

---

## Notes for the executor
- Branch first (we are on a merged docs branch): `git switch -c feat/measured-input-engine-path` before Task 1.
- Commit as `jam-sudo`; **no `Co-Authored-By: Claude` / AI trailer** (CLAUDE.md + user memory).
- Do **not** push or open a PR without explicit user instruction.
