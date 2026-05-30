# ECM Generalization Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute pre-registered ECM generalization test on 3 non-statin OATP1B1 substrates (bosentan, valsartan, repaglinide) under IV dosing, producing a classified outcome (Mode A/B/C/D) per the spec.

**Architecture:** Literature-extracted kinetics + clinical observations are committed to frozen JSON files. A validation script loads the data, runs the existing pipeline (SMILES → DrugOnGraph → engine MC → Cmax) for each drug, and classifies per-drug pass/fail + aggregate mode. No engine or ADME code changes; only new data + new script.

**Tech Stack:** Python 3.10+, existing Sisyphus engine (`propagate_fast` MC API), pytest TDD, scipy (via existing solver), JSON for frozen artifacts.

**Spec reference:** `docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md` (committed `9115e63`).

---

## File Structure

```
data/
  transporters/
    oatp1b1.json              # MODIFY — add 3 drug entries under "drugs"
  validation/
    oatp_generalization_drugs.json   # CREATE — IV dose + observed Cmax + citations
    oatp_generalization_result.json  # CREATE — engine run output (committed post-execution)

scripts/
  validate_oatp_generalization.py   # CREATE — execution driver

src/sisyphus/validation/
  oatp_generalization.py    # CREATE — classifier module (pure functions)

tests/
  unit/
    test_oatp_generalization_data.py  # CREATE — schema + value-range tests for JSON
    test_oatp_generalization_classifier.py  # CREATE — per-drug pass/fail + mode taxonomy
  integration/
    test_oatp_generalization_pipeline.py  # CREATE — 3-drug end-to-end smoke

docs/claude/
  experiment-log.md    # MODIFY — append entry after execution

docs/superpowers/specs/
  2026-04-21-ecm-generalization-test-design.md  # already committed (read-only during execution)
```

Responsibilities:
- `data/transporters/oatp1b1.json` — per-drug Jmax/Km + CV + source (freeze contract)
- `data/validation/oatp_generalization_drugs.json` — per-drug IV dose + observed Cmax + source (freeze contract)
- `src/sisyphus/validation/oatp_generalization.py` — pure classifier logic (pass/fail per drug, mode A/B/C/D aggregation) — separated from script for unit testability
- `scripts/validate_oatp_generalization.py` — orchestration: load data, run MC for each drug, call classifier, emit result JSON
- Tests split: unit for data schema + classifier logic; integration for the pipeline

---

## Task 1: Create `data/validation/oatp_generalization_drugs.json`

**Files:**
- Create: `data/validation/oatp_generalization_drugs.json`
- Create: `tests/unit/test_oatp_generalization_data.py`

**Sub-skill:** Use `superpowers:test-driven-development`.

**Background — exactly what values go into the file.** Each drug needs `dose_mg` (IV), `observed_cmax_mg_l`, `administration` (`iv_bolus` or `iv_infusion_{minutes}`), `smiles`, and `source` (DOI + table/figure reference). Literature lookup required; target values below are best-estimate starting points to verify against primary sources.

Expected value envelopes (plausibility check — if extracted value falls outside these, re-verify the source):

| Drug | IV dose (mg) | Expected Cmax (mg/L, envelope) | Primary source |
|---|---|---|---|
| bosentan | 100 or 250 | 0.5–3.0 | Weber 1996 JCP; cross-check Dingemanse 2003 Clin Pharmacokinet |
| valsartan | 20–160 | 0.3–6.0 | Flesch 1997 Eur J Clin Pharmacol |
| repaglinide | 2 | 0.020–0.080 | Hatorp 2002 Clin Pharmacokinet |

SMILES (hand-curated, canonical protonation state):
- bosentan: `COc1ccccc1Oc1nc(Nc2ncccn2)nc(-c2ccc(C(C)(C)C)cc2)c1S(=O)(=O)Nc1ccc(OC)cc1`
- valsartan: `CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1`
- repaglinide: `CCOc1ccc(CC(=O)N[C@@H](CC(C)C)c2ccccc2N2CCCCC2C)cc1C(=O)O`

- [ ] **Step 1: Write the failing schema test**

Create `tests/unit/test_oatp_generalization_data.py`:

```python
"""Schema and value-envelope tests for the OATP generalization observation file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "validation" / "oatp_generalization_drugs.json"

_EXPECTED_DRUGS = {"bosentan", "valsartan", "repaglinide"}

_CMAX_ENVELOPES = {
    "bosentan": (0.5, 3.0),
    "valsartan": (0.3, 6.0),
    "repaglinide": (0.020, 0.080),
}

_DOSE_ENVELOPES = {
    "bosentan": (100.0, 250.0),
    "valsartan": (20.0, 160.0),
    "repaglinide": (2.0, 2.0),
}


def _load() -> dict:
    assert _DATA_FILE.exists(), f"{_DATA_FILE} does not exist"
    with _DATA_FILE.open() as f:
        return json.load(f)


def test_top_level_schema():
    data = _load()
    assert "drugs" in data
    assert isinstance(data["drugs"], dict)
    assert set(data["drugs"].keys()) == _EXPECTED_DRUGS


@pytest.mark.parametrize("drug", sorted(_EXPECTED_DRUGS))
def test_per_drug_required_fields(drug: str):
    data = _load()
    entry = data["drugs"][drug]
    for field in ("dose_mg", "observed_cmax_mg_l", "administration", "smiles", "source"):
        assert field in entry, f"{drug} missing field {field}"


@pytest.mark.parametrize("drug", sorted(_EXPECTED_DRUGS))
def test_dose_within_envelope(drug: str):
    data = _load()
    dose = float(data["drugs"][drug]["dose_mg"])
    lo, hi = _DOSE_ENVELOPES[drug]
    assert lo <= dose <= hi, f"{drug} dose {dose} outside envelope [{lo}, {hi}]"


@pytest.mark.parametrize("drug", sorted(_EXPECTED_DRUGS))
def test_cmax_within_envelope(drug: str):
    data = _load()
    cmax = float(data["drugs"][drug]["observed_cmax_mg_l"])
    lo, hi = _CMAX_ENVELOPES[drug]
    assert lo <= cmax <= hi, f"{drug} cmax {cmax} outside envelope [{lo}, {hi}]"


@pytest.mark.parametrize("drug", sorted(_EXPECTED_DRUGS))
def test_administration_is_iv(drug: str):
    data = _load()
    admin = data["drugs"][drug]["administration"]
    assert admin.startswith("iv_"), f"{drug} admin {admin!r} must be iv_bolus or iv_infusion_Xmin"


@pytest.mark.parametrize("drug", sorted(_EXPECTED_DRUGS))
def test_source_has_doi(drug: str):
    data = _load()
    src = data["drugs"][drug]["source"]
    assert "doi" in src.lower() or "10." in src, f"{drug} source missing DOI: {src!r}"
```

- [ ] **Step 2: Run test — confirm it fails**

```
pytest tests/unit/test_oatp_generalization_data.py -v
```
Expected: FAIL with "FileNotFoundError" or assertion error (file doesn't exist yet).

- [ ] **Step 3: Create the JSON file with verified literature values**

**Procedure** (each drug requires independent verification):

For each of `bosentan`, `valsartan`, `repaglinide`:
1. Locate the primary source listed in the table above.
2. Read the table/figure that reports IV pharmacokinetic data.
3. Extract: IV dose, observed Cmax (mean across subjects), subject count N, infusion duration if applicable.
4. Convert units if needed: ng/mL → mg/L (divide by 1000), µg/mL → mg/L (equivalent numerically).
5. Cross-check against a review if available (e.g., Kunze 2014 for all three). If primary-value falls >3× from review midpoint, prefer review midpoint and document discrepancy in `notes`.
6. Enter into JSON with `source` = full citation + DOI + exact table/figure reference.

Final file format (example shape — actual values from literature extraction):

```json
{
  "description": "Pre-registered IV clinical observations for ECM generalization test. Committed 2026-04-21. DO NOT MODIFY after engine run — see spec docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md §Freeze Contract.",
  "spec_commit": "9115e63",
  "drugs": {
    "bosentan": {
      "dose_mg": 100.0,
      "observed_cmax_mg_l": 0.9,
      "observed_cmax_cv": 0.30,
      "patient_n": 8,
      "administration": "iv_infusion_15min",
      "smiles": "COc1ccccc1Oc1nc(Nc2ncccn2)nc(-c2ccc(C(C)(C)C)cc2)c1S(=O)(=O)Nc1ccc(OC)cc1",
      "source": "Weber 1996 J Clin Pharmacol 36:1149-1156, Table 1; DOI 10.1002/j.1552-4604.1996.tb04170.x. Single IV 100 mg over 15 min in healthy subjects.",
      "notes": "Cross-checked vs Dingemanse 2003 Clin Pharmacokinet 42:509-516; Cmax consistent."
    },
    "valsartan": {
      "dose_mg": 20.0,
      "observed_cmax_mg_l": 1.2,
      "observed_cmax_cv": 0.35,
      "patient_n": 12,
      "administration": "iv_infusion_60min",
      "smiles": "CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
      "source": "Flesch 1997 Eur J Clin Pharmacol 52:115-120, Table 1; DOI 10.1007/s002280050264. IV 20 mg over 60 min.",
      "notes": ""
    },
    "repaglinide": {
      "dose_mg": 2.0,
      "observed_cmax_mg_l": 0.040,
      "observed_cmax_cv": 0.40,
      "patient_n": 6,
      "administration": "iv_infusion_60min",
      "smiles": "CCOc1ccc(CC(=O)N[C@@H](CC(C)C)c2ccccc2N2CCCCC2C)cc1C(=O)O",
      "source": "Hatorp 2002 Clin Pharmacokinet 41:471-483, Table 2; DOI 10.2165/00003088-200241070-00002. IV 2 mg over 60 min.",
      "notes": ""
    }
  }
}
```

**If literature extraction yields different values than the envelope test expects**, update the envelope test in Step 1 to match the extracted value (with explicit `git diff` shown in the commit message). This is acceptable ONLY if the primary source confirms the divergent value; fishing for values that pass the test is a cherry-picking violation.

- [ ] **Step 4: Re-run tests — confirm they pass**

```
pytest tests/unit/test_oatp_generalization_data.py -v
```
Expected: 21 passes (3 drugs × 7 test parameters).

- [ ] **Step 5: Commit**

```bash
git add data/validation/oatp_generalization_drugs.json tests/unit/test_oatp_generalization_data.py
git commit -m "data(oatp): IV clinical obs for 3 generalization substrates

Pre-registered observations per spec 9115e63. Bosentan/valsartan/repaglinide
IV Cmax extracted from Weber 1996 / Flesch 1997 / Hatorp 2002 respectively.
Schema + value-envelope tests (21 assertions) included."
```

---

## Task 2: Extend `data/transporters/oatp1b1.json` with 3 new drugs

**Files:**
- Modify: `data/transporters/oatp1b1.json`
- Modify: `tests/unit/test_transporter_db.py` (add 3 tests — existing style)

**Background — extraction sources.**

| Drug | OATP1B1 Km (expected, µM) | OATP1B1 Jmax (expected, pmol/min/mg) | Source |
|---|---|---|---|
| bosentan | 40–50 | 40–80 | Treiber 2007 DMD 35:1400–1407, Table 2 |
| valsartan | 1.0–1.8 | 30–80 | Yamashiro 2006 DMD 34:1247–1254, Fig 3/Table 2 |
| repaglinide | 0.3–0.5 | 20–50 | Niemi 2005 CPT 77:468–478 or Bidstrup 2003 Br J Clin Pharmacol 56:305–314 |

CV propagation: literature CV ≥ 0.30 for Jmax, ≥ 0.25 for Km. If reported CV is smaller, widen to 0.40/0.35 (conservative direction per spec §Literature Data Extraction Protocol).

- [ ] **Step 1: Write failing unit tests for the 3 new drug entries**

Append to `tests/unit/test_transporter_db.py`:

```python
import pytest as _pytest

_GENERALIZATION_DRUGS_EXPECTED = {
    "bosentan": {"km_range_uM": (40.0, 50.0), "jmax_range": (40.0, 80.0)},
    "valsartan": {"km_range_uM": (1.0, 1.8), "jmax_range": (30.0, 80.0)},
    "repaglinide": {"km_range_uM": (0.3, 0.5), "jmax_range": (20.0, 50.0)},
}


@_pytest.mark.parametrize("drug", sorted(_GENERALIZATION_DRUGS_EXPECTED.keys()))
def test_generalization_drug_has_entry(drug: str):
    kinetics = load_oatp1b1_kinetics(drug)
    assert kinetics is not None, f"{drug} missing from oatp1b1.json"
    assert "OATP1B1" in kinetics


@_pytest.mark.parametrize("drug", sorted(_GENERALIZATION_DRUGS_EXPECTED.keys()))
def test_generalization_drug_km_in_envelope(drug: str):
    kinetics = load_oatp1b1_kinetics(drug)
    km = kinetics["OATP1B1"].km.mean
    lo, hi = _GENERALIZATION_DRUGS_EXPECTED[drug]["km_range_uM"]
    assert lo <= km <= hi, f"{drug} Km {km} outside [{lo}, {hi}] uM"


@_pytest.mark.parametrize("drug", sorted(_GENERALIZATION_DRUGS_EXPECTED.keys()))
def test_generalization_drug_jmax_in_envelope(drug: str):
    kinetics = load_oatp1b1_kinetics(drug)
    jmax = kinetics["OATP1B1"].jmax.mean
    lo, hi = _GENERALIZATION_DRUGS_EXPECTED[drug]["jmax_range"]
    assert lo <= jmax <= hi, f"{drug} Jmax {jmax} outside [{lo}, {hi}] pmol/min/mg"


@_pytest.mark.parametrize("drug", sorted(_GENERALIZATION_DRUGS_EXPECTED.keys()))
def test_generalization_drug_cv_widened(drug: str):
    kinetics = load_oatp1b1_kinetics(drug)
    assert kinetics["OATP1B1"].jmax.cv >= 0.30, f"{drug} Jmax CV must be >= 0.30"
    assert kinetics["OATP1B1"].km.cv >= 0.25, f"{drug} Km CV must be >= 0.25"
```

- [ ] **Step 2: Run tests — confirm they fail**

```
pytest tests/unit/test_transporter_db.py -v
```
Expected: 12 new tests FAIL (KeyError or None for the 3 new drugs).

- [ ] **Step 3: Extend `data/transporters/oatp1b1.json`**

**Procedure:**
1. Read primary source for each drug (Treiber 2007 / Yamashiro 2006 / Niemi 2005).
2. Locate Jmax and Km in Table/Figure.
3. Apply scaling if needed: Treiber 2007 reports pmol/min/mg transfected cell protein — scale to per-mg microsomal protein via Kunze 2014 RAF ~0.5 (i.e., divide by 2). Yamashiro 2006 uses HEK293-OATP1B1 similar basis — confirm scaling needed.
4. Append entry under `drugs` key.
5. Widen CV if reported < 0.30 (Jmax) or < 0.25 (Km).

Example final state (actual values from extraction):

```json
{
  "transporter": "OATP1B1",
  "source_primary": "Varma 2014 JPET (pravastatin); Niemi 2009 review Km compilation (others, statins); Treiber 2007 / Yamashiro 2006 / Niemi 2005 (generalization drugs)",
  "source_crosscheck": "...",
  "notes": "...",
  "drugs": {
    "pravastatin": { ... existing ... },
    "rosuvastatin": { ... existing ... },
    "atorvastatin": { ... existing ... },
    "pitavastatin": { ... existing ... },
    "fluvastatin": { ... existing ... },
    "bosentan": {
      "jmax_pmol_per_min_per_mg": {"mean": 62.0, "cv": 0.40},
      "km_uM": {"mean": 44.3, "cv": 0.30},
      "source": "Treiber 2007 DMD 35:1400-1407 Table 2; HEK293-OATP1B1 uptake assay. DOI 10.1124/dmd.107.015230."
    },
    "valsartan": {
      "jmax_pmol_per_min_per_mg": {"mean": 50.0, "cv": 0.40},
      "km_uM": {"mean": 1.4, "cv": 0.35},
      "source": "Yamashiro 2006 DMD 34:1247-1254 Fig 3; HEK293-OATP1B1 uptake. DOI 10.1124/dmd.106.009365."
    },
    "repaglinide": {
      "jmax_pmol_per_min_per_mg": {"mean": 35.0, "cv": 0.40},
      "km_uM": {"mean": 0.40, "cv": 0.35},
      "source": "Niemi 2005 CPT 77:468-478; HEK293-OATP1B1 Km consistent with Bidstrup 2003 Br J Clin Pharmacol 56:305-314. DOI 10.1016/j.clpt.2005.01.006."
    }
  }
}
```

**IMPORTANT:** `_load_oatp1b1_table` uses `functools.lru_cache`. If the JSON is edited during an interactive python session, clear the cache via `_load_oatp1b1_table.cache_clear()`. In pytest this is not an issue (fresh process per run).

- [ ] **Step 4: Re-run tests — confirm they pass**

```
pytest tests/unit/test_transporter_db.py -v
```
Expected: existing pravastatin tests + 12 new tests all PASS.

- [ ] **Step 5: Commit**

```bash
git add data/transporters/oatp1b1.json tests/unit/test_transporter_db.py
git commit -m "data(transporters): OATP1B1 Jmax/Km for bosentan/valsartan/repaglinide

Pre-registered kinetics per spec 9115e63. Extracted from Treiber 2007 /
Yamashiro 2006 / Niemi 2005. CV widened per spec: Jmax >= 0.40, Km >= 0.30.
Value-envelope tests (12) gate correctness."
```

---

## Task 3: Build classifier module (pure functions, unit-tested)

**Files:**
- Create: `src/sisyphus/validation/oatp_generalization.py`
- Create: `tests/unit/test_oatp_generalization_classifier.py`

**Responsibility:** Pure functions for per-drug pass/fail + aggregate mode classification. Zero I/O, zero engine calls. Script calls into this module with numbers. Separation enables fast unit testing.

- [ ] **Step 1: Write failing unit tests**

Create `tests/unit/test_oatp_generalization_classifier.py`:

```python
"""Unit tests for ECM generalization test classifier.

Tests pure classifier logic per spec docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md.
"""

from __future__ import annotations

import math

import pytest

from sisyphus.validation.oatp_generalization import (
    DrugOutcome,
    Mode,
    classify_aggregate,
    classify_drug,
)


# ---------- per-drug classification ----------

def test_pass_when_both_conditions_met():
    """Drug passes iff 90% PI contains obs AND |log10 FE| <= 0.48."""
    out = classify_drug(
        drug="x",
        observed=1.0,
        point_estimate=1.2,  # FE = 1.2, log10 FE = 0.079, passes FE gate
        pi_low=0.8,
        pi_high=1.5,  # contains obs=1.0
    )
    assert out.passed is True
    assert out.log10_fe == pytest.approx(math.log10(1.2))


def test_fail_when_point_estimate_out_of_fe_gate():
    """FE = 3.5 (log10 = 0.544 > 0.48) → fail, even if PI contains obs."""
    out = classify_drug(
        drug="x",
        observed=1.0,
        point_estimate=3.5,
        pi_low=0.5,
        pi_high=10.0,  # contains obs
    )
    assert out.passed is False


def test_fail_when_pi_does_not_contain_obs():
    """PI must contain observed."""
    out = classify_drug(
        drug="x",
        observed=1.0,
        point_estimate=0.5,
        pi_low=0.3,
        pi_high=0.7,  # does NOT contain obs=1.0
    )
    assert out.passed is False


def test_fe_gate_boundary_inclusive():
    """|log10 FE| = 0.48 exactly passes the gate (FE = 3.02 approx)."""
    fe_boundary = 10 ** 0.48
    out = classify_drug(
        drug="x",
        observed=1.0,
        point_estimate=fe_boundary,
        pi_low=0.1,
        pi_high=100.0,
    )
    assert out.passed is True


# ---------- aggregate mode classification ----------

def _make(obs: float, pred: float, pi: tuple[float, float], name: str = "d") -> DrugOutcome:
    return classify_drug(name, obs, pred, pi[0], pi[1])


def test_mode_A_when_all_pass():
    outcomes = [
        _make(1.0, 1.1, (0.5, 2.0), "a"),
        _make(1.0, 0.9, (0.5, 2.0), "b"),
        _make(1.0, 1.2, (0.5, 2.0), "c"),
    ]
    assert classify_aggregate(outcomes) == Mode.A


def test_mode_B_two_fail_same_direction_large_magnitude():
    """2/3 fail, both over-predict, median log10 FE > 0.5 → Mode B."""
    outcomes = [
        _make(1.0, 1.1, (0.5, 2.0), "pass"),
        _make(1.0, 5.0, (2.0, 12.0), "over1"),  # log10 FE = 0.699
        _make(1.0, 6.0, (3.0, 15.0), "over2"),  # log10 FE = 0.778
    ]
    assert classify_aggregate(outcomes) == Mode.B


def test_mode_B_all_fail_same_direction():
    """3/3 fail, same direction → Mode B (systematic), not Mode D."""
    outcomes = [
        _make(1.0, 5.0, (2.0, 12.0), "o1"),
        _make(1.0, 6.0, (3.0, 15.0), "o2"),
        _make(1.0, 4.0, (1.5, 10.0), "o3"),
    ]
    assert classify_aggregate(outcomes) == Mode.B


def test_mode_C_mixed_direction_failures():
    """2/3 fail, one over + one under → Mode C."""
    outcomes = [
        _make(1.0, 1.1, (0.5, 2.0), "pass"),
        _make(1.0, 5.0, (2.0, 12.0), "over"),
        _make(1.0, 0.2, (0.05, 0.5), "under"),
    ]
    assert classify_aggregate(outcomes) == Mode.C


def test_mode_C_single_failure():
    """Single failure regardless of magnitude → Mode C."""
    outcomes = [
        _make(1.0, 1.1, (0.5, 2.0), "pass1"),
        _make(1.0, 0.9, (0.5, 2.0), "pass2"),
        _make(1.0, 5.0, (2.0, 12.0), "fail"),
    ]
    assert classify_aggregate(outcomes) == Mode.C


def test_mode_C_two_same_direction_small_magnitude():
    """2/3 fail, same direction, but median |log10 FE| ≤ 0.5 → Mode C (not B).

    Both failures have |log10 FE| just above the 0.48 pass gate but the median
    is below 0.5 — below the Mode B magnitude threshold.
    """
    outcomes = [
        _make(1.0, 1.1, (0.5, 2.0), "pass"),
        _make(1.0, 3.05, (1.2, 7.0), "f1"),  # log10 FE ~ 0.484 (fails gate, below 0.5)
        _make(1.0, 3.10, (1.2, 7.0), "f2"),  # log10 FE ~ 0.491
    ]
    assert classify_aggregate(outcomes) == Mode.C


def test_mode_D_all_fail_mixed_direction():
    """3/3 fail, mixed directions → Mode D."""
    outcomes = [
        _make(1.0, 5.0, (2.0, 12.0), "over"),
        _make(1.0, 6.0, (3.0, 15.0), "over2"),
        _make(1.0, 0.15, (0.05, 0.4), "under"),
    ]
    assert classify_aggregate(outcomes) == Mode.D


def test_precedence_A_over_B():
    """All-pass never reaches Mode B check."""
    outcomes = [
        _make(1.0, 1.0, (0.5, 2.0), "a"),
        _make(1.0, 1.0, (0.5, 2.0), "b"),
        _make(1.0, 1.0, (0.5, 2.0), "c"),
    ]
    assert classify_aggregate(outcomes) == Mode.A
```

- [ ] **Step 2: Run tests — confirm they fail**

```
pytest tests/unit/test_oatp_generalization_classifier.py -v
```
Expected: all FAIL with `ImportError: cannot import name 'DrugOutcome' from 'sisyphus.validation.oatp_generalization'`.

- [ ] **Step 3: Implement the classifier**

Create `src/sisyphus/validation/oatp_generalization.py`:

```python
"""ECM generalization test classifier.

Pure functions for pre-registered pass/fail logic per spec
docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md.

Separation from the execution script enables fast unit tests and keeps
the classification logic frozen independently of orchestration changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


_FE_GATE_LOG10 = 0.48  # |log10 FE| <= 0.48 iff FE <= 3.02
_MODE_B_MAGNITUDE = 0.5  # |median log10 FE of failures| > 0.5 for Mode B


class Mode(str, Enum):
    """Aggregate outcome taxonomy."""

    A = "A"  # all-pass
    B = "B"  # systematic bias (same-direction fail, magnitude > 0.5)
    C = "C"  # inconclusive (default)
    D = "D"  # all-fail mixed direction


@dataclass(frozen=True)
class DrugOutcome:
    """Per-drug classification result."""

    drug: str
    observed: float
    point_estimate: float
    pi_low: float
    pi_high: float
    log10_fe: float
    passed: bool


def classify_drug(
    drug: str,
    observed: float,
    point_estimate: float,
    pi_low: float,
    pi_high: float,
) -> DrugOutcome:
    """Classify one drug as pass/fail per spec §Per-drug criterion.

    Passes iff:
    1. 90% PI contains observed.
    2. |log10 FE| <= 0.48 (= FE <= 3.02x).
    """
    if observed <= 0 or point_estimate <= 0:
        raise ValueError("observed and point_estimate must be positive")
    log10_fe = math.log10(point_estimate / observed)
    pi_contains = pi_low <= observed <= pi_high
    fe_ok = abs(log10_fe) <= _FE_GATE_LOG10
    return DrugOutcome(
        drug=drug,
        observed=observed,
        point_estimate=point_estimate,
        pi_low=pi_low,
        pi_high=pi_high,
        log10_fe=log10_fe,
        passed=pi_contains and fe_ok,
    )


def classify_aggregate(outcomes: list[DrugOutcome]) -> Mode:
    """Classify the set of per-drug outcomes into Mode A/B/C/D.

    Precedence: A → B → D → C (C is the fallback).

    - A: all pass.
    - B: (>=2 failures with same-direction log10 FE) AND
         |median log10 FE of failures| > 0.5. Includes 3/3 same-direction fail.
    - D: 3/3 fail AND failures are NOT same-direction (mixed signs).
    - C: everything else.
    """
    n = len(outcomes)
    failures = [o for o in outcomes if not o.passed]
    n_fail = len(failures)

    if n_fail == 0:
        return Mode.A

    fail_signs = {1 if o.log10_fe > 0 else -1 for o in failures}
    same_direction = len(fail_signs) == 1

    if n_fail >= 2 and same_direction:
        median_log10_fe = _median([abs(o.log10_fe) for o in failures])
        if median_log10_fe > _MODE_B_MAGNITUDE:
            return Mode.B

    if n_fail == n and not same_direction:
        return Mode.D

    return Mode.C


def _median(values: list[float]) -> float:
    s = sorted(values)
    k = len(s)
    if k == 0:
        return 0.0
    if k % 2 == 1:
        return s[k // 2]
    return 0.5 * (s[k // 2 - 1] + s[k // 2])
```

- [ ] **Step 4: Re-run tests — confirm they pass**

```
pytest tests/unit/test_oatp_generalization_classifier.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/validation/oatp_generalization.py tests/unit/test_oatp_generalization_classifier.py
git commit -m "feat(validation): ECM generalization test classifier

Pure functions for per-drug pass/fail + Mode A/B/C/D aggregation per spec
9115e63 §Pass/Fail Criteria. Precedence A->B->D->C. Unit tests cover all
4 modes + boundary conditions."
```

---

## Task 4: Build execution script

**Files:**
- Create: `scripts/validate_oatp_generalization.py`

No TDD on this file directly — it's orchestration glue with no pure logic. Correctness gated by the integration test in Task 5.

- [ ] **Step 1: Write the script**

Create `scripts/validate_oatp_generalization.py`:

```python
#!/usr/bin/env python3
"""ECM generalization test — engine execution driver.

Pre-registered per spec docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md.

For each of 3 non-statin OATP1B1 substrates:
1. Load IV dose + observed Cmax from data/validation/oatp_generalization_drugs.json
2. Load frozen Jmax/Km from data/transporters/oatp1b1.json
3. Build DrugOnGraph with route=iv, administration_node=venous_blood
4. Propagate uncertainty (N=1000 MC samples, fast mode)
5. Classify per-drug pass/fail, aggregate Mode A/B/C/D

Writes data/validation/oatp_generalization_result.json.

Usage:
    python scripts/validate_oatp_generalization.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sisyphus.engine.flux  # noqa: F401,E402 -- register flux specs
from sisyphus.engine.compiler import ODECompiler  # noqa: E402
from sisyphus.engine.uncertainty import UncertaintyEngine  # noqa: E402
from sisyphus.graph.builder import build_from_yaml  # noqa: E402
from sisyphus.predict.adme import predict_adme  # noqa: E402
from sisyphus.predict.chemistry import compute_profile  # noqa: E402
from sisyphus.predict.ivive import build_drug_on_graph  # noqa: E402
from sisyphus.predict.transporter_db import (  # noqa: E402
    load_hepatic_ecm_params,
    load_oatp1b1_kinetics,
)
from sisyphus.validation.oatp_generalization import (  # noqa: E402
    Mode,
    classify_aggregate,
    classify_drug,
)

_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"
_OBS_FILE = ROOT / "data" / "validation" / "oatp_generalization_drugs.json"
_OUT = ROOT / "data" / "validation" / "oatp_generalization_result.json"

import os as _os

# Frozen per spec §Execution Constraints. SMOKE mode for Task 4 Step 2 only
# (writes to .smoke.json, not the pre-registered result path).
_IS_SMOKE = _os.environ.get("SISYPHUS_OATP_GEN_SMOKE") == "1"
_MC_N_SAMPLES = 10 if _IS_SMOKE else 1000


def _run_one(name: str, entry: dict, graph, liver_enzymes: dict) -> dict:
    """Run MC for one drug. Returns per-drug result dict."""
    smiles = entry["smiles"]
    dose_mg = float(entry["dose_mg"])
    observed = float(entry["observed_cmax_mg_l"])

    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    oatp_kinetics = load_oatp1b1_kinetics(name)
    ecm_params = load_hepatic_ecm_params(name)

    drug = build_drug_on_graph(
        profile,
        adme,
        dose_mg=dose_mg,
        route="iv",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=oatp_kinetics,
        hepatic_ecm_params=ecm_params,
    )

    compiler = ODECompiler()
    compiled = compiler.compile(graph)
    ue = UncertaintyEngine()

    t0 = time.time()
    mc = ue.propagate_fast(
        compiled=compiled,
        graph=graph,
        drug=drug,
        n_samples=_MC_N_SAMPLES,
        seed=42,
        t_span=(0.0, 24.0),
        observation_node="venous_blood",
    )
    elapsed = time.time() - t0

    cmax_samples = mc.cmax_samples
    point_estimate = float(np.median(cmax_samples))  # spec: median per §P6 morphine pattern
    pi_low, pi_high = mc.cmax_90ci

    outcome = classify_drug(name, observed, point_estimate, pi_low, pi_high)

    confound = {
        "fup_predicted": float(adme.fup.mean),
        "rbp_predicted": float(adme.rbp.mean),
        "kp_liver_predicted": float(adme.kp_overrides.get("liver", adme.kp_overrides.get("_default", type("X", (), {"mean": float("nan")}))).mean)
            if adme.kp_overrides else float("nan"),
        "notes": "Flag drugs where fup/rbp/kp >3x off published value — predict-layer confound. Manual check required.",
    }

    return {
        "drug": name,
        "dose_mg": dose_mg,
        "observed_cmax_mg_l": observed,
        "point_estimate_cmax_mg_l": point_estimate,
        "pi_90_low_mg_l": float(pi_low),
        "pi_90_high_mg_l": float(pi_high),
        "log10_fe": outcome.log10_fe,
        "passed": outcome.passed,
        "wall_seconds": elapsed,
        "mc_n_samples": int(mc.n_samples),
        "mc_n_failures": int(mc.n_failures),
        "predict_layer_confound_diagnostics": confound,
    }


def main() -> None:
    with _OBS_FILE.open() as f:
        obs_data = json.load(f)

    graph = build_from_yaml(_PHYS)
    liver_enzymes = {
        tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
    }

    drug_results = []
    outcomes = []
    for name in sorted(obs_data["drugs"].keys()):
        entry = obs_data["drugs"][name]
        print(f"\n[{name}] dose={entry['dose_mg']}mg iv observed={entry['observed_cmax_mg_l']} mg/L")
        result = _run_one(name, entry, graph, liver_enzymes)
        drug_results.append(result)
        outcomes.append(
            classify_drug(
                name,
                result["observed_cmax_mg_l"],
                result["point_estimate_cmax_mg_l"],
                result["pi_90_low_mg_l"],
                result["pi_90_high_mg_l"],
            )
        )
        print(f"  point={result['point_estimate_cmax_mg_l']:.4f} "
              f"PI=[{result['pi_90_low_mg_l']:.4f}, {result['pi_90_high_mg_l']:.4f}] "
              f"log10_FE={result['log10_fe']:+.3f} passed={result['passed']} "
              f"wall={result['wall_seconds']:.1f}s")

    mode = classify_aggregate(outcomes)
    print(f"\n=== Aggregate Mode: {mode.value} ===")

    report = {
        "spec": "docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md",
        "spec_commit": "9115e63",
        "mc_n_samples": _MC_N_SAMPLES,
        "drugs": drug_results,
        "aggregate_mode": mode.value,
        "mode_descriptions": {
            "A": "All-pass. ECM is confirmed as a general mechanism within domain.",
            "B": "Systematic bias. Same-direction failures with |median log10 FE| > 0.5.",
            "C": "Inconclusive. Fallback for patterns not matching A/B/D.",
            "D": "All-fail mixed. ECM = statin-specialized; architecture review required.",
        },
        "notes": "Single run per spec §Execution Constraints. No post-run parameter adjustment.",
    }

    out_path = _OUT.with_suffix(".smoke.json") if _IS_SMOKE else _OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")
    if _IS_SMOKE:
        print("(smoke run — not the pre-registered result; delete before Task 6)")


if __name__ == "__main__":
    main()
```

**Note on `build_drug_on_graph` signature:** verify that it accepts `route="iv"` + `transporter_kinetics` + `hepatic_ecm_params` kwargs. Check `src/sisyphus/predict/ivive.py` before running. If the signature is different, update the script accordingly — do NOT modify `ivive.py` itself (freeze contract).

**Note on `adme.kp_overrides` shape:** this is defensive — different versions of ADMEProperties may store Kp as dict or as attr. If `kp_overrides` is not a dict or does not have a "liver" key, confound diagnostic will be NaN and log-only; it is not gating.

- [ ] **Step 2: Smoke-run the script (N=10, separate output path)**

Before the pre-registered run, verify script plumbing with N=10 samples. The smoke mode writes to `oatp_generalization_result.smoke.json` (NOT the pre-registered path) to prevent the working tree from containing a wrong-N result when Task 6 checks.

```
SISYPHUS_OATP_GEN_SMOKE=1 python scripts/validate_oatp_generalization.py
```
Expected: ~30s total (3 drugs × ~10s per drug at N=10), writes `data/validation/oatp_generalization_result.smoke.json`, prints Mode classification.

If it crashes, fix the script (not the data or engine). Common issues:
- `build_drug_on_graph` signature mismatch — inspect `src/sisyphus/predict/ivive.py`
- `adme.kp_overrides` schema difference — inspect `src/sisyphus/predict/adme.py`
- Solver stall on any drug — widen `t_span` to `(0.0, 48.0)` in the script (will persist to the real run; that's OK if justified)

After smoke passes, delete the smoke artifact:
```
rm data/validation/oatp_generalization_result.smoke.json
```

- [ ] **Step 3: Commit the script (NOT the result file)**

```bash
git add scripts/validate_oatp_generalization.py
git commit -m "script(oatp): ECM generalization test driver

Loads frozen kinetics + clinical obs per spec 9115e63, runs MC N=1000
per drug, classifies outcomes via src/sisyphus/validation/oatp_generalization.
No result artifact committed in this step — see Task 6."
```

---

## Task 5: Integration test for the pipeline

**Files:**
- Create: `tests/integration/test_oatp_generalization_pipeline.py`

**Purpose:** Smoke-test that each of the 3 drugs can run end-to-end through the pipeline without errors (MC returns valid numbers). Does NOT assert on pass/fail — that's the live experiment.

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_oatp_generalization_pipeline.py`:

```python
"""Integration smoke test: 3 ECM generalization drugs run end-to-end.

Uses N_SAMPLES=10 for speed; the full pre-registered run uses N=1000.
Assertions are "did it run, did it produce finite numbers" — NOT pass/fail
(that is the actual experiment, only run via scripts/validate_oatp_generalization.py).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.engine.compiler import ODECompiler
from sisyphus.engine.uncertainty import UncertaintyEngine
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph
from sisyphus.predict.transporter_db import (
    load_hepatic_ecm_params,
    load_oatp1b1_kinetics,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"
_OBS_FILE = ROOT / "data" / "validation" / "oatp_generalization_drugs.json"

_DRUGS = ["bosentan", "valsartan", "repaglinide"]


@pytest.fixture(scope="module")
def obs_data() -> dict:
    with _OBS_FILE.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def graph_and_enzymes():
    graph = build_from_yaml(_PHYS)
    liver_enzymes = {tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()}
    return graph, liver_enzymes


@pytest.mark.parametrize("drug_name", _DRUGS)
def test_pipeline_end_to_end_smoke(drug_name: str, obs_data: dict, graph_and_enzymes):
    """MC(N=10) completes, returns finite Cmax, valid 90% PI."""
    graph, liver_enzymes = graph_and_enzymes
    entry = obs_data["drugs"][drug_name]

    profile = compute_profile(entry["smiles"])
    adme = predict_adme(profile)
    drug = build_drug_on_graph(
        profile,
        adme,
        dose_mg=float(entry["dose_mg"]),
        route="iv",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=load_oatp1b1_kinetics(drug_name),
        hepatic_ecm_params=load_hepatic_ecm_params(drug_name),
    )

    compiled = ODECompiler().compile(graph)
    ue = UncertaintyEngine()
    mc = ue.propagate_fast(
        compiled=compiled, graph=graph, drug=drug,
        n_samples=10, seed=42, t_span=(0.0, 24.0),
        observation_node="venous_blood",
    )

    # Structural checks, not scientific assertions
    assert mc.n_samples >= 8, f"{drug_name}: too many MC failures ({mc.n_failures})"
    assert np.all(np.isfinite(mc.cmax_samples))
    assert np.all(mc.cmax_samples > 0)
    pi_low, pi_high = mc.cmax_90ci
    assert pi_low < pi_high
    assert pi_low > 0
```

- [ ] **Step 2: Run test — confirm it fails (or fails for expected reason)**

```
pytest tests/integration/test_oatp_generalization_pipeline.py -v
```
Expected: FAIL if data files from Tasks 1-2 are not yet committed (expected). After Tasks 1-2 are complete, this test should PASS directly (script under test, Task 4, doesn't need to be written for THIS integration test to pass — the test uses the same APIs directly).

- [ ] **Step 3: Confirm tests pass (after Tasks 1-2 complete + script verified in Task 4 Step 2)**

```
pytest tests/integration/test_oatp_generalization_pipeline.py -v
```
Expected: 3 PASS (one per drug).

If any drug fails with solver error, do NOT proceed to Task 6. Investigate:
- Check `predict_adme` output for NaNs
- Check `build_drug_on_graph` accepts all kwargs
- Check `reference_man.yaml` has OATP1B1 abundance (post-ECM value 5.0e5)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_oatp_generalization_pipeline.py
git commit -m "test(integration): ECM generalization pipeline smoke

3 drugs × MC(N=10) through predict→engine. Structural assertions only —
actual pass/fail is via scripts/validate_oatp_generalization.py (Task 6)."
```

---

## Task 6: Pre-registered execution + result commit

**Files:**
- Create: `data/validation/oatp_generalization_result.json`

**This is the single authorized engine run.** Any failure to complete this task (solver stall, NaN, etc.) means investigation followed by a new run, with BOTH runs documented in the final report.

- [ ] **Step 1: Verify all frozen artifacts are committed**

```
git status
git log --oneline -5
```
Expected: clean working tree. Last commits include the 3 commits from Tasks 1, 2, 3, 4, 5 (data + classifier + script + tests).

If working tree is dirty, stash or commit before proceeding. **Running with uncommitted changes is a freeze-contract violation.**

- [ ] **Step 2: Execute the pre-registered run**

```
python scripts/validate_oatp_generalization.py 2>&1 | tee /tmp/oatp_gen_stdout.log
```
Expected: ~10-20 minutes wall time total. Mode classification printed at end.

Record the printed Mode and per-drug outcomes. **Do not run again after this.**

- [ ] **Step 3: Verify result file is well-formed**

```python
python -c "
import json, pathlib
p = pathlib.Path('data/validation/oatp_generalization_result.json')
d = json.loads(p.read_text())
assert d['mc_n_samples'] == 1000
assert len(d['drugs']) == 3
assert d['aggregate_mode'] in ('A', 'B', 'C', 'D')
print('OK:', d['aggregate_mode'], [r['passed'] for r in d['drugs']])
"
```
Expected: `OK: <mode> [<pass1>, <pass2>, <pass3>]`

- [ ] **Step 4: Commit the result**

```bash
git add data/validation/oatp_generalization_result.json
git commit -m "result(oatp): ECM generalization test — Mode <X>

Per-drug: bosentan=<pass|fail>, valsartan=<pass|fail>, repaglinide=<pass|fail>.
Single pre-registered run per spec 9115e63 §Execution Constraints.
MC N=1000, seed=42. No post-run parameter adjustment."
```

Replace `<X>` with the actual mode from Step 2, and `<pass|fail>` with per-drug outcomes.

---

## Task 7: Report + memory update

**Files:**
- Modify: `docs/claude/experiment-log.md`
- Create: `.claude-memory/project_ecm_generalization_test.md` (user memory path)

- [ ] **Step 1: Append entry to experiment-log.md**

Prepend (at top of file, under any existing header) an entry following the existing format:

```markdown
## 2026-04-21 — ECM generalization test on non-statin OATP1B1 substrates

**Spec:** `docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md` (commit 9115e63)
**Plan:** `docs/superpowers/plans/2026-04-21-ecm-generalization-test.md`
**Result:** `data/validation/oatp_generalization_result.json`

**Outcome:** Mode <X>

**Per drug:**
- Bosentan: observed <obs> mg/L, point <point> mg/L, PI [<lo>, <hi>], log10 FE <fe>, passed=<bool>
- Valsartan: (same format)
- Repaglinide: (same format)

**Interpretation** (tailor to Mode):
- Mode A: ECM is confirmed as a general OATP1B1 uptake mechanism within the declared domain. Claim promoted to landmarks.md.
- Mode B: Systematic bias, direction = <over|under>, |median log10 FE among failures| = <val>. Post-hoc investigation authorized; direction-matched hypothesis: <e.g., ECM ps_active formulation under-models flow-limit transition>.
- Mode C: Inconclusive. Drug-specific confounds suspected — see per-drug predict_layer_confound_diagnostics in result file. ECM generalization claim remains unconfirmed, not refuted.
- Mode D: ECM = statin-specialized. Architecture review required. ECM merge `a60a14e` remains valid for statins; generalization beyond statins is retracted.

**No post-run parameter adjustment.** Spec §Execution Constraints honored.
```

Replace placeholders with actual values from `oatp_generalization_result.json`.

- [ ] **Step 2: Commit experiment log update**

```bash
git add docs/claude/experiment-log.md
git commit -m "docs(experiment-log): ECM generalization test Mode <X>"
```

- [ ] **Step 3: Save memory**

Save memory to `~/.claude/projects/-home-jam-Sisyphus/memory/project_ecm_generalization_test.md`:

```markdown
---
name: ECM Generalization Test Outcome
description: ECM hepatic-clearance test on 3 non-statin OATP1B1 substrates (bosentan, valsartan, repaglinide) under hard pre-registration. Spec 9115e63, Mode <X> result 2026-04-21.
type: project
---
Executed 2026-04-21 per spec docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md.

**Mode <X>** on 3/3 substrates.

**Per drug:** bosentan <result>, valsartan <result>, repaglinide <result>.

**Why:** Decorrelation test for ECM (merged a60a14e) — verify ECM is general OATP1B1 mechanism, not statin-specialized. Cherry-picking audit response to 35+ error-cancellation rejection pattern.

**How to apply:**
- "ECM generalizes?" / "OATP non-statin" → this result applies.
- Mode A: proceed with ECM-based features for in-domain OATP substrates.
- Mode B: direction-matched follow-up per experiment-log interpretation.
- Mode C: treat ECM generalization as unconfirmed; do not make claims beyond 3/5 statin + 0-3 non-statin pass.
- Mode D: ECM is statin-specialized; do NOT extend to non-statin features without new evidence.

**Frozen artifacts (do not modify):** data/transporters/oatp1b1.json (bosentan/valsartan/repaglinide entries), data/validation/oatp_generalization_drugs.json, data/validation/oatp_generalization_result.json.
```

Add a pointer line in `MEMORY.md`:

```markdown
- [project_ecm_generalization_test.md](project_ecm_generalization_test.md) — ECM OATP1B1 generalization test Mode <X> (3 non-statin substrates, 2026-04-21)
```

Replace `<X>` and `<result>` with actual values.

- [ ] **Step 4: Final verification**

```
pytest tests/unit/test_oatp_generalization_data.py tests/unit/test_transporter_db.py tests/unit/test_oatp_generalization_classifier.py tests/integration/test_oatp_generalization_pipeline.py -v
git log --oneline -10
```
Expected: all tests PASS; git log shows 6-7 commits from this plan (data, transporters+tests, classifier, script, integration test, result, experiment-log).

---

## Summary of Commits Produced

1. `data(oatp): IV clinical obs for 3 generalization substrates` (Task 1)
2. `data(transporters): OATP1B1 Jmax/Km for bosentan/valsartan/repaglinide` (Task 2)
3. `feat(validation): ECM generalization test classifier` (Task 3)
4. `script(oatp): ECM generalization test driver` (Task 4)
5. `test(integration): ECM generalization pipeline smoke` (Task 5)
6. `result(oatp): ECM generalization test — Mode <X>` (Task 6)
7. `docs(experiment-log): ECM generalization test Mode <X>` (Task 7)

Plus memory file saved outside git (user-level memory).

## Violation Log (cherry-picking defense)

If any of the following occurs during execution, it MUST be documented in the final report and treated as a spec violation:

1. Any data file edited after Task 6 Step 2 completes.
2. Any engine/ADME code modification.
3. Any re-run of Task 6 Step 2 with different parameters.
4. Any drug dropped from analysis post-hoc.
5. Any envelope test widened to accommodate extracted values that exceed the envelope without primary-source confirmation.
