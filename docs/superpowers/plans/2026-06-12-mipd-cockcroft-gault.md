# Cockcroft-Gault CrCl Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Cockcroft-Gault creatinine-clearance estimator and a renal-only `Covariates` factory, so a caller with routine labs (age/weight/sex/serum-creatinine) can individualize the engine's renal clearance without a measured CrCl.

**Architecture:** Two additions to the existing `src/sisyphus/mipd/covariates.py`: a pure module-level function `cockcroft_gault(...)` (formula + validation + advisory warnings) and an additive classmethod `Covariates.from_cockcroft_gault(...)` that returns `Covariates(crcl_ml_min=estimate)` (renal-only — weight/age are estimate inputs, not stored, so no `generate_physiology` rebuild). `renal_factor()` is unchanged.

**Tech Stack:** Python 3.10+, stdlib `math`/`logging`, frozen dataclass, pytest.

**Spec:** `docs/superpowers/specs/2026-06-12-mipd-cockcroft-gault-crcl-design.md`

**Conventions:** `from __future__ import annotations`; `logging` (not print); type hints; `ruff` line length 100; constants `UPPER_SNAKE` with a source comment; commits `type(mipd): description` with NO `Co-Authored-By`/AI trailer.

**Current `covariates.py` shape (for orientation):**
- Imports: `from __future__ import annotations` then `from dataclasses import dataclass`.
- `_REFERENCE_GFR_ML_MIN = 125.0`.
- `@dataclass(frozen=True) class Covariates` with fields `crcl_ml_min/body_weight_kg/age_years` (all `float | None = None`), `__post_init__` (positivity validation), `renal_factor()` (returns `crcl_ml_min/125.0` or `1.0`), `has_physiology()`, `warnings()`.
- No `math`/`logging` imports yet.

**Test file:** `tests/unit/test_mipd_covariates.py` already exists — APPEND to it.

---

## File Structure

- **Modify** `src/sisyphus/mipd/covariates.py` — add CG constants + `cockcroft_gault()` (Task 1) and the `from_cockcroft_gault()` classmethod (Task 2). One file, one responsibility (covariate derivation). ~73 → ~120 lines.
- **Modify** `tests/unit/test_mipd_covariates.py` — append CG function tests (Task 1) + factory tests (Task 2).

No `mipd/__init__.py` change (per spec §8 — `cockcroft_gault`/`Covariates` stay in the submodule).

---

### Task 1: `cockcroft_gault` pure function

**Files:**
- Modify: `src/sisyphus/mipd/covariates.py`
- Test: `tests/unit/test_mipd_covariates.py`

- [ ] **Step 1: Write the failing tests** — APPEND to `tests/unit/test_mipd_covariates.py`. Ensure these imports exist at the top of the file (add only the missing ones): `import logging`, `import pytest`, and `from sisyphus.mipd.covariates import Covariates, cockcroft_gault`. (Do NOT add `import math` to the test file — the tests use `float("nan")`/`float("inf")`, so a `math` import would be an unused-import ruff F401 failure.)

```python
def test_cockcroft_gault_male_cancelling_anchor():
    # weight 72, scr 1.0 cancels the 72 denominator -> CrCl == 140 - age
    assert cockcroft_gault(60, 72.0, 1.0, "male") == pytest.approx(80.0)


def test_cockcroft_gault_female_factor():
    assert cockcroft_gault(60, 72.0, 1.0, "female") == pytest.approx(68.0)  # 80 * 0.85


def test_cockcroft_gault_non_cancelling_anchor():
    # full arithmetic: (140-40)*80 / (72*1.2) = 8000 / 86.4 = 92.5926
    assert cockcroft_gault(40, 80.0, 1.2, "male") == pytest.approx(92.5926, rel=1e-4)


def test_cockcroft_gault_sex_case_insensitive():
    assert cockcroft_gault(60, 72.0, 1.0, "Male") == pytest.approx(80.0)
    assert cockcroft_gault(60, 72.0, 1.0, "FEMALE") == pytest.approx(68.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"age_years": 60, "weight_kg": 72, "scr_mg_dl": 1.0, "sex": "x"}, "sex"),
        ({"age_years": 0, "weight_kg": 72, "scr_mg_dl": 1.0, "sex": "male"}, "age_years"),
        ({"age_years": 140, "weight_kg": 72, "scr_mg_dl": 1.0, "sex": "male"}, "age_years"),
        ({"age_years": 60, "weight_kg": 0, "scr_mg_dl": 1.0, "sex": "male"}, "weight_kg"),
        ({"age_years": 60, "weight_kg": 72, "scr_mg_dl": 0, "sex": "male"}, "scr_mg_dl"),
        ({"age_years": float("nan"), "weight_kg": 72, "scr_mg_dl": 1.0, "sex": "male"}, "finite"),
        ({"age_years": 60, "weight_kg": float("inf"), "scr_mg_dl": 1.0, "sex": "male"}, "finite"),
    ],
)
def test_cockcroft_gault_rejects_bad_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        cockcroft_gault(**kwargs)


def test_cockcroft_gault_pediatric_warns(caplog):
    with caplog.at_level(logging.WARNING):
        cockcroft_gault(10, 30.0, 1.0, "male")
    assert "not validated" in caplog.text.lower()


def test_cockcroft_gault_low_scr_warns(caplog):
    with caplog.at_level(logging.WARNING):
        cockcroft_gault(60, 72.0, 0.4, "male")
    assert "overestimate" in caplog.text.lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/test_mipd_covariates.py -k cockcroft -q`
Expected: FAIL — `ImportError: cannot import name 'cockcroft_gault'`.

- [ ] **Step 3: Implement the function + constants** in `src/sisyphus/mipd/covariates.py`.

(a) Change the imports block at the top from:
```python
from __future__ import annotations

from dataclasses import dataclass
```
to:
```python
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)
```

(b) Immediately AFTER the existing `_REFERENCE_GFR_ML_MIN = 125.0` line, add the CG constants and the function:
```python
# Cockcroft & Gault, Nephron 1976: CrCl = (140-age)*wt*(0.85 if female)/(72*SCr_mg/dL).
_CG_AGE_CONSTANT = 140.0
_CG_SCR_DENOMINATOR_DL = 72.0
_CG_FEMALE_FACTOR = 0.85
_CG_MAX_AGE_YEARS = 140.0          # validation floor: 140 - age must stay > 0
_CG_PEDIATRIC_AGE_YEARS = 18.0     # CG is not validated below this age
_CG_LOW_SCR_MG_DL = 0.6            # low SCr -> CrCl overestimation risk (low muscle mass)


def cockcroft_gault(
    age_years: float, weight_kg: float, scr_mg_dl: float, sex: str
) -> float:
    """Estimate creatinine clearance (mL/min) by the Cockcroft-Gault equation.

    ``CrCl = (140 - age) * weight_kg * (0.85 if female else 1.0) / (72 * SCr_mg/dL)``.
    ``scr_mg_dl`` is serum creatinine in mg/dL (NOT SI µmol/L; divide µmol/L by 88.42).
    ``sex`` is "male"/"female" (case-insensitive). Returns an ABSOLUTE CrCl in mL/min
    (body size enters via the weight term) — the quantity ``Covariates.crcl_ml_min``
    expects; do not pass a BSA-normalized eGFR there instead. With actual body weight,
    CG overestimates CrCl in obese patients (adjusted body weight needs height — future).

    Raises ``ValueError`` on non-finite, non-positive, or out-of-range inputs. Logs a
    (non-blocking) warning for age < 18 (CG unvalidated) and for very low SCr
    (overestimation in reduced muscle mass).
    """
    for _name, _value in (
        ("age_years", age_years), ("weight_kg", weight_kg), ("scr_mg_dl", scr_mg_dl)
    ):
        if not math.isfinite(_value):
            raise ValueError(f"{_name} must be finite, got {_value}")
    sex_l = sex.strip().lower()
    if sex_l not in ("male", "female"):
        raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")
    if age_years <= 0 or age_years >= _CG_MAX_AGE_YEARS:
        raise ValueError(
            f"age_years must be in (0, {_CG_MAX_AGE_YEARS}), got {age_years}"
        )
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if scr_mg_dl <= 0:
        raise ValueError(f"scr_mg_dl must be > 0, got {scr_mg_dl}")

    if age_years < _CG_PEDIATRIC_AGE_YEARS:
        logger.warning(
            "Cockcroft-Gault is not validated for age < %s (got %s); consider Schwartz.",
            _CG_PEDIATRIC_AGE_YEARS, age_years,
        )
    if scr_mg_dl < _CG_LOW_SCR_MG_DL:
        logger.warning(
            "serum creatinine %s mg/dL is low; in patients with reduced muscle mass "
            "(elderly/cachectic) Cockcroft-Gault may overestimate CrCl.", scr_mg_dl,
        )

    crcl = (
        (_CG_AGE_CONSTANT - age_years) * weight_kg
        / (_CG_SCR_DENOMINATOR_DL * scr_mg_dl)
    )
    if sex_l == "female":
        crcl *= _CG_FEMALE_FACTOR
    return crcl
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/unit/test_mipd_covariates.py -k cockcroft -q`
Expected: PASS (all `cockcroft_*` tests, ~7 cases incl. the 7 parametrized rejections).

- [ ] **Step 5: ruff**

Run: `ruff check src/sisyphus/mipd/covariates.py tests/unit/test_mipd_covariates.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/mipd/covariates.py tests/unit/test_mipd_covariates.py
git commit -m "feat(mipd): Cockcroft-Gault CrCl estimator (cockcroft_gault)"
```

---

### Task 2: `Covariates.from_cockcroft_gault` factory (renal-only)

**Files:**
- Modify: `src/sisyphus/mipd/covariates.py`
- Test: `tests/unit/test_mipd_covariates.py`

- [ ] **Step 1: Write the failing tests** — APPEND to `tests/unit/test_mipd_covariates.py` (imports from Task 1 already cover `Covariates`/`pytest`):

```python
def test_from_cockcroft_gault_is_renal_only():
    cov = Covariates.from_cockcroft_gault(60, 72.0, 1.0, "male")
    assert cov.crcl_ml_min == pytest.approx(80.0)
    assert cov.body_weight_kg is None   # renal-only: weight/age are estimate inputs, not stored
    assert cov.age_years is None
    assert cov.has_physiology() is False  # so no generate_physiology rebuild is triggered


def test_from_cockcroft_gault_feeds_renal_factor():
    cov = Covariates.from_cockcroft_gault(60, 72.0, 1.0, "male")
    assert cov.renal_factor() == pytest.approx(80.0 / 125.0)


def test_from_cockcroft_gault_propagates_validation():
    with pytest.raises(ValueError, match="sex"):
        Covariates.from_cockcroft_gault(60, 72.0, 1.0, "unknown")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/test_mipd_covariates.py -k from_cockcroft -q`
Expected: FAIL — `AttributeError: type object 'Covariates' has no attribute 'from_cockcroft_gault'`.

- [ ] **Step 3: Implement the classmethod.** Add it to the `Covariates` class body, placed AFTER the `warnings()` method (the last method), at class-body indentation:

```python
    @classmethod
    def from_cockcroft_gault(
        cls, age_years: float, weight_kg: float, scr_mg_dl: float, sex: str
    ) -> "Covariates":
        """Build Covariates with crcl_ml_min ESTIMATED via Cockcroft-Gault (renal-only).

        Use when CrCl is not directly measured. Weight/age are inputs to the estimate
        ONLY — they are not stored, so ``has_physiology()`` stays False and no
        ``generate_physiology`` graph rebuild is triggered: this individualizes renal
        CL only. For whole-body size individualization, pass body_weight_kg/age_years to
        ``Covariates(...)`` directly. The estimate then flows through ``renal_factor()``
        exactly like a measured CrCl.
        """
        return cls(crcl_ml_min=cockcroft_gault(age_years, weight_kg, scr_mg_dl, sex))
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/unit/test_mipd_covariates.py -k from_cockcroft -q`
Expected: PASS (3 tests). Then the whole file: `python -m pytest tests/unit/test_mipd_covariates.py -q` — expect all green.

- [ ] **Step 5: ruff**

Run: `ruff check src/sisyphus/mipd/covariates.py tests/unit/test_mipd_covariates.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/mipd/covariates.py tests/unit/test_mipd_covariates.py
git commit -m "feat(mipd): Covariates.from_cockcroft_gault renal-only factory"
```

---

## Final Verification (after both tasks)

- [ ] **Run the covariates + dependent MIPD suites:**

```bash
python -m pytest tests/unit/test_mipd_covariates.py tests/unit/test_mipd_api.py \
       tests/unit/test_mipd_tdm.py -q
```
Expected: all pass. (The factory is additive and renal_factor() is unchanged, so the api/tdm covariate paths are unaffected — confirm.)

- [ ] **Confirm scope:** `git diff main --stat` shows ONLY `src/sisyphus/mipd/covariates.py`, `tests/unit/test_mipd_covariates.py`, and the spec/plan docs. No `engine/`, `predict/`, `pipeline/`, `__init__.py`, or data changes.

- [ ] **Update graphify graph:** `graphify update .` (AST-only, no API cost).

## Notes for the implementer

- The `from __future__ import annotations` already present makes the `-> "Covariates"` self-reference fine; keep the quotes for clarity.
- `caplog.text` (not `record.message`) is used in the warning tests because the warnings use `%s` lazy formatting — `caplog.text` contains the formatted output.
- Do NOT round SCr up to 1.0 and do NOT cap the output — the spec deliberately prefers the honest estimate + warning (KDIGO does not endorse round-up).
- Two commits total (one per task). No `Co-Authored-By`/AI trailer.
