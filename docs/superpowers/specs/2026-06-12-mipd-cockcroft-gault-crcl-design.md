# Cockcroft-Gault CrCl Estimation — Design

**Date:** 2026-06-12
**Status:** Approved (design); pending implementation plan
**Module:** `src/sisyphus/mipd/covariates.py` (extend)
**Composes with:** `Covariates.renal_factor()` (the renal individualization lever)

---

## 1. Purpose

The MIPD covariate path individualizes the engine's renal clearance from a **measured**
creatinine clearance (`Covariates.crcl_ml_min` → `renal_factor() = CrCl/125`). In practice
CrCl is often not measured directly; it is **estimated** from age, weight, sex, and serum
creatinine. This adds the **Cockcroft-Gault** estimator and a factory that feeds its output
into the existing renal path, so a caller with routine labs (not a 24 h urine collection) can
still individualize renal clearance.

## 2. The equation (Cockcroft & Gault, *Nephron* 1976)

```
CrCl (mL/min) = (140 - age_years) * weight_kg * (0.85 if female else 1.0) / (72 * SCr_mg/dL)
```

The output is an **absolute** CrCl in mL/min (body size enters via the weight term). This is
the right quantity for this engine: the reference renal model is glomerular-filtration-only
with an **absolute** reference GFR of 125 mL/min (7.5 L/h, reference 70 kg / 1.73 m² man; see
`predict/ivive.py`), so `renal_factor = CG_CrCl / 125` is dimensionally coherent and a larger
patient correctly scales renal CL up.

**Pitfall (documented):** do NOT feed a BSA-normalized eGFR (CKD-EPI/MDRD, mL/min/1.73 m²)
into `crcl_ml_min` — only absolute CrCl composes with this path. CG gives absolute, so it is
the correct estimator to wire in here.

## 3. Surface (additions to `covariates.py`)

```python
def cockcroft_gault(age_years: float, weight_kg: float, scr_mg_dl: float, sex: str) -> float:
    """Estimate creatinine clearance (mL/min) by the Cockcroft-Gault equation."""

# classmethod on Covariates:
@classmethod
def from_cockcroft_gault(cls, age_years, weight_kg, scr_mg_dl, sex) -> "Covariates":
    """Covariates with crcl_ml_min ESTIMATED via Cockcroft-Gault (renal-only)."""
    return cls(crcl_ml_min=cockcroft_gault(age_years, weight_kg, scr_mg_dl, sex))
```

`renal_factor()` is **unchanged** — the estimate flows through it identically to a measured CrCl.

### Constants (cited)
`_CG_AGE_CONSTANT = 140.0`, `_CG_SCR_DENOMINATOR_DL = 72.0`, `_CG_FEMALE_FACTOR = 0.85` (Cockcroft
& Gault 1976); `_CG_MAX_AGE_YEARS = 140.0` (validation floor so `140-age > 0`);
`_CG_PEDIATRIC_AGE_YEARS = 18.0`, `_CG_LOW_SCR_MG_DL = 0.6` (advisory thresholds).

## 4. Inputs & conventions

- **sex**: `"male"` / `"female"`, case-insensitive (`.strip().lower()`); anything else → `ValueError`.
- **scr_mg_dl**: serum creatinine in **mg/dL** (the formula's native unit; the `_mg_dl` suffix in
  the parameter name is the unit guardrail). SI µmol/L is NOT auto-converted (÷88.42 to convert).
- **weight**: **actual** body weight (classic CG). Caveat (documented): with actual weight, CG
  **overestimates** CrCl in obese patients; ideal/adjusted body weight needs height and is a
  future extension.

## 5. Renal-only output (the deliberate trade-off)

`from_cockcroft_gault` returns `Covariates(crcl_ml_min=estimate)` — it does **not** store
`body_weight_kg`/`age_years`, so `has_physiology()` stays False and **no `generate_physiology`
graph rebuild** is triggered. Weight/age are inputs to the *estimate* only.

**Documented consequence:** this individualizes **renal CL only**. A large/small patient gets
renal CL scaled by CG (which used their weight), but Vd / hepatic CL remain at the 70 kg
reference body — an internal inconsistency that is the accepted renal-only scope. For
whole-body size individualization, a caller passes `body_weight_kg`/`age_years` explicitly to
`Covariates` (which triggers `generate_physiology`), separately from this factory.

## 6. Validation & edge handling

`cockcroft_gault` raises `ValueError` on:
- any input not finite (`not math.isfinite(x)`) — prevents silent `nan`/`inf` propagation
  (a `nan` would pass every `<= 0` check and poison `renal_factor()`).
- `sex` not in `{"male", "female"}` (after `strip().lower()`).
- `age_years <= 0` or `age_years >= _CG_MAX_AGE_YEARS` (keeps `140 - age > 0`).
- `weight_kg <= 0`.
- `scr_mg_dl <= 0`.

`cockcroft_gault` emits (non-blocking) `logging.warning`:
- when `age_years < _CG_PEDIATRIC_AGE_YEARS` (18): "Cockcroft-Gault is not validated for age
  < 18; consider Schwartz."
- when `scr_mg_dl < _CG_LOW_SCR_MG_DL` (0.6): "serum creatinine {x} mg/dL is low; in patients
  with reduced muscle mass (elderly/cachectic) Cockcroft-Gault may overestimate CrCl."

No SCr round-up (Jelliffe) and no output capping: KDIGO does not endorse round-up and it can
cause underdosing — the honest estimate plus a warning is preferred. Extreme *results* are
still caught downstream by the existing `Covariates.warnings()` (`crcl` outside [5, 200]).

**Known limitation (documented):** because the output is renal-only (weight/age not stored),
the CG advisories above are emitted via `logging` at construction and do NOT appear in the
prediction's structured `warnings` (`PosteriorPK.warnings` / `recommend_dose`). `logging` is
the sanctioned advisory channel; structured propagation is a possible follow-up.

## 7. Testing (TDD)

- **Formula, cancelling anchor:** `cockcroft_gault(60, 72, 1.0, "male") == 80.0`; `"female"` → `68.0`.
- **Formula, non-cancelling anchor** (verifies the full arithmetic, not the 72/1.0 coincidence):
  `cockcroft_gault(40, 80, 1.2, "male") == pytest.approx(92.5926, rel=1e-4)`.
- **Case-insensitive sex:** `"Male"`, `"FEMALE"` accepted.
- **Validation `ValueError`:** bad sex; `age <= 0`; `age >= 140`; `weight <= 0`; `scr <= 0`;
  `nan`/`inf` input.
- **Factory renal-only:** `from_cockcroft_gault(60, 72, 1.0, "male")` →
  `crcl_ml_min == pytest.approx(80.0)`, `body_weight_kg is None`, `age_years is None`;
  `renal_factor() == pytest.approx(80.0 / 125.0)`.
- **Advisories (`caplog`):** `age_years = 10` logs the pediatric warning; `scr_mg_dl = 0.4`
  logs the low-SCr warning.

## 8. Scope / invariants

- New helper + additive classmethod only. **No `engine/`, `predict()`, `renal_factor()`, or
  `Covariates` field changes.** `Covariates` stays frozen; the classmethod is additive.
- **No `mipd/__init__` export** — `cockcroft_gault` and `Covariates` live in
  `sisyphus.mipd.covariates` (Covariates is not package-exported today; this does not expand
  that). Callers: `from sisyphus.mipd.covariates import cockcroft_gault, Covariates`.
- Bit-identical to current behavior when the factory is not used (no production-path change;
  holdout / 2.731 headline untouched).
- `covariates.py` grows ~73 → ~120 lines (gains `import math`, `import logging`).

## 9. Out of scope (future)

- Ideal/adjusted body weight for obesity (needs height).
- SI (µmol/L) serum-creatinine auto-conversion.
- Pediatric Schwartz estimator.
- Structured propagation of CG advisories into the prediction `warnings`.
- BSA-normalized eGFR (CKD-EPI/MDRD) ingestion with de-normalization.
