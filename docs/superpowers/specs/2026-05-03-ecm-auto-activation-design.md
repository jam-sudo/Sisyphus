# v0.3 ECM Auto-Activation — Design Spec

**Date:** 2026-05-03
**Status:** Draft for user review
**Spawn from:** PR #22 (merged 2026-05-02, closed #12 #13 #14) — explicit follow-up. Successor cleanup of issue triage 2026-05-03 (closed #8, #21; DE-33 root-cause confirmed).

## Goal

Wire `pipeline.predict.predict()` to automatically activate the ECM (Extended Clearance Model) machinery — OATP1B1 saturable hepatic uptake plus passive PS / biliary CL_int — for drugs registered as OATP-rate-limited substrates. This replaces the current default path (XGBoost-CYP only) with a mechanistically correct two-path hepatic clearance for those drugs.

In one sentence: **pravastatin's predict() default Cmax should match its ECM-active simulation, not the XGBoost-CYP-only fallback.**

## Architecture

Three-layer extension of PR #22's registry pattern, no engine code changes:

1. **Substrate detection**: a new boolean field `ecm_applicable` per drug entry in `data/transporters/oatp1b1.json`. Default `false` (key absent = false). Detection is by InChIKey of `compute_profile(smiles).inchikey`.
2. **Conditional kinetics loading**: in `pipeline.predict.predict()`, after `compute_profile()` and before `build_drug_on_graph()`, look up the applicability flag. If `true`, populate `transporter_kinetics` and `hepatic_ecm_params` from the registries.
3. **Phenotype API (additive)**: extend `predict()` with an optional `phenotypes: dict[str, str] | None = None` parameter that, when provided, applies `apply_phenotype_to_graph` to the BodyGraph before drug binding. This is orthogonal to ECM activation: phenotype scaling makes sense only when the relevant transporter abundance is already feeding into a clearance path.

The `metabolic_fraction` registry (PR #22) is already wired through `build_drug_on_graph` — no changes needed.

## Tech Stack

Python 3.10, existing modules:
- `src/sisyphus/pipeline/predict.py` (modify `predict()`)
- `src/sisyphus/predict/transporter_db.py` (extend lookup helpers)
- `data/transporters/oatp1b1.json` (schema extension)

No new dependencies, no engine changes, no ML retraining.

---

## Section 1 — Substrate Registry Schema Extension

Add `ecm_applicable: bool` to each drug entry in `data/transporters/oatp1b1.json`. Default behavior when the field is absent: treated as `false`.

A new helper in `predict/transporter_db.py`:

```python
def is_oatp_ecm_applicable(inchikey: str) -> bool:
    """Return True if the InChIKey is registered with ecm_applicable=true."""
```

The lookup is keyed on InChIKey (matching the existing oatp1b1.json `inchikey` field per drug). SMILES variation across callers is normalized through RDKit canonicalization.

### Initial seed list

| drug | flag | rationale |
|---|---|---|
| pravastatin | `true` | Niemi 2009 PM/EM ~ 2.6×; OATP1B1 rate-limits hepatic uptake. PR #22 calibrated FE 1.066 vs FDA. |
| pitavastatin | `true` | Niemi 2009 PM/EM ~ 3×; passes test_oatp_ecm_statins gate. |
| fluvastatin | `false` | Niemi 2009 PM/EM ~ 1.0×; ~75% CYP2C9 dominant. Issue #21 closure documents this. |
| valsartan | `false` | Vss over-prediction confirmed (DE-33). Activating ECM amplifies the under-prediction. |
| glimepiride | `false` | Same as valsartan (DE-33). |
| rosuvastatin | `false` (defer) | Niemi PM/EM ~ 2× clinically supports `true`, but test_oatp_ecm_statins xfails on Peff over-prediction (FE ~12). Promote when Peff issue is addressed. |
| atorvastatin | `false` (defer) | Same as rosuvastatin. Niemi PM/EM ~ 1.5×; xfail on Peff. |

The seed list is intentionally conservative. Adding `true` for borderline drugs is a follow-up commit, not a v0.3 scope item.

---

## Section 2 — predict() Change

In `src/sisyphus/pipeline/predict.py`, after the existing `profile = compute_profile(smiles)` and `adme = predict_adme(profile)` calls, before `build_drug_on_graph(...)`:

```python
from sisyphus.predict.transporter_db import (
    is_oatp_ecm_applicable,
    load_hepatic_ecm_params,
    load_oatp1b1_kinetics,
)

transporter_kinetics = None
hepatic_ecm_params = None
if is_oatp_ecm_applicable(profile.inchikey):
    transporter_kinetics = load_oatp1b1_kinetics_by_inchikey(profile.inchikey)
    hepatic_ecm_params = load_hepatic_ecm_params_by_inchikey(profile.inchikey)
```

Then pass these into `build_drug_on_graph(...)`. The function already accepts both as keyword arguments.

The lookup helpers will need an InChIKey-keyed variant since the current `load_oatp1b1_kinetics(name: str)` is name-keyed. The InChIKey lookup is necessary because `predict()` does not have a canonical drug name for arbitrary SMILES inputs.

`metabolic_fraction` is already looked up by InChIKey/SMILES inside `build_drug_on_graph` (PR #22 wired this). No change needed.

---

## Section 3 — Phenotype API (additive, optional)

Extend `predict()` signature:

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

When `phenotypes` is `None` (current default), behavior is the reference-population graph as built from YAML. When `phenotypes` is provided (e.g. `{"SLCO1B1": "PM"}`), call `apply_phenotype_to_graph(graph, phenotypes)` before drug binding.

This is orthogonal to ECM activation: providing `phenotypes={"SLCO1B1": "PM"}` for a drug that is NOT `ecm_applicable=true` will scale the OATP1B1 abundance, but with no clearance path consuming that abundance the result is invariant. Documented in the docstring.

GenoADME consumes this API by passing `phenotypes` per individual; tests/integration use it for SLCO1B1 EM/PM checks.

---

## Section 4 — Holdout Impact Validation

The seed list activates ECM for two drugs (pravastatin, pitavastatin). Both are in the 107-holdout. Expected Cmax shifts:

| drug | current predict() | post-v0.3 predict() | source |
|---|---|---|---|
| pravastatin | 0.0500 mg/L (XGBoost-CYP only, 20 mg) | ~0.0211 mg/L (ECM-active, scaled) | scripts/diagnose_pravastatin_ecm.py |
| pitavastatin | TBD before promotion | TBD before promotion | run before merge |

Engine-track AAFE will shift for these two drugs (and only these two). Meta-track AAFE will shift less due to ML/classifier blending.

**Pre-merge gate**: re-run `scripts/run_engine_benchmark.py`, regenerate `data/training/4track_holdout_predictions.json`, refresh bootstrap CIs (10k resamples, seed=20260422), update `data/validation/4track_ci_2026-05-03.json`. Headline AAFE values in `CLAUDE.md` and `README.md` updated to the new baseline.

The shift is documented as a model improvement (not regression), with the rationale that XGBoost-CYP-only was a known under-spec for OATP-rate-limited drugs (PR #22 only fixed the multi-path double-counting; predict() never connected the registry to the engine until now).

If the Engine-track AAFE worsens by >10% relative, halt: that signals the seed list is wrong (some drug shouldn't be flagged). Investigate and adjust the flag set, do not merge with a regression.

---

## Section 5 — Initial Seed Decisions (already covered in Section 1)

See Section 1 table. Two drugs flagged `true`. The conservative posture is deliberate — it lets us measure the AAFE impact at the smallest possible step, then promote individual drugs as their ECM parameters are validated. No part of the v0.3 plan promotes a drug from `false` to `true`.

---

## Section 6 — Testing Plan

### 6.1 Existing tests

- **`tests/integration/test_oatp_pravastatin.py`**: keep manual `build_drug_on_graph` path (unit-level coverage of ECM/PGx machinery). Adds an additional check: `predict(smiles_pravastatin)` produces the same Cmax as the manual path within 1% tolerance.
- **`tests/integration/test_oatp_ecm_statins.py`**: unchanged. Continues to gate ECM behavior directly.
- **`tests/regression/test_holdout_regression.py`**: pin updated to new Meta AAFE.
- **`tests/regression/test_prodrug_v2_snapshot.py`**: unchanged (4 prodrugs not OATP substrates).

### 6.2 New tests

`tests/integration/test_predict_auto_ecm.py`:
1. `test_pravastatin_predict_auto_ecm_active` — `predict(PRAVA_SMILES)` and direct `build_drug_on_graph(...transporter_kinetics=...)` produce Cmax within 1%.
2. `test_fluvastatin_predict_no_auto_ecm` — `predict(FLUVA_SMILES)` Cmax matches the no-ECM path (no triple-counting).
3. `test_predict_phenotype_api` — `predict(PRAVA_SMILES, phenotypes={"SLCO1B1": "PM"})` Cmax > `predict(PRAVA_SMILES)` Cmax.
4. `test_phenotype_orthogonal_for_non_substrate` — `predict(MIDAZOLAM_SMILES, phenotypes={"SLCO1B1": "PM"})` invariant vs no phenotypes (midazolam not OATP substrate).

### 6.3 Schema test

`tests/regression/test_oatp_registry_schema.py`:
1. Every drug entry in `oatp1b1.json` has either `ecm_applicable: bool` set explicitly OR a comment justifying the absence.
2. The 7-drug seed list values match Section 1 exactly (regression-pinned to catch silent flag flips).

---

## Error Handling

- Missing InChIKey on `oatp1b1.json` entry → schema test fails CI, registry not loaded as substrate. Production path unaffected.
- `is_oatp_ecm_applicable(inchikey)` for an unknown InChIKey → returns `False`. No exception. Same as PR #22's `metabolic_fraction` lookup.
- `phenotypes` dict with unknown key (e.g. `{"SLCO1B7": "PM"}`) → `apply_phenotype_to_graph` already raises `ValueError`. Documented in `predict()` docstring.

## Out of Scope

- Re-promoting rosuvastatin / atorvastatin to `ecm_applicable: true` (blocked by Peff over-prediction; separate work).
- Adding new OATP substrates beyond the registry (clopidogrel, simvastatin acid, irinotecan are tracked in #11 prodrug expansion).
- Kp method retuning for high-fup acids (DE-33 root cause; engine-layer work).
- BCRP / OATP1B3 / NTCP transporters (engine extension).
- Auto-activation for non-OATP transporters (e.g. P-gp at gut wall) — separate spec.

## Self-review

- **Placeholder scan**: pitavastatin's "TBD" Cmax row in Section 4 is a deliberate "must measure before merge" gate, not a documentation gap. Acceptable for a draft spec; the pre-merge run will fill it in.
- **Internal consistency**: Section 5 references Section 1 (consistent). Section 6 references Section 1 seed list (consistent). Phenotype API in Section 3 mentions "orthogonal to ECM activation" — confirmed by Section 6.2 test 4.
- **Scope check**: single subsystem (`predict` orchestration + registry data). No engine, no ML, no graph topology. Single implementation plan.
- **Ambiguity check**: `predict()` Cmax meaning is plasma (matches existing convention from `pk/endpoints.py`); `phenotypes` dict key format follows `apply_phenotype_to_graph` (existing convention). All explicit.

## Decision summary

| design choice | resolved | rationale |
|---|---|---|
| substrate detection | InChIKey + per-drug `ecm_applicable` flag (option C from brainstorm) | explicit per-drug audit trail; no external DB dependency |
| default behavior for unflagged | OFF | conservative; AAFE shift bounded to flagged set |
| auto-activation policy | always when flag is `true` | mechanistic correctness over data-only invariance |
| phenotype API | additive `phenotypes=` parameter on `predict()` | orthogonal to substrate detection; matches existing GenoADME usage pattern |
| seed `true` set | pravastatin, pitavastatin | smallest measurable step; promote others after Peff fix and Vss work |

## Open questions for user review

None blocking. Spec is ready for implementation plan.
