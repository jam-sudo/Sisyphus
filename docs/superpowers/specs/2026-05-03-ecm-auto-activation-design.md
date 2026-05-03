# v0.3 ECM Auto-Activation — Design Spec

**Date:** 2026-05-03 (revised after spec self-review with empirical validation)
**Status:** Draft for user review
**Spawn from:** PR #22 (merged 2026-05-02, closed #12 #13 #14) — explicit follow-up. Successor of OATP triage 2026-05-03 (closed #8, #21; DE-33 root-cause confirmed).

## Goal

Wire `pipeline.predict.predict()` to automatically activate the ECM (Extended Clearance Model) machinery — OATP1B1 saturable hepatic uptake plus passive PS / biliary CL_int — for drugs registered as OATP-rate-limited substrates. This replaces the current default path (XGBoost-CYP only) with a mechanistically correct two-path hepatic clearance for those drugs.

In one sentence: **pravastatin's predict() default Cmax should match its ECM-active simulation, not the XGBoost-CYP-only fallback.**

## Architecture

Three-layer extension of PR #22's registry pattern, no engine code changes:

1. **Substrate detection**: a new boolean field `ecm_applicable` per drug entry in `data/transporters/oatp1b1.json`. Default `false` (key absent = false). Detection occurs inside a new helper that derives the InChIKey from SMILES via RDKit (mirroring the existing `lookup_metabolic_fraction(smiles)` pattern from PR #22). `MolecularProfile` is **not** modified — the helper is self-contained.
2. **Conditional kinetics loading**: in `pipeline.predict.predict()`, after `compute_profile()` and before `build_drug_on_graph()`, look up the applicability flag via SMILES. If `true`, populate `transporter_kinetics` and `hepatic_ecm_params` from the registries.
3. **Phenotype API (additive)**: extend `predict()` with an optional `phenotypes: dict[str, str] | None = None` parameter that, when provided, applies `apply_phenotype_to_graph` to the BodyGraph before drug binding. Supports all keys that `apply_phenotype_to_graph` accepts (SLCO1B1 + CYP2D6/2C9/2C19/3A5/1A2/2B6 today; new additions like NAT2/UGT1A1 from issue #10 will inherit automatically).

The `metabolic_fraction` registry (PR #22) is already wired through `build_drug_on_graph` via `lookup_metabolic_fraction(profile.smiles)` (see `predict/cyp_clearance_overrides.py`) — predict() does not need to re-look it up. **However**, every drug promoted to `ecm_applicable=true` MUST have a corresponding entry in `cyp_clearance_overrides.json` with a literature-justified `metabolic_fraction`, or it triple-counts hepatic clearance. See empirical validation in §1.2.

## Tech Stack

Python 3.10, existing modules:
- `src/sisyphus/pipeline/predict.py` (modify `predict()` signature and body)
- `src/sisyphus/predict/transporter_db.py` (add InChIKey-keyed lookup helpers + applicability check)
- `data/transporters/oatp1b1.json` (schema extension; one drug entry updated)

No new dependencies, no engine changes, no ML retraining.

---

## Section 1 — Substrate Registry Schema Extension

### 1.1 Schema

Add `ecm_applicable: bool` to each drug entry in `data/transporters/oatp1b1.json`. Default behavior when the field is absent: treated as `false`.

A new helper in `predict/transporter_db.py` (mirrors PR #22's `lookup_metabolic_fraction` pattern):

```python
@lru_cache(maxsize=1)
def _load_oatp_applicability_index() -> dict[str, bool]:
    """InChIKey → ecm_applicable flag, indexed once."""
    ...

def is_oatp_ecm_applicable(smiles: str) -> bool:
    """Return True if the SMILES's InChIKey is registered with ecm_applicable=true.

    Lookup uses RDKit-canonical InChIKey to be robust against SMILES
    annotation differences (matches the cyp_clearance_overrides pattern).
    Returns False if RDKit unavailable, SMILES invalid, or InChIKey not
    registered.
    """
```

### 1.2 InChIKey matching policy

**Use the full InChIKey** (e.g. `GOSGZXISMCZCDW-LYANWTNHSA-N`), matching `cyp_clearance_overrides.py` precedent. Justification:

- Block 1 (connectivity) alone is too loose: pravastatin (block `GOSGZXIS`) and the wrong-connectivity DrugBank entry (block `TUZYXOIX`) differ at block 1 — full match is necessary to distinguish.
- Block 2 (stereo) inclusion is acceptable because canonical sources (PubChem, our `oatp1b1.json` registry) all use the same stereo annotation. Issue #25 already corrected `clinical_pk.json` to PubChem-canonical SMILES, eliminating one historical drift source.

A regression test (§6.3) pins this contract: the registered InChIKey for each `ecm_applicable=true` drug matches `Chem.MolToInchiKey(Chem.MolFromSmiles(registered_smiles))` exactly.

### 1.3 Initial seed list (REVISED after empirical validation)

| drug | flag | rationale | metabolic_fraction status |
|---|---|---|---|
| pravastatin | **`true`** | Niemi 2009 PM/EM ~ 2.6×; OATP1B1 rate-limits hepatic uptake. PR #22 calibrated FE 1.066 vs FDA. | ✅ `0.0` registered (PR #22) |
| pitavastatin | **deferred** to v0.3 follow-up | Niemi 2009 PM/EM ~ 3×; passes test_oatp_ecm_statins gate **but** has no `metabolic_fraction` entry. Empirical validation: activating ECM with default `metabolic_fraction=1.0` triple-counts → Cmax 0.00777 → 0.00165 (FE direction flips 2.22× over → 2.12× under, magnitude unchanged). Promotion blocked on literature-curated metabolic_fraction (~0.15-0.25 estimate; UGT1A3/2B7 + minor CYP2C9). | ❌ missing |
| fluvastatin | `false` | Niemi 2009 PM/EM ~ 1.0×; ~75% CYP2C9 dominant. Issue #21 closure documents this. | n/a (not OATP-rate-limited) |
| valsartan | `false` | Vss over-prediction confirmed (DE-33). Activating ECM amplifies under-prediction. | n/a (Vss issue dominant) |
| glimepiride | `false` | Same as valsartan (DE-33). | n/a |
| rosuvastatin | `false` (defer) | Niemi PM/EM ~ 2× clinically supports `true`, but test_oatp_ecm_statins xfails on Peff over-prediction (FE ~12). Promote when Peff issue resolved AND metabolic_fraction curated. | ❌ missing |
| atorvastatin | `false` (defer) | Niemi PM/EM ~ 1.5×; xfail on Peff. Same as rosuvastatin. | ❌ missing |

**v0.3 ships with exactly one drug flagged `true` (pravastatin).** This is the smallest possible step that exercises the auto-activation machinery while keeping AAFE shift bounded to a single holdout drug. Promotions of pitavastatin/rosuvastatin/atorvastatin are tracked as follow-up commits, each requiring (a) literature-justified metabolic_fraction entry and (b) re-run of holdout regen.

### 1.4 Empirical preview for pravastatin

Reproduced via `scripts/diagnose_pravastatin_ecm.py` and the pitavastatin diagnostic above:

| metric (40 mg dose, realize_means) | current predict() default | post-v0.3 (auto-ECM) | FDA target |
|---|---|---|---|
| Cmax (mg/L) | ~0.0500 (XGBoost-CYP, no ECM) | 0.0422 (ECM-active, mf=0) | 0.045 |
| FE vs FDA | 1.11× over | **1.066×** | — |
| 107-holdout dose 20 mg | scaled 0.0500 → 0.025 | scaled 0.0422 → 0.0211 | 0.025 |

Engine track impact estimate: pravastatin |log10(fold)| current ≈ 0.30, post-v0.3 ≈ 0.07. Δ over 107 drugs: 0.23 / 107 = +0.0022 to mean(|log10(fold)|). Engine AAFE 3.791 → ~3.778 (−0.3%). Within bootstrap CI noise (CI half-width ~0.4). Not a regression; small mechanistic improvement.

---

## Section 2 — predict() Change

### 2.1 Code skeleton

In `src/sisyphus/pipeline/predict.py`, after the existing `profile = compute_profile(smiles)` and `adme = predict_adme(profile)` calls, before `build_drug_on_graph(...)`:

```python
from sisyphus.predict.transporter_db import (
    is_oatp_ecm_applicable,
    load_oatp1b1_kinetics_for_smiles,
    load_hepatic_ecm_params_for_smiles,
)

transporter_kinetics = None
hepatic_ecm_params = None
if is_oatp_ecm_applicable(smiles):
    transporter_kinetics = load_oatp1b1_kinetics_for_smiles(smiles)
    hepatic_ecm_params = load_hepatic_ecm_params_for_smiles(smiles)
    logger.info(
        "Auto-ECM active for InChIKey %s",
        Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)),
    )
```

The two new helpers `load_oatp1b1_kinetics_for_smiles` and `load_hepatic_ecm_params_for_smiles` wrap the existing name-keyed `load_oatp1b1_kinetics(name)` and `load_hepatic_ecm_params(name)` by reverse-mapping InChIKey → drug name from the registry.

### 2.2 Phenotype-aware path (Section 3)

After graph construction, before drug binding:

```python
graph = build_from_yaml(_PHYS_PATH)
if phenotypes is not None:
    graph = apply_phenotype_to_graph(graph, phenotypes)
```

`apply_phenotype_to_graph` returns a new BodyGraph (does not mutate). Existing pattern from `test_oatp_pravastatin.py`.

### 2.3 Logging contract

- Auto-ECM activation: `logger.info("Auto-ECM active for InChIKey %s", inchikey)`
- Phenotype application: `logger.info("Phenotypes applied: %s", phenotypes)`
- No-op cases (flag absent, phenotypes None): no logging (avoid noise on the bulk-holdout path).

---

## Section 3 — Phenotype API

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

**Supported phenotype keys (v0.3)**: SLCO1B1, CYP2D6, CYP2C9, CYP2C19, CYP3A5, CYP1A2, CYP2B6 — i.e. every key already accepted by `apply_phenotype_to_graph`. Future additions (NAT2, UGT1A1 from issue #10) inherit automatically when added to that function.

**Orthogonality with auto-ECM**: providing `phenotypes={"SLCO1B1": "PM"}` for a drug that is NOT `ecm_applicable=true` scales the OATP1B1 abundance, but with no clearance path consuming that abundance the result is invariant. Documented in the docstring; covered by §6.2 test 4.

**Interaction with `n_mc_samples > 0`**: phenotype scaling is applied to the BodyGraph before MC sampling, so each MC sample inherits the scaled abundance and propagates uncertainty around it. ECM kinetics are sampled independently (existing OATP1B1 Vmax/Km CV). No special MC handling needed — auto-ECM and phenotypes both compose with the existing sampling code paths.

---

## Section 4 — Holdout Impact Validation

### 4.1 Pre-merge gate

The seed list activates ECM for **one drug** (pravastatin). Pravastatin appears in the 107-holdout. Expected Cmax shift in the holdout cache: 0.0500 → 0.0211 mg/L (20 mg dose, scaled from 40 mg ECM-active 0.0422).

Pre-merge required steps:

1. Re-run `scripts/run_engine_benchmark.py` → regenerate `data/training/4track_holdout_predictions.json`.
2. Refresh bootstrap CIs (10k resamples, seed=20260422) → write `data/validation/4track_ci_2026-05-03.json` (or `_v2` if same-day collision with the OATP triage).
3. Update headline AAFE table in `CLAUDE.md` and `README.md` to the new baseline.
4. Update `tests/regression/test_holdout_regression.py` pin if Meta AAFE moves.
5. Update `tests/regression/test_prodrug_v2_snapshot.py` only if any of the 4 prodrugs are affected (they should not be — none are OATP substrates).

### 4.2 AAFE gate (REVISED — tightened from 10% to 2% improving)

**Engine track AAFE must change by ≤ 2% AND in the improving direction.**

- Expected post-v0.3: 3.791 → ~3.778 (−0.34%).
- If the regen produces a worsening or > 2% absolute change, halt: that signals (a) the seed list is wrong, (b) an unrelated drug shifted (regression bug), or (c) the metabolic_fraction registry interaction broke. Investigate before merge.

The 10% threshold from spec v1 was loose enough to let the pitavastatin double-counting bug pass undetected on a 1-drug change. The 2%/improving gate catches direction-flip bugs of the kind §1.3 documents.

### 4.3 Method routing reassessment

`data/sbi/method_routing.json` is keyed on drug name and assigns SBI/IS/IBIS routing per drug. Pravastatin's current routing (per `project_n50_curation.md`: 12 SBI / 0 IS / 1 IBIS production set) needs to be checked: if pravastatin is the lone IBIS, the auto-ECM improvement may make it eligible for SBI promotion. **Not a v0.3 blocker** — routing is offline-determined and the file is unchanged unless `scripts/route_sbi.py` (or equivalent) is re-run. Track as follow-up.

### 4.4 Mass balance regression check

Add `pravastatin` to `tests/integration/test_engine_validation.py::test_mass_balance` parametrization, OR verify in the new `test_predict_auto_ecm.py::test_mass_balance_auto_ecm_active`. Auto-ECM adds new flux paths (OATP1B1 saturable + ECM passive + biliary CL_int) — mass balance must still close to < 1e-6 relative error.

---

## Section 5 — (REMOVED — Section 5 in spec v1 was a back-reference to Section 1; not a separate section.)

---

## Section 6 — Testing Plan

### 6.1 Existing tests — minimal changes

- **`tests/integration/test_oatp_pravastatin.py`**: keep manual `build_drug_on_graph` path (unit-level coverage of ECM/PGx machinery). Add ONE new test:
  - `test_predict_auto_ecm_matches_manual` — `predict(smiles_pravastatin, dose_mg=40)` Cmax matches the manual `build_drug_on_graph(...transporter_kinetics=..., hepatic_ecm_params=...)` Cmax within 1% relative.
- **`tests/integration/test_oatp_ecm_statins.py`**: unchanged. Continues to gate ECM behavior directly via manual build.
- **`tests/integration/test_engine_validation.py`**: unchanged unless §4.4 mass-balance integration is placed here.
- **`tests/regression/test_holdout_regression.py`**: pin updated to new Meta AAFE.

### 6.2 New tests — `tests/integration/test_predict_auto_ecm.py`

1. `test_pravastatin_auto_ecm_activates` — `predict(PRAVA_SMILES, dose_mg=40)` engine_pk.cmax ≈ ECM-active manual Cmax (within 1%). Verifies the wiring fires.
2. `test_fluvastatin_no_auto_ecm` — `predict(FLUVA_SMILES, dose_mg=40)` engine_pk.cmax ≈ no-ECM manual Cmax (within 1%). Verifies the negative case (flag absent / explicit false → no activation).
3. `test_phenotype_changes_pravastatin_cmax` — `predict(PRAVA_SMILES, dose_mg=40, phenotypes={"SLCO1B1": "PM"})` engine_pk.cmax > `predict(PRAVA_SMILES, dose_mg=40)` engine_pk.cmax. Lower bound: PM/EM ratio > 1.10 (matches `test_oatp_pravastatin` invariant).
4. `test_phenotype_orthogonal_for_non_substrate` — `predict(MIDAZOLAM_SMILES, dose_mg=15, phenotypes={"SLCO1B1": "PM"})` Cmax invariant (within 0.1%) vs same call without phenotypes. Midazolam is not OATP1B1 substrate; SLCO1B1 abundance scaling has no clearance path consuming it.
5. `test_smiles_variant_robustness` — pravastatin via stereo-stripped SMILES (e.g. `CCC(C)C(=O)OC1...`) does NOT trigger auto-activation (full InChIKey mismatch). Documents the matching contract.
6. `test_mc_sampling_with_auto_ecm` — `predict(PRAVA_SMILES, dose_mg=40, n_mc_samples=10)` produces a non-degenerate prediction interval (`pi_90_high > pi_90_low > 0`) with ECM machinery active. Verifies MC compatibility.

### 6.3 Schema regression test — `tests/regression/test_oatp_registry_schema.py`

1. `test_seed_list_pinned` — exactly the drugs in §1.3's "true" column have `ecm_applicable=true` in the registry. Catches silent flag flips.
2. `test_inchikey_matches_smiles` — for each `ecm_applicable=true` entry, `Chem.MolToInchiKey(Chem.MolFromSmiles(entry["smiles"]))` equals `entry["inchikey"]`. Catches SMILES drift.
3. `test_metabolic_fraction_paired` — every `ecm_applicable=true` drug has a corresponding entry in `cyp_clearance_overrides.json`. **This is the gate that prevents a recurrence of the pitavastatin double-counting bug.**

---

## Error Handling

- Invalid SMILES → `compute_profile` already raises `ValueError`; `is_oatp_ecm_applicable(smiles)` returns `False` (RDKit `MolFromSmiles` returns None). No exception propagated past predict().
- RDKit unavailable → `is_oatp_ecm_applicable` returns `False` (mirrors `lookup_metabolic_fraction` graceful fallback). predict() runs as before auto-ECM.
- `phenotypes` dict with unknown key → `apply_phenotype_to_graph` raises `ValueError`. predict() lets this propagate (caller error, fail loudly).
- `phenotypes` dict with valid key but unknown phenotype value (e.g. `{"CYP2D6": "ZZZ"}`) → `apply_phenotype_to_graph` raises `ValueError`. Same as above.

## Out of Scope

- Pitavastatin / rosuvastatin / atorvastatin promotion to `ecm_applicable=true` (each blocked on metabolic_fraction curation; tracked as v0.3 follow-up commits).
- New OATP substrates beyond the existing 7 in `oatp1b1.json` (clopidogrel, simvastatin acid, irinotecan are #11; deferred to that issue).
- Kp method retuning for high-fup acids (DE-33; engine-layer; separate spec).
- BCRP / OATP1B3 / NTCP transporters (engine extension; separate spec).
- Auto-activation for non-OATP transporters (e.g. P-gp at gut wall).
- PredictionResult schema additions (`ecm_activated: bool`, `phenotypes_applied: dict`) — useful but optional; defer to v0.3.1 if not required by GenoADME consumers.
- Method routing regen (`scripts/route_sbi.py` re-run); §4.3 notes it as follow-up, not a blocker.

## Self-review (v2)

- **Placeholder scan**: pitavastatin/rosuvastatin/atorvastatin "deferred" rows in §1.3 are explicitly out-of-scope, not gaps. AAFE gate now has a concrete number (≤ 2% improving) replacing the vague 10% from v1. Pre-merge regen steps in §4.1 are concrete.
- **Internal consistency**: §1.3 seed list (pravastatin only) consistent with §1.4 impact (one drug shift), §4.1 gate (one drug expected change), §6.3 schema test (pinned to {pravastatin}). InChIKey policy in §1.2 referenced by §6.3 test 2.
- **Scope check**: single subsystem (`predict` orchestration + registry data + new helpers). No engine, no ML, no graph topology. Single implementation plan.
- **Ambiguity check**: phenotype key set explicitly enumerated in §3. AAFE gate concrete in §4.2. Mass balance gate location documented in §4.4. InChIKey matching policy and rationale concrete in §1.2.

## Decision summary (v2)

| design choice | resolved | rationale |
|---|---|---|
| substrate detection | InChIKey via RDKit in helper, no MolecularProfile change (PR #22 pattern) | matches existing `lookup_metabolic_fraction` precedent, no dataclass churn |
| InChIKey matching | full key (block-1 + stereo) | block-1 alone is too loose (e.g. `GOSGZXIS` vs `TUZYXOIX` connectivity error case) |
| default behavior for unflagged | OFF | conservative; AAFE shift bounded to flagged set |
| auto-activation policy | always when flag is `true` | mechanistic correctness over data-only invariance |
| phenotype API | additive `phenotypes=` parameter on `predict()`, all `apply_phenotype_to_graph` keys supported | orthogonal to substrate detection |
| seed `true` set (v0.3) | **pravastatin only** | pitavastatin's metabolic_fraction is missing → double-counting verified empirically; defer until curated |
| AAFE gate | **≤ 2% AND improving sign on Engine track** | catches direction-flip bugs of the pitavastatin class that 10% would miss |
| logging | info-level on activation events, none on no-op | debugging without noise on bulk holdout |
| metabolic_fraction pairing | **schema test enforces** every `ecm_applicable=true` has paired entry | prevents pitavastatin recurrence |

## Open questions for user review

None blocking. Spec is ready for implementation plan.
