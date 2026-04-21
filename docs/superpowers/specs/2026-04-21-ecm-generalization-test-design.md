# ECM Generalization Test — Design Spec

**Date:** 2026-04-21
**Status:** Pre-registration draft (awaiting user review)
**Author:** Hypatia (session)

## Purpose

Verify that the Extended Clearance Model (ECM, merged `a60a14e` 2026-04-21) is a genuine general mechanism for OATP1B1-mediated hepatic uptake, not a mechanism implicitly fit to five statin chemistries. This is a **decorrelation test under hard pre-registration**: all substrates, parameters, success criteria, and failure modes are declared and committed BEFORE engine execution. No post-hoc tuning of any kind is permitted.

## Motivation

ECM was developed and calibrated against 5 statins (pravastatin, rosuvastatin, atorvastatin, pitavastatin, fluvastatin). Pravastatin drives the abundance calibration (`liver.transporters.OATP1B1 = 5.0e5` from Phase 1). If ECM generalizes, non-statin OATP1B1 substrates of similar chemistry must pass engine Cmax validation without re-tuning. If they systematically fail, ECM is statin-specialized and the 2026-04-21 merge's claim of "general hepatic clearance refactor" is overstated.

This test directly addresses the cherry-picking concern flagged in session 2026-04-21: 35+ per-drug error-cancellation failures are defensible only if the remaining joint fit is a true mechanism. This test operationalizes that question.

## Scope

**In scope:**
- Engine Cmax validation for 3 non-statin OATP1B1 substrates under IV dosing.
- Pre-registered domain of applicability declaration.
- Frozen per-drug literature Jmax/Km with CV propagation.
- 4-mode outcome taxonomy (pass / systematic-bias / inconclusive / fail).

**Out of scope:**
- Oral dosing (Peff confound deliberately excluded — see §Peff Isolation).
- DDI with OATP1B1 inhibitors (separate spec, reserved for #4 direction).
- SLCO1B1 phenotype extension (already validated at `a60a14e` for pravastatin; not re-tested here).
- Any ADME XGBoost retraining.
- TDM/SBI integration for the new drugs.

## Pre-registration Commitment

This section is binding. Any deviation from the declared protocol must be documented as a cherry-picking violation in the final report.

### Frozen Artifacts (created once, never modified after engine run)

1. `data/transporters/oatp1b1.json` — extended with 3 new drug entries (bosentan, valsartan, repaglinide) containing literature Jmax/Km and CV.
2. `data/validation/oatp_generalization_drugs.json` — IV dose, Cmax observation, citation for each drug.
3. This spec file (current document).
4. The implementation plan derived from this spec.

### Frozen Existing State (no modification permitted for this test)

- `src/sisyphus/engine/flux.py` — ECM branch in `ClearanceFluxSpec.apply`.
- `src/sisyphus/engine/compiler.py` — resolution logic.
- `data/physiology/reference_man.yaml` — OATP1B1 abundance (5.0e5).
- All ADME XGBoost models (fup, rbp, Kp, Peff, enzyme_affinity, ps_passive, ps_eff, cl_int_bile).
- `data/transporters/hepatic_ecm.json` — existing ECM parameters for 5 statins.

### Execution Constraints

- **Single engine run per drug.** Best-of-N is prohibited.
- **No parameter adjustment between run and report.** Including Distribution CVs, abundance, Jmax, Km, Peff, or any other field.
- **Single MC sample count (N=1000)** declared upfront, same as current pipeline default. No re-run at higher N to change PI width.
- **No substrate substitution after data extraction.** If literature data turns out insufficient for a chosen drug, the drug is dropped and the result reports `N_effective < 3`. No back-fill with an alternative drug.

## Substrates

### Selection Rationale

Three drugs chosen for maximal chemistry diversity from statins while remaining inside the declared domain of applicability (§Domain). Selection fixed before any engine execution.

### Substrate 1: Bosentan

- **Class**: Endothelin receptor antagonist (non-selective ETA/ETB).
- **MW**: 551.6 g/mol.
- **logD(7.4)**: ~2.7.
- **Ionization at pH 7.4**: Sulfonamide anion (pKa_sulfonamide ~5.5).
- **OATP1B1 Km**: 44 µM (Treiber 2007 DMD, HEK293-OATP1B1, triplicate).
- **OATP1B1 Jmax**: To be extracted from Treiber 2007 (pmol/min/mg transfected cell protein; scaling factor to pmol/min/10⁶ hepatocytes applied per Kunze 2014 review).
- **IV Cmax observation**: Weber 1999 JCP, single-dose IV infusion; dose + Cmax from Table 1. Cross-check with Dingemanse 2003 Clin Pharmacokinet.
- **Metabolism**: CYP3A4 (primary), CYP2C9 (secondary). Single-dose only — chronic dosing auto-induces CYP3A4 (Dingemanse 2004).
- **Rationale**: High-Km (low-affinity) OATP1B1 substrate; distinct sulfonamide ionization; tests ECM in the flow-limited + metabolism-heavy regime.

### Substrate 2: Valsartan

- **Class**: Angiotensin II receptor blocker (ARB).
- **MW**: 435.5 g/mol.
- **logD(7.4)**: ~0.9.
- **Ionization at pH 7.4**: Tetrazole + carboxylate di-anion (pKa_tetrazole ~4.7, pKa_carboxylate ~3.9).
- **OATP1B1 Km**: 1.4 µM (Yamashiro 2006 DMD).
- **OATP1B1 Jmax**: To be extracted from Yamashiro 2006.
- **IV Cmax observation**: Flesch 1997 Eur J Clin Pharmacol, IV doses 20-160 mg; dose + Cmax from reported tables.
- **Metabolism**: Minimal — ~80% unchanged in bile/feces, ~13% as metabolite M1 (valeryl-4-hydroxyvalsartan, inactive). No significant CYP involvement.
- **Rationale**: Low-Km (high-affinity) OATP1B1 substrate; di-anion ionization; minimal metabolic interference — tests ECM in the uptake-dominated regime.

### Substrate 3: Repaglinide

- **Class**: Meglitinide (insulin secretagogue).
- **MW**: 452.6 g/mol.
- **logD(7.4)**: ~3.3.
- **Ionization at pH 7.4**: Carboxylate anion (pKa_a ~4.2).
- **OATP1B1 Km**: 0.4 µM (Niemi 2005 CPT; also reported 0.35-0.5 µM range across studies).
- **OATP1B1 Jmax**: To be extracted from Niemi 2005 or Bidstrup 2003.
- **IV Cmax observation**: Hatorp 2002 Clin Pharmacokinet, IV 2 mg infusion.
- **Metabolism**: CYP2C8 (primary, ~60%), CYP3A4 (secondary, ~30%). Gemfibrozil-repaglinide DDI is well-characterized but not relevant here (no inhibitor co-administered).
- **Rationale**: Very low Km, highest-affinity OATP1B1 substrate in set; tests ECM in the near-saturated regime while maintaining a distinct meglitinide scaffold. Mixed metabolism exercises ECM's `cl_int_metab` leg.

### Diversity Axes Verification

| Axis | Bosentan | Valsartan | Repaglinide | Range covered |
|---|---|---|---|---|
| OATP1B1 Km (µM) | 44 | 1.4 | 0.4 | 2 orders of magnitude |
| Metabolism contribution | Heavy (CYP3A4) | Minimal | Moderate (CYP2C8) | Full range |
| Ionization | Sulfonamide mono-anion | Tetrazole + carboxylate di-anion | Carboxylate mono-anion | 3 distinct |
| Scaffold | Endothelin antagonist | ARB | Meglitinide | 3 distinct, none statin-like |
| MW | 551 | 435 | 453 | Compact, all inside domain |
| logD(7.4) | 2.7 | 0.9 | 3.3 | Full [-1, 5] domain range spanned |

## Domain of Applicability

ECM generalization claim applies **only** to drugs satisfying all of:

1. MW ∈ [300, 700] g/mol.
2. logD(7.4) ∈ [-1, 5].
3. Anionic at physiological pH (pKa_a < 6 for the primary acidic group).
4. OATP1B1 is a documented significant hepatic uptake pathway (literature evidence: OATP1B1 contribution to total hepatic uptake ≥ 30% from transfected-cell uptake ratio OR validated transporter DDI studies).
5. Single-dose administration (auto-induction / time-dependent inhibition excluded).
6. Cmax observation within 24 h post-dose.
7. No metabolite that confounds parent Cmax measurement via interconversion back to parent (e.g., glucuronide deconjugation-driven enterohepatic recirculation, lactone-acid ring equilibrium at magnitudes > 10%). Separately measured independent metabolites that do NOT feed back to parent do not count as confounds.

Failures on drugs **outside** this box are NOT counted as ECM failures (declared up front to prevent post-hoc domain narrowing). Failures on drugs **inside** this box ARE ECM failures.

All three chosen substrates satisfy all 7 criteria — verified in each substrate's listing above.

## Peff Isolation (Confound Control)

### Problem

Prior ECM validation (2026-04-21 merge) showed rosuvastatin and atorvastatin xfail due to Peff XGBoost over-prediction for high-MW polar acids, NOT ECM failure. If this test used oral dosing, Peff would be on the critical path to Cmax, and a Peff-related failure would be misattributed to ECM.

### Mitigation

- **IV dosing only** for all 3 substrates. Bolus or short infusion with clinical Cmax data.
- Dose goes to `venous_blood` administration node, bypassing absorption entirely.
- Peff, ka_fraction, solubility, particle_radius_um, gut enzyme contributions are **off the critical path** under IV. They may still load (uncertain sampling machinery is indifferent to administration route) but they do not affect Cmax since no mass flows through gut.

### Residual confounds

Even under IV:

- **fup prediction** still in path (affects ECM `fup * ps_inf * cl_int_h`).
- **Kp prediction** affects distribution phase and therefore Cmax timing.
- **Enzyme_affinity predictions** affect `cl_int_metab` → ECM `cl_int_h` → steady hepatic extraction.
- **rbp prediction** affects all organ C_out calculations.

These are accepted as inherent to Sisyphus's SMILES-to-Cmax architecture and are NOT considered part of the ECM generalization test's scope. A failure driven by (e.g.) bosentan's fup being severely miscalibrated would be logged but not classified as an ECM failure.

To quantify this, report **per-drug fup, rbp, Kp_liver point predictions** alongside each Cmax result for diagnostic transparency. If any of these sits >3× from a published experimental value, flag the drug as having a predict-layer confound and downgrade the ECM signal from that drug.

## Literature Data Extraction Protocol

For each substrate, extract and commit the following before any engine execution:

### Transporter kinetics (→ `data/transporters/oatp1b1.json`)

- **Jmax**: pmol/min/mg microsomal protein (preferred) OR pmol/min/10⁶ HEK293-OATP1B1 cells (converted via Kunze 2014 scaling factor).
- **Km**: µM, from HEK293-OATP1B1 initial uptake assay OR hepatocyte uptake (specify which).
- **CV**: minimum 0.30 (reported inter-study variance is usually 0.30-0.60 for these assays); 0.40 if reported CV < 0.30 (to absorb residual inter-lab bias). **Widen, do not narrow, relative to reported uncertainty** — this is the conservative direction.
- **Source**: primary citation + DOI + figure/table number.

### Clinical observation (→ `data/validation/oatp_generalization_drugs.json`)

- **Dose**: mg (IV route, bolus or infusion).
- **Cmax observed**: mg/L (converted from ng/mL if original in ng/mL).
- **Cmax CV**: reported inter-patient CV or, if single-patient, leave as null (flag in diagnostic output).
- **Patient N**: N of reported trial.
- **Source**: primary citation + DOI + table number.
- **Notes**: any relevant caveats (e.g., "dose-normalized from AUC-dose proportionality", "infusion duration 15 min", "fed vs fasted").

### Verification requirement

Before commit, visually cross-check Jmax/Km against at least one review (Kunze 2014 or Niemi 2009) to confirm values are in the reported inter-study range. If a primary value falls >3× from the review's mid-range, prefer the review midpoint and note the primary/review discrepancy.

## Pass/Fail Criteria

### Per-drug criterion

A drug **passes** iff BOTH:

1. Engine-predicted 90% PI contains the observed Cmax.
2. |log10(point_estimate / observed)| ≤ 0.48 (= FE ≤ 3.0×).

Point estimate = median of MC samples (consistent with P6 morphine resolution: median is robust to right-skewed lognormal posterior).

A drug **fails** if either condition breaks.

### Aggregate 4-mode taxonomy

For N=3 substrates, classify the outcome into exactly one mode:

- **Mode A (all-pass)**: 3/3 pass. Conclusion: ECM is confirmed as a general mechanism within the declared domain. Update `landmarks.md` and `experiment-log.md`.

- **Mode B (systematic bias)**: Either (i) 2/3 fail AND both failures have same sign of log10 FE AND |median log10 FE among failures| > 0.5, or (ii) 3/3 fail with same-sign log10 FE (which also triggers Mode D but Mode B takes precedence for classification when direction is consistent). Conclusion: systematic ECM bias discovered. Direction-appropriate post-hoc investigation is legitimate (e.g., if all over-predict, investigate whether ECM's ps_active formulation has a missing flow-limit term). Report Mode B outcome + direction + magnitude.

- **Mode C (inconclusive)**: Any failure pattern not matching Modes A, B, or D. Covers: single fail (regardless of direction or magnitude), 2/3 fail with mixed direction, 2/3 fail with same direction but |median log10 FE| ≤ 0.5. Conclusion: inconclusive — drug-specific confound likely. Report per-drug diagnostic table with predict-layer confound flags.

- **Mode D (all-fail)**: 3/3 fail AND failures are NOT same-direction (mixed signs, which would otherwise be Mode B). Conclusion: ECM is statin-specialized within the tested domain. Architectural review required. Do NOT revert the ECM merge (still valid for statins) but flag the generalization claim in CLAUDE.md.

**Precedence rule:** evaluate in order A → B → D → C (Mode C is the default fallback). This guarantees every outcome maps to exactly one mode.

### Cherry-picking violation conditions

The following, if observed, constitute violations that must be documented in the final report:

1. Changing any frozen artifact after the engine run.
2. Re-running with adjusted parameters to obtain better results.
3. Dropping a failed drug from the analysis post-hoc.
4. Reclassifying a failed drug as "out of domain" without the criterion being in this spec.
5. Averaging multiple runs to smooth variance.
6. Using MC N > 1000 or < 1000 (fixed at 1000).
7. Re-declaring success criteria after seeing results.

## Execution Protocol

1. Literature extraction and commit of frozen artifacts.
2. Spec review + plan approval.
3. Single engine run: `scripts/validate_oatp_generalization.py` (to be created in plan phase) invokes `predict → DrugOnGraph → engine → pk` for each of 3 drugs.
4. Result file: `data/validation/oatp_generalization_result.json` containing per-drug Cmax point, 90% PI, log10 FE, pass/fail, predict-layer confound flags, and classified mode.
5. Final report in `docs/claude/experiment-log.md` + memory update.

## Failure Handling (per Mode)

- **Mode A** — no follow-up. Update memory + log.
- **Mode B** — investigation authorized under new spec; do not proceed with ECM-based features outside domain until direction is understood.
- **Mode C** — inspect predict-layer diagnostics; if confound confirmed, re-run is permitted ONLY after fixing confound (and the re-run's outcome classification starts fresh).
- **Mode D** — halt all OATP generalization claims; ECM remains valid for statins only.

## Risks

1. **Literature extraction error** — manual data collection can introduce typos. Mitigation: cross-check against review articles, document each number's source to table/figure level.
2. **Sample size** — N=3 is small. Mitigation: acknowledged; conclusions scaled to N (Mode A claim is "3/3 pass" not "definitively general").
3. **Predict-layer confound** — fup/rbp/Kp misprediction could mask or mimic ECM signal. Mitigation: per-drug confound flags in output.
4. **Observed Cmax variance** — reported Cmax is mean of N patients with inter-patient CV. A single published Cmax is a noisy target. Mitigation: 90% PI containment criterion already absorbs this.

## Artifacts to Produce

- `data/transporters/oatp1b1.json` — extended (3 new drug entries).
- `data/validation/oatp_generalization_drugs.json` — new file.
- `scripts/validate_oatp_generalization.py` — execution script.
- `data/validation/oatp_generalization_result.json` — execution output.
- `docs/claude/experiment-log.md` — final entry.
- Memory update: `project_ecm_generalization_test.md` (new).

## Open Questions (resolve in planning phase)

1. Should the MC sampling use `sisyphus.engine.uncertainty.mc_sample` directly or go through the full pipeline (`pipeline/predict.py`)? Preference: full pipeline, so the test reflects production Cmax path including 4-track meta aggregation's engine component in isolation.
2. Is Kunze 2014's Jmax scaling factor appropriate for all 3 drugs, or do any require a drug-specific scaling (e.g., bosentan's reported Jmax in pmol/min/mg is from transfected cells)? Resolve during literature extraction.
3. For valsartan's di-anion, does the existing `DrugOnGraph` support representing two distinct pKa_a values, or is it truncated to one? Verify during `predict` layer inspection. If truncated, may be a predict-layer confound for valsartan specifically.
