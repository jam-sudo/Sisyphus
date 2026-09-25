---
last_updated: 2026-09-24
charter: Historical Cmax accuracy hypotheses and tests, with current corrections
---

# Accuracy Diagnosis

**Current status (2026-09-24):** The source-screened, repeatedly used development N=73 has Meta AAFE **2.8846** (conditional 95% bootstrap CI **2.3370–3.6113**) after the [felbamate unit correction](../validation/development_felbamate_source_correction_2026-09-24.md). The earlier-model, AI-assisted P0 N=18 scored **3.343**. The consumed AI-assisted [P1 diagnostic](../validation/blind_p1_result_2026-09-24.md), corrected to exclude a whole-blood Cmax label, scored **6.629** on seven plasma-matrix arms with an earlier model; dose-basis and formulation questions remain. None is an independent external accuracy estimate. Historical experiments below show many tested changes failed to improve the selected benchmark; they do not prove a universal architecture ceiling, an intrinsic CLint limit, or F as the dominant human PK error. The current [F source audit](../validation/f_reference_source_audit_2026-09-24.md) invalidated the ten-drug absolute-F calibration premise. An untouched, source-verified external cohort is still required for a new accuracy claim.

Before proposing an accuracy intervention, review the tested candidates in [dead-ends.md](./dead-ends.md) and specify how a new test differs.

---

## 1. Historical CLint prediction limits

- **XGBoost v1** on TDC Hepatocyte_AZ (1,213 compounds): R²=0.24.
- **v2** augmented to ~3,700 compounds: marginal improvement — target noise dominates.
- **ChEMBL expansion** (539 unique new compounds, 2026-03-27): scaffold CV R² 0.279→0.333 (+0.054). Engine AAFE +0.099, Meta AAFE +0.038 — **homogeneous data expansion destroys error cancellation**.
- **Foundation model shootout** (MoLFormer, ChemBERTa, Uni-Mol, frozen embedding + Ridge/MLP/XGB, 2026): Morgan FP + XGB (R²=0.205) outperformed the tested alternatives on that split. This does not distinguish a target-noise floor from a representation or dataset limit.
- **BDE features** (ALFABET, 978 compounds): r=+0.033 vs log10(CLint) — zero correlation. Hepatocyte CLint integrates kcat + Km + enzyme complement; C-H BDE captures only the kcat component.

These splits showed limited CLint predictability and poor Cmax transfer for the tested replacements. They do not establish that an intrinsic target-noise floor prevents all future prediction improvements.

## 2. Error cancellation

Cmax = f(fup, CLint, Peff, Kp, ...). Each ADME predictor has its own error, and the production pipeline is calibrated on the **joint** error profile inherited from Omega. Partial ADME replacements that improve one component's R² in isolation **destroy the joint calibration** and worsen pipeline AAFE.

Evidence:
- **ALL-ON** experiment (pKa + Berezhkovskiy + expanded CLint simultaneously): engine AAFE 2.945→3.016 (+0.072), meta AAFE 2.058→2.135 (+0.077). The individual harms sum.
- **Full predict replacement** (2026-03-30): CLint +0.033, fup +0.042, VDss +0.057 in R². Engine AAFE +0.165, Meta +0.023 worse.
- **Post-hoc meta-learner tournament**: 33 methods tested, all have error correlation r > 0.986 with the baseline. Mathematically, Engine + ML post-hoc combinations cannot break 2.277 (pre-VDss baseline).
- **ADME fup override** (2026-04-11): DrugBank measured fup prioritized over XGBoost. Principled, empirically harmful: engine AAFE +0.306. **35th error-cancellation failure.** Reverted.

## 3. Measured ADME PoC (Pattern C)

The original probe listed 12 drugs with measured fup + CLint and used the engine only (no meta). On the current source-screened reference, 10 have evaluable observed Cmax; excluding montelukast leaves nine. This is consumed development evidence.

**Current production rerun (2026-09-24):** `scripts/run_measured_adme_benchmark.py` reports engine AAFE **6.708 → 5.968** (SMILES-only → measured fup+CLint, N=10), or **5.514 → 4.783** excluding montelukast (N=9). Clozapine and abiraterone lack current Cmax arms and are skipped. Two label-backed absolute-F point examples score **4.083 → 2.619** with measured F, but N=2 and clinical contexts are unmatched; this does not validate measured-F improvement. These numbers replace the earlier clean-10 2.63→2.33 as a *current* claim.

**Historical 2026-06-02 reconciliation.** The earlier "AAFE 2.329 → **1.980**" figures came from a stale engine state. At that checkpoint, re-running `scripts/measured_adme_poc.py` gave clean-10 **2.81 → 2.69**. The following production figure also belongs to that checkpoint, not the current source-screened reference:
- **Historical production `predict(measured_adme=fup+clint)` result:** clean-10 SMILES 2.63 → measured 2.33 on the former arms and engine state. It is not reproduced by the current source-screened reference and must not be used as a present accuracy figure.
- The legacy fup-matched (1.91→1.79) / fup-corrected (5.15→2.96) subgroup splits were tied to the stale state and are not re-derived.

**Conclusion:** The current small, selected probe shows a lower engine AAFE with measured fup+CLint, but it cannot establish a general accuracy gain. Some individual compounds worsen, so input substitution and model calibration must be evaluated together on independent clinical arms.

## 4. The VDss exception — when a new track *is* OK

2026-04-10: VDss analytical 4th track dropped Meta AAFE from 2.808 to **2.695 (−4.0%)**. The original "partial replacement is impossible" conclusion is **falsified in the limit**.

**Why VDss worked where CL/F · t½ failed:**
- CL · t½ · Cmax all depend on hepatic clearance / CYP-dominant kinetics → errors correlate across tracks.
- VDss depends on tissue partitioning (lipophilicity + tissue binding) → **clearance-orthogonal** error component.
- dose / (Vd · BW) 1-compartment analytical Cmax at 20% weight scales down the 3 existing tracks to 0.80 and adds uncorrelated signal.

**Decorrelation criterion (for future track proposals):** measure per-drug error correlation (Pearson r on log Cmax residuals) between the candidate track and the existing 4 tracks. Only consider tracks with |r| < 0.5 against all 4 existing tracks. This gate precedes any integration work.

## 5. Direct CL/F · t½ — confirmed negative path

- **Direct CL/F 3rd track** (IVIVE bypass, 2026-03-27): CV R²=0.232, analytical 1-cpt Cmax. LOOCV w_clf = 0.00 (both base and other regions). Standalone AAFE 3.133 (ML 2.336 wins). Meta AAFE Δ = −0.005 (noise). Oracle 3-track 1.788 (28/107 drugs CL/F is best individually) — but no fixed weight unlocks it.
- **Post-VDss direct CL/F · t½ predictors** (6 variants): all negative, `data/validation/post_vdss_negative_results.json`. Falsifies "IVIVE bypass is what made VDss work" — the real reason is **decorrelation**, not bypass.

## 6. Practical next evidence

An accuracy proposal needs a stated clinical-arm target, source-verified matched inputs, and an untouched external evaluation. The prior error-decorrelation gate can reject redundant candidate tracks on development data, but passing it would not itself establish generalization. Observation-conditioned TDM remains a separate research capability requiring its own clinical validation.

## 7. Population-level AAFE ≤1.7 evaluation

No current independent external test supports population-level Cmax AAFE ≤1.7 from SMILES alone. Historical CLint splits and failed Cmax interventions suggest this would require a substantial new source of predictive information, but do not prove impossibility. TDM Bayesian updating is a separate, observation-conditioned research setting and does not establish population-level structure-only accuracy.

---

## 8. Novel-drug underprediction and the unverified F hypothesis (2026-06-01)

**2026-09-24 correction:** The dated F diagnosis below is a research hypothesis,
not a verified root cause. Its ten "literature F" anchors were not a valid
human absolute-F cohort; the [source audit](../validation/f_reference_source_audit_2026-09-24.md)
retains only three unmatched label references, with engine/reference ratios
0.98, 0.42, and 0.20–0.23. The asserted 10/10 systematic under-call and
0.46–0.51 median are withdrawn. Moreover, literature oral **CL/F** cannot be
equated with systemic **CL**, so that comparison does not isolate F from
clearance. Prospective Cmax underprediction and absorption-scalar experiments
remain observed results, but their physiological attribution and the claim that
measured F is the only remaining lever need a matched oral/IV human cohort.
The following dated paragraphs preserve the original reasoning and should be
read under this correction.

The 2026-06-01 prospective expansion (N=28; prospective Meta AAFE 3.21 > retrospective 2.698) exposed severe underprediction for some 2025 NMEs (mirdametinib 30×, sevabertinib 18×). The original IV-vs-oral engine decomposition suggested an absorption/first-pass hypothesis, but did not verify its human cause:

- Engine systemic CL was reported as 4.8 L/h for mirdametinib, while the cited 4.6 L/h is **oral apparent CL/F**. These cannot establish that systemic clearance is correct or separate clearance error from F error.
- Engine F was 0.08 for mirdametinib and 0.05 for sevabertinib. No matched human absolute-F observations were established here, so the proposed real F ≈ 0.5–1.0 and the claim that F explains the entire Cmax gap are unsupported.
- On the prospective new-16, corr(engine_F, |log10 fold|) = −0.54; this association does not identify the causal PK parameter. The engine track (5.10) scored worse than direct ML (3.40) on those drugs.

**Refinement to §1:** prospective underprediction exceeds what the retrospective CLint model assessment alone explains. Whether absorption, first-pass extraction, systemic clearance, formulation, or their combination dominates requires matched human oral/IV and clinical-arm evidence.

**No tested predict-time AD signal recovered the error** (see dead-ends.md DE-41): low predicted-F and engine↔ML divergence both correlate ≈0 with error on the consumed holdout (the engine predicts low F for nearly everything — median 0.18 — so it is not discriminative). Measured-F routing is an available input path; its accuracy effect has not been established by a matched external cohort.

**2026-06-03 intervention result (dead-ends.md DE-42):** `ka` enters the tested ODE linearly. Uniformly scaling its constant by 5.25× improved historical engine N=107 AAFE 3.831→3.336, but the unrefitted meta regressed about 3%, and a refined candidate scored 3.405 on that engine set. The attempted calibration target—median engine/literature F 0.46→1.0—was invalid because the ten F references were not comparable human absolute-F measurements. These experiments reject that particular global scaling as a scored-meta improvement; they do not establish a general absorption or first-pass mechanism, nor foreclose a new mechanism supported by matched human data.

**Later intervention checks:** FLUX-1 corrected a flow-limitation double count in the engine; on a historical consumed benchmark the Meta AAFE moved 2.762→2.784, and a later N=28 prospective-set re-score moved 3.21→3.29. Correcting the implemented equation did not improve those Cmax scores. DE-43 tested absorption and gut-CYP3A scaling: some engine-only scores improved, while the fixed meta was neutral or worse. These are outcomes for specific model versions and interventions, not proof that absorption is the dominant human error or that all future engine changes are ineffective. Detailed dated experiments remain in [experiment-log.md](experiment-log.md) and [dead-ends.md](dead-ends.md).

---

## 9. External benchmark context

An earlier comparison with the IMI OrBiTo expert-harmonised PBPK evaluation was used to claim that Sisyphus had reached a commercial accuracy ceiling. That claim is unsupported: the study populations, input curation, endpoints, and evaluation sets differ, while Sisyphus's current 2.8846 AAFE is from repeatedly used development data. The earlier-model P0 pilot scored 3.343 and also lacks independent curation. Published PBPK results provide context for study design, not a matched head-to-head rank or a bound on achievable error. Independent source-verified evaluation remains necessary.

---

## 10. The label-noise estimate and failed added-track tests (2026-06-08)

A full layered analysis (`layered-analysis-and-leap-2026-06-08.md`) ran four candidate levers against the §4 decorrelation gate. Two structural facts emerged that refine this document.

**(a) The Cmax label-noise floor is estimated at AAFE ≈ 1.18, not ~2.** §1/§2 and dead-ends.md line 32 framed the residual as "≈ experimental + formulation + inter-patient variability." Fourteen same-drug/same-dose study-replicate pairs in `mmpk_expanded_full.csv` gave a between-study geomean fold difference of 1.261. Under independent, identically distributed normal log errors, that implies a single-label noise AAFE of **1.18**; a broader plausible band was 1.18–1.5. The corresponding estimate of label-noise variance is **3–16%** of the historical model-error variance (σ_total=0.557 log10 units at AAFE 2.784). **A perfect latent-Cmax predictor would score near the label-noise floor, not 2.56–2.75.** The latter was an estimate of the *current model's residual error after hypothetically removing label noise*, assuming independent Gaussian errors. Thus the data suggest material model-side error, but N=14 and the noise assumptions limit the precision of any floor or headroom estimate.

**(b) The tested added tracks did not decorrelate Cmax errors.** Dose-number (DE-45), renal-secretion (DE-46), and a measured-ADME ML regressor (DE-47) failed their respective gates; on N=93, the measured-ADME track's residual correlations were +0.69/+0.78/+0.79 against engine/direct-ML/meta. In a separate representative measured-input probe, engine AAFE was ~3.84 versus meta 2.78, and measured-regime engine up-weighting worsened accuracy (DE-48). These results reject those specific tracks and weights. The proposed shared bioavailability-F mechanism remains a hypothesis: the §8 human absolute-F evidence was invalidated, and correlated Cmax residuals alone cannot identify F as the cause. A new measured input or mechanism must be tested on matched clinical arms and then on an untouched external cohort before it can support an accuracy claim.

---

## See also

- [dead-ends.md](./dead-ends.md) — failed-intervention records and their current corrections.
- [experiment-log.md](./experiment-log.md) — chronological record of experiments, successes and failures.
- `docs/breakthrough_path.md` — UDE roadmap (Phase 1 falsified; Phase 2 / 3 pending).
- `docs/holdout_contamination_audit.md` — the 2026-04-04 leakage discovery and fix (AAFE 2.283 → invalidated).
