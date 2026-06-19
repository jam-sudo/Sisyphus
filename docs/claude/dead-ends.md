---
last_updated: 2026-06-03
parent: ../../CLAUDE.md
charter: Authoritative list of failed Sisyphus experiments. Read before proposing any accuracy improvement.
---

# Dead Ends — Do Not Retry

Every experiment here was run, reverted, and documented. **Before proposing any accuracy improvement, open this file and search for the approach.** New track proposals must first pass the error-decorrelation gate described in [diagnosis.md §4](./diagnosis.md).

**Canonical count:** 43 enumerated experiments below. Narrative references in commit messages or prose (e.g. "#35 error cancellation", "14번째 시도", "누적 33 methods") use **informal** numbering that counts early exploration attempts separately; those narrative numbers are **not authoritative** and do not match the table count below. When in doubt, cite the table entry (`DE-NN`).

## 1. Theme summary (14 categories)

| Category | Representative entries | Headline outcome |
|---|---|---|
| Post-hoc meta-learner variants (33 method-combinations total) | DE-23, DE-24, DE-30 | error correlation r > 0.986 with baseline; provably near-optimal already |
| CLint R² improvement (14 attempts) | DE-08, DE-11, DE-13, DE-16, DE-17, DE-18 | R² gains 0.02–0.12 achievable, but all destroy pipeline error cancellation |
| Foundation models (MoLFormer / ChemBERTa / Uni-Mol) | DE-14 | Morgan FP + XGB dominates every frozen-embedding combination |
| Docking features for CLint | DE-13 | ΔR² = +0.005 (noise); binding affinity ≠ metabolic rate |
| UDE / gradient-through-solver | DE-29 | residual CV R² < 0 (falsified); Phase 2/3 unexecuted |
| E2E Neural PK (Pharos, E2E MLP) | DE-05, DE-17 | data scale insufficient; GNN needs >>5k Cmax |
| Training-data expansion (ChEMBL / DrugBank / Biogen) | DE-09, DE-11 | breaks error cancellation |
| ADME replacement (partial or full) | DE-18, DE-31, DE-15 | 18+ error-cancellation regressions |
| Class-aware / batch-specific meta weighting | DE-25 | kinase under-prediction diagnosed; no weight combo beats baseline |
| F% bioavailability predictor | DE-26 | trained, negative; VDss-style unlock does not apply |
| Direct CL/F + t½ predictors | DE-27, DE-28 | CL/F R²=0.232 + t½ variants all negative; falsifies "IVIVE bypass" as the reason VDss worked |
| Hepatic intracellular fu correction (PPB-targeted) | DE-37 | Phase A infra shipped; primary literature corpus paywall-locked, 4 PPB candidates dispositioned ceiling_accepted, Meta AAFE shift 0.0% |
| UGT path / abundance / IVIVE interventions | DE-36, DE-38, DE-39, DE-40 | four consecutive metric-neutral UGT cycles; no per-substrate hepatocyte-basis scaling factor exists; ΔMeta AAFE ≤ 0.003 |
| Absorption / first-pass bioavailability-F recalibration | DE-41, DE-42, DE-43 | engine F under-call is bidirectional first-pass dispersion (not absorption); the absorption knob is linear = a flat scalar; and the fixed-weight meta damps **any** engine recalibration to ~18% pass-through on **both** the retrospective and prospective sets — the engine is not a headline lever on any benchmark |

**Root cause (shared across categories):** Cmax residuals are not learnable from molecular structure (CV R² < 0). Remaining error ≈ experimental variability + formulation + inter-patient variability. SMILES → Cmax carries a fundamental information-channel ceiling.

---

## 2. Enumerated experiments

Each entry: ID, date, what was tried, outcome (numeric), why it failed, telltale sign if the idea returns under a new name.

### DE-01 — fup re-training (DrugBank + TDC)
AAFE ± 0.02, noise level. fup XGBoost alternatives do not move the pipeline.

### DE-02 — logP residual correction
AAFE ± 0.02, noise level.

### DE-03 — IVIVE chain ensemble (R&R/PT × WS/PT, 4 chains)
Negative result.

### DE-04 — UGT metabolism enabled in engine (pre-v0.3.2 pipeline)
Engine AAFE 2.861 → 3.090. Revert. **Re-measured 2026-05-13 under current pipeline as DE-36 — prior conclusion does NOT generalize; today UGT path is mildly Engine-positive and Meta-neutral.** See DE-36 for the refreshed measurement; DE-04 retained for historical record only.

### DE-05 — E2E differentiable MLP
Holdout 3.265. N=65 insufficient to learn a full SMILES→Cmax map.

### DE-06 — MMPK CLint deconvolution
R²=0.166. Apparent CLint not learnable from molecular features alone.

### DE-07 — Transporter scaffolding (pre-Phase-1)
Quantitative kinetics absent at the time; zero drugs active. Superseded by Phase 1 OATP1B1 (2026-04-15).

### DE-08 — pKa XGBoost model (DrugBank 9,974, R²=0.79, MAE 1.6) as engine input
Engine AAFE +0.005 (noise), meta AAFE 2.058 → 2.153 worse. Error cancellation destroyed. Revert.

### DE-09 — Berezhkovskiy Kp correction
Engine AAFE +0.021, meta 2.058 → 2.067 worse. Revert.

### DE-10 — pKa + Berezhkovskiy combo
Engine AAFE +0.021 (noise). Kp is not the engine-error driver.

### DE-11 — CLint expansion (Hep_AZ 986 + Mic_AZ 420 = 1,402 compounds)
CV R² 0.229 → 0.273 (+0.044). Engine AAFE 2.945 → 2.930 (−0.015), meta AAFE 2.058 → 2.110 (+0.052 worse). Error cancellation broken. Revert.

### DE-12 — ALL-ON (pKa + Berezhkovskiy + expanded CLint, simultaneously)
Engine AAFE +0.072, meta +0.077. Individual harms sum. Simultaneous improvement cannot establish a new balance. Revert.

### DE-13 — CYP docking features (DiffDock NIM + Vina)
DiffDock CYP3A4 1,114 drugs: CLint CV R² 0.190 → 0.196 (ΔR²=+0.005, noise). Vina: ΔR² = −0.026 (worse). Docking feature importance 0.2–0.4%, top-30 has zero docking features. **Binding affinity ≠ metabolic rate.** Do not retry.

### DE-14 — Foundation model shootout (MoLFormer / ChemBERTa / Uni-Mol)
Frozen embedding + Ridge / MLP / XGBoost, every combination. **Morgan FP + XGB R²=0.205 dominates every alternative** (MoLFormer mean 0.184, ChemBERTa 0.170, Uni-Mol 0.083). Ensembling also worsens. CLint R²≈0.20 is a **target-noise** ceiling, not a representation ceiling.

### DE-15 — Direct CL/F 3rd track (IVIVE bypass, 2026-03-27)
MMPK AUC → CL/F direct prediction (N=1,014), Vd/F inverse (N=940). CL/F XGB CV R²=0.232, Vd/F R²=0.332. Analytical 1-cpt Cmax. 3-track LOOCV: w_clf=0.00 (base / other both). Standalone AAFE=3.133. Meta Δ=−0.005 (noise). Oracle 1.788 across 28/107 drugs but not unlockable with fixed weights. Benet hypothesis ("IVIVE bypass → accuracy gain") **not verified**. Infrastructure retained, w_clf=0.00.

### DE-16 — ChEMBL CLint expansion (2026-03-27)
ChEMBL 36 all-extract: 539 unique compounds (534 net new). TDC Hep 978 + ChEMBL 517 = 1,910 compounds. Scaffold CV R² 0.279 → 0.333 (+0.054). Engine AAFE 3.416 → 3.515 (+0.099 worse), meta AAFE 2.277 → 2.316 (+0.038 worse). LOOCV w_base 0.45 → 0.25 (meta-learner loses confidence in engine). Revert. Data archived under `data/chembl/` and `data/training/clint_expanded_v2.csv`.

### DE-17 — CLint 3-class classification (2026-03-29)
Low / Med / High (10/50 cutoff), XGB classifier accuracy 53.5% (kappa 0.299, scaffold CV). Probability-weighted MC mixture. Engine AAFE +0.108 worse; Meta AAFE 2.277 → 2.255 (Δ=−0.023, **marginal noise-level improvement**). w_base=0.45 retained. Coarser prediction destroys less error cancellation, but the effect is within noise.

### DE-18 — BDE reactivity features (2026-03-29)
ALFABET BDE on 978 compounds. BDE_min vs log10(CLint): r=+0.033 (no correlation). CYP subset: r=+0.043. **Gate |r|<0.15 failed.** Hepatocyte CLint integrates kcat + Km + enzyme complement; C-H BDE (CYP kcat-only) cannot explain it.

### DE-19 — Pharos v0 E2E prototype (2026-03-29)
GNN encoder + MoE(K=3) + 1-comp PK backbone. 3,551 compounds, 1,074 with Cmax. Best AAFE=3.006 (GNN+MoE), worse than Sisyphus ML-only 2.336. 465K parameters vs 1,074 samples (ratio 433:1). **Data scale, not architecture, is the bottleneck.** GNN needs >>5,000 Cmax samples. Branch: `pharos-prototype`.

### DE-20 — CLint descriptor upgrade (2026-03-30)
Feature selection top-300 + Optuna: CLint scaffold CV R² 0.279 → 0.399 (+0.120). Holdout Meta AAFE +0.012 (17th error-cancellation regression). Regularization is not the ceiling; data quality is.

### DE-21 — Full predict replacement (2026-03-30)
All ADME models re-optimized simultaneously. CLint +0.033, fup +0.042, VDss +0.057 in R². Engine AAFE +0.165, Meta AAFE +0.023 worse. Partial OR whole replacement fails under the current pipeline.

### DE-22 — ML Mordred features (2026-03-30)
Mordred 1,613 descriptors + ensemble (XGB + LGB + Ridge). CV AAFE 3.410 < Morgan 3.750, but Holdout AAFE 2.848 > Morgan 2.336. At N≈1,100, dense features CV-overfit.

### DE-23 — Delta model / MOS (2026-03-31)
log10(Cmax) = log10(Engine) + Delta(features). Delta variance 46% of Cmax variance. Holdout: Delta-only 3.528, Delta+ADME 8.450 (catastrophic overfit). Engine error is non-systematic → ML correction impossible.

### DE-24 — k-NN read-across (2026-03-31)
Morgan FP Tanimoto (median 0.464), k=20 similarity-weighted: AAFE 3.049. 3-way blend w_knn=0.00. r(ML, kNN) = 0.690 (correlated errors). Oracle 3-track 1.689 (28/107 drugs kNN best) but no fixed weight exploits it.

### DE-25 — Post-hoc meta-learner (2026-04-01)
OOF Stacking (Ridge), ACF (Analog Correction Factor), Winsorized — 6 variants. **None beat baseline meta 2.277.** Stacking V1: 2.420 (OOF-Full gap r=0.81 destroys transfer), ACF k=5: 3.005 (neighbor fold-error std 0.67, noisy), Winsorized cap=0.5: 2.300 (same). Stacking+ACF combined — no effect. 23rd negative.

### DE-26 — 10-method meta-learner tournament (2026-04-01)
5 PK-domain + 5 cross-domain. Every entry: Isotonic Engine Cal. (+0.325), ER-Proxy Routing (tie), Error Direction Clf (+0.055, 64.2% acc), CLint-Stratified (+0.006), AAFE-Direct Optim (+0.082), Quantile XGB (+0.602), Local BMA (+0.081), Caruana Ensemble (+0.090), Disagree-Sigmoid (+0.014), Trimmed AAFE (+0.097). **All 10 have error correlation r>0.986 with baseline.** Compound-type-adaptive geometric blend is provably near-optimal. 24th negative (cumulative 33 methods).

### DE-27 — Kinase class-aware meta weights
After diagnosing kinase batch under-prediction, `scripts/class_aware_meta_benchmark.py` swept kinase-class weights. Meta AAFE 2.277 tie; 1,765-cell weight cache generated; no combo beats baseline. `data/validation/class_aware_meta_results.json`.

### DE-28 — F% bioavailability predictor
DrugBank 527 drugs, XGB (`scripts/train_bioavailability.py`). Standalone + meta integration both negative. `data/validation/f_predictor_negative_result.json`. F% does not unlock error cancellation the way VDss did.

### DE-29 — Direct CL / half-life predictors (post-VDss, 6 variants)
`xgboost_clearance_v1.json` + `xgboost_thalf_v1.json`. 6 combinations all negative. `data/validation/post_vdss_negative_results.json`. **Falsifies the interpretation that "VDss's IVIVE bypass" is what made VDss work** — the real reason is clearance-orthogonality (see [diagnosis.md §4](./diagnosis.md)).

### DE-30 — UDE prototype Phase 1 (Diffrax gradient-through-solver)
Residual learning. `data/validation/phase1_ude_prototype_result.json` records the falsification. Residual not learnable from molecular structure (CV R² < 0). Phase 2 (amortized SBI) and Phase 3 (flow matching) unexecuted.

### DE-31 — ADME fup override (2026-04-11)
DrugBank measured fup always preferred over XGBoost prediction (inverting the >5× disagreement fallback). Principled, empirically harmful: Engine AAFE 3.421 → 3.726 (+0.306 — the **34+ error-cancellation failure pattern** reproduced), Meta AAFE 2.695 → 2.728 (+0.033 noise). Revert. Narrative "35th error cancellation failure" entry.

### DE-32 — SBI v3 OATP training expansion (2026-04-14)
To fix pravastatin SBC (cov_dev 0.223), OATP1B1 substrates added: atorvastatin, fluvastatin, pitavastatin, valsartan, bosentan. 55 → 60 drugs. Result: pravastatin 0.223 → 0.237 (worse), posaconazole 0.073 → 0.173 (much worse, SBI → IBIS regression). SBI 12 → 11. **Pravastatin failure is engine-level (OATP1B1 not modeled), not training-data.** Revert. Narrative "36th failure" entry.

### DE-33 — ECM fup override for V3 OATP underprediction (2026-04-22)
Valsartan fup predicted 0.009 vs DrugBank/clinical 0.050. Hypothesized this 5.6× deficit drove V3 Cmax underprediction (FE 0.48× on valsartan, 0.39× on glimepiride under V3 windowed IV-Cmax). Override: Cmax changed by 0.97× — essentially unchanged. Glimepiride predicted fup already matches clinical. **fup RULED OUT as cause of V3 OATP non-statin underprediction.** `scripts/diagnose_v3_underpredict.py`, result `5ff72eb`. Candidates remaining (not tested): Jmax calibration, Vss/Kp over-distribution, ECM architecture limit outside statin Km range. Do not re-test fup override for this class.

**2026-05-03 root cause confirmed: Vss over-prediction.** Post-Hardening diagnostic with `realize_means()` (issue #8/#21 closure session): XGBoost VDss predictor over-distributes both drugs by 3-6× (valsartan pred 107.8 L vs FDA 17 L; glimepiride 28.5 L vs FDA 8.8 L). Direct mechanism for Cmax under-prediction: IV Cmax ~ dose/Vss → 6× over-Vss → ~2× under-Cmax (residual absorbed by distribution kinetics). Jmax calibration ruled out as primary contributor (would not affect C0 IV behavior). Engine uses Kp (Rodgers-Rowland), not Vss directly, so a registry-only Vss override would be architecturally inconsistent. Wholesale Kp recalibration for high-fup acid class is engine-layer work, deferred. Re-running fup override post-Hardening would still hit the DE-31 error-cancellation trap. **Structural limitation acknowledged; not actionable without coordinated Kp method retuning.**

### DE-34 — 3D conformer descriptors for ML Cmax (2026-04-01)
20 RDKit 3D descriptors (asphericity, NPR1/2, PMI1/2/3, WHIM, etc.) on ETKDG-generated conformers (99.8% success). Two evaluations, both falsified:
- **N=1029 holdout-excluded** (`feature/3d-cyp-multidist` Experiment A): Morgan+3D ML AAFE 2.930 vs Morgan-only 3.030 (Δ=-0.100, first gate PASS); meta blend 2.818 vs production-then meta 2.277. Orthogonality r=0.655 with Morgan, insufficient to clear ensemble error cancellation. Archive tag `archive/3d-cyp-multidist-2026-04-01`.
- **N=1128 production-scale retest** (`feature/morgan3d-retrain` Phase 12c, 2026-04-01, 324 hyperparameter configs): Morgan+3D ML AAFE 2.402 vs Morgan-only 2.341 (Δ=+0.061, **WORSE**); meta 2.370 vs production 2.277 (Δ=+0.093). Error correlation r=0.952 — at production training scale, the 3D features lose their pseudo-orthogonality. 3D feature share = 6.9% importance, only 3 in top-20 (whim_5/6/7). Archive tag `archive/morgan3d-retrain-2026-04-01`.

The two evaluations together rule out "but it might work at production scale" as an escape: the orthogonality observed at N=1029 was a small-data artifact, not a feature-engineering signal. Telltale if it returns: "asphericity / NPR / PMI / WHIM / 3D shape descriptors / conformer features" added to ML feature set.

### DE-35 — Differentiable engine surrogate for E2E SMILES→Cmax (2026-04-02)
Distinct from DE-30 (UDE / gradient-through-solver): pre-train an MLP operator to mimic the engine, then fine-tune SMILES→Encoder→Operator→Cmax end-to-end with frozen operator. Two-gate evaluation:
- **Gate 1 — operator approximation: PASS.** MLP (128, 64) on 20K synthetic ADME→Cmax pairs, R²=0.9985, fold error 1.09×. Top features: dose 62%, Peff 24%, CLint 7%. Engine is perfectly approximable as a pure function of physiological inputs.
- **Gate 2 — E2E fine-tuning: FAIL.** Scaffold-CV AAFE 3.544 vs XGBoost 3.369 (+5.2% worse). Error correlation r(E2E, XGB)=0.867 — somewhat orthogonal, but E2E too inaccurate to contribute. The 12-D latent bottleneck through the physics operator over-constrains the model at N=1,239 training samples.

Mechanism: the operator captures engine physics perfectly, but the engine's systematic bias (AAFE 3.42) transfers through unchanged — fine-tuning with N=1,239 cannot correct it via the encoder. Same SMILES information ceiling as DE-05/DE-17/DE-30 reached from a different architectural angle. Branch `feature/neural-operator-surrogate` (commit `b85b18d`); archive tag `archive/neural-operator-surrogate-2026-04-02`. Telltale if it returns: "differentiable surrogate / amortized engine / pre-trained operator + encoder fine-tuning" with N < 5K Cmax training data.

### DE-36 — UGT fm redistribution re-measurement (2026-05-13)
**Refresh of a prior unrecorded sensitivity test** that had concluded "UGT fm redistribution degrades Engine AAFE 2.861 → 3.090" (cited as a comment in `src/sisyphus/predict/ivive.py` pre-2026-05-13). That measurement was pre-v0.3.2 + pre-public-only-headline + pre-ECM-auto-activation; current pipeline is materially different. Re-measured under current main + DrugBank-present:

- **Engine (overall N=107):** 3.791 → 3.762 (**−0.029**, marginally helpful)
- **Meta (overall N=107):** 2.679 → 2.679 (**+0.0002**, invariant — error cancellation)
- **Engine (in-domain N=79):** 3.466 → 3.440 (**−0.026**)
- **Per-drug (16 shifts ≥5% on Engine):** 11 improved (dapagliflozin FE 15.8 → 13.7, etodolac 8.4 → 7.0, ketorolac 7.1 → 5.8, metronidazole 10.6 → 9.8, glasdegib 4.0 → 3.2 — mostly UGT-substrate NSAIDs / gliflozins / etc that were under-predicting); 5 worsened (codeine 2.0 → 2.4, morphine 1.9 → 2.1, losartan 2.2 → 2.5 — over-predicting drugs got more over-prediction).

Old narrative ("UGT path is harmful") **does not generalize** to the current pipeline. New finding: UGT path is mildly Engine-positive and **headline-neutral**. The Engine improvement gets re-absorbed by the 4-track meta-learner's track weights (DE-08~DE-18 error-cancellation family).

Not activated in production because: (a) zero Meta benefit, (b) UGT annotations sourced only from DrugBank → public-clone reproducibility would require a curated `data/enzymes/ugt2b7_substrates.json`-style registry (separate cycle, parallel to the v0.3.2 NAT2/UGT1A1 pattern). Comment in `ivive.py` updated to point here. Telltale if it returns: "UGT path / UGT fm redistribution / DrugBank-driven UGT enrichment" without a public-clone-reproducible UGT substrate registry AND without re-running the error-decorrelation gate at the meta-learner level.

Artifacts: `/tmp/4track_state_A_ugt_off.json` and `/tmp/4track_state_B_ugt_on.json` (not committed; comparison summary in `ivive.py:640-655` comment).

### DE-37 — Hepatic intracellular fu correction (B-11)

**Date:** 2026-05-22
**Hypothesis:** Per-drug `fu_correction_liver` from primary literature (Watanabe 2009 DMD 37:1471 / Yamazaki 2010 DMD 38:998 / Riccardi 2017 DMD 45:781 / Patilea-Vrana 2017 Clin Pharmacokinet) would reduce systematic over-prediction for highly bound drugs in the 107-holdout by gating hepatic CLint on fu_inc instead of fup at well-stirred and parallel-tube extraction sites.

**What was measured:** 19 holdout drugs with meta_fold > 3 were mechanism-triaged (T11). 4 identified as PPB-related candidates (paroxetine, oxybutynin, abiraterone, progesterone); the other 15 dispositioned `not_applicable` (P-gp, renal, CES1, extreme first-pass, formulation, prodrug-entity mismatch, autoinhibition, lysosomal trapping, gastric-pH absorption, MAO-A, cytidine deaminase, high-fup, UGT2B7 — see `data/transporters/hepatic_fu_correction.json`). T12 literature search across the 4 primary corpus papers + secondary PubMed queries (hepatic uptake, albumin-facilitated, Kp,uu,liver) returned 0 usable fu_inc/fu_p ratios for any of the 4 PPB candidates — every primary paper is paywalled subscription-only and WebFetch retrieves only abstracts, not the supplemental tables where per-drug ratios live. Final registry: 19 audit rows, all `fu_correction_liver = {mean: 1.0, cv: 0.0}` (identity); 0 literature_applied, 0 class_extrapolated, 4 ceiling_accepted, 15 not_applicable.

**Outcome:** Meta AAFE shift = 0.0% (2.7715238009 → 2.7715238009, bit-identical 107/107 per-drug, T14 verification). Phase A infrastructure (`DrugOnGraph.fu_correction_liver`, loader, ClearanceFluxSpec + ProdrugActivationFluxSpec gating, liver-node applicability flag, identity-blind random-rename invariance test) is shipped to main (commit `a0c90f8`). Curation rows retained as audit trail.

**Why it failed:** The primary literature corpus that publishes hepatocyte uptake fu_inc/fu_p ratios (Watanabe / Yamazaki / Riccardi / Patilea-Vrana et al.) is paywalled, and the per-drug ratios live in supplemental tables that WebFetch cannot retrieve from abstracts. Secondary PubMed sources for the 4 PPB candidates returned mechanism-context papers (autoinhibition PBPK, transdermal PBPK, SULT2A1-class steroid PBPK) without the specific fu_inc measurement. Without literature support, no defensible non-identity multiplier could be curated, so the infra runs on an all-identity registry.

**What this implies:** Either (a) fu_inc/fu_p ratios are not the dominant over-prediction mechanism for the 4 audited PPB candidates, (b) the accessible literature does not measure these ratios for the specific drugs of interest, or (c) both. Future iterations may revisit per spec §6.2 with: subscription access to the four primary DMD/CPK papers, in-house hepatocyte uptake assay data for the PPB candidates, or transporter-mediated alternatives (OATP / NTCP uptake clearance instead of fu_inc gating). Telltale if it returns: "hepatic intracellular fu / fu_inc / Kp,uu,liver / albumin-facilitated uptake" applied to highly-bound holdout drugs without a paired public-corpus or experimentally-measured ratio per drug.

Artifacts: `data/transporters/hepatic_fu_correction.json` (19 audit rows), curation log `docs/superpowers/specs/2026-05-22-B11-Phase-B-curation-log.md`, spec `docs/superpowers/specs/2026-05-21-B11-hepatic-fu-correction-design.md`, plan `docs/superpowers/plans/2026-05-21-B11-hepatic-fu-correction.md`. Phase A infra shipped at `a0c90f8`; Phase B curation at `d10bbef`.

### DE-38 — Morphine / Codeine over-prediction worsens under UGT2B7 activation (secondary finding from B-02)

**Date:** 2026-05-27

**Hypothesis (not the primary B-02 hypothesis):** activating literature-anchored UGT2B7 path for canonical substrates would improve per-drug FE for all 4 UGT2B7 entries (morphine, codeine, ketorolac, indomethacin).

**What was measured (same-numerics-stack regen):**
- morphine: engine FE 1.90 → **2.94** (over-prediction worsened, eng 0.0354 → 0.0549 vs obs 0.0186)
- codeine: engine FE 1.98 → **2.71** (eng 0.276 → 0.377 vs obs 0.139)
- ketorolac: 6.61 → 6.15 (improved, under-prediction reduced)
- indomethacin: 7.87 → 7.79 (slightly improved)
- The 4 UGT1A9 entries (dapagliflozin, etodolac, bexagliflozin, glasdegib) all improved.

**Outcome:** Net Meta AAFE Δ = +0.0067 (within bootstrap noise [2.3151, 3.1690], 1.6% of CI half-width). 6 of 8 seeds improved; 2 of 8 (morphine + codeine) worsened. The Δ direction is determined by the 2 worsening drugs offsetting the 6 improvements.

**Why morphine + codeine worsened:** activating UGT2B7 redirects 70-85% of XGBoost CLint from the default CYP route to the UGT2B7 route. The UGT2B7 effective clearance (abundance × literature-fm × XGBoost CLint) is LOWER than the default CYP allocation, so total hepatic CL drops → Cmax rises. Pre-B-02 morphine engine FE 1.90 was a **coincidental cancellation** — over-extraction via CYP-default + missing UGT path summed to a moderate FE. Activating the correct UGT path REVEALED that the CYP-default routing was over-extracting these drugs.

**What this implies:** B-02 ships as designed (literature-anchored, anti-fudge preserved). The morphine/codeine worsening is a **secondary diagnostic finding** about the engine's CYP-default routing balance for UGT-dominant substrates — orthogonal to B-02's capability + reproducibility mandate. Phase 2.x (backlog B-13) will address UGT2B7 abundance + IVIVE recalibration to reconcile.

**Telltale if it returns under a new label:** "morphine over-prediction" or "UGT abundance recalibration" applied to UGT2B7 substrates without a CYP-route IVIVE rebalancing alongside.

Artifacts: `data/training/4track_holdout_predictions.json` (post-B-02 cache), `data/validation/4track_ci_2026-05-27_B02.json` (bootstrap CI), spec `docs/superpowers/specs/2026-05-26-B02-ugt-public-registry-design.md`, plan `docs/superpowers/plans/2026-05-26-B02-ugt-public-registry.md`.

---

### DE-39 — Gut UGT abundance does not fix UGT2B7-substrate over-prediction (B-13, follow-up to DE-38)

**Date:** 2026-05-29

**Hypothesis (B-13 original framing):** adding literature-anchored gut-wall UGT2B7 + UGT1A9 abundance would supply the first-pass clearance B-02 left missing (DE-38), pulling morphine/codeine over-prediction back down.

**What was measured (corrected B-13, same-numerics-stack regen):** gut UGT2B7 added at the defensible literature value **3.6e3 pmol** (0.60 pmol/mg total-mucosal × 6000; Al-Majdoub 2021 CPT 109:1136 / PMC8048492, corroborated Couto 2020 DMD 48:245). Only the 4 UGT2B7 gut-paired seeds shift, all DOWN but trivially: morphine −0.112%, codeine −0.034%, ketorolac −0.033%, indomethacin −0.004%. Meta 2.69828 → 2.69825 (Δ −2.7e-05). Morphine stays ~3.4× over-predicted.

**Why it cannot work:** the defensible gut UGT2B7 abundance (3.6e3) is **~0.15% of hepatic UGT2B7** (2.43e6). Gut first-pass via UGT2B7 is a sub-percent clearance term — it cannot close a 3.4× over-prediction. The morphine/codeine fix, if any, must come from a **hepatic** UGT2B7 IVIVE/extraction differential, not the gut node.

**Citation-confabulation sub-finding (process, important):** the B-13 spec's gut abundances rested on confabulated literature. The claimed intestinal UGT2B7 "15 pmol/mg (5-30 range)" is ~25× the real median (0.60); "Bhatt 2019 DMD 47:498" resolves to an unrelated Kimoto maraviroc DDI paper (PMID 30862625); "Akabane 2012 DMD 40:1310" does not exist (NCBI esearch count=0). Gut UGT1A9 was DROPPED — not expressed in human small intestine (Oda 2012 isoform-specific antibody finds it in kidney+liver only; UGT1A10 is the intestine-specific 1A isoform; absent from Couto 2020 >5000-protein global proteomics). Caught by an 11-agent adversarial verification workflow (`verify-gut-ugt-citations`, 2026-05-29; both committed values refuted 3/3 + 3/3, high confidence). **Lesson:** spec-stage abundance/citation values must be verified against primary sources before reaching a committed YAML — "fallback citation" lists authored from memory are a confabulation risk.

**Outcome:** B-13 ships as an enzyme-level gut-wall **correctness** term (proper literature-grounded gut UGT2B7; UGT1A9 correctly absent), NOT a morphine fix. Metric-neutral within bootstrap noise.

**Telltale if it returns under a new label:** "gut UGT abundance," "extra-hepatic UGT first-pass," or "intestinal UGT2B7" proposed as a fix for morphine/codeine/UGT2B7-substrate over-prediction. The hepatic UGT2B7 IVIVE differential remains the only plausible lever and is a separate (un-started) backlog item.

Artifacts: `data/physiology/reference_man.yaml` (gut_wall UGT2B7), `data/training/4track_holdout_predictions.json` (corrected B-13 cache), spec `docs/superpowers/specs/2026-05-27-B13-gut-ugt-expansion-design.md` (+ 2026-05-29 amendment), `tests/regression/test_gut_ugt_abundance.py`.

---

### DE-40 — Hepatic UGT IVIVE differential has no verified per-substrate value (B-14; closes DE-39's named lever)

**Date:** 2026-05-30

**Hypothesis (the lever DE-39 named):** correct the hepatic UGT in-vitro→in-vivo under-prediction with a per-substrate scaling factor (SF) on the UGT-routed affinity, pulling morphine/codeine over-prediction down. Designed as a *bounded blind decisive experiment* (spec `2026-05-30-hepatic-ugt-ivive-differential-design.md` v2) after a 3-critic adversarial review demolished a v1 "fix morphine" framing as cherry-picking.

**What was built (ships, audited no-op):** a predict-side, per-enzyme SF hook — `data/enzymes/ugt_ivive_sf.json` registry + `get_ugt_ivive_sf()` loader + a one-line multiply in `_decompose_clint` (engine untouched; identity-blind preserved). With an all-1.0 registry it is a **107/107 bit-identical no-op** (Gate D1). Infrastructure shipped per the B-11/DE-37 precedent.

**What the decisive blind verification found — no applicable SF exists (all dispositions → 1.0):**
1. **Wrong basis.** The famous morphine under-prediction (up to ~16×, Gill/Galetin 2012 PMC3310423) is **HLM/microsomal**; the engine's ML CLint is **hepatocyte**-trained (TDC Hepatocyte_AZ). Intact hepatocytes already recover most of that albumin/binding correction, so the HLM fold is the wrong multiplier — applying it would double-count and over-clear (flip morphine to under-predicted).
2. **Renal, not hepatic.** Morphine glucuronidation is substantially **renal** (renal CLint,UGT ≈ ⅓ of hepatic per-gram for UGT2B7, Gill 2012; Knights 2016 PMID 26808419). Renal clearance does **not** cause hepatic first-pass; it is explicitly out of B-14 scope.
3. **No per-substrate hepatocyte number.** The only hepatocyte-basis figure is a **13-drug class geomean ~2.7×** (suspended-hepatocyte AFE 0.37, AAPS J 2020, DOI 10.1208/s12248-020-00482-9) — not disaggregable to morphine within budget, and individual drugs vary (dapagliflozin's own AFE ≈ 1, i.e. its PBPK is well-predicted, so the class number must NOT be applied uniformly). Per the spec's single-point/anti-confabulation rule, unverifiable-per-substrate ⇒ SF = 1.0.

**Quantitative prior (pre-registered):** even a *full* morphine 3.38→2.0 + codeine 1.78→1.3 fix moves Meta AAFE only ≈ −0.021, so any realistic partial, honest hepatic SF is sub-threshold. NO-GO.

**Outcome:** predicted/realized Meta Δ = 0 (no-op). This is the **fourth** UGT intervention to land neutral (DE-36/38/39/40) and it **closes DE-39's "only remaining lever"**: the hepatic UGT IVIVE differential, evaluated honestly (hepatocyte-basis, hepatic-fraction-only, blind, per-substrate-verified), has no applicable literature value. The clean no-op infra remains for any *future* verified per-substrate hepatocyte SF.

**Telltale if it returns:** "hepatic UGT IVIVE / UGT under-prediction correction / albumin-effect scaling" proposed for morphine/codeine without a **verified, hepatocyte-basis, hepatic-fraction-only, per-substrate** number. The HLM albumin fold and the renal contribution are the two traps. The DE-38-complete idea (UGT IVIVE **plus** a CYP-route rebalance) remains theoretically open but is a different cycle — B-14 shows the UGT-IVIVE half alone has no honest large value.

Artifacts: `data/enzymes/ugt_ivive_sf.json` (all-1.0 audited registry), `src/sisyphus/predict/non_cyp_substrates.py` (`get_ugt_ivive_sf`), `src/sisyphus/predict/ivive.py` (`_decompose_clint` hook), `tests/unit/test_ugt_ivive_sf.py`, `tests/regression/test_ugt_ivive_sf_registry_schema.py`, spec `docs/superpowers/specs/2026-05-30-hepatic-ugt-ivive-differential-design.md` (v2), plan `docs/superpowers/plans/2026-05-30-B14-hepatic-ugt-ivive-differential.md`.

---

### DE-41 — Predict-time AD signal for catastrophic novel-drug Cmax errors (low-F / track-divergence) (2026-06-01)

**Date:** 2026-06-01

**Context:** the 2026-06-01 prospective expansion (N=28, Meta AAFE 3.21 > retrospective 2.698) showed the engine catastrophically under-predicts some 2025 NMEs (mirdametinib 30×, sevabertinib 18×). Root-caused via IV/oral decomposition to **bioavailability (F) under-prediction, not clearance** — engine F = 0.05–0.08 vs implied real F ≈ 1.0, while engine CL_systemic ≈ literature (mirdametinib 4.8 vs 4.6 L/h). See diagnosis.md §8.

**Hypothesis:** the engine's own low predicted-F (or engine↔ML track disagreement) is a predict-time signal of an OOD / unreliable Cmax that the applicability-domain detector could flag, excluding the catastrophic cases from in-domain.

**Result — falsified on the 107-holdout:** corr(engine_F, |log10 fold|) = **−0.037** on the holdout (vs −0.54 on the prospective new-16 — does **not** generalize). Of 21 holdout drugs with engine F<0.10, **17 are within 2-fold** (the engine predicts low F for nearly everything — median 0.18 — and it is co-calibrated). Flagging F<0.08 removes 7 in-domain drugs but only moves in-domain AAFE 2.760→2.732 — it removes *well*-predicted drugs. engine↔ML divergence is also flat (holdout r=−0.033; top-20 vs bottom-20 divergence AAFE 2.48 vs 2.57).

**Why it failed:** the per-drug Cmax error is **not recoverable from the model's own outputs** — consistent with the structural-error ceiling (~30% PI coverage). The F under-prediction is real but near-uniform, so it carries no discriminative OOD signal. The honest lever is measured-F routing or an absorption-model recalibration, not an AD flag.

**Telltale if it returns:** "flag low predicted-F / high track-disagreement as out-of-domain." Re-check the holdout correlation (≈0) before building — it looks predictive on a prospective slice but does not generalize.

---

### DE-42 — Absorption-model recalibration as an F-accuracy lever (DE-41's "honest lever", now tested end-to-end) (2026-06-03)

**Date:** 2026-06-03

**Context:** DE-41 / diagnosis.md §8 left "an absorption-model recalibration" as the one un-tested honest lever for the systematic engine bioavailability-F under-call (median engine-F/lit-F ≈ 0.46, 10/10 measured-fup+CLint PoC drugs). Two measurement-only multi-agent decompositions tested it (engine `F = fa·Fg·Fh`; runtime monkeypatch only, no tracked file changed; headline Meta 2.698 / engine 3.831 reproduced exactly as controls).

**Confirmed diagnostic:** the *median* under-call localises to **fa** (fraction absorbed) — fa median bias 0.55 (vs physiological ~0.9), Fg ≈ 1.0, Fh ≈ 1.05 — because `ka = 2.88·Peff·ka_fraction/radius` (~6%/segment) loses the race to gut transit (~3.85/h), so most dose transits to faeces unabsorbed (dasatinib fa 0.16, sildenafil 0.22). Decisive: the non-CYP3A acids (diclofenac/etodolac/febuxostat) have an empty `metabolized_gut` sink (Fg ≈ 1 real) yet suppressed F ⇒ the loss is fa, not first-pass.

**Why the lever fails:** `ka` enters the ODE **linearly** (`rate = ka·y`), so any uniform multiplier — the `2.88` constant, a villous-amplification factor, a corrected particle radius, or a literature transit-window — is mathematically the *same flat scalar*. It nulls the median (5.25× → engine-F/lit-F 1.0; engine N=107 3.831→3.336) but **cannot reduce per-drug dispersion**: all 4 candidates plateau at geomean fold-error 1.43–1.45 (flat-scalar 1.40, itself inside the ±15% lit-F noise band); the one nonlinear candidate (Peff Caco-2→in-vivo remap) made it *worse* (1.52); engine SITT (195 min) already matches literature (Yu 1996, 199 min). On the full N=107 holdout the best refinement scored engine AAFE **3.405 — worse than the plain scalar (3.336)** — and flipped the engine from 14 to 30 `>3×`-over-predictors (the co-calibration-break signature; un-refit Meta regresses +3%, **meta-regression risk HIGH**).

**The real residual is bidirectional first-pass, not absorption:** once fa→1, the per-drug error splits into two *opposing* modes one knob cannot reconcile — (a) **CYP3A first-pass over-extraction** for bases (alprazolam/carbamazepine/quinine cap at F ≈ 0.5 vs lit 0.8–0.9 even at fa=1; candidate cause: gut-CYP3A abundance scaled-to-midazolam over-extracting non-midazolam substrates) and (b) **well-stirred Fh under-extraction** for high-PPB acids (diclofenac fup=0.003, febuxostat, etodolac overshoot — the DE-37/B-11 hepatic-fu problem). Fixing the bases worsens the acids. Both halves are already data-blocked / co-calibrated.

**Telltale if it returns:** "recalibrate the absorption constant / villous amplification / particle radius / transit time to fix the engine's low bioavailability F." It nulls the median F on a PoC set but is a flat scalar in disguise (ka is linear), worsens the holdout vs the simpler scalar, and breaks meta co-calibration. The only un-foreclosed F lever is **measured-F routing**; the recoverable structural residual is first-pass (gut/hepatic CYP3A IVIVE ⊕ hepatic-fu for high-PPB acids), not absorption.

---

### DE-43 — Engine first-pass recalibration as a *prospective*-set lever; the meta damps engine changes to ~18% on BOTH benchmarks (2026-06-03)

**Date:** 2026-06-03

**Context:** DE-42 foreclosed absorption recalibration for the *retrospective* headline. Open question: the *prospective* N=28 set (Meta AAFE 3.21 — the real novel-drug failure, §8) is **not** part of the meta co-calibration, so a first-pass lever foreclosed retrospectively might still net-improve it. A measurement-only test decomposed the prospective catastrophes and measured two levers on **both** benchmarks via the production meta path (runtime monkeypatch only; before-controls bit-exact: retro meta 2.69825 / engine 3.8314).

**Decomposition (production predicted-ADME, `F = fa·Fg·Fh`):** the catastrophic under-predictors (mirdametinib engine 74×, sevabertinib 53×, pirtobrutinib, pacritinib, tovorafenib … mostly kinase inhibitors) are **fa-first, Fg-second** — fa 0.08–0.32 (absorption starved: low Peff, or low RDKit-solubility → `particle_radius=50µm` → `ka ≪` gut transit), then gut-CYP3A Fg 0.37–0.55 (the midazolam-calibrated `gut_wall` CYP3A4 over-extracting). Fh is correct (consistent with §8: CL_systemic correct). The over-predictors (imlunestrant, taletrectinib) are `not_F` (Vdss/distribution, out-of-AD) — a blunt F lever *worsens* them.

**Result — both levers fail at the meta:** absorption scalar (5.25×): prospective meta 3.171→3.102 (−0.069) but retro meta 2.698→**2.780** (+0.082) → **net −0.012** (costs the headline more than it gains). Gut-CYP3A 0.5×: prospective meta 3.171→3.151 (−0.020), retro meta neutral (−0.0006) → net +0.020 but **inside the N=28 bootstrap CI** (statistically zero) and **not literature-anchored** (halving a midazolam-calibrated abundance = tuning to Cmax, Invariant #8).

**Why it failed (the unifying mechanism):** both levers move the **engine track** materially on prospective (absorption 4.11→3.75; gut-CYP3A 4.11→4.00; mirdametinib engine fold 58→13 / 58→51) — but the fixed-weight **meta-learner damps this to ~18–19% pass-through, the SAME on prospective as on retrospective.** The meta is robust to engine errors by construction (down-weights outlier engine predictions), which symmetrically prevents engine *improvements* from propagating. **Prospective is NOT exempt from co-calibration** — the engine is structurally not a headline lever on *any* benchmark. Plus the DE-42 bidirectional tension: relieving the catastrophic unders blows up the `not_F` over-predictors (imlunestrant 17×→62× under the absorption scalar).

**Telltale if it returns:** "the prospective / novel-drug set isn't co-calibrated, so an engine F / first-pass / gut-CYP3A recalibration will fix it." It improves the engine track on both sets but the fixed-weight meta mutes it to ~18%; net is neutral-to-negative and within N=28 noise. The only un-foreclosed F lever is per-drug **measured-F routing**, not an engine recalibration.

---

### DE-44 — CLint-floor "breakthrough" survey: 7 candidates, all foreclosed as headline levers (2026-06-04)

**Date:** 2026-06-04

**Context:** A schema-free adversarial workflow (8 agents, web + repo, error-cancellation-gated) stress-tested 7 *non-trivial* ways to break the CLint floor, seeded after the external deep-research confirmed the floor is real. All 7 returned **kill as a 2.78-headline lever**; the convergence is the result. Honest probability any moves the headline: **~3–5%** (the residual lives entirely in the `invivo-F-prior` surprise branch, which DE-28/42/43 make unlikely).

**Per-candidate verdicts (with the repo fact that kills each):**
- **Per-isoform rCYP CLint** — data-infeasible: ChEMBL CYP targets are *inhibition IC50*, not substrate-CLint; the merged corpus has ~2 recombinant-basis records of 552. rCYP→in-vivo noise is *relocated* into the substrate-dependent ISEF/RAF (CYP3A4 ISEF ~8-fold), not removed. Collides DE-11/16/20/21/36/40 (fm reallocation = Engine-positive / Meta-neutral).
- **kcat×Km×[E] structured decomposition** — both factor arms already dead in-repo: kcat (C–H BDE) = DE-18 (r=+0.033); Km (docking) = DE-13 ("binding affinity ≠ metabolic rate, do not retry"). A product of two zero-signal estimators compounds variance, it does not create signal.
- **Hierarchical multi-assay target-denoising** — data fatal: of 5,338 unique CLint compounds **exactly 1** has cross-source measurements (field max ~50) → Dawid-Skene unidentifiable. Mechanism also wrong: Bowman & Benet 2019 (our own cited source) shows interlab variance is *systematic per-protocol bias*, not random noise — averaging averages bias.
- **Calibrated-CLint-uncertainty → meta *routing*** — the inverse-variance mechanism already ran: DE-26 M7 Local-BMA **+0.082 worse**, M9 +0.014 worse. Only the *PI-recalibration* half survives (touches the interval, not the point estimate → cannot trip co-calibration), and only for *coverage/UQ*, not point-AAFE. CLint-CV ranks |Cmax fold-error| weakly (SRCC≈0.16).
- **Per-subclass measured-input routing (aggregate)** — the ML track is `predict_cmax(smiles, dose)`, **structurally blind to measured ADME**, so the strongest meta track (3.01) is frozen and the shipped measured "≈11% gain" is *engine-only* and never propagates to the meta headline. Capability already shipped; the "targeted-selection" wrapper adds nothing testable.
- **+BSA / albumin-mediated PSu,inf (OATP)** — *keep as correctness only, not a lever*: a real primary-literature mechanistic fix (Li/Benet 2020, ~1.9–2.0 fold, no scaling) but **inert** — touches 1 holdout drug (pravastatin, inviolable) and 0 prospective → ΔMeta≈0.000. Ships under [[correctness-over-benchmark]] (spec `2026-06-04-oatp-ecm-reanchor-design.md`).
- **Structure→F as engine prior/anchor** — collides DE-28 (repo's own structure→F predictor = scaffold-CV R²=−0.09; the external "Q²=0.58" is the optimistic/AD-filtered tail) + DE-43 damping + circularity (engine E and structure→F are both functions of the same SMILES → no orthogonal information = DE-23 in F clothing).

**Why the whole class fails (unifying):** CLint enters the pipeline **only through the engine track**, which the fixed-weight meta damps to ~18% pass-through (DE-43) under r>0.986 track co-calibration (DE-26). So any CLint improvement *in isolation* either re-learns the existing weights (null) or breaks co-calibration (regression) — the pattern of 18+ prior reverts. The floor is two floors (DE-14 target-noise + mechanistic transporter–enzyme-interplay IVIVE), neither reachable from the CLint side.

**Survivors (correctness/UQ, NOT headline):** (1) +BSA/ECM as correct-but-inert physics; (2) conformal/CQR PI-recalibration for the 29.9%@90% coverage problem — run the ~minutes cached-data kill-test first (Spearman ρ between |log Cmax fold-error| and propagated engine CLint-CV; if |ρ|<0.3 even the UQ value is marginal). One cheap falsification worth running to *retire the premise*: the `invivo-F-prior` test (reuse `xgboost_bioavailability.json` + `_apply_f_correction` at shrinkage w∈{0.25,0.5,1.0}, re-score; pre-registered to rise +0.02–0.08, kill if so).

**Telltale if it returns:** "a better structure→CLint / per-isoform / denoised / uncertainty-routed CLint will move the headline." It will not — CLint is engine-only and the meta damps it; the headline is already in the OrBiTo band (2.08–2.74 vs 2.78). Ship CLint-side work **only** as correctness or calibrated-UQ, never sold as an accuracy lever.

---

### DE-45 — Dose-number / dissolution as a SMILES-only orthogonal track (2026-06-08)

**Date:** 2026-06-08

**Hypothesis (diagnosis §6 named candidate):** a biopharmaceutics **dose-number** track (`Do = (Dose/V0)/S`, fa_cap = min(1,1/Do)) would be clearance/distribution-orthogonal (like VDss) and target the documented prospective fa-starvation failure (low-solubility BCS II/IV kinase inhibitors, §8/DE-43), passing the |r|<0.5 decorrelation gate.

**What was measured (N=107 holdout, public-clone state):** existing-track residual matrix confirmed co-calibration (engine↔ml r=0.596, meta↔both 0.84–0.92). Candidate dose-number track: signal-vs-residual corr(log Do, engine residual)=+0.222 full / +0.135 in-domain (n.s.); candidate-Cmax residual **vs ML r=+0.544 (FAIL)** / vs engine +0.136; in-domain vs ML +0.581 (FAIL). Standalone AAFE 15–18 (instantaneous-absorption 1-cpt proxy, irrelevant — fails the gate regardless).

**Why it failed (decisive number):** in this codebase solubility is `log10 S = −logP + 0.5` (Yalkowsky-from-logP), so **corr(log Do, logP) = +0.912** — the dose-number *is* logP wearing a costume, and the engine already ingests logP (→Peff, Kp, solubility, first-pass). No *residual* dissolution information exists for the engine to lack (the DE-44 structure→F circularity trap). Robust to removing the solubility floor (uncapped log Do: engine r=+0.217, n.s.) and to the fa formula (invDo vs Amidon agree ±0.004). Diagnostic against the fa-starvation thesis itself: dissolution-limited drugs (Do>1, N=82) are *less* under-predicted (mean engine residual −0.353) than solubility-OK drugs (Do≤1, N=25; −0.635).

**Telltale if it returns:** "dose-number / BCS-class / dissolution-limited fa as a new track." It is logP in this codebase → circular. The only un-foreclosed variant is a **pKa-aware pH-dependent** solubility (Henderson-Hasselbalch ionization term, not reducible to logP) — un-run, low probability, pre-register |r|<0.5 AND the pH-term (not logP) carrying the signal. A genuinely orthogonal absorption track needs **measured** aqueous solubility / crystal form / particle size (a measured-input lever), not a SMILES proxy.

Artifacts: scratch `/tmp/dose_number_gate.py`, `/tmp/dose_number_robust.py`; analysis `docs/claude/layered-analysis-and-leap-2026-06-08.md` (Exp 2). `src/sisyphus/predict/adme.py:272` (`_estimate_solubility`).

---

### DE-46 — Renal active-secretion as a SMILES-only orthogonal track (2026-06-08)

**Date:** 2026-06-08

**Hypothesis (diagnosis §6 named candidate):** a charge/polarity proxy for active tubular secretion (`renal_score = charged·(TPSA/100)·max(0,2−logP)`) would seed a transporter-axis-orthogonal track.

**What was measured (N=107):** only **23/107** holdout drugs score >0 (secretion-relevant); signal-vs-residual corr ≈ 0 with every track (engine −0.084 p=0.39, ml +0.117 p=0.23, **meta +0.011 p=0.91**).

**Why it failed:** the retrospective holdout has too few secretion-dominated drugs to carry orthogonal signal, and renal clearance is a terminal-elimination (not Cmax) driver for most of them; the renal subclass is already among the best-predicted (AAFE 2.06, 0 drugs in the >3-fold tail). Quantitation would anyway need OATP1B1-style Jmax/Km registry data (a measured/curated input), gated behind the latent transporter-MM unit bug (D-1, `flux.py:726-736`).

**Telltale if it returns:** "renal secretion / OAT-OCT-MATE analytical track to move the headline." No signal on this holdout; it is a correctness/breadth item (needs the D-1 unit fix + a transporter kinetics registry), not a decorrelated headline track.

Artifacts: `docs/claude/layered-analysis-and-leap-2026-06-08.md` (Exp 3).

---

### DE-47 — Measured-ADME-aware ML regressor track (2026-06-08)

**Date:** 2026-06-08

**Hypothesis (the leading 2026-06-08 leap candidate):** the meta's strongest track (ML, ~3.01) is `predict_cmax(smiles, dose)` — blind to measured ADME (DE-44), so measured fup/CLint reach only the meta-damped engine track. A NEW XGBoost track trained on **measured** {fup, clint, dose, descriptors}→clinical Cmax (corpus `mmpk_pbpk_features.csv`, 1,128 drugs, fup/clint verified measured) would inject orthogonal measured information (like VDss), pass the |r|<0.5 gate, and let the meta exploit measured inputs.

**What was measured (high power, N=93 holdout overlap, clean holdout-excluded train of 1035):** provenance clean (fup/clint literature-measured, not model outputs; cmax_obs is the clinical value — not circular). Track CV AAFE **3.79** (worse than SMILES-ML 3.01 and far worse than engine-measured 2.33). **Decorrelation gate FAIL:** residual r = **+0.685 / +0.784 / +0.787** vs engine/ml/meta (p<1e-13; partial-r controlling for log-obs +0.59/+0.70/+0.70). Even the oracle per-drug min(track, engine) blend = 2.40 > engine-measured 2.33.

**Why it failed (the unifying mechanism — W2):** the dominant error mode is **bioavailability-F blindness, shared across all tracks regardless of input**. A flat regressor has no F (absorption/first-pass/volume) mechanism either, so it re-makes the *same directional errors on the same drugs* (the DE-41 low-F cluster) → correlated residuals. Measured fup/clint were **nearly inert** features (importance 0.045/0.053 vs dose 0.49): measured ADME is only usable *mechanistically* (through the engine), not by a regressor. **The error, not the input, is what must decorrelate** — and F-error is everywhere. This is the fourth gate failure of 2026-06-08 and the empirical heart of the F wall.

**Scope caveat:** concerns the MEASURED-INPUT regime only; zero bearing on the SMILES-only 2.784 holdout (measured ADME absent there). The genuine measured-regime lever is **measured-regime routing** (trust the measured-*engine* more when measured ADME is present — an input-availability regime switch, the extension of shipped measured-F routing), NOT a measured-ADME regressor track.

**Telltale if it returns:** "feed measured fup/CLint into a new ML/analytical track so the meta can use them." The regressor shares the F-blindness and fails the gate; route measured inputs to the measured-aware *engine* instead.

Artifacts: corpus `data/training/mmpk_pbpk_features.csv`, exclusions `mmpk_sisyphus_holdout_exclusions.json`, per-track residuals `data/training/4track_holdout_predictions.json`; analysis `docs/claude/layered-analysis-and-leap-2026-06-08.md` (Exp 4).

---

### DE-48 — Measured-regime engine up-weighting / routing (2026-06-08)

**Date:** 2026-06-08

**Hypothesis (the leading 2026-06-08 leap candidate, proposed in `layered-analysis-and-leap-2026-06-08.md` Leap B):** in the measured-input regime, measured fup/CLint reach mainly the engine track (ML strictly blind, VDss blind, CLF partially Tmax-coupled), so the fixed-weight meta "drags" the measured-aware engine back toward SMILES-blind tracks. A measured-regime weight switch (up-weight / bypass to the measured-engine when measured ADME is present — an input-availability switch, NOT class-aware DE-25/27) was expected to recover toward a "~2.33 engine-measured floor."

**What was measured (high power, N=93 holdout drugs with measured fup+clint; gate workflow + adversarial verification, runtime-monkeypatch only, no tracked-file changes):**
- SMILES-only production meta AAFE **2.78**; measured-ADME production meta AAFE **~2.9** (a small *regression* from supplying measured ADME); engine-measured AAFE **~3.84** (reproduced 3 independent ways + by a skeptic — NOT the prior-phase's stale 3.51).
- **The "2.33 floor" was a hand-curated clean-10 PoC artifact** (reproduced exactly on those 10 drugs); on the representative N=93 the engine-measured is 3.84.
- Every routing variant *loses*: V1 engine=1.0 → 3.84; best reweight (engine 0.7/ml 0.3) → ~3.12; hindsight-optimal w=0.3 → 2.89 — **all > the SMILES meta 2.78**. AAFE rises monotonically with engine weight. Invariance intact (bit-identical when measured ADME absent; 2.784 headline unmoved).

**Why it failed:** engine-measured (3.84) is **worse** than the meta (2.78) on a representative set, because measured high-CLint blows up first-pass actives in the engine (selegiline 126×, acamprosate 100×, methylphenidate 64×, progesterone 49×, abiraterone 40× over) and the ML/VDss tracks were correctly **damping exactly those**. **This is DE-43 in the measured regime: the meta's engine-damping is a feature, not a bug** — up-weighting the engine degrades. Holds on the obs-agree subset (N=52: engine 3.92 vs meta 3.05).

**What this retracts:** the `layered-analysis-and-leap-2026-06-08.md` Leap B recommendation and its "2.33 engine-measured floor / ≤2.0 measured operating point" claim. The measured-input regime is **not** a better operating point on a representative set. *(Caveat: measured on a dirty tree with uncommitted engine changes; exact 3.84 is non-canonical and should be CI-regenerated, but the 3.84-vs-2.78 inversion margin is too large to flip.)*

**Telltale if it returns:** "in the measured regime, trust/up-weight the measured engine more." It degrades — the engine-measured number cited (2.33) is a clean-subset cherry-pick; on a representative set it is ~3.84 and the meta already beats it. The only surviving measured lever is a **new F-orthogonal data modality** (measured solubility/permeability/transporter kinetics / measured F), not a reweight.

Artifacts: gate workflow `next-step-measured-regime-gate`; analysis `docs/claude/layered-analysis-and-leap-2026-06-08.md` (Leap B, retracted); `scripts/run_measured_adme_benchmark.py`, `data/training/mmpk_pbpk_features.csv`, `src/sisyphus/ml/ensemble.py`.

---

### DE-49 — Genotype-nonlinearity "two-arm" validation: clinical data HALT + systemic-sign non-invariance (2026-06-17)

**Date:** 2026-06-17

**Hypothesis (spec `2026-06-16-pgx-genotype-nonlinearity-two-arm-design.md`):** the v2.2a saturable MM engine, with literature `Km` on the gene fraction, reproduces genotype-stratified *nonlinear dose-dependence* a linear model cannot — in **two arms with opposite, topology-determined signs**: systemic clearance (phenytoin/CYP2C9) → PM's low Vmax saturates first → fold **diverges** (`Δβ>0`); first-pass extraction (propafenone/CYP2D6, axial liver) → EM's high extraction saturates → fold **converges** (`Δβ<0`). Harness-isolated; headline 2.731 untouched throughout.

**Two independent failures:**
1. **Data-availability HALT (§4.1).** No citable ≥2-dose **genotype-stratified** exposure grid exists for either clean-primary. Phenytoin: only dose-*requirement* at constant Css (van der Weide 2001) or paywalled per-genotype Vmax (Mamiya 1998); the `*3/*3` activity fraction was **not found** and **not invented**. Propafenone: clean EM/PM cross only at a **single** 300 mg dose (Tran 2022) + within-EM dose-nonlinearity (Siddoway 1987). The full genotype cross-term (P2) is **not validatable on citable human data**.
2. **Systemic-sign non-invariance (refutes spec §5.2).** A synthetic Km×clearance sweep showed the **systemic `Δβ` sign flips**: divergence at mild saturation (high Km), **convergence at deep saturation (low Km — the phenytoin-relevant regime)**. "Systemic always diverges" is false; it holds only where PM nears its own Vmax ceiling while EM stays mildly saturated. The **first-pass convergence (`Δβ<0`) IS robust** (PM≈0 activity pins `β_PM≡1` → structural contrast).

**What survives (shipped, honest):** the **first-pass arm** — the axial engine robustly produces EM-nonlinear/PM-linear genotype-modulated first-pass saturation (`Δβ<0` across the param box), the axial liver resolves a higher hepatic-inlet `C_u` than `well_stirred` (the mechanism), and the engine reproduces propafenone EM within-dose **supra-proportionality** (`β_EM>1`) matching Siddoway 1987's direction (linear-null `β=1`). Also corrected: the `1/(1−fm)` oracle is **well_stirred-only** (parallel_tube ≈3.98 vs 5.0 — topology, not a bug; strict `xfail`).

**Telltale if it returns:** "the engine produces a clean opposite-sign genotype-nonlinearity invariant (systemic diverges / first-pass converges)." Only the first-pass convergence is robust; the systemic sign is **regime-dependent** and in the phenytoin-relevant regime gives the *wrong* sign vs phenytoin's clinical divergence — and there is no citable crossed-grid data to validate the systemic arm anyway. Reusable: the data-independent first-pass/axial demonstration + Siddoway EM P1; NOT a systemic genotype-fold claim. Headline 2.731 untouched (harness-isolated).

Artifacts: `scripts/validate_pgx_cmax_v2b.py`, `tests/integration/test_pgx_arm_sign_mechanism.py`, `data/validation/pgx_genotype_nonlinearity_folds.json` (HALT recorded), `src/sisyphus/validation/pgx_metrics.py`.

---

### DE-51 — Divided-dose (bolus vs split) as a clean signal of GSH-pool memory: escape-saturation compresses it (2026-06-18)

**Date:** 2026-06-18

**Hypothesis (B1.x, the GSH-pool depletion sub-project):** with a dynamic, depleting per-zone GSH pool (`gsh_pool_hazard`), the same total dose given as one bolus vs two spaced half-doses should show the pool's *cumulative depletion memory* as **excess path-dependence** — the dynamic bolus/divided hazard ratio exceeding the static (envelope-only) ratio, because the pool partially regenerates between divided doses. Harness-isolated; headline 2.731 untouched.

**Refuted on the engine (a-priori-pinned params, not tuned).** The measured excess is **negative**: dynamic ratio 4.35 < static ratio 5.45, excess **−1.09** (config: APAP zonation, k_syn=0.231/h, τ=4 h). The pool's **escape-saturation** is the cause — once a zone's pool is overwhelmed the escape factor `1 − GSH/(Kg+GSH) → 1` and *caps* the bolus hazard, so the dynamic model **compresses** the bolus-vs-divided separation **below** the static threshold model (which has no such ceiling and keeps scaling the tall-peak excess). So the static pointwise model actually *overstates* bolus-vs-divided risk separation relative to the saturating pool.

**Why it fails as a demonstration:** the physical divided-dose envelope effect is large and present in BOTH models (the static control is path-sensitive, not flat — the original confound the 2026-06-18 ultrathink caught), and the pool's saturation ceiling pushes the *dynamic* ratio the wrong way. **Do not** use bolus-vs-divided as the pool-memory signal. The clean, confound-free signature of pool/history memory is the **G-order test** (a pure concentration reordering: static hazard fp-invariant by the measure-preserving-reordering identity, dynamic hazard moves — rel diff 0.0 vs 0.029), which B1.x ships as the centerpiece. The rest of B1.x is POSITIVE (G-cliff sharper, G-NAC monotone, G2 orthogonality) — only this one arm is the dead end. See experiment-log 2026-06-18 (B1.x).

### DE-50 — Liver-zonation as a bulk-first-pass / Cmax lever: plug-flow invariance (2026-06-17)

**Date:** 2026-06-17

**Hypothesis (Bridge A of the virtual-cell fusion research):** parameterize the axial liver (`liver__ax1..N`, PR #79) with cell/spatial-atlas **zonation** — pericentral CYP3A4/2E1/1A2 gradients instead of the uniform `1/N` split — to refine first-pass / Cmax. Harness-isolated; headline 2.731 untouched.

**Refuted analytically, then confirmed on the engine.** In the continuous (plug-flow) limit the axial tube obeys `Km·ln(c_out/c_in) + (c_out − c_in) = −V_total/Q` — extraction depends **only on total enzyme**, not its spatial distribution (MM, and as Km→∞ linear). The probe (`scripts/probe_liver_zonation.py`) demonstrates `ΔE(N) = E_zonated − E_uniform → 0` as `N→∞` (clean ~1/N decay; linear |ΔE| 1.9e-4→8.4e-6, saturable 8.8e-4→5.5e-5 over N=5→80, total preserved). The finite-N effect is a **CSTR-discretization artifact** of `_DEFAULT_AXIAL_N=10`, not physiology.

**Why it fails:** spatially redistributing a fixed total enzyme along a plug-flow sinusoid does not change how much of the throughput is cleared — the catalytic capacity it meets is the same regardless of arrangement. Zonation's real consequence is **local/zonal** (per-zone metabolite/reactive-exposure → zonal toxicity, the acetaminophen/DILI pattern), NOT bulk extraction. (Side finding: the finite-N artifact is direction-symmetric for linear clearance but direction-asymmetric for saturable — **periportal > pericentral** extraction — both vanishing with N.)

**Telltale if it returns:** "zonate the liver enzymes (from scRNA-seq/spatial atlases) to improve PBPK first-pass/Cmax." It is invariant in the continuum; any axial-model effect is a discretization artifact that vanishes with N. Zonation belongs to **Bridge B (PD/zonal toxicity)**, consuming per-sub-tank concentrations — not bulk-PK parameterization. (Bonus: the probe doubles as a convergence check on the PR-#79 axial cascade.)

Artifacts: `scripts/probe_liver_zonation.py`, `tests/integration/test_liver_zonation_invariance.py`, `tests/unit/test_zonation_weights.py`, `src/sisyphus/validation/pgx_metrics.py` (`zonation_weights`, `plugflow_E_linear`), `data/validation/liver_zonation_invariance_2026-06-17.{json,md}`.

---

## 3. When to consult this list

- Before writing a design spec for any accuracy improvement.
- Before proposing "let's try SMILES → X for X in {CLint, fup, VDss, CL/F, t½, F%, ...}" — check the relevant category first.
- When a teammate / agent suggests an idea that "sounds new" — grep this file for the keyword before investing time.

## 4. How to add a new entry

When a new experiment concludes negative, append as the next `DE-NN` with: ID, date, one-sentence description, numeric outcome (ΔR² or ΔAAFE or ratio), why it failed (1 sentence), telltale sign if it returns under a new label. Keep entries under 5 lines.
