---
last_updated: 2026-05-22
parent: ../../CLAUDE.md
charter: Authoritative list of failed Sisyphus experiments. Read before proposing any accuracy improvement.
---

# Dead Ends — Do Not Retry

Every experiment here was run, reverted, and documented. **Before proposing any accuracy improvement, open this file and search for the approach.** New track proposals must first pass the error-decorrelation gate described in [diagnosis.md §4](./diagnosis.md).

**Canonical count:** 37 enumerated experiments below. Narrative references in commit messages or prose (e.g. "#35 error cancellation", "14번째 시도", "누적 33 methods") use **informal** numbering that counts early exploration attempts separately; those narrative numbers are **not authoritative** and do not match the table count below. When in doubt, cite the table entry (`DE-NN`).

## 1. Theme summary (11 categories)

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

## 3. When to consult this list

- Before writing a design spec for any accuracy improvement.
- Before proposing "let's try SMILES → X for X in {CLint, fup, VDss, CL/F, t½, F%, ...}" — check the relevant category first.
- When a teammate / agent suggests an idea that "sounds new" — grep this file for the keyword before investing time.

## 4. How to add a new entry

When a new experiment concludes negative, append as the next `DE-NN` with: ID, date, one-sentence description, numeric outcome (ΔR² or ΔAAFE or ratio), why it failed (1 sentence), telltale sign if it returns under a new label. Keep entries under 5 lines.
