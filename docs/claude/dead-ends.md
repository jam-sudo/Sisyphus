---
last_updated: 2026-05-02
parent: ../../CLAUDE.md
charter: Authoritative list of failed Sisyphus experiments. Read before proposing any accuracy improvement.
---

# Dead Ends — Do Not Retry

Every experiment here was run, reverted, and documented. **Before proposing any accuracy improvement, open this file and search for the approach.** New track proposals must first pass the error-decorrelation gate described in [diagnosis.md §4](./diagnosis.md).

**Canonical count:** 34 enumerated experiments below. Narrative references in commit messages or prose (e.g. "#35 error cancellation", "14번째 시도", "누적 33 methods") use **informal** numbering that counts early exploration attempts separately; those narrative numbers are **not authoritative** and do not match the table count below. When in doubt, cite the table entry (`DE-NN`).

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

### DE-04 — UGT metabolism enabled in engine
Engine AAFE 2.861 → 3.090. Revert.

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

### DE-34 — 3D conformer descriptors for ML Cmax (2026-04-01)
20 RDKit 3D descriptors (asphericity, NPR1/2, PMI1/2/3, WHIM, etc.) on ETKDG-generated conformers (99.8% success on 1029 training molecules). Morgan+3D ML AAFE 2.930 vs Morgan-only 3.030 (Δ=-0.100, first gate PASS); but meta blend 2.818 vs production meta 2.277 (then) / 2.695 (4-track now) — orthogonality (r=0.655 with Morgan-only ML) was real but insufficient to clear the ensemble's already-tight error cancellation. Branch `feature/3d-cyp-multidist` Experiment A (commit `5c2737b`); see archive tag `archive/3d-cyp-multidist-2026-04-01`. Telltale if it returns: "asphericity / NPR / PMI / WHIM / 3D shape descriptors / conformer features" added to ML feature set.

---

## 3. When to consult this list

- Before writing a design spec for any accuracy improvement.
- Before proposing "let's try SMILES → X for X in {CLint, fup, VDss, CL/F, t½, F%, ...}" — check the relevant category first.
- When a teammate / agent suggests an idea that "sounds new" — grep this file for the keyword before investing time.

## 4. How to add a new entry

When a new experiment concludes negative, append as the next `DE-NN` with: ID, date, one-sentence description, numeric outcome (ΔR² or ΔAAFE or ratio), why it failed (1 sentence), telltale sign if it returns under a new label. Keep entries under 5 lines.
