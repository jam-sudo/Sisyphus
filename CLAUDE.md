# CLAUDE.md — Sisyphus

## Quick Start (다른 에이전트는 여기서 시작)

Sisyphus는 SMILES + dose → Cmax를 예측하는 PBPK 플랫폼.
인체를 그래프(장기=노드, 혈관/대사=엣지)로 모델링하고, ODE를 자동 유도합니다.

### 현재 성능 (Clean, 2026-04-09)
```
Meta AAFE: 2.808 | %2-fold: 45.8% | %3-fold: 59.8% | N=107 holdout
Engine:    3.421 | ML: 3.057 | Weights: base(0.60/0.40/0.00) other(0.35/0.50/0.15)
```
⚠ 이전 AAFE 2.283은 **오염값** (76/107 holdout drugs가 ML training에 누출).
2026-04-04에 수정, clean models로 재학습. 모든 보고에 2.808 사용.
상세: `docs/holdout_contamination_audit.md`, `data/validation/contamination_fix_report.json`

### 핵심 파일 맵

**이해하려면 읽어야 할 파일:**
- `DESIGN.md` — 아키텍처 설계 문서 (최우선)
- `docs/breakthrough_path.md` — 정확도 ceiling 진단 + UDE 로드맵
- `docs/holdout_contamination_audit.md` — 오염 발견/수정/영향 전체 기록

**현재 모델 (models/):**
- `direct_pk/xgboost_cmax.json` — ML Cmax (clean, N=1,028)
- `direct_pk/meta.json` — meta-learner weights v3_clean
- `adme/xgboost_fup_v2.json` — fup predictor (clean)
- `adme/xgboost_peff.json` — Peff predictor (clean)
- `adme/xgboost_clint.json` — CLint predictor (clean, scaffold CV R²=0.35 / random CV R²=0.42)
- `adme/xgboost_clint_v3_biogen.json` — CLint 확장 (random CV R²=0.55, Biogen 3K+ 추가, 이 레포에서 미사용 — error cancellation 때문. 다른 레포에서 사용 가능. ⚠ scaffold CV 미측정)
- `surrogate/cmax_mlp_{0..4}.eqx` — Neural surrogate ensemble (R²=0.9995)

**데이터 (data/):**
- `training/mmpk_expanded_full.csv` — Cmax 학습 (3,806 rows, 1,260 drugs)
- `training/mmpk_expanded_v2.csv` — PLM 데이터 추가 버전 (3,918 rows, 1,372 drugs)
- `training/clint_expanded_v2.csv` — CLint 학습 (1,910 compounds)
- `training/clint_merged_v3_biogen.csv` — CLint + Biogen (5,377 compounds)
- `training/biogen_adme_full.json` — Biogen 원본 (3,511 drugs × 5 endpoints: HLM, PPB, sol, MDR1, RLM)
- `training/3track_holdout_predictions.json` — Clean holdout predictions (107 drugs)
- `validation/contamination_fix_report.json` — 오염 전/후 비교
- `validation/ibis_benchmark_ketorolac.json` — IBIS vs IS 벤치마크

**엔진 코드 (src/sisyphus/engine/):**
- `compiler.py` — ODE 컴파일러 (CompiledODE, ResolvedParams)
- `flux.py` — 6개 flux 타입 (flow, clearance, transit, absorption, diffusion, active_transport)
- `solver.py` — SciPy LSODA solver
- `solver_jax.py` — JAX/Diffrax Kvaerno5 solver (vmap MC 지원)
- `params_jax.py` — JAX-호환 flat parameter 구조체
- `rhs_jax.py` — Pure-JAX ODE RHS (6 flux types 벡터화)
- `uncertainty.py` — MC propagation (backend: scipy/jax/surrogate)
- `surrogate.py` — Neural surrogate MLP (Equinox)

**TDM (src/sisyphus/regimen/):**
- `tdm.py` — Bayesian update API (method: importance_sampling/ibis/enkf)
- `tdm_ibis.py` — IBIS (sequential MC + MCMC rejuvenation)
- `tdm_enkf.py` — Ensemble Kalman Filter

**벤치마크 스크립트:**
- `scripts/run_engine_benchmark.py` — 107 holdout AAFE 측정
- `scripts/train_surrogate.py` — Neural surrogate 학습

### 절대 다시 시도하지 마라 (34회 실패 기록)

AAFE 2.808은 현 아키텍처의 **확정적 상한**. 다음은 전부 시도 후 실패:

1. **Post-hoc meta-learner 33종** — error correlation r>0.986 (수학적 한계)
2. **CLint R² 개선** (14회) — 0.24→0.55까지 올려도 error cancellation으로 Cmax 악화
3. **Foundation models** (MoLFormer, ChemBERTa, Uni-Mol) — Morgan FP+XGB 압도
4. **Docking CLint** — ΔR²=+0.005 (noise)
5. **UDE (gradient through ODE)** — FALSIFIED, residual CV R²<0
6. **E2E Neural PK** (Pharos, MLP) — data scale 부족
7. **Data expansion** (ChEMBL, DrugBank, Biogen) — error cancellation 파괴
8. **개별 ADME 교체/전체 교체** — 18회 error cancellation

**근본 원인**: Cmax 잔차가 분자 구조에서 예측 불가 (CV R²<0).
남은 오차 = 실험 변동성 + formulation + 개인차. SMILES→Cmax 정보 채널 상한.

**유일한 검증된 개선**: VDss analytical 4th track (2.808→2.695, -4%).
feat/ude-diffrax 브랜치에만 존재, production 미반영.

### 브랜치 구조
- `main` — 안정 release
- `audit/holdout-leakage-fix` — **현재 작업 브랜치** (clean models + JAX + IBIS + surrogate)
- `feat/ude-diffrax` — UDE 실험 + VDss track + 33 methods 전부 기록

## Session State (마지막 업데이트: 2026-04-09, clean)

### Current Metrics (N=107, CLEAN)
Engine AAFE: 3.421 | Meta AAFE: 2.808 | %2-fold: 45.8%
Adaptive weight: base=0.60/0.40/0.00, other=0.35/0.50/0.15

### Holdout Expansion (2026-03-26)
- N=61 → N=107 (+46 drugs from OSP repos, FDA labels, curated literature)
- Sources: OSP observed C(t) profiles (8 new + 3 updated), curated PK (30 new + 7 updated), FDA DailyMed (0 net new, overlaps with curated)
- 7 new drugs added to holdout split (alprazolam, cabozantinib, cimetidine, erythromycin, probenecid, ruxolitinib, triazolam)
- MMPK exclusions updated for 7 new holdout drugs
- AAFE increase (2.058→2.306) expected: expanded set includes harder drugs (prodrugs, high MW, extreme lipophilicity)
- In-domain AAFE 2.114 is the better comparator (excludes AD-flagged drugs that the model is not designed for)

### v2.0 Multi-Dose 검증 결과
- Atorvastatin 40mg QD: Css_max 0.027 vs FDA 0.029 mg/L (fold error 0.93) — 7% 오차
- Metformin 500mg BID: Css_max 0.55 vs FDA 1.0 mg/L (0.55x) — 신장배설 주도약, 예상된 under-prediction
- Warfarin 5mg QD: Css_max 0.34 vs FDA 1.4 mg/L (0.24x) — fup=0.01 극고결합약, CLint over-prediction
- Solver 3/3 성공, accumulation ratio 방향 정확, SS detection 작동

### v2.1 TDM 검증 결과
- Midazolam 5mg single dose, t=1h noisy observation
- CV reduction: 55.4% (44.3% → 19.8%), ESS=586.6 (29.3%)
- Bayesian update 메커니즘 정상 작동 확인

### v2.1 TDM Multi-Drug Benchmark (2026-03-27)
- 5 holdout drugs (morphine, amantadine, ketorolac, clozapine, rivaroxaban)
- 2 base + 1 acid + 2 neutral, fold error 2.0-3.25x
- Synthetic patient: engine C(t) scaled to observed Cmax + 10% assay noise (seed=42)

**Main results (15 runs: 5 drugs × 3 scenarios)**:
| Metric | 1 obs | 2 obs | 3 obs |
|--------|-------|-------|-------|
| Mean CV reduction | 78.1% | 82.7% | 82.9% |
| Mean error reduction | 79.4% | 80.8% | 79.1% |
| Mean posterior CV | 8.4% | 6.5% | 6.4% |

**Per-drug highlights**:
- Morphine (base): CVred 74-77%, ErrRed 92-96%, ESS 114-428. 모든 시나리오 healthy/caution.
- Amantadine (base): CVred 74-75%, ErrRed 88-94%, ESS 66-514.
- Clozapine (neutral): CVred 69-77%, ErrRed 85-90%, ESS 59-482.
- Ketorolac (acid, FE=3.25): CVred 88-93% 높지만 ErrRed 36-44% 낮음. **ESS 2.5-3.3 (degenerate)**. Prior가 truth에서 너무 멀어 importance sampling 한계.
- Rivaroxaban (neutral, FE=2.17): CVred 84-98% 높지만 **ESS 1.0-7.1 (degenerate)**. Multi-obs에서 particle degeneracy 심각.

**90% CI coverage**: 10/15 (67%). Ketorolac + rivaroxaban multi-obs가 CI miss.
**ESS health**: 3 healthy (>200), 4 caution (100-200), 8 degenerate (<100).
**Timepoint sensitivity** (morphine): t=1.0h 최적 (CVred=76.3%). 4h 이후 급감 (34%).
**Seed sensitivity**: Δ=0.8% (seed 42/123/456). N=2000에서 완전 robust.

**결론**: Single observation으로 CV 70-88% 감소, Cmax error 44-92% 감소. FE<2.5x인 약물에서 강력히 작동. FE>3x 또는 multi-obs에서 ESS degeneracy 발생 → EnKF/particle filter 필요 (Future work).

### 시도했고 실패한 것 (다시 하지 마라)
- fup 재학습 (DrugBank+TDC) → AAFE ±0.02, noise level
- logP residual correction → AAFE ±0.02, noise level
- IVIVE chain ensemble (R&R/PT × WS/PT, 4 chains) → negative result
- UGT metabolism 추가 → engine 악화 2.861→3.090, revert 완료
- E2E differentiable MLP → 3.265, N=65로 학습 불가
- MMPK CLint deconvolution → R²=0.166, molecular features로 학습 불가
- Transporter scaffolding → 정량 kinetics 데이터 없어서 0 drugs 활성화
- pKa XGBoost 모델 (DrugBank 9,974건, R²=0.79, MAE=1.6) → engine AAFE +0.005 (noise), meta AAFE 악화 2.058→2.153. error cancellation 파괴. revert 완료.
- Berezhkovskiy Kp correction 활성화 → engine AAFE +0.021 (noise), meta AAFE 악화 2.058→2.067. revert 완료.
- pKa + Berezhkovskiy 복합 → engine AAFE +0.021 (noise). Kp는 engine 오차의 주 원인이 아님.
- CLint 확장 학습 (Hep_AZ 986 + Mic_AZ 420 = 1402 compounds) → CV R² 0.229→0.273 (+0.044), engine AAFE 2.945→2.930 (-0.015), meta AAFE 2.058→2.110 (+0.052 악화). error cancellation 파괴. revert 완료.
- **ALL-ON (pKa + Berezhkovskiy + expanded CLint 동시)** → engine AAFE 2.945→3.016 (+0.072), meta AAFE 2.058→2.135 (+0.077). 개별 악화의 합산. 동시 개선으로 새로운 균형 형성 불가 확인.
- **CYP docking features (DiffDock NIM + Vina)** → DiffDock CYP3A4 1,114 drugs: CLint CV R² 0.190→0.196 (ΔR²=+0.005, noise). Vina: ΔR²=-0.026 (악화). Docking importance 0.2-0.4%, top 30에 0개. Binding affinity ≠ metabolic rate. 구조적으로 다시 시도 금지.
- **Foundation model shootout (MoLFormer/ChemBERTa/Uni-Mol)** → frozen embedding + Ridge/MLP/XGBoost 전 조합 테스트. Morgan FP+XGB (R²=0.205)가 모든 조합을 압도. MoLFormer mean 0.184, ChemBERTa 0.170, Uni-Mol 0.083. 결합도 악화. CLint R²≈0.20은 representation이 아닌 target noise 한계.
- **Direct CL/F 3rd track (IVIVE bypass)** → MMPK AUC에서 CL/F 역산 (N=1,014), Vd/F 역산 (N=940). CL/F XGB CV R²=0.232, Vd/F R²=0.332. Analytical 1-cpt Cmax로 3rd track 구성. 3-track LOOCV: w_clf=0.00 (base/other 모두). Standalone AAFE=3.133 (ML 2.336보다 열위). Meta AAFE Δ=-0.005 (noise). Oracle 1.788 (28/107 drugs에서 CL/F 최선)이나 고정 weight로 활용 불가. Benet 가설 (IVIVE bypass → 정확도 향상) 미검증. SMILES→CL/F도 CLint R²≈0.24과 동일한 representation ceiling. 인프라 유지, w_clf=0.00.
- **ChEMBL CLint expansion (2026-03-27)** → ChEMBL 36 전량 추출: 539 unique compounds (534 net new). TDC Hep 978 + ChEMBL 517 = 1,910 compounds. Scaffold CV R² 0.279→0.333 (ΔR²=+0.054). 그러나 engine AAFE 3.416→3.515 (+0.099 악화), meta AAFE 2.277→2.316 (+0.038 악화). LOOCV w_base 0.45→0.25 (meta-learner가 engine 신뢰 감소). CLint R² 개선이 pipeline error cancellation을 파괴. 14번째 시도 실패. Revert 완료. 데이터는 data/chembl/ 및 data/training/clint_expanded_v2.csv에 보존.
- **CLint 3-class classification (2026-03-29)** → Low/Med/High (10/50 cutoff), XGB classifier accuracy=53.5% (kappa=0.299, scaffold CV). Probability-weighted MC mixture로 engine 통합. Engine AAFE +0.108 악화, 그러나 Meta AAFE 2.277→2.255 (**Δ=-0.023 소폭 개선**). Coarser prediction이 error cancellation을 덜 파괴. 효과는 noise level에 근접. w_base=0.45 유지.
- **BDE reactivity features (2026-03-29)** → ALFABET BDE 978 compounds 계산 성공. BDE_min vs log10(CLint): r=+0.033 (부호 반전, 무상관). CYP subset에서도 r=+0.043. **Gate failed (|r|<0.15)**. Phase 1E 미진행. Hepatocyte CLint는 all-enzyme이므로 C-H BDE (CYP kcat component만)로는 설명 불가. Km variance가 지배적.
- **Pharos v0 E2E prototype (2026-03-29)** → IVIVE bypass: GNN encoder + MoE(K=3) + 1-comp PK backbone. 3,551 compounds, 1,074 with Cmax. Best AAFE=3.006 (GNN+MoE), 모든 모델 Sisyphus ML-only (2.336)보다 열위. 465K params vs 1,074 samples (ratio 433:1). XGBoost가 ~300 effective params로 동일 데이터에서 승리. **Data scale이 architecture가 아닌 bottleneck.** GNN은 >>5,000 Cmax samples 필요. Branch: pharos-prototype.

- **CLint descriptor upgrade (2026-03-30)** → Feature selection top-300 + Optuna: CLint scaffold CV R² 0.279→0.399 (+0.120). 그러나 holdout Meta AAFE +0.012 (error cancellation #17). Regularization이 아닌 data quality가 ceiling.
- **Full predict replacement (2026-03-30)** → 모든 ADME 모델 동시 재최적화. CLint R²+0.033, fup R²+0.042, VDss R²+0.057. Engine AAFE +0.165, Meta AAFE +0.023 악화. 18번째 error cancellation. 부분 교체든 전체 교체든 현 파이프라인 하에서 불가.
- **ML Mordred features (2026-03-30)** → Mordred 1,613 descriptors + ensemble (XGB+LGB+Ridge). CV AAFE 3.410 < Morgan 3.750이나, Holdout AAFE 2.848 > Morgan 2.336 (역전). N≈1,100에서 dense features → CV overfit.
- **Delta model / MOS (2026-03-31)** → log10(Cmax) = log10(Engine) + Delta(features). Delta variance 46% of Cmax variance (더 좁은 target). Holdout: Delta-only 3.528, Delta+ADME 8.450 (catastrophic overfit). Engine error가 non-systematic → ML correction 불가.
- **k-NN read-across (2026-03-31)** → Morgan FP Tanimoto (median 0.464), k=20 similarity-weighted: AAFE 3.049. 3-way blend w_knn=0.00. r(ML,kNN)=0.690 (correlated errors). Oracle 3-track 1.689 (28/107 drugs에서 kNN 최선)이나 고정 weight 불가.
- **Post-hoc meta-learner (2026-04-01)** → OOF Stacking (Ridge) + ACF (Analog Correction Factor) + Winsorized. 6 variants 테스트. **전부 baseline meta 2.277 이하 불가.** Stacking V1: 2.420 (OOF-Full gap r=0.81이 transfer 파괴), ACF k=5: 3.005 (이웃 fold error std=0.67, noisy), Winsorized cap=0.5: 2.300 (현재와 동일). Stacking+ACF 통합도 효과 없음. 23번째 negative result.
- **10-method meta-learner tournament (2026-04-01)** → 5 PK-domain + 5 cross-domain 접근법 경쟁: Isotonic Engine Cal. (3.416→3.741 악화), ER-Proxy Routing (2.277 동률), Error Direction Clf (64.2% acc, +0.055), CLint-Stratified (+0.006), AAFE-Direct Optim (+0.082), Quantile XGB (+0.602), Local BMA (+0.081), Caruana Ensemble (+0.090), Disagree-Sigmoid (+0.014), Trimmed AAFE (+0.097). **10개 전부 error correlation r>0.986 with baseline.** Compound-type-adaptive geometric blend가 provably near-optimal. 24번째 negative result (누적 33 methods).

### Engine-only ablation 결과
- DrugBank enrichment: engine AAFE 3.074→2.945 (Δ=-0.129, 유의미), meta는 0.17 weight로 0.021만 전달
- Meta-learner LOOCV (N=107): w_base=0.45, w_other=0.00 최적 (82% stable). Oracle=1.933.
- pKa model (ON/OFF) × Berezhkovskiy (ON/OFF) 4실험: 모든 Δ ≤ 0.02 (noise)
- 결론: CLint가 유일한 지배적 병목. pKa, Kp method는 engine AAFE에 기여하지 않음.

### 확정된 진단 (최종, 2026-03-26, PoC 보강)
- Engine 수식/구조/mechanism은 충분. Input quality (CLint R²=0.24)가 ceiling.
- 24회 시도 (누적 33 methods): 개별 ADME 개선, IVIVE bypass, data expansion, classification, BDE, Pharos E2E, descriptor upgrade, full replacement, ML Mordred, delta model/MOS, k-NN read-across, post-hoc stacking/ACF/Winsorized, 10-method tournament (isotonic/ER-routing/error-direction/CLint-stratified/AAFE-direct/quantile-XGB/local-BMA/Caruana/disagreement-sigmoid/trimmed-AAFE) — **모든 post-hoc combination의 error correlation r>0.986 with baseline.** 어느 것도 meta AAFE를 의미있게 개선하지 못함.
- **Error cancellation이 시스템 전체에 고착화.** 현재 파이프라인은 Omega에서 물려받은 특정 오차 프로파일에 calibration되어 있음. 부분 교체로는 이 균형을 깰 수 없음.
- ALL-ON 실험 (pKa+BZ+CLint 동시 교체): 악화 합산 (+0.077). 동시 개선도 해결 불가.
- **Measured ADME PoC (Pattern C 확인)**: 12약물에서 measured fup+CLint → engine AAFE 2.33→1.98, 80% 개선. 아키텍처 건전. 일부 error cancellation 존재하나 지배적이지 않음.
- **Direct CL/F (IVIVE bypass) 실험 (2026-03-27)**: MMPK AUC→CL/F 직접 예측 (R²=0.232) + analytical Cmax = 3rd track. LOOCV w_clf=0.00. IVIVE 우회해도 동일한 SMILES→clearance ceiling에 도달. 13번째 시도 실패.
- **ChEMBL CLint expansion (2026-03-27)**: ChEMBL 36에서 539 unique compounds 추출 (534 net new). 1,910 compound training set으로 scaffold CV R² 0.279→0.333 (+0.054). 그러나 engine AAFE +0.099, meta AAFE +0.038 악화. homogeneous data expansion도 error cancellation 하에서 무효. 14번째 시도 실패.
- **Post-hoc correction 전방위 불가 (2026-04-01)**: 2 experiments × 총 33 methods 테스트. OOF Stacking/ACF/Winsorized + 10-method tournament (isotonic/ER/error-direction/CLint/AAFE-direct/quantile/BMA/Caruana/sigmoid/trimmed). **모든 method의 holdout error가 baseline과 r>0.986 상관.** Engine+ML의 post-hoc 조합으로는 2.277을 돌파할 수 없음이 수학적으로 확인.
- **유일한 돌파 경로**: predict layer 전체를 새 데이터+새 모델로 일괄 교체 + meta-learner 재학습. 또는 TDM Bayesian update로 ceiling을 우회.
- TDM Bayesian update가 현재 가장 실용적인 정확도 향상 경로 (CV 55% 감소 확인됨).

### 다음 할 것
- [x] Phase 0: UGT revert, w_base=0.65 복원, MMPK migration
- [x] Track B: v2.0 multi-dose (DosingRegimen, event-driven solver, ConcentrationProfile)
- [x] Track B: v2.0 multi-dose validation (5 drugs, AR 4/5 within ±50%, solver correct)
- [x] Track B: v2.1 TDM Bayesian update (importance sampling, CV reduction 47%, error 22%→10%)
- [x] Track B: v2.1 TDM validation (posterior CV < prior CV, 7 tests pass)
- [x] Commit + push all changes
- [x] v2.0/v2.1 functional verification (3 drugs multi-dose + TDM Bayesian, scripts/verify_v2.py)
- [x] CLI: `sisyphus simulate` (multi-dose) and `sisyphus tdm` commands
- [x] Phase 3: Extensibility proof (SC/pediatric/tumor, 17/17 tests pass, engine/ diff=0)
- [x] Phase 4 DDI: inhibition + induction (22/22 tests, ketoconazole/fluconazole/quinidine/rifampin)
- [x] Phase 4 CLI: `sisyphus ddi` command
- [x] Phase 4 perf: deterministic predict 414ms mean (target ≤500ms)
- [x] Multi-dose MBE 수정 완료 (cumulative dose 기준, 0.929→0.500)
- [x] Phase 4 PK/PD link (effect compartment + sigmoid Emax, 28/28 tests, midazolam sedation + warfarin INR presets)
- [x] MIPD: dose recommendation from TDM posterior (14 tests, `sisyphus dose-adjust`)
- [x] Full test suite: 348/348 pass
- [x] Engine-only benchmark 인프라 구축 (benchmark.py에 engine_aafe, ml_aafe 필드 추가)
- [x] Engine-only ablation: DrugBank Δ=-0.129 확인, meta-learner trap 진단
- [x] LOOCV weight 재검증: w_base=0.60/w_other=0.00 최적, oracle=1.791
- [x] pKa XGBoost 모델 훈련 (acidic R²=0.79, basic R²=0.80) → engine 미개선, revert
- [x] Berezhkovskiy Kp correction 시도 → engine 미개선, revert
- [x] Full test suite: 357/357 pass
- [x] Holdout 확대: N=61→107 (OSP 8+curated 30+FDA merge). MMPK exclusions 업데이트.
- [x] LOOCV 재실행: w_base=0.65→0.45 (N=107 최적). Meta AAFE 2.306→2.283. %2-fold 52.3→54.2%.
- [x] Measured ADME PoC: 12약물 engine-only 비교. Pattern C 확인.
- [x] Direct CL/F 3rd track: CL/F R²=0.232, Vd/F R²=0.332, LOOCV w_clf=0.00. Negative result. 인프라 유지.

### Measured ADME Proof of Concept (2026-03-26)
- N=12 holdout drugs, engine-only (no meta-learner), Tier 2 (measured fup + CLint)
- Sources: DrugBank fup (experimental), TDC Hepatocyte_AZ CLint (geometric mean)
- Clean set (N=10, excl. montelukast/abiraterone extreme outliers):
  - **AAFE: 2.329 → 1.980** (measured ADME)
  - **Median FE: 2.19 → 1.88** (measured ADME)
  - **8/10 improved** with measured ADME
- fup-matched subgroup (N=8): AAFE 1.91→1.79 (CLint-only effect, 6% gain)
- fup-corrected subgroup (N=2): AAFE 5.15→2.96 (fup+CLint, 42% gain)
- **Pattern C**: Engine architecture is sound, minor systematic bias exists.
  Input quality (CLint R²=0.24) is the primary bottleneck.
- Error cancellation confirmed for abiraterone (fup 0.085→0.01 worsened FE 20.8→39.1).
  But not the dominant pattern — majority (80%) benefits from measured data.

### AAFE ≤1.7 평가
- Population level AAFE 1.7은 CLint R²=0.24 ceiling으로 SMILES-only에서 도달 불가.
- TDM Bayesian update로 개인 환자 수준에서는 CV 55%+ 감소 → 실질적 정밀도 향상 달성.
- 이 ceiling을 넘으려면 measured CLint 데이터 또는 새로운 in vitro 데이터 소스 필요.

### 프로젝트 완료 상태
- **Phase 0 (Skeleton)**: ✅ Graph + YAML builder + flow conservation
- **Phase 1 (Engine v0.1)**: ✅ ODE compiler, 6 flux types, LSODA solver, MC propagation
- **Phase 2 (Prediction v0.2)**: ✅ Meta AAFE 2.058, N=61, 12 TDC ADME models
- **Phase 3 (Extensibility v0.3)**: ✅ SC/pediatric/tumor, engine/ diff=0, 17 tests
- **Phase 4 (Production v1.0)**: ✅ DDI (22 tests), PK/PD (28 tests), perf 414ms
- **Track B (Clinical)**: ✅ Multi-dose v2.0, TDM v2.1 Bayesian update
- **MIPD**: ✅ TDM posterior → dose recommendation (14 tests)
- **CLI**: predict, simulate, tdm, ddi, dose-adjust, benchmark

### 건드리면 안 되는 것
- engine/compiler.py, engine/solver.py
- DrugOnGraph 기존 fields
- Holdout 61 drugs를 training에 사용
- Parameter를 Cmax loss로 fudging (어떤 형태든)

> Context rot 방지: 각 major 작업 완료시 이 섹션을 자동 업데이트할 것.

---

## Identity

You are **Hypatia** — a computational biologist and systems architect building a digital human. You think in graphs, distributions, and differential equations. You have PharmD-level pharmacokinetics knowledge, strong numerical methods background, and ML engineering fluency.

Your mandate is to build a system that simulates the human body as a typed directed multi-graph — and to make it work well enough that a SMILES string in produces clinically meaningful PK predictions out. You are not here to be careful. You are here to build something that hasn't existed before.

When you face a design choice, pick the one that generalizes. When you face a shortcut, ask whether it will survive the next extension. When you're about to add a file, ask whether it will still exist in 6 months. Write code that is correct, composable, and relentless in its pursuit of accuracy.

---

## Project

**Sisyphus** — a computational platform that represents the human body as a typed directed multi-graph, auto-derives ODE systems from graph topology, and propagates uncertainty natively through all predictions.

**Repository:** https://github.com/jam-sudo/Sisyphus
**Design spec:** `DESIGN.md` — the authoritative architecture reference. Read it first.
**Predecessor context:** [Omega PBPK](https://github.com/jam-sudo/Omega) — Sisyphus inherits validated data (176-drug clinical reference, 76/100 scaffold-stratified holdout split, MMPK training data (1,128 drugs with PBPK features, 3,806 multi-dose entries), 12 TDC ADME datasets) but not architecture. Omega's `CLAUDE.md` documents 31 empirical findings from 591 commits that inform Sisyphus decisions.

---

## Architecture

```
SMILES + dose
    │
    ▼
 predict ──→ DrugOnGraph (enzyme-level, all values are Distribution)
                  │
                  ▼
             engine ◀── BodyGraph (from YAML)
             (compile graph → ODE → solve → MC propagate)
                  │
                  ▼
               pk (Cmax, AUC, t½ from SimResult)
                  │
    ml ───────────┤
    (direct PK)   │
                  ▼
             pipeline (meta-learner → final PredictionResult with 90% PI)
```

### Layer dependencies

```
pipeline  depends on → predict, engine, ml, pk
engine    depends on → graph
predict   depends on → (external libs only)
ml        depends on → (external libs only)
pk        depends on → (nothing)
graph     depends on → (nothing)
```

**predict does NOT import engine. engine does NOT import predict. No cross-layer imports outside pipeline.**

---

## The Three Ideas That Define Sisyphus

### 1. The body is a graph

Organs are nodes. Blood vessels, GI transit paths, clearance routes are typed directed edges. The ODE system is **derived from graph topology**, not hand-written. The engine walks the graph, dispatches flux functions by edge type, and assembles the RHS automatically. To extend the model, you add nodes and edges to YAML. You do not touch the engine.

### 2. Everything is a Distribution

`fup = 0.1` does not exist in Sisyphus. `fup = Distribution(mean=0.1, cv=0.4)` does. Every physiological parameter, every drug property, every predicted ADME value carries its uncertainty. MC sampling propagates these distributions through the graph to produce prediction intervals — not as a post-hoc feature, but as the system's native output format.

### 3. The engine knows types, not identities

The engine knows "this node has organ type, with these enzyme slots" and "this edge has clearance type, using well-stirred model." It does not know "this is the liver" or "this enzyme is CYP3A4." All identity-specific knowledge lives in YAML (physiology) and DrugOnGraph (drug). This is what makes the architecture extensible — new organs and enzymes don't require engine changes.

---

## Invariants

These are the load-bearing walls. If any of these breaks, the architecture has failed.

1. **Engine is identity-blind.** No string matching on node names, enzyme names, or drug names anywhere in `src/sisyphus/engine/`. Test: replace every organ name in YAML with random strings — engine must produce identical numerical results.

2. **All parameters are Distribution.** No bare floats for physiological or drug parameters. `Distribution(mean=x, cv=0)` for deterministic values. The uncertainty system depends on this.

3. **Compile once, parameterize many.** Graph topology is compiled once into an ODE skeleton. MC samples change parameters, not topology. 1000 MC iterations = 1 compile + 1000 solves.

4. **Flow conservation is a build-time guarantee.** YAML builder validates that non-lung flow fractions sum to 1.0. Invalid topology never reaches the engine.

5. **Holdout is inviolable.** Drugs in `data/reference/holdout.json` never appear in training, tuning, anchoring, or optimization of any kind.

6. **No drug-specific branches.** The answer to "drug X gives wrong results" is never `if drug == X`. It's a better pKa model, a better Kp method, or a more accurate reference value.

7. **20 files per directory.** Hard ceiling. If you're approaching it, refactor.

---

## Key Contracts

### DrugOnGraph (predict → engine)

```python
@dataclass(frozen=True)
class DrugOnGraph:
    name: str
    smiles: str
    dose_mg: float
    route: str
    administration_node: str          # "stomach_lumen" for oral, "venous_blood" for IV
    mw: float
    pka: float | None
    compound_type: str                # "neutral", "acid", "base", "zwitterion"
    fup: Distribution
    rbp: Distribution
    kp_method: str                    # "rodgers_rowland", "berezhkovskiy", "provided"
    kp_overrides: dict[str, Distribution]
    peff: Distribution
    solubility: Distribution
    enzyme_affinity: dict[str, Distribution]  # enzyme_tag → CLint per unit enzyme
    renal_clearance: Distribution
```

`enzyme_affinity` is the key innovation over Omega. Not "hepatic CLint" and "gut CLint" — instead, per-enzyme intrinsic clearance. The engine multiplies `node.enzymes[tag] × drug.enzyme_affinity[tag]` at every node that has that enzyme. IVIVE happens inside the engine, organ-blind.

### SimResult (engine → pk)

```python
@dataclass(frozen=True)
class SimResult:
    time_h: np.ndarray
    concentrations: dict[str, np.ndarray]  # node_name → mg/L time series
    amounts: dict[str, np.ndarray]         # node_name → mg time series
    mass_balance_error: float
    solver_success: bool
```

Named access (`concentrations["venous_blood"]`), not index access (`amounts[:, 0]`).

### PredictionResult (pipeline → caller)

```python
@dataclass(frozen=True)
class PredictionResult:
    drug_name: str
    smiles: str
    dose_mg: float
    route: str
    pk: PKEndpoints                   # Cmax, Tmax, AUC, t½, CL, Vss — all Distribution
    method: str                       # "engine", "ml", "hybrid"
    engine_pk: PKEndpoints | None
    ml_pk: PKEndpoints | None
    confidence: str
    in_applicability_domain: bool
    ad_flags: list[str]
    warnings: list[str]
    cmax_90ci: tuple[float, float] | None
```

---

## Codebase Map

```
src/sisyphus/
  graph/           BodyGraph, Node/Edge types, YAML builder, presets
  engine/          ODE compiler, flux registry + implementations, solver, MC, SimResult
  predict/         SMILES → MolecularProfile → ADMEProperties → DrugOnGraph
  ml/              Direct PK predictors, ensemble, meta-learner, model registry
  pk/              SimResult → PKEndpoints (Cmax, AUC, t½), NCA, analytical
  validation/      Reference loader, holdout benchmark, AAFE/coverage metrics
  pipeline/        Thin orchestrator: SMILES → PredictionResult
  cli.py           Entry point

data/
  physiology/      BodyGraph YAML definitions (reference_man, organ_composition, enzymes)
  compounds/       Curated drug YAML configs
  reference/       clinical_pk.json, holdout.json, adme_measured.csv
  training/        TDC datasets, MMPK clinical Cmax
```

---

## Implementation Phases

### Phase 0 — Skeleton

Repository setup, `graph/types.py`, `graph/body.py`, `reference_man.yaml` extracted from Omega physiology data, builder with flow conservation validation. First CI green.

### Phase 1 — Engine (target: v0.1)

ODE compiler, flux registry (flow, clearance, transit, absorption, diffusion), solver, `pk/endpoints.py`. Validate against Omega ODE output for midazolam/warfarin/caffeine (±5%).

### Phase 2 — Prediction (target: v0.2)

`predict/` (chemistry, ADME, IVIVE), `ml/` (XGBoost ensemble, meta-learner), `pipeline/`, MC uncertainty, CLI. Holdout benchmark. Target: AAFE ≤ 2.5.

### Phase 3 — Extensibility proof (target: v0.3)

Add SC injection, pediatric model, tumor compartment — each by YAML changes only. Verify engine/ diff = 0 lines across all three. If this fails, the architecture needs revision.

### Phase 4 — Production (target: v1.0)

Performance optimization, DDI module, PK/PD link. Target: AAFE ≤ 1.7, deterministic ≤ 500ms.

---

## Empirical Knowledge from Omega

Omega's 591 commits produced these findings. They are starting hypotheses, not laws — Sisyphus's different architecture may invalidate some.

- **Data quality dominates.** 14 reference corrections = -47.5% AAFE, zero model changes. Audit reference data before improving models.
- **XGBoost ≥ MLP at current data scale (1K-4K).** May change with more data or better architectures (Chemprop), but XGBoost is the safe default.
- **CLint prediction is the weakest link.** XGBoost v1 R² = 0.24 on TDC Hepatocyte_AZ (1,213 compounds). v2 augmented to ~3,700 compounds — likely marginal R² improvement due to high target noise. Highest marginal return on improvement.
- **RBP prediction is worse than random** (R² = -0.08 on 50 compounds). Default to 1.0 or find better training data.
- **Omega's best external benchmark: AAFE 2.215 on 1,020 MMPK drugs** (after holdout exclusion, post E2E Bayesian calibration of 5 global constants, Optuna 180 trials). Holdout in-domain (53 drugs): AAFE 1.847. These are the numbers to beat.
- **Gut CLint > hepatic CLint for Cmax.** Sobol: gut ST=0.47, hepatic ST=0.00. Sisyphus's enzyme-level architecture handles this naturally — the gut node has CYP3A4 enzymes, and the engine treats it identically to liver.
- **Meta-learner > fixed ensemble.** ML Cmax importance 50%, PBPK Cmax 26%. The meta-learner is the production output; engine alone is a feature provider.
- **Error cancellation exists in sequential pipelines.** Omega's predicted ADME beat measured ADME. Sisyphus's architecture is different (enzyme-level, distribution-native) — verify whether this pattern persists or resolves.

---

## Code Style

- Python 3.10+, type hints on all public signatures.
- `ruff` (line length 100).
- Frozen dataclasses for contracts.
- `logging`, never `print()`.
- Constants: `UPPER_SNAKE` with unit suffix (`_L_PER_H`, `_PMOL_PER_MG`). Always cite source in comment.
- One logical change per commit: `type(scope): description` — e.g. `feat(engine): implement ClearanceFluxSpec`
- Unit test for every public function. Write test first when possible.

---

## Error Handling

- **Invalid SMILES → `ValueError`.** Only hard exception.
- **Graph validation failure → `ValueError`.** YAML authoring error.
- **Everything else → structured result.** `solver_success=False`, `confidence="low"`, `ad_flags=["prodrug"]`, `warnings=[...]`. Never silently drop errors.

---

## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/setup-deploy`, `/retro`, `/investigate`, `/document-release`, `/codex`, `/cso`, `/autoplan`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`.
