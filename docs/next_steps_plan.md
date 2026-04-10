# Next Steps Plan — Post-SBI-POC

**작성일**: 2026-04-10
**Base branch**: `audit/holdout-leakage-fix` at commit `4438a24`
**Author**: Hypatia (autonomous session 2026-04-10)

## Executive Summary

SBI POC는 **완료되었고 kill-switch gate를 두 드럭(morphine, clozapine)에서 통과**했습니다. 이제 결정할 것은 "이 capability를 어떻게 확장하고 production으로 정착시킬 것인가"입니다.

네 개의 평행 트랙이 식별됩니다.

- **Track A (primary)**: Phase 2.0 — Multi-drug conditional amortizer. SBI POC를 "드럭 하나당 네트워크 하나"에서 "모든 드럭 공통 네트워크"로 확장.
- **Track B**: Phase 2.1 — Production integration. `regimen/tdm.bayesian_update`에 `method="sbi"` dispatch, CLI 노출, multi-observation.
- **Track C**: Phase 2.2 — Advanced capabilities. Hierarchical populations, active learning, inverse molecular design.
- **Track D**: Technical debt — Surrogate OOD 버그, TDM CI calibration, CLI dispatch 통일, in-domain AAFE 재측정, prospective N=15→40 확장.

권장은 **Track A 먼저 단일 세션으로 완결, 그 다음 Track D 하나씩, 그 다음 Track B** 순서입니다. Track C는 A+B가 정착된 이후.

---

## 현 상태 체크포인트

**완료된 것 (commit `4438a24` 기준)**:

- Post-merge 4-track pipeline, Meta AAFE 2.695 (N=107 holdout, 4-track 재벤치 확정)
- JAX/Diffrax differentiable engine (Phase 0+1.1-1.3+2.1), neural surrogate (production R²=0.9995, OOD for real drugs — known bug)
- TDM: IBIS + EnKF production-ready, `regimen/tdm.bayesian_update` dispatch
- Amortized SBI POC: morphine + clozapine 둘 다 SBC 통과, morphine에서 IBIS 대비 2164x 속도 향상
- Full test suite 384/384 pass
- Branch 합본, archive tag 보존, origin 동기화

**알려진 결함**:

1. **Neural surrogate feature extraction 단위 불일치**: training은 drug-level `p["clint"]`, production은 `Σ(abundance×affinity)`. 생산 드럭에 대해 OOD. (Sev: medium — SBI POC가 이를 우회했지만 neural surrogate의 다른 사용 경로가 영향받음.)
2. **SBI 포스테리어 predictive Cmax 편향**: IBIS 대비 +28% (morphine). (Sev: low — POC 통과 가능, Phase 2.0에서 개선.)
3. **TDM 90% CI coverage가 67%**: importance sampling의 prior miscalibration. (Sev: medium — TDM의 신뢰성에 직접 영향.)
4. **CLI `--method` 옵션 불일치**: `cli.py`는 `{is, enkf}`만 노출, `tdm.bayesian_update` API는 `ibis`도 지원. (Sev: low — API 불일치.)
5. **4-track holdout predictions JSON 미저장**: benchmark는 출력만 하고 저장하지 않음. 재현성 gap. (Sev: low.)
6. **In-domain AAFE는 pre-VDss 2.591에 고착**: feat 브랜치에서 측정한 값, 4-track에서 재측정 필요. (Sev: low — 문서 gap.)
7. **Prospective N=15**는 pre-merge. 4-track에서 재벤치 필요. (Sev: low — 논문 쓸 때 필요.)

---

## Track A (권장 1순위): Multi-drug Conditional Amortizer

### 목표

POC는 "drug 1개당 네트워크 1개" 구조였습니다. 실제 production에서는 "모든 드럭에 대해 하나의 네트워크가 즉시 응답"해야 합니다. 이는 conditional density estimation 문제입니다:

```
p(θ | x_obs, drug_features) — x_obs는 1D Cmax, drug_features는 12-64D 분자 특성
```

**성공 기준**:
- 5-10개의 holdout 드럭에 대해 SBC 통과 (KS p > 0.01, coverage within 10pp)
- IBIS 대비 speedup이 여전히 10^2~10^3 단위로 유지
- 한 번의 네트워크 훈련으로 drug-generic inference 가능

### 설계 결정

**Conditioning 입력**: **12D ADME features** (neural surrogate가 쓰는 feature space와 동일).
- Pros: 이미 predict pipeline이 생산, 차원 낮음, 해석 가능.
- Cons: Morgan fingerprint 같은 raw descriptor보다 덜 expressive. 첫 이터레이션엔 충분.

**Amortizer 입력 총 차원**:
- observation `x` = concat([drug_features(12), log10_cmax(1)]) = 13D
- 단, SBC와 포스테리어 의미론을 보존하려면 `drug_features`는 관측의 일부가 아니라 **fixed conditioning**으로 다뤄야 함. `sbi` 라이브러리는 이걸 embedding_net으로 처리.

**훈련 데이터 설계**:
- Training 드럭 세트: 50-100 drugs (NOT in holdout — from `data/training/mmpk_expanded_v2.csv`)
- Per-drug theta 샘플 수: 500-1000
- 총 시뮬레이션: 50 × 1000 = **50,000 사이즈**
- 단일 스레드 예상: 50,000 × 0.2s = 2.8시간
- Multiprocessing 4 workers: ~45분 (I/O + worker startup 포함)

**Parallelization 전략**: `multiprocessing.Pool` + drug-partitioned 작업 분할. 각 worker는 자기 드럭만 시뮬레이션. 공유 state 없음, embarrassingly parallel.

**Density estimator**: NSF (POC에서 MAF 대비 우수 확인). hidden_features=64, num_transforms=8 유지.

**Prior**: POC의 box prior 그대로. 추후 drug-type별로 좁히는 건 Phase 2.0.5.

### 작업 단계 (실제 실행 순서)

1. **Training drug set 선정** (~30분)
   - `data/training/mmpk_expanded_v2.csv`에서 holdout 제외 드럭 전체 목록
   - 필터: SMILES 유효, dose 알려짐, Cmax 학습 데이터 있음
   - 계층화 샘플링: compound_type (neutral/acid/base/zwitter), MW, logP, CLint 범위 고루
   - 50개 선정 → `data/sbi/train_drug_set.json`

2. **Multi-drug simulator wrapper** (~1-2시간)
   - `src/sisyphus/sbi/multi_drug.py`: `MultiDrugSimulator` 클래스
   - 드럭별 `EngineSimulator` 캐시 보관
   - `simulate_batch(drug_ids, thetas)` 인터페이스
   - Drug feature 추출 (12D) 함수
   - Unit tests

3. **Parallel training data generation** (~1시간 실행)
   - `scripts/sbi_generate_multi_drug_data.py`
   - `multiprocessing.Pool` with 4 workers (or auto-detect)
   - Checkpoint 저장 (drug_id별로 .npz, 최종 concat)
   - Progress logging
   - 산출: `data/sbi/multi_drug_train.npz` (N_drugs, N_theta, theta+x+context)

4. **Conditional NPE 훈련** (~15-30분)
   - `scripts/sbi_train_multi_drug.py`
   - `sbi.inference.SNPE`에 embedding_net 제공 — drug_features(12)를 condensed representation(8D)으로 매핑
   - x = log10_cmax만 (drug_features는 embedding_net을 통해)
   - NSF density estimator over (theta | embedded_context, x)
   - 산출: `models/sbi/multi_drug_posterior_nsf.pt`

5. **SBC per-holdout-drug** (~30-60분)
   - `scripts/sbi_run_sbc_multi_drug.py`
   - 5-10개 holdout 드럭 (morphine, clozapine, amantadine, caffeine, etc.) 각각에 대해 SBC
   - 각 드럭에 대한 300 calibration × 500 posterior samples
   - Gate: 모든 드럭이 KS p > 0.01, coverage within 10pp
   - 산출: `data/validation/sbi_sbc_multi_drug.json`

6. **Cross-drug IBIS 비교** (~1-2시간, 실제로 IBIS가 드럭당 26분)
   - 3-5개 holdout 드럭에 대해 amortized posterior vs IBIS
   - Speedup은 한 번 훈련한 네트워크로 여러 드럭 처리 → 누적 speedup 훨씬 큼
   - 산출: `data/validation/sbi_vs_ibis_multi_drug.json`

7. **문서 + 테스트 + 커밋**
   - `docs/sbi_multi_drug_results.md`: 방법, 결과, cross-drug generalization 분석
   - `tests/unit/test_sbi_multi_drug.py`: embedding_net 정합성, 드럭 features 추출, cross-drug sampling
   - CLAUDE.md 업데이트
   - Full test suite 재검증
   - Commit + push

### 예상 소요

- 순수 실행 시간: 4-5시간 (주로 시뮬레이션 + IBIS 비교)
- 코드 작성 시간: 2-3시간
- 총: **한 세션 안에 완결 가능**

### 리스크와 kill-switch

1. **Conditional amortizer가 generalization 실패** → SBC가 multi-drug에서 부분 통과 (일부 드럭만)
   - Diagnostic: 어느 드럭에서 실패하는가? 해당 드럭의 ADME feature가 training 분포에서 얼마나 벗어나는가?
   - Response: training drug set 확대, 또는 embedding_net 심화
2. **훈련 시간이 예상보다 김** → multiprocessing overhead, ODE stiffness
   - Diagnostic: drug별 평균 시뮬레이션 시간 로깅
   - Response: drug 수를 50 → 30으로 축소
3. **Memory blowup** — 50k samples × 13D + embedding 상태
   - Diagnostic: psutil monitoring
   - Response: chunked training, gradient accumulation

**Kill-switch gate**: Week 1에 5000 샘플 × 20 drugs로 mini-run 먼저. Coverage 확인 후 full 50k로 확대. 이게 Phase 2.0 첫 단계의 결정 게이트.

---

## Track B: Phase 2.1 — Production Integration

Phase 2.0 완료를 전제로 함. Track A 후에 할 것.

### 목표

Amortized SBI를 Sisyphus의 기존 TDM API에 매끄럽게 통합해서 사용자가 `method="sbi"` 한 줄로 호출 가능하게 만든다.

### 작업

1. **API dispatch** (~1-2시간)
   - `src/sisyphus/regimen/tdm.py`의 `bayesian_update`에 `method="sbi"` 분기 추가
   - Multi-drug posterior 로드 + drug features 계산 + 추론 + TDMResult 변환
   - Fallback: posterior 파일이 없거나 로드 실패 시 IBIS로 fallback (경고 출력)
   - Unit tests

2. **CLI 노출** (~30분)
   - `cli.py`의 `--method` argparse choices 확장: `{is, ibis, enkf, sbi}`
   - `cli.py`가 `tdm.bayesian_update(method=args.method)` 하나로 모든 method dispatch (현재 EnKF만 직접 호출)
   - 기존 cli.py의 EnKF 직접 호출 제거 (API 일관성)

3. **Multi-observation 지원** (~2-3시간)
   - Phase 2.0의 amortizer는 single observation x=log10_cmax만 가정
   - 실제 TDM은 multi-point (t=1h, t=4h, ...)
   - 선택 1: context를 (drug_features, [log10_c(t1), log10_c(t2), ...])로 확장
   - 선택 2: sequential SBI (한 observation씩 posterior 업데이트)
   - 선택 1이 단순, 선택 2가 더 일반적
   - 첫 구현은 선택 1, fixed 2-observation (t=1h, t=4h)
   - 추후 variable n_obs를 위해 transformer-style encoder 고려

4. **SBC CI automation** (~1시간)
   - `scripts/sbi_ci_sbc.py`: 기존 multi-drug posterior에 대해 빠른 SBC 실행 (50 calibration × 100 posterior)
   - 매 engine 변경 시 자동 실행 (pre-commit 또는 GitHub Actions)
   - Calibration drift 감지

5. **Benchmark 확장** (~2시간)
   - `scripts/benchmark_tdm_methods.py`: 5-drug set에 대해 IS/IBIS/EnKF/SBI 4-way 비교
   - CV 감소, error 감소, ESS, wall time 전부 리포트
   - 산출: `data/validation/tdm_method_tournament.json`

### 예상 소요
7-10시간. 한 세션 or 두 세션.

---

## Track C: Phase 2.2 — Advanced Capabilities

Track A+B 완결 후. 장기 플레이.

### 옵션 C1: Hierarchical amortization

- Population-level conditioning: pediatric, renal impairment, hepatic impairment
- Amortizer가 (drug_features, obs, population_class)로 conditioning
- 성인 + 소아 + 신장장애 3 classes로 시작
- 훈련 데이터: 각 class의 BodyGraph variant 사용 (이미 `data/physiology/`에 pediatric.yaml 있음, Phase 3 extensibility proof에서)

### 옵션 C2: Active learning

- 목표: "어느 wet-lab 측정치가 예측 불확실성을 가장 많이 줄이는가?"를 자동 결정
- Amortizer의 posterior entropy + gradient w.r.t. 측정 가능 feature로 acquisition function
- Use case: limited wet-lab budget → prioritize highest-info-gain measurements

### 옵션 C3: Inverse molecular design

- 목표: "target Cmax profile을 달성하는 SMILES를 design"
- Differentiable engine + molecular graph gradients
- 매우 어려움, 5년 연구 주제
- POC: 기존 드럭의 작은 변형으로 target Cmax 맞추기

권장 순서: C1 → C2 → C3

---

## Track D: Technical Debt (Parallel, Low-coupling)

이것들은 **Track A와 독립적**으로 어느 때든 할 수 있음. 각각 1-2시간 작업.

### D1: Neural surrogate OOD 버그 수정

- **문제**: `params_to_features_single`이 drug-level clint가 아닌 `Σ(abundance×affinity)`를 반환 → training 분포 완전히 벗어남 → 생산 드럭에 대해 예측 신뢰 불가
- **해결 방법 A**: `params_to_features_single`을 재작성해서 training과 동일한 drug-level clint 반환
- **해결 방법 B**: `train_surrogate.py`를 재작성해서 production feature 분포에 맞춰 재훈련
- 방법 A가 적절 (훈련 데이터는 보존 가치). 재작성 후 기존 MC propagation 파이프라인에서 accuracy gate 재검증.
- 산출: `src/sisyphus/engine/surrogate.py` 수정 + 해당 부분 retest + surrogate accuracy 재보고서

### D2: TDM 90% CI calibration fix

- **문제**: TDM 5-drug benchmark에서 90% CI coverage가 67% (spec 90%)
- 원인 후보:
  - Prior가 너무 좁아서 (actual dispersion > prior CV)
  - Importance sampling degenerate on some drugs
  - Assay noise model mismatch
- **실험**:
  - Prior CV 확대 (0.2→0.3)하고 coverage 재측정
  - IBIS 전환 (이미 가능) 후 coverage 재측정
  - Conformal calibration layer 추가 (posterior에 post-hoc scaling)
- 산출: `docs/tdm_calibration_study.md` + patched prior CVs

### D3: CLI method dispatch 통일

- **문제**: `cli.py`는 `{is, enkf}`만 노출, `tdm.bayesian_update` API는 ibis/enkf/is 전부 지원. EnKF 경로는 직접 호출.
- **해결**: CLI를 전부 `tdm.bayesian_update(method=args.method)` 경유로 통일. choices에 `ibis` 추가.
- 1시간 작업. Low-risk.

### D4: 4-track holdout predictions JSON 정식 저장

- **문제**: `run_engine_benchmark.py`가 결과를 stdout만 뱉고 저장 안 함. 3-track JSON은 merge 이후 stale.
- **해결**: `scripts/run_engine_benchmark.py`에 `--save-json PATH` 옵션 추가. 4-track 결과를 `data/training/4track_holdout_predictions.json`에 저장.
- 1시간 작업.

### D5: In-domain AAFE 재측정

- **문제**: 현재 CLAUDE.md의 in-domain AAFE 2.591은 pre-VDss, pre-merge 값 (feat 브랜치에서 측정됨).
- **해결**: 4-track에서 N=82 subset (AD-flagged 제외)에 대해 AAFE 재측정. CLAUDE.md 업데이트.
- 30분 작업.

### D6: Prospective N=15 → N=40 확장

- **문제**: Prospective validation은 2024-2025 FDA NME 기준. N=15 → N=40+으로 확장하면 gap 평가가 더 견고.
- **해결**: `scripts/fda_cmax_extractor.py` 재실행 (2025 포함), `scripts/prospective_batch_validator.py`로 배치 실행, 4-track에서.
- 2-3시간 작업. 데이터 수집 quality에 따라 달라짐.

---

## 권장 세션 플랜

### Next session (immediate)

**Primary**: Track A (Multi-drug conditional amortizer) — 한 세션 완결 목표.

- Week-1-equivalent mini-run (5000 sims × 20 drugs)으로 kill-switch gate 먼저.
- 통과하면 50k full run.
- 실패하면 diagnostic + 축소 스코프로 재도전.

### Session +1

**Parallel**: 
- Track A 완결되었다면 Track D1 (surrogate OOD fix) → D2 (TDM calibration fix).
- 이 둘은 production quality에 직접 영향, 빠르게 끝남.

### Session +2

**Primary**: Track B (Production integration) — Phase 2.0 결과를 기존 API로 endpoint 연결.

### Session +3 이후

**Choice**: Track C (advanced, 장기) or 또 다른 방향.

---

## 우선순위 결정 근거

왜 A → D → B → C 순서인가?

- **A 먼저**: Track B와 C는 A의 결과물(multi-drug amortizer)에 의존한다. A가 완결되지 않으면 B/C는 시작도 불가.
- **D는 언제든지**: Track D는 A/B/C와 독립적. "일감 부족"이거나 "블록 해소" 용도로 언제든 insert 가능.
- **B는 A 이후**: Phase 2.0의 결과물이 있어야 API integration이 의미 있음.
- **C는 마지막**: C1 hierarchical은 A 패턴 확장, C2 active learning은 B의 downstream, C3 inverse design은 장기.

왜 D1 (surrogate OOD)을 Track A 전에 하지 않는가?
- A는 surrogate를 우회해서 full engine을 쓰므로 D1에 blocked되지 않음.
- D1을 먼저 하면 A가 delay됨.
- D1은 A가 완료된 후 "집 청소" 페이즈에서 깨끗이 처리하면 됨.

왜 D2 (TDM calibration fix)을 B의 일부가 아닌 D로 분리하는가?
- D2는 기존 IBIS/EnKF 경로의 버그 수정이고, B는 새 SBI 경로 통합.
- 두 트랙은 독립적. D2는 언제든 할 수 있고 B의 선행조건이 아님.

---

## 남은 질문 (Blockers / 결정 대기)

1. **Training drug set 선정 기준**: 50개 드럭을 어떻게 고를지. 계층화 vs 랜덤 vs "가장 학습에 도움 되는" 기준?
2. **Compute budget**: 50k 시뮬레이션 = ~1시간 (multiprocessing) 또는 3시간 (단일). 사용자 승인?
3. **Holdout validation drugs**: SBC 검증용 5-10개를 어떻게 고를지. 다양한 compound type + 다양한 CLint 범위?
4. **Paper timing**: 논문 작성이 목표라면 Track F1 (prospective N=40) + F3 (negative results paper) + SBI methodology paper 세 편을 묶을 수도. 우선순위?
5. **Surrogate OOD 버그**: "집 청소 페이즈"로 뒤로 미루는 것이 맞나, 아니면 A의 일부로 먼저 처리할까? Phase 2.0 자체가 full engine을 쓰므로 영향 없지만, Phase 2.1이 surrogate를 쓴다면 prereq.

---

## 기록

이 플랜은 POC commit `4438a24` 시점에 작성됨. 계획은 수정 가능하고, 각 Track/단계는 독립적으로 re-scoping 가능. Kill-switch gate와 결정 게이트를 존중할 것.

다음 세션에서 Track A로 진행하려면 "진행" 신호 주시면 Training drug set 선정부터 시작합니다. 다른 Track이나 방향을 원하시면 이 문서를 기준으로 재조정합니다.
