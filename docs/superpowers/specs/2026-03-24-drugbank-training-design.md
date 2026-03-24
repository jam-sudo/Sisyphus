# DrugBank Training Integration — Phase A

**Date:** 2026-03-24
**Status:** Draft
**Scope:** fup retraining + logP correction + holdout audit + benchmark 3-way
**Prerequisite:** DrugBank lookup integration (merged to main)

---

## 1. Motivation

DrugBank 5.1 데이터를 predict layer에 활용하는 방법은 두 가지:

1. **Training (primary):** measured values를 XGBoost training set에 추가하여
   **모든 SMILES** (novel 포함)의 prediction 정확도 개선.
2. **Lookup (secondary, already implemented):** matched drugs에 대해 measured 값 override.

이전 구현 (lookup-only)은 DrugBank에 있는 ~1,400 drugs만 혜택.
Training은 **모든 약물**의 일반화 성능을 개선 — 이것이 Sisyphus의 core value.

**Silver path AAFE (lookup OFF, retrained model, N=100)가 primary metric.**

---

## 2. Design Principles

1. **Training이 primary.** Novel SMILES에서의 prediction 개선이 목표.
2. **Holdout inviolable.** Holdout 100 drugs는 어떤 training set에도 포함 안 됨.
   SMILES 기반 제외 (isomeric canonical matching).
3. **Lookup은 secondary.** 기존 predict/drugbank.py 구현 유지. 재학습 후에도
   known drugs의 measured 값이 prediction보다 정확하면 override.
4. **IsomericSMILES=True 명시.** 모든 `Chem.MolToSmiles` 호출에 `isomericSmiles=True`
   를 명시적으로 전달. 광학 이성질체 구분 보장.
5. **Engine 코드 변경 금지.** DrugOnGraph contract 변경 금지.
6. **data/drugbank/ git-ignored.** DrugBank license 준수.

---

## 3. Scope (Phase A)

| 항목 | Phase A (이번) | Phase B (후속) |
|------|---------------|---------------|
| Holdout contamination audit | ✓ | — |
| fup 재학습 (logit transform) | ✓ | — |
| logP residual correction | ✓ | — |
| pKa | 현재 유지 (ChemAxon lookup + SMARTS) | pkasolver 평가 |
| CYP substrate classifier | — | PU learning 연구 필요 |
| Lookup (secondary) | 기존 구현 유지 | — |
| Ablation study | ✓ (5 experiments) | — |

**Phase B 제외 근거:**
- CYP classifier: DrugBank annotation은 positive-only → PU learning 또는 careful
  negative sampling 필요. Class imbalance 심각 (CYP3A4: 869 vs CYP2E1: 75). 별도 연구.
- pKa predictor: 9,324 ChemAxon 값으로 XGBoost 학습은 pkasolver (GNN, 714K) 대비
  열등. 기존 lookup이 pragmatic하게 작동 중.

---

## 4. Step 0: Holdout Contamination Audit

**목적:** 기존 XGBoost models의 baseline AAFE가 이미 낙관적인지 확인.

TDC training data와 holdout 100 drugs의 SMILES overlap을 확인한다.
Holdout drug names → clinical_pk.json SMILES → canonical SMILES (isomeric) 변환 후 매칭.

```
1. TDC PPBR_AZ (~1,614 drugs) vs holdout 100 → SMILES overlap 수
2. TDC Hepatocyte_AZ (~1,213 drugs) vs holdout 100 → SMILES overlap 수
3. TDC VDss_Lombardo (~1,130 drugs) vs holdout 100 → SMILES overlap 수
4. 결과 기록: docs/holdout_contamination_audit.md
5. Overlap > 20%이면 baseline AAFE 재해석 필요
```

**TDC data 위치:** `data/training/` (현재 비어있음 — Omega에서 migration 필요)
또는 TDC Python package (`pip install PyTDC`)에서 직접 로드.

---

## 5. Step 1: Baseline AAFE 기록

DrugBank training 적용 전, 현재 모델의 holdout AAFE를 기록한다.

```
Baseline = v1 models, lookup OFF → holdout AAFE (N=100)
```

Feature flags `enable_*_lookup=False`로 lookup을 끄고 benchmark 실행.
이 값이 모든 이후 비교의 기준점.

---

## 6. Step 2: fup Model 재학습

### 6.1 Target Transform: Logit (NOT log10)

**현재 v1:** `log10(fup)` → XGBoost → `10^prediction` → clip [0.001, 1.0]

**문제:** log10은 fup 0.50과 0.99의 차이를 0.297로 압축 (XGBoost가 구분 불가).
`10^(positive) > 1.0` → clip으로 강제 보정 (비물리적 예측의 증거).

**v2:** `logit(fup)` → XGBoost → `sigmoid(prediction)` → (0,1) guaranteed

```python
def logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logit transform. Bounded proportions → unbounded reals."""
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Inverse logit. Unbounded reals → (0, 1)."""
    return 1 / (1 + np.exp(-x))
```

**해상도 비교:**

| fup | log10 | logit | 비고 |
|-----|-------|-------|------|
| 0.001 | -3.00 | -6.91 | low-fup: 둘 다 적절 |
| 0.05 | -1.30 | -2.94 | |
| 0.50 | -0.30 | 0.00 | |
| 0.95 | -0.02 | 2.94 | **log10: 0.50과 0.28 차이. logit: 2.94 차이** |
| 0.99 | -0.004 | 4.60 | log10: 거의 0. logit: 충분히 구분 |

logit 도함수 `1/(x(1-x))`는 x=0.01과 x=0.99에서 동일 (~101) → 전 범위 균등 해상도.
log10 도함수 `1/(x·ln10)`는 x=0.01에서 43.4, x=0.99에서 0.44 → 99:1 해상도 차이.

### 6.2 Training Data 통합

```
Source 1: TDC PPBR_AZ (~1,614 compounds)
  - TDC 문서: target "Y" = fraction bound. 범위 확인 필수:
    if max(values) > 1.0: 값은 % bound (0-100), fup = (100 - value) / 100
    else: 값은 fraction bound (0-1), fup = 1 - value
  - Implementation에서 히스토그램으로 범위 확인 후 적절한 변환 적용

Source 2: DrugBank parsed fup (~1,439 drugs)
  - 값: fraction unbound (0-1) — parser가 "% bound" → 1 - pct/100로 변환
  - Quality filter: 0.001 ≤ fup ≤ 0.999 (물리적 범위)

Holdout 제외 절차 (CRITICAL):
  1. holdout.json에서 drug names 로드: ["alosetron", "apixaban", ...]
  2. clinical_pk.json에서 name → SMILES 매핑
  3. SMILES → canonical SMILES (isomericSmiles=True)
  4. Canonical SMILES set을 holdout_exclusion으로 저장
  5. TDC/DrugBank training data에서 holdout_exclusion에 있는 SMILES 제거
  6. holdout drug이 clinical_pk.json에 SMILES가 없으면 → name matching fallback
  7. 학습 완료 후 POST-TRAINING 오염 검사:
     holdout canonical SMILES가 training set에 있는지 재확인

Merge rules:
  - TDC-DrugBank overlap: canonical SMILES dedup, DrugBank 우선 (더 curated)

Target: y_train = logit(fup)
Features: 2057-element vector (2048 Morgan FP + 9 normalized descriptors)
Model: XGBoost regression, 5-fold CV for hyperparameter tuning
Output: models/adme/xgboost_fup_v2.json
```

### 6.3 Prediction (adme.py)

```python
def _predict_fup_v2(features: np.ndarray) -> Distribution:
    """v2: logit-space model, sigmoid inverse. Output in (0, 1) guaranteed."""
    model = _load_model("xgboost_fup_v2.json")
    logit_fup = float(model.predict(features)[0])
    fup = float(1.0 / (1.0 + np.exp(-logit_fup)))  # sigmoid
    return Distribution(mean=fup, cv=_FUP_CV_V2)  # CV from cross-validation
```

v1/v2 선택 — **model file 존재 여부가 유일한 gate:**
```python
# adme.py — 기존 _predict_fup를 rename하고 새 _predict_fup를 dispatcher로
_FUP_V2_PATH = _MODEL_DIR / "xgboost_fup_v2.json"

def _predict_fup(features: np.ndarray) -> Distribution:
    """Auto-select fup model: v2 (logit) if available, v1 (log10) fallback."""
    if _FUP_V2_PATH.exists():
        return _predict_fup_v2(features)
    return _predict_fup_v1(features)  # existing log10 model
```

**Config flag 불필요.** v2 model file이 있으면 v2, 없으면 v1.
Ablation에서 v1을 강제하려면 v2 file을 임시 rename. 단순하고 명시적.

**Lookup과의 상호작용:** adme.py에서 fup prediction 후, 기존 DrugBank lookup이
measured fup override를 시도 (enable_fup_lookup=True일 때).
v2 model이 DrugBank data로 학습되었으므로, v2 prediction과 DrugBank measured 값이
유사할 것 → 5x guard가 거의 trigger되지 않음 → lookup이 measured 값으로 정상 override.
이는 의도된 동작: training이 prediction을 개선하고, lookup이 known drugs를 fine-tune.

### 6.4 CV 산출

v2 model의 CV는 5-fold cross-validation에서 산출한다.
구체적 공식은 implementation에서 RMSE(logit space) → approximate CV로 결정.
기대값: v1 CV=0.50보다 낮을 것 (더 많은 data + 더 나은 transform).

---

## 7. Step 3: logP Residual Correction

### 7.1 Approach: Residual Learning

Crippen logP를 baseline으로 두고, systematic bias만 학습.

```
Target: y = experimental_logP - crippen_logP (residual)
Features: [logP_crippen, MW, TPSA, HBD, HBA, rotatable_bonds] (6 features)
           — full 2057 features는 1,463 samples에서 overfitting. Small feature set 사용.
Model: XGBoost regression, 5-fold CV for hyperparameter tuning (fup와 동일)
Holdout 제외: §6.2와 동일한 절차
Output: models/adme/logp_correction.json
```

### 7.2 Prediction (chemistry.py)

```python
def _correct_logp(logp_crippen: float, mw: float, tpsa: float,
                  hbd: int, hba: int, rotatable: int) -> float:
    """Apply residual correction to Crippen logP."""
    model = _load_model("logp_correction.json")
    features = np.array([[logp_crippen, mw, tpsa, hbd, hba, rotatable]])
    correction = float(model.predict(features)[0])
    return logp_crippen + correction
```

**Fallback:** model 파일 없으면 Crippen 그대로 사용. `correction = 0`.

**Cascade:** corrected logP → Kp (R&R), Peff heuristic, solubility heuristic, AD check.
이 cascade는 이미 chemistry.py 내에서 logP 변수 하나를 통해 흐르므로
insertion point만 올바르면 자동 전파.

### 7.3 Training Data

```
Source: DrugBank experimental logP (~1,463 drugs)
Holdout 제외: canonical SMILES matching
Features 계산: 각 drug의 SMILES → RDKit Crippen logP + MW + TPSA + HBD + HBA + rotatable
Target: experimental_logP - crippen_logP
```

---

## 8. Step 4: Benchmark 3-Way Comparison

### 8.1 정의

| 조건 | Model | Lookup | N | 의미 |
|------|-------|--------|---|------|
| **Baseline** | v1 XGBoost | OFF | 100 | 현재 성능 |
| **Silver** | v2 retrained | OFF | 100 | **Training의 순수 효과** |
| **Gold** | v2 retrained | ON | 100 | 최대 성능 (training + lookup) |

동일한 holdout 100 drugs에서 3회 benchmark.

**Primary metric: Silver AAFE.** Baseline 대비 Silver 개선 = DrugBank training의 진짜 효과.
Gold - Silver = lookup 추가 효과 (참고용).

### 8.2 Feature Flags

```python
@dataclass
class DrugBankConfig:
    # Lookup flags (measured value override 여부) — 기존 구현 유지
    enable_fup: bool = True
    enable_logp: bool = True
    enable_pka: bool = True
    enable_enzyme_fm: bool = True
```

**Training flags 불필요.** Retrained model 사용 여부는 model file 존재로 결정:
- `models/adme/xgboost_fup_v2.json` 존재 → v2 사용
- `models/adme/logp_correction.json` 존재 → logP correction 적용
- 파일 없으면 → 기존 v1 동작

Ablation: model file rename으로 제어. Config complexity 최소화.

### 8.3 Ablation Study

```
Exp 1: Baseline (v1 models, lookup OFF)
Exp 2: fup v2 only (retrained fup, v1 logP, lookup OFF)
Exp 3: logP correction only (v1 fup, corrected logP, lookup OFF)
Exp 4: fup v2 + logP correction (both retrained, lookup OFF) = Silver
Exp 5: Silver + lookup ON = Gold
```

각 experiment에서 holdout AAFE 보고. Delta 기록.

**Ablation 제어:**
- Model swap: v2 file rename으로 v1 fallback 강제
- Lookup toggle: DrugBankConfig flags
- **Fixed seed:** 모든 experiment에서 deterministic mode (n_mc_samples=0).
  MC propagation OFF — engine 단일 solve로 MC noise 제거.
  Ablation의 AAFE delta가 순수 model 차이를 반영하도록.

---

## 9. IsomericSMILES Enforcement

모든 `Chem.MolToSmiles` 호출에 `isomericSmiles=True` 명시:

| File | Line | Current | Change |
|------|------|---------|--------|
| `scripts/extract_drugbank.py` | 303 | `Chem.MolToSmiles(mol)` | `Chem.MolToSmiles(mol, isomericSmiles=True)` |
| `predict/chemistry.py` | 318 | `Chem.MolToSmiles(mol)` | `Chem.MolToSmiles(mol, isomericSmiles=True)` |

**근거:** RDKit 2025.09 default는 True이지만, 버전 변경 시 동작 변화 방지 +
의도 명시 + 광학 이성질체 구분 보장 (esomeprazole ≠ omeprazole).

Holdout contamination audit에서도 isomeric matching 사용.

---

## 10. File Changes

### New files
- `scripts/train_fup_v2.py` — fup 재학습 스크립트 (TDC + DrugBank, logit transform)
- `scripts/train_logp_correction.py` — logP correction 학습 스크립트
- `scripts/holdout_audit.py` — TDC vs holdout overlap 확인
- `scripts/run_ablation.py` — ablation study runner
- `docs/holdout_contamination_audit.md` — audit 결과
- `models/adme/xgboost_fup_v2.json` — retrained fup model (git-ignored)
- `models/adme/logp_correction.json` — logP correction model (git-ignored)

### Modified files
- `src/sisyphus/predict/adme.py` — v1/v2 model selection logic
- `src/sisyphus/predict/chemistry.py` — logP correction + isomericSmiles=True
- `src/sisyphus/predict/drugbank.py` — DrugBankConfig에 training flags 추가
- `scripts/extract_drugbank.py` — isomericSmiles=True

### Not modified
- `engine/` — zero changes
- `core.py` — DrugOnGraph contract unchanged
- `predict/ivive.py` — fm fractions unchanged (CYP classifier는 Phase B)

---

## 11. Acceptance Criteria

1. Holdout contamination audit 완료. TDC-holdout SMILES overlap 수 기록.
2. Holdout 100 drugs의 SMILES가 어떤 training set에도 포함되지 않음 확인.
3. Baseline AAFE 기록 (v1 models, lookup OFF).
4. **Silver AAFE ≤ Baseline AAFE** (재학습이 regression 아님 확인).
5. Silver/Gold/Baseline 3-way 비교 보고.
6. Ablation study (5 experiments) 결과 기록.
7. fup v2 model의 CV가 cross-validation에서 산출됨.
8. DrugBank CSV 없이 모든 기존 test 통과.
9. `isomericSmiles=True` 모든 MolToSmiles 호출에 적용.

---

## 12. Known Limitations

### 12.1 TDC 값 범위 미확인
TDC PPBR_AZ가 % bound (0-100)인지 fraction (0-1)인지 implementation에서 확인 필요.
Data loading 시 `max(values) > 1`이면 percentage로 판단.

### 12.2 DrugBank parsed fup 품질
Parser accuracy 80.7%. 일부 오파싱 (erythrocyte % → protein binding).
Training data에서는 noise로 작용 (tree model에 robustness 있음).
5x cross-validation guard는 lookup에서만 적용 (training에서는 불필요).

### 12.3 logP correction model의 외삽 한계
1,463 training samples로 학습. Chemical space 외삽 시 correction이 부정확할 수 있음.
Residual learning이므로 worst case = correction=0 = Crippen 그대로. 안전.

### 12.4 Sobol sensitivity 미재측정
Omega에서 gut CYP3A4 Sobol ST=0.47이었으나, Sisyphus architecture에서 재측정되지 않음.
fm 분배 변경의 Cmax 영향은 Omega 수치 기반 추정. Phase B에서 Sobol 재측정 고려.

---

## 13. Out of Scope (Phase A)

- CYP substrate classifier (PU learning 문제, 별도 연구).
- pKa predictor (pkasolver 평가, 또는 현재 lookup 유지).
- RBP model 재학습 (DrugBank에 직접적 RBP 데이터 없음).
- CLint 재학습 (DrugBank에 정량적 CLint 없음, R²=0.24 미해결).
- Meta-learner 재학습 (training integration 효과 확인 후 결정).
- Engine 코드 변경.
- DrugOnGraph field 변경.
