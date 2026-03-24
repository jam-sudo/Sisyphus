# DrugBank Data Integration into Sisyphus v1.0

**Date:** 2026-03-24
**Status:** Draft (post-review, 19-pass self-feedback incorporated)
**Scope:** predict layer enhancement — zero engine changes

---

## 1. Motivation

Sisyphus v1.0의 ADME prediction pipeline은 전부 XGBoost/heuristic 기반이다.
핵심 약점:

- **fup**: XGBoost CV=0.50. 실측값이 있으면 CV=0.20으로 불확실성 대폭 감소.
  Clearance 계산(well-stirred model)과 renal CL에 직접 입력.
- **logP**: Crippen 계산값. 실측값과 2+ unit 차이 발생 가능
  (e.g., Cyclosporine: 1.4 vs 3.64). Kp 전 조직, Peff, solubility에 cascade.
- **pKa**: SMARTS class average (acid→4.5, base→9.0). R&R Kp ionization ratio에 직접 영향.
- **CLint 분배**: compound type만으로 CYP별 fm fraction 결정 (neutral → CYP3A4 50%).
  Sobol sensitivity: gut CYP3A4 fm이 Cmax ST=0.47.
  단, CLint total R²=0.24 + meta-learner engine weight 17%로 최종 Cmax에 대한
  fm 개선의 실질 impact은 다른 enrichment보다 작을 수 있음.

DrugBank 5.1 full database (15,485 small molecules) 접근이 확보되어,
1,136 drugs의 CYP substrate annotation, 1,439 drugs의 parsed fup,
9,324 drugs의 ChemAxon pKa, 1,463 drugs의 experimental logP를 활용할 수 있다.

**각 enrichment의 실제 AAFE impact은 ablation study (§6.4)로 측정한다.**

---

## 2. Design Principles

1. **predict layer에 통합 (방법 A).** 별도 enrichment layer를 만들지 않는다.
   adme.py가 models/adme/ XGBoost를 로드하는 것과 동일 패턴으로 data/drugbank/ CSV를 로드.
2. **DrugOnGraph contract 변경 없음.** 출력 type이 동일.
3. **Drug-specific 분기 아님.** `if smiles in drugbank_index`는 data availability routing.
   DrugBank에 있는 모든 약물에 동일한 메커니즘 적용.
4. **파일 없으면 graceful fallback.** data/drugbank/ CSV가 없거나 파싱 실패 시
   전부 기존 경로. FileNotFoundError/ImportError/csv 파싱 에러 모두 catch →
   silver path + logger.warning.
5. **Engine 코드 변경 금지.**
6. **Holdout에서 gold path 사용은 허용.** Pipeline intended behavior. Training data 오염은 금지.
7. **data/drugbank/는 git-ignored.** DrugBank license 준수. 사용자가 직접
   DrugBank access를 확보하고 `scripts/extract_drugbank.py`를 실행해야 함.
   CI에서 CSV 없음 → 자동으로 silver path → 기존 test 영향 없음.
8. **데이터 소스 우선순위.** DrugBank annotation > ChemAxon pKa > compound_type baseline.
   CYP substrate annotation이 있으면 fm 분배에서 compound_type을 override.
   ChemAxon pKa가 있으면 compound_type 분류 + R&R pKa에서 SMARTS를 override.
   둘 다 없으면 기존 SMARTS + compound_type fallback.

---

## 3. Data Availability (Verified)

DrugBank 5.1 (exported 2026-03-05)에서 추출 완료. `data/drugbank/` CSV 5개.

**주의:** 아래 holdout hit count는 drug name matching 기반. 실제 구현은 SMILES/InChIKey
matching을 사용하며, stereo/tautomer 차이로 hit rate가 다를 수 있음 (§4.1 참조).

| Data | Source CSV | Usable Records | Holdout (100 drugs, holdout.json) Hit (name기준) |
|------|-----------|---------------|-------------------|
| CYP substrate annotation | enzyme_annotations.csv | 1,136 drugs (Sisyphus 5 CYP 매칭: 665, 추가 CYP 포함: 1,111) | ~71 |
| fup (parsed from text) | pk_data.csv | 1,439 drugs (parse rate 80.7%) | ~90 |
| ChemAxon pKa (acidic+basic) | drugs.csv | 9,324 drugs with both values | ~97 |
| Experimental logP | experimental_properties.csv | 1,463 drugs | 확인 필요 |
| Experimental caco2 | experimental_properties.csv | 85 drugs | **Scope 밖** (너무 적음) |
| Experimental solubility | experimental_properties.csv | 1,531 drugs (대부분 정성적) | **Scope 밖** |

**DrugBank에 없는 것:** Km, Vmax, Ki, IC50 등 정량적 enzyme kinetics.
Enzyme annotation은 전부 정성적 (substrate/inhibitor/inducer).

---

## 4. Architecture

### 4.1 New Module: `predict/drugbank.py`

SMILES/InChIKey matching으로 DrugBank 데이터를 조회하는 singleton lookup.

```
DrugBankLookup (lazy-loaded singleton)
  ├── _smiles_to_id: dict[canonical_smiles → drugbank_id]
  ├── _inchikey14_to_id: dict[inchikey_first_14_chars → drugbank_id]    ◀ fallback
  ├── _enzyme_roles: dict[drugbank_id → dict[sisyphus_cyp_tag → set[role]]]
  ├── _fup: dict[drugbank_id → float]
  ├── _pka: dict[drugbank_id → (acidic: float, basic: float)]
  ├── _logp_experimental: dict[drugbank_id → float]
  │
  ├── lookup(canonical_smiles) → drugbank_id | None
  ├── get_substrate_enzymes(canonical_smiles) → set[str] | None
  ├── get_fup(canonical_smiles) → float | None
  ├── get_pka(canonical_smiles) → tuple[float, float] | None
  └── get_logp(canonical_smiles) → float | None
```

**매칭 전략 (2-tier):**

1. **Canonical SMILES exact match** (stereo 포함). 대부분의 약물이 여기서 match.
2. **InChIKey connectivity layer fallback** (첫 14자). Stereo/tautomer 차이를 해결.
   drugs.csv에 이미 `inchikey` column 존재. 첫 14자는 connectivity layer hash로
   stereo, protonation, tautomer에 독립적. fup/logP/pKa에 안전. CYP annotation에서
   enantiomer-specific 대사 (S-warfarin CYP2C9 vs R-warfarin CYP3A4) 구분 불가 →
   known limitation (§7.6).

**Build time:** `scripts/extract_drugbank.py`에서 `canonical_smiles` + `inchikey_14`
column 추가. RDKit version 기록.

**Runtime:** CSV 로드 → 두 dict 구성만 (~0.1초). RDKit 호출 없음.

**CYP Normalization:** CSV 로드 시 DrugBank enzyme 이름을 Sisyphus CYP tag로 변환.
`get_substrate_enzymes()`는 이미 정규화된 Sisyphus CYP tag set을 반환.

### 4.2 Integration Points

```
SMILES
  │
  ▼
compute_profile(smiles)
  ├── [RDKit] MW, TPSA, HBD/HBA, rotatable_bonds  (변경 없음)
  ├── [DrugBank] logP experimental lookup → fallback: Crippen logP     ◀ NEW
  ├── [DrugBank] ChemAxon pKa → compound_type 분류 → fallback: SMARTS  ◀ NEW
  └── MolecularProfile (profile.smiles is already canonical)
       │
       ▼
  predict_adme(profile)    ← 기존 signature 유지 (profile.smiles로 lookup)
       ├── [DrugBank] fup measured → fallback: XGBoost                  ◀ NEW
       ├── [XGBoost] CLint prediction  (변경 없음 — 총량은 그대로)
       └── ADMEProperties
            │
            ▼
  build_drug_on_graph(profile, adme, ...)
       ├── [DrugBank] CYP substrate annotation → fm 분배 개선           ◀ NEW
       │    fallback: compound_type 기반 _FM_ADJUSTMENTS
       ├── [R&R] Kp 계산  (pKa/logP 개선에 의한 간접 개선)
       └── DrugOnGraph  (contract 동일)
```

**Signature 원칙:** `predict_adme(profile)` signature는 변경하지 않는다.
`profile.smiles`가 이미 canonical SMILES이므로 (`chemistry.py:297`에서
`Chem.MolToSmiles(mol)`로 생성), 이 값을 DrugBank lookup key로 직접 사용.
pipeline/predict.py 호출부 변경 없음.

**ML layer 영향:** ML direct Cmax predictor (`ml/models.py`)는 `compute_features(smiles)`로
Morgan FP + Crippen logP 기반 features를 직접 생성. DrugBank integration은 ML features에
영향 없음 — ML 모델은 학습 시와 동일한 Crippen logP feature를 사용. Engine path만 변경.

---

## 5. Detailed Design

### 5.1 Enzyme fm Fraction 개선

**현재:** `_get_fm_fractions(compound_type)` — 4가지 compound type별 고정 분배.

**변경:** `_get_fm_fractions(compound_type, substrate_enzymes=None)` — DrugBank substrate
annotation이 있으면 해당 CYP에 가중치 집중.

```python
def _get_fm_fractions(
    compound_type: str,
    substrate_enzymes: set[str] | None = None,
) -> dict[str, float]:
    base_fm = dict(_FM_ADJUSTMENTS.get(compound_type, _DEFAULT_FM))

    if not substrate_enzymes:
        return _normalize_fm(base_fm)

    # Filter to enzymes that exist in base_fm (Sisyphus's 5 CYPs)
    known_substrates = substrate_enzymes & set(base_fm.keys())
    if not known_substrates:
        return _normalize_fm(base_fm)

    # Substrate enzymes get equal share, non-substrates get floor
    _NON_SUBSTRATE_FLOOR = 0.05
    for enzyme in base_fm:
        if enzyme in known_substrates:
            base_fm[enzyme] = 1.0 / len(known_substrates)
        else:
            base_fm[enzyme] = _NON_SUBSTRATE_FLOOR

    return _normalize_fm(base_fm)


def _normalize_fm(fm: dict[str, float]) -> dict[str, float]:
    """Normalize fm fractions to sum to 1.0."""
    total = sum(fm.values())
    if total > 0:
        return {k: v / total for k, v in fm.items()}
    return fm
```

**Call chain threading:** `substrate_enzymes`는 `_get_fm_fractions`에 추가되고,
`_decompose_clint`에도 새 parameter로 전달되어야 한다:

```python
def _decompose_clint(
    clint: Distribution,
    compound_type: str,
    pka: float | None,
    enzyme_abundances: dict[str, float] | None = None,
    substrate_enzymes: set[str] | None = None,  # NEW
) -> dict[str, Distribution]:
    fm = _get_fm_fractions(compound_type, substrate_enzymes)  # pass through
    # ... rest unchanged
```

`build_drug_on_graph`에서 DrugBank lookup → `substrate_enzymes` → `_decompose_clint`:

```python
def build_drug_on_graph(profile, adme, dose_mg, route="oral", liver_enzymes=None):
    # DrugBank CYP substrate lookup
    substrate_enzymes = drugbank_lookup().get_substrate_enzymes(profile.smiles)

    enzyme_affinity = _decompose_clint(
        adme.clint, profile.compound_type, profile.pka,
        enzyme_abundances=abundances,
        substrate_enzymes=substrate_enzymes,  # NEW
    )
    # ... rest unchanged
```

**CYP Normalization Table** (DrugBank → Sisyphus, drugbank.py CSV 로드 시 적용):

| DrugBank CYP | Sisyphus CYP | 근거 |
|-------------|-------------|------|
| Cytochrome P450 3A4 | CYP3A4 | 직접 매핑 |
| Cytochrome P450 2D6 | CYP2D6 | 직접 매핑 |
| Cytochrome P450 1A2 | CYP1A2 | 직접 매핑 |
| Cytochrome P450 2C9 | CYP2C9 | 직접 매핑 |
| Cytochrome P450 2E1 | CYP2E1 | 직접 매핑 |
| Cytochrome P450 3A5 | CYP3A4 | 동일 유전자 패밀리, 205/211 공동 annotated |
| Cytochrome P450 2C19 | CYP2C9 | 동일 2C subfamily. Known limitation §7.1 참조 |
| Cytochrome P450 2C8 | CYP2C9 | 동일 2C subfamily |
| Cytochrome P450 2B6 | (무시) | Sisyphus에 대응 없음 |

**예시:**

| Drug | compound_type | DrugBank substrate | known_substrates (after CYP map) | 현재 fm(CYP3A4) | 변경 fm(CYP3A4) |
|------|-------------|-------------------|----------------------------------|----------------|----------------|
| Midazolam | neutral | CYP3A4 | {CYP3A4} | 0.50 | **0.83** (= 1.0 / (1.0 + 4×0.05)) |
| Metoprolol | base | CYP2D6 | {CYP2D6} | 0.35 | **0.04** (= 0.05 / 1.20) |
| Omeprazole | neutral | CYP2C19+CYP3A4 | {CYP2C9, CYP3A4} | 0.50 | **0.43** (= 0.50 / 1.15) |

**CLint 보존 범위:** Liver total CLint는 보존됨 (Σ fm_i = 1.0).
Gut CLint는 fm_CYP3A4에 비례하여 변화 — gut_wall은 CYP3A4만 보유하므로,
fm_CYP3A4 0.50→0.83 변경 시 gut CLint가 66% 증가.
이것이 gut first-pass extraction 정확도 개선의 메커니즘이다.

### 5.2 pKa Lookup (ChemAxon)

**현재:** `_estimate_pka_type(mol, logp)` — SMARTS 패턴 → class average pKa.

**변경:** DrugBank ChemAxon pKa가 있으면 우선 사용. 없으면 기존 SMARTS fallback.
ChemAxon pKa는 계산값 (RMSE ~0.7 pKa units)이지, 실측값이 아님.
그러나 SMARTS class average (error ~2-3 units)보다 약물별 정확도가 높음.

**compound_type 결정 규칙 (ChemAxon pKa 기반):**

```python
def _classify_from_pka(pka_acidic: float, pka_basic: float) -> tuple[float | None, str]:
    """ChemAxon pKa → (pka_for_rr, compound_type).

    Physiological pH 7.4 기준 ionization 판단.
    Thresholds: acidic < 9 → R&R acid pathway에서 의미 있는 ionization
                basic > 4 → R&R base pathway에서 의미 있는 protonation
    경계값 (e.g., acidic=8.9)에서 acid pathway의 ionization ratio는 ~1.13으로
    neutral과 거의 동일 → threshold 불연속점은 Kp에 실질적 영향 없음.
    """
    is_acidic = pka_acidic < 9.0
    is_basic = pka_basic > 4.0

    if is_acidic and is_basic:
        return pka_basic, "zwitterion"    # R&R base 경로 사용
    elif is_acidic:
        return pka_acidic, "acid"
    elif is_basic:
        return pka_basic, "base"
    else:
        return None, "neutral"
```

**R&R Kp 영향:**
- acid: `10^(tissue_pH - pka)` → pKa 정확도가 ionization ratio에 직접 영향.
- base: `10^(pka - tissue_pH)` + phospholipid binding term.
- 현재 acid pKa=4.5 (class average) → ChemAxon drug-specific 값으로 교체 시
  Kp가 크게 변할 수 있음. 이는 올바른 변화이지만 사전 검증 필요 (§6.2).

### 5.3 fup Measured Lookup

**현재:** `predict_adme()` → XGBoost fup (CV=0.50).

**변경:**

```python
def predict_fup(profile: MolecularProfile) -> Distribution:
    db_fup = _drugbank.get_fup(profile.smiles)  # profile.smiles is canonical
    if db_fup is not None and 0.001 <= db_fup <= 1.0:  # sanity check
        # Cross-validation: DrugBank와 XGBoost가 5x 이상 차이나면 파싱 오류 의심
        xgb_fup = _predict_fup_xgboost(profile).mean
        if xgb_fup > 0 and (db_fup / xgb_fup > 5.0 or xgb_fup / db_fup > 5.0):
            logger.warning(
                "DrugBank fup (%.3f) disagrees with XGBoost (%.3f) by >5x, using XGBoost",
                db_fup, xgb_fup,
            )
            return _predict_fup_xgboost(profile)
        return Distribution(mean=db_fup, cv=0.20)  # measured → lower CV
    return _predict_fup_xgboost(profile)             # silver path
```

**CV=0.20 선택 근거:** 0.15는 measurement precision만 반영하고 inter-individual
variability (IIV)를 과소반영. v1.0의 MC propagation은 prediction uncertainty + IIV를
혼합하므로, 0.20이 타협점 (measured confidence + 최소 IIV).

**5x threshold 근거:** XGBoost fup R²~0.6-0.7이면 typical error ~3x.
5x threshold = DrugBank가 XGBoost 대비 5x 차이 → 둘 중 하나가 확실히 틀림.
예: Cyclosporine parser → fup=0.50, XGBoost → 0.08, ratio=6.25x > 5x → fallback. ✓

### 5.4 Experimental logP Lookup

**현재:** `compute_profile()` → `Descriptors.MolLogP(mol)` (Crippen).

**변경:** DrugBank experimental logP가 있으면 우선 사용.

```python
def compute_profile(smiles: str) -> MolecularProfile:
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol)
    logp = Descriptors.MolLogP(mol)  # default: Crippen

    db_logp = _drugbank.get_logp(canonical)
    if db_logp is not None:
        logp = db_logp  # experimental overrides computed

    # ... rest unchanged, logp flows into pKa, Kp, Peff, solubility
```

**Cascade 영향:** logP는 다음에 입력됨:
- Kp 계산 (R&R): `P = 10^logP` — 가장 큰 영향
- Peff heuristic: `10^(0.4*logP - 0.4)`
- Solubility heuristic: `log10(S) = -logP + 0.5`
- AD check: `EXTREME_LIPOPHILIC` threshold

**AD check 결정:** AD check (`_check_ad`)도 experimental logP를 사용한다.
AD check의 목적은 "이 약물이 예측 가능한 범위 내인가"이며, 실제 lipophilicity가
높으면 정당한 경고. Crippen이 부정확한 경우 거짓 양성 AD flag를 제거하는 것이 올바르다.

실측 logP가 Crippen과 크게 다를 수 있음 (§7.4 known limitation).
Benchmark에서 분리 보고로 영향 측정.

### 5.5 Warning Tag 전파

**문제:** predict layer 함수들은 frozen dataclass를 반환한다 (`MolecularProfile`,
`ADMEProperties`, `DrugOnGraph`). warnings 필드가 없으므로 DrugBank 사용 여부를
직접 반환할 수 없다.

**해결:** `pipeline/predict.py`에서 predict-layer 호출 후, `DrugBankLookup` singleton을
직접 조회하여 어떤 경로가 사용되었는지 판단한다.

**주의:** tag 로직은 predict-layer 내부의 acceptance 로직을 반영해야 한다.
단순히 "데이터가 있는가"가 아니라 "데이터가 사용되었는가"를 추적.

```python
# pipeline/predict.py (기존 warnings_list 변수 활용)
from sisyphus.predict.drugbank import drugbank_lookup

canonical = profile.smiles
db = drugbank_lookup()

# enzyme_fm: data availability = usage (no rejection logic)
if db.get_substrate_enzymes(canonical) is not None:
    warnings_list.append("drugbank:enzyme_fm")

# fup: replicate sanity check + cross-validation
db_fup = db.get_fup(canonical)
if db_fup is not None and 0.001 <= db_fup <= 1.0:
    xgb_fup = adme.fup.mean  # from predict_adme (XGBoost result before override)
    if xgb_fup <= 0 or (db_fup / xgb_fup <= 5.0 and xgb_fup / db_fup <= 5.0):
        warnings_list.append("drugbank:fup")

# pka, logp: data availability = usage (no rejection logic)
if db.get_pka(canonical) is not None:
    warnings_list.append("drugbank:pka")
if db.get_logp(canonical) is not None:
    warnings_list.append("drugbank:logp")
```

**Known imprecision:** fup tag에서 5x cross-validation을 복제하지만,
pipeline에서의 `adme.fup.mean`은 이미 DrugBank override된 값일 수 있음.
XGBoost-only fup를 별도로 계산하지 않으면 완벽한 복제가 불가.
실무적으로: rejection은 ~5-10% 빈도이므로, 소수의 false positive tag를
허용한다 (gold group에 silver drug 소수 혼입). Benchmark 해석에 큰 영향 없음.

### 5.6 Benchmark 분리 보고

**PredictionResult contract 변경 없음.** warnings tuple에 §5.5의 태그가 포함됨.

```python
# validation/benchmark.py
def run_holdout_benchmark(...) -> BenchmarkResult:
    # ... existing logic ...
    results_gold = [r for r in results if any("drugbank:" in w for w in r.warnings)]
    results_silver = [r for r in results if not any("drugbank:" in w for w in r.warnings)]
```

**`BenchmarkResult` 변경:** frozen dataclass이므로 새 fields에 defaults 필수.
```python
@dataclass(frozen=True)
class BenchmarkResult:
    # ... existing fields ...
    aafe_gold: float | None = None
    aafe_silver: float | None = None
    n_gold: int = 0
    n_silver: int = 0
```

### 5.7 Feature Flags (Ablation 지원)

개별 enrichment을 독립적으로 toggle하여 어느 enrichment이 AAFE에
어떤 영향을 미치는지 진단 가능하게 한다.

```python
# predict/drugbank.py
@dataclass
class DrugBankConfig:
    enable_enzyme_fm: bool = True
    enable_fup: bool = True
    enable_pka: bool = True
    enable_logp: bool = True
```

DrugBankLookup의 각 get_* 메서드는 해당 flag가 False이면 None 반환.
Default는 all True. 테스트에서는 config 주입으로 개별 제어.

`data/drugbank/` CSV가 없으면 모든 lookup이 자동으로 None → feature flag 불필요.
Feature flags는 CSV가 있을 때 개별 enrichment 효과 측정용.

---

## 6. Validation Plan

### 6.0 Baseline 기록

DrugBank 적용 전 holdout benchmark를 실행하여 v1.0 baseline AAFE를 기록한다.
이후 모든 비교의 기준점.

### 6.1 Regression Gate

DrugBank CSV 없이 모든 기존 test 통과 확인. 253 tests green.
data/drugbank/가 git-ignored이므로 CI에서 자동으로 이 조건 충족.

### 6.2 pKa Compound Type 검증

**전체** holdout 약물 (~97 drugs with ChemAxon pKa)에서 비교:

| Drug | SMARTS compound_type | ChemAxon compound_type | 일치? | Kp 차이 |
|------|---------------------|----------------------|------|---------|

불일치 비율 > 30%이면 threshold 재조정 후 재검증.
compound_type이 다르더라도 Kp 차이가 < 2x이면 benign (threshold 경계의 약한 ionization).

### 6.3 Gold/Silver AAFE 분리

Holdout benchmark에서:
- All drugs AAFE (baseline 대비)
- Gold path (DrugBank-enhanced) AAFE
- Silver path (XGBoost-only) AAFE
- Gold vs Silver AAFE 차이

**Rollback 기준:** Gold AAFE > baseline AAFE이면 DrugBank integration이
error cancellation을 깨뜨린 신호. Feature flags로 원인 enrichment 식별 후
해당 enrichment만 disable.

**통계적 주의:** Silver group (N ≈ 29)은 작으므로 AAFE 추정이 noisy할 수 있음.

### 6.4 Ablation Study

Feature flags로 하나씩 켜면서 AAFE 변화 측정:
1. fup only
2. logP only
3. pKa only
4. enzyme_fm only
5. all combined

### 6.5 logP Impact 검증

DrugBank에 Vd parsed 값 (1,044 drugs) 존재. Experimental logP 적용 전/후
engine predicted Vd를 DrugBank measured Vd와 비교 — Kp "변화량"이 아닌
Vd "정확도" 개선 여부를 직접 측정.

### 6.6 UGT-dominant Drug 검증

Known risk: UGT-dominant drugs (morphine, lamotrigine, zidovudine 등)에서
CYP3A4 annotation 적용 시 gut first-pass 과대예측 여부 확인.
이 약물들의 Cmax 변화 방향이 일관되게 감소 (과도한 gut extraction)하면
fm algorithm에 non-CYP damping 또는 해당 약물군 제외 필요.

---

## 7. Known Limitations

### 7.1 CYP2C19 → CYP2C9 매핑은 약리학적으로 부정확

CYP2C19 substrate는 216 drugs (omeprazole, clopidogrel 등).
Sisyphus BodyGraph에 CYP2C19이 없으므로 CYP2C9에 매핑.
후속 작업: `reference_man.yaml`에 CYP2C19 추가 (engine 변경 0줄, YAML만).

### 7.2 DrugBank enzyme annotation은 정성적

"CYP3A4 substrate"만 있고 fm 비율은 없음.
균등 분배 (1/N substrates) 가정은 실제 fm과 다를 수 있음.
특히 major/minor substrate 구분이 없으므로, minor CYP pathway에 과대 가중치.
Major/minor substrate 구분 데이터가 확보되면 개선 가능.

### 7.3 UGT-dominant drugs에서 fm 악화 가능

Sisyphus v1.0은 CYP-only clearance model. Non-CYP 대사 (UGT, esterase 등)가
모델에 없으므로, 총 CLint가 전부 CYP에 배분됨.

DrugBank에서 CYP annotation만 사용하면 (UGT annotation 무시), UGT-dominant drugs
(e.g., morphine: CYP3A4 minor + UGT2B7 major)에서 fm_CYP3A4가 과대 추정 →
gut first-pass 과대예측 → Cmax 과소예측.

이건 DrugBank integration의 문제가 아니라 CYP-only model의 기존 한계이나,
DrugBank fm이 compound_type baseline보다 더 aggressive하므로 (0.83 vs 0.50)
일부 약물에서 regression 발생 가능. §6.6에서 targeted 검증.

후속 작업: YAML에 UGT enzyme 추가 (engine 변경 0줄).

### 7.4 Experimental logP와 Crippen logP의 차이

Cyclosporine: experimental 1.4 vs Crippen 3.64 (ΔlogP = 2.24).
Kp, Peff, solubility heuristic에 cascade 영향.
Benchmark 분리 보고로 영향 측정 후 필요 시 rollback.

### 7.5 Parsed fup 정확도

자동 파싱 정확도: 80.7%. 파싱된 값 중에도 오류 가능
(e.g., erythrocyte distribution % → protein binding으로 오파싱).
5x cross-validation guard가 대부분 잡지만, XGBoost도 같은 방향으로 틀리면 검출 불가.

### 7.6 InChIKey matching의 stereo 미구분

InChIKey connectivity layer (첫 14자)는 enantiomer를 구분하지 않음.
S-warfarin (CYP2C9 primary) ≠ R-warfarin (CYP3A4 primary)이지만
같은 InChIKey connectivity → 같은 annotation 반환.
대부분의 약물에서 무해하지만, 특정 chiral drugs에서 부정확 가능.

### 7.7 Error cancellation 위험

Omega 경험: "predicted ADME beat measured ADME" — sequential pipeline에서
prediction error가 상쇄되는 현상. Sisyphus의 enzyme-level architecture에서
동일 패턴이 존재하는지는 미확인.

DrugBank measured values가 이 error cancellation을 깨뜨리면 AAFE가 악화될 수 있음.
§6.3의 rollback 기준이 안전망.

### 7.8 Meta-learner feature distribution shift

Meta-learner (`xgboost_meta.json`)의 `combine()` 메서드는 engine_pk, ml_pk 외에
logP, fup, compound_type 등을 feature로 받음. 이 값들이 DrugBank integration으로
변경되면 학습 시와 다른 feature 분포를 보게 됨 (measured values는 XGBoost predicted
values보다 noise가 적음). Feature shift의 magnitude는 작을 것으로 예상되나,
§6.4 ablation에서 간접 측정됨.

### 7.9 Caco2, solubility는 이번 scope 밖

Caco2: 85 drugs only — 모델 학습이나 lookup 모두 불충분.
Solubility: 대부분 정성적 ("insoluble") — 정량 heuristic 대체 불가.

---

## 8. File Changes

### New files
- `predict/drugbank.py` — DrugBank lookup module + DrugBankConfig
- `.gitignore` — `data/drugbank/` 추가

### Modified files
- `scripts/extract_drugbank.py` — canonical_smiles, inchikey_14 column 추가, RDKit version 기록
- `predict/chemistry.py` — logP/pKa DrugBank lookup integration
- `predict/adme.py` — fup DrugBank lookup with 5x cross-validation
- `predict/ivive.py` — `_get_fm_fractions()` substrate_enzymes parameter 추가
- `pipeline/predict.py` — DrugBank warning tags 추가 (§5.5)
- `validation/benchmark.py` — gold/silver 분리 보고
- `Sisyphus_Design_v4.md` — §3.2 predict 의존성 업데이트

### Not modified
- `engine/` — 전체 디렉토리 변경 없음
- `core.py` — DrugOnGraph, PredictionResult contract 변경 없음
- `graph/` — BodyGraph 변경 없음
- `ml/` — ML layer 변경 없음
- `data/physiology/` — YAML 변경 없음 (CYP2C19/UGT 추가는 후속)

### predict/ file count
현재 4 files → drugbank.py 추가 = 5 files. (상한 20)

---

## 9. Acceptance Criteria

1. `data/drugbank/` CSV 없이 253 tests all pass (regression zero).
2. DrugBank CSV 있을 때 hit drug → gold path, miss drug → silver path.
   로그에서 `"drugbank:enzyme_fm"`, `"drugbank:fup"` 등 태그로 확인 가능.
3. Holdout benchmark에서 gold/silver 분리 보고 출력.
4. Gold path drugs (N≥20)에서 AAFE 측정 (개선 보장 아님 — 측정이 목표).
5. pKa compound_type 검증: **전체** holdout drugs에서 SMARTS vs ChemAxon 비교 완료.
6. `scripts/extract_drugbank.py`에 canonical_smiles + inchikey_14 생성 + RDKit version 기록.
7. Feature flags로 개별 enrichment toggle 가능.
8. Baseline AAFE 기록 후 DrugBank 적용 전후 비교 가능.

---

## 10. Out of Scope

- Drug-specific `if drug == "warfarin"` 분기.
- Engine 코드 변경.
- DrugOnGraph dataclass field 추가/삭제.
- Holdout drugs의 DrugBank 데이터를 ADME model training에 사용.
- Caco2 permeability lookup (85 drugs — 불충분).
- Solubility lookup (정성적 데이터).
- CYP2C19 YAML 추가 (후속 작업).
- UGT enzyme YAML 추가 (후속 작업).
- fup/CLint model 재학습 (별도 scope).
- Reference data 교차 검증 스크립트 (별도 scope).
- ChEMBL/PubChem에서 quantitative CYP kinetics (IC50, Ki, Km/Vmax) 확보 →
  fm 정량화 (별도 data science project).
- DrugBank inhibitor/inducer annotation → DDI module 자동 파라미터 생성 (별도 scope).
