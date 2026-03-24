# DrugBank Data Integration into Sisyphus v1.0

**Date:** 2026-03-23
**Status:** Draft
**Scope:** predict layer enhancement — zero engine changes

---

## 1. Motivation

Sisyphus v1.0의 ADME prediction pipeline은 전부 XGBoost/heuristic 기반이다.
핵심 약점:

- **CLint 분배**: compound type만으로 CYP별 fm fraction 결정 (neutral → CYP3A4 50%).
  Metoprolol (CYP2D6 primary substrate)도 neutral이면 CYP3A4에 50% 배정.
  Sobol sensitivity: gut CYP3A4 fm이 Cmax ST=0.47 — fm 정확도가 Cmax에 직결.
- **pKa**: SMARTS class average (acid→4.5, base→9.0). R&R Kp 계산에 직접 영향.
- **fup**: XGBoost CV=0.50. 실측값이 있으면 CV=0.15로 불확실성 대폭 감소.
- **logP**: Crippen 계산값. 실측값과 2+ unit 차이 발생 가능 (e.g., Cyclosporine: 1.4 vs 3.64).

DrugBank 5.1 full database (15,485 small molecules) 접근이 확보되어,
1,136 drugs의 CYP substrate annotation, 1,439 drugs의 parsed fup,
9,324 drugs의 ChemAxon pKa, 1,463 drugs의 experimental logP를 활용할 수 있다.

---

## 2. Design Principles

1. **predict layer에 통합 (방법 A).** 별도 enrichment layer를 만들지 않는다.
   adme.py가 models/adme/ XGBoost를 로드하는 것과 동일 패턴으로 data/drugbank/ CSV를 로드.
2. **DrugOnGraph contract 변경 없음.** 출력 type이 동일.
3. **Drug-specific 분기 아님.** `if smiles in drugbank_index`는 data availability routing.
   DrugBank에 있는 모든 약물에 동일한 메커니즘 적용.
4. **파일 없으면 graceful fallback.** data/drugbank/ CSV가 없으면 전부 기존 경로.
   FileNotFoundError/ImportError 발생 안 함.
5. **Engine 코드 변경 금지.**
6. **Holdout에서 gold path 사용은 허용.** Pipeline intended behavior. Training data 오염은 금지.

---

## 3. Data Availability (Verified)

DrugBank 5.1 (exported 2026-03-05)에서 추출 완료. `data/drugbank/` CSV 5개.

| Data | Source CSV | Usable Records | Holdout (100 drugs, from holdout.json) 중 Hit |
|------|-----------|---------------|-------------------|
| CYP substrate annotation | enzyme_annotations.csv | 1,136 drugs (Sisyphus 5 CYP 매칭: 665, 추가 CYP 포함: 1,111) | 71 |
| fup (parsed from text) | pk_data.csv | 1,439 drugs (parse rate 80.7%) | 90 |
| ChemAxon pKa (acidic+basic) | drugs.csv | 9,324 drugs with both values | ~97 |
| Experimental logP | experimental_properties.csv | 1,463 drugs | 확인 필요 |
| Experimental caco2 | experimental_properties.csv | 85 drugs | **Scope 밖** (너무 적음) |
| Experimental solubility | experimental_properties.csv | 1,531 drugs (대부분 정성적) | **Scope 밖** |

**DrugBank에 없는 것:** Km, Vmax, Ki, IC50 등 정량적 enzyme kinetics.
Enzyme annotation은 전부 정성적 (substrate/inhibitor/inducer).

---

## 4. Architecture

### 4.1 New Module: `predict/drugbank.py`

SMILES canonical matching으로 DrugBank 데이터를 조회하는 singleton lookup.

```
DrugBankLookup (lazy-loaded singleton)
  ├── _smiles_to_id: dict[canonical_smiles → drugbank_id]
  ├── _enzyme_roles: dict[drugbank_id → dict[cyp_tag → set[role]]]
  ├── _fup: dict[drugbank_id → float]
  ├── _pka: dict[drugbank_id → (acidic: float, basic: float)]
  ├── _logp_experimental: dict[drugbank_id → float]
  │
  ├── lookup(canonical_smiles) → DrugBankMatch | None
  ├── get_substrate_enzymes(canonical_smiles) → set[str] | None
  ├── get_fup(canonical_smiles) → float | None
  ├── get_pka(canonical_smiles) → tuple[float, float] | None
  └── get_logp(canonical_smiles) → float | None
```

**매칭 전략:**
- Build time: `scripts/extract_drugbank.py`에서 RDKit canonical SMILES를 생성하여
  `drugs.csv`에 `canonical_smiles` column 추가.
- Runtime: CSV 로드 → dict 구성만 (~0.1초). RDKit 호출 없음.
- `compute_profile(smiles)` 에서 이미 canonical SMILES를 생성하므로 (`Chem.MolToSmiles(mol)`),
  이 값을 lookup key로 사용.

**데이터 소스:** 이미 추출된 CSV를 직접 로드. 별도 JSON 가공 불필요.

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
       ├── [R&R] Kp 계산  (pKa 개선에 의한 간접 개선)
       └── DrugOnGraph  (contract 동일)
```

**Signature 원칙:** `predict_adme(profile)` signature는 변경하지 않는다.
`profile.smiles`가 이미 canonical SMILES이므로 (`chemistry.py:297`에서
`Chem.MolToSmiles(mol)`로 생성), 이 값을 DrugBank lookup key로 직접 사용.
pipeline/predict.py 호출부 변경 없음.

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

**CYP Normalization Table** (DrugBank → Sisyphus):

| DrugBank CYP | Sisyphus CYP | 근거 |
|-------------|-------------|------|
| Cytochrome P450 3A4 | CYP3A4 | 직접 매핑 |
| Cytochrome P450 2D6 | CYP2D6 | 직접 매핑 |
| Cytochrome P450 1A2 | CYP1A2 | 직접 매핑 |
| Cytochrome P450 2C9 | CYP2C9 | 직접 매핑 |
| Cytochrome P450 2E1 | CYP2E1 | 직접 매핑 |
| Cytochrome P450 3A5 | CYP3A4 | 동일 유전자 패밀리, 205/211 공동 annotated |
| Cytochrome P450 2C19 | CYP2C9 | 동일 2C subfamily. Known limitation §7 참조 |
| Cytochrome P450 2C8 | CYP2C9 | 동일 2C subfamily |
| Cytochrome P450 2B6 | (무시) | Sisyphus에 대응 없음 |

**예시:**

| Drug | compound_type | DrugBank substrate | known_substrates (after CYP map) | 현재 fm(CYP3A4) | 변경 fm(CYP3A4) |
|------|-------------|-------------------|----------------------------------|----------------|----------------|
| Midazolam | neutral | CYP3A4 | {CYP3A4} | 0.50 | **0.83** (= 1.0 / (1.0 + 4×0.05)) |
| Metoprolol | base | CYP2D6 | {CYP2D6} | 0.35 | **0.04** (= 0.05 / 1.20) |
| Omeprazole | neutral | CYP2C19+CYP3A4 | {CYP2C9, CYP3A4} | 0.50 | **0.43** (= 0.50 / 1.15) |

총 CLint는 변경 없음 — XGBoost predicted CLint의 분배만 개선.

### 5.2 pKa Lookup (ChemAxon)

**현재:** `_estimate_pka_type(mol, logp)` — SMARTS 패턴 → class average pKa.

**변경:** DrugBank ChemAxon pKa가 있으면 우선 사용. 없으면 기존 SMARTS fallback.

**compound_type 결정 규칙 (ChemAxon pKa 기반):**

```python
def _classify_from_pka(pka_acidic: float, pka_basic: float) -> tuple[float | None, str]:
    """ChemAxon pKa → (pka_for_rr, compound_type).

    Physiological pH 7.4 기준 ionization 판단.
    Thresholds: acidic < 9 → 산성 이온화 의미 있음
                basic > 4 → 염기성 이온화 의미 있음
    """
    is_acidic = pka_acidic < 9.0   # meaningfully ionized at pH 7.4
    is_basic = pka_basic > 4.0     # meaningfully protonated at pH 7.4

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
    match = _drugbank.get_fup(profile.smiles)  # profile.smiles is canonical
    if match is not None and 0.001 <= match <= 1.0:  # sanity check
        return Distribution(mean=match, cv=0.15)      # measured → lower CV
    return _predict_fup_xgboost(profile)               # silver path
```

**Sanity check 필수:** Parsed fup 중 일부는 ambiguous text에서 추출됨
(e.g., "50% taken up by erythrocytes" — RBP가 아닌 protein binding으로 오파싱 가능).
0.001-1.0 범위 밖이면 fallback.

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
높으면 정당한 경고. Crippen이 부정확한 경우 (e.g., computed logP=3.6이지만
experimental=1.4), 거짓 양성 AD flag를 제거하는 것이 올바르다.

실측 logP가 Crippen과 크게 다를 수 있음 (§7 known limitation).
Benchmark에서 분리 보고로 영향 측정.

### 5.5 Warning Tag 전파

**문제:** predict layer 함수들은 frozen dataclass를 반환한다 (`MolecularProfile`,
`ADMEProperties`, `DrugOnGraph`). warnings 필드가 없으므로 DrugBank 사용 여부를
직접 반환할 수 없다.

**해결:** `pipeline/predict.py`에서 predict-layer 호출 후, `DrugBankLookup` singleton을
직접 조회하여 어떤 경로가 사용되었는지 판단한다. predict layer 내부에서 tag를 반환하지
않아도 됨 — pipeline이 canonical_smiles로 DrugBank에 데이터가 있었는지 확인 가능.

```python
# pipeline/predict.py (기존 warnings_list 변수 활용)
from sisyphus.predict.drugbank import drugbank_lookup

# Step 1 후 (compute_profile + predict_adme + build_drug_on_graph)
canonical = profile.smiles
db = drugbank_lookup()
if db.get_substrate_enzymes(canonical) is not None:
    warnings_list.append("drugbank:enzyme_fm")
if db.get_fup(canonical) is not None:
    warnings_list.append("drugbank:fup")
if db.get_pka(canonical) is not None:
    warnings_list.append("drugbank:pka")
if db.get_logp(canonical) is not None:
    warnings_list.append("drugbank:logp")
```

이 방식은 predict layer의 signature/return type을 변경하지 않으면서,
pipeline의 기존 `warnings_list` 패턴에 자연스럽게 통합된다.

### 5.6 Benchmark 분리 보고

**PredictionResult contract 변경 없음.** warnings tuple에 §5.5의 태그가 포함됨.

```python
# validation/benchmark.py
def run_holdout_benchmark(...) -> BenchmarkResult:
    # ... existing logic ...
    # 추가: gold/silver 분리
    results_gold = [r for r in results if any("drugbank:" in w for w in r.warnings)]
    results_silver = [r for r in results if not any("drugbank:" in w for w in r.warnings)]

    # BenchmarkResult에 optional fields 추가 (frozen dataclass 호환)
    # aafe_gold: float | None = None
    # aafe_silver: float | None = None
    # n_gold: int = 0
    # n_silver: int = 0
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

---

## 6. Validation Plan

### 6.1 Regression Gate

DrugBank 파일 없이 모든 기존 test 통과 확인. 253 tests green.

### 6.2 pKa Compound Type 검증

Holdout 약물 중 10개를 선택하여 비교:

| Drug | SMARTS compound_type | ChemAxon compound_type | 일치? |
|------|---------------------|----------------------|------|

불일치 비율 > 30%이면 threshold 재조정 후 재검증.

### 6.3 Gold/Silver AAFE 분리

Holdout benchmark에서:
- All drugs AAFE (기존과 비교)
- Gold path (DrugBank-enhanced) AAFE
- Silver path (XGBoost-only) AAFE
- Gold vs Silver AAFE 차이 — 측정이 목적, 개선 보장은 아님.

### 6.4 logP Impact 검증

Experimental logP 적용 전/후 Kp 비교 (약물 5개).
Kp 변화가 >3x인 경우 수동 확인.

---

## 7. Known Limitations

1. **CYP2C19 → CYP2C9 매핑은 약리학적으로 부정확.**
   CYP2C19 substrate는 216 drugs (omeprazole, clopidogrel 등).
   Sisyphus BodyGraph에 CYP2C19이 없으므로 CYP2C9에 매핑.
   후속 작업: `reference_man.yaml`에 CYP2C19 추가 (engine 변경 0줄, YAML만).

2. **DrugBank enzyme annotation은 정성적.**
   "CYP3A4 substrate"만 있고 fm 비율은 없음.
   균등 분배 (1/N substrates) 가정은 실제 fm과 다를 수 있음.
   Major/minor substrate 구분 데이터가 확보되면 개선 가능.

3. **Parsed fup/pKa/logP는 text extraction 기반.**
   자동 파싱 정확도: fup 80.7%, logP/pKa 파싱은 structured field라 높음.
   개별 값의 정확성은 보장되지 않으며, sanity check으로 이상치 제거.

4. **Experimental logP와 Crippen logP의 차이가 클 수 있음.**
   Cyclosporine: experimental 1.4 vs Crippen 3.64 (ΔlogP = 2.24).
   Kp, Peff, solubility heuristic에 cascade 영향.
   benchmark 분리 보고로 영향 측정 후 필요 시 rollback.

5. **Caco2, solubility는 이번 scope 밖.**
   Caco2: 85 drugs only — 모델 학습이나 lookup 모두 불충분.
   Solubility: 대부분 정성적 ("insoluble") — 정량 heuristic 대체 불가.

---

## 8. File Changes

### New files
- `predict/drugbank.py` — DrugBank lookup module

### Modified files
- `scripts/extract_drugbank.py` — canonical_smiles column 추가, RDKit version 기록
- `predict/chemistry.py` — logP/pKa DrugBank lookup integration
- `predict/adme.py` — fup DrugBank lookup
- `predict/ivive.py` — `_get_fm_fractions()` substrate_enzymes parameter 추가
- `validation/benchmark.py` — gold/silver 분리 보고
- `Sisyphus_Design_v4.md` — §3.2 predict 의존성 업데이트

### Not modified
- `engine/` — 전체 디렉토리 변경 없음
- `core.py` — DrugOnGraph, PredictionResult contract 변경 없음
- `graph/` — BodyGraph 변경 없음
- `ml/` — ML layer 변경 없음
- `data/physiology/` — YAML 변경 없음 (CYP2C19 추가는 후속)

### predict/ file count
현재 4 files → drugbank.py 추가 = 5 files. (상한 20)

---

## 9. Acceptance Criteria

1. `data/drugbank/` CSV 없이 253 tests all pass (regression zero).
2. DrugBank CSV 있을 때 hit drug → gold path, miss drug → silver path.
   로그에서 `"drugbank:enzyme_fm"`, `"drugbank:fup"` 등 태그로 확인 가능.
3. Holdout benchmark에서 gold/silver 분리 보고 출력.
4. Gold path drugs (N≥20)에서 AAFE 측정 (개선 보장 아님 — 측정이 목표).
5. pKa compound_type 검증: holdout 10개에서 SMARTS vs ChemAxon 비교 완료.
6. `scripts/extract_drugbank.py`에 canonical_smiles 생성 + RDKit version 기록.

---

## 10. Out of Scope

- Drug-specific `if drug == "warfarin"` 분기.
- Engine 코드 변경.
- DrugOnGraph dataclass field 추가/삭제.
- Holdout drugs의 DrugBank 데이터를 ADME model training에 사용.
- Caco2 permeability lookup (85 drugs — 불충분).
- Solubility lookup (정성적 데이터).
- CYP2C19 YAML 추가 (후속 작업).
- fup/CLint model 재학습 (별도 scope).
- Reference data 교차 검증 스크립트 (별도 scope — 이번은 predict 통합에 집중).
