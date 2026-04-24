# Sisyphus 현재 디렉토리 완성도 평가 및 감사안

> **STATUS: 최신 감사 (CURRENT) — 2026-04-24 하드닝 개정판**
>
> 이 문서가 Sisyphus 디렉토리 완성도에 대한 **현재 기준 감사**다. 상위 감사 문서(`docs/claude_md_audit.md` 2026-04-20)는 CLAUDE.md 문서 품질만 다루며 스코프가 다르다. 차기 감사가 나오기 전까지 이 파일이 reference다.

**작성일:** 2026-04-24 (최초판) / 2026-04-24 (하드닝 개정) / 2026-04-24 (counter-audit patch, Option α)
**대상 디렉토리:** `/home/jam/Sisyphus`
**평가 방식:** 읽기 전용 구조 감사. 코드/데이터/설정/문서/테스트 구성을 확인했으며, 테스트 실행은 하지 않았다. 초안 사실관계는 `git ls-files`, `grep`, `python3 json` 로 교차 검증함.
**주의:** 이 문서는 현재 작업 디렉토리 상태 기준의 스냅샷이다. git 추적 상태와 로컬 artifact 상태가 다를 수 있다.

## 1. 종합 판정

현재 Sisyphus 디렉토리는 단순 실험 폴더가 아니라, 패키징, CLI, 검증 데이터, 문서, CI, 모델 artifact, 연구 로그를 갖춘 성숙한 연구형 Python 프로젝트다. 다만 공개 릴리스나 외부 사용자용 안정 제품으로 보기에는 git 상태 정리, 문서 최신화, 일부 미구현 API 정리, CI 게이트 강화가 더 필요하다.

| 기준 | 완성도 |
|---|---:|
| 연구 코드 / 내부 실험 플랫폼 | 80-85% |
| 재현 가능한 공개 릴리스 | 65-70% |
| 안정적 프로덕션 / 외부 사용자 도구 | 55-60% |

**한 줄 결론:** 핵심 연구 플랫폼으로는 상당히 완성되어 있으나, 릴리스 품질로 올리려면 "새 기능"보다 "상태 정리와 재현성 고정"이 우선이다.

## 2. 확인한 주요 근거

### 2.1 프로젝트 구조

- `pyproject.toml` 존재: Python 패키지 메타데이터, optional dependency, CLI entrypoint 정의.
- `README.md` 존재: 방법론, Quickstart, Validation, Architecture, Limitations, Citation까지 포함.
- `.github/workflows/ci.yml` 존재: GitHub Actions CI 구성.
- `requirements-lock.txt` 존재: lockfile 기반 재현 설치 경로 제공.
- `src/sisyphus/` 아래 주요 레이어가 분리되어 있음:
  - `engine/`
  - `graph/`
  - `predict/`
  - `pipeline/`
  - `regimen/`
  - `sbi/`
  - `validation/`
  - `ml/`
  - `pk/`

### 2.2 규모

확인 시점의 대략적 규모:

| 항목 | 수량 |
|---|---:|
| `src/sisyphus` Python 파일 | 66 |
| `tests` 테스트 파일 | 63 |
| `scripts` Python 스크립트 | 98 |
| `src/sisyphus` LOC | 약 14,033 |
| `tests` LOC | 약 10,617 |
| `scripts` LOC | 약 31,434 |
| 테스트 케이스 총 | 약 635 (top-level `def test_` 222 + 클래스 내 메서드 413) |
| 테스트 클래스 | 92 |

**수치 정정 기록:** 초기 판본은 "테스트 222"로만 적어 클래스 내 메서드 413개를 누락, 테스트 규모를 3배가량 저평가했다. 파일 수도 59 → 63로 정정.

### 2.3 테스트 및 검증 자산

- unit/integration/benchmark 테스트 디렉토리가 분리되어 있다.
- README는 전체 테스트 수치와 validation 결과를 제시한다.
- 다수의 validation artifact가 `data/validation/`에 존재한다.
- `docs/claude/`, `docs/superpowers/`, `docs/science/` 아래에 실험 이력, 실패 기록, 설계 문서, hardening backlog가 축적되어 있다.

### 2.4 모델 및 데이터 자산

- `models/adme/`, `models/direct_pk/`, `models/sbi/`, `models/surrogate/`에 모델 artifact가 존재한다.
- `data/reference/`, `data/training/`, `data/validation/`, `data/transporters/`, `data/physiology/`, `data/sbi/`가 구성되어 있다.
- `data/sbi/method_routing.json` 기준 production routing 요약은 다음과 같다:
  - `sbi`: 12
  - `ibis`: 1
  - `is`: 0
  - total: 13

## 3. 강점

### 3.1 아키텍처 분리가 좋다

엔진, 그래프, 예측, validation, regimen/TDM, SBI가 비교적 분명히 분리되어 있다. 연구 코드에서 흔히 보이는 거대한 단일 스크립트 형태가 아니라, 패키지로 유지하려는 구조가 이미 잡혀 있다.

### 3.2 문서화 수준이 높다

`README.md`는 사용법뿐 아니라 모델 방법론, validation 결과, limitation까지 적고 있다. 특히 한계와 실패 이력을 명시한 점은 연구 프로젝트로서 신뢰도를 높인다.

관련 문서 예시:

- `docs/reproducibility.md`
- `docs/claude/hardening_backlog.md`
- `docs/claude/propranolol_cmax_drift.md`
- `docs/holdout_contamination_audit.md`
- `docs/sbi_multi_drug_results.md`
- `docs/science/ecm_unit_audit.md`

### 3.3 검증 문화가 있다

테스트 파일 수와 validation artifact가 많고, xfail/skip에도 이유가 붙어 있다. 특히 propranolol drift, statin Peff limitation, cherry-picking caveat 등이 문서로 남아 있는 점은 "결과만 좋은 척"하는 코드보다 훨씬 건강하다.

### 3.4 재현성 인프라가 일부 갖춰져 있다

`requirements-lock.txt`, CI, benchmark smoke, reproducibility 문서가 있다. 완전하지는 않지만, 재현 가능성에 대한 의식은 분명하다.

## 4. 주요 미완성 / 리스크

### 4.1 git 작업트리가 깨끗하지 않다 — 일부는 **릴리스 블로커**

확인 시점의 `git status` 기준:

- 수정됨 (tracked):
  - `data/validation/prospective_batch_N5_kinase.json` — 5개 drug의 `meta`/`fold` 값이 재계산된 흔적 (11줄 insertion/deletion). P4.5 Achour 머지 이후 재실행 결과로 추정. **커밋 대상인지, 실험 중 의도치 않은 덮어쓰기인지 명시적 결정 필요.**
- untracked:
  - `.claude/` — agent/로컬 설정. artifact가 아니라 `.gitignore`로 고정 대상.
  - `data/sbi/*.npz`, `*.meta.json` (학습 데이터)
  - `docs/superpowers/...` (스펙/플랜 문서)
  - `models/sbi/*.pt`, `*.aux.pt`

#### ⚠ production 모델 untracked 교차검증

`src/sisyphus/regimen/tdm_sbi.py` 기본 로더가 참조하는 production 모델 경로와 tracked 여부를 대조:

| 경로 | 참조 위치 | tracked? |
|---|---|---|
| `models/sbi/multi_drug_nsf.pt` | `tdm_sbi.py:40` (`_DEFAULT_POSTERIOR_PATH`) | ✗ untracked |
| `models/sbi/hierarchical_nsf_2k.pt` | `tdm_sbi.py:54` (`_DEFAULT_HIERARCHICAL_POSTERIOR_PATH`) | ✗ untracked |
| `models/sbi/continuous_hierarchical_nsf.pt` | `tdm_sbi.py:55` (`_DEFAULT_CONTINUOUS_POSTERIOR_PATH`) | ✓ tracked |

즉 **TDM/SBI의 두 핵심 production 모델은 fresh clone 시 존재하지 않는다.** 이는 단순한 "추적 정책 불일치"가 아니라 **공개 릴리스의 재현성 블로커**다. P0로 승격 필요.

`models/sbi/` 전체 19개 파일 중 git에 추적된 것은 5개 (`clozapine_posterior_nsf.pt`, `continuous_hierarchical_nsf{.pt,.aux.pt}`, `morphine_posterior.pt`, `morphine_posterior_nsf.pt`)에 불과하다.

### 4.2 `NotImplementedError` 3-way 분류 (개정)

초안은 `NotImplementedError`를 **통합해서 "미구현 API"**로 묶었으나, 실제 코드를 읽으면 세 사이트가 서로 다른 의미를 가진다.

| 사이트 | 의미 | 올바른 조치 |
|---|---|---|
| `graph/builder.py:191` `merge_overlay(...)` | 진짜 TODO (docstring만 있고 본문 없음) | public surface 결정: 필요하면 구현, 아니면 삭제 |
| `validation/split.py:29` `scaffold_split(...)` | **Dead code** — holdout은 이미 `data/reference/holdout.json`에 동결 (docstring §3.1에서도 "generated once ... and frozen"). 이 함수로 생성된 적 없음 | **삭제** (유지 비용 > 0, 가치 0) |
| `engine/rhs_jax.py:137-141, 143-145` | 방어적 backend-limit 신호 (주석: *"fail loudly here so experiments with backend='jax' do not silently zero hepatic clearance for OATP substrates"*) | **유지 + README Limitations에 JAX 제약 명시**. 구현 지시로 오독되지 않도록 분류 필요 |

초안에서 셋을 같은 bullet로 묶으면 "모두 구현 대기" 인상을 준다. 실제로는 1개만 구현 후보, 1개는 삭제, 1개는 의도된 가드.

### 4.3 문서와 실제 파일 사이에 drift가 있다

확인된 예:

- README의 프로젝트 구조에는 `src/sisyphus/ml/vdss_predictor.py`가 언급되지만 실제 파일 목록에는 없다.
- README의 프로젝트 구조에는 `src/sisyphus/sbi/hierarchical.py`가 언급되지만 실제 파일 목록에는 없다.
- `src/sisyphus/graph/presets.py`는 `reference_woman.yaml`을 기대하지만 `data/physiology/`에는 `reference_woman.yaml`이 없다.
- README와 내부 감사 문서의 테스트 수치가 서로 다른 시점의 값을 담고 있다.

문서가 매우 풍부한 만큼, 오래된 정보가 섞일 가능성도 같이 커졌다.

### 4.4 CI가 완전한 품질 게이트는 아니다

`.github/workflows/ci.yml`은 존재하지만:

- Ruff가 `--exit-zero`로 non-gating이다.
- CI 범위는 unit tests + 일부 integration + benchmark smoke 중심이다.
- torch/sbi 계열은 기본 lockfile/CI 경로에서 제외되어 있다.

즉 core path는 어느 정도 보호되지만, 전체 연구 기능이 매번 검증되는 구조는 아니다.

### 4.6 버저닝 / 릴리스 태깅 인프라 부재 (신설)

- `pyproject.toml:7` `version = "0.1.0"` — 선언은 있으나 변경 이력 파일 없음.
- `CHANGELOG.md` **부재** (`ls CHANGELOG*` → not found).
- `git tag --list` 기준 릴리스 태그 정책 없음 (audit 초안 누락).
- 결과: 릴리스 시점 간 변경이 사용자에게 노출되는 경로가 git log뿐. 하드닝 H1-H5 같은 큰 인프라 변화가 사용자 관점에서 불투명.

릴리스 준비도 90% 이상을 목표로 한다면 필수:

1. `CHANGELOG.md` 도입 (Keep a Changelog 포맷 권장).
2. 태깅 정책: 최소한 `v0.1.0` 태그 + 이후 major/minor 기준 문서화.
3. README에 현재 버전 배지 또는 명시.

### 4.5 대형 로컬 artifact와 ignore 정책이 혼재되어 있다

확인된 대형/로컬 자산:

- `full database.xml`: 약 1.8GB, git 추적 안 됨.
- `data/drugbank/`: 약 6.5MB, git 추적 안 됨.
- `models/sbi/` 일부 핵심 모델은 추적되어 있고, 일부 최신 모델은 untracked.
- `data/sbi/` 일부 학습 데이터는 추적되어 있고, 일부는 untracked.

이는 의도된 정책일 수 있지만, 어떤 artifact가 "필수", "재생성 가능", "라이선스상 제외", "현재 실험 산출물"인지 더 명확히 구분하면 좋다.

## 5. 세부 평가

| 영역 | 평가 | 코멘트 |
|---|---:|---|
| 코드 구조 | 높음 | 레이어 분리가 좋고 패키지 형태가 확립되어 있다. |
| 테스트 커버리지 | 중상-높음 | 테스트 자산은 많지만, skip/xfail/선택 dependency 영역이 있다. |
| 문서화 | 높음 | 매우 풍부하지만 최신성 drift 관리가 필요하다. |
| 재현성 | 중상 | lockfile과 CI가 있으나 artifact 정책과 optional feature 재현성이 남아 있다. |
| 릴리스 준비도 | 중간 | dirty worktree, untracked 모델/데이터, 문서 drift가 걸림돌이다. |
| 연구 투명성 | 높음 | 실패 기록과 limitation이 잘 남아 있다. |
| 외부 사용자 경험 | 중간 | Quickstart는 있으나 필수 모델/데이터와 optional dependency 경계가 더 선명해야 한다. |

## 6. 우선 정리 과제

### P0: 상태 고정

1. **production SBI 모델 배포 전략을 즉시 정한다.**
   - `multi_drug_nsf.pt`, `hierarchical_nsf_2k.pt` 는 `tdm_sbi.py` 기본 경로인데 untracked.
   - 선택지: (a) Git LFS로 tracked 전환, (b) Release artifact로 분리 + 해시 검증 다운로드 스크립트.
   - (초안 option (c) "재학습 스크립트 + 에러 메시지 개선"은 **재현성 블로커를 해결하지 못한다**. SBI 학습은 확률적 (NSF stochastic optimization) → 같은 데이터 + 같은 seed라도 PyTorch/CUDA 비결정성으로 posterior 미소 차이 발생. 배포된 posterior와 bit-wise 일치 불가 ⇒ (a)/(b) 중 택1.)
2. **tracked modified 파일 처리.** `data/validation/prospective_batch_N5_kinase.json` (11줄 변경)이 의도된 갱신인지 결정 — 의도면 커밋, 아니면 `git restore`.
3. 나머지 untracked artifact 분류:
   - 추적해야 하는 것
   - release artifact로 별도 배포해야 하는 것
   - 재생성 가능해서 ignore할 것 (`.claude/` 포함)
   - 로컬 실험 산출물로 보관만 할 것
4. `.gitignore`와 문서의 artifact 정책을 맞춘다.

### P1: 문서-실제 상태 동기화

1. README의 프로젝트 구조에서 실제 없는 파일명을 정리한다.
2. 테스트 수치와 validation 수치의 기준 날짜를 명확히 한다.
3. `reference_woman` preset의 방향을 결정한다.
   - 실제 YAML 추가
   - preset 제거
   - "planned/unsupported"로 문서화

### P1: 버저닝/릴리스 인프라 (신설)

1. `CHANGELOG.md` 초기화 — H1-H5 하드닝 머지 + P4.5 Achour 머지 + N50 FROZEN을 0.1.0 엔트리로 요약.
2. `v0.1.0` 태그 부여 (현재 main HEAD).
3. 태깅 정책 문서화 (major/minor 기준, release note 템플릿 위치).

### P1: 미구현 API 처리 (개정 — §4.2 재분류 반영)

1. `merge_overlay`: public surface인지 결정. 필요하면 구현, 아니면 삭제.
2. `scaffold_split`: **dead code로 확인 → 삭제**. holdout은 이미 frozen JSON이라 이 함수 경로가 실행된 적 없음.
3. JAX ECM (`rhs_jax.py:137`): **구현 대상 아님 — 유지**. README/Limitations 섹션에 "JAX backend는 ECM extended clearance 미지원, scipy가 default"만 명시.

### P2: CI 강화 — 최근 머지 반영 후 남은 항목

2026-04-23 하드닝 PR로 이미 머지된 항목 (감사 초안과 중복되어 제거):

| 항목 | 커밋 | 상태 |
|---|---|---|
| CI workflow (H5) | `2ed7b12` (#3) | ✓ |
| Reproducible lockfile (H1) | `2ed7b12` (#3) | ✓ |
| `pi_coverage_90` via `--compute-pi` (H3) | `824f8f7` (#4) | ✓ |
| Model manifest + feature-schema hash (H2) | `0d0a3a9` (#5) | ✓ |
| ECM unit audit doc (H4) | `9ef2072` (#6) | ✓ |

이 중 H1-H5는 메모리의 `project_hardening_backlog.md` 5개 항목에 정확히 대응하며, 모두 완료됐다. 실제로 남은 P2 과제는:

1. **Ruff를 gating으로 승격.** 현재 `ci.yml:28`이 `--exit-zero`라 경고만 발생. 코드베이스 규모(src 14k LOC)에서 스타일 드리프트 위험이 커짐.
2. **optional feature별 CI matrix 분리.**
   - core (현재 유일한 게이트 경로)
   - chem/ml (RDKit + XGBoost)
   - sbi/torch (torch + sbi) — 현재 lockfile/CI에서 제외돼 있음
3. **full benchmark vs smoke benchmark 경계 문서화.** H3로 `--compute-pi` 훅은 생겼지만 언제 full을 돌려야 하는지 기준은 없음.
4. **(다운그레이드 → P3)** lockfile ↔ pyproject.toml 일치성 CI 체크. 초안 P2로 제시했으나 "일치성"의 정의 자체가 애매 (lockfile은 정당하게 `>=`을 구체 pin함). 실질 효용 낮음 → 우선순위 하락.
5. **CI fail-closed 게이트 구체화 (추가).** 역사적 회귀 사고 (holdout leakage 2026-04-04, propranolol Cmax drift)에 대응하는 구체 guard:
   - holdout-leakage linter: ML training 스크립트가 `data/reference/holdout.json` 드럭을 참조하면 실패
   - model artifact sha256 check: `models/adme/*.json`, `models/direct_pk/*.json`의 기준 해시 목록 비교 (feature_schema hash와 분리)
   - PI coverage regression: H3 `pi_coverage_90`이 기준 band 이탈 시 실패 (single-number gate)

### P2: 릴리스 패키징

1. 모델 artifact 배포 전략을 정한다.
2. DrugBank/raw XML 같은 license-sensitive data의 설치/생성 절차를 더 명확히 한다.
3. fresh checkout에서 가능한 최소 기능과 full 기능을 나눠 검증한다.

## 7. 최종 판단

Sisyphus는 현재 "작동하는 연구 플랫폼"으로는 상당히 높은 완성도에 있다. 핵심 엔진, 예측 파이프라인, TDM/SBI 실험, validation artifact, 문서가 모두 존재한다. 그러나 "누군가 새로 clone해서 같은 결과를 안정적으로 재현하고 확장하는 공개 릴리스" 기준에서는 아직 정리할 부분이 남아 있다.

### 7.1 상한은 두 개의 **독립적** 축이다 (개정)

초안은 "재현성이 해결돼도 AAFE 5.249가 풀리지 않으면 릴리스 품질은 [75%] 상한을 넘지 못한다"라고 썼다. 이는 **두 개의 직교 축을 하나의 상한으로 혼합한 것**이다. 개정판은 다음과 같이 분리한다.

#### 축 A — 과학적 정확도 상한

- **N50 2026Q2 FROZEN (2026-04-23, commit b366035): AAFE 5.249 [3.79-7.77].** 메인 N=107 holdout의 Meta AAFE 2.695와 벌어짐 — secondary holdout에서 generalization 약화 시그널.
- **DE-33 미해결.** OATP1B1 ECM V3 underpredict (FE 0.39/0.48× Mode C). TransPortal Km audit은 DE-33 수정 후보에서 제외됨 (flat-CLuptake가 Km 변경을 no-op으로 만듦).
- **P4.5a (SBI 재학습 + SBC)** 큐에만 있음.

→ 이 축의 개선은 **모델 품질 작업** (track 개발, DE-33 연구). 릴리스를 **막지 않는다** — README/Limitations에 정직하게 기록하면 충분.

#### 축 B — 릴리스 준비도 상한

- production SBI 모델 untracked (§4.1)
- 문서 drift (§4.3)
- CHANGELOG / tag 정책 부재 (§4.6 신설)
- CI 커버리지 홀 (§4.4 + §6 P2-5)

→ 이 축의 개선은 **인프라·문서 작업**. AAFE와 무관하게 100%까지 가능.

**결론**: "릴리스 블로커"는 **축 B**의 4개 P0/P1. AAFE 5.249은 "릴리스 안 됨"의 근거가 아니라, "정직한 릴리스 노트" 작성의 근거다. 독자가 초안을 "DE-33 해결 전엔 릴리스 불가"로 오독할 소지가 있어 명시 분리.

### 7.2 우선순위

기능 추가보다 다음이 먼저 — 축 B 4건 + 축 A는 **평행 트랙**:

**축 B (릴리스 준비도, 차단 제거용)**:
1. production SBI 모델 (`multi_drug_nsf.pt`, `hierarchical_nsf_2k.pt`) 배포 전략 결정 — 릴리스 블로커
2. git 상태와 artifact 정책 정리 (tracked modified + untracked 분류, `.claude/` gitignore 포함)
3. README/문서 drift 제거 (`vdss_predictor.py`, `hierarchical.py`, `reference_woman.yaml`)
4. 미구현 API 3-way 분류 반영 (`scaffold_split` 삭제, JAX ECM Limitations 명시)

**축 A (정확도, 평행 진행 가능)**:
- P4.5a SBI 재학습 + SBC → DE-33 재검증

축 B 4건이 정리되면 **릴리스 준비도는 90%+**. 축 A는 별개 사이클.

## 8. 감사 중 확인한 제한

- 테스트는 실행하지 않았다. 사용자의 이전 요청이 "수정/편집 금지"였고, pytest 실행은 캐시나 pyc를 갱신할 수 있어 읽기 중심 감사로 제한했다.
- 인터넷 검색은 사용하지 않았다. 현재 디렉토리의 완성도 평가는 로컬 파일과 git 상태만으로 판단했다.
- 이 문서 작성 자체는 사용자의 후속 요청에 따라 새 Markdown 파일을 추가한 것이다.

## 9. 개정 이력

- **2026-04-24 (최초판):** 초기 감사 문서 작성.
- **2026-04-24 (하드닝 개정):** 초안의 사실관계를 교차 검증하여 다음을 반영:
  - 테스트 수치 정정 (222 → 약 635, 59 → 63)
  - §4.1에 production SBI 모델(`multi_drug_nsf.pt`, `hierarchical_nsf_2k.pt`) untracked = 재현성 블로커 교차검증 추가
  - §4.1에 tracked modified 파일(`prospective_batch_N5_kinase.json`)의 실제 diff 내용 반영
  - §6 P0에 위 두 블로커 명시적 과제화
  - §6 P2에서 이미 머지된 H1-H5 항목 (커밋 `2ed7b12`, `824f8f7`, `0d0a3a9`, `9ef2072`) 제거하고 실제 남은 과제만 유지
  - §7에 N50 2026Q2 AAFE 5.249 및 DE-33 미해결이 릴리스 점수 상한에 미치는 영향 명시
- **2026-04-24 (counter-audit patch — Option α):** 본 감사를 재감사한 결과 반영 (5건 정정, 1건 신설):
  - §4.2 `NotImplementedError` 3-way 재분류 (TODO vs dead code vs 방어적 가드 구분)
  - §6 P0-1 option (c) "재학습 스크립트" 부적절 판정 추가 (SBI 학습 확률성 ⇒ 재현성 해결 불가)
  - §6 P1 `scaffold_split` → "삭제" 명시 / JAX ECM → "유지 + 문서화" 명시
  - §6 P2-4 lockfile 일치성 체크 P3로 다운그레이드
  - §6 P2-5 CI fail-closed 게이트 구체화 신설 (leakage linter, model sha256, PI coverage regression)
  - §7 릴리스 상한 논리 분리 (축 A 과학 정확도 ⊥ 축 B 릴리스 준비도; AAFE 5.249가 릴리스 차단 근거 아님 명시)
  - §4.6 / §6 P1 버저닝/태깅 인프라 신설 (CHANGELOG 부재, tag 정책 부재)
  - **커밋 시 self-tracked**: 초안/개정판은 untracked였으나 Option α 반영 시점에 git add로 tracked 전환 (§8 한계 해소)
  - **`.gitignore`에 `.claude/` 추가** (§6 P0-3 "artifact 분류" 중 `.claude/` 부분 즉시 처리)
