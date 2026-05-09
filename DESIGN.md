# Sisyphus — Design Specification

**Version:** 0.2.0-draft
**Author:** Jae Min Yoon
**Date (original):** 2026-03-22
**Status:** **Architectural rationale only — superseded by `README.md` §Methodology for current implementation details.**
**Language policy:** 본문 한국어, 코드·타입·용어 영어
**Predecessor:** [Omega PBPK](https://github.com/jam-sudo/Omega) — 35-state ODE 기반 whole-body PBPK. Sisyphus의 설계 결정 상당수는 Omega 개발 과정에서 확인된 구조적 한계에 기반한다. Omega repo의 `CLAUDE.md`에 31개 settled decisions이 기록되어 있다.

> **Note (2026-05-08):** This document captures the original design rationale (Phase 0–1 era). Substantial architectural additions have shipped since the original date and are *not* reflected here:
>
> - Extended Clearance Model (ECM) with closed-form QSSA hepatocyte kinetics + auto-activation gating (PR #9, #20, #22, #29, #30)
> - Prodrug activation v1 → v2 → v3 (PR #6, #7, #15, #34) — `ProdrugActivation` flux + registry
> - NAT2/UGT1A1 phenotype propagation + back-solve cancellation fix (PR #32, v0.3.2)
> - `phenotype_scale_overrides` API hook (PR #33, v0.3.3)
> - 4-track meta-learner with VDss orthogonal track (`_W_VDSS=0.20`)
> - Method dispatch SBI / IBIS / IS for TDM (P6, 2026-04-19)
> - Mean-only deterministic realization (Hardening, 2026-05-01)
>
> For the **current** mathematical model, ODE forms, prediction pipeline, applicability-domain rules, and validation status, refer to `README.md` §Methodology and §Validation. For **architectural rationale and invariants** (graph-as-body, distribution-native, identity-blind engine), this document remains canonical. For **chronological experiment history**, see `docs/claude/experiment-log.md`.

---

## 1. 목적

Sisyphus는 인체의 약물 거동을 시뮬레이션하는 computational 플랫폼이다.

인체를 typed directed multi-graph로 표현하고, 이 graph 위에서 물질의 이동·변환·제거를 시뮬레이션한다. 모든 파라미터와 예측은 distribution이며, uncertainty가 시스템 전체에 걸쳐 전파된다.

### 1.1 Scope 정의

**v1 scope (이 문서):** 경구 투여된 small molecule의 pharmacokinetics — Cmax, AUC, t½ 예측. SMILES + dose → PK endpoints (with uncertainty).

**향후 scope:** v1의 graph architecture가 PK/PD, disease physiology, multi-drug DDI, population simulation으로 확장 가능해야 한다. 이 확장이 engine 수정 없이 가능한 것은 v0.3에서 실증한다 (§11.3).

### 1.2 Scope 밖

- Discrete event simulation (action potential, stochastic gene expression)은 ODE graph의 범위 밖이다. 이런 system은 별도 simulation engine이 필요하며, 이 문서에서 다루지 않는다.
- Biologics (antibody, ADC) PK는 v1 scope 밖이다. FcRn-mediated recycling 등의 mechanism은 v1 graph로 표현 가능하지만, 검증 데이터가 없으므로 포함하지 않는다.
- GUI / web frontend는 v1 범위 밖이다.

---

## 2. 핵심 아이디어

### 2.1 인체는 graph다

인체의 continuous transport system은 directed graph로 표현할 수 있다.

```
혈류:   heart → artery → [organs] → vein → heart
림프:   [tissues] → [lymph_nodes] → thoracic_duct → blood
담즙:   liver → bile_duct → duodenum → (enterohepatic recirculation)
```

이것들은 **같은 node set (organs/tissues) 위에 다른 edge types이 overlay된 multi-graph**다.

**적용 범위:** 이 graph 모델은 연속적 물질 이동(blood flow, diffusion, active transport, secretion)에 적합하다. 이산적 사건(neural firing, immune cell migration)은 ODE로 자연스럽게 표현되지 않으므로 별도 메커니즘이 필요하다. v1에서는 혈류 + GI transit + 담즙 순환까지만 다룬다.

기존 PBPK 도구들(PK-Sim, Simcyp, Omega)은 ODE state vector에 organ을 하드코딩한다. 새 compartment를 추가하려면 state vector를 확장하고, ODE indexing을 수정하고, flow fraction을 재계산해야 한다.

Graph 표현에서는 YAML에 node/edge를 추가하면 된다. ODE는 graph topology에서 **자동으로 유도**된다. Engine은 graph를 받아서 ODE를 생성하고 풀 뿐, 특정 organ이나 substance의 identity를 알지 못한다.

### 2.2 모든 값은 distribution이다

Point estimate 기반 시스템의 근본 한계: 입력 하나가 2배 틀리면 출력이 4배 틀릴 수 있고, 어디서 얼마나 틀렸는지 알 수 없다.

Sisyphus에서는 모든 physiological parameter와 drug property가 distribution이다.

```
fup ~ LogNormal(μ=-2.3, σ=0.4)
CYP3A4_abundance ~ LogNormal(μ=4.68, σ=0.40)  # inter-individual variability CV=40%
cardiac_output ~ Normal(μ=390, σ=39)            # L/h
```

이 distribution이 graph를 통해 전파되면 최종 예측도 distribution이 된다.

```
Cmax ~ LogNormal(μ=1.16, σ=0.52)
→ median: 3.2 mg/L
→ 90% PI: [1.1, 9.3] mg/L
→ P(Cmax > 10 mg/L): 6.2%
```

이것의 의미:
- **Population prediction**: distribution의 median.
- **Individual prediction**: 환자의 genotype/체중/신기능으로 conditioning → PI가 좁아짐.
- **Risk assessment**: P(Cmax > toxicity_threshold)를 직접 계산.

### 2.3 Engine은 identity를 모른다

Engine은 node와 edge의 **type**은 알지만 **identity**는 모른다.

Engine이 아는 것:
- 이 node는 `organ` type이고, 이 enzyme slot에 abundance가 있다.
- 이 edge는 `blood_flow` type이고, flow_rate가 있다.
- 이 edge는 `clearance` type이고, 이 node의 enzyme abundance × drug의 enzyme affinity로 CLint를 계산한다.

Engine이 모르는 것:
- 이 node가 "liver"라는 것.
- 이 enzyme이 "CYP3A4"라는 것.
- 이 drug이 "midazolam"이라는 것.

이 구분이 중요한 이유: engine에 identity-specific 분기(`if organ == "liver"`)가 없으면, **새 organ이나 enzyme을 추가할 때 engine을 수정할 필요가 없다.** 모든 specificity는 graph YAML과 drug parameterization에 있다.

---

## 3. 아키텍처

### 3.1 데이터 흐름

```
SMILES + dose
    │
    ▼
┌─────────┐     ┌─────────────┐
│ predict │────▶│ DrugOnGraph │
└─────────┘     └──────┬──────┘
                       │
    ┌──────────────────┤
    │                  │
    ▼                  ▼
┌────────┐      ┌──────────┐
│   ml   │      │  engine  │◀── BodyGraph (from YAML)
│ (direct│      │ (compile │
│  Cmax) │      │  + solve)│
└───┬────┘      └────┬─────┘
    │                │
    │    ┌───────────┘
    │    ▼
    │  ┌────┐
    │  │ pk │ (Cmax, AUC, t½)
    │  └──┬─┘
    │     │
    ▼     ▼
┌────────────┐
│  pipeline  │ (meta-learner: engine result + ML result → final)
└─────┬──────┘
      │
      ▼
PredictionResult (with uncertainty)
```

`predict`가 `DrugOnGraph`를 생성하고, 이것이 `engine`에 주입된다. `ml`은 `predict`의 출력을 직접 feature로 받아서 independent Cmax 예측을 수행한다. `pipeline`이 engine 결과와 ML 결과를 조합한다.

### 3.2 Layer 정의

| Layer | 책임 | 의존 대상 | 의존하지 않는 것 |
|-------|------|----------|----------------|
| `graph` | BodyGraph 정의, node/edge types | 없음 | |
| `engine` | Graph → ODE 유도, solver, MC propagation | `graph` | `predict`, `ml` |
| `predict` | SMILES → MolecularProfile → ADMEProperties → DrugOnGraph | reference data (DrugBank CSV, optional) | `engine`, `ml` |
| `ml` | Direct PK prediction, ensemble | 없음 | `engine`, `predict` |
| `pk` | SimResult → PKEndpoints | 없음 | |
| `validation` | Reference data, benchmark, metrics | 없음 | |
| `pipeline` | 전체 조합 | `predict`, `engine`, `ml`, `pk` | |

### 3.3 디렉토리 구조

```
sisyphus/
├── pyproject.toml
├── README.md
├── DESIGN.md
│
├── src/sisyphus/
│   ├── graph/                # BodyGraph: node, edge, topology
│   │   ├── body.py           # BodyGraph class
│   │   ├── types.py          # Node/Edge type definitions
│   │   ├── builder.py        # YAML → BodyGraph constructor
│   │   └── presets.py        # reference_man(), reference_woman()
│   │
│   ├── engine/               # Graph → ODE → solution
│   │   ├── compiler.py       # BodyGraph + DrugOnGraph → ODE system
│   │   ├── flux.py           # Flux functions + registry
│   │   ├── solver.py         # ODE solver wrapper
│   │   ├── uncertainty.py    # MC propagation (compile once, parameterize many)
│   │   └── result.py         # SimResult
│   │
│   ├── predict/              # SMILES → drug properties
│   │   ├── chemistry.py      # Descriptors, pKa, logP, AD check
│   │   ├── adme.py           # fup, CLint, Peff, VDss, RBP
│   │   └── ivive.py          # ADME → DrugOnGraph parameter translation
│   │
│   ├── ml/                   # Data-driven PK prediction
│   │   ├── features.py       # Feature engineering
│   │   ├── models.py         # XGBoost/LightGBM wrappers
│   │   ├── ensemble.py       # Ensemble + meta-learner
│   │   └── registry.py       # Model versioning
│   │
│   ├── pk/                   # PK endpoint computation
│   │   ├── endpoints.py      # Cmax, AUC, t½
│   │   ├── nca.py            # Non-compartmental analysis
│   │   └── analytical.py     # 1-cpt, 2-cpt closed-form
│   │
│   ├── validation/           # Reference, benchmark, metrics
│   │   ├── reference.py      # Single source of truth
│   │   ├── benchmark.py      # Holdout benchmark runner
│   │   ├── metrics.py        # AAFE, fold error, coverage
│   │   └── split.py          # Scaffold/temporal split
│   │
│   ├── pipeline/             # Thin orchestrator
│   │   ├── predict.py        # SMILES → PredictionResult
│   │   └── config.py         # Configuration
│   │
│   └── cli.py
│
├── data/
│   ├── physiology/           # BodyGraph YAML definitions
│   ├── compounds/            # Manually curated drug configs (validation용)
│   ├── reference/            # clinical_pk.json, holdout.json
│   └── training/             # TDC, MMPK
│
├── models/                   # Trained artifacts (git-ignored)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmark/
├── scripts/
└── docs/
```

Source files: ~30개. 디렉토리당 상한: 20개.

---

## 4. BodyGraph

### 4.1 정의

BodyGraph는 typed directed multi-graph다.

```python
class BodyGraph:
    nodes: dict[str, Node]
    edges: list[Edge]
    global_params: dict[str, Distribution]  # cardiac_output 등 graph-level parameters
    
    def add_node(self, node: Node) -> None: ...
    def add_edge(self, edge: Edge) -> None: ...
    def remove_node(self, name: str) -> None: ...
    def validate(self) -> list[str]:
        """Flow conservation, mass balance 등 물리적 제약 검증.
        위반 사항을 문자열 목록으로 반환. 빈 목록 = valid."""
```

### 4.2 Node Types

```python
@dataclass
class Node:
    name: str
    node_type: str                     # "organ", "barrier_organ", "blood_pool", "lumen", "sink"
    volume: Distribution               # L
    composition: TissueComposition | None  # fn, fp, fw, pH (Kp 계산용)
    enzymes: dict[str, Distribution]   # enzyme tag → abundance distribution
    transporters: dict[str, Distribution]

@dataclass
class TissueComposition:
    fn: float    # neutral lipid fraction
    fp: float    # phospholipid fraction
    fw: float    # water fraction
    pH: float    # intracellular pH
```

`enzymes`의 key는 arbitrary string tag다. "CYP3A4"일 수도 있고 "enz_001"일 수도 있다. Engine은 이 string을 해석하지 않는다 — drug의 enzyme affinity와 매칭할 때만 사용한다.

### 4.3 Edge Types

```python
@dataclass
class Edge:
    source: str
    target: str
    edge_type: str

@dataclass
class FlowEdge(Edge):
    """Convective transport (blood flow, lymphatic flow)."""
    edge_type: str = "flow"
    flow_rate: Distribution            # L/h (absolute, not fraction)

@dataclass
class DiffusionEdge(Edge):
    """Permeability-surface area limited transfer."""
    edge_type: str = "diffusion"
    ps_product: Distribution           # L/h

@dataclass
class TransitEdge(Edge):
    """First-order transit (GI lumen segments)."""
    edge_type: str = "transit"
    transit_rate: Distribution         # h⁻¹

@dataclass
class AbsorptionEdge(Edge):
    """Drug absorption (lumen → tissue). Rate depends on drug properties."""
    edge_type: str = "absorption"

@dataclass
class ClearanceEdge(Edge):
    """Elimination. Rate = f(node.enzymes, drug.enzyme_affinity, drug.fup)."""
    edge_type: str = "clearance"
    model: str                         # "well_stirred", "gfr_filtration"
```

### 4.4 Flow Specification

Edge의 `flow_rate`는 **absolute (L/h)**로 지정한다. YAML에서는 편의상 `flow_fraction`으로 적되, builder가 `cardiac_output × flow_fraction`으로 변환한다.

```yaml
global_params:
  cardiac_output: { mean: 390, cv: 0.10, unit: L/h }

edges:
  - source: arterial_blood
    target: brain
    type: flow
    flow_fraction: 0.12       # builder가 390 × 0.12 = 46.8 L/h로 변환
```

**Flow conservation:** builder는 생성 후 `validate()`를 호출한다. Non-lung node에서 inflow ≠ outflow이면 validation error. 이 검증이 YAML 작성 시점에 오류를 잡는다 — 실행 시점이 아니라.

### 4.5 Node 추가 시 flow 재분배

새 node를 추가하면 (e.g., tumor), cardiac output의 일부가 새 node로 간다. **기존 node의 flow_fraction을 수정해야 한다.** 이건 engine 수정이 아니라 YAML 수정이다.

```yaml
# tumor_overlay.yaml — base reference_man.yaml에 merge
nodes:
  tumor:
    type: organ
    volume: { mean: 0.05, cv: 0.50 }

edges:
  - { source: arterial_blood, target: tumor, type: flow, flow_fraction: 0.005 }
  - { source: tumor, target: venous_blood, type: flow }

# 기존 'rest' node의 flow_fraction을 0.069 → 0.064로 조정
overrides:
  edges:
    - { source: arterial_blood, target: rest, flow_fraction: 0.064 }
```

Builder는 base YAML + overlay YAML을 merge한 후 flow conservation을 검증한다.

---

## 5. DrugOnGraph — Drug-Graph 매핑

`DrugOnGraph`는 drug의 properties를 graph의 node/edge에 **매핑하는 선언**이다. Organ-specific field를 두지 않고, enzyme-level / property-level로 정의한다.

```python
@dataclass(frozen=True)
class DrugOnGraph:
    name: str
    smiles: str
    dose_mg: float
    route: str                          # "oral", "iv"
    administration_node: str            # "stomach_lumen" (oral), "venous_blood" (iv)
    
    # Physicochemical
    mw: float
    pka: float | None
    compound_type: str                  # "neutral", "acid", "base", "zwitterion"
    
    # Binding & partitioning
    fup: Distribution                   # fraction unbound in plasma
    rbp: Distribution                   # blood:plasma ratio
    kp_method: str                      # "rodgers_rowland", "berezhkovskiy", "provided"
    kp_overrides: dict[str, Distribution]  # node name → Kp (kp_method="provided"일 때)
    
    # Absorption
    peff: Distribution                  # effective permeability (×10⁻⁴ cm/s)
    solubility: Distribution            # mg/mL
    
    # Metabolism — enzyme-level, NOT organ-level
    enzyme_affinity: dict[str, Distribution]
    # enzyme tag → CLint per unit enzyme (µL/min/pmol)
    # e.g., {"CYP3A4": Distribution(mean=0.9), "CYP2D6": Distribution(mean=0.1)}
    # Engine이 각 node의 enzyme abundance와 곱해서 organ-level CLint를 자동 계산
    
    # Renal
    renal_clearance: Distribution       # L/h, total plasma basis (GFR × fup + secretion)
    
    def sample(self, rng) -> "DrugOnGraph":
        """모든 Distribution을 point value로 sample."""
        ...
```

**핵심 결정:** `clint_hepatic_L_per_h`가 없다. 대신 `enzyme_affinity`가 있다. 간의 CYP3A4 abundance는 간 node의 `enzymes["CYP3A4"]`에, 장벽의 CYP3A4 abundance는 장벽 node의 `enzymes["CYP3A4"]`에 있다. Engine의 `ClearanceFlux`가 `node.enzymes[tag] × drug.enzyme_affinity[tag]`를 합산하여 organ-level CLint를 계산한다. **Organ identity 없이 IVIVE가 수행된다.**

---

## 6. Engine

### 6.1 ODE Compiler — Compile Once, Parameterize Many

```python
class ODECompiler:
    def compile(self, graph: BodyGraph) -> CompiledODE:
        """Graph topology를 분석하여 ODE skeleton을 생성.
        
        이 단계에서 state indexing, edge → flux 매핑, 
        RHS function의 구조가 확정된다. Drug parameter나
        distribution sample에 의존하지 않는다.
        """
        states = self._assign_states(graph)
        flux_specs = []
        for edge in graph.edges:
            flux_cls = FLUX_REGISTRY[edge.edge_type]
            flux_specs.append(flux_cls.from_edge(edge, states))
        return CompiledODE(states=states, flux_specs=flux_specs)


class CompiledODE:
    def make_rhs(self, params: ResolvedParams) -> Callable:
        """Compiled skeleton + resolved parameters → RHS function.
        
        params에는 sampled (point) flow rates, volumes, Kp,
        enzyme abundances, drug properties가 들어있다.
        Graph topology는 compile 시 확정되었으므로 여기서 변하지 않는다.
        """
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            dydt = np.zeros(self.n_states)
            for spec in self.flux_specs:
                spec.apply(t, y, dydt, params)
            return dydt
        return rhs
```

**MC에서의 활용:** `compile()`은 1회. `make_rhs()`가 sample마다 다른 `params`로 호출. Graph traversal과 state indexing이 반복되지 않으므로 MC 1000 samples의 overhead는 ODE solve 1000회뿐.

### 6.2 Flux Registry

```python
FLUX_REGISTRY: dict[str, type[FluxSpec]] = {}

@register_flux("flow")
class FlowFluxSpec(FluxSpec):
    """Convective transport: dA_target/dt += Q × C_in — Q × C_out"""
    
    def apply(self, t, y, dydt, params):
        q = params.edge_param(self.edge_id, "flow_rate")
        c_source = y[self.source_idx] / params.node_param(self.source, "volume")
        
        kp = params.drug_kp(self.target)     # drug Kp for target node
        rbp = params.drug_param("rbp")
        v_target = params.node_param(self.target, "volume")
        c_out = y[self.target_idx] * rbp / (v_target * kp)
        
        dydt[self.source_idx] -= q * c_source
        dydt[self.target_idx] += q * c_source - q * c_out
        # venous return은 별도 edge

@register_flux("clearance")
class ClearanceFluxSpec(FluxSpec):
    """Enzyme-mediated elimination.
    
    CLint_organ = Σ(enzyme_abundance_i × drug_affinity_i) × IVIVE_scaling
    CLh = well_stirred(Q, fup, CLint_organ)   — or other model
    rate = CLh × C_in
    """
    
    def apply(self, t, y, dydt, params):
        # Engine이 organ identity를 모른 채 enzyme-level IVIVE 수행
        clint_organ = 0.0
        for tag, abundance in params.node_enzymes(self.source).items():
            affinity = params.drug_enzyme_affinity(tag)
            if affinity > 0 and abundance > 0:
                clint_organ += abundance * affinity * params.ivive_scaling
        
        if self.model == "well_stirred":
            fup = params.drug_param("fup")
            q = params.total_inflow(self.source)  # 이 node의 총 inflow
            clh = (q * fup * clint_organ) / max(q + fup * clint_organ, 1e-12)
            c_in = ...  # mixed input concentration
            rate = clh * c_in
        elif self.model == "gfr_filtration":
            rate = params.drug_param("renal_clearance") * c_plasma
        
        dydt[self.source_idx] -= rate
        dydt[self.sink_idx] += rate
```

`ClearanceFluxSpec`이 "CYP3A4"라는 string을 해석하지 않는다. `params.node_enzymes()`가 `{"CYP3A4": 108, "CYP2D6": 10}`을 반환하고, `params.drug_enzyme_affinity("CYP3A4")`가 0.9를 반환하면, 둘을 곱한다. 이게 liver든 gut이든 engine은 모른다 — node에 enzyme이 있으면 clearance가 발생할 뿐.

### 6.3 Uncertainty Propagation

```python
class UncertaintyEngine:
    def propagate(self, compiled: CompiledODE, graph: BodyGraph,
                  drug: DrugOnGraph, n_samples: int = 1000) -> UncertaintyResult:
        results = []
        for i in range(n_samples):
            rng = np.random.default_rng(seed=42 + i)
            realized_graph = graph.sample(rng)        # Distribution → point
            realized_drug = drug.sample(rng)
            params = ResolvedParams(realized_graph, realized_drug)
            
            rhs = compiled.make_rhs(params)            # compile은 재사용
            sol = self.solver.solve(rhs, y0, t_span)
            pk = compute_endpoints(sol)
            results.append(pk)
        
        return UncertaintyResult.from_samples(results)
```

---

## 7. Data Contracts (나머지)

### 7.1 SimResult (engine → pk)

```python
@dataclass(frozen=True)
class SimResult:
    time_h: np.ndarray
    concentrations: dict[str, np.ndarray]  # node name → mg/L time series
    amounts: dict[str, np.ndarray]         # node name → mg time series
    mass_balance_error: float              # max |total - dose| / dose
    solver_success: bool
```

### 7.2 PKEndpoints (pk → pipeline)

```python
@dataclass(frozen=True)
class PKEndpoints:
    cmax: Distribution
    tmax: Distribution
    auc_0t: Distribution
    auc_0inf: Distribution | None
    t_half: Distribution | None
    cl: Distribution | None
    vss: Distribution | None
```

MC 결과: N개 sample에서 집계한 distribution. 단일 simulation: cv=0인 deterministic distribution.

### 7.3 PredictionResult (pipeline → caller)

```python
@dataclass(frozen=True)
class PredictionResult:
    drug_name: str
    smiles: str
    dose_mg: float
    route: str
    pk: PKEndpoints
    method: str                     # "engine", "ml", "hybrid"
    engine_pk: PKEndpoints | None   # engine 단독 결과
    ml_pk: PKEndpoints | None       # ML 단독 결과
    confidence: str
    in_applicability_domain: bool
    ad_flags: list[str]
    warnings: list[str]
    cmax_90ci: tuple[float, float] | None
```

---

## 8. Validation

### 8.1 Single Reference

`data/reference/clinical_pk.json` 단일 파일. 각 drug에 `in_holdout`, `in_training` flag.

### 8.2 Holdout Protocol

Murcko scaffold stratified split, seed=42, 생성 시 확정. Holdout은 training/tuning에 사용 금지. Reference value 수정은 허용. 반복적 모델수정-재평가 금지.

### 8.3 Acceptance Criteria

| Metric | v0.1 | v0.2 | v1.0 |
|--------|------|------|------|
| Holdout Cmax AAFE | — | ≤ 2.5 | ≤ 1.7 |
| Holdout %2-fold | — | ≥ 40% | ≥ 60% |
| 90% PI coverage | — | ≥ 80% | ≥ 90% |
| Mass balance error | ≤ 1e-6 | ≤ 1e-6 | ≤ 1e-6 |
| 단일 prediction (deterministic) | — | ≤ 2s | ≤ 500ms |
| MC prediction (N=1000) | — | ≤ 60s | ≤ 15s |

MC latency 근거: ODE 1회 ~50ms (Omega 200ms의 4x 최적화 — graph precompile + LSODA warm-start) × 1000 = 50s + overhead ~10s. v1.0에서는 15s가 목표이며, vectorized parameter sampling + Numba JIT가 필요할 수 있다.

---

## 9. Error Handling

| Failure | 처리 |
|---------|------|
| Invalid SMILES | `ValueError`. 유일한 hard failure. |
| Graph validation 실패 (flow conservation 등) | `ValueError`. Graph 정의 오류. |
| ADME 예측 실패 | conservative defaults, `confidence="low"`, `warnings`에 기록. |
| AD 밖 | `in_applicability_domain=False`, `ad_flags`에 사유. 예측은 계속. |
| ODE solver 발산 | `solver_success=False`, analytical fallback, `warnings`. |
| Mass balance > 1e-4 | `warnings`에 기록, 신뢰도 하향. |
| Terminal t½ 추정 불가 | `t_half=None`. |

---

## 10. Invariants

1. **Engine은 node/edge의 identity를 모른다.** `if node.name == "liver"` 같은 코드가 `engine/` 내에 있으면 CI fail.
2. **하위 layer가 상위 layer를 import하지 않는다.**
3. **Source code에 하드코딩된 physiological 상수 없음.** Float literal에는 출처 주석 필수.
4. **Holdout drug은 training/tuning에 사용 금지.**
5. **Drug-specific 분기 금지.**
6. **디렉토리당 .py 파일 20개 이하.**
7. **Deployed model은 holdout metric 기록 필수.**
8. **Physiology YAML의 flow conservation은 builder가 자동 검증.** 위반 시 build 실패.

---

## 11. 구현 로드맵

### 11.1 v0.1 — Graph Engine (Week 1–3)

**목표:** reference_man.yaml → BodyGraph → ODE compiler → Cmax. Omega ODE와 ±5% 일치.

**범위:** `graph/`, `engine/`, `pk/endpoints.py`, `validation/`, physiology YAML, compound YAML 4개 (midazolam, warfarin, caffeine, propranolol).

**Acceptance:** 4개 약물의 YAML → SimResult → Cmax가 Omega ODE ±5%.

### 11.2 v0.2 — SMILES → PK + Uncertainty (Week 3–5)

**목표:** End-to-end SMILES → PK prediction. MC uncertainty. Holdout AAFE 측정.

**범위:** `predict/`, `ml/`, `pipeline/`, `cli.py`. XGBoost ensemble, meta-learner.

**Acceptance:** Holdout AAFE ≤ 2.5, 90% PI coverage ≥ 80%, deterministic ≤ 2s.

### 11.3 v0.3 — 확장성 실증 (Week 5–7)

**목표:** Graph architecture의 composability를 실증한다. 3가지 확장을 engine 수정 없이 수행.

| 확장 | 방법 | Engine 수정 |
|------|------|------------|
| SC injection route | depot node + first-order absorption edge 추가 | 없음 |
| Pediatric model | pediatric YAML (scaled volumes/flows, ontogeny enzyme abundance) | 없음 |
| Tumor compartment | tumor node + flow edges + overlay YAML | 없음 |

**Acceptance:** 3가지 확장 모두 `src/sisyphus/engine/` 내 diff 0줄. SC/pediatric에서 physiologically plausible한 PK 변화 재현.

**v0.3이 실패하면:** Graph architecture가 주장만큼 composable하지 않다는 뜻. 이 경우 실패 원인을 분석하고, engine에 최소한의 수정으로 해결 가능한지 검토한다. 해결 불가하면 아키텍처를 재검토한다.

### 11.4 v1.0 — Production (Week 7+)

**범위:** 성능 최적화, DDI (inhibitor node → enzyme activity modification), PK/PD link (effect node + Emax edge), API.

**Acceptance:** Holdout AAFE ≤ 1.7, %2-fold ≥ 60%, deterministic ≤ 500ms, MC N=1000 ≤ 15s.

---

## 12. CI

```yaml
lint:        ruff check + format
unit:        pytest tests/unit/ (< 30s)
integration: pytest tests/integration/ (< 2min)
invariants:
  - 디렉토리당 .py 파일 ≤ 20
  - 하위 → 상위 layer import 없음
  - engine/ 내에 node/edge identity string 없음 (grep 기반)
  - physiology YAML flow conservation (builder validate)
  - deployed model holdout_aafe ≠ null
benchmark:   pytest tests/benchmark/ (holdout AAFE gate, main merge 시)
```

---

## 부록 A: Omega와의 관계

Sisyphus는 [Omega PBPK](https://github.com/jam-sudo/Omega)의 후속 프로젝트다.

**가져오는 것 (데이터 자산):**
- Platinum reference (176 drugs clinical Cmax) → `data/reference/clinical_pk.json`
- Holdout split (Murcko, seed=42) → `data/reference/holdout.json`
- ICRP reference man physiology, R&R tissue composition → `data/physiology/`
- Kp estimation 수식 3종 → engine flux 계산에 활용
- MMPK clinical Cmax data → `data/training/`
- ADME reference (153 compounds) → `data/reference/`
- Compound YAML configs (22 drugs) → `data/compounds/`

**가져오지 않는 것 (아키텍처 차이):**
- 35-state hardcoded ODE → graph-compiled ODE
- Sequential chain (SMILES→ADME→IVIVE→ODE) → enzyme-level DrugOnGraph + graph IVIVE
- Organ-specific CLint fields → enzyme-level affinity
- Point estimate pipeline → distribution-native
- Post-hoc hybrid selector → meta-learner
- 1,600+ _pNNN files, 11K-line api/app.py, 3중 benchmark 체계 → 없음

**Omega settled decisions의 Sisyphus 적용:**
- "Data quality >> model improvements" → 유효. Reference 품질이 최우선.
- "XGBoost > MLP at 1K-4K scale" → 유효. ML layer에서 XGBoost 사용.
- "Hybrid Cmax selector 필수" → **무효**. Meta-learner로 대체.
- "Error cancellation은 structural" → **구조가 다르므로 재검증 필요.**

---

## 부록 B: 용어 정의

| 용어 | 정의 |
|------|------|
| BodyGraph | 인체를 표현하는 typed directed multi-graph. |
| Distribution | mean + cv + distribution type. 불확실성을 포함하는 파라미터 값. |
| DrugOnGraph | Drug properties를 graph에 매핑하는 선언. Organ-specific이 아닌 enzyme-level. |
| ODE Compiler | BodyGraph → ODE skeleton 생성. Topology만 분석, parameter에 독립. |
| Flux Registry | Edge type → flux 계산식 매핑. |
| ResolvedParams | Distribution이 sample되어 point value가 된 parameter set. Compiled ODE에 주입. |
| IVIVE | In Vitro to In Vivo Extrapolation. Engine의 ClearanceFlux에서 enzyme abundance × drug affinity로 수행. |
| Holdout | 모델 개발에 사용하지 않는 영구 평가용 데이터. |
| AAFE | Absolute Average Fold Error. 10^(mean(\|log10(pred/obs)\|)). |
| MC | Monte Carlo. Compile once, parameterize many로 prediction interval 생성. |
