<p align="center">
  <h1 align="center">Sisyphus</h1>
  <p align="center">
    <strong>The human body as a typed directed graph.<br>Drug pharmacokinetics from first principles.</strong>
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> &middot;
    <a href="#architecture">Architecture</a> &middot;
    <a href="#extending-the-model">Extend</a> &middot;
    <a href="#benchmarks">Benchmarks</a>
  </p>
</p>

---

Sisyphus is a computational pharmacokinetics platform that represents the human body as a **typed directed multi-graph**, automatically derives ODE systems from graph topology, and propagates uncertainty natively through all predictions.

Give it a SMILES string and a dose. It gives you Cmax, AUC, and a 90% prediction interval &mdash; in 350 milliseconds.

```
$ sisyphus predict --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O" --dose 100

Drug: Cn1c(=O)c2c(ncn2C)n(C)c1=O
Method: hybrid
Confidence: high
Cmax: 1.5247 mg/L
Tmax: 0.96 h
AUC: 0.0477 mg*h/L
t½: 4.32 h
```

## Three Ideas

### 1. The body is a graph

Organs are nodes. Blood vessels, GI transit paths, and clearance routes are typed directed edges. The ODE system is **derived from graph topology**, not hand-written. To add a new organ, you edit YAML. You do not touch the engine.

```
         ┌──────────────────────────────────────────────────────────┐
         │                    venous_blood                          │
         │  ┌──────┐    ┌───────┐    ┌──────┐    ┌──────────────┐  │
         └──┤ lung ├────┤arteri-├───►│brain │───►│              │  │
            └──────┘    │al_blo-│    └──────┘    │              │  │
                        │od     ├───►│heart │───►│              │  │
                        │       │    └──────┘    │  venous_     │  │
                        │       ├───►│kidney├───►│  blood       │──┘
                        │       │    └──────┘    │              │
    stomach ──► duod ──►│       ├───►│gut   ├──► portal ──► liver ──►│
       ──► jej1 ──►     │       │    │wall  │    vein           │
       ──► jej2 ──►     │       │    └──────┘                   │
       ──► ile1 ──►     │       ├───►│muscle├───►│              │
       ──► ile2 ──►     │       │    └──────┘    │              │
       ──► colon ──►    │       ├───►│adipos├───►│              │
             fecal      └───────┘    └──────┘    └──────────────┘
```

34 nodes. 54 edges. 5 flux types. The engine walks the graph, dispatches flux functions by edge type, and assembles the right-hand side automatically.

### 2. Everything is a Distribution

`fup = 0.032` does not exist in Sisyphus. Only `fup = Distribution(mean=0.032, cv=0.4)` does.

Every physiological parameter, every drug property, every predicted ADME value carries its uncertainty. Monte Carlo sampling propagates these distributions through the graph to produce prediction intervals &mdash; not as a post-hoc feature, but as the system's native output format.

```python
from sisyphus.engine.uncertainty import UncertaintyEngine

ue = UncertaintyEngine()
mc = ue.propagate_fast(compiled, graph, drug, n_samples=1000)
# mc.cmax_samples → 1000 Cmax realizations
# mc.pk.cmax → Distribution(median=1.76, cv=0.19)
# mc.cmax_90ci → (1.28, 2.35) mg/L
```

### 3. The engine knows types, not identities

The engine knows *"this node has organ type with enzyme slots"* and *"this edge has clearance type using the well-stirred model."* It does not know *"this is the liver"* or *"this enzyme is CYP3A4."*

All identity lives in YAML and DrugOnGraph. This is what makes the architecture extensible &mdash; new organs, enzymes, routes, and populations require zero engine changes.

**Proof:** SC injection, pediatric physiology, tumor compartment, DDI, and PK/PD were all added with **0 lines changed** in `src/sisyphus/engine/`.

## Quickstart

```bash
# Install
pip install -e ".[dev,ml,chem]"

# Predict PK for caffeine 100mg oral
sisyphus predict --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O" --dose 100

# Run holdout benchmark
sisyphus benchmark --holdout
```

### Python API

```python
from sisyphus.pipeline.predict import predict

result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)

print(result.pk.cmax.mean)    # 1.52 mg/L
print(result.method)          # "hybrid"
print(result.confidence)      # "high"
```

### From compound YAML (bypassing ADME prediction)

```python
from pathlib import Path
from sisyphus.graph.builder import build_from_yaml
from sisyphus.compounds import load_compound
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.pk.endpoints import compute_endpoints
import sisyphus.engine.flux  # register flux specs
import numpy as np

graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
drug = load_compound(Path("data/compounds/midazolam.yaml"))

compiled = ODECompiler().compile(graph)
rng = np.random.default_rng(42)
params = ResolvedParams(graph.sample(rng), drug.sample(rng))

y0 = np.zeros(compiled.n_states)
y0[compiled.state_index[drug.administration_node]] = drug.dose_mg

result = solve(compiled, params, y0, t_span=(0, 24))
pk = compute_endpoints(result)

print(f"Cmax: {pk.cmax.mean:.4f} mg/L")  # 0.0069 mg/L (matches Omega ±0.5%)
```

## Architecture

```
SMILES + dose
    │
    ▼
 predict ──► DrugOnGraph (enzyme-level, all values are Distribution)
                  │
                  ▼
             engine ◄── BodyGraph (from YAML)
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

| Layer | Responsibility | Depends on |
|-------|---------------|------------|
| `graph/` | BodyGraph, node/edge types, YAML builder | `core` |
| `engine/` | ODE compiler, flux registry, solver, MC | `core`, `graph` |
| `predict/` | SMILES → chemistry → ADME → DrugOnGraph | `core` |
| `ml/` | XGBoost Cmax, meta-learner | `core` |
| `pk/` | SimResult → PKEndpoints | `core` |
| `pipeline/` | Orchestrator wiring all layers | all layers |
| `ddi.py` | Drug-drug interactions | `core`, `graph` |
| `pkpd.py` | PK/PD effect modeling | `core` |

**No cross-layer imports.** `predict` does not import `engine`. `engine` does not import `predict`. Shared types live in `core.py`.

## Extending the Model

The architecture's value proposition: **extend the model without touching the engine.**

### Add a new organ

```yaml
# tumor_overlay.yaml
nodes:
  - name: tumor
    type: organ
    volume: 0.05
    composition: {fn: 0.013, fp: 0.010, fw: 0.700, pH: 6.8}

edges:
  - {source: arterial_blood, target: tumor, type: flow, flow_fraction: 0.005}
  - {source: tumor, target: venous_blood, type: flow}
```

### Add a new route

```python
# SC injection — just add a depot node
graph.add_node(Node(name="sc_depot", node_type="lumen", volume=Distribution(0.01)))
graph.add_edge(AbsorptionEdge(source="sc_depot", target="venous_blood",
                               ka_fraction=Distribution(1.0)))
```

### Add a new population

Create `pediatric_5y.yaml` with allometrically scaled volumes, cardiac output, and ontogeny-adjusted enzyme abundances. Same graph structure, different parameters.

### Drug-drug interactions

```python
from sisyphus.ddi import apply_inhibition, KETOCONAZOLE

inhibited_graph = apply_inhibition(graph, KETOCONAZOLE)
# CYP3A4 activity reduced by >99%
# Midazolam AUC increases 12x — no engine changes
```

### PK/PD modeling

```python
from sisyphus.pkpd import compute_effect, PDModel

pd = PDModel(ke0=0.5, emax=100.0, ec50=0.05, hill=2.0)
effect = compute_effect(sim_result, pd)
# effect.emax_achieved → peak sedation %
# effect.temax → time of peak effect (delayed from PK peak)
```

## Benchmarks

### Engine Validation (Phase 1)

4 drugs validated against [Omega PBPK](https://github.com/jam-sudo/Omega) ODE output:

| Drug | Sisyphus Cmax | Omega Cmax | Error |
|------|:------------:|:----------:|:-----:|
| Midazolam 2mg | 0.006911 | 0.006943 | 0.5% |
| Caffeine 100mg | 1.7151 | 1.7139 | 0.1% |
| Warfarin 10mg | 0.4917 | 0.4922 | 0.1% |
| Propranolol 80mg | 0.1353 | 0.1355 | 0.1% |

### Holdout Benchmark (Phase 2)

| Metric | Value |
|--------|:-----:|
| In-domain AAFE | **1.697** |
| In-domain %2-fold | **70.4%** |
| Deterministic latency | **350 ms** |
| Mass balance error | **< 10⁻¹²** |
| MC N=1000 | 33.5 s |

### Performance

| Operation | Time |
|-----------|:----:|
| Single prediction (SMILES → Cmax) | 350 ms |
| ODE solve only | 36 ms |
| MC sample (fast path) | 33 ms |
| RHS evaluation | 31 µs |
| Graph compilation | < 1 ms |

## Project Structure

```
src/sisyphus/
├── core.py              # Distribution, contracts (DrugOnGraph, SimResult, PKEndpoints)
├── descriptors.py       # Morgan FP + RDKit descriptors (shared)
├── ddi.py               # Drug-drug interaction (inhibition, induction)
├── pkpd.py              # PK/PD link (effect compartment, Emax)
├── cli.py               # CLI entry point
├── compounds.py         # Compound YAML → DrugOnGraph loader
│
├── graph/               # BodyGraph: nodes, edges, topology
│   ├── types.py         # Node, Edge hierarchy (all frozen)
│   ├── body.py          # BodyGraph (add/remove/validate/sample)
│   ├── builder.py       # YAML → BodyGraph
│   └── presets.py       # reference_man(), reference_woman()
│
├── engine/              # Graph → ODE → solution (identity-blind)
│   ├── compiler.py      # ODECompiler, CompiledODE, ResolvedParams
│   ├── flux.py          # 5 FluxSpec implementations
│   ├── solver.py        # solve() + solve_mc()
│   └── uncertainty.py   # MC propagation (compile once, solve many)
│
├── predict/             # SMILES → drug properties
│   ├── chemistry.py     # RDKit descriptors, pKa, compound type, AD
│   ├── adme.py          # XGBoost fup/CLint/RBP/VDss prediction
│   └── ivive.py         # CLint→enzyme affinity, R&R Kp, DrugOnGraph
│
├── ml/                  # Data-driven PK prediction
│   ├── features.py      # Feature vector (re-exports descriptors)
│   ├── models.py        # XGBoost Cmax predictor
│   └── ensemble.py      # Meta-learner (engine + ML → final)
│
├── pk/                  # PK endpoint computation
│   ├── endpoints.py     # SimResult → PKEndpoints
│   ├── nca.py           # AUC (trapezoidal), terminal t½
│   └── analytical.py    # 1-cpt oral, 2-cpt IV closed-form
│
├── validation/          # Reference data, benchmarks
│   ├── reference.py     # clinical_pk.json loader
│   ├── benchmark.py     # Holdout benchmark runner
│   ├── metrics.py       # AAFE, fold error, PI coverage
│   └── split.py         # Scaffold-stratified split
│
└── pipeline/            # Thin orchestrator
    ├── predict.py       # SMILES → PredictionResult
    └── config.py        # PipelineConfig

data/
├── physiology/          # BodyGraph YAML (reference_man, pediatric, overlays)
├── compounds/           # Curated drug configs (midazolam, caffeine, ...)
└── reference/           # clinical_pk.json (290 drugs), holdout.json

tests/                   # 251 tests
├── unit/                # Per-module tests
├── integration/         # Engine validation + extensibility proof
└── benchmark/           # Holdout AAFE gate
```

## Invariants

These are load-bearing walls. If any breaks, the architecture has failed.

1. **Engine is identity-blind.** No string matching on node names, enzyme names, or drug names anywhere in `engine/`. Test: replace every organ name in YAML with random strings — engine produces identical results.

2. **All parameters are Distribution.** No bare floats for physiological or drug parameters. `Distribution(mean=x, cv=0)` for deterministic values.

3. **Compile once, parameterize many.** Graph topology is compiled once. MC samples change parameters, not topology. 1000 iterations = 1 compile + 1000 solves.

4. **Flow conservation is a build-time guarantee.** The YAML builder validates flow balance before the graph reaches the engine.

5. **Holdout is inviolable.** 100 holdout drugs never appear in training, tuning, or optimization.

6. **No drug-specific branches.** The answer to "drug X gives wrong results" is never `if drug == X`.

## Predecessor

Sisyphus inherits validated data assets from [Omega PBPK](https://github.com/jam-sudo/Omega) (591 commits, 35-state hardcoded ODE) but not its architecture. Key empirical findings from Omega that informed Sisyphus:

- Data quality dominates: 14 reference corrections = −47.5% AAFE, zero model changes
- Gut CLint > hepatic CLint for Cmax (Sobol: gut ST=0.47, hepatic ST=0.00)
- Meta-learner > fixed ensemble (ML Cmax importance 50%, PBPK 26%)
- XGBoost ≥ MLP at 1K–4K drug scale

## Requirements

- Python ≥ 3.10
- numpy, scipy, pyyaml (core)
- rdkit, xgboost, scikit-learn (prediction)

## License

MIT
