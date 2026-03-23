# Sisyphus

**The human body as a typed directed graph. Drug pharmacokinetics from first principles.**

[Quickstart](#quickstart) &middot; [Architecture](#architecture) &middot; [Extend](#extending-the-model) &middot; [Benchmarks](#benchmarks)

---

Sisyphus is a computational pharmacokinetics platform that represents the human body as a **typed directed multi-graph**, automatically derives ODE systems from graph topology, and propagates uncertainty natively through all predictions.

Give it a SMILES string and a dose. It returns Cmax, Tmax, and t&frac12; &mdash; in 350 milliseconds.

```
$ sisyphus predict --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O" --dose 100

Drug: Cn1c(=O)c2c(ncn2C)n(C)c1=O
Method: hybrid
Confidence: high
Cmax: 1.5247 mg/L
Tmax: 0.96 h
t½: 4.32 h
  Engine Cmax: 0.0145 mg/L
  ML Cmax: 0.7410 mg/L
```

## Three Ideas

### 1. The body is a graph

Organs are nodes. Blood vessels, GI transit paths, and clearance routes are typed directed edges. The ODE system is **derived from graph topology**, not hand-written. To add a new organ, you edit YAML. You do not touch the engine.

```
                     ┌───────────────────────────────────────────────┐
                     │                                               │
   ┌──────┐    ┌─────┴────┐                                   ┌─────┴────┐
   │ lung │───►│ arterial │──► brain ─────────────────────────►│ venous   │
   └──┬───┘    │  blood   │──► heart ─────────────────────────►│  blood   │
      │        │          │──► kidney ────────────────────────►│          │
      │        │          │                                    │          │
      │        │          │──► gut wall ──┐                    │          │
      │        │          │──► spleen  ───┤ portal ──► liver ─►│          │
      │        │          │──► pancreas ──┘  vein    (CYP450)  │          │
      │        │          │                                    │          │
      │        │          │──► muscle ─────────────────────────►│          │
      │        │          │──► adipose ────────────────────────►│          │
      │        │          │──► skin, bone, rest, ... ─────────►│          │
      │        └──────────┘                                    └────┬─────┘
      │                                                             │
      └─────────────────────────────────────────────────────────────┘

   stomach ──► duodenum ──► jejunum ──► ileum ──► colon ──► fecal
                  │            │          │                 (excretion)
                  └────────────┴──────────┘
                       absorption ──► gut wall
```

34 nodes. 54 edges. 5 flux types. The engine walks the graph, dispatches flux functions by edge type, and assembles the right-hand side automatically.

### 2. Everything is a Distribution

`fup = 0.032` does not exist in Sisyphus. Only `fup = Distribution(mean=0.032, cv=0.4)` does.

Every physiological parameter, every drug property, every predicted ADME value carries its uncertainty. Monte Carlo sampling propagates these distributions through the graph to produce prediction intervals &mdash; not as a post-hoc feature, but as the system's native output format.

```python
from sisyphus.engine.uncertainty import UncertaintyEngine

ue = UncertaintyEngine()
mc = ue.propagate_fast(compiled, graph, drug, n_samples=1000)

print(mc.pk.cmax)        # Distribution(mean=1.76, cv=0.19)
print(mc.cmax_90ci)      # (1.28, 2.35) mg/L
print(len(mc.cmax_samples))  # 1000 individual Cmax realizations
```

### 3. The engine knows types, not identities

The engine knows *"this node has organ type with enzyme slots"* and *"this edge has clearance type using the well-stirred model."* It does not know *"this is the liver"* or *"this enzyme is CYP3A4."*

All identity lives in YAML and DrugOnGraph. This is what makes the architecture extensible &mdash; new organs, enzymes, routes, and populations require zero engine changes.

**Proof:** SC injection, pediatric physiology, tumor compartment, DDI, and PK/PD were all added with **0 lines changed** in `src/sisyphus/engine/`.

## Quickstart

### Installation

```bash
pip install -e ".[dev,ml,chem]"
```

> **Note:** Pre-trained XGBoost models (fup, CLint, Cmax, meta-learner) are required in `models/`.
> These are not tracked in git due to size. See [Predecessor](#predecessor) for data provenance.

### CLI

```bash
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

### Engine-only mode (compound YAML, no ADME prediction)

```python
from pathlib import Path
import numpy as np
from sisyphus.graph.builder import build_from_yaml
from sisyphus.compounds import load_compound
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.pk.endpoints import compute_endpoints
import sisyphus.engine.flux  # register flux specs

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
             pipeline (meta-learner → final PredictionResult)
```

| Layer | Responsibility | Depends on |
|-------|---------------|------------|
| `graph/` | BodyGraph, node/edge types, YAML builder | `core` |
| `engine/` | ODE compiler, flux registry, solver, MC | `core`, `graph` |
| `predict/` | SMILES → chemistry → ADME → DrugOnGraph | `core` |
| `ml/` | XGBoost Cmax, meta-learner | `core` |
| `pk/` | SimResult → PKEndpoints | `core` |
| `pipeline/` | Orchestrator wiring all layers | all layers |
| `ddi.py` | Drug-drug interactions (inhibition, induction) | `core`, `graph` |
| `pkpd.py` | PK/PD effect modeling (Emax) | `core` |

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
# SC injection — add a depot node and absorption edge
graph.add_node(Node(name="sc_depot", node_type="lumen", volume=Distribution(0.01)))
graph.add_edge(AbsorptionEdge(source="sc_depot", target="venous_blood",
                               ka_fraction=Distribution(1.0)))
```

### Add a new population

Create `pediatric_5y.yaml` with allometrically scaled volumes, cardiac output `CO × (18/70)^0.75`, and ontogeny-adjusted enzyme abundances (CYP3A4 at 50% of adult). Same graph structure, different parameters. Engine processes it identically.

### Drug-drug interactions

```python
from sisyphus.ddi import apply_inhibition, KETOCONAZOLE

inhibited_graph = apply_inhibition(graph, KETOCONAZOLE)
# CYP3A4 activity reduced >99%
# Midazolam AUC increases 12x — zero engine changes
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

### Engine Validation

4 drugs validated against [Omega PBPK](https://github.com/jam-sudo/Omega) ODE output:

| Drug | Sisyphus Cmax | Omega Cmax | Error |
|------|:------------:|:----------:|:-----:|
| Midazolam 2mg | 0.006911 | 0.006943 | 0.5% |
| Caffeine 100mg | 1.7151 | 1.7139 | 0.1% |
| Warfarin 10mg | 0.4917 | 0.4922 | 0.1% |
| Propranolol 80mg | 0.1353 | 0.1355 | 0.1% |

### Holdout Benchmark

Evaluated on scaffold-stratified holdout drugs using [AAFE](https://en.wikipedia.org/wiki/Average_fold_error) (Absolute Average Fold Error: 10^mean(|log10(pred/obs)|); 1.0 = perfect).

| Metric | Value |
|--------|:-----:|
| In-domain AAFE | **1.697** |
| In-domain %2-fold | **70.4%** |
| Deterministic latency | **350 ms** |
| Mass balance error | **< 10⁻¹²** |
| MC N=1000 | 33.5 s |

### Performance

| Operation | Time | Note |
|-----------|:----:|------|
| Full prediction (SMILES → Cmax) | 350 ms | Includes ADME + ODE + ML |
| ODE solve (full fidelity) | 106 ms | rtol=10⁻⁸ |
| ODE solve (MC fast path) | 33 ms | rtol=10⁻⁴ |
| RHS evaluation | 31 µs | 54 flux specs, pure Python |
| Graph compilation | < 1 ms | One-time cost |

## Project Structure

```
src/sisyphus/
├── core.py              # Distribution, TissueComposition, contracts
├── descriptors.py       # Morgan FP + RDKit descriptors (shared)
├── compounds.py         # Compound YAML → DrugOnGraph loader
├── ddi.py               # Drug-drug interaction (inhibition, induction)
├── pkpd.py              # PK/PD link (effect compartment, Emax)
├── cli.py               # CLI entry point
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
│   └── ivive.py         # CLint → enzyme affinity, R&R Kp, DrugOnGraph
│
├── ml/                  # Data-driven PK prediction
│   ├── features.py      # Feature vector (re-exports descriptors)
│   ├── models.py        # XGBoost Cmax predictor (v2, 1128 drugs)
│   └── ensemble.py      # Meta-learner (engine + ML → final Cmax)
│
├── pk/                  # PK endpoint computation
│   ├── endpoints.py     # SimResult → PKEndpoints
│   ├── nca.py           # AUC (trapezoidal), terminal t½
│   └── analytical.py    # 1-cpt oral, 2-cpt IV closed-form
│
├── validation/          # Reference data, benchmarks
│   ├── reference.py     # clinical_pk.json loader (290 drugs)
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

models/                  # Pre-trained XGBoost models (not in git)
├── adme/                # fup, CLint, RBP, VDss
├── direct_pk/           # Cmax v2
└── meta_learner/        # Engine + ML combination

tests/                   # 251 tests
├── unit/
├── integration/
└── benchmark/
```

## Invariants

These are load-bearing walls. If any breaks, the architecture has failed.

1. **Engine is identity-blind.** No string matching on node names, enzyme names, or drug names anywhere in `engine/`. Test: replace every organ name with random strings &mdash; engine produces identical results.

2. **All parameters are Distribution.** No bare floats for physiological or drug parameters. `Distribution(mean=x, cv=0)` for deterministic values.

3. **Compile once, parameterize many.** Graph topology compiled once. MC samples change parameters, not topology. 1000 iterations = 1 compile + 1000 solves.

4. **Flow conservation at build time.** The YAML builder validates flow balance before the graph reaches the engine.

5. **Holdout is inviolable.** 100 holdout drugs never appear in training, tuning, or optimization.

6. **No drug-specific branches.** The answer to "drug X gives wrong results" is never `if drug == X`.

## Predecessor

Sisyphus inherits validated data assets from [Omega PBPK](https://github.com/jam-sudo/Omega) (591 commits, 35-state hardcoded ODE) but not its architecture:

| Inherited (data) | Not inherited (architecture) |
|-------------------|------------------------------|
| 290-drug clinical reference | 35-state hardcoded ODE |
| 76/100 scaffold holdout split | Organ-specific CLint fields |
| ICRP physiology values | Sequential ADME → IVIVE chain |
| Trained XGBoost models | Point estimate pipeline |
| Rodgers & Rowland tissue compositions | Post-hoc hybrid selector |

Key empirical findings from Omega that informed Sisyphus design:

- **Data quality dominates.** 14 reference corrections = &minus;47.5% AAFE, zero model changes.
- **Gut CLint > hepatic CLint for Cmax.** Sobol analysis: gut ST=0.47, hepatic ST=0.00.
- **Meta-learner > fixed ensemble.** ML Cmax importance 50%, PBPK Cmax 26%.
- **XGBoost &ge; MLP** at 1K&ndash;4K drug scale.

## Requirements

- Python &ge; 3.10
- numpy, scipy, pyyaml (core)
- rdkit, xgboost, scikit-learn (prediction)

## License

MIT
