# Architecture

**Analysis Date:** 2026-03-24

## System Overview

Sisyphus is a PBPK (Physiologically-Based Pharmacokinetic) simulation platform that predicts drug concentrations in the human body from a SMILES string and dose. The system represents the human body as a typed directed multi-graph, auto-derives ODE systems from graph topology, and propagates uncertainty natively through Monte Carlo sampling.

The core pipeline: SMILES + dose -> molecular profiling -> ADME prediction -> IVIVE (in vitro to in vivo extrapolation) -> ODE simulation on body graph -> PK endpoint extraction -> meta-learner combination with ML direct prediction -> final PredictionResult.

## Pattern Overview

**Overall:** Layered pipeline with graph-driven ODE compilation and registry-based dispatch.

**Key Characteristics:**
- Identity-blind engine: operates on node/edge *types* and *tags*, never on names like "liver" or "CYP3A4"
- Everything is a Distribution: all parameters carry uncertainty (mean + CV), propagated via Monte Carlo
- Compile once, parameterize many: graph topology compiles to an ODE skeleton reused across MC samples
- Frozen dataclass contracts between layers: `DrugOnGraph`, `SimResult`, `PKEndpoints`, `PredictionResult`

## Core Abstractions

### Distribution (`src/sisyphus/core.py`)
- **Purpose**: Atomic unit of uncertainty. Every physiological and drug parameter is a Distribution.
- **Key types**: `Distribution(mean, cv, dist_type)` -- supports lognormal, normal, uniform
- **Relationships**: Sampled by `BodyGraph.sample()` and `DrugOnGraph.sample()` for MC propagation. Used in every layer.

### BodyGraph (`src/sisyphus/graph/body.py`)
- **Purpose**: Typed directed multi-graph of the human body. Organs are nodes, transport pathways are edges.
- **Key types**: `BodyGraph` (container), `Node` (compartment), `Edge` (base), `FlowEdge`, `ClearanceEdge`, `TransitEdge`, `AbsorptionEdge`, `DiffusionEdge`, `ActiveTransportEdge`
- **Relationships**: Built from YAML by `build_from_yaml()` (`src/sisyphus/graph/builder.py`). Consumed by `ODECompiler`. Self-validates flow conservation.

### DrugOnGraph (`src/sisyphus/core.py`)
- **Purpose**: Drug properties mapped onto graph topology. The contract between predict and engine layers.
- **Key types**: `DrugOnGraph` (frozen dataclass) with `enzyme_affinity: dict[str, Distribution]` (per-enzyme CLint, not per-organ), `kp_overrides`, `transporter_kinetics`, `fup`, `rbp`, `peff`, `solubility`, `renal_clearance`
- **Relationships**: Produced by `build_drug_on_graph()` in `src/sisyphus/predict/ivive.py`. Consumed by `ResolvedParams` in the engine.

### CompiledODE (`src/sisyphus/engine/compiler.py`)
- **Purpose**: Reusable ODE skeleton. Fixes state indexing and edge-to-flux mappings from graph topology.
- **Key types**: `CompiledODE` (state_index, flux_specs, n_states), `ODECompiler`, `ResolvedParams`
- **Relationships**: Produced by `ODECompiler.compile(graph)`. Used by `solve()` and `solve_mc()` in `src/sisyphus/engine/solver.py`. `make_rhs(params)` binds point-value parameters to produce the ODE RHS callable.

### FluxSpec (`src/sisyphus/engine/flux.py`)
- **Purpose**: Abstract flux computation per edge type. Each edge type dispatches to a FluxSpec subclass.
- **Key types**: `FluxSpec` (ABC), `FlowFluxSpec`, `ClearanceFluxSpec`, `TransitFluxSpec`, `AbsorptionFluxSpec`, `DiffusionFluxSpec`, `ActiveTransportFluxSpec`
- **Relationships**: Registered via `@register_flux(edge_type)` decorator into `FLUX_REGISTRY`. Instantiated during `ODECompiler.compile()`. Called during RHS evaluation.

### SimResult (`src/sisyphus/core.py`)
- **Purpose**: Raw ODE solution. Named access by node: `concentrations["venous_blood"]`.
- **Key types**: `SimResult(time_h, concentrations, amounts, mass_balance_error, solver_success)`
- **Relationships**: Produced by `solve()` in `src/sisyphus/engine/solver.py`. Consumed by `compute_endpoints()` in `src/sisyphus/pk/endpoints.py`.

### PKEndpoints (`src/sisyphus/core.py`)
- **Purpose**: Pharmacokinetic endpoints (Cmax, Tmax, AUC, t_half, CL, Vss).
- **Key types**: `PKEndpoints` -- all fields are Distribution
- **Relationships**: Produced by PK layer (`src/sisyphus/pk/endpoints.py`) or ML layer (`src/sisyphus/ml/models.py`). Consumed by MetaLearner.

### PredictionResult (`src/sisyphus/core.py`)
- **Purpose**: Final pipeline output combining engine and ML predictions.
- **Key types**: `PredictionResult(drug_name, pk, method, engine_pk, ml_pk, confidence, cmax_90ci, ...)`
- **Relationships**: Produced by `predict()` in `src/sisyphus/pipeline/predict.py`. Consumed by CLI and validation.

## Layer Dependencies

```
pipeline/predict.py
    depends on -> predict (chemistry, adme, ivive, drugbank)
                  engine (compiler, flux, solver, uncertainty)
                  ml (models, ensemble)
                  pk (endpoints)
                  graph (builder)

engine/         depends on -> graph, core
predict/        depends on -> core, descriptors (external libs: rdkit, xgboost)
ml/             depends on -> core, descriptors (external libs: xgboost)
pk/             depends on -> core (nothing else)
graph/          depends on -> core (nothing else)
validation/     depends on -> pipeline, core
ddi.py          depends on -> core, graph
pkpd.py         depends on -> core
descriptors.py  depends on -> (external libs: rdkit, numpy)
compounds.py    depends on -> core
```

**Critical constraint: predict does NOT import engine. engine does NOT import predict. No cross-layer imports outside pipeline.**

The `descriptors.py` module at the package root is a shared utility used by both `predict/adme.py` and `ml/models.py` to avoid cross-layer dependencies.

## Data Flow

### Primary Pipeline (oral drug prediction)

```
SMILES + dose_mg + route
    |
    v
compute_profile(smiles)                 [src/sisyphus/predict/chemistry.py]
    MolecularProfile (mw, logp, pka, compound_type, ad_flags)
    |                                    DrugBank enrichment: logp override, pka override
    v
predict_adme(profile)                   [src/sisyphus/predict/adme.py]
    ADMEProperties (fup, clint, peff, solubility, vdss, rbp)
    |                                    XGBoost models from models/adme/
    |                                    DrugBank fup measured values
    v
build_drug_on_graph(profile, adme)      [src/sisyphus/predict/ivive.py]
    DrugOnGraph                          CLint decomposition to enzyme_affinity
    |                                    Kp via Rodgers & Rowland
    |                                    Renal CL estimation
    v
build_from_yaml("reference_man.yaml")   [src/sisyphus/graph/builder.py]
    BodyGraph                            YAML -> nodes + edges + flow validation
    |
    v
ODECompiler().compile(graph)            [src/sisyphus/engine/compiler.py]
    CompiledODE                          State indexing, FluxSpec mapping
    |
    v
graph.sample(rng) + drug.sample(rng)    [core.py, graph/body.py]
    realized_graph, realized_drug        Point-value sampling of Distributions
    |
    v
ResolvedParams(graph, drug)             [src/sisyphus/engine/compiler.py]
    params                               Pre-computed lookups for fast RHS eval
    |
    v
solve(compiled, params, y0, t_span)     [src/sisyphus/engine/solver.py]
    SimResult                            SciPy solve_ivp (LSODA), mass balance check
    |
    v
compute_endpoints(sim_result)           [src/sisyphus/pk/endpoints.py]
    engine_pk: PKEndpoints               Cmax, Tmax, AUC, t_half from venous_blood
    |
    +--- (parallel) ---+
    |                   |
    v                   v
PKPredictor()          UncertaintyEngine()
    ml_pk               MCResult (optional)
    |                   |
    v                   v
MetaLearner().combine(engine_pk, ml_pk)  [src/sisyphus/ml/ensemble.py]
    final PKEndpoints                     Adaptive geometric weighting in log space
    |
    v
PredictionResult                        [src/sisyphus/pipeline/predict.py]
```

### MC Uncertainty Propagation

```
CompiledODE (compiled once)
    |
    for i in range(N):
        rng = default_rng(seed + i)
        realized_graph = graph.sample(rng)        # sample physiology distributions
        realized_drug = drug.sample(rng)           # sample drug property distributions
        params = ResolvedParams(realized_graph, realized_drug)
        cmax, tmax, auc, ok = solve_mc(compiled, params, y0, t_span)
    |
    v
MCResult (median, CV, 90% PI from N samples)
```

### State Management

- **No persistent state.** Each prediction is stateless. Models are loaded lazily and cached in module-level variables (`_model_cache` in adme.py, `DrugBankLookup` singleton).
- **BodyGraph is mutable during construction** (add_node, add_edge), then treated as immutable after validation. `sample()` returns a new graph.
- **All contract types are frozen dataclasses.** `DrugOnGraph.sample()` returns a new instance.

## Entry Points

### CLI (`src/sisyphus/cli.py`)
- **Location**: `src/sisyphus/cli.py`, registered as `sisyphus` console script in `pyproject.toml`
- **Triggers**: `sisyphus predict --smiles "..." --dose 500` or `sisyphus benchmark --holdout`
- **Responsibilities**: Parse args, set up logging, dispatch to `pipeline.predict.predict()` or `validation.benchmark.run_benchmark()`

### Pipeline predict (`src/sisyphus/pipeline/predict.py`)
- **Location**: `src/sisyphus/pipeline/predict.py::predict()`
- **Triggers**: Called by CLI, tests, benchmarks, scripts
- **Responsibilities**: Orchestrate all layers in correct order, handle failures gracefully (engine failure falls back to ML-only)

### Benchmark runner (`src/sisyphus/validation/benchmark.py`)
- **Location**: `src/sisyphus/validation/benchmark.py::run_benchmark()`
- **Triggers**: `sisyphus benchmark --holdout` or test harness
- **Responsibilities**: Load reference drugs, run predictions, compute AAFE/pct_2fold, report gold/silver stratification

## Error Handling

**Strategy:** Structured results with warnings, not exceptions (except invalid SMILES).

**Patterns:**
- Invalid SMILES raises `ValueError` immediately in `compute_profile()` (`src/sisyphus/predict/chemistry.py`)
- Graph validation failure raises `ValueError` in `build_from_yaml()` (`src/sisyphus/graph/builder.py`)
- Engine solver failure: `SimResult.solver_success=False`, warning appended, prediction falls back to ML-only
- MC sample failure: individual sample silently skipped (logged at DEBUG), `n_failures` counted in result
- ML model load failure: warning appended, prediction continues with engine-only or zero
- DrugBank lookup failure: silently returns `None`, predict layer falls back to XGBoost/heuristic
- All failures in `pipeline/predict.py` are wrapped in try/except with warnings appended to `PredictionResult.warnings`

## Cross-Cutting Concerns

### Logging
- `logging` module throughout, never `print()` (except CLI output)
- Each module creates `logger = logging.getLogger(__name__)`
- CLI sets level: `--verbose` -> `DEBUG`, default -> `WARNING`

### Validation
- **Graph**: Flow conservation validated at build time in `BodyGraph.validate()` -- invalid topology never reaches the engine
- **ADME**: DrugBank fup 5x cross-validation guard (reject if DrugBank and XGBoost disagree by >5x)
- **Benchmark**: Holdout invariant enforced by `validation/reference.py` (`in_holdout` flag from `holdout.json`)

### Authentication/Authorization
- Not applicable -- this is a scientific computation tool, not a web service

### DDI (Drug-Drug Interaction)
- Implemented in `src/sisyphus/ddi.py` by modifying enzyme abundances on the BodyGraph *before* ODE compilation
- Engine remains identity-blind -- it just sees different enzyme abundances
- Presets: ketoconazole, fluconazole, quinidine (inhibitors), rifampin (inducer)

### PK/PD
- Effect compartment model in `src/sisyphus/pkpd.py`, post-processing on `SimResult`
- Uses analytical convolution, not ODE -- keeps engine clean
- Presets: midazolam sedation, warfarin INR

## Key Design Decisions

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| Body as typed directed multi-graph (YAML-defined) | Extensibility: add organs/routes by editing YAML, not engine code. ODE system derived from topology automatically. | More complex than hardcoded compartment models. YAML schema must be carefully validated. |
| Identity-blind engine | Engine code never changes when adding new organs, enzymes, or transporters. Invariant tested by renaming all nodes. | Requires lookup_name indirection for Kp resolution. Parameter access through ResolvedParams is less direct. |
| Everything is Distribution | Native uncertainty propagation. No post-hoc bolted-on intervals. MC sampling is a first-class operation. | TissueComposition is a conscious exception (low Kp sensitivity). Performance cost of MC sampling. |
| Compile once, parameterize many | 1000 MC iterations = 1 compile + 1000 solves. Topology analysis is done once. | CompiledODE is mutable (flux_specs list). ResolvedParams rebuilt per sample. |
| Per-enzyme CLint (not per-organ) | Drug's `enzyme_affinity` matched to node's `enzymes` at any node. IVIVE is organ-blind. Same CYP3A4 affinity applies in liver and gut. | CLint decomposition into fm fractions is uncertain. Default fm uses heuristic/DrugBank annotations. |
| Registry-based flux dispatch | `FLUX_REGISTRY[edge_type]` maps edge types to FluxSpec classes. Adding a new transport mechanism = one decorated class. | Global mutable registry. Must import `flux.py` to register specs (done in pipeline via `import sisyphus.engine.flux`). |
| Meta-learner (adaptive geometric weighting) | Engine helps for base drugs only (w=0.65). ML-only for non-bases (w=0.00). LOOCV-validated on N=61 holdout. | Compound-type-dependent weighting is coarse. Engine's value limited to one drug class currently. |
| DrugBank enrichment (singleton lookup) | Measured fup, experimental logP, ChemAxon pKa, CYP substrate annotations improve prediction accuracy. | Adds 6MB of CSV data. Singleton pattern makes testing harder (requires `_reset_singleton()`). |
| SciPy LSODA solver | Automatic stiffness detection (switches BDF/Adams). Robust for PBPK ODE systems. | No parallelism (Python GIL). MC iterations are sequential. |
| Frozen dataclass contracts | Immutability guarantees correctness across layers. `sample()` returns new instances. | Verbose construction. `dataclasses.replace()` needed for modifications. |

---

*Architecture analysis: 2026-03-24*
