# Codebase Structure

**Analysis Date:** 2026-03-24

## Directory Layout

```
Sisyphus/
├── src/sisyphus/               # Main package (5,620 lines Python)
│   ├── __init__.py             # Package init (3 lines)
│   ├── core.py                 # Shared contract types: Distribution, DrugOnGraph, SimResult, PKEndpoints, PredictionResult (296 lines)
│   ├── cli.py                  # CLI entry point: predict + benchmark commands (92 lines)
│   ├── compounds.py            # YAML compound loader -> DrugOnGraph (90 lines)
│   ├── descriptors.py          # Shared Morgan FP + RDKit descriptor computation (71 lines)
│   ├── ddi.py                  # Drug-drug interaction: inhibition/induction via enzyme abundance modification (256 lines)
│   ├── pkpd.py                 # PK/PD link: effect compartment + sigmoid Emax model (236 lines)
│   ├── graph/                  # Body graph layer (650 lines)
│   │   ├── __init__.py         # Re-exports BodyGraph, Node, Edge types (25 lines)
│   │   ├── types.py            # Node, Edge hierarchy (FlowEdge, ClearanceEdge, etc.) (155 lines)
│   │   ├── body.py             # BodyGraph: typed directed multi-graph + validation + sampling (199 lines)
│   │   ├── builder.py          # YAML -> BodyGraph: flow fraction conversion, 2-pass edge inference (239 lines)
│   │   └── presets.py          # Convenience loaders: reference_man(), reference_woman() (32 lines)
│   ├── engine/                 # ODE engine layer (1,237 lines)
│   │   ├── __init__.py         # Re-exports CompiledODE, ODECompiler, FluxSpec, etc. (13 lines)
│   │   ├── compiler.py         # ODECompiler + ResolvedParams + CompiledODE (292 lines)
│   │   ├── flux.py             # FluxSpec ABC + 6 implementations + FLUX_REGISTRY (493 lines)
│   │   ├── solver.py           # solve() + solve_mc() wrapping SciPy solve_ivp (145 lines)
│   │   ├── uncertainty.py      # UncertaintyEngine: MC propagate + propagate_fast (288 lines)
│   │   └── result.py           # Re-exports SimResult from core (9 lines)
│   ├── predict/                # SMILES -> DrugOnGraph layer (1,586 lines)
│   │   ├── __init__.py         # Empty (1 line)
│   │   ├── chemistry.py        # SMILES -> MolecularProfile: descriptors, pKa, prodrug detection, AD (377 lines)
│   │   ├── adme.py             # XGBoost ADME predictions: fup, CLint, RBP, VDss, Peff, solubility (278 lines)
│   │   ├── ivive.py            # IVIVE: CLint decomposition, Kp (R&R/PT/BZ), renal CL, DrugOnGraph assembly (654 lines)
│   │   └── drugbank.py         # DrugBank CSV lookup: SMILES -> fup, pKa, logP, CYP substrates (277 lines)
│   ├── ml/                     # ML direct PK prediction layer (246 lines)
│   │   ├── __init__.py         # Empty (1 line)
│   │   ├── models.py           # PKPredictor: XGBoost Cmax from SMILES + dose (63 lines)
│   │   ├── ensemble.py         # MetaLearner: adaptive geometric weighting of engine + ML (129 lines)
│   │   ├── features.py         # Re-export of compute_features from descriptors.py (13 lines)
│   │   └── registry.py         # ModelRecord + ModelRegistry (not yet implemented) (41 lines)
│   ├── pk/                     # SimResult -> PKEndpoints layer (164 lines)
│   │   ├── __init__.py         # Empty (1 line)
│   │   ├── endpoints.py        # compute_endpoints(): Cmax, Tmax, AUC, t_half from SimResult (37 lines)
│   │   ├── nca.py              # NCA: trapezoidal AUC, terminal half-life regression (61 lines)
│   │   └── analytical.py       # 1-compartment oral, 2-compartment IV closed-form solutions (66 lines)
│   ├── pipeline/               # Orchestrator layer (250 lines)
│   │   ├── __init__.py         # Empty (1 line)
│   │   ├── predict.py          # predict(): end-to-end SMILES -> PredictionResult (219 lines)
│   │   └── config.py           # PipelineConfig dataclass (30 lines)
│   └── validation/             # Benchmark and metrics (437 lines)
│       ├── __init__.py         # Empty (1 line)
│       ├── benchmark.py        # run_benchmark(): holdout AAFE, gold/silver stratification (211 lines)
│       ├── reference.py        # load_reference(): clinical_pk.json -> DrugReference list (124 lines)
│       ├── metrics.py          # aafe(), pct_within_n_fold(), pi_coverage() (72 lines)
│       └── split.py            # scaffold_split() (not implemented) (29 lines)
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests (15 files)
│   ├── integration/            # Integration tests (2 files)
│   └── benchmark/              # Benchmark tests (1 file)
├── data/                       # Data assets
│   ├── physiology/             # YAML body graph definitions
│   ├── compounds/              # Drug YAML configurations (5 drugs)
│   ├── reference/              # Clinical PK reference data + holdout split
│   ├── drugbank/               # Extracted DrugBank CSVs (6 files, ~6.7 MB)
│   └── training/               # TDC training datasets (empty placeholder)
├── models/                     # Trained ML model artifacts
│   ├── adme/                   # XGBoost ADME models (7 JSON files)
│   ├── direct_pk/              # XGBoost Cmax model (1 JSON file + meta)
│   └── meta_learner/           # XGBoost meta-learner (1 JSON file)
├── scripts/                    # Training, benchmarking, and analysis scripts (13 files)
├── docs/                       # Documentation
├── pyproject.toml              # Build config, dependencies, ruff/pytest settings
├── DESIGN.md                   # Full design specification (authoritative)
└── README.md                   # Project readme
```

## Module Responsibilities

| Module | Purpose | Key Files | Lines |
|--------|---------|-----------|-------|
| `core` | Shared contract types (Distribution, DrugOnGraph, SimResult, PKEndpoints, PredictionResult) | `core.py` | 296 |
| `graph` | Body graph types, construction, validation, sampling | `types.py`, `body.py`, `builder.py`, `presets.py` | 650 |
| `engine` | ODE compilation, flux registry, solver, MC uncertainty | `compiler.py`, `flux.py`, `solver.py`, `uncertainty.py` | 1,237 |
| `predict` | SMILES -> molecular profiling -> ADME prediction -> IVIVE -> DrugOnGraph | `chemistry.py`, `adme.py`, `ivive.py`, `drugbank.py` | 1,586 |
| `ml` | Direct PK prediction (XGBoost Cmax) + meta-learner ensemble | `models.py`, `ensemble.py`, `features.py`, `registry.py` | 246 |
| `pk` | SimResult -> PKEndpoints (Cmax, AUC, t_half) | `endpoints.py`, `nca.py`, `analytical.py` | 164 |
| `pipeline` | Thin orchestrator: SMILES -> PredictionResult | `predict.py`, `config.py` | 250 |
| `validation` | Benchmark runner, reference data loader, accuracy metrics | `benchmark.py`, `reference.py`, `metrics.py`, `split.py` | 437 |
| root utils | CLI, compounds loader, descriptors, DDI, PK/PD | `cli.py`, `compounds.py`, `descriptors.py`, `ddi.py`, `pkpd.py` | 745 |

**Total source:** 5,620 lines Python

## Key File Locations

### Entry Points
- `src/sisyphus/cli.py`: CLI entry point (`sisyphus predict`, `sisyphus benchmark`)
- `src/sisyphus/pipeline/predict.py::predict()`: Programmatic entry point for single-drug prediction
- `src/sisyphus/validation/benchmark.py::run_benchmark()`: Programmatic entry point for holdout benchmarks

### Configuration
- `pyproject.toml`: Build system, dependencies, ruff config, pytest config
- `data/physiology/reference_man.yaml`: Primary body graph definition (ICRP Reference Man)
- `data/physiology/pediatric_5y.yaml`: Pediatric body graph
- `data/physiology/sc_overlay.yaml`: Subcutaneous injection overlay
- `data/physiology/tumor_overlay.yaml`: Tumor compartment overlay
- `src/sisyphus/pipeline/config.py`: PipelineConfig dataclass (defaults, not yet wired)

### Core Logic
- `src/sisyphus/core.py`: All inter-layer contract types
- `src/sisyphus/engine/compiler.py`: ODECompiler + ResolvedParams (the ODE skeleton builder)
- `src/sisyphus/engine/flux.py`: FluxSpec implementations (6 transport mechanisms)
- `src/sisyphus/predict/ivive.py`: IVIVE chain (CLint decomposition, Kp calc, DrugOnGraph assembly)
- `src/sisyphus/predict/chemistry.py`: Molecular profiling + prodrug detection + AD checks
- `src/sisyphus/ml/ensemble.py`: MetaLearner (adaptive engine/ML weighting)

### Data
- `data/reference/clinical_pk.json`: Clinical PK reference for 176+ drugs
- `data/reference/holdout.json`: Frozen holdout drug names (inviolable)
- `data/drugbank/drugs.csv`: DrugBank drug index with SMILES, InChIKey, pKa
- `data/drugbank/enzyme_annotations.csv`: CYP/UGT substrate annotations
- `data/drugbank/experimental_properties.csv`: Experimental logP values
- `data/drugbank/pk_data.csv`: Protein binding (fup) measurements
- `data/compounds/*.yaml`: Curated drug configs (midazolam, warfarin, caffeine, propranolol, midazolam_sc)

### ML Models
- `models/adme/xgboost_fup.json`: fup v1 model (log10 space)
- `models/adme/xgboost_fup_v2.json`: fup v2 model (logit space, preferred)
- `models/adme/xgboost_clint.json`: CLint model (log10 space)
- `models/adme/xgboost_rbp.json`: RBP model (Morgan FP only, 2048 features)
- `models/adme/xgboost_vdss.json`: VDss model (log10 space)
- `models/adme/xgboost_peff.json`: Peff model (Caco-2 -> in vivo calibrated)
- `models/adme/logp_correction.json`: logP residual correction model
- `models/direct_pk/xgboost_cmax.json`: Direct Cmax predictor (1,128 MMPK drugs)
- `models/meta_learner/xgboost_meta.json`: Meta-learner (currently unused -- hardcoded adaptive weights in ensemble.py)

### Testing
- `tests/unit/`: 15 test files covering all modules
- `tests/integration/test_engine_validation.py`: Engine accuracy validation against Omega
- `tests/integration/test_extensibility.py`: Extensibility proof (SC, pediatric, tumor)
- `tests/benchmark/test_holdout.py`: Holdout AAFE acceptance test

### Scripts
- `scripts/train_fup_v2.py`: Train fup v2 (logit-space) model
- `scripts/train_peff.py`: Train Peff XGBoost on Caco-2 data
- `scripts/train_logp_correction.py`: Train logP residual correction
- `scripts/train_e2e.py`: E2E differentiable PBPK experiment (MLP + finite diff)
- `scripts/run_engine_benchmark.py`: Engine-only benchmark
- `scripts/run_chain_benchmark.py`: IVIVE chain benchmark
- `scripts/run_loocv_validation.py`: LOOCV validation
- `scripts/run_loocv_weights.py`: LOOCV weight optimization
- `scripts/run_ablation.py`: DrugBank feature ablation study
- `scripts/run_mechanism_audit.py`: Mechanism audit script
- `scripts/run_ugt_sensitivity.py`: UGT sensitivity analysis
- `scripts/extract_drugbank.py`: Extract data from DrugBank XML
- `scripts/holdout_audit.py`: Holdout set audit

## Naming Conventions

### Files
- Lowercase snake_case: `predict_adme.py`, `body_graph.py`
- Layer modules: directory per layer (`graph/`, `engine/`, `predict/`, `ml/`, `pk/`, `pipeline/`, `validation/`)
- Cross-layer utilities at package root: `core.py`, `descriptors.py`, `compounds.py`
- Scripts: `run_*.py` for execution scripts, `train_*.py` for training scripts

### Directories
- Lowercase snake_case
- Maximum 20 files per directory (hard ceiling per invariants)
- `data/` for data assets, `models/` for ML artifacts, `scripts/` for one-off scripts

### Classes
- PascalCase: `BodyGraph`, `DrugOnGraph`, `CompiledODE`, `FluxSpec`, `MetaLearner`
- Frozen dataclasses for contracts: `Distribution`, `SimResult`, `PKEndpoints`, `PredictionResult`
- Registry pattern: `FLUX_REGISTRY` (UPPER_SNAKE for module-level registries)

### Functions
- snake_case: `compute_profile()`, `predict_adme()`, `build_drug_on_graph()`
- Private with underscore: `_compute_kp_rodgers_rowland()`, `_predict_fup()`
- Constants: `UPPER_SNAKE` with unit suffix: `_GFR_L_PER_H`, `_CLINT_SCALING`, `_HPGL`

## Where to Add New Code

### New ADME predictor (e.g., solubility XGBoost)
- Train script: `scripts/train_solubility.py`
- Model artifact: `models/adme/xgboost_solubility.json`
- Predictor function: `src/sisyphus/predict/adme.py` -- add `_predict_solubility()`, update `predict_adme()`
- Test: `tests/unit/test_adme_ivive.py`

### New transport mechanism (e.g., saturable absorption)
- Edge type: `src/sisyphus/graph/types.py` -- add `SaturableAbsorptionEdge(Edge)`
- FluxSpec: `src/sisyphus/engine/flux.py` -- add `@register_flux("saturable_absorption") class SaturableAbsorptionFluxSpec(FluxSpec)`
- YAML edge: `data/physiology/reference_man.yaml` -- add edges with `type: saturable_absorption`
- Test: `tests/unit/test_flux.py`

### New organ/compartment
- YAML only: `data/physiology/reference_man.yaml` -- add node + edges
- No engine changes required (identity-blind invariant)
- If new edge type needed, follow "New transport mechanism" above

### New population model (e.g., elderly)
- YAML: `data/physiology/elderly_70y.yaml`
- Preset: `src/sisyphus/graph/presets.py` -- add `elderly()` function
- Test: `tests/integration/test_extensibility.py`

### New drug compound (curated, with known parameters)
- YAML: `data/compounds/<drug_name>.yaml` -- follow existing format (e.g., `midazolam.yaml`)
- Load via: `src/sisyphus/compounds.py::load_compound(path)`

### New ML model (e.g., Chemprop)
- Model class: `src/sisyphus/ml/models.py` -- add class alongside `PKPredictor`
- Feature computation: If different features needed, add to `src/sisyphus/descriptors.py` or new file in `ml/`
- Integration in pipeline: `src/sisyphus/pipeline/predict.py`
- Test: `tests/unit/test_ml_models.py`

### New validation metric
- Metric function: `src/sisyphus/validation/metrics.py`
- Integration: `src/sisyphus/validation/benchmark.py`
- Test: `tests/unit/test_validation.py`

### New DDI perpetrator
- Preset constant: `src/sisyphus/ddi.py` -- add `Inhibitor()` or `Inducer()` instance
- Test: `tests/unit/test_ddi.py`

### New PD model
- Preset constant: `src/sisyphus/pkpd.py` -- add `PDModel()` instance
- Test: `tests/unit/test_pkpd.py`

## Special Directories

### `models/`
- Purpose: Serialized XGBoost model artifacts (JSON format)
- Generated: Yes (by `scripts/train_*.py` scripts)
- Committed: Yes (checked into git for reproducibility)
- Note: Models reference training data that may live outside the repo

### `data/drugbank/`
- Purpose: Extracted DrugBank CSVs for experimental property enrichment
- Generated: Yes (by `scripts/extract_drugbank.py` from `full database.xml`)
- Committed: Yes
- Note: `full database.xml` (1.9 GB) is NOT committed (in `.gitignore`)

### `data/reference/`
- Purpose: Clinical PK ground truth for benchmarking
- Generated: No (curated manually from FDA labels and literature)
- Committed: Yes
- Critical: `holdout.json` is frozen and inviolable

### `data/training/`
- Purpose: TDC ADME datasets for model training
- Generated: No (downloaded from Therapeutic Data Commons)
- Committed: Placeholder only (`.gitkeep`)

### `.planning/`
- Purpose: GSD planning documents
- Generated: Yes (by mapping tools)
- Committed: Yes

### `__pycache__/`
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No (in `.gitignore`)

## Import Patterns

### Layer imports follow strict dependency rules:

```python
# core.py -- imports only numpy
# graph/* -- imports from core only
# engine/* -- imports from graph and core only
# predict/* -- imports from core and descriptors (external: rdkit, xgboost)
# ml/* -- imports from core and descriptors (external: xgboost)
# pk/* -- imports from core only
# pipeline/* -- imports from ALL layers (the sole orchestrator)
# validation/* -- imports from pipeline and core
```

### Circular import avoidance in engine:
```python
# engine/compiler.py uses TYPE_CHECKING for DrugOnGraph
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sisyphus.core import DrugOnGraph
```

### Lazy imports in pipeline/predict.py:
```python
# All sub-layer imports are inside the predict() function body
# to avoid circular imports and ensure flux specs are registered
def predict(smiles, dose_mg, route, n_mc_samples):
    import sisyphus.engine.flux  # noqa: F401 -- register flux specs
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.ml.ensemble import MetaLearner
    # ...
```

---

*Structure analysis: 2026-03-24*
