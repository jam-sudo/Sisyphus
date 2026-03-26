# Coding Conventions

**Analysis Date:** 2026-03-24

## Style & Formatting

**Formatter/Linter:** `ruff` v0.4+ (configured in `pyproject.toml`)
- Line length: 100 characters
- Target: Python 3.10
- Lint rules: `["E", "F", "I", "W", "UP"]` (pycodestyle errors/warnings, pyflakes, isort, upgrade)
- No `.prettierrc`, `.flake8`, or standalone `ruff.toml` -- all config in `pyproject.toml`

**Python Version:** 3.10+ (uses `X | Y` union syntax, `from __future__ import annotations` everywhere)

## Naming Conventions

**Files:**
- `snake_case.py` for all modules: `body.py`, `compiler.py`, `flux.py`, `predict.py`
- Test files: `test_{module_name}.py` -- e.g., `test_compiler.py`, `test_flux.py`

**Classes:**
- `PascalCase`: `BodyGraph`, `FlowFluxSpec`, `ODECompiler`, `DrugOnGraph`, `PKEndpoints`
- All data containers are `@dataclass(frozen=True)`: `Distribution`, `Node`, `Edge`, `SimResult`, `PKEndpoints`, `PredictionResult`, `DrugOnGraph`, `ADMEProperties`, `BenchmarkResult`, `MCResult`

**Functions:**
- `snake_case`: `compute_profile()`, `build_drug_on_graph()`, `predict_adme()`
- Private helpers prefixed with `_`: `_decompose_clint()`, `_compute_kp_rodgers_rowland()`, `_estimate_peff_heuristic()`

**Variables:**
- `snake_case` for locals and instance attrs
- Module-level private data prefixed with `_`: `_LIVER_ENZYME_ABUNDANCE`, `_DEFAULT_FM`, `_TISSUE_COMPOSITIONS`

**Constants:**
- `UPPER_SNAKE` with unit suffix where applicable: `_GFR_L_PER_H = 7.5`, `_HPGL = 120e6`, `_CLINT_SCALING`
- Always include source comment for non-obvious values:
  ```python
  _GFR_L_PER_H = 7.5  # ~125 mL/min = 7.5 L/h
  _HPGL = 120e6  # cells/g
  ```

**Types:**
- `PascalCase` for type aliases and dataclasses
- `str` literals for discriminated unions: `node_type: str  # "organ" | "blood_pool" | "lumen" | "sink"`
- Use `X | None` (not `Optional[X]`): `pka: float | None`, `t_half: Distribution | None = None`

## Import Organization

**Order:**
1. `from __future__ import annotations` (always first, present in every module)
2. Standard library: `logging`, `dataclasses`, `pathlib`, `abc`, `collections.abc`
3. Third-party: `numpy`, `scipy`, `xgboost`, `rdkit`, `yaml`
4. Internal: `from sisyphus.core import ...`, `from sisyphus.graph.types import ...`

**Path Aliases:** None. All imports use full dotted paths: `from sisyphus.engine.compiler import ODECompiler`

**TYPE_CHECKING Guard:** Used to break circular imports in `compiler.py` and `flux.py`:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sisyphus.core import DrugOnGraph
```

**Flux Registration Side-Effect Import:** Test files and pipeline code that need FluxSpec implementations to be registered must import the flux module even if not directly referencing it:
```python
import sisyphus.engine.flux  # noqa: F401 -- register flux specs
```

## Error Handling Patterns

**ValueError for Input Validation:**
- Invalid SMILES: `raise ValueError(f"Invalid SMILES: {smiles!r}")`
- Graph validation: `raise ValueError(f"Graph validation failed: {errors}")`
- Duplicate nodes: `raise ValueError(f"duplicate node name: {node.name!r}")`
- Unknown edge types: `raise ValueError(f"No FluxSpec registered for edge type: {edge.edge_type!r}")`

**Structured Results Instead of Exceptions for Runtime Failures:**
- Solver failure: `SimResult.solver_success = False` (never raises)
- Pipeline failures: `warnings=["Engine failed: {e}"]`, `confidence="low"` (never raises)
- MC sample failures: silently counted as `n_failures`, logged at `DEBUG` level

**Guard Clauses in Flux Computations:**
Every FluxSpec `apply()` method uses early returns for degenerate cases:
```python
if clint_organ <= 0:
    return  # No metabolism at this node
if denom < 1e-12:
    return  # Avoid division by zero
```

**Try/Except Wrapping in Pipeline:**
Each pipeline step (engine, ML, MC, DrugBank enrichment) is wrapped independently:
```python
try:
    engine_pk = ...
except Exception as e:
    warnings_list.append(f"Engine failed: {e}")
    logger.warning("Engine simulation failed: %s", e)
```

## Logging & Observability

**Framework:** Python `logging` module exclusively. Never `print()` in library code (only `cli.py` uses `print()`).

**Logger Initialization:** Module-level, one per file:
```python
import logging
logger = logging.getLogger(__name__)
```

**Log Levels Used:**
- `logger.debug()` -- per-MC-sample outcomes, model loading
- `logger.info()` -- ADME predictions, DrugOnGraph assembly, pipeline step results, benchmark per-drug results
- `logger.warning()` -- RBP far from 1.0, DrugBank fup disagreement, route fallback, solver failures
- `logger.error()` -- not commonly used (failures go to `warnings_list` or `DEBUG`)

**Log Message Format:** Uses `%`-style formatting (not f-strings) per logging best practice:
```python
logger.info("ADME predicted: fup=%.3f, CLint=%.1f uL/min/10^6", fup.mean, clint.mean)
logger.warning("RBP prediction %.2f far from 1.0, defaulting to 1.0", rbp)
```

**Modules with logging (14 of ~40):** `pipeline/predict.py`, `predict/ivive.py`, `predict/adme.py`, `predict/chemistry.py`, `predict/drugbank.py`, `engine/uncertainty.py`, `ml/models.py`, `ml/ensemble.py`, `validation/benchmark.py`, `validation/reference.py`, `ddi.py`, `pkpd.py`, `cli.py`

## Common Patterns

### Frozen Dataclass Contracts

```python
@dataclass(frozen=True)
class DrugOnGraph:
    name: str
    fup: Distribution
    enzyme_affinity: dict[str, Distribution]
    # ...
```
- **Where used**: All cross-layer data types in `src/sisyphus/core.py`, `src/sisyphus/graph/types.py`, `src/sisyphus/predict/adme.py`, `src/sisyphus/engine/uncertainty.py`, `src/sisyphus/validation/benchmark.py`
- **Why**: Immutability guarantees. MC sampling creates new instances via `dataclasses.replace()` rather than mutating.

### Distribution Wrapping

Every numeric parameter that participates in uncertainty propagation is a `Distribution`:
```python
fup = Distribution(mean=0.5, cv=0.4)      # uncertain
rbp = Distribution(mean=1.0, cv=0.0)      # deterministic
```
- **Where used**: All of `core.py`, `graph/types.py`, `graph/builder.py`, `predict/adme.py`, `predict/ivive.py`, `compounds.py`
- **Why**: Invariant 2. The MC engine calls `.sample(rng)` on every Distribution to produce point-value copies. Use `cv=0.0` for deterministic values, never bare floats.

### Registry Pattern

```python
FLUX_REGISTRY: dict[str, type[FluxSpec]] = {}

def register_flux(edge_type: str):
    def decorator(cls):
        FLUX_REGISTRY[edge_type] = cls
        return cls
    return decorator

@register_flux("flow")
class FlowFluxSpec(FluxSpec): ...
```
- **Where used**: `src/sisyphus/engine/flux.py`
- **Why**: Decouples edge types from the compiler. Adding a new transport mechanism means adding a new FluxSpec class -- no changes to compiler.py.

### Factory Helpers for Tests

```python
def _make_drug(**overrides) -> DrugOnGraph:
    defaults = dict(name="test", smiles="C", dose_mg=100.0, ...)
    defaults.update(overrides)
    return DrugOnGraph(**defaults)
```
- **Where used**: `tests/unit/test_flux.py`, `tests/unit/test_compiler.py`, `tests/unit/test_uncertainty.py`
- **Why**: DrugOnGraph has 18+ required fields. Factory with overrides keeps tests readable.

### Lazy Model Loading

```python
class PKPredictor:
    def __init__(self):
        self._model = None  # Loaded on first call

    def _ensure_loaded(self):
        if self._model is None:
            self._model = xgb.XGBRegressor()
            self._model.load_model(str(path))
```
- **Where used**: `src/sisyphus/ml/models.py`, `src/sisyphus/predict/adme.py` (module-level `_model_cache`)
- **Why**: Models are large (~50MB total). Only load when actually needed.

### Path Resolution from Source File

```python
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "adme"
_PHYSIOLOGY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "physiology"
```
- **Where used**: `src/sisyphus/predict/adme.py`, `src/sisyphus/ml/models.py`, `src/sisyphus/pipeline/predict.py`, `src/sisyphus/predict/chemistry.py`
- **Why**: Works regardless of working directory. Resolves relative to repository root.

### Numpy Compatibility Shim

```python
_trapz = getattr(np, "trapezoid", np.trapz)  # numpy 2.0+ vs 1.x
auc = float(_trapz(conc, time))
```
- **Where used**: `src/sisyphus/pk/nca.py`, `src/sisyphus/engine/solver.py`
- **Why**: `np.trapz` renamed to `np.trapezoid` in numpy 2.0.

### Section Dividers in Source Files

```python
# ---------------------------------------------------------------------------
# Section Name
# ---------------------------------------------------------------------------
```
- **Where used**: All major source files
- **Why**: Visual separation of logical sections within a module. Use a full 75-char divider line.

### Module-Level Docstrings with Dependency Declaration

```python
"""ODE Compiler -- graph topology -> ODE skeleton.

Imports from ``sisyphus.graph`` and ``sisyphus.core`` only.
The compiler is **identity-blind**: it never inspects node names,
enzyme names, or drug names.
"""
```
- **Where used**: Every module in `src/sisyphus/`
- **Why**: Makes layer dependencies explicit. The docstring declares what the module imports from and what invariant it upholds.

## Anti-patterns to Avoid

**Never use bare floats for physiological or drug parameters:**
```python
# WRONG
fup = 0.5
# RIGHT
fup = Distribution(mean=0.5, cv=0.0)
```
`TissueComposition` is the sole documented exception (cv sensitivity too low to warrant Distribution wrapping).

**Never inspect node/enzyme/drug names in engine code:**
```python
# WRONG (in engine/)
if node_name == "liver":
    do_liver_specific_thing()
# RIGHT
for tag, abundance in params.node_enzymes(source_name).items():
    affinity = params.drug_enzyme_affinity(tag)
```
This is Invariant 1.

**Never use `print()` in library code:**
Use `logger.info()` / `logger.warning()`. Only `cli.py` uses `print()`.

**Never use f-strings in logging calls:**
```python
# WRONG
logger.info(f"Cmax = {cmax:.4f}")
# RIGHT
logger.info("Cmax = %.4f", cmax)
```

**Never add drug-specific branches:**
```python
# WRONG
if drug.name == "warfarin":
    ...
# RIGHT -- fix the model or reference data instead
```
This is Invariant 6.

**Never import across non-adjacent layers:**
```python
# WRONG
from sisyphus.engine.solver import solve  # in predict/ module
# RIGHT -- pipeline/ is the only module that wires layers together
```

---

*Convention analysis: 2026-03-24*
