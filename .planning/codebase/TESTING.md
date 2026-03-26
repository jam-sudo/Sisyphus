# Testing Patterns

**Analysis Date:** 2026-03-24

## Test Framework & Tools

**Runner:**
- pytest >= 7.0 (dev dependency in `pyproject.toml`)
- Config: `[tool.pytest.ini_options]` in `pyproject.toml`
- Test paths: `["tests"]`
- Custom markers: `slow` for long-running benchmarks

**Assertion Library:**
- pytest native `assert` statements
- `pytest.approx(value, rel=..., abs=...)` for floating-point comparisons
- `numpy.testing.assert_array_almost_equal()` for array comparisons
- `numpy.testing.assert_array_equal()` for exact array equality
- `pytest.raises(ExceptionType, match="pattern")` for error assertions

**Run Commands:**
```bash
python3 -m pytest tests/                     # Run all tests
python3 -m pytest tests/unit/                # Unit tests only
python3 -m pytest tests/integration/         # Integration tests only
python3 -m pytest tests/benchmark/ -m slow   # Benchmark tests (slow)
python3 -m pytest tests/ -v                  # Verbose output
python3 -m pytest tests/ -k "test_compiler"  # Run specific test pattern
```

## Test File Organization

**Location:** Separate `tests/` directory (not co-located with source)

**Naming:** `test_{module_or_feature}.py`

**Structure:**
```
tests/
├── __init__.py
├── unit/                          # 20 test files, ~2800 lines
│   ├── __init__.py
│   ├── test_distribution.py       # Distribution type (core.py)
│   ├── test_body_graph.py         # BodyGraph operations (graph/body.py)
│   ├── test_builder.py            # YAML → graph (graph/builder.py)
│   ├── test_compiler.py           # ODE compiler + ResolvedParams
│   ├── test_flux.py               # All FluxSpec implementations (630 lines, largest)
│   ├── test_solver.py             # ODE solver wrapper
│   ├── test_endpoints.py          # PK endpoint extraction
│   ├── test_uncertainty.py        # MC propagation engine
│   ├── test_adme_ivive.py         # ADME prediction + IVIVE translation
│   ├── test_features_chemistry.py # Feature engineering + chemistry module
│   ├── test_ml_models.py          # PKPredictor + MetaLearner
│   ├── test_pipeline.py           # End-to-end pipeline
│   ├── test_validation.py         # Validation metrics
│   ├── test_reference.py          # Reference data loading
│   ├── test_drugbank.py           # DrugBank lookup
│   ├── test_drugbank_integration.py # DrugBank integration tests
│   ├── test_training.py           # Training data utilities
│   ├── test_ddi.py                # Drug-drug interaction module
│   └── test_pkpd.py               # PK/PD module
├── integration/                   # 2 test files, ~565 lines
│   ├── __init__.py
│   ├── test_engine_validation.py  # 4-drug validation against Omega ODE output
│   └── test_extensibility.py      # Phase 3 proof: SC, pediatric, tumor (0 engine changes)
└── benchmark/                     # 1 test file, 38 lines
    ├── __init__.py
    └── test_holdout.py            # Holdout set AAFE benchmark (marked slow)
```

## Test Categories

| Category | Location | Count | What They Test |
|----------|----------|-------|---------------|
| Unit | `tests/unit/` | ~250 tests | Individual functions and classes in isolation |
| Integration | `tests/integration/` | ~30 tests | Full pipeline: YAML -> graph -> ODE -> PK |
| Benchmark | `tests/benchmark/` | ~2 tests | Holdout AAFE against acceptance criteria |

**Total:** 298 tests collected

## Test Structure

**Suite Organization:**
```python
class TestFlowFlux:
    def test_registered(self):
        """Verify flux type is in the registry."""
        assert "flow" in FLUX_REGISTRY

    def test_mass_conservation(self):
        """Flow from A to B: total dydt sums to zero."""
        # Build minimal graph
        g = BodyGraph()
        g.add_node(Node(name="a", ...))
        g.add_node(Node(name="b", ...))
        g.add_edge(FlowEdge(source="a", target="b", ...))

        # Create drug and params
        drug = _make_drug()
        params = ResolvedParams(g, drug)

        # Exercise the flux spec
        spec = FlowFluxSpec.from_edge(0, g.edges[0], state_index)
        y = np.array([50.0, 30.0])
        dydt = np.zeros(2)
        spec.apply(0.0, y, dydt, params)

        # Assert physical invariant
        assert abs(dydt[0] + dydt[1]) < 1e-12
```

**Pattern:** Tests are organized by class per module/concept. Each test class groups related tests with descriptive docstrings. No `conftest.py` exists -- fixtures are defined as module-level helper functions within each test file.

## Fixtures and Test Helpers

**No `conftest.py`.** Helpers are defined as module-level functions in each test file.

**Factory Functions (most common pattern):**
```python
# tests/unit/test_flux.py
def _make_drug(**overrides) -> DrugOnGraph:
    """Create a minimal DrugOnGraph with sensible defaults."""
    defaults = dict(
        name="test", smiles="C", dose_mg=100.0, route="oral",
        administration_node="a", mw=100.0, pka=None,
        compound_type="neutral", fup=Distribution(0.5),
        rbp=Distribution(1.0), kp_method="provided",
        kp_overrides={}, peff=Distribution(1.0),
        solubility=Distribution(10.0), enzyme_affinity={},
        renal_clearance=Distribution(0.0), particle_radius_um=25.0,
        ps_overrides={},
    )
    defaults.update(overrides)
    return DrugOnGraph(**defaults)
```
- **Where used:** `tests/unit/test_flux.py`, `tests/unit/test_compiler.py`

**Graph Builder Helpers:**
```python
# tests/unit/test_compiler.py
def make_two_node_graph():
    g = BodyGraph()
    g.add_node(Node(name="a", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_node(Node(name="b", node_type="organ", volume=Distribution(2.0)))
    g.add_edge(FlowEdge(source="a", target="b", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="b", target="a", flow_rate=Distribution(10.0)))
    g.global_params["cardiac_output"] = Distribution(390.0)
    return g
```
- **Where used:** `tests/unit/test_compiler.py`, `tests/unit/test_uncertainty.py`

**Simulation Runner Helper (integration tests):**
```python
# tests/integration/test_extensibility.py
def _run_simulation(graph, drug, t_span=(0.0, 24.0)):
    """Compile graph, solve deterministically, return (pk, result)."""
    compiled = ODECompiler().compile(graph)
    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = drug.sample(rng)
    params = ResolvedParams(realized_graph, realized_drug)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    result = solve(compiled, params, y0, t_span=t_span, t_eval=t_eval)
    pk = compute_endpoints(result, observation_node="venous_blood")
    return pk, result
```

**Well-Known SMILES Constants:**
```python
# tests/unit/test_adme_ivive.py
_MIDAZOLAM_SMILES = "Clc1ccc2c(c1)C(=NCc3nccn3C)c1cc(F)ccc1N2"
_BENZENE_SMILES = "c1ccccc1"
_ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
```

## Mocking

**Framework:** No mocking framework is used. Tests construct real objects with minimal configurations.

**No Mocks Policy:** Tests build lightweight real graphs and drugs rather than mocking. The architecture supports this because:
- A 2-node graph with 2 edges is a valid PBPK system
- FluxSpec implementations can be tested in isolation with minimal `ResolvedParams`
- DrugOnGraph factory functions create valid drugs with any subset of overrides

**Where mocking might be expected but is absent:**
- ML model predictions: tests load the real XGBoost models from `models/` directory
- YAML parsing: tests load real YAML files from `data/`
- RDKit: tests use real molecule computations

## Coverage

**Requirements:** No coverage targets enforced. No coverage tool configured.

**Gaps:**
- No `conftest.py` with shared fixtures (each test file is self-contained)
- `src/sisyphus/graph/presets.py` -- no dedicated test file
- `src/sisyphus/pk/analytical.py` -- no dedicated test file
- `src/sisyphus/ml/registry.py` -- no dedicated test file
- `src/sisyphus/pipeline/config.py` -- no dedicated test file
- `src/sisyphus/engine/result.py` -- no dedicated test file
- No coverage for `cli.py` (only manual testing)
- CI pipeline: no `.github/workflows/` configuration exists

## Test Types

### Unit Tests (`tests/unit/`)

**Scope:** Individual functions, classes, and methods. Tests construct minimal test fixtures (2-3 node graphs, simple drugs).

**Characteristic Assertions:**
```python
# Physical invariants
assert abs(dydt[0] + dydt[1]) < 1e-12           # mass conservation
assert all(s > 0 for s in samples)                # positivity
assert 0.001 <= adme.fup.mean <= 1.0              # range clamping
assert d.std == pytest.approx(10.0)               # numerical accuracy

# Contract/type verification
assert isinstance(result, PredictionResult)
assert isinstance(adme.fup, Distribution)
with pytest.raises(ValueError, match="Invalid SMILES"):
    compute_profile("INVALID")

# Frozen dataclass enforcement
with pytest.raises(AttributeError):
    d.mean = 2.0
```

### Integration Tests (`tests/integration/`)

**Engine Validation (`test_engine_validation.py`):**
Tests 4 drugs (midazolam, caffeine, warfarin, propranolol) against Omega ODE Cmax values within +/-5%:
```python
OMEGA_TARGETS = {
    "midazolam": {"cmax": 0.006943, "tmax": 1.5},
    "caffeine": {"cmax": 1.7139, "tmax": 1.0},
    ...
}

class TestEngineValidation:
    @pytest.mark.parametrize("drug", OMEGA_TARGETS.keys())
    def test_cmax_within_5pct(self, drug):
        pk, _ = run_drug(drug)
        target = OMEGA_TARGETS[drug]["cmax"]
        rel_error = abs(actual - target) / target
        assert rel_error < 0.05
```

**Extensibility Proof (`test_extensibility.py`):**
Proves 3 extensions work with zero engine changes:
1. SC injection: add depot node + absorption edge
2. Pediatric model: load separate YAML with scaled parameters
3. Tumor compartment: add organ node with diverted blood flow

Each extension tests: solver success, mass balance < 1e-6, positive Cmax, flow conservation.

### Benchmark Tests (`tests/benchmark/`)

**Holdout Benchmark (`test_holdout.py`):**
Marked with `@pytest.mark.slow`. Runs full pipeline on holdout drugs:
```python
class TestHoldoutBenchmark:
    @pytest.mark.slow
    def test_holdout_aafe(self):
        result = run_benchmark(holdout_only=True, max_drugs=10)
        assert result.n_drugs >= 5
        assert result.aafe < 10.0
```

## Common Test Patterns

### Parametrized Validation Tests
```python
@pytest.mark.parametrize("drug", OMEGA_TARGETS.keys())
def test_cmax_within_5pct(self, drug):
    pk, _ = run_drug(drug)
    ...
```
- **Example**: `tests/integration/test_engine_validation.py:57`
- **When to use**: Testing the same assertion across multiple drugs or inputs.

### Physical Invariant Assertions
```python
# Mass conservation: dydt sums to zero
assert abs(dydt[0] + dydt[1]) < 1e-12

# Mass balance error within tolerance
assert result.mass_balance_error < 1e-6

# All concentrations non-negative
assert np.all(conc >= -1e-10)
```
- **Example**: `tests/unit/test_flux.py:84`, `tests/unit/test_solver.py:49`
- **When to use**: Any test involving ODE solutions or flux computations.

### Range Clamping Verification
```python
assert 0.001 <= adme.fup.mean <= 1.0    # fup bounds
assert 0.01 <= kp.mean <= 200.0         # Kp bounds
assert 0.1 <= adme.peff.mean <= 50.0    # Peff bounds
```
- **Example**: `tests/unit/test_adme_ivive.py:47`
- **When to use**: Testing ADME predictor outputs that should be clamped.

### Frozen Dataclass Verification
```python
def test_frozen(self):
    d = Distribution(mean=1.0)
    with pytest.raises(AttributeError):
        d.mean = 2.0
```
- **Example**: `tests/unit/test_distribution.py:33`
- **When to use**: Verifying immutability of contract types.

### Seed Reproducibility Testing
```python
def test_propagate_reproducible(self):
    """Same seed produces same results."""
    r1 = ue.propagate(compiled, graph, drug, n_samples=3, seed=42)
    r2 = ue.propagate(compiled, graph, drug, n_samples=3, seed=42)
    for sr1, sr2 in zip(r1.sim_results, r2.sim_results):
        np.testing.assert_array_almost_equal(
            sr1.concentrations["venous_blood"],
            sr2.concentrations["venous_blood"],
        )
```
- **Example**: `tests/unit/test_uncertainty.py:85`
- **When to use**: Testing MC or sampling-based code.

### Performance Bound Tests
```python
def test_performance_n100(self):
    """MC N=100 on simple graph should complete in <5s."""
    t0 = time.perf_counter()
    mc = ue.propagate_fast(compiled, graph, drug, n_samples=100, seed=42)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"MC N=100 took {elapsed:.1f}s (limit: 5s)"
```
- **Example**: `tests/unit/test_uncertainty.py:178`
- **When to use**: Guarding against performance regressions in critical paths.

### Side-Effect Import for Registry
```python
import sisyphus.engine.flux  # noqa: F401 -- register flux specs
```
- **Example**: `tests/unit/test_compiler.py:6`, `tests/unit/test_solver.py:6`
- **When to use**: Any test that compiles a graph or invokes the solver. The FluxSpec registry must be populated before compilation.

## Adding New Tests

**For a new FluxSpec implementation:**
1. Add to `tests/unit/test_flux.py`
2. Test registry presence: `assert "new_type" in FLUX_REGISTRY`
3. Test mass conservation: `abs(dydt[source] + dydt[target]) < 1e-12`
4. Test zero/degenerate inputs: zero volume, zero rate, etc.
5. Test a numerical example with hand-calculated expected values

**For a new ADME predictor:**
1. Add to `tests/unit/test_adme_ivive.py`
2. Verify output is `Distribution` with `cv > 0`
3. Verify output is in physiological range
4. Test with at least midazolam, benzene, aspirin SMILES

**For a new graph extension (Phase 3 pattern):**
1. Add to `tests/integration/test_extensibility.py`
2. Build the extended graph from `reference_man.yaml` base
3. Test: solver_success, mass_balance < 1e-6, positive Cmax, flow conservation
4. Test: physiologically meaningful comparison (e.g., SC vs oral Cmax)

---

*Testing analysis: 2026-03-24*
