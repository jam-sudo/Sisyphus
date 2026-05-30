# V3 IV-Cmax Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Alternative E from spec `docs/superpowers/specs/2026-04-22-iv-cmax-observation-design.md` — route-aware minimum observation time with adaptive `t_eval` anchor, enabling non-degenerate PI for IV bolus Cmax without affecting 107 oral holdout drugs.

**Architecture:** Add `t_min_h` parameter (default `0.0` = backward-compatible) to `solve()`, `solve_mc()`, `compute_endpoints()`, and `propagate_fast()`. Inject `t_min_h` as a guaranteed anchor point in `t_eval` for `solve()`. Cmax is extracted as `max(conc[t >= t_min_h])`, skipping the pre-distribution `t=0` spike for IV bolus. Pipeline sets `t_min_h = _IV_CMAX_DELAY_H` (= 5/60 h) when `route == "iv"`, else `0.0`.

**Tech Stack:** Python 3.10+, NumPy, SciPy `solve_ivp` (LSODA), existing frozen dataclasses (`SimResult`, `PKEndpoints`).

---

## File Structure

**Modify:**
- `src/sisyphus/engine/solver.py` — add `t_min_h` to `solve()` and `solve_mc()`; define `_IV_CMAX_DELAY_H = 5.0/60.0`; inject `t_min_h` into `t_eval`; windowed-max extraction
- `src/sisyphus/pk/endpoints.py` — add `t_min_h` to `compute_endpoints()`; windowed-max extraction
- `src/sisyphus/engine/uncertainty.py` — add `t_min_h` to `propagate_fast()`; thread to `solve_mc()`
- `src/sisyphus/pipeline/predict.py` — route-conditional `t_min_h` for engine `solve()` + `propagate_fast()` + `compute_endpoints()`

**Create:**
- `tests/unit/test_solver_iv_cmax.py` — unit tests for V3 windowed max in `solve()` and `solve_mc()`
- `tests/unit/test_endpoints_iv_cmax.py` — unit tests for windowed `compute_endpoints()`
- `tests/integration/test_pipeline_iv_cmax.py` — IV drug regression check (variance propagation) + oral drug unchanged check

**Do NOT modify:**
- `data/physiology/reference_man.yaml` (Alternative C explicitly rejected in spec §5)
- Any file under `data/reference/`, `data/validation/`, `models/`
- `scripts/validate_oatp_generalization.py` (re-run comes AFTER V3 lands, in a separate non-code task)

---

### Task 1: Add `t_min_h` to `solver.solve()` with anchored `t_eval`

**Files:**
- Modify: `src/sisyphus/engine/solver.py`
- Create: `tests/unit/test_solver_iv_cmax.py`

**Scope:** Extend `solve()` (currently line 16) with a `t_min_h: float = 0.0` keyword argument. When `t_min_h > 0` and `t_eval is None`, build `t_eval` as `np.concatenate([[0.0, t_min_h], np.linspace(t_min_h, t_span[1], 498)])`. Cmax is computed elsewhere (`compute_endpoints` in Task 2); `solve()` only changes the time grid. Define module-level constant `_IV_CMAX_DELAY_H = 5.0 / 60.0  # 5 min post-bolus, clinical first-draw convention`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_solver_iv_cmax.py`:

```python
"""V3: verify solve() injects t_min_h into t_eval when provided."""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import _IV_CMAX_DELAY_H, solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph

_PHYS = "data/physiology/reference_man.yaml"


def _setup(route: str = "iv"):
    graph = build_from_yaml(_PHYS)
    profile = compute_profile("CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1")  # valsartan
    adme = predict_adme(profile)
    drug = build_drug_on_graph(profile, adme, dose_mg=20.0, route=route)
    compiler = ODECompiler()
    compiled = compiler.compile(graph)
    rng = np.random.default_rng(42)
    params = ResolvedParams(graph.sample(rng), drug.sample(rng))
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    return compiled, params, y0


def test_iv_cmax_delay_constant_is_five_minutes():
    assert _IV_CMAX_DELAY_H == pytest.approx(5.0 / 60.0)


def test_solve_t_min_h_injects_anchor_point():
    compiled, params, y0 = _setup("iv")
    result = solve(compiled, params, y0, t_span=(0.0, 24.0), t_min_h=_IV_CMAX_DELAY_H)
    # t_min_h must appear as an exact time point in the grid.
    assert np.any(np.isclose(result.time_h, _IV_CMAX_DELAY_H))


def test_solve_t_min_h_zero_is_backward_compatible():
    compiled, params, y0 = _setup("oral")
    result_default = solve(compiled, params, y0, t_span=(0.0, 24.0))
    result_zero = solve(compiled, params, y0, t_span=(0.0, 24.0), t_min_h=0.0)
    np.testing.assert_allclose(result_default.time_h, result_zero.time_h)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_solver_iv_cmax.py -v`
Expected: FAIL — `_IV_CMAX_DELAY_H` not defined and `solve()` does not accept `t_min_h`.

- [ ] **Step 3: Implement the changes in `solver.py`**

Edit `src/sisyphus/engine/solver.py`:

1. After the imports (after line 13), add the module constant:

```python
# Clinical first-draw convention for IV bolus Cmax (5 min post-injection).
# Used by route-aware Cmax extraction to skip the deterministic t=0 spike
# (see docs/superpowers/specs/2026-04-22-iv-cmax-observation-design.md §5).
_IV_CMAX_DELAY_H = 5.0 / 60.0
```

2. Modify the `solve()` signature to add `t_min_h`:

```python
def solve(
    compiled: CompiledODE,
    params: ResolvedParams,
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
    t_min_h: float = 0.0,
) -> SimResult:
```

3. Replace the `t_eval` construction block (lines 38-39) with:

```python
    if t_eval is None:
        if t_min_h > 0.0:
            t_eval = np.concatenate(
                [[0.0, t_min_h], np.linspace(t_min_h, t_span[1], 498)]
            )
        else:
            t_eval = np.linspace(t_span[0], t_span[1], 500)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_solver_iv_cmax.py -v`
Expected: 3/3 PASS.

- [ ] **Step 5: Regression — existing solver tests still pass**

Run: `pytest tests/unit/test_solver.py tests/unit/test_endpoints.py tests/unit/test_uncertainty.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/engine/solver.py tests/unit/test_solver_iv_cmax.py
git commit -m "feat(engine): add t_min_h anchor to solve() for V3 IV-Cmax"
```

---

### Task 2: Add `t_min_h` to `solve_mc()` and windowed-max extraction

**Files:**
- Modify: `src/sisyphus/engine/solver.py`
- Modify: `tests/unit/test_solver_iv_cmax.py`

**Scope:** Extend `solve_mc()` (currently line 88) with `t_min_h: float = 0.0`. For `t_min_h > 0`, pass `t_eval=np.array([0.0, t_min_h] + np.linspace(t_min_h, t_span[1], 98).tolist())` (100 points — fewer than `solve()` because MC is speed-critical) to `solve_ivp`. Extract Cmax via `mask = sol.t >= t_min_h; cmax = float(np.max(conc[mask]))`. Tmax uses the same mask: `tmax = float(sol.t[mask][np.argmax(conc[mask])])`. AUC is unchanged (full trapezoid over `sol.t`, `conc`).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_solver_iv_cmax.py`:

```python
from sisyphus.engine.solver import solve_mc


def test_solve_mc_iv_cmax_excludes_t0_spike():
    """Zero-width PI root cause: IV Cmax at t=0 = dose/V_venous (deterministic).
    With t_min_h > 0, the windowed max must be < the t=0 value."""
    compiled, params, y0 = _setup("iv")
    # Unfiltered (V2 behavior): Cmax should equal dose/V_venous = 20/3.7 ≈ 5.405
    cmax_v2, _, _, ok_v2 = solve_mc(compiled, params, y0, t_span=(0.0, 24.0))
    assert ok_v2
    assert cmax_v2 == pytest.approx(20.0 / 3.7, rel=1e-3)
    # Windowed (V3 behavior): Cmax must be strictly less than t=0 value.
    cmax_v3, tmax_v3, _, ok_v3 = solve_mc(
        compiled, params, y0, t_span=(0.0, 24.0), t_min_h=_IV_CMAX_DELAY_H
    )
    assert ok_v3
    assert cmax_v3 < cmax_v2
    assert tmax_v3 >= _IV_CMAX_DELAY_H


def test_solve_mc_t_min_h_zero_is_backward_compatible():
    compiled, params, y0 = _setup("oral")
    cmax_default, tmax_default, auc_default, ok_d = solve_mc(
        compiled, params, y0, t_span=(0.0, 24.0)
    )
    cmax_zero, tmax_zero, auc_zero, ok_z = solve_mc(
        compiled, params, y0, t_span=(0.0, 24.0), t_min_h=0.0
    )
    assert ok_d and ok_z
    assert cmax_default == pytest.approx(cmax_zero, rel=1e-6)
    assert tmax_default == pytest.approx(tmax_zero, rel=1e-6)
    assert auc_default == pytest.approx(auc_zero, rel=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_solver_iv_cmax.py::test_solve_mc_iv_cmax_excludes_t0_spike tests/unit/test_solver_iv_cmax.py::test_solve_mc_t_min_h_zero_is_backward_compatible -v`
Expected: FAIL — `solve_mc` does not accept `t_min_h`.

- [ ] **Step 3: Implement the changes**

Edit `src/sisyphus/engine/solver.py`:

1. Modify `solve_mc()` signature:

```python
def solve_mc(
    compiled: CompiledODE,
    params: ResolvedParams,
    y0: np.ndarray,
    t_span: tuple[float, float],
    observation_node: str = "venous_blood",
    t_min_h: float = 0.0,
) -> tuple[float, float, float, bool]:
```

2. Add `t_eval` construction before the `solve_ivp` call (replace the call starting at line 114):

```python
    if t_min_h > 0.0:
        t_eval = np.concatenate(
            [[0.0, t_min_h], np.linspace(t_min_h, t_span[1], 98)]
        )
    else:
        t_eval = None  # adaptive grid (V2 behavior)

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-6,
    )
```

3. Replace the Cmax/Tmax extraction (lines 140-141) with masked version:

```python
    if t_min_h > 0.0:
        mask = sol.t >= t_min_h
        if not np.any(mask):
            return 0.0, 0.0, 0.0, False
        conc_window = conc[mask]
        t_window = sol.t[mask]
        cmax = float(np.max(conc_window))
        tmax = float(t_window[np.argmax(conc_window)])
    else:
        cmax = float(np.max(conc))
        tmax = float(sol.t[np.argmax(conc)])
```

(AUC line unchanged — full-interval trapezoid is still correct.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_solver_iv_cmax.py -v`
Expected: 5/5 PASS.

- [ ] **Step 5: Regression check**

Run: `pytest tests/unit/test_solver.py tests/unit/test_uncertainty.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/engine/solver.py tests/unit/test_solver_iv_cmax.py
git commit -m "feat(engine): windowed Cmax in solve_mc() for V3 IV-Cmax"
```

---

### Task 3: Add `t_min_h` to `compute_endpoints()`

**Files:**
- Modify: `src/sisyphus/pk/endpoints.py`
- Create: `tests/unit/test_endpoints_iv_cmax.py`

**Scope:** Extend `compute_endpoints()` signature with `t_min_h: float = 0.0`. Replace `cmax = float(np.max(conc))` with the same mask pattern as `solve_mc()`. AUC and terminal half-life remain unchanged (they operate on the full time series).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_endpoints_iv_cmax.py`:

```python
"""V3: verify compute_endpoints() windowed Cmax extraction."""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.core import SimResult
from sisyphus.pk.endpoints import compute_endpoints


def _build_sim_result() -> SimResult:
    # IV bolus-like profile: max at t=0, monotonic decline.
    time = np.linspace(0.0, 24.0, 500)
    conc = 5.405 * np.exp(-0.1 * time)  # decays from 5.405 at t=0
    conc[time < 0.083] = 5.405  # plateau at the very-early t
    return SimResult(
        time_h=time,
        concentrations={"venous_blood": conc},
        amounts={"venous_blood": conc * 3.7},
        mass_balance_error=0.0,
        solver_success=True,
    )


def test_compute_endpoints_default_picks_t0():
    result = _build_sim_result()
    pk = compute_endpoints(result)
    assert pk.cmax.mean == pytest.approx(5.405, rel=1e-3)
    assert pk.tmax.mean == pytest.approx(0.0, abs=1e-3)


def test_compute_endpoints_windowed_skips_t0():
    result = _build_sim_result()
    pk = compute_endpoints(result, t_min_h=5.0 / 60.0)
    # At t = 0.083h, conc = 5.405 * exp(-0.1 * 0.083) ≈ 5.360
    assert pk.cmax.mean < 5.405
    assert pk.cmax.mean == pytest.approx(5.405 * np.exp(-0.1 * 5.0 / 60.0), rel=1e-2)
    assert pk.tmax.mean >= 5.0 / 60.0


def test_compute_endpoints_t_min_h_zero_is_backward_compatible():
    result = _build_sim_result()
    pk_default = compute_endpoints(result)
    pk_zero = compute_endpoints(result, t_min_h=0.0)
    assert pk_default.cmax.mean == pytest.approx(pk_zero.cmax.mean)
    assert pk_default.tmax.mean == pytest.approx(pk_zero.tmax.mean)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_endpoints_iv_cmax.py -v`
Expected: FAIL — `compute_endpoints` does not accept `t_min_h`.

- [ ] **Step 3: Implement changes**

Edit `src/sisyphus/pk/endpoints.py` — replace entire `compute_endpoints` function body:

```python
def compute_endpoints(
    result: SimResult,
    observation_node: str = "venous_blood",
    t_min_h: float = 0.0,
) -> PKEndpoints:
    """Extract PK endpoints from a SimResult.

    Args:
        result: Raw ODE simulation output.
        observation_node: Node to use for plasma concentrations.
        t_min_h: Minimum time for Cmax extraction (skips t < t_min_h). Used
            for IV bolus to avoid the deterministic t=0 spike; default 0.0
            (V2-compatible). See
            docs/superpowers/specs/2026-04-22-iv-cmax-observation-design.md.

    Returns:
        PKEndpoints with Cmax, Tmax, AUC, t½.
    """
    conc = result.concentrations[observation_node]
    time = result.time_h

    if t_min_h > 0.0:
        mask = time >= t_min_h
        if not np.any(mask):
            cmax = 0.0
            tmax = 0.0
        else:
            conc_window = conc[mask]
            time_window = time[mask]
            cmax = float(np.max(conc_window))
            tmax = float(time_window[np.argmax(conc_window)])
    else:
        cmax = float(np.max(conc))
        tmax = float(time[np.argmax(conc)])

    auc = auc_trapezoidal(time, conc)
    t_half = terminal_half_life(time, conc)

    return PKEndpoints(
        cmax=Distribution(cmax),
        tmax=Distribution(tmax),
        auc_0t=Distribution(auc),
        t_half=Distribution(t_half) if t_half is not None else None,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_endpoints_iv_cmax.py tests/unit/test_endpoints.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/pk/endpoints.py tests/unit/test_endpoints_iv_cmax.py
git commit -m "feat(pk): windowed Cmax in compute_endpoints() for V3 IV-Cmax"
```

---

### Task 4: Thread `t_min_h` through `UncertaintyEngine.propagate_fast()`

**Files:**
- Modify: `src/sisyphus/engine/uncertainty.py`

**Scope:** Extend `propagate_fast()` signature with `t_min_h: float = 0.0`. Pass it to `solve_mc()` in the scipy-backend loop (line 274). JAX and surrogate backends are out-of-scope — they are not used by the ECM generalization test and raise no regression risk for current IV work. Document this restriction in the docstring.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_solver_iv_cmax.py`:

```python
from sisyphus.engine.compiler import ODECompiler
from sisyphus.engine.uncertainty import UncertaintyEngine


def test_propagate_fast_t_min_h_matches_solve_mc():
    """propagate_fast with t_min_h must produce non-degenerate, decreased Cmax
    median vs the V2 (t_min_h=0) path for an IV bolus."""
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    graph = build_from_yaml("data/physiology/reference_man.yaml")
    profile = compute_profile("CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1")
    adme = predict_adme(profile)
    drug = build_drug_on_graph(profile, adme, dose_mg=20.0, route="iv")
    compiler = ODECompiler()
    compiled = compiler.compile(graph)

    ue = UncertaintyEngine()
    mc_v2 = ue.propagate_fast(
        compiled=compiled, graph=graph, drug=drug,
        n_samples=50, seed=42, t_span=(0.0, 24.0),
    )
    mc_v3 = ue.propagate_fast(
        compiled=compiled, graph=graph, drug=drug,
        n_samples=50, seed=42, t_span=(0.0, 24.0),
        t_min_h=_IV_CMAX_DELAY_H,
    )
    # V3 must yield strictly smaller median and non-degenerate PI (low < high).
    assert mc_v3.pk.cmax.mean < mc_v2.pk.cmax.mean
    low_v3, high_v3 = mc_v3.cmax_90ci
    assert high_v3 > low_v3  # non-degenerate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_solver_iv_cmax.py::test_propagate_fast_t_min_h_matches_solve_mc -v`
Expected: FAIL — `propagate_fast` does not accept `t_min_h`.

- [ ] **Step 3: Implement changes**

Edit `src/sisyphus/engine/uncertainty.py`:

1. Modify the `propagate_fast()` signature (at line 167):

```python
    def propagate_fast(
        self,
        compiled: CompiledODE,
        graph: BodyGraph,
        drug: DrugOnGraph,
        n_samples: int = 1000,
        seed: int = 42,
        t_span: tuple[float, float] = (0.0, 24.0),
        observation_node: str = "venous_blood",
        backend: str = "scipy",
        t_min_h: float = 0.0,
    ) -> MCResult:
```

2. In the scipy-backend loop (around line 274), update the `solve_mc` call:

```python
                cmax, tmax, auc, success = solve_mc(
                    compiled,
                    params,
                    y0_template,
                    t_span,
                    observation_node,
                    t_min_h=t_min_h,
                )
```

3. Append to the docstring (after line 194):

```
            t_min_h: Minimum time for Cmax extraction (skips t < t_min_h).
                Used for IV bolus to avoid the deterministic t=0 spike.
                scipy backend only — JAX/surrogate backends ignore this.
                Default 0.0 (V2-compatible).
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_solver_iv_cmax.py tests/unit/test_uncertainty.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/engine/uncertainty.py tests/unit/test_solver_iv_cmax.py
git commit -m "feat(engine): thread t_min_h through propagate_fast() scipy backend"
```

---

### Task 5: Route-conditional wiring in `pipeline/predict.py`

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py`
- Create: `tests/integration/test_pipeline_iv_cmax.py`

**Scope:** In `pipeline/predict.py`, when `route == "iv"`, pass `t_min_h=_IV_CMAX_DELAY_H` to the three engine entry points: `solve()` (line 121), `compute_endpoints()` (line 123), and `propagate_fast()` (line 141). For non-IV routes, behavior is unchanged (`t_min_h` defaults to 0.0). Import `_IV_CMAX_DELAY_H` from `sisyphus.engine.solver`.

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_pipeline_iv_cmax.py`:

```python
"""V3 integration: verify pipeline route-conditional Cmax behavior.

IV drugs: windowed Cmax (V3) produces Cmax strictly less than V2 t=0 spike.
Oral drugs: behavior unchanged from V2 (regression guard).
"""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.pipeline.predict import predict


# SMILES + doses chosen to avoid network / DB access. Valsartan 20 mg IV
# (ECM generalization test substrate). Atorvastatin 20 mg oral (in holdout,
# represents the oral regression path).
_VALSARTAN_SMILES = "CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"
_ATORVASTATIN_SMILES = (
    "CC(C)c1c(/C=C/[C@H](O)C[C@H](O)CC(=O)O)n(CCc2ccc(F)cc2)c(-c2ccccc2)c1-c1ccc(F)cc1"
)


def test_iv_pipeline_cmax_uses_windowed_max():
    """For IV bolus, engine Cmax must be strictly less than dose/V_venous."""
    result = predict(
        smiles=_VALSARTAN_SMILES, dose_mg=20.0, route="iv",
        n_mc_samples=50,
    )
    assert result.engine_pk is not None
    # V2 Cmax would be 20/3.7 ≈ 5.405. V3 windowed must be less.
    assert result.engine_pk.cmax.mean < 20.0 / 3.7


def test_iv_pipeline_90pi_is_non_degenerate():
    """For IV bolus with MC, PI must have positive width (non-degenerate)."""
    result = predict(
        smiles=_VALSARTAN_SMILES, dose_mg=20.0, route="iv",
        n_mc_samples=100,
    )
    assert result.cmax_90ci is not None
    low, high = result.cmax_90ci
    assert high > low  # non-degenerate


def test_oral_pipeline_unchanged_by_v3():
    """Oral drugs must see no V3 behavior change — Tmax > 5 min trivially."""
    result = predict(
        smiles=_ATORVASTATIN_SMILES, dose_mg=20.0, route="oral",
        n_mc_samples=0,
    )
    assert result.engine_pk is not None
    # Oral Tmax must be well above the IV threshold (absorption takes hours).
    assert result.engine_pk.tmax.mean > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_pipeline_iv_cmax.py -v`
Expected: `test_iv_pipeline_90pi_is_non_degenerate` FAILS (degenerate PI from V2 behavior); other tests may pass incidentally.

- [ ] **Step 3: Implement changes**

Edit `src/sisyphus/pipeline/predict.py`:

1. Add import near the top (next to other engine imports):

```python
from sisyphus.engine.solver import _IV_CMAX_DELAY_H, solve
```

2. Wrap the existing three call sites with route-conditional `t_min_h`. Just before line 121 (the `solve()` call), add:

```python
        t_min_h = _IV_CMAX_DELAY_H if route == "iv" else 0.0
```

3. Update the `solve()` call (line 121):

```python
        sim_result = solve(compiled, params, y0, t_span=(0, 24), t_min_h=t_min_h)
```

4. Update the `compute_endpoints()` call (line 123):

```python
            engine_pk = compute_endpoints(sim_result, t_min_h=t_min_h)
```

5. Update the `propagate_fast()` call (line 141):

```python
            mc = ue.propagate_fast(
                compiled, graph, drug, n_samples=n_mc_samples, t_min_h=t_min_h
            )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/test_pipeline_iv_cmax.py tests/unit/test_solver_iv_cmax.py tests/unit/test_endpoints_iv_cmax.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/pipeline/predict.py tests/integration/test_pipeline_iv_cmax.py
git commit -m "feat(pipeline): route-conditional t_min_h for V3 IV-Cmax"
```

---

### Task 6: Oral-holdout regression guard

**Files:**
- Create: `tests/integration/test_v3_oral_regression.py`

**Scope:** Confirm that for a representative oral drug, the V3 pipeline produces byte-identical results to the V2 behavior. Because `route == "oral"` forces `t_min_h = 0.0`, the entire code path is structurally backward-compatible. This test locks that invariant in place so future refactors cannot break it.

- [ ] **Step 1: Write the test**

Create `tests/integration/test_v3_oral_regression.py`:

```python
"""V3 regression guard: oral drugs must produce identical results to V2.

V3 is route-conditional — when route == "oral", t_min_h = 0.0 and the engine
code path is structurally unchanged. This test pins that invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve, solve_mc
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph


def _setup_oral(dose_mg: float = 20.0):
    graph = build_from_yaml("data/physiology/reference_man.yaml")
    # Caffeine — well-behaved oral reference.
    profile = compute_profile("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    adme = predict_adme(profile)
    drug = build_drug_on_graph(profile, adme, dose_mg=dose_mg, route="oral")
    compiler = ODECompiler()
    compiled = compiler.compile(graph)
    rng = np.random.default_rng(42)
    params = ResolvedParams(graph.sample(rng), drug.sample(rng))
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    return compiled, params, y0


def test_oral_solve_v3_equals_v2():
    compiled, params, y0 = _setup_oral()
    sim_v2 = solve(compiled, params, y0, t_span=(0.0, 24.0))
    sim_v3 = solve(compiled, params, y0, t_span=(0.0, 24.0), t_min_h=0.0)
    np.testing.assert_allclose(sim_v2.time_h, sim_v3.time_h)
    np.testing.assert_allclose(
        sim_v2.concentrations["venous_blood"],
        sim_v3.concentrations["venous_blood"],
    )


def test_oral_solve_mc_v3_equals_v2():
    compiled, params, y0 = _setup_oral()
    cmax_v2, tmax_v2, auc_v2, ok_v2 = solve_mc(compiled, params, y0, t_span=(0.0, 24.0))
    cmax_v3, tmax_v3, auc_v3, ok_v3 = solve_mc(
        compiled, params, y0, t_span=(0.0, 24.0), t_min_h=0.0
    )
    assert ok_v2 and ok_v3
    assert cmax_v2 == pytest.approx(cmax_v3, rel=1e-6)
    assert tmax_v2 == pytest.approx(tmax_v3, rel=1e-6)
    assert auc_v2 == pytest.approx(auc_v3, rel=1e-6)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/integration/test_v3_oral_regression.py -v`
Expected: 2/2 PASS (trivially — `t_min_h=0.0` routes through the unchanged code path).

- [ ] **Step 3: Full regression sweep**

Run: `pytest tests/ -x --timeout=300`
Expected: all PASS (or pre-existing skips/xfails only — no new failures).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_v3_oral_regression.py
git commit -m "test(v3): oral-route regression guard for t_min_h=0 compatibility"
```

---

## Post-Implementation (NOT part of this plan — separate task)

After all 6 tasks merge, the ECM generalization test re-run under V3 is a separate, non-code execution:

1. Delete or archive `data/validation/oatp_generalization_result.json` → move to `data/validation/oatp_generalization_result.v2.json`.
2. Run `python scripts/validate_oatp_generalization.py` (pipeline now route-conditional, no code change required).
3. New result path: write a V3-tagged result file (e.g., `oatp_generalization_result_v3.json`) — modify `_OUT` in the script or add a V3 env switch in a separate micro-PR.
4. Freeze result.

This is tracked in task #124 in the parent session task list, not in this plan.
