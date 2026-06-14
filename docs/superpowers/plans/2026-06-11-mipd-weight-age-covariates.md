# MIPD Weight/Age Covariates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Individualize the engine prior for body weight and age by swapping the reference graph for `sbi.physiology_generator.generate_physiology(weight, age)` at the shared `_build_grid_engine` point — for both the oral CL-grid and the IV steady-state TDM paths. Renal stays CrCl-only.

**Architecture:** `Covariates` gains `body_weight_kg`/`age_years`. `_build_grid_engine` builds the graph via `generate_physiology` when either is supplied (else `build_from_yaml`). Both grid builders thread the params through; both entry points expose them via `Covariates`. `renal_factor()` is unchanged (CrCl-only).

**Tech Stack:** Python 3.10+, numpy, pytest, ruff. Reference spec: `docs/superpowers/specs/2026-06-11-mipd-weight-age-covariates-design.md`. Reused: `sbi.physiology_generator.generate_physiology`.

---

## File Structure

- **MODIFY** `src/sisyphus/mipd/covariates.py` — `body_weight_kg`/`age_years` fields, validation, `has_physiology()`, `warnings()` (consolidates the extreme-covariate flags).
- **MODIFY** `src/sisyphus/mipd/grid.py` — `_build_grid_engine` `generate_physiology` swap; `build_cl_grid` threads weight/age.
- **MODIFY** `src/sisyphus/mipd/renal_grid.py` — `build_renal_cl_grid` threads weight/age.
- **MODIFY** `src/sisyphus/mipd/api.py` — `predict_posterior` dispatch (broaden `individualized`, thread weight/age, use `covariates.warnings()`).
- **MODIFY** `src/sisyphus/mipd/tdm.py` — `predict_tdm` threads weight/age, uses `covariates.warnings()`.

---

## Task 1: `Covariates` — weight/age fields + `has_physiology()` + `warnings()`

**Files:** Modify `src/sisyphus/mipd/covariates.py`. Test: `tests/unit/test_mipd_covariates.py`.

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/test_mipd_covariates.py`:

```python
def test_covariates_weight_age_fields_and_validation():
    import pytest
    from sisyphus.mipd.covariates import Covariates
    c = Covariates(body_weight_kg=10.0, age_years=2.0)
    assert c.body_weight_kg == 10.0 and c.age_years == 2.0
    with pytest.raises(ValueError):
        Covariates(body_weight_kg=0.0)
    with pytest.raises(ValueError):
        Covariates(age_years=-1.0)


def test_covariates_has_physiology():
    from sisyphus.mipd.covariates import Covariates
    assert Covariates().has_physiology() is False
    assert Covariates(crcl_ml_min=50).has_physiology() is False  # CrCl is not physiology
    assert Covariates(body_weight_kg=10).has_physiology() is True
    assert Covariates(age_years=80).has_physiology() is True


def test_covariates_renal_factor_unaffected_by_weight_age():
    from sisyphus.mipd.covariates import Covariates
    # renal is CrCl-only — weight/age never change renal_factor
    assert Covariates(body_weight_kg=10, age_years=80).renal_factor() == 1.0
    assert Covariates(crcl_ml_min=62.5, body_weight_kg=10, age_years=80).renal_factor() == 0.5


def test_covariates_warnings_flags_extremes():
    from sisyphus.mipd.covariates import Covariates
    assert Covariates().warnings() == ()
    assert Covariates(crcl_ml_min=90, body_weight_kg=70, age_years=30).warnings() == ()
    assert any("crcl" in w.lower() for w in Covariates(crcl_ml_min=3).warnings())
    assert any("weight" in w.lower() for w in Covariates(body_weight_kg=1.0).warnings())
    assert any("age" in w.lower() for w in Covariates(age_years=120).warnings())
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/unit/test_mipd_covariates.py -k "weight_age or has_physiology or warnings or unaffected" -v` → FAIL (`TypeError: ... unexpected keyword argument 'body_weight_kg'`).

- [ ] **Step 3: Edit `src/sisyphus/mipd/covariates.py`.** Replace the `Covariates` class body (currently fields `crcl_ml_min`, `__post_init__`, `renal_factor`) with:

```python
    crcl_ml_min: float | None = None
    body_weight_kg: float | None = None
    age_years: float | None = None

    def __post_init__(self) -> None:
        if self.crcl_ml_min is not None and self.crcl_ml_min <= 0:
            raise ValueError(f"crcl_ml_min must be > 0, got {self.crcl_ml_min}")
        if self.body_weight_kg is not None and self.body_weight_kg <= 0:
            raise ValueError(f"body_weight_kg must be > 0, got {self.body_weight_kg}")
        if self.age_years is not None and self.age_years <= 0:
            raise ValueError(f"age_years must be > 0, got {self.age_years}")

    def renal_factor(self) -> float:
        """Multiplicative scale for ``drug.renal_clearance`` (CrCl-only; 1.0 at CrCl=125).

        Weight/age never affect this — renal individualization is measured-CrCl-only
        (estimating GFR from age/weight is deferred; see the design spec).
        """
        if self.crcl_ml_min is None:
            return 1.0
        return self.crcl_ml_min / _REFERENCE_GFR_ML_MIN

    def has_physiology(self) -> bool:
        """True iff weight or age is set — triggers generate_physiology graph build."""
        return self.body_weight_kg is not None or self.age_years is not None

    def warnings(self) -> tuple[str, ...]:
        """Structured flags for physiologically extreme covariates (extrapolation risk)."""
        w: list[str] = []
        if self.crcl_ml_min is not None and not (5.0 <= self.crcl_ml_min <= 200.0):
            w.append(
                f"crcl:extreme:{self.crcl_ml_min}: the engine renal model is "
                "glomerular-filtration-only and least reliable outside [5, 200] mL/min"
            )
        if self.body_weight_kg is not None and not (2.0 <= self.body_weight_kg <= 250.0):
            w.append(
                f"weight:extreme:{self.body_weight_kg}: allometric/ontogeny scaling "
                "extrapolates poorly outside [2, 250] kg"
            )
        if self.age_years is not None and not (0.0 < self.age_years <= 100.0):
            w.append(
                f"age:extreme:{self.age_years}: ontogeny/aging scaling extrapolates "
                "poorly outside (0, 100] yr"
            )
        return tuple(w)
```

(Update the class docstring's `Attributes:` to mention `body_weight_kg`/`age_years` if present; otherwise leave prose as is.)

- [ ] **Step 4: Run tests** — `python -m pytest tests/unit/test_mipd_covariates.py -v` → PASS (all, incl. the existing CrCl tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/covariates.py tests/unit/test_mipd_covariates.py
git commit -m "feat(mipd): Covariates weight/age fields + has_physiology + warnings"
```

---

## Task 2: `_build_grid_engine` generate_physiology swap + `build_cl_grid` threading

**Files:** Modify `src/sisyphus/mipd/grid.py`. Test: `tests/unit/test_mipd_grid.py`.

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/test_mipd_grid.py` (the file already defines `MIDAZOLAM`/`DOSE`):

```python
def test_build_cl_grid_weight_age_individualizes_and_solves():
    # generate_physiology path must run augment->expand->compile->solve end-to-end
    # (this path has no existing caller — proven here).
    g = build_cl_grid(MIDAZOLAM, DOSE, n_grid=3, s_range=(0.5, 2.0),
                      body_weight_kg=40.0, age_years=70.0)
    import numpy as np
    assert g.cmax.shape == (3,)
    assert np.all(np.isfinite(g.cmax)) and np.all(g.cmax > 0)


def test_build_cl_grid_no_weight_age_is_unchanged():
    import numpy as np
    a = build_cl_grid(MIDAZOLAM, DOSE, n_grid=3, s_range=(0.5, 2.0))
    b = build_cl_grid(MIDAZOLAM, DOSE, n_grid=3, s_range=(0.5, 2.0),
                      body_weight_kg=None, age_years=None)
    assert np.array_equal(a.cmax, b.cmax) and np.array_equal(a.auc, b.auc)


def test_build_cl_grid_lower_weight_higher_exposure():
    # fixed dose, lower body weight -> lower absolute clearance -> higher AUC
    ref = build_cl_grid(MIDAZOLAM, DOSE, n_grid=3, s_range=(0.5, 2.0))            # 70 kg
    light = build_cl_grid(MIDAZOLAM, DOSE, n_grid=3, s_range=(0.5, 2.0),
                          body_weight_kg=40.0)
    assert light.auc[1] > ref.auc[1]  # s=1.0 (middle of geomspace(0.5,2.0,3))
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/unit/test_mipd_grid.py -k "weight" -v` → FAIL (`TypeError: build_cl_grid() got an unexpected keyword argument 'body_weight_kg'`).

- [ ] **Step 3: Edit `_build_grid_engine`** in `src/sisyphus/mipd/grid.py`. Change the signature to add the two params:

```python
def _build_grid_engine(
    smiles: str,
    dose_mg: float,
    route: str,
    renal_factor: float,
    kp_method: str,
    body_weight_kg: float | None = None,
    age_years: float | None = None,
):
```

Then replace the single line `graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")` with:

```python
    if body_weight_kg is not None or age_years is not None:
        # Weight/age covariate individualization: scale the reference graph
        # (volumes, flows, enzyme ontogeny/aging) via the verified drop-in. Pass
        # base_yaml ABSOLUTE — generate_physiology's default is CWD-relative.
        from sisyphus.sbi.physiology_generator import generate_physiology
        graph = generate_physiology(
            body_weight_kg if body_weight_kg is not None else 70.0,
            age_years if age_years is not None else 30.0,
            base_yaml=_PHYSIOLOGY_DIR / "reference_man.yaml",
        )
    else:
        graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")
```

- [ ] **Step 4: Edit `build_cl_grid`** in `src/sisyphus/mipd/grid.py`. Add the two params to its signature (after `renal_factor: float = 1.0,`):

```python
    renal_factor: float = 1.0,
    body_weight_kg: float | None = None,
    age_years: float | None = None,
) -> CLGrid:
```

And change its `_build_grid_engine` call from `_build_grid_engine(smiles, dose_mg, route, renal_factor, kp_method)` to:

```python
    compiled, realized_graph, drug, obs_node = _build_grid_engine(
        smiles, dose_mg, route, renal_factor, kp_method, body_weight_kg, age_years
    )
```

- [ ] **Step 5: Run tests** — `python -m pytest tests/unit/test_mipd_grid.py -v` → PASS (all, incl. the existing faithfulness pin and renal_factor tests). If `test_build_cl_grid_lower_weight_higher_exposure` fails, STOP and report `ref.auc` and `light.auc` (a lighter patient at fixed dose MUST have higher AUC); do not loosen.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/mipd/grid.py tests/unit/test_mipd_grid.py
git commit -m "feat(mipd): _build_grid_engine weight/age via generate_physiology"
```

---

## Task 3: `build_renal_cl_grid` threading (IV path)

**Files:** Modify `src/sisyphus/mipd/renal_grid.py`. Test: `tests/unit/test_mipd_renal_grid.py` (defines `ATENOLOL` + `_iv_regimen`).

- [ ] **Step 1: Write the failing test** — append to `tests/unit/test_mipd_renal_grid.py`:

```python
def test_build_renal_cl_grid_weight_age_individualizes_and_solves():
    # IV solve_regimen + generate_physiology graph must run end-to-end.
    from sisyphus.mipd.renal_grid import build_renal_cl_grid
    g = build_renal_cl_grid(ATENOLOL, _iv_regimen(), n_grid=3, r_range=(0.5, 2.0),
                            body_weight_kg=50.0, age_years=75.0)
    assert g.cmax.shape == (3,)
    assert np.all(np.isfinite(g.cmax)) and np.all(g.cmax > 0)


def test_build_renal_cl_grid_no_weight_age_unchanged():
    from sisyphus.mipd.renal_grid import build_renal_cl_grid
    a = build_renal_cl_grid(ATENOLOL, _iv_regimen(), n_grid=3, r_range=(0.5, 2.0))
    b = build_renal_cl_grid(ATENOLOL, _iv_regimen(), n_grid=3, r_range=(0.5, 2.0),
                            body_weight_kg=None, age_years=None)
    assert np.array_equal(a.cmax, b.cmax)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/unit/test_mipd_renal_grid.py -k "weight_age" -v` → FAIL (`TypeError: ... unexpected keyword argument 'body_weight_kg'`).

- [ ] **Step 3: Edit `build_renal_cl_grid`** in `src/sisyphus/mipd/renal_grid.py`. Add the two params to its signature (after `renal_factor: float = 1.0,`):

```python
    renal_factor: float = 1.0,
    body_weight_kg: float | None = None,
    age_years: float | None = None,
    kp_method: str = "rodgers_rowland",
    dt_output: float = 0.1,
) -> RenalCLGrid:
```

And change its `_build_grid_engine` call to pass them. The current call is:

```python
    compiled, realized_graph, drug, obs_node = _build_grid_engine(
        smiles, regimen.events[0].dose_mg, "iv", renal_factor, kp_method
    )
```

Change to:

```python
    compiled, realized_graph, drug, obs_node = _build_grid_engine(
        smiles, regimen.events[0].dose_mg, "iv", renal_factor, kp_method,
        body_weight_kg, age_years,
    )
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/unit/test_mipd_renal_grid.py -v` → PASS (all, incl. the existing faithfulness + monotonicity tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/renal_grid.py tests/unit/test_mipd_renal_grid.py
git commit -m "feat(mipd): build_renal_cl_grid threads weight/age (IV path)"
```

---

## Task 4: `predict_posterior` wiring (oral)

**Files:** Modify `src/sisyphus/mipd/api.py`. Test: `tests/unit/test_mipd_api.py` (defines `MIDAZOLAM`/`DOSE`/`Covariates`).

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/test_mipd_api.py`:

```python
def test_predict_posterior_weight_age_routes_through_individualized_solve():
    # weight (no obs, no CrCl) must reach the engine via the individualized solve,
    # not the reference F-only path; the clint latent stays fixed.
    post = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(body_weight_kg=10, age_years=2), seed=0)
    assert post.cl_scale is None
    assert post.cmax.point > 0


def test_predict_posterior_lower_weight_higher_exposure():
    ref = predict_posterior(MIDAZOLAM, DOSE, seed=0)
    light = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(body_weight_kg=40), seed=0)
    assert light.auc.point > ref.auc.point


def test_predict_posterior_empty_covariates_bit_identical():
    a = predict_posterior(MIDAZOLAM, DOSE, seed=0)
    b = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(), seed=0)
    assert np.array_equal(a.cmax.samples, b.cmax.samples)


def test_predict_posterior_extreme_weight_warns():
    post = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(body_weight_kg=1.0), seed=0)
    assert any("weight" in w.lower() for w in post.warnings)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/unit/test_mipd_api.py -k "weight" -v` → FAIL (`TypeError: ... unexpected keyword argument 'body_weight_kg'` from `Covariates`, OR the weight is ignored so the directional assert fails).

- [ ] **Step 3: Edit `predict_posterior`** in `src/sisyphus/mipd/api.py`.

(a) Replace the lines that compute `renal_factor`/`individualized` and the inline CrCl-warning block. The current code is:

```python
    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    individualized = renal_factor != 1.0
```
... and a few lines below, the warnings block:
```python
    warnings_list: list[str] = []
    if covariates is not None and covariates.crcl_ml_min is not None:
        if not (5.0 <= covariates.crcl_ml_min <= 200.0):
            warnings_list.append(
                f"crcl:extreme:{covariates.crcl_ml_min}: the engine renal model is "
                "glomerular-filtration-only and least reliable outside [5, 200] mL/min"
            )
```

Replace the `renal_factor`/`individualized` lines with:

```python
    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    has_phys = covariates.has_physiology() if covariates is not None else False
    body_weight_kg = covariates.body_weight_kg if covariates is not None else None
    age_years = covariates.age_years if covariates is not None else None
    individualized = renal_factor != 1.0 or has_phys
```

Replace the entire `warnings_list` block (the `warnings_list: list[str] = []` through the `crcl:extreme` append) with:

```python
    warnings_list: list[str] = list(covariates.warnings()) if covariates is not None else []
```

(b) Pass `body_weight_kg`/`age_years` to BOTH `build_cl_grid` calls. The `needs_grid` call currently ends with `renal_factor=renal_factor,` — add after it:

```python
            renal_factor=renal_factor,
            body_weight_kg=body_weight_kg,
            age_years=age_years,
        )
```

The `elif individualized:` single-solve call (`build_cl_grid(... n_grid=1, s_range=(1.0, 1.0), ... renal_factor=renal_factor)`) — add the two kwargs there too:

```python
            renal_factor=renal_factor,
            body_weight_kg=body_weight_kg,
            age_years=age_years,
        )
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/unit/test_mipd_api.py -v` → PASS (all, incl. existing CrCl/measured-F/meta tests). If `test_predict_posterior_lower_weight_higher_exposure` fails, STOP and report `ref.auc.point` and `light.auc.point`.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/api.py tests/unit/test_mipd_api.py
git commit -m "feat(mipd): predict_posterior weight/age dispatch + warnings"
```

---

## Task 5: `predict_tdm` wiring (IV)

**Files:** Modify `src/sisyphus/mipd/tdm.py`. Test: `tests/unit/test_mipd_tdm.py` (defines `ATENOLOL`/`_iv_regimen`/`Covariates`).

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/test_mipd_tdm.py`:

```python
def test_predict_tdm_age_individualizes():
    post = predict_tdm(ATENOLOL, _iv_regimen(), [], covariates=Covariates(age_years=80), n_grid=5, seed=0)
    assert post.cmax.point > 0
    assert post.renal_scale is not None


def test_predict_tdm_extreme_age_warns():
    post = predict_tdm(ATENOLOL, _iv_regimen(), [], covariates=Covariates(age_years=120), n_grid=5, seed=0)
    assert any("age" in w.lower() for w in post.warnings)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/unit/test_mipd_tdm.py -k "age" -v` → FAIL (the age is ignored / `TypeError` from threading, or no warning emitted).

- [ ] **Step 3: Edit `predict_tdm`** in `src/sisyphus/mipd/tdm.py`.

(a) Replace the `renal_factor = ...` line and the inline CrCl-warning block. Current:

```python
    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    rng = np.random.default_rng(seed)

    warnings_list: list[str] = []
    if covariates is not None and covariates.crcl_ml_min is not None:
        if not (5.0 <= covariates.crcl_ml_min <= 200.0):
            warnings_list.append(
                f"crcl:extreme:{covariates.crcl_ml_min}: the engine renal model is "
                "glomerular-filtration-only and least reliable outside [5, 200] mL/min"
            )
```

Replace with:

```python
    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    body_weight_kg = covariates.body_weight_kg if covariates is not None else None
    age_years = covariates.age_years if covariates is not None else None
    rng = np.random.default_rng(seed)

    warnings_list: list[str] = list(covariates.warnings()) if covariates is not None else []
```

(b) Pass `body_weight_kg`/`age_years` to `build_renal_cl_grid`. The current call:

```python
    grid = build_renal_cl_grid(
        smiles, regimen, n_grid=n_grid, renal_factor=renal_factor, kp_method=kp_method
    )
```

Change to:

```python
    grid = build_renal_cl_grid(
        smiles, regimen, n_grid=n_grid, renal_factor=renal_factor,
        body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
    )
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/unit/test_mipd_tdm.py -v` → PASS (all, incl. existing IV-guard/output-honesty/directional/CrCl tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/tdm.py tests/unit/test_mipd_tdm.py
git commit -m "feat(mipd): predict_tdm threads weight/age covariates"
```

---

## Task 6: Regression + lint + graphify

**Files:** none (verification).

- [ ] **Step 1: Full mipd suite + holdout pin**

Run: `python -m pytest tests/unit/test_mipd_api.py tests/unit/test_mipd_core.py tests/unit/test_mipd_grid.py tests/unit/test_mipd_meta.py tests/unit/test_mipd_clgrid.py tests/unit/test_mipd_covariates.py tests/unit/test_mipd_renal_grid.py tests/unit/test_mipd_tdm.py tests/integration/test_holdout_regression.py -q`
Expected: PASS — all mipd tests + the holdout pin (`predict()` untouched).

- [ ] **Step 2: Lint** — `ruff check src/sisyphus/mipd/ tests/unit/test_mipd_covariates.py tests/unit/test_mipd_grid.py tests/unit/test_mipd_renal_grid.py tests/unit/test_mipd_api.py tests/unit/test_mipd_tdm.py` → no errors (run `ruff check --fix` for import-order, then re-check).

- [ ] **Step 3: Update the graph** — `graphify update .` (AST-only, no API cost).

- [ ] **Step 4: Commit any lint changes**

```bash
git add -A src/sisyphus/mipd/ tests/
git commit -m "chore(mipd): ruff after weight/age covariates" || echo "nothing to commit"
```

---

## Self-Review (planner checklist — completed)

**1. Spec coverage:**
- §4.1 `Covariates` weight/age + validation + `has_physiology()` + `renal_factor()` unchanged → Task 1; `warnings()` consolidation → Task 1, consumed Tasks 4/5.
- §4.2 `_build_grid_engine` generate_physiology swap (absolute base_yaml) → Task 2.
- §4.3 entry-point wiring (build_cl_grid Task 2, build_renal_cl_grid Task 3, predict_posterior broaden `individualized` Task 4, predict_tdm Task 5).
- §4.4 gating/faithfulness (no weight/age → unchanged) → Task 2/3 `*_unchanged` tests + Task 4 bit-identity.
- §4.5 extreme-weight/age warnings → Task 1 `warnings()`, asserted Tasks 4/5.
- §5 invariants (engine/predict untouched, renal_factor unchanged, covariates=None bit-identical) → Tasks 2/3/4 + Task 6 holdout pin.
- §7 directional (lower weight → higher AUC) → Tasks 2 + 4; both-grid integration (the C1 gap) → Task 2 (oral) + Task 3 (IV).

**2. Placeholder scan:** none — every step has concrete code/commands.

**3. Type consistency:** `Covariates.has_physiology()`/`warnings()`/`body_weight_kg`/`age_years` (Task 1) ↔ used Tasks 4/5; `_build_grid_engine(..., body_weight_kg, age_years)` (Task 2) ↔ called by `build_cl_grid` (Task 2) and `build_renal_cl_grid` (Task 3); `build_cl_grid(..., body_weight_kg=, age_years=)` ↔ Task 4 calls; `build_renal_cl_grid(..., body_weight_kg=, age_years=)` ↔ Task 5 call. ✓
