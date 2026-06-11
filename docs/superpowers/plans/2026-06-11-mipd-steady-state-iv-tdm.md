# MIPD Steady-State IV TDM (Renal-CL Latent) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Condition the engine-as-prior on a steady-state IV trough to individualize renal clearance — a free renal-CL latent `r` (prior centered on the CrCl-implied value) updated by a measured trough, over a multi-dose engine solve (`regimen.solver.solve_regimen`, reused).

**Architecture:** A new renal-CL grid (`mipd/renal_grid.py`) re-solves `solve_regimen` across a renal-CL scale `r` (scaling `drug.renal_clearance`); a low-D SIR over `r` is conditioned on a steady-state `MeasuredConc` trough. A new entry point `mipd/tdm.py::predict_tdm` orchestrates it. F ≡ 1 (IV, no F latent); the oral-calibrated `meta_cmax`/`cmax_90ci` are NOT attached. The engine-build setup is factored out of `build_cl_grid` and shared.

**Tech Stack:** Python 3.10+, numpy, pytest, ruff. Reference spec: `docs/superpowers/specs/2026-06-11-mipd-steady-state-iv-tdm-design.md`. Reused: `regimen.solver.solve_regimen`, `regimen.types.DosingRegimen.iv_infusion`, `mipd.clgrid.MeasuredConc`, `mipd.core._softmax_resample`, `mipd.covariates.Covariates`.

---

## File Structure

- **MODIFY** `src/sisyphus/mipd/core.py` — add `PosteriorPK.renal_scale` field (additive).
- **MODIFY** `src/sisyphus/mipd/grid.py` — extract `_build_grid_engine` (shared engine setup); `build_cl_grid` calls it (behavior unchanged, pinned).
- **NEW** `src/sisyphus/mipd/renal_grid.py` — `RenalCLPrior`, `RenalCLGrid`, `RenalCLForward`, `build_renal_cl_grid`, `sir_posterior_renal`.
- **NEW** `src/sisyphus/mipd/tdm.py` — `predict_tdm` (IV steady-state entry).
- **NEW** `tests/unit/test_mipd_renal_grid.py`, `tests/unit/test_mipd_tdm.py`.

---

## Task 1: Add `PosteriorPK.renal_scale` field

**Files:** Modify `src/sisyphus/mipd/core.py` (PosteriorPK; last field is `warnings: tuple[str, ...] = ()`). Test: `tests/unit/test_mipd_core.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/unit/test_mipd_core.py`:

```python
def test_posterior_pk_renal_scale_defaults_none():
    from sisyphus.mipd.core import Posterior, PosteriorPK
    import numpy as np

    s = Posterior(np.array([1.0, 2.0, 3.0]))
    post = PosteriorPK(f=s, cmax=s, auc=s, n_eff=10.0)
    assert post.renal_scale is None
    post2 = PosteriorPK(f=s, cmax=s, auc=s, n_eff=10.0, renal_scale=s)
    assert post2.renal_scale is s
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/unit/test_mipd_core.py::test_posterior_pk_renal_scale_defaults_none -v` → FAIL (`TypeError: ... unexpected keyword argument 'renal_scale'`).

- [ ] **Step 3: Add the field** — in `src/sisyphus/mipd/core.py`, in `PosteriorPK`, immediately after the `warnings: tuple[str, ...] = ()` line, add:

```python
    # Renal-CL latent posterior (IV steady-state TDM path, mipd.tdm.predict_tdm):
    # the individualized renal-clearance scale relative to the CrCl-implied value.
    # None on every other path. Additive — preserves the prior contract.
    renal_scale: Posterior | None = None
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/unit/test_mipd_core.py -v` → PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/core.py tests/unit/test_mipd_core.py
git commit -m "feat(mipd): add PosteriorPK.renal_scale field for IV TDM"
```

---

## Task 2: Extract `_build_grid_engine` shared setup from `build_cl_grid`

**Why:** `build_renal_cl_grid` needs the identical engine-build setup (profile→adme→graph→drug→compile). Factor it out so the two grids share one source of truth instead of duplicating ~25 lines. `build_cl_grid`'s behavior must stay bit-identical (pinned by `test_cl_grid_at_unit_scale_reproduces_predict_engine_pk`).

**Files:** Modify `src/sisyphus/mipd/grid.py`. Test: `tests/unit/test_mipd_grid.py` (existing pin must still pass).

- [ ] **Step 1: Add the helper** — in `src/sisyphus/mipd/grid.py`, add this module-level function ABOVE `build_cl_grid` (move the relevant imports to module level or keep them function-local inside the helper):

```python
def _build_grid_engine(
    smiles: str,
    dose_mg: float,
    route: str,
    renal_factor: float,
    kp_method: str,
):
    """Build + compile the engine for a grid: profile -> adme -> graph -> drug.

    Returns ``(compiled, realized_graph, drug, obs_node)``. Applies the CrCl
    renal_factor once to the base drug. Shared by ``build_cl_grid`` (single-bolus
    clint grid) and ``build_renal_cl_grid`` (multi-dose renal grid).
    """
    from sisyphus.engine.compiler import ODECompiler
    from sisyphus.graph.axial import expand_axial
    from sisyphus.graph.builder import augment_for_active_species, build_from_yaml
    from sisyphus.pipeline.predict import _PHYSIOLOGY_DIR, _resolve_observation_node
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph, detect_disposition

    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")
    auto_oatp_kinetics, auto_ecm_params, non_cyp_fractions = detect_disposition(profile)

    liver_pre: dict[str, float] | None = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_pre = {tag: d.mean for tag, d in graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(
        profile, adme, dose_mg, route,
        liver_enzymes=liver_pre,
        kp_method=kp_method,
        transporter_kinetics=auto_oatp_kinetics,
        hepatic_ecm_params=auto_ecm_params,
        non_cyp_fractions=non_cyp_fractions,
    )
    if renal_factor != 1.0:
        drug = dataclasses.replace(
            drug,
            renal_clearance=Distribution(
                mean=drug.renal_clearance.mean * renal_factor,
                cv=drug.renal_clearance.cv,
            ),
        )
    graph = augment_for_active_species(graph, drug)
    graph = expand_axial(graph)
    compiled = ODECompiler().compile(graph)
    realized_graph = graph.realize_means()
    obs_node = _resolve_observation_node(drug)
    return compiled, realized_graph, drug, obs_node
```

- [ ] **Step 2: Rewrite `build_cl_grid` to use the helper** — replace the block in `build_cl_grid` that runs from `profile = compute_profile(smiles)` down through `obs_node = _resolve_observation_node(drug)` (the setup, including the `renal_factor` scaling block and `augment`/`expand`/`compile`/`realize`) with:

```python
    compiled, realized_graph, drug, obs_node = _build_grid_engine(
        smiles, dose_mg, route, renal_factor, kp_method
    )
```

Keep everything after it (`t_min_h = ...`, `admin_idx = compiled.state_index[drug.administration_node]`, the per-`s` loop, the nan handling, the `return CLGrid(...)`). Keep `s_grid = np.geomspace(...)` and the `t_grid` default at the top of `build_cl_grid`. Remove now-unused function-local imports in `build_cl_grid` that moved into the helper (`compute_profile`, `predict_adme`, `build_from_yaml`, `detect_disposition`, `build_drug_on_graph`, `augment_for_active_species`, `expand_axial`, `ODECompiler`, `_resolve_observation_node`), but KEEP the ones the loop still needs (`ResolvedParams`, `solve`, `_IV_CMAX_DELAY_H`, `compute_endpoints`, `_engine_oral_bioavailability`).

- [ ] **Step 3: Run the grid suite (behavior must be unchanged)** — `python -m pytest tests/unit/test_mipd_grid.py -v` → PASS (all, including `test_cl_grid_at_unit_scale_reproduces_predict_engine_pk` and the renal_factor tests). If ANY grid test changes outcome, the extraction altered behavior — STOP and report; do not proceed.

- [ ] **Step 4: Commit**

```bash
git add src/sisyphus/mipd/grid.py
git commit -m "refactor(mipd): extract _build_grid_engine shared by clint + renal grids"
```

---

## Task 3: `RenalCLPrior` + `RenalCLGrid` + `RenalCLForward` (pure numpy)

**Files:** Create `src/sisyphus/mipd/renal_grid.py`. Test: `tests/unit/test_mipd_renal_grid.py` (NEW).

- [ ] **Step 1: Write the failing test** — create `tests/unit/test_mipd_renal_grid.py`:

```python
"""Tests for mipd.renal_grid (IV steady-state renal-CL grid + SIR)."""
import numpy as np

from sisyphus.mipd.renal_grid import RenalCLForward, RenalCLGrid, RenalCLPrior


def _toy_grid() -> RenalCLGrid:
    # 3-point renal grid; lower r (lower renal CL) -> higher exposure.
    r_grid = np.array([0.5, 1.0, 2.0])
    t_grid = np.array([0.0, 1.0, 2.0, 3.0])
    # conc curves decreasing in r (more CL -> lower conc), decaying in t
    conc = np.array([
        [10.0, 6.0, 4.0, 2.0],   # r=0.5
        [8.0, 4.0, 2.0, 1.0],    # r=1.0
        [6.0, 2.0, 1.0, 0.5],    # r=2.0
    ])
    cmax = conc.max(axis=1)
    auc = np.trapz(conc, t_grid, axis=1)
    return RenalCLGrid(r_grid=r_grid, t_grid=t_grid, conc=conc, cmax=cmax, auc=auc)


def test_renal_prior_median_one_clipped_to_range():
    rng = np.random.default_rng(0)
    r = RenalCLPrior(cv=1.0, r_min=0.2, r_max=5.0).sample(20000, rng)
    assert abs(np.median(r) - 1.0) < 0.05
    assert r.min() >= 0.2 and r.max() <= 5.0


def test_renal_grid_conc_at_interpolates_and_guards_horizon():
    g = _toy_grid()
    # at r=1.0, t=1.0 -> exactly 4.0
    assert np.isclose(g.conc_at(np.array([1.0]), 1.0)[0], 4.0)
    import pytest
    with pytest.raises(ValueError):
        g.conc_at(np.array([1.0]), 99.0)  # beyond horizon


def test_renal_forward_interpolates_cmax_auc_monotone():
    fwd = RenalCLForward(_toy_grid())
    state = fwd(np.array([0.5, 2.0]))
    # lower r -> higher cmax and auc
    assert state["cmax"][0] > state["cmax"][1]
    assert state["auc"][0] > state["auc"][1]
    assert callable(state["conc_at"])
    assert state["conc_at"](1.0).shape == (2,)
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/unit/test_mipd_renal_grid.py -v` → FAIL (`ModuleNotFoundError: sisyphus.mipd.renal_grid`).

- [ ] **Step 3: Create `src/sisyphus/mipd/renal_grid.py`** with the prior, grid, and forward (the builder + SIR come in Tasks 4–5):

```python
"""IV steady-state renal-CL grid: a free renal-clearance latent + multi-dose solve.

For an IV drug the engine-as-prior's dominant structural error (bioavailability F)
is absent (F == 1). The latent that a steady-state trough constrains is renal
clearance. This module builds the engine over a renal-CL scale ``r`` (scaling
``drug.renal_clearance``) by re-solving the multi-dose regimen with
``regimen.solver.solve_regimen``, and runs a low-D SIR over ``r`` conditioned on a
``MeasuredConc`` trough. ``r`` multiplies the CrCl-set renal CL, so its prior is
centered on the CrCl-implied value (r=1.0). See the design spec
docs/superpowers/specs/2026-06-11-mipd-steady-state-iv-tdm-design.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sisyphus.mipd.core import Posterior, PosteriorPK, _softmax_resample


@dataclass(frozen=True)
class RenalCLPrior:
    """Prior over the renal-CL scale ``r``, centered at 1.0 (= CrCl-implied CL).

    ``cv`` defaults wide (renal CL is the engine's individual-level unknown).
    Samples are clipped to the grid range so the forward never extrapolates.
    """

    cv: float = 1.0
    r_min: float = 0.2
    r_max: float = 5.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        sigma = math.sqrt(math.log(1.0 + self.cv * self.cv))
        r = rng.lognormal(mean=0.0, sigma=sigma, size=n)  # median 1.0
        return np.clip(r, self.r_min, self.r_max)


@dataclass(frozen=True)
class RenalCLGrid:
    """Precomputed engine response over a renal-CL grid (ascending ``r_grid``).

    ``conc`` is the venous concentration-time curve per scale, shape (G, H) over
    the common ``t_grid`` (H,). ``cmax``/``auc`` are the steady-state final-interval
    quantities (G,). No ``f_engine`` — F == 1 for IV.
    """

    r_grid: np.ndarray
    t_grid: np.ndarray
    conc: np.ndarray
    cmax: np.ndarray
    auc: np.ndarray

    def conc_at(self, r: np.ndarray, t: float) -> np.ndarray:
        """Model venous concentration at time ``t`` for each renal-scale in ``r``.

        Interpolates each grid curve at ``t`` (linear in time), then across scale
        in log-log space. ``r`` is clipped to the grid range. ``t`` must lie within
        the simulated horizon — a later time (a trough past the regimen) raises.
        """
        if t < self.t_grid[0] or t > self.t_grid[-1]:
            raise ValueError(
                f"observation time t={t} h is outside the engine grid "
                f"[{self.t_grid[0]}, {self.t_grid[-1]}] h; extend the regimen to cover it"
            )
        r = np.asarray(r, dtype=float)
        c_at_t = np.array(
            [np.interp(t, self.t_grid, self.conc[g]) for g in range(self.r_grid.size)]
        )
        lr = np.log(np.clip(r, self.r_grid[0], self.r_grid[-1]))
        return np.exp(
            np.interp(lr, np.log(self.r_grid), np.log(np.maximum(c_at_t, 1e-300)))
        )


class RenalCLForward:
    """Forward map renal-scale ``r`` -> PK state over a precomputed ``RenalCLGrid``."""

    def __init__(self, grid: RenalCLGrid) -> None:
        self.grid = grid
        self._lr = np.log(grid.r_grid)
        self._lcmax = np.log(grid.cmax)
        self._lauc = np.log(grid.auc)

    def _interp(self, ltable: np.ndarray, r: np.ndarray) -> np.ndarray:
        lr = np.log(np.clip(r, self.grid.r_grid[0], self.grid.r_grid[-1]))
        return np.exp(np.interp(lr, self._lr, ltable))

    def __call__(self, r: np.ndarray) -> dict:
        r = np.asarray(r, dtype=float)
        grid = self.grid

        def conc_at(t: float, _r: np.ndarray = r) -> np.ndarray:
            return grid.conc_at(_r, t)

        return {
            "renal_scale": r,
            "cmax": self._interp(self._lcmax, r),
            "auc": self._interp(self._lauc, r),
            "conc_at": conc_at,
        }
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/unit/test_mipd_renal_grid.py -v` → PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/renal_grid.py tests/unit/test_mipd_renal_grid.py
git commit -m "feat(mipd): RenalCLPrior + RenalCLGrid + RenalCLForward (pure numpy)"
```

---

## Task 4: `build_renal_cl_grid` (multi-dose engine grid via `solve_regimen`)

**Files:** Modify `src/sisyphus/mipd/renal_grid.py` (add the builder). Test: `tests/unit/test_mipd_renal_grid.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/unit/test_mipd_renal_grid.py`:

```python
ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"  # high-fup, renally-cleared


def _iv_regimen():
    from sisyphus.regimen.types import DosingRegimen
    return DosingRegimen.iv_infusion(dose_mg=50.0, duration_h=0.5, interval_h=8.0, n_doses=5)


def test_build_renal_cl_grid_shape_and_horizon():
    from sisyphus.mipd.renal_grid import build_renal_cl_grid
    g = build_renal_cl_grid(ATENOLOL, _iv_regimen(), n_grid=5, r_range=(0.5, 2.0))
    assert g.cmax.shape == g.auc.shape == (5,)
    assert np.all(g.r_grid[1:] > g.r_grid[:-1])
    # horizon spans last_dose (4*8=32h) + max(interval 8, 24) = 56h
    assert g.t_grid[-1] >= 56.0 - 1e-6
    import pytest
    with pytest.raises(ValueError):
        g.conc_at(np.array([1.0]), 999.0)


def test_build_renal_cl_grid_lower_r_higher_exposure():
    from sisyphus.mipd.renal_grid import build_renal_cl_grid
    g = build_renal_cl_grid(ATENOLOL, _iv_regimen(), n_grid=5, r_range=(0.5, 2.0))
    # ascending r -> descending steady-state AUC (more renal CL -> less exposure)
    assert g.auc[0] > g.auc[-1]


def test_build_renal_cl_grid_faithful_to_solve_regimen():
    # The grid's per-r curve must equal a direct solve_regimen at that r.
    from sisyphus.mipd.grid import _build_grid_engine
    from sisyphus.mipd.renal_grid import build_renal_cl_grid
    import dataclasses
    from sisyphus.core import Distribution
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.regimen.solver import solve_regimen

    reg = _iv_regimen()
    g = build_renal_cl_grid(ATENOLOL, reg, n_grid=3, r_range=(1.0, 1.0))  # single r=1
    compiled, realized_graph, drug, obs_node = _build_grid_engine(
        ATENOLOL, reg.events[0].dose_mg, "iv", 1.0, "rodgers_rowland"
    )
    params = ResolvedParams(realized_graph, drug.realize_means())
    sim = solve_regimen(compiled, params, reg, t_total_h=float(g.t_grid[-1]))
    direct = np.interp(g.t_grid, sim.time_h, sim.concentrations[obs_node])
    assert np.allclose(g.conc[0], direct, rtol=1e-6, atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/unit/test_mipd_renal_grid.py -k build_renal -v` → FAIL (`ImportError: cannot import name 'build_renal_cl_grid'`).

- [ ] **Step 3: Add `build_renal_cl_grid`** to `src/sisyphus/mipd/renal_grid.py`:

```python
def _regimen_interval_h(regimen) -> float:
    """Dosing interval tau from a regimen (events[1]-events[0]); 24.0 if single-dose."""
    ev = regimen.events
    return float(ev[1].time_h - ev[0].time_h) if len(ev) >= 2 else 24.0


def build_renal_cl_grid(
    smiles: str,
    regimen,
    *,
    n_grid: int = 13,
    r_range: tuple[float, float] = (0.2, 5.0),
    renal_factor: float = 1.0,
    kp_method: str = "rodgers_rowland",
    dt_output: float = 0.1,
) -> RenalCLGrid:
    """Solve the IV regimen over a renal-CL grid and return a ``RenalCLGrid``.

    Reuses ``grid._build_grid_engine`` (engine setup, CrCl renal_factor) and
    ``regimen.solver.solve_regimen`` (multi-dose solve). ``cmax``/``auc`` are the
    steady-state final-dosing-interval quantities. F == 1 (no f_engine column).
    """
    import dataclasses

    from sisyphus.core import Distribution
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.mipd.grid import (
        _build_grid_engine,
        _fill_nan_log_s,
        _nearest_finite_backfill,
    )
    from sisyphus.regimen.solver import solve_regimen

    compiled, realized_graph, drug, obs_node = _build_grid_engine(
        smiles, regimen.events[0].dose_mg, "iv", renal_factor, kp_method
    )
    r_grid = np.geomspace(r_range[0], r_range[1], n_grid)

    last = float(regimen.last_dose_time_h)
    tau = _regimen_interval_h(regimen)
    t_total = last + max(tau, 24.0)
    n_points = max(2, int(round(t_total / dt_output)) + 1)
    t_grid = np.linspace(0.0, t_total, n_points)

    conc_rows: list[np.ndarray] = []
    cmaxs: list[float] = []
    aucs: list[float] = []
    for r in r_grid:
        drug_r = dataclasses.replace(
            drug,
            renal_clearance=Distribution(
                mean=drug.renal_clearance.mean * float(r), cv=drug.renal_clearance.cv
            ),
        )
        params_r = ResolvedParams(realized_graph, drug_r.realize_means())
        sim = solve_regimen(compiled, params_r, regimen, t_total_h=t_total, dt_output=dt_output)
        if not sim.solver_success:
            conc_rows.append(np.full(t_grid.size, np.nan))
            cmaxs.append(np.nan)
            aucs.append(np.nan)
            continue
        t_native = sim.time_h
        c_native = sim.concentrations[obs_node]
        conc_rows.append(np.interp(t_grid, t_native, c_native))
        # steady-state final dosing interval [last, last+tau]
        mask = (t_native >= last - 1e-9) & (t_native <= last + tau + 1e-9)
        if mask.sum() >= 2:
            cmaxs.append(float(np.max(c_native[mask])))
            aucs.append(float(np.trapz(c_native[mask], t_native[mask])))
        else:
            cmaxs.append(np.nan)
            aucs.append(np.nan)

    cmaxs_arr = np.array(cmaxs)
    if not np.isfinite(cmaxs_arr).any():
        raise ValueError(
            f"engine failed at all {n_grid} renal-scale grid points; cannot build the grid"
        )
    cmax = _fill_nan_log_s(cmaxs_arr, r_grid)
    auc = _fill_nan_log_s(np.array(aucs), r_grid)
    conc = _nearest_finite_backfill(np.array(conc_rows))
    return RenalCLGrid(r_grid=r_grid, t_grid=t_grid, conc=conc, cmax=cmax, auc=auc)
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/unit/test_mipd_renal_grid.py -v` → PASS (all, including faithfulness). If the faithfulness test fails, the grid diverges from `solve_regimen` — STOP and report the max abs diff; do not loosen the tolerance without diagnosis.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/renal_grid.py tests/unit/test_mipd_renal_grid.py
git commit -m "feat(mipd): build_renal_cl_grid via solve_regimen (steady-state IV)"
```

---

## Task 5: `sir_posterior_renal` (SIR over the renal-CL latent)

**Files:** Modify `src/sisyphus/mipd/renal_grid.py`. Test: `tests/unit/test_mipd_renal_grid.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/unit/test_mipd_renal_grid.py`:

```python
def test_sir_posterior_renal_trough_moves_r_correctly():
    # Higher trough than the r=1 curve predicts -> patient clears SLOWER -> r < 1.
    from sisyphus.mipd.clgrid import MeasuredConc
    from sisyphus.mipd.renal_grid import (
        RenalCLForward,
        RenalCLPrior,
        sir_posterior_renal,
    )
    grid = _toy_grid()  # at r=1.0, conc_at(t=2.0) == 2.0
    fwd = RenalCLForward(grid)
    rng = np.random.default_rng(0)
    # measured trough HIGHER than the r=1 prediction (4.0 vs 2.0) -> slower CL -> r<1
    high = sir_posterior_renal(
        RenalCLPrior(cv=1.0, r_min=0.5, r_max=2.0), fwd,
        [MeasuredConc(value=4.0, t=2.0, cv=0.1)], n_samples=20000, rng=rng,
    )
    assert high.renal_scale.point < 1.0
    assert high.cmax.point > grid.cmax[1]  # slower CL -> higher exposure than r=1
    assert high.n_eff > 100


def test_sir_posterior_renal_iv_has_degenerate_f():
    from sisyphus.mipd.clgrid import MeasuredConc
    from sisyphus.mipd.renal_grid import (
        RenalCLForward,
        RenalCLPrior,
        sir_posterior_renal,
    )
    post = sir_posterior_renal(
        RenalCLPrior(r_min=0.5, r_max=2.0), RenalCLForward(_toy_grid()),
        [MeasuredConc(value=3.0, t=1.0, cv=0.2)], n_samples=5000,
        rng=np.random.default_rng(0),
    )
    assert np.allclose(post.f.samples, 1.0)  # F == 1 for IV
    assert post.renal_scale is not None
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/unit/test_mipd_renal_grid.py -k sir_posterior_renal -v` → FAIL (`ImportError`).

- [ ] **Step 3: Add `sir_posterior_renal`** to `src/sisyphus/mipd/renal_grid.py`:

```python
def sir_posterior_renal(
    prior: RenalCLPrior,
    forward: RenalCLForward,
    observations,
    n_samples: int = 20000,
    rng: np.random.Generator | None = None,
) -> PosteriorPK:
    """SIR posterior over the renal-CL scale ``r`` given observations.

    Draws ``r`` from the prior, weights by the joint observation likelihood
    (e.g. a steady-state ``MeasuredConc`` trough via ``conc_at``), resamples.
    F is degenerate (== 1) for the IV path. Reports ``n_eff``.
    """
    if rng is None:
        rng = np.random.default_rng()
    r = prior.sample(n_samples, rng)
    state = forward(r)
    loglik = np.zeros(n_samples)
    for obs in observations:
        loglik = loglik + obs.log_likelihood(state)
    idx, n_eff = _softmax_resample(loglik, rng)
    return PosteriorPK(
        f=Posterior(np.ones(idx.size)),
        cmax=Posterior(state["cmax"][idx]),
        auc=Posterior(state["auc"][idx]),
        n_eff=n_eff,
        renal_scale=Posterior(r[idx]),
    )
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/unit/test_mipd_renal_grid.py -v` → PASS (all). If `test_sir_posterior_renal_trough_moves_r_correctly` fails on the direction (`renal_scale.point < 1.0`), STOP — a higher-than-predicted trough must pull `r` DOWN (slower clearance); report the actual `renal_scale.point`.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/renal_grid.py tests/unit/test_mipd_renal_grid.py
git commit -m "feat(mipd): sir_posterior_renal (renal-CL latent SIR)"
```

---

## Task 6: `predict_tdm` entry point (IV steady-state)

**Files:** Create `src/sisyphus/mipd/tdm.py`. Test: `tests/unit/test_mipd_tdm.py` (NEW).

- [ ] **Step 1: Write the failing tests** — create `tests/unit/test_mipd_tdm.py`:

```python
"""Integration tests for mipd.tdm.predict_tdm (IV steady-state TDM)."""
import numpy as np
import pytest

from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.covariates import Covariates
from sisyphus.mipd.tdm import predict_tdm
from sisyphus.regimen.types import DosingRegimen

ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"  # high-fup, renally-cleared


def _iv_regimen():
    return DosingRegimen.iv_infusion(dose_mg=50.0, duration_h=0.5, interval_h=8.0, n_doses=5)


def test_predict_tdm_rejects_oral_regimen():
    oral = DosingRegimen.oral_repeated(dose_mg=50.0, interval_h=8.0, n_doses=3)
    with pytest.raises(ValueError, match="IV"):
        predict_tdm(ATENOLOL, oral, [], seed=0)


def test_predict_tdm_output_is_honest_for_iv():
    post = predict_tdm(ATENOLOL, _iv_regimen(), [], n_grid=5, seed=0)
    assert post.meta_cmax is None      # no oral-calibrated population blend for IV
    assert post.cmax_90ci is None      # no oral-calibrated conformal band for IV
    assert post.renal_scale is not None
    assert post.cmax.point > 0


def test_predict_tdm_low_trough_means_faster_clearance_lower_exposure():
    reg = _iv_regimen()
    # baseline (no obs) predicted trough at t=39h
    base = predict_tdm(ATENOLOL, reg, [], n_grid=9, seed=0)
    # a trough far BELOW baseline -> faster clearance -> renal_scale > 1 -> lower exposure
    base_trough = float(base.cmax.point)  # rough scale ref
    low = predict_tdm(
        ATENOLOL, reg, [MeasuredConc(value=base_trough * 0.1, t=39.0, cv=0.2)],
        n_grid=9, seed=0,
    )
    assert low.renal_scale.point > 1.0
    assert low.auc.point < base.auc.point


def test_predict_tdm_extreme_crcl_warns():
    post = predict_tdm(
        ATENOLOL, _iv_regimen(), [], covariates=Covariates(crcl_ml_min=3), n_grid=5, seed=0
    )
    assert any("crcl" in w.lower() for w in post.warnings)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/unit/test_mipd_tdm.py -v` → FAIL (`ModuleNotFoundError: sisyphus.mipd.tdm`).

- [ ] **Step 3: Create `src/sisyphus/mipd/tdm.py`**:

```python
"""Public IV steady-state TDM: SMILES + IV regimen + trough -> renal-CL posterior.

``predict_tdm`` conditions the engine-as-prior on a steady-state IV trough to
individualize renal clearance. The latent is a free renal-CL scale (prior centered
on the CrCl-implied value); F == 1 (IV). The oral-train-calibrated ``meta_cmax`` /
``cmax_90ci`` are NOT attached — they are oral-Cmax artifacts, invalid for IV. The
primary output is the conditioned engine posterior ``cmax``/``auc`` with the
``renal_scale`` posterior; ``cmax.ci90`` is a parameter-uncertainty band (does not
carry calibrated structural coverage). See the design spec.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from sisyphus.mipd.core import PosteriorPK
from sisyphus.mipd.covariates import Covariates
from sisyphus.regimen.types import DEFAULT_IV_NODE, DosingRegimen


def predict_tdm(
    smiles: str,
    regimen: DosingRegimen,
    observations,
    *,
    covariates: Covariates | None = None,
    renal_prior_cv: float = 1.0,
    n_samples: int = 20000,
    n_grid: int = 13,
    seed: int = 0,
    kp_method: str = "rodgers_rowland",
) -> PosteriorPK:
    """Posterior PK for an IV ``regimen`` given steady-state trough ``observations``.

    Args:
        regimen: an IV ``DosingRegimen`` (e.g. ``DosingRegimen.iv_infusion(...)``).
            Every event must target the IV node; an oral regimen is rejected.
        observations: ``MeasuredConc`` troughs at times within the regimen horizon.
        covariates: v1 supports a measured CrCl, which sets the renal-CL prior
            center (``CrCl/125``); the trough updates the latent around it.
    """
    from sisyphus.mipd.renal_grid import (
        RenalCLForward,
        RenalCLPrior,
        build_renal_cl_grid,
        sir_posterior_renal,
    )

    if any(ev.node != DEFAULT_IV_NODE for ev in regimen.events):
        raise ValueError(
            "predict_tdm supports IV regimens only (every event must target the IV "
            f"node {DEFAULT_IV_NODE!r}); oral steady-state TDM is a future extension."
        )

    observations = list(observations)
    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    rng = np.random.default_rng(seed)

    warnings_list: list[str] = []
    if covariates is not None and covariates.crcl_ml_min is not None:
        if not (5.0 <= covariates.crcl_ml_min <= 200.0):
            warnings_list.append(
                f"crcl:extreme:{covariates.crcl_ml_min}: the engine renal model is "
                "glomerular-filtration-only and least reliable outside [5, 200] mL/min"
            )

    grid = build_renal_cl_grid(
        smiles, regimen, n_grid=n_grid, renal_factor=renal_factor, kp_method=kp_method
    )
    post = sir_posterior_renal(
        RenalCLPrior(cv=renal_prior_cv, r_min=float(grid.r_grid[0]), r_max=float(grid.r_grid[-1])),
        RenalCLForward(grid),
        observations,
        n_samples=n_samples,
        rng=rng,
    )
    return dataclasses.replace(post, warnings=tuple(warnings_list))
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/unit/test_mipd_tdm.py -v` → PASS (4 tests). If `test_predict_tdm_low_trough_means_faster_clearance_lower_exposure` fails the direction, STOP and report `low.renal_scale.point` and both AUC points — a far-below trough MUST pull `renal_scale` ABOVE 1 and AUC DOWN.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/tdm.py tests/unit/test_mipd_tdm.py
git commit -m "feat(mipd): predict_tdm IV steady-state TDM entry (renal-CL latent)"
```

---

## Task 7: Regression + lint + graphify

**Files:** none (verification).

- [ ] **Step 1: Full mipd suite + holdout pin**

Run: `python -m pytest tests/unit/test_mipd_api.py tests/unit/test_mipd_core.py tests/unit/test_mipd_grid.py tests/unit/test_mipd_meta.py tests/unit/test_mipd_clgrid.py tests/unit/test_mipd_covariates.py tests/unit/test_mipd_renal_grid.py tests/unit/test_mipd_tdm.py tests/integration/test_holdout_regression.py -q`
Expected: PASS — all mipd tests (the CrCl/clint paths unchanged) and the holdout pin (`predict()` untouched).

- [ ] **Step 2: Lint** — `ruff check src/sisyphus/mipd/ tests/unit/test_mipd_renal_grid.py tests/unit/test_mipd_tdm.py tests/unit/test_mipd_core.py` → no errors (fix any import-order issues and re-run).

- [ ] **Step 3: Update the graph** — `graphify update .` (AST-only, no API cost).

- [ ] **Step 4: Commit any lint changes**

```bash
git add -A src/sisyphus/mipd/ tests/
git commit -m "chore(mipd): ruff after steady-state IV TDM" || echo "nothing to commit"
```

---

## Self-Review (planner checklist — completed)

**1. Spec coverage:**
- §4.1 renal-CL grid via solve_regimen, final-interval cmax/auc, horizon `last+max(tau,24)` → Task 4. ✓
- §4.2 renal-CL latent + CrCl prior center → Task 3 (`RenalCLPrior`) + Task 6 (`renal_factor` from covariates). ✓
- §4.3 forward + SIR, F≡1 → Task 3 (`RenalCLForward`) + Task 5 (`sir_posterior_renal`, degenerate f). ✓
- §4.4 `predict_tdm` entry, IV-only guard → Task 6. ✓
- §4.5 honest output (meta/conformal None) → Task 6 (`predict_tdm` never attaches them) + `test_predict_tdm_output_is_honest_for_iv`. ✓
- §4.6 reuse `MeasuredConc`; horizon guard → Tasks 3–4 (`conc_at` raises). ✓
- §5 invariants: engine identity-blind (predict/grid-layer scaling), `predict()`/`predict_posterior` untouched, `PosteriorPK.renal_scale` additive, reuse `solve_regimen`/`DosingRegimen`/`_softmax_resample`/`MeasuredConc` → Tasks 1–6 + Task 7 holdout pin. ✓
- §7 directional tests (low trough → r>1 → lower exposure; lower r → higher exposure) → Tasks 4–6. ✓

**2. Placeholder scan:** none — every step has concrete code/commands.

**3. Type consistency:** `RenalCLPrior(cv, r_min, r_max).sample` ↔ used in Task 6; `RenalCLGrid(r_grid,t_grid,conc,cmax,auc)` ↔ built in Task 4, consumed by `RenalCLForward`/`conc_at`; `build_renal_cl_grid(smiles, regimen, *, n_grid, r_range, renal_factor, kp_method, dt_output)` ↔ called in Task 6; `sir_posterior_renal(prior, forward, observations, n_samples, rng)` ↔ Task 6; `_build_grid_engine(smiles, dose_mg, route, renal_factor, kp_method)` (Task 2) ↔ used in Task 4 and the faithfulness test; `PosteriorPK.renal_scale` (Task 1) ↔ set in Task 5. ✓

**Deliberate note:** the `f` field is a degenerate F≡1 `Posterior` on the IV path (PosteriorPK requires `f`); this is the documented IV smell, acceptable for v1.
