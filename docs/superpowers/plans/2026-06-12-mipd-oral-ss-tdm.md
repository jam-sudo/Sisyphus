# Oral Steady-State TDM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `mipd.predict_tdm` and `mipd.recommend_dose` from IV-only to oral multi-dose, conditioning the engine-as-prior on a steady-state oral trough (F-only) or peak+trough/MeasuredF (free-both), and recommending a target-attainment oral dose.

**Architecture:** A new `build_oral_cl_grid` reuses the existing `CLGrid`/`CLGridForward`/`sir_posterior_2d` (2-latent F+clint) machinery; `predict_tdm`/`recommend_dose` gain a route pre-check that leaves the IV numerical path bit-identical and adds an oral branch. F re-enters as the dominant latent (IV froze only renal-CL since F≡1). Measured-input feature — the 2.731 holdout headline is untouched.

**Tech Stack:** Python 3.10+, numpy, frozen dataclasses, pytest. Reuses `regimen.solver.solve_regimen`, `regimen.profile.compute_steady_state_metrics`, `pk.nca.terminal_half_life`, `grid._build_grid_engine`, `pipeline.predict._engine_oral_bioavailability`.

**Spec:** `docs/superpowers/specs/2026-06-12-mipd-oral-ss-tdm-design.md` (revised through two adversarial go-overs; findings B1–B3/I1–I2/M1–M7 and Bx1–Bx6/Mx1–Mx3 are folded into the spec and tagged in the tasks below).

**Branch:** `feat/mipd-oral-ss-tdm` (already checked out, off `c76ebcb`).

**Invariants (hold across every task):** engine/ & predict() untouched; IV numerical path bit-identical (run the existing IV TDM tests after every task); identity-blind; all PK quantities `Posterior`/`Distribution`; directional tests anchor to the engine's own s=1 prediction (no magic numbers — `df4492c` discipline); commit as jam-sudo noreply, NO `Co-Authored-By`/AI trailer; 20-file-per-directory ceiling (mipd/ will be 13 files — OK).

---

## Task 1: Shared regimen helpers (`mipd/_regimen.py`)

Owns route classification, the uniform-regimen precondition, the final-interval helper, and the wraparound-aware shape test — so `tdm.py`/`dosing.py`/`oral_grid.py` import one home and `renal_grid._regimen_interval_h` is re-pointed to it (Bx5/Bx6).

**Files:**
- Create: `src/sisyphus/mipd/_regimen.py`
- Modify: `src/sisyphus/mipd/renal_grid.py` (re-point its local `_regimen_interval_h` to the shared one)
- Test: `tests/unit/test_mipd_regimen_helpers.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mipd_regimen_helpers.py
import numpy as np
import pytest

from sisyphus.mipd._regimen import (
    _SHAPE_PHASE_TOL,
    _distinct_phases,
    _regimen_interval_h,
    _regimen_route,
    _require_uniform_regimen,
)
from sisyphus.regimen.types import DosingRegimen


def _oral(dose=100.0, tau=12.0, n=5):
    return DosingRegimen.oral_repeated(dose_mg=dose, interval_h=tau, n_doses=n)


def _iv(dose=100.0, dur=1.0, tau=12.0, n=5):
    return DosingRegimen.iv_infusion(dose_mg=dose, duration_h=dur, interval_h=tau, n_doses=n)


def test_route_oral_and_iv():
    assert _regimen_route(_oral()) == "oral"
    assert _regimen_route(_iv()) == "iv"


def test_route_mixed_raises():
    oral = _oral()
    iv = _iv()
    mixed = DosingRegimen(events=oral.events[:2] + iv.events[:1])
    with pytest.raises(ValueError):
        _regimen_route(mixed)


def test_uniform_ok_nonuniform_raises():
    _require_uniform_regimen(_oral(tau=12.0))  # no raise
    reg = _oral(tau=12.0)
    bad = DosingRegimen(events=[
        reg.events[0],
        reg.events[1]._replace(time_h=reg.events[1].time_h + 5.0)
        if hasattr(reg.events[1], "_replace")
        else __import__("dataclasses").replace(reg.events[1], time_h=reg.events[1].time_h + 5.0),
        *reg.events[2:],
    ])
    with pytest.raises(ValueError):
        _require_uniform_regimen(bad)


def test_interval_is_final_interval():
    assert _regimen_interval_h(_oral(tau=8.0)) == pytest.approx(8.0)
    # single dose -> 24.0 default
    single = DosingRegimen.oral_repeated(dose_mg=100.0, interval_h=12.0, n_doses=1)
    assert _regimen_interval_h(single) == pytest.approx(24.0)


def test_distinct_phases_circular():
    tau = 12.0
    from sisyphus.mipd.clgrid import MeasuredConc
    # two troughs one interval apart -> same phase -> NOT distinct
    same = [MeasuredConc(value=1.0, t=12.0), MeasuredConc(value=1.0, t=24.0)]
    assert _distinct_phases(same, tau) is False
    # peak (t=2) + trough (t=12) -> distinct phases
    shape = [MeasuredConc(value=5.0, t=2.0), MeasuredConc(value=1.0, t=12.0)]
    assert _distinct_phases(shape, tau) is True
    # 0/tau wraparound: t=0.1 (just after dose) and t=11.9 -> circular distance ~0.2 < tol -> NOT distinct
    wrap = [MeasuredConc(value=1.0, t=0.1), MeasuredConc(value=1.0, t=11.9)]
    assert _distinct_phases(wrap, tau) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_mipd_regimen_helpers.py -x -q`
Expected: FAIL (`ModuleNotFoundError: sisyphus.mipd._regimen`).

- [ ] **Step 3: Implement `_regimen.py`**

```python
# src/sisyphus/mipd/_regimen.py
"""Shared regimen helpers for the MIPD TDM stack (route, uniformity, interval, shape).

Co-locating these keeps ``tdm.py``/``dosing.py``/``oral_grid.py`` on one contract and
gives ``renal_grid`` a single ``_regimen_interval_h`` to import. All operate on a
``regimen.types.DosingRegimen`` and are identity-blind.
"""
from __future__ import annotations

import numpy as np

from sisyphus.regimen.types import DEFAULT_IV_NODE, DEFAULT_ORAL_NODE

# Phase-distinctness tolerance as a fraction of tau (spec §6). A pair must be
# separated by more than this (circular) to justify freeing the clint axis.
_SHAPE_PHASE_TOL_FRAC: float = 0.1


def _regimen_route(regimen) -> str:
    """'iv' if every event targets the IV node, 'oral' if every event the oral node.

    Mixed or unknown nodes raise — the TDM stack models a single route at a time.
    """
    nodes = {ev.node for ev in regimen.events}
    if nodes == {DEFAULT_IV_NODE}:
        return "iv"
    if nodes == {DEFAULT_ORAL_NODE}:
        return "oral"
    raise ValueError(
        f"regimen mixes/uses unsupported administration nodes {sorted(nodes)!r}; "
        f"TDM supports a pure IV ({DEFAULT_IV_NODE!r}) or pure oral "
        f"({DEFAULT_ORAL_NODE!r}) regimen."
    )


def _require_uniform_regimen(regimen) -> None:
    """Raise ``ValueError`` if dosing intervals are non-uniform (>~1% spread).

    Steady-state TDM assumes a uniform interval; a non-uniform reconstructed history
    is out of scope (spec §10) and is rejected rather than silently mis-modeled.
    """
    times = np.array([ev.time_h for ev in regimen.events], dtype=float)
    if times.size < 3:
        return
    gaps = np.diff(times)
    median = float(np.median(gaps))
    if median <= 0:
        raise ValueError("regimen event times are non-increasing")
    if float(np.max(np.abs(gaps - median))) > 0.01 * median:
        raise ValueError(
            "non-uniform dosing interval detected; oral/IV steady-state TDM assumes "
            "a uniform interval (non-uniform regimens are out of scope)"
        )


def _regimen_interval_h(regimen) -> float:
    """The dosing interval tau (h): the FINAL interval, or 24.0 for a single dose.

    Returns the final gap so tau is physically correct even within the uniformity
    tolerance; under ``_require_uniform_regimen`` it equals the first interval.
    """
    events = regimen.events
    if len(events) < 2:
        return 24.0
    return float(events[-1].time_h - events[-2].time_h)


def _distinct_phases(observations, tau: float) -> bool:
    """True if the MeasuredConc phases span distinct within-interval positions.

    Phase ``phi = t mod tau``; distinctness uses the maximum pairwise CIRCULAR
    distance ``min(|dphi|, tau-|dphi|)`` so a 0/tau (both trough-like) pair reads as
    same-phase (spec §6, Bx6). Only ``MeasuredConc`` obs carry a time; others ignored.
    """
    phis = [float(o.t) % tau for o in observations if hasattr(o, "t")]
    if len(phis) < 2:
        return False
    tol = _SHAPE_PHASE_TOL_FRAC * tau
    max_d = 0.0
    for i in range(len(phis)):
        for j in range(i + 1, len(phis)):
            d = abs(phis[i] - phis[j])
            d = min(d, tau - d)
            max_d = max(max_d, d)
    return max_d > tol
```

- [ ] **Step 4: Re-point the renal helper**

In `src/sisyphus/mipd/renal_grid.py`, replace the local `_regimen_interval_h` definition (around line 108) and its usage with an import from the shared module. At the top-of-file imports add `from sisyphus.mipd._regimen import _regimen_interval_h` and delete the local `def _regimen_interval_h(...)`. (Value-identical for uniform regimens — `events[-1]-events[-2] == events[1]-events[0]`.)

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_mipd_regimen_helpers.py tests/unit/test_mipd_renal_grid.py -q`
Expected: PASS (both the new helpers and the re-pointed renal grid).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/mipd/_regimen.py src/sisyphus/mipd/renal_grid.py tests/unit/test_mipd_regimen_helpers.py
git commit -m "feat(mipd): shared regimen helpers (route/uniformity/interval/shape)"
```

---

## Task 2: `build_oral_cl_grid` (`mipd/oral_grid.py`)

Three engine solves per grid point (single-dose oral → `f_engine`; single-dose IV reference inside `_engine_oral_bioavailability`; multi-dose `solve_regimen` → SS), returning `(CLGrid, is_steady_state, t_half_h)` (Bx1/Bx4/α). It is a merge of `build_cl_grid` (metabolic-`s` scaling + `f_engine`) and `build_renal_cl_grid` (multi-dose final-interval mask).

**Files:**
- Create: `src/sisyphus/mipd/oral_grid.py`
- Test: `tests/unit/test_mipd_oral_grid.py`

- [ ] **Step 1: Write the failing tests** (load-bearing first: LTI exactness, grid faithfulness, n_grid=1)

```python
# tests/unit/test_mipd_oral_grid.py
import numpy as np
import pytest

from sisyphus.mipd.clgrid import CLGrid, CLGridForward
from sisyphus.mipd.oral_grid import build_oral_cl_grid
from sisyphus.regimen.types import DosingRegimen

SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine — in-domain, fast to solve


def _reg(dose=100.0, tau=12.0, n=6):
    return DosingRegimen.oral_repeated(dose_mg=dose, interval_h=tau, n_doses=n)


def test_returns_tuple_grid_bool_thalf():
    grid, is_ss, t_half = build_oral_cl_grid(SMILES, _reg(), n_grid=5)
    assert isinstance(grid, CLGrid)
    assert isinstance(is_ss, bool)
    assert t_half is None or t_half > 0.0
    assert grid.f_engine.shape == grid.s_grid.shape
    assert np.all(grid.f_engine > 0) and np.all(grid.f_engine <= 1.0)


def test_lti_exactness_dose_linear():
    # SS cmax/auc scale linearly with dose (LTI premise of the dose solve).
    g1, _, _ = build_oral_cl_grid(SMILES, _reg(dose=100.0), n_grid=5)
    g2, _, _ = build_oral_cl_grid(SMILES, _reg(dose=200.0), n_grid=5)
    np.testing.assert_allclose(g2.cmax, 2.0 * g1.cmax, rtol=1e-6)
    np.testing.assert_allclose(g2.auc, 2.0 * g1.auc, rtol=1e-6)


def test_n_grid_1_s1_slice():
    grid, _, _ = build_oral_cl_grid(SMILES, _reg(), n_grid=1, s_range=(1.0, 1.0))
    assert grid.s_grid.shape == (1,)
    fwd = CLGridForward(grid)
    f = np.array([grid.f_engine[0]])  # F == f_engine -> scale 1
    s = np.ones(1)
    state = fwd(f, s)
    # trough at end of final interval must be readable (in-horizon) and positive
    last = float(_reg().last_dose_time_h)
    assert state["conc_at"](last + 12.0) > 0.0


def test_grid_faithful_at_s1_matches_direct_solve():
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.mipd.grid import _build_grid_engine
    from sisyphus.regimen.solver import solve_regimen
    reg = _reg()
    grid, _, _ = build_oral_cl_grid(SMILES, reg, n_grid=3, s_range=(0.5, 2.0))
    i1 = int(np.argmin(np.abs(np.log(grid.s_grid))))
    compiled, rgraph, drug, obs = _build_grid_engine(
        SMILES, reg.events[0].dose_mg, "oral", 1.0, "rodgers_rowland", None, None
    )
    sim = solve_regimen(compiled, ResolvedParams(rgraph, drug.realize_means()), reg,
                        t_total_h=float(reg.last_dose_time_h) + 24.0, dt_output=0.1)
    last = float(reg.last_dose_time_h)
    m = (sim.time_h >= last - 1e-9) & (sim.time_h <= last + 12.0 + 1e-9)
    direct_cmax = float(np.max(sim.concentrations[obs][m]))
    assert grid.cmax[i1] == pytest.approx(direct_cmax, rel=0.05)


def test_all_grid_points_fail_raises(monkeypatch):
    with pytest.raises(ValueError):
        build_oral_cl_grid("[Na+].[Cl-]", _reg(), n_grid=3)  # non-drug -> engine fails
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_mipd_oral_grid.py -x -q`
Expected: FAIL (`ModuleNotFoundError: sisyphus.mipd.oral_grid`).

- [ ] **Step 3: Implement `oral_grid.py`**

```python
# src/sisyphus/mipd/oral_grid.py
"""Oral steady-state clint-scale grid for the engine-as-prior TDM stack.

Composes ``build_cl_grid`` (metabolic-s scaling + single-dose f_engine) and
``build_renal_cl_grid`` (multi-dose SS final-interval extraction). Per grid point s:
(1) single-dose oral solve -> oral AUC + endpoints; (2) single-dose IV reference inside
``_engine_oral_bioavailability`` -> f_engine; (3) multi-dose ``solve_regimen`` -> SS
cmax/auc/conc on the final interval. Returns the existing ``CLGrid`` plus the SS flag and
terminal t1/2 (computed at s~=1), so no grid type changes (spec §3, Bx1/Bx4).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from sisyphus.mipd.clgrid import CLGrid


def build_oral_cl_grid(
    smiles: str,
    regimen,
    *,
    n_grid: int = 13,
    s_range: tuple[float, float] = (0.05, 20.0),
    renal_factor: float = 1.0,
    body_weight_kg: float | None = None,
    age_years: float | None = None,
    kp_method: str = "rodgers_rowland",
    dt_output: float = 0.1,
) -> tuple[CLGrid, bool, float | None]:
    """Build the oral SS clint grid. Returns ``(grid, is_steady_state, t_half_h)``."""
    from sisyphus.core import Distribution
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.mipd._regimen import _regimen_interval_h
    from sisyphus.mipd.grid import (
        _build_grid_engine,
        _default_t_grid,
        _fill_nan_log_s,
        _nearest_finite_backfill,
    )
    from sisyphus.pipeline.predict import _engine_oral_bioavailability
    from sisyphus.pk.endpoints import compute_endpoints
    from sisyphus.regimen.profile import compute_steady_state_metrics
    from sisyphus.regimen.solver import solve_regimen

    dose_mg = float(regimen.events[0].dose_mg)
    s_grid = np.geomspace(s_range[0], s_range[1], n_grid)
    compiled, realized_graph, drug, obs_node = _build_grid_engine(
        smiles, dose_mg, "oral", renal_factor, kp_method, body_weight_kg, age_years
    )
    admin_idx = compiled.state_index[drug.administration_node]

    last = float(regimen.last_dose_time_h)
    tau = _regimen_interval_h(regimen)
    t_total = last + max(tau, 24.0)
    n_points = max(2, int(round(t_total / dt_output)) + 1)
    t_grid = np.linspace(0.0, t_total, n_points)
    i1 = int(np.argmin(np.abs(np.log(s_grid))))  # grid point nearest s=1

    conc_rows: list[np.ndarray] = []
    cmaxs: list[float] = []
    aucs: list[float] = []
    fengs: list[float] = []
    is_ss = False
    t_half_h: float | None = None

    for gi, s in enumerate(s_grid):
        drug_s = dataclasses.replace(
            drug,
            enzyme_affinity={
                k: Distribution(mean=v.mean * float(s), cv=v.cv)
                for k, v in drug.enzyme_affinity.items()
            },
        )
        realized_drug_s = drug_s.realize_means()
        params_s = ResolvedParams(realized_graph, realized_drug_s)

        # (1)+(2) single-dose oral solve -> oral AUC, endpoints, f_engine.
        y0 = np.zeros(compiled.n_states)
        y0[admin_idx] = dose_mg
        sim_sd = solve(compiled, params_s, y0, t_span=(0, 24), t_min_h=0.0)
        feng = np.nan
        if sim_sd.solver_success:
            pk_sd = compute_endpoints(sim_sd, observation_node=obs_node, t_min_h=0.0)
            feng = _engine_oral_bioavailability(
                compiled, params_s, realized_drug_s, pk_sd.auc_0t.mean, obs_node
            )
            feng = feng if (feng is not None and feng > 0) else np.nan
            if gi == i1 and pk_sd.t_half is not None:
                t_half_h = float(pk_sd.t_half.mean)
        fengs.append(feng)

        # (3) multi-dose SS solve -> final-interval cmax/auc/conc.
        sim_ss = solve_regimen(compiled, params_s, regimen, t_total_h=t_total, dt_output=dt_output)
        if not sim_ss.solver_success:
            conc_rows.append(np.full(t_grid.size, np.nan))
            cmaxs.append(np.nan)
            aucs.append(np.nan)
            continue
        t_native = sim_ss.time_h
        c_native = sim_ss.concentrations[obs_node]
        conc_rows.append(np.interp(t_grid, t_native, c_native))
        mask = (t_native >= last - 1e-9) & (t_native <= last + tau + 1e-9)
        if mask.sum() >= 2:
            cmaxs.append(float(np.max(c_native[mask])))
            from sisyphus.mipd.renal_grid import _trapz
            aucs.append(float(_trapz(c_native[mask], t_native[mask])))
        else:
            cmaxs.append(np.nan)
            aucs.append(np.nan)
        if gi == i1:
            try:
                is_ss = bool(
                    compute_steady_state_metrics(sim_ss, regimen, node=obs_node).is_steady_state
                )
            except ValueError:
                is_ss = False

    cmaxs_arr = np.array(cmaxs)
    fengs_arr = np.array(fengs)
    if not np.isfinite(cmaxs_arr).any():
        raise ValueError(
            f"engine failed at all {n_grid} oral clint-scale grid points; cannot build grid"
        )
    if not np.isfinite(fengs_arr).any():
        raise ValueError(
            f"engine produced no valid oral bioavailability at any of {n_grid} grid points"
        )
    cmax = _fill_nan_log_s(cmaxs_arr, s_grid)
    auc = _fill_nan_log_s(np.array(aucs), s_grid)
    f_engine = np.clip(_fill_nan_log_s(fengs_arr, s_grid), 1e-4, 1.0)
    conc = _nearest_finite_backfill(np.array(conc_rows))
    grid = CLGrid(
        s_grid=s_grid, t_grid=t_grid, conc=conc, cmax=cmax, auc=auc, f_engine=f_engine
    )
    return grid, is_ss, t_half_h
```

Note for implementer: confirm `_default_t_grid` and `_trapz` import paths (`grid._default_t_grid` exists; `_trapz` lives in `renal_grid`). If `_default_t_grid` is unused after the rewrite (the oral grid builds its own `t_grid`), drop that import. Verify `compiled.state_index` and `compiled.n_states` names against `build_cl_grid` (grid.py:147,172).

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_mipd_oral_grid.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/oral_grid.py tests/unit/test_mipd_oral_grid.py
git commit -m "feat(mipd): build_oral_cl_grid (3-solve SS grid, f_engine + SS flag + t_half)"
```

---

## Task 3: Oral `predict_tdm` (route dispatch + adaptive inference)

Add the route pre-check; oral branch does free-F-only (B3) or free-both (shape) with the Mx2 ridge-width warning; `renal_prior_cv` → `float | None` (Mx1); conformal omitted with `cmax.ci90` fallback (M5/Mx3). IV branch unchanged.

**Files:**
- Modify: `src/sisyphus/mipd/tdm.py`
- Test: `tests/unit/test_mipd_oral_tdm.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mipd_oral_tdm.py
import numpy as np
import pytest

from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.oral_grid import build_oral_cl_grid
from sisyphus.mipd.tdm import predict_tdm
from sisyphus.regimen.types import DosingRegimen

SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def _reg(dose=100.0, tau=12.0, n=6):
    return DosingRegimen.oral_repeated(dose_mg=dose, interval_h=tau, n_doses=n)


def _engine_trough(grid, last, tau):
    from sisyphus.mipd.clgrid import CLGridForward
    f = np.array([grid.f_engine[grid.s_grid.size // 2]])
    return float(CLGridForward(grid)(f, np.ones(1))["conc_at"](last + tau))


def test_single_trough_is_f_only():
    reg = _reg()
    post = predict_tdm(SMILES, reg, [MeasuredConc(value=1.0, t=72.0)], n_grid=5)
    assert post.cl_scale is None           # F-only
    assert post.f is not None
    assert post.meta_cmax is None and post.cmax_90ci is None   # no SS conformal (M5)
    assert post.cmax.samples.size > 0      # ci90 fallback populated (Mx3)
    _ = post.cmax.ci90                       # finite tuple, no raise


def test_peak_plus_trough_is_free_both():
    reg = _reg()
    obs = [MeasuredConc(value=5.0, t=62.0), MeasuredConc(value=1.0, t=72.0)]
    post = predict_tdm(SMILES, reg, obs, n_grid=7)
    assert post.cl_scale is not None       # free-both


def test_inference_direction_low_trough_lowers_F():
    reg = _reg()
    grid, _, _ = build_oral_cl_grid(SMILES, reg, n_grid=7)
    last, tau = float(reg.last_dose_time_h), 12.0
    eng = _engine_trough(grid, last, tau)
    post = predict_tdm(SMILES, reg, [MeasuredConc(value=0.4 * eng, t=last + tau)], n_grid=7)
    assert post.f.mean < grid.f_engine[grid.s_grid.size // 2]


def test_iv_path_unchanged():
    # IV regimen still works and frees the renal latent (cl_scale untouched).
    iv = DosingRegimen.iv_infusion(dose_mg=1000.0, duration_h=1.0, interval_h=8.0, n_doses=5)
    post = predict_tdm("CCO", iv, [MeasuredConc(value=1.0, t=36.0)], n_grid=5)
    assert post.renal_scale is not None


def test_renal_prior_cv_warning_on_oral_only_when_set():
    reg = _reg()
    obs = [MeasuredConc(value=1.0, t=72.0)]
    default = predict_tdm(SMILES, reg, obs, n_grid=5)
    assert not any("renal_prior_cv" in w for w in default.warnings)
    explicit = predict_tdm(SMILES, reg, obs, n_grid=5, renal_prior_cv=0.3)
    assert any("renal_prior_cv" in w for w in explicit.warnings)


def test_attribution_honesty_trough_agrees_cmax_auc_diverge():
    # Spec §7.5 / Bx2: only the observed-trough conc is attribution-independent.
    # free-both's Cmax/AUC band is wider than free-F-only's; they are NOT equal.
    reg = _reg()
    grid, _, _ = build_oral_cl_grid(SMILES, reg, n_grid=9)
    last, tau = float(reg.last_dose_time_h), 12.0
    eng = _engine_trough(grid, last, tau)
    trough_obs = MeasuredConc(value=eng, t=last + tau)
    # single trough -> F-only; same trough + a distinct-phase peak -> free-both.
    peak = MeasuredConc(value=1.5 * eng, t=last + 2.0)
    f_only = predict_tdm(SMILES, reg, [trough_obs], n_grid=9)
    both = predict_tdm(SMILES, reg, [trough_obs, peak], n_grid=9)
    assert both.cl_scale is not None and f_only.cl_scale is None
    # AUC band strictly wider under free-both (shape marginalized, not frozen at s=1).
    def _w(p):
        lo, hi = p.auc.ci90
        return hi - lo
    assert _w(both) > _w(f_only)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_mipd_oral_tdm.py -x -q`
Expected: FAIL (oral regimen currently raises `ValueError`).

- [ ] **Step 3: Refactor `predict_tdm`**

1. Change the signature default `renal_prior_cv: float = 1.0` → `renal_prior_cv: float | None = None`.
2. Replace the IV-only guard (the `if any(ev.node != DEFAULT_IV_NODE ...)` raise) with a route dispatch:

```python
    from sisyphus.mipd._regimen import _regimen_route, _require_uniform_regimen
    route = _regimen_route(regimen)
    _require_uniform_regimen(regimen)
    if route == "oral":
        return _predict_tdm_oral(
            smiles, regimen, list(observations), covariates=covariates,
            renal_prior_cv=renal_prior_cv, n_samples=n_samples, n_grid=n_grid,
            seed=seed, kp_method=kp_method,
        )
    # ---- IV branch (unchanged) ----
    renal_cv = 1.0 if renal_prior_cv is None else renal_prior_cv
```

   Then in the existing IV body, replace `RenalCLPrior(cv=renal_prior_cv, ...)` with `RenalCLPrior(cv=renal_cv, ...)` (the None→1.0 coalesce keeps IV byte-identical).

3. Add the oral branch implementation:

```python
def _predict_tdm_oral(
    smiles, regimen, observations, *, covariates, renal_prior_cv,
    n_samples, n_grid, seed, kp_method,
):
    import dataclasses
    import numpy as np
    from sisyphus.mipd._regimen import _distinct_phases, _regimen_interval_h
    from sisyphus.mipd.clgrid import (
        CLGridForward, CLPrior, FPrior, MeasuredConc, sir_posterior_2d,
    )
    from sisyphus.mipd.oral_grid import build_oral_cl_grid

    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    body_weight_kg = covariates.body_weight_kg if covariates is not None else None
    age_years = covariates.age_years if covariates is not None else None
    warnings_list = list(covariates.warnings()) if covariates is not None else []
    if renal_prior_cv is not None:
        warnings_list.append(
            "renal_prior_cv is ignored on an oral regimen (oral frees F, not renal CL)"
        )
    rng = np.random.default_rng(seed)
    tau = _regimen_interval_h(regimen)
    has_f = any(not hasattr(o, "t") for o in observations)  # MeasuredF has no .t
    free_both = has_f or _distinct_phases(observations, tau)

    if free_both:
        grid, is_ss, _ = build_oral_cl_grid(
            smiles, regimen, n_grid=n_grid, renal_factor=renal_factor,
            body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
        )
        f0 = float(np.clip(grid.f_engine[int(np.argmin(np.abs(np.log(grid.s_grid))))], 1e-4, 1.0))
        post = sir_posterior_2d(
            FPrior(f0, 1.0),
            CLPrior(cv=1.0, s_min=float(grid.s_grid[0]), s_max=float(grid.s_grid[-1])),
            CLGridForward(grid), observations, n_samples=n_samples, rng=rng,
        )
        # Mx2: warn if the supplied distinct phases are flat-tail (ridge-wide clint).
        if _flat_tail(grid, observations, tau):
            warnings_list.append(
                "cl_scale marginal may be prior-ridge-wide; supplied phases do not "
                "identify the metabolic clint axis"
            )
    else:
        grid, is_ss, _ = build_oral_cl_grid(
            smiles, regimen, n_grid=1, s_range=(1.0, 1.0), renal_factor=renal_factor,
            body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
        )
        f0 = float(np.clip(grid.f_engine[0], 1e-4, 1.0))
        post = sir_posterior_2d(
            FPrior(f0, 1.0), CLPrior(cv=1.0, s_min=1.0, s_max=1.0),
            CLGridForward(grid), observations, n_samples=n_samples, rng=rng,
        )
        post = dataclasses.replace(post, cl_scale=None)  # F-only: no clint latent

    if not is_ss:
        warnings_list.append(
            "observed regimen has not reached steady state at the grid horizon; "
            "the SS trough/posterior may be biased"
        )
    return dataclasses.replace(
        post, renal_scale=None, meta_cmax=None, cmax_90ci=None,
        warnings=tuple(warnings_list),
    )
```

   Add the `_flat_tail` helper (Mx2) in `tdm.py` (or `_regimen.py`): compare `|Δlog c|` between the two most phase-separated MeasuredConc times against the engine s=1 curve; return True when below a small tolerance. Minimal form:

```python
def _flat_tail(grid, observations, tau, *, tol=0.2):
    import numpy as np
    ts = sorted(float(o.t) for o in observations if hasattr(o, "t"))
    if len(ts) < 2:
        return False
    s1 = np.ones(1)
    c = [float(grid.conc_at(s1, t)[0]) for t in (ts[0], ts[-1])]
    if min(c) <= 0:
        return False
    return abs(np.log(c[1]) - np.log(c[0])) < tol
```

   Confirm `dataclasses.replace` covers all the `PosteriorPK` fields set to None (it leaves `f`/`cmax`/`auc`/`cl_scale`/`n_eff` from `post`). Confirm `PosteriorPK` has `renal_scale`, `meta_cmax`, `cmax_90ci`, `warnings` fields (core.py:211-223) — set them via `replace`.

- [ ] **Step 4: Run tests + IV regression**

Run: `pytest tests/unit/test_mipd_oral_tdm.py tests/unit/test_tdm.py tests/unit/test_mipd_renal*.py -q`
Expected: PASS (oral works; all existing IV TDM tests still pass).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/tdm.py tests/unit/test_mipd_oral_tdm.py
git commit -m "feat(mipd): oral predict_tdm (F-only/free-both adaptive, route dispatch)"
```

---

## Task 4: `DoseRecommendation` contract relaxation (B2)

Make the MIPD `DoseRecommendation` carry an oral recommendation (no renal latent; F[+clint] latents).

**Files:**
- Modify: `src/sisyphus/mipd/dosing.py` (the `DoseRecommendation` dataclass, ~line 76)
- Test: `tests/unit/test_mipd_dosing.py` (add a contract test)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_mipd_dosing.py
def test_dose_recommendation_optional_renal_and_latents():
    from sisyphus.mipd.core import Posterior
    from sisyphus.mipd.dosing import DoseRecommendation, DoseTarget
    import numpy as np
    rec = DoseRecommendation(
        dose_mg=100.0, interval_h=12.0, attainment_prob=0.9,
        cmax=Posterior(np.ones(3)), trough=Posterior(np.ones(3)),
        auc24=Posterior(np.ones(3)),
        target=DoseTarget(trough=(0.5, 2.0)),
        candidates=(), n_eff=3.0, warnings=(),
        f=Posterior(np.ones(3)),          # oral latent
    )
    assert rec.renal_scale is None        # defaults None now
    assert rec.f is not None and rec.cl_scale is None
```

(Use the real `DoseTarget` constructor signature from `dosing.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_mipd_dosing.py::test_dose_recommendation_optional_renal_and_latents -x -q`
Expected: FAIL (`renal_scale` is required; `f`/`cl_scale` don't exist).

- [ ] **Step 3: Relax + reorder the dataclass**

In `src/sisyphus/mipd/dosing.py`, change `DoseRecommendation` so the three defaulted fields trail the required ones:

```python
@dataclass(frozen=True)
class DoseRecommendation:
    """The recommended regimen and the exposure posteriors it produces."""
    dose_mg: float
    interval_h: float
    attainment_prob: float
    cmax: Posterior
    trough: Posterior
    auc24: Posterior
    target: DoseTarget
    candidates: tuple[CandidateEval, ...]
    n_eff: float
    warnings: tuple[str, ...]
    renal_scale: Posterior | None = None   # IV latent (None for oral)
    f: Posterior | None = None             # oral F latent
    cl_scale: Posterior | None = None      # oral metabolic-clint latent (free-both)
```

Update the existing IV construction site (dosing.py:337) to keyword args (it already is); the reorder is safe because all construction is keyword-based.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_mipd_dosing.py -q`
Expected: PASS (new contract test + all existing dosing tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/dosing.py tests/unit/test_mipd_dosing.py
git commit -m "feat(mipd): DoseRecommendation carries optional oral F/clint latents"
```

---

## Task 5: Oral `_interval_reference` + oral `recommend_dose` (B1/I2/Bx3)

The oral sibling: per-candidate-τ grid, trough/cmax/auc read off the **F-scaled forward state** (B1), half-life-aware `n_doses` (I2/Bx4), τ-sweep shape caveat (Bx3). The post-`q_ref` LTI dose-solve core is reused verbatim.

**Files:**
- Modify: `src/sisyphus/mipd/dosing.py` (route dispatch + `_interval_reference_oral` + oral branch)
- Test: `tests/unit/test_mipd_oral_dosing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mipd_oral_dosing.py
import numpy as np
import pytest

from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.dosing import DoseTarget, recommend_dose
from sisyphus.regimen.types import DosingRegimen

SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def _reg(dose=100.0, tau=12.0, n=6):
    return DosingRegimen.oral_repeated(dose_mg=dose, interval_h=tau, n_doses=n)


def test_oral_recommend_returns_no_renal_latent():
    rec = recommend_dose(
        SMILES, _reg(), [MeasuredConc(value=1.0, t=72.0)],
        DoseTarget(trough=(0.5, 2.0)), candidate_intervals=(12.0, 24.0), n_grid=5,
    )
    assert rec.renal_scale is None
    assert rec.f is not None
    assert rec.dose_mg > 0


def test_oral_b1_trough_carries_F_scale():
    # The oral interval-reference trough must scale with F (not the raw engine conc).
    from sisyphus.mipd.clgrid import CLGridForward
    from sisyphus.mipd.dosing import _interval_reference_oral
    from sisyphus.mipd.oral_grid import build_oral_cl_grid
    reg = _reg()
    grid, _, _ = build_oral_cl_grid(SMILES, reg, n_grid=5)
    s1 = grid.s_grid.size // 2
    f = np.array([2.0 * grid.f_engine[s1]])  # F = 2*f_engine -> scale 2
    s = np.array([grid.s_grid[s1]])
    last, tau = float(reg.last_dose_time_h), 12.0
    raw = float(grid.conc_at(s, last + tau))
    q, _ = _interval_reference_oral(SMILES, reg, tau, f, s, n_grid=5)
    assert q["trough"][0] == pytest.approx(2.0 * raw, rel=0.05)


def test_oral_auc24_factor():
    from sisyphus.mipd.dosing import _interval_reference_oral
    from sisyphus.mipd.oral_grid import build_oral_cl_grid
    reg = _reg(tau=8.0)
    grid, _, _ = build_oral_cl_grid(SMILES, reg, n_grid=5)
    s1 = grid.s_grid.size // 2
    f = np.array([grid.f_engine[s1]]); s = np.array([grid.s_grid[s1]])
    q, _ = _interval_reference_oral(SMILES, reg, 8.0, f, s, n_grid=5)
    # auc24 == interval-auc * 24/8 == interval-auc * 3
    assert q["auc24"][0] == pytest.approx(q["auc"][0] * 3.0, rel=1e-6)


def test_oral_tau_change_emits_shape_caveat():
    rec = recommend_dose(
        SMILES, _reg(tau=12.0), [MeasuredConc(value=1.0, t=72.0)],
        DoseTarget(trough=(0.5, 2.0)), candidate_intervals=(8.0, 24.0), n_grid=5,
    )
    assert any("shape" in w.lower() for w in rec.warnings)
```

(Reconcile `_interval_reference_oral`'s return-dict keys with the IV `_interval_reference` — it must expose at least `trough`, `cmax`, `auc`, `auc24`. Check the IV helper's keys at dosing.py:196-201 and mirror them.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_mipd_oral_dosing.py -x -q`
Expected: FAIL (oral regimen raises; `_interval_reference_oral` doesn't exist).

- [ ] **Step 3: Implement the oral dosing branch**

1. Change `recommend_dose`'s `renal_prior_cv: float = 1.0` → `renal_prior_cv: float | None = None` and forward it verbatim to `predict_tdm`.
2. Replace the IV-only guard with route dispatch (mirror Task 3): on `"oral"`, call a new `_recommend_dose_oral(...)`; else the existing IV body with `renal_prior_cv` forwarded (the None→1.0 coalesce lives in `predict_tdm`, so IV stays byte-identical).
3. `_recommend_dose_oral` mirrors the IV orchestrator but threads `f`(+`cl_scale`) and uses `_interval_reference_oral`:

```python
def _interval_reference_oral(
    smiles, reg_tau, tau, f_samples, s_samples, *, renal_factor=1.0,
    body_weight_kg=None, age_years=None, n_grid=13, kp_method="rodgers_rowland",
):
    import numpy as np
    from sisyphus.mipd.clgrid import CLGridForward
    from sisyphus.mipd.oral_grid import build_oral_cl_grid
    grid, _, _ = build_oral_cl_grid(
        smiles, reg_tau, n_grid=n_grid, renal_factor=renal_factor,
        body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
    )
    state = CLGridForward(grid)(np.asarray(f_samples), np.asarray(s_samples))
    last = float(reg_tau.last_dose_time_h)
    trough = state["conc_at"](last + tau)
    q = {
        "trough": np.asarray(trough),
        "cmax": np.asarray(state["cmax"]),
        "auc": np.asarray(state["auc"]),
        "auc24": np.asarray(state["auc"]) * 24.0 / tau,
    }
    d_ref = float(reg_tau.events[0].dose_mg)
    return q, d_ref
```

```python
def _recommend_dose_oral(
    smiles, regimen, observations, target, *, covariates, candidate_intervals,
    dose_step_mg, dose_bounds_mg, renal_prior_cv, n_samples, n_grid, seed, kp_method,
):
    import numpy as np
    from sisyphus.mipd.tdm import predict_tdm
    from sisyphus.regimen.types import DosingRegimen
    from sisyphus.mipd._regimen import _regimen_interval_h

    post = predict_tdm(
        smiles, regimen, observations, covariates=covariates,
        renal_prior_cv=renal_prior_cv, n_samples=n_samples, n_grid=n_grid,
        seed=seed, kp_method=kp_method,
    )
    f_samples = post.f.samples
    s_samples = post.cl_scale.samples if post.cl_scale is not None else np.ones(f_samples.size)
    warnings_list = list(post.warnings)

    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    body_weight_kg = covariates.body_weight_kg if covariates is not None else None
    age_years = covariates.age_years if covariates is not None else None

    cur_dose = float(regimen.events[0].dose_mg)
    cur_last = float(regimen.last_dose_time_h)
    cur_tau = _regimen_interval_h(regimen)
    _grid, _is_ss, t_half = build_oral_cl_grid(  # reuse the observed-regimen t_half
        smiles, regimen, n_grid=1, s_range=(1.0, 1.0), renal_factor=renal_factor,
        body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
    )
    base = tuple(candidate_intervals) if candidate_intervals is not None else (8.0, 12.0, 24.0)
    taus = sorted(set(base + (cur_tau,)))
    if any(abs(t - cur_tau) > 1e-9 for t in taus):
        warnings_list.append(
            "candidate intervals other than the observed interval extrapolate the "
            "curve shape (engine s=1 estimate); free-both for a shape-identified band"
        )

    rows = []
    for tau in taus:
        if t_half is not None:
            n_doses = max(5, int(np.ceil(5.0 * t_half / tau)))
        else:
            n_doses = max(2, int(round(cur_last / tau)) + 1)
        reg_tau = DosingRegimen.oral_repeated(dose_mg=cur_dose, interval_h=tau, n_doses=n_doses)
        q_ref, d_ref = _interval_reference_oral(
            smiles, reg_tau, tau, f_samples, s_samples, renal_factor=renal_factor,
            body_weight_kg=body_weight_kg, age_years=age_years, n_grid=n_grid,
            kp_method=kp_method,
        )
        rows.append(_eval_candidate(q_ref, d_ref, tau, target, dose_step_mg, dose_bounds_mg))
    win = _select_winner(rows)   # the verbatim IV winner-selection
    return DoseRecommendation(
        dose_mg=win.dose_mg, interval_h=win.interval_h, attainment_prob=win.attainment_prob,
        cmax=..., trough=..., auc24=...,        # win_q * win_m, as in the IV body
        target=target, candidates=tuple(r[0] for r in rows),
        n_eff=post.n_eff, warnings=tuple(warnings_list),
        renal_scale=None, f=post.f, cl_scale=post.cl_scale,
    )
```

   **Implementer (DRY refactor — do this first):** the IV `recommend_dose` body from just after the `_interval_reference(...)` call through the `return DoseRecommendation(...)` (dosing.py ~**301-348**: the per-`tau` `_sample_m_intervals`/`_max_overlap_region`/`_center_m`/`_attainment` evaluation, `rows.append(...)`, winner selection, infeasibility warning, and the final construction) is **identical** for oral except (a) it consumes the oral `q_ref` and (b) the final `DoseRecommendation` sets `renal_scale=None, f=post.f, cl_scale=post.cl_scale` instead of `renal_scale=post.renal_scale`. Extract that block into shared helpers (`_eval_candidate` for the per-`tau` row, `_select_winner` + the infeasibility-warning for the tail) so the IV and oral orchestrators call them identically; do not duplicate the LTI math. Read dosing.py:301-348 for the exact existing code to extract — it already exists in the file you are editing.

- [ ] **Step 4: Run tests + IV regression**

Run: `pytest tests/unit/test_mipd_oral_dosing.py tests/unit/test_mipd_dosing.py -q`
Expected: PASS (oral dosing + all existing IV dosing tests, including `test_recommend_renal_scale_shifts_with_observation`).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/dosing.py tests/unit/test_mipd_oral_dosing.py
git commit -m "feat(mipd): oral recommend_dose (F-scaled interval reference, t_half n_doses)"
```

---

## Task 6: Exports, integration, and full regression

**Files:**
- Modify: `src/sisyphus/mipd/__init__.py` (export `build_oral_cl_grid` if the package exports the renal/cl grid builders)
- Test: `tests/unit/test_mipd_oral_integration.py`

- [ ] **Step 1: Write integration + invariant tests**

```python
# tests/unit/test_mipd_oral_integration.py
import pytest
from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.tdm import predict_tdm
from sisyphus.mipd.dosing import DoseTarget, recommend_dose
from sisyphus.regimen.types import DosingRegimen

SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def test_mixed_route_regimen_raises():
    oral = DosingRegimen.oral_repeated(dose_mg=100.0, interval_h=12.0, n_doses=3)
    iv = DosingRegimen.iv_infusion(dose_mg=100.0, duration_h=1.0, interval_h=12.0, n_doses=1)
    mixed = DosingRegimen(events=oral.events[:2] + iv.events[:1])
    with pytest.raises(ValueError):
        predict_tdm(SMILES, mixed, [MeasuredConc(value=1.0, t=12.0)])


def test_nonuniform_oral_raises():
    import dataclasses
    reg = DosingRegimen.oral_repeated(dose_mg=100.0, interval_h=12.0, n_doses=4)
    ev = list(reg.events)
    ev[2] = dataclasses.replace(ev[2], time_h=ev[2].time_h + 6.0)
    with pytest.raises(ValueError):
        predict_tdm(SMILES, DosingRegimen(events=ev), [MeasuredConc(value=1.0, t=48.0)])
```

- [ ] **Step 2: Run + verify mixed/non-uniform reject**

Run: `pytest tests/unit/test_mipd_oral_integration.py -q`
Expected: PASS.

- [ ] **Step 3: Export + ruff**

Add `build_oral_cl_grid` to `mipd/__init__.py` exports if sibling grid builders are exported there (check first; if `__init__` only re-exports the public API like `predict_tdm`/`recommend_dose`, leave the grid builder internal). Run `ruff check src/sisyphus/mipd tests/unit` and fix any lint.

- [ ] **Step 4: Full MIPD + IV-invariance regression**

Run: `pytest tests/unit/test_mipd_*.py tests/unit/test_tdm*.py -q`
Expected: PASS — every existing IV TDM/dosing test unchanged (the 2.731 headline and renal path are untouched), all new oral tests green.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/__init__.py tests/unit/test_mipd_oral_integration.py
git commit -m "feat(mipd): oral SS-TDM exports + mixed/non-uniform route guards"
```

---

## Final review

After all 6 tasks: dispatch a final code-reviewer subagent over the whole branch diff (engine/predict untouched? IV numerically bit-identical? no identity-blind violation? every spec finding B1–B3/I1–I2/M1–M7/Bx1–Bx6/Mx1–Mx3 reflected?), then use **superpowers:finishing-a-development-branch** (the user finishes features via option 2 = push + PR, then merges).

## Acceptance criteria

Spec **§7 is the authoritative test list (18 cases)**. This plan spells out the load-bearing and contract tests inline; the per-task spec-compliance review must confirm every applicable §7 case is present before the task is marked done. §7 cases not written out above but required: **#5** attribution honesty (added to Task 3), **#7** free-F-only internal-`s`≈1 assertion (Task 3), **#13** auc24 factor (Task 5), **#15** flat-tail Mx2 warning (Task 3 — add a free-both case where the two concs are both on the elimination tail and assert the "prior-ridge-wide" warning fires), **#18** `cmax.ci90` finite-tuple fallback (Task 3). Engine/predict diff must be **zero lines**; the IV numerical path must be bit-identical (verified by the unchanged IV test suite each task).

## Self-review notes (cross-task type consistency)

- `build_oral_cl_grid` returns `tuple[CLGrid, bool, float | None]` everywhere it's called (Tasks 2, 3, 5) — callers unpack 3 values.
- `_regimen_route`/`_require_uniform_regimen`/`_regimen_interval_h`/`_distinct_phases` live only in `mipd/_regimen.py` (Task 1); Tasks 2/3/5 import from there.
- `DoseRecommendation` gains `f`/`cl_scale`/`renal_scale` as trailing `| None = None` (Task 4) before Task 5 constructs the oral variant.
- `_interval_reference_oral` returns a dict with keys `trough`/`cmax`/`auc`/`auc24` (Task 5) — reconcile against the IV `_interval_reference` keys before wiring `_solve_dose_from_q`.
- `predict_tdm`/`recommend_dose` both take `renal_prior_cv: float | None = None`; the None→1.0 coalesce is in the IV branch of `predict_tdm` only (Tasks 3, 5).
