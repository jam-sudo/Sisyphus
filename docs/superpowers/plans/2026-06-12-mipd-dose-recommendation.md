# MIPD Dose Recommendation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `recommend_dose()` — given an IV regimen, a measured trough, and a clinical target (constraints on trough/Cmax/AUC24), recommend the (dose, interval) maximizing the probability the target is met under the conditioned posterior.

**Architecture:** A new self-contained `src/sisyphus/mipd/dosing.py`. The patient's renal-CL posterior is inferred once via `predict_tdm` (a patient property, regimen-invariant). For each candidate interval τ, one engine re-solve (`build_renal_cl_grid`) yields per-posterior-sample reference exposures; the dose is then solved **analytically** because the engine is linear in dose (LTI) — a max-interval-overlap sweep over the per-sample feasible dose-multiplier ranges. Reuses the forward model verbatim; `engine/`, `predict()`, and `PosteriorPK` are untouched.

**Tech Stack:** Python 3.10+, numpy, frozen dataclasses, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-12-mipd-dose-recommendation-design.md`

**Conventions to follow** (from the existing `mipd/` modules):
- `from __future__ import annotations` at the top of every module.
- Module-level docstring; `logging` not `print`.
- Frozen dataclasses; type hints on all public signatures.
- `ruff` line length 100.
- Commit messages: `type(mipd): description` — NO `Co-Authored-By` / AI trailer (project directive).
- Reference SMILES used in existing tests: `ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"` (high-fup, renally cleared — the right drug for renal-CL TDM tests).

**Reused APIs (verified — use exactly these signatures):**
- `sisyphus.mipd.core.Posterior(samples: np.ndarray)` → `.point` (median), `.mean`, `.ci90`.
- `sisyphus.mipd.tdm.predict_tdm(smiles, regimen, observations, *, covariates=None, renal_prior_cv=1.0, n_samples=20000, n_grid=13, seed=0, kp_method="rodgers_rowland") -> PosteriorPK`. Returns `post.renal_scale: Posterior`, `post.n_eff: float`, `post.warnings: tuple[str,...]`.
- `sisyphus.mipd.renal_grid.build_renal_cl_grid(smiles, regimen, *, n_grid=13, r_range=(0.2,5.0), renal_factor=1.0, body_weight_kg=None, age_years=None, kp_method="rodgers_rowland", dt_output=0.1) -> RenalCLGrid`.
- `RenalCLGrid.conc_at(r: np.ndarray, t: float) -> np.ndarray` (raises if `t` outside the grid horizon).
- `sisyphus.mipd.renal_grid.RenalCLForward(grid).__call__(r) -> {"renal_scale","cmax","auc","conc_at"}`.
- `sisyphus.mipd.renal_grid._regimen_interval_h(regimen) -> float` (module-private; the orchestrator passes τ explicitly instead — do NOT import it into `dosing.py`).
- `sisyphus.mipd.clgrid.MeasuredConc(value: float, t: float, cv: float = 0.25)`.
- `sisyphus.regimen.types.DosingRegimen.iv_infusion(dose_mg, duration_h, interval_h, n_doses)`; `DEFAULT_IV_NODE = "venous_blood"`; events carry `.dose_mg`, `.duration_h`, `.node`; `regimen.last_dose_time_h`.
- `sisyphus.mipd.covariates.Covariates` → `.renal_factor()`, `.body_weight_kg`, `.age_years`.

---

## File Structure

- **Create** `src/sisyphus/mipd/dosing.py` — all types + the analytic solve + `recommend_dose`. One responsibility: turn a posterior + target into a (dose, interval) recommendation. (mipd/ goes 9→10 `.py` files — within the 20-file ceiling.)
- **Modify** `src/sisyphus/mipd/__init__.py` — export the new public names.
- **Create** `tests/unit/test_mipd_dosing.py` — type/validation + analytic-solve unit tests + behavioral integration tests.

Task order: types → analytic solve (pure numpy) → per-interval reference (engine) → orchestrator → exports.

---

### Task 1: Constraint / DoseTarget / CandidateEval / DoseRecommendation types

**Files:**
- Create: `src/sisyphus/mipd/dosing.py`
- Test: `tests/unit/test_mipd_dosing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mipd_dosing.py
"""Unit + integration tests for mipd.dosing (dose recommendation / target attainment)."""
import numpy as np
import pytest

from sisyphus.mipd.clgrid import MeasuredConc
from sisyphus.mipd.covariates import Covariates
from sisyphus.mipd.dosing import (
    CandidateEval,
    Constraint,
    DoseRecommendation,
    DoseTarget,
    recommend_dose,
)
from sisyphus.regimen.types import DosingRegimen

ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"  # high-fup, renally cleared


def _iv_regimen():
    return DosingRegimen.iv_infusion(dose_mg=50.0, duration_h=0.5, interval_h=8.0, n_doses=5)


def test_constraint_rejects_unknown_quantity():
    with pytest.raises(ValueError, match="quantity"):
        Constraint(quantity="halflife", low=1.0)


def test_constraint_rejects_no_bound():
    with pytest.raises(ValueError, match="at least one"):
        Constraint(quantity="trough")


def test_constraint_rejects_low_above_high():
    with pytest.raises(ValueError, match="low"):
        Constraint(quantity="trough", low=5.0, high=2.0)


def test_constraint_accepts_one_sided():
    c = Constraint(quantity="cmax", high=10.0)
    assert c.low is None and c.high == 10.0


def test_dose_target_rejects_empty():
    with pytest.raises(ValueError, match="at least one constraint"):
        DoseTarget(constraints=())
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_mipd_dosing.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sisyphus.mipd.dosing'`.

- [ ] **Step 3: Implement the types**

```python
# src/sisyphus/mipd/dosing.py
"""Dose recommendation (target attainment) over the IV steady-state TDM posterior.

``predict_tdm`` infers the patient's renal-clearance posterior from a measured
steady-state trough. This module closes the clinical loop: given a target (a set of
constraints on the steady-state trough, peak Cmax, and/or AUC24), recommend the
(dose, interval) that maximizes the probability the target is met under that
posterior.

The engine is **linear in dose** (concentration-independent clearance — no saturable
Michaelis-Menten in this path), so at a fixed disposition every steady-state exposure
scales linearly with dose. The dose knob is therefore inverted *analytically* (one
solve per interval, then a max-interval-overlap sweep over per-sample feasible
dose-multiplier ranges); only the interval knob (nonlinear accumulation) costs an
engine re-solve. See docs/superpowers/specs/2026-06-12-mipd-dose-recommendation-design.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sisyphus.mipd.core import Posterior

_QUANTITIES = ("trough", "cmax", "auc24")


@dataclass(frozen=True)
class Constraint:
    """A bound on one steady-state PK quantity, evaluated under the posterior.

    ``quantity`` is one of ``"trough"`` / ``"cmax"`` / ``"auc24"``. At least one of
    ``low`` / ``high`` must be set (mg/L for trough/cmax, mg*h/L for auc24).
    """

    quantity: str
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if self.quantity not in _QUANTITIES:
            raise ValueError(f"quantity must be one of {_QUANTITIES}, got {self.quantity!r}")
        if self.low is None and self.high is None:
            raise ValueError("Constraint needs at least one of low/high")
        if self.low is not None and self.low < 0:
            raise ValueError(f"low must be >= 0, got {self.low}")
        if self.high is not None and self.high <= 0:
            raise ValueError(f"high must be > 0, got {self.high}")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"low {self.low} > high {self.high}")


@dataclass(frozen=True)
class DoseTarget:
    """A set of constraints. Attainment = P(ALL constraints satisfied) under the posterior."""

    constraints: tuple[Constraint, ...]

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError("DoseTarget needs at least one constraint")


@dataclass(frozen=True)
class CandidateEval:
    """One (dose, interval) row of the recommendation search — for transparency."""

    dose_mg: float
    interval_h: float
    attainment_prob: float
    trough_median: float
    cmax_median: float
    auc24_median: float


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
    renal_scale: Posterior
    n_eff: float
    warnings: tuple[str, ...]
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/unit/test_mipd_dosing.py -x -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/dosing.py tests/unit/test_mipd_dosing.py
git commit -m "feat(mipd): dose-recommendation Constraint/DoseTarget/result types"
```

---

### Task 2: Analytic LTI dose solve (pure numpy)

The core math. Pure numpy over synthetic per-sample arrays — no engine, fast, and **stack-independent** (no floating engine numerics). This is the load-bearing algorithm.

**Files:**
- Modify: `src/sisyphus/mipd/dosing.py`
- Test: `tests/unit/test_mipd_dosing.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_mipd_dosing.py
from sisyphus.mipd.dosing import (  # noqa: E402  (extend the existing import)
    _attainment,
    _center_m,
    _max_overlap_region,
    _sample_m_intervals,
)


def test_sample_m_intervals_two_sided_window():
    # q_ref = 1.0 for all; trough window [2, 4] -> m must be in [2, 4].
    q_ref = {"trough": np.array([1.0, 1.0, 1.0])}
    target = DoseTarget((Constraint("trough", low=2.0, high=4.0),))
    m_lo, m_hi = _sample_m_intervals(q_ref, target)
    assert np.allclose(m_lo, 2.0)
    assert np.allclose(m_hi, 4.0)


def test_sample_m_intervals_unbounded_sides():
    q_ref = {"cmax": np.array([2.0, 2.0])}
    # ceiling only -> m_lo == 0, m_hi == high/q
    lo, hi = _sample_m_intervals(q_ref, DoseTarget((Constraint("cmax", high=10.0),)))
    assert np.allclose(lo, 0.0) and np.allclose(hi, 5.0)
    # floor only -> m_lo == low/q, m_hi == inf
    lo, hi = _sample_m_intervals(q_ref, DoseTarget((Constraint("cmax", low=4.0),)))
    assert np.allclose(lo, 2.0) and np.all(np.isinf(hi))


def test_max_overlap_region_picks_densest_segment():
    # intervals: [0,2], [1,3], [1,4]  -> max overlap (3) on [1, 2]
    m_lo = np.array([0.0, 1.0, 1.0])
    m_hi = np.array([2.0, 3.0, 4.0])
    a, b, count = _max_overlap_region(m_lo, m_hi)
    assert count == 3
    assert a == pytest.approx(1.0) and b == pytest.approx(2.0)


def test_attainment_counts_covering_intervals():
    m_lo = np.array([0.0, 1.0, 1.0])
    m_hi = np.array([2.0, 3.0, 4.0])
    assert _attainment(1.5, m_lo, m_hi) == pytest.approx(1.0)   # all three cover 1.5
    assert _attainment(3.5, m_lo, m_hi) == pytest.approx(1.0 / 3.0)  # only [1,4]


def test_center_m_rules():
    assert _center_m(2.0, 8.0) == pytest.approx(4.0)        # bounded -> geometric mid sqrt(16)
    assert _center_m(3.0, math.inf) == pytest.approx(3.0)   # only floors -> smallest (a)
    assert _center_m(0.0, 5.0) == pytest.approx(5.0)        # only ceilings -> largest (b)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_mipd_dosing.py -k "m_intervals or overlap or attainment or center_m" -q`
Expected: FAIL with `ImportError` (helpers not defined).

- [ ] **Step 3: Implement the helpers**

```python
# append to src/sisyphus/mipd/dosing.py

def _sample_m_intervals(
    q_ref: dict[str, np.ndarray], target: DoseTarget
) -> tuple[np.ndarray, np.ndarray]:
    """Per posterior-sample feasible dose-multiplier interval ``[m_lo, m_hi]``.

    Each exposure scales linearly with the dose multiplier ``m`` (LTI). A constraint
    ``low <= m*q_ref <= high`` becomes ``m in [low/q_ref, high/q_ref]``; intersecting
    across constraints gives ``[m_lo[i], m_hi[i]]`` per sample (0 / +inf when a side
    is unbounded).
    """
    n = len(next(iter(q_ref.values())))
    m_lo = np.zeros(n)
    m_hi = np.full(n, np.inf)
    for c in target.constraints:
        q = np.maximum(np.asarray(q_ref[c.quantity], dtype=float), 1e-300)
        if c.low is not None:
            m_lo = np.maximum(m_lo, c.low / q)
        if c.high is not None:
            m_hi = np.minimum(m_hi, c.high / q)
    return m_lo, m_hi


def _attainment(m: float, m_lo: np.ndarray, m_hi: np.ndarray) -> float:
    """Fraction of posterior samples whose feasible interval covers dose-multiplier ``m``."""
    return float(np.mean((m_lo <= m) & (m <= m_hi)))


def _max_overlap_region(
    m_lo: np.ndarray, m_hi: np.ndarray
) -> tuple[float, float, int]:
    """Dose-multiplier segment ``[a, b]`` where the most sample-intervals overlap.

    Classic max-interval-overlap sweep. At a tie, a start is ordered before an end so
    a point shared by ``[., x]`` and ``[x, .]`` counts both (closed intervals). Returns
    ``(a, b, count)``; ``b`` may be ``inf`` (only-floor constraints). Empty sample
    intervals (``m_lo > m_hi``) are dropped.
    """
    feasible = m_lo <= m_hi
    los = m_lo[feasible]
    his = m_hi[feasible]
    if los.size == 0:
        return (0.0, 0.0, 0)
    pts = np.concatenate([los, his])
    kinds = np.concatenate([np.ones(los.size), -np.ones(his.size)])  # +1 start, -1 end
    order = np.lexsort((-kinds, pts))  # by point asc; starts (+1) before ends (-1) at ties
    pts = pts[order]
    cov = np.cumsum(kinds[order])
    best_i = int(np.argmax(cov))
    a = float(pts[best_i])
    b = float(pts[best_i + 1]) if best_i + 1 < pts.size else a
    return (a, b, int(cov[best_i]))


def _center_m(a: float, b: float) -> float:
    """Pick the dose multiplier within the max-overlap region ``[a, b]``.

    Bounded window -> geometric midpoint (max margin). Only floors (b == inf) ->
    smallest dose meeting them (``a``). Only ceilings (a == 0) -> largest dose under
    them (``b``).
    """
    if not math.isfinite(b):
        return a if a > 0.0 else 1.0
    if a <= 0.0:
        return b
    return math.sqrt(a * b)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/unit/test_mipd_dosing.py -k "m_intervals or overlap or attainment or center_m" -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/dosing.py tests/unit/test_mipd_dosing.py
git commit -m "feat(mipd): analytic LTI dose solve (max-interval-overlap sweep)"
```

---

### Task 3: Per-interval reference exposures + LTI exactness (the load-bearing premise)

**Files:**
- Modify: `src/sisyphus/mipd/dosing.py`
- Test: `tests/unit/test_mipd_dosing.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_mipd_dosing.py
from sisyphus.mipd.dosing import _interval_reference  # noqa: E402


def test_lti_exactness_grid_scales_linearly_with_dose():
    # The whole layer rests on the engine being linear in dose. Doubling the dose must
    # exactly double steady-state Cmax and AUC at every renal scale. A future saturable
    # nonlinearity would fail HERE.
    from sisyphus.mipd.renal_grid import build_renal_cl_grid

    g1 = build_renal_cl_grid(
        ATENOLOL, DosingRegimen.iv_infusion(50.0, 0.5, 8.0, 5), n_grid=5
    )
    g2 = build_renal_cl_grid(
        ATENOLOL, DosingRegimen.iv_infusion(100.0, 0.5, 8.0, 5), n_grid=5
    )
    assert np.allclose(g2.cmax, 2.0 * g1.cmax, rtol=1e-6)
    assert np.allclose(g2.auc, 2.0 * g1.auc, rtol=1e-6)


def test_interval_reference_returns_quantities_at_reference_dose():
    reg = _iv_regimen()
    r = np.array([1.0, 1.0, 1.0])
    q_ref, d_ref = _interval_reference(
        ATENOLOL, reg, 8.0, r, renal_factor=1.0, body_weight_kg=None,
        age_years=None, n_grid=5, kp_method="rodgers_rowland",
    )
    assert d_ref == 50.0
    assert set(q_ref) == {"trough", "cmax", "auc24"}
    for key in q_ref:
        assert q_ref[key].shape == (3,)
        assert np.all(q_ref[key] > 0)
    # trough must equal the grid's own conc at the end of the final interval at r=1.
    from sisyphus.mipd.renal_grid import build_renal_cl_grid

    g = build_renal_cl_grid(ATENOLOL, reg, n_grid=5)
    expect = float(g.conc_at(np.array([1.0]), reg.last_dose_time_h + 8.0)[0])
    assert q_ref["trough"][0] == pytest.approx(expect, rel=1e-9)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_mipd_dosing.py -k "lti or interval_reference" -q`
Expected: `test_lti_exactness...` PASSES already (it only uses existing code); `test_interval_reference...` FAILS with `ImportError: cannot import name '_interval_reference'`.

- [ ] **Step 3: Implement `_interval_reference`**

```python
# append to src/sisyphus/mipd/dosing.py

def _interval_reference(
    smiles: str,
    regimen,
    tau: float,
    r_samples: np.ndarray,
    *,
    renal_factor: float,
    body_weight_kg: float | None,
    age_years: float | None,
    n_grid: int,
    kp_method: str,
) -> tuple[dict[str, np.ndarray], float]:
    """Per posterior-sample steady-state exposures at the regimen's reference dose.

    Builds one renal-CL grid at this interval (one engine re-solve), then reads each
    quantity at the posterior's renal-scale samples: ``trough`` = the curve at the end
    of the final dosing interval; ``cmax`` / per-interval ``auc`` from the forward;
    ``auc24`` = per-interval AUC * (24/tau). Returns the quantity dict and the
    reference dose ``D_ref`` (= the regimen's per-dose amount).
    """
    from sisyphus.mipd.renal_grid import RenalCLForward, build_renal_cl_grid

    grid = build_renal_cl_grid(
        smiles, regimen, n_grid=n_grid, renal_factor=renal_factor,
        body_weight_kg=body_weight_kg, age_years=age_years, kp_method=kp_method,
    )
    state = RenalCLForward(grid)(r_samples)
    trough = grid.conc_at(r_samples, float(regimen.last_dose_time_h) + tau)
    q_ref = {
        "trough": np.asarray(trough, dtype=float),
        "cmax": np.asarray(state["cmax"], dtype=float),
        "auc24": np.asarray(state["auc"], dtype=float) * (24.0 / tau),
    }
    d_ref = float(regimen.events[0].dose_mg)
    return q_ref, d_ref
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/unit/test_mipd_dosing.py -k "lti or interval_reference" -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/dosing.py tests/unit/test_mipd_dosing.py
git commit -m "feat(mipd): per-interval reference exposures + LTI-exactness guard"
```

---

### Task 4: `recommend_dose` orchestration

**Files:**
- Modify: `src/sisyphus/mipd/dosing.py`
- Test: `tests/unit/test_mipd_dosing.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_mipd_dosing.py


def _engine_trough_at_unit_scale(reg, dose_mg=50.0):
    """The engine's own r=1 steady-state trough — a stack-independent anchor."""
    from sisyphus.mipd.renal_grid import build_renal_cl_grid

    g = build_renal_cl_grid(ATENOLOL, reg, n_grid=5)
    return float(g.conc_at(np.array([1.0]), reg.last_dose_time_h + 8.0)[0])


def test_recommend_rejects_oral_regimen():
    oral = DosingRegimen.oral_repeated(dose_mg=50.0, interval_h=8.0, n_doses=3)
    with pytest.raises(ValueError, match="IV"):
        recommend_dose(ATENOLOL, oral, [], DoseTarget((Constraint("trough", low=0.1),)))


def test_recommend_hits_feasible_trough_window():
    # Condition on a trough at the r=1 prediction so the posterior concentrates near r=1,
    # then ask for a window centered at 1.5x that trough -> recommender scales the dose ~1.5x
    # and the median trough lands in the window. Window defined from the engine's own
    # prediction -> stack-independent.
    reg = _iv_regimen()
    base = _engine_trough_at_unit_scale(reg)
    obs = [MeasuredConc(value=base, t=reg.last_dose_time_h + 8.0, cv=0.1)]
    lo, hi = 1.3 * base, 1.7 * base
    rec = recommend_dose(
        ATENOLOL, reg, obs, DoseTarget((Constraint("trough", low=lo, high=hi),)),
        candidate_intervals=(8.0,), n_grid=5, n_samples=4000, seed=0,
    )
    assert lo <= rec.trough.point <= hi
    assert rec.attainment_prob > 0.5
    assert isinstance(rec, DoseRecommendation)


def test_recommend_tie_break_prefers_longer_interval():
    # A loose Cmax ceiling both q8 and q24 attain ~1.0 -> the longer interval wins.
    reg = _iv_regimen()
    from sisyphus.mipd.renal_grid import build_renal_cl_grid

    g = build_renal_cl_grid(ATENOLOL, reg, n_grid=5)
    i1 = int(np.argmin(np.abs(np.log(g.r_grid))))  # r≈1 index
    loose = 10.0 * float(g.cmax[i1])
    rec = recommend_dose(
        ATENOLOL, reg, [], DoseTarget((Constraint("cmax", high=loose),)),
        candidate_intervals=(8.0, 24.0), n_grid=5, n_samples=2000, seed=0,
    )
    assert rec.interval_h == 24.0
    assert rec.attainment_prob == pytest.approx(1.0, abs=1e-6)
    assert len(rec.candidates) == 2


def test_recommend_dose_step_rounds_dose():
    reg = _iv_regimen()
    base = _engine_trough_at_unit_scale(reg)
    obs = [MeasuredConc(value=base, t=reg.last_dose_time_h + 8.0, cv=0.1)]
    rec = recommend_dose(
        ATENOLOL, reg, obs, DoseTarget((Constraint("trough", low=1.3 * base, high=1.7 * base),)),
        candidate_intervals=(8.0,), dose_step_mg=25.0, n_grid=5, n_samples=2000, seed=0,
    )
    assert rec.dose_mg % 25.0 == pytest.approx(0.0, abs=1e-9)


def test_recommend_extreme_crcl_warns_and_individualizes():
    reg = _iv_regimen()
    rec = recommend_dose(
        ATENOLOL, reg, [], DoseTarget((Constraint("trough", low=0.01),)),
        covariates=Covariates(crcl_ml_min=3), candidate_intervals=(8.0,),
        n_grid=5, n_samples=2000, seed=0,
    )
    assert any("crcl" in w.lower() for w in rec.warnings)


def test_recommend_infeasible_target_warns():
    # A trough window far above anything any dose+posterior sample can place the MEDIAN
    # into consistently -> attainment < 0.5 soft warning. Use contradictory two-sided
    # bounds the spread can't satisfy: a razor-thin window with a very wide prior (no obs).
    reg = _iv_regimen()
    base = _engine_trough_at_unit_scale(reg)
    rec = recommend_dose(
        ATENOLOL, reg, [],  # no obs -> wide renal prior -> wide trough spread
        DoseTarget((Constraint("trough", low=0.999 * base, high=1.001 * base),)),
        candidate_intervals=(8.0,), n_grid=5, n_samples=3000, seed=0,
    )
    assert rec.attainment_prob < 0.5
    assert any("attainment" in w.lower() for w in rec.warnings)


def test_recommend_renal_scale_shifts_with_observation():
    # Observing a trough well BELOW the r=1 prediction => patient clears faster => r>1.
    reg = _iv_regimen()
    base = _engine_trough_at_unit_scale(reg)
    obs = [MeasuredConc(value=0.5 * base, t=reg.last_dose_time_h + 8.0, cv=0.2)]
    rec = recommend_dose(
        ATENOLOL, reg, obs, DoseTarget((Constraint("trough", low=0.01),)),
        candidate_intervals=(8.0,), n_grid=7, n_samples=4000, seed=0,
    )
    assert rec.renal_scale.point > 1.0


def test_recommend_joint_peak_trough_satisfies_both():
    # The motivating case for adding the interval knob: a Cmax window AND a trough ceiling
    # at once. Bounds are derived from the engine's OWN q24 r=1 prediction so a q24 regimen
    # at ~1x dose is feasible BY CONSTRUCTION (and stack-independent). Assert the
    # recommendation's median Cmax/trough satisfy BOTH constraints.
    from sisyphus.mipd.renal_grid import build_renal_cl_grid

    reg = _iv_regimen()
    reg24 = DosingRegimen.iv_infusion(50.0, 0.5, 24.0, 2)
    g24 = build_renal_cl_grid(ATENOLOL, reg24, n_grid=5)
    i24 = int(np.argmin(np.abs(np.log(g24.r_grid))))  # r≈1
    c24 = float(g24.cmax[i24])
    t24 = float(g24.conc_at(np.array([1.0]), reg24.last_dose_time_h + 24.0)[0])
    target = DoseTarget((
        Constraint("cmax", low=0.5 * c24, high=2.0 * c24),
        Constraint("trough", high=2.0 * t24),
    ))
    # condition near the q8 r=1 trough so the posterior concentrates
    obs = [MeasuredConc(
        value=_engine_trough_at_unit_scale(reg), t=reg.last_dose_time_h + 8.0, cv=0.1
    )]
    rec = recommend_dose(
        ATENOLOL, reg, obs, target, candidate_intervals=(8.0, 24.0),
        n_grid=5, n_samples=4000, seed=0,
    )
    assert rec.attainment_prob > 0.5
    assert 0.5 * c24 <= rec.cmax.point <= 2.0 * c24
    assert rec.trough.point <= 2.0 * t24
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_mipd_dosing.py -k recommend -q`
Expected: FAIL — `recommend_dose` raises `NotImplementedError`/returns nothing (not yet implemented).

- [ ] **Step 3: Implement `recommend_dose`**

```python
# append to src/sisyphus/mipd/dosing.py

_INFEASIBLE_ATTAINMENT = 0.5  # soft-warn when the best candidate falls below this
_TIE_EPS = 1e-3               # attainment ties within this prefer the longer interval


def recommend_dose(
    smiles: str,
    regimen,
    observations,
    target: DoseTarget,
    *,
    covariates=None,
    candidate_intervals: tuple[float, ...] | None = None,
    dose_step_mg: float | None = None,
    dose_bounds_mg: tuple[float, float] | None = None,
    renal_prior_cv: float = 1.0,
    n_samples: int = 20000,
    n_grid: int = 13,
    seed: int = 0,
    kp_method: str = "rodgers_rowland",
) -> DoseRecommendation:
    """Recommend the (dose, interval) that best attains ``target`` under the posterior.

    ``regimen`` is the CURRENT IV regimen the ``observations`` (steady-state troughs)
    were measured under. The renal-CL posterior is inferred once (a patient property),
    then propagated to each candidate interval; the dose is solved analytically per
    interval (LTI). The winner maximizes attainment, breaking ties toward the longer
    interval. IV-only (oral steady-state TDM is a future layer).
    """
    from sisyphus.mipd.tdm import predict_tdm
    from sisyphus.regimen.types import DEFAULT_IV_NODE, DosingRegimen

    if any(ev.node != DEFAULT_IV_NODE for ev in regimen.events):
        raise ValueError(
            "recommend_dose supports IV regimens only (every event must target the IV "
            f"node {DEFAULT_IV_NODE!r}); oral steady-state TDM is a future extension."
        )

    observations = list(observations)
    post = predict_tdm(
        smiles, regimen, observations, covariates=covariates,
        renal_prior_cv=renal_prior_cv, n_samples=n_samples, n_grid=n_grid,
        seed=seed, kp_method=kp_method,
    )
    r_samples = post.renal_scale.samples
    warnings_list = list(post.warnings)

    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    body_weight_kg = covariates.body_weight_kg if covariates is not None else None
    age_years = covariates.age_years if covariates is not None else None

    cur_dose = float(regimen.events[0].dose_mg)
    cur_dur = float(regimen.events[0].duration_h)
    cur_last = float(regimen.last_dose_time_h)
    cur_tau = (
        float(regimen.events[1].time_h - regimen.events[0].time_h)
        if regimen.n_doses >= 2 else 24.0
    )

    base = tuple(candidate_intervals) if candidate_intervals is not None else (8.0, 12.0, 24.0)
    taus = sorted(set(base + (cur_tau,)))

    rows: list[tuple[CandidateEval, dict[str, np.ndarray], float]] = []
    for tau in taus:
        n_doses = max(2, int(round(cur_last / tau)) + 1)
        reg_tau = DosingRegimen.iv_infusion(
            dose_mg=cur_dose, duration_h=cur_dur, interval_h=tau, n_doses=n_doses
        )
        q_ref, d_ref = _interval_reference(
            smiles, reg_tau, tau, r_samples, renal_factor=renal_factor,
            body_weight_kg=body_weight_kg, age_years=age_years, n_grid=n_grid,
            kp_method=kp_method,
        )
        m_lo, m_hi = _sample_m_intervals(q_ref, target)
        a, b, _ = _max_overlap_region(m_lo, m_hi)
        dose = _center_m(a, b) * d_ref
        if dose_step_mg:
            dose = round(dose / dose_step_mg) * dose_step_mg
        if dose_bounds_mg is not None:
            dose = min(max(dose, dose_bounds_mg[0]), dose_bounds_mg[1])
        m_actual = dose / d_ref if d_ref > 0 else 0.0
        attain = _attainment(m_actual, m_lo, m_hi)
        rows.append((
            CandidateEval(
                dose_mg=float(dose), interval_h=float(tau), attainment_prob=attain,
                trough_median=float(np.median(q_ref["trough"] * m_actual)),
                cmax_median=float(np.median(q_ref["cmax"] * m_actual)),
                auc24_median=float(np.median(q_ref["auc24"] * m_actual)),
            ),
            q_ref, m_actual,
        ))

    best_attain = max(row[0].attainment_prob for row in rows)
    winners = [row for row in rows if row[0].attainment_prob >= best_attain - _TIE_EPS]
    win_cand, win_q, win_m = max(winners, key=lambda row: row[0].interval_h)

    if win_cand.attainment_prob < _INFEASIBLE_ATTAINMENT:
        warnings_list.append(
            f"best attainment {win_cand.attainment_prob:.2f} < "
            f"{_INFEASIBLE_ATTAINMENT:.2f}; target may be infeasible for this patient"
        )

    return DoseRecommendation(
        dose_mg=win_cand.dose_mg,
        interval_h=win_cand.interval_h,
        attainment_prob=win_cand.attainment_prob,
        cmax=Posterior(win_q["cmax"] * win_m),
        trough=Posterior(win_q["trough"] * win_m),
        auc24=Posterior(win_q["auc24"] * win_m),
        target=target,
        candidates=tuple(row[0] for row in rows),
        renal_scale=post.renal_scale,
        n_eff=post.n_eff,
        warnings=tuple(warnings_list),
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/unit/test_mipd_dosing.py -k recommend -q`
Expected: PASS (8 tests). If `test_recommend_infeasible_target_warns` is flaky on the spread, widen the razor window's contradiction by using a window entirely above the prior's trough support (e.g. `low=3*base, high=3.01*base`) — keep the assertion that attainment < 0.5 and the warning fires.

- [ ] **Step 5: Run the whole file + ruff**

Run: `pytest tests/unit/test_mipd_dosing.py -q && ruff check src/sisyphus/mipd/dosing.py tests/unit/test_mipd_dosing.py`
Expected: all green; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/mipd/dosing.py tests/unit/test_mipd_dosing.py
git commit -m "feat(mipd): recommend_dose orchestration (per-interval LTI dose solve)"
```

---

### Task 5: Public exports

**Files:**
- Modify: `src/sisyphus/mipd/__init__.py`
- Test: `tests/unit/test_mipd_dosing.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_mipd_dosing.py
def test_public_names_importable_from_package():
    import sisyphus.mipd as m

    for name in ("Constraint", "DoseTarget", "CandidateEval", "DoseRecommendation",
                 "recommend_dose"):
        assert hasattr(m, name)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_mipd_dosing.py::test_public_names_importable_from_package -q`
Expected: FAIL (names not exported from the package).

- [ ] **Step 3: Add the exports**

In `src/sisyphus/mipd/__init__.py`, add an import block after the existing `from sisyphus.mipd.grid import build_cl_grid` line:

```python
from sisyphus.mipd.dosing import (
    CandidateEval,
    Constraint,
    DoseRecommendation,
    DoseTarget,
    recommend_dose,
)
```

And add these names to `__all__` (keep the list sorted-ish, matching the existing style):

```python
    "CandidateEval",
    "Constraint",
    "DoseRecommendation",
    "DoseTarget",
    "recommend_dose",
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/unit/test_mipd_dosing.py::test_public_names_importable_from_package -q`
Expected: PASS.

- [ ] **Step 5: Full module test + ruff**

Run: `pytest tests/unit/test_mipd_dosing.py -q && ruff check src/sisyphus/mipd/`
Expected: all green; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/mipd/__init__.py tests/unit/test_mipd_dosing.py
git commit -m "feat(mipd): export dose-recommendation public API"
```

---

## Final Verification (after all tasks)

- [ ] **Run the full MIPD test suite + the headline-invariance guard:**

```bash
pytest tests/unit/test_mipd_dosing.py tests/unit/test_mipd_tdm.py \
       tests/unit/test_mipd_api.py tests/unit/test_mipd_grid.py \
       tests/unit/test_mipd_renal_grid.py -q
```
Expected: all pass. (`recommend_dose` adds no production-path change, so the cached-holdout pin `test_cached_holdout_aafe_is_2p731` is unaffected — but if a broader run is cheap, confirm it still passes.)

- [ ] **Confirm invariants:** `git diff --stat main` shows changes ONLY under `src/sisyphus/mipd/dosing.py`, `src/sisyphus/mipd/__init__.py`, `tests/unit/test_mipd_dosing.py`, and the two `docs/superpowers/` files. No `engine/`, no `predict/`, no `pipeline/`, no holdout/data changes.

- [ ] **Update graphify graph:** `graphify update .` (AST-only, no API cost).

---

## Notes for the implementer

- **Stack-independence is mandatory in directional tests.** Never assert against an absolute concentration magic number; anchor every window to the engine's *own* r=1 prediction (the `_engine_trough_at_unit_scale` helper). The macOS/CI numerics stacks differ ~12% per-drug; a hard-coded concentration will pass locally and fail CI (this exact failure mode bit the IV-TDM directional test, fixed in `df4492c`).
- **Keep `n_grid` / `n_samples` small in tests** (5–7 / 2000–4000) so each engine re-solve stays fast; production defaults are 13 / 20000.
- **Do NOT import `_regimen_interval_h`** from `renal_grid` into `dosing.py` — the orchestrator already knows τ (it constructs each candidate regimen). The current interval is read inline from the event times.
- **`predict_tdm` with empty `observations`** returns the prior as the posterior (renal scale ~ wide lognormal around 1.0) — this is the a-priori recommendation path and is intended.
