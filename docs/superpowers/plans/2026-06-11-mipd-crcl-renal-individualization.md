# MIPD CrCl Renal Individualization + Conditioned-Output Surfacing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Individualize the MIPD engine-as-prior posterior's renal elimination from a measured creatinine clearance (CrCl), and surface the un-damped individualized engine posterior (`post.cmax`) as the primary TDM estimate — without changing the existing `meta_cmax`/`cmax_90ci` contract.

**Architecture:** CrCl scales the drug-level `renal_clearance` Distribution by `CrCl/125` (reference GFR) once, in the predict/grid layer; the engine stays identity-blind. A new dispatch branch routes a CrCl-only prediction through a single renal-scaled engine solve (clint latent stays fixed) so the metabolic latent is freed only by a `MeasuredConc`. The output keeps the validated `meta_cmax`/conformal band and adds a `warnings` field; `post.cmax` is documented as the individualized primary.

**Tech Stack:** Python 3.10+, numpy, pytest, ruff. Module: `src/sisyphus/mipd/`. Frozen dataclasses for contracts. Reference: `docs/superpowers/specs/2026-06-11-mipd-crcl-renal-individualization-design.md`.

---

## File Structure

- **NEW** `src/sisyphus/mipd/covariates.py` — `Covariates` frozen dataclass + `_REFERENCE_GFR_ML_MIN`. One responsibility: patient-covariate → deterministic engine-prior scaling.
- **MODIFY** `src/sisyphus/mipd/core.py` — add `PosteriorPK.warnings` field (additive); add module-level `ci_floor` pure helper; docstring clarifying the three outputs.
- **MODIFY** `src/sisyphus/mipd/grid.py` — add `renal_factor: float = 1.0` param to `build_cl_grid`, applied once to the base drug.
- **MODIFY** `src/sisyphus/mipd/api.py` — `predict_posterior` gains `covariates`; 3-way dispatch (grid / single renal-scaled solve / reference F-only); extreme-CrCl warning; `post.cmax`-primary docstring.
- **NEW** `tests/unit/test_mipd_covariates.py` — `Covariates`, `ci_floor`.
- **MODIFY** `tests/unit/test_mipd_api.py` — CrCl routing, bit-identity, directional, warnings.

**Deviation from spec §4.4 (flagged):** the `min_ci_half_width_fraction` floor is shipped as the **pure helper `ci_floor` only**, NOT wired into `predict_posterior`. Reason: `post.cmax.ci90` is a computed property of an immutable `Posterior`, so wiring a floor would require a shadow field; the floor is opt-in/default-off, so a caller applies `mipd.ci_floor(post.cmax.ci90, post.cmax.mean, frac)` instead. Surfaced again in the handoff for confirmation.

---

## Task 1: Add `PosteriorPK.warnings` field (additive, non-breaking)

**Files:**
- Modify: `src/sisyphus/mipd/core.py` (the `PosteriorPK` dataclass, ends at line 195 with `cl_scale`)
- Test: `tests/unit/test_mipd_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_mipd_core.py`:

```python
def test_posterior_pk_has_empty_warnings_by_default():
    from sisyphus.mipd.core import Posterior, PosteriorPK
    import numpy as np

    s = Posterior(np.array([1.0, 2.0, 3.0]))
    post = PosteriorPK(f=s, cmax=s, auc=s, n_eff=10.0)
    assert post.warnings == ()


def test_posterior_pk_carries_warnings():
    from sisyphus.mipd.core import Posterior, PosteriorPK
    import numpy as np

    s = Posterior(np.array([1.0, 2.0, 3.0]))
    post = PosteriorPK(f=s, cmax=s, auc=s, n_eff=10.0, warnings=("crcl:extreme:3.0",))
    assert post.warnings == ("crcl:extreme:3.0",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_mipd_core.py::test_posterior_pk_has_empty_warnings_by_default -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'warnings'` is NOT raised yet (the default-field test) — actually the first test fails with `AttributeError: 'PosteriorPK' object has no attribute 'warnings'`.

- [ ] **Step 3: Add the field**

In `src/sisyphus/mipd/core.py`, in the `PosteriorPK` dataclass, immediately after the `cl_scale: Posterior | None = None` line (line 195), add:

```python
    # Structured non-fatal flags (e.g. extreme CrCl). Empty by default
    # (additive — preserves the prior PosteriorPK contract). Project doctrine:
    # never silently drop a warning.
    warnings: tuple[str, ...] = ()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mipd_core.py -v`
Expected: PASS (all, including the two new tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/core.py tests/unit/test_mipd_core.py
git commit -m "feat(mipd): add PosteriorPK.warnings structured-flag field"
```

---

## Task 2: Add `ci_floor` pure helper

**Files:**
- Modify: `src/sisyphus/mipd/core.py` (add module-level function after the `Posterior` class, ~line 164)
- Test: `tests/unit/test_mipd_covariates.py` (NEW)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_mipd_covariates.py`:

```python
"""Tests for mipd.covariates (Covariates) and mipd.core.ci_floor."""
from sisyphus.mipd.core import ci_floor


def test_ci_floor_off_by_default_fraction():
    assert ci_floor((0.9, 1.1), 1.0, 0.0) == (0.9, 1.1)


def test_ci_floor_widens_overtight_interval():
    # half-width 0.05 < 0.2*1.0 -> widen to +/- 0.2 around the mean
    assert ci_floor((0.95, 1.05), 1.0, 0.2) == (0.8, 1.2)


def test_ci_floor_leaves_wide_interval_unchanged():
    assert ci_floor((0.5, 1.5), 1.0, 0.2) == (0.5, 1.5)


def test_ci_floor_none_passthrough():
    assert ci_floor(None, 1.0, 0.2) is None


def test_ci_floor_nonpositive_mean_passthrough():
    assert ci_floor((0.9, 1.1), 0.0, 0.2) == (0.9, 1.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_mipd_covariates.py -v`
Expected: FAIL — `ImportError: cannot import name 'ci_floor' from 'sisyphus.mipd.core'`.

- [ ] **Step 3: Implement the helper**

In `src/sisyphus/mipd/core.py`, after the `Posterior` class block (after the `ci90` property, ~line 164, before `@dataclass class PosteriorPK`), add:

```python
def ci_floor(
    ci: tuple[float, float] | None, mean: float, frac: float
) -> tuple[float, float] | None:
    """Widen a 90% interval to half-width ``frac*mean`` if it is narrower.

    Opt-in guard (``frac<=0`` is a no-op) against an over-tight conditioned
    posterior that pathologically excludes truth. Mirrors the formula of
    ``regimen.tdm._apply_ci_floor`` (which takes a TDMResult, so it is not a
    drop-in here). The widened interval is centered on ``mean``.
    """
    if frac <= 0.0 or ci is None or mean <= 0.0:
        return ci
    lo, hi = ci
    floor = frac * mean
    half = max(mean - lo, hi - mean)
    if half >= floor:
        return ci
    return (max(mean - floor, 0.0), mean + floor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mipd_covariates.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/core.py tests/unit/test_mipd_covariates.py
git commit -m "feat(mipd): add ci_floor pure helper (opt-in CI-widening guard)"
```

---

## Task 3: Add `Covariates` dataclass

**Files:**
- Create: `src/sisyphus/mipd/covariates.py`
- Test: `tests/unit/test_mipd_covariates.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_mipd_covariates.py`:

```python
def test_covariates_renal_factor_unity_at_reference():
    from sisyphus.mipd.covariates import Covariates
    assert Covariates(crcl_ml_min=125.0).renal_factor() == 1.0


def test_covariates_renal_factor_scales_linearly():
    from sisyphus.mipd.covariates import Covariates
    assert Covariates(crcl_ml_min=62.5).renal_factor() == 0.5


def test_covariates_empty_is_no_op():
    from sisyphus.mipd.covariates import Covariates
    assert Covariates().renal_factor() == 1.0
    assert Covariates(crcl_ml_min=None).renal_factor() == 1.0


def test_covariates_rejects_nonpositive_crcl():
    import pytest
    from sisyphus.mipd.covariates import Covariates
    with pytest.raises(ValueError):
        Covariates(crcl_ml_min=0.0)
    with pytest.raises(ValueError):
        Covariates(crcl_ml_min=-5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_mipd_covariates.py -k covariates_renal -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sisyphus.mipd.covariates'`.

- [ ] **Step 3: Create the module**

Create `src/sisyphus/mipd/covariates.py`:

```python
"""Patient covariates that deterministically individualize the engine prior.

v1: renal function only — a measured creatinine clearance (CrCl). CrCl scales
the drug's renal (glomerular-filtration) clearance: the engine's reference renal
model is ``CL_renal = GFR*fup`` with GFR = 7.5 L/h (~125 mL/min), so an
individual's renal CL is scaled by ``CrCl / 125``. Weight/age covariates (via
sbi.physiology_generator) are a documented future extension — see the design spec
docs/superpowers/specs/2026-06-11-mipd-crcl-renal-individualization-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

# Reference glomerular filtration rate the engine's renal_clearance assumes
# (_GFR_L_PER_H = 7.5 L/h ~= 125 mL/min; src/sisyphus/predict/ivive.py:42-43).
_REFERENCE_GFR_ML_MIN = 125.0


@dataclass(frozen=True)
class Covariates:
    """Deterministic patient covariates for the engine-as-prior individualization.

    Attributes:
        crcl_ml_min: measured creatinine clearance (mL/min). None -> no renal
            individualization (renal_factor 1.0).
    """

    crcl_ml_min: float | None = None

    def __post_init__(self) -> None:
        if self.crcl_ml_min is not None and self.crcl_ml_min <= 0:
            raise ValueError(f"crcl_ml_min must be > 0, got {self.crcl_ml_min}")

    def renal_factor(self) -> float:
        """Multiplicative scale for ``drug.renal_clearance`` (1.0 at CrCl=125)."""
        if self.crcl_ml_min is None:
            return 1.0
        return self.crcl_ml_min / _REFERENCE_GFR_ML_MIN
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mipd_covariates.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/covariates.py tests/unit/test_mipd_covariates.py
git commit -m "feat(mipd): add Covariates(crcl_ml_min) with renal_factor"
```

---

## Task 4: Add `renal_factor` to `build_cl_grid`

**Files:**
- Modify: `src/sisyphus/mipd/grid.py` (signature at line 58; apply after `drug = build_drug_on_graph(...)` at line 102-109, before `augment_for_active_species` at line 110)
- Test: `tests/unit/test_mipd_grid.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_mipd_grid.py` (the file already imports `build_cl_grid`, `MIDAZOLAM`, `DOSE`; reuse them — if not in scope, import `from sisyphus.mipd.grid import build_cl_grid` and define `MIDAZOLAM = "C[C@@H]1N=C(c2ccccc2F)c2cc(Cl)ccc2-n2c1nc2C"`, `DOSE = 7.5` at the top alongside the existing constants):

```python
def test_build_cl_grid_renal_factor_scales_exposure_and_preserves_f_engine():
    import numpy as np
    base = build_cl_grid(MIDAZOLAM, DOSE, n_grid=5, s_range=(0.5, 2.0))
    low_renal = build_cl_grid(
        MIDAZOLAM, DOSE, n_grid=5, s_range=(0.5, 2.0), renal_factor=0.25
    )
    # Lower renal clearance -> higher total AUC at every clint scale (renal_cl>0).
    assert np.all(low_renal.auc >= base.auc)
    assert low_renal.auc[0] > base.auc[0]
    # F_engine = AUC_oral/AUC_iv is invariant to systemic renal scaling (LTI).
    assert np.allclose(low_renal.f_engine, base.f_engine, rtol=1e-3)


def test_build_cl_grid_renal_factor_unity_is_unchanged():
    import numpy as np
    a = build_cl_grid(MIDAZOLAM, DOSE, n_grid=5, s_range=(0.5, 2.0))
    b = build_cl_grid(MIDAZOLAM, DOSE, n_grid=5, s_range=(0.5, 2.0), renal_factor=1.0)
    assert np.array_equal(a.cmax, b.cmax)
    assert np.array_equal(a.auc, b.auc)


def test_build_cl_grid_single_point_at_unit_scale():
    # n_grid=1 at s=1 is the single renal-scaled solve the API uses for CrCl-only.
    g = build_cl_grid(MIDAZOLAM, DOSE, n_grid=1, s_range=(1.0, 1.0), renal_factor=0.5)
    assert g.cmax.shape == (1,)
    assert g.cmax[0] > 0 and g.auc[0] > 0
    assert 0.0 < g.f_engine[0] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_mipd_grid.py -k renal_factor -v`
Expected: FAIL — `TypeError: build_cl_grid() got an unexpected keyword argument 'renal_factor'`.

- [ ] **Step 3: Add the parameter and scaling**

In `src/sisyphus/mipd/grid.py`, change the `build_cl_grid` signature (line 58) to add `renal_factor`:

```python
def build_cl_grid(
    smiles: str,
    dose_mg: float,
    route: str = "oral",
    *,
    n_grid: int = 13,
    s_range: tuple[float, float] = (0.1, 10.0),
    kp_method: str = "rodgers_rowland",
    t_grid: np.ndarray | None = None,
    renal_factor: float = 1.0,
) -> CLGrid:
```

Then, immediately after the `drug = build_drug_on_graph(...)` call (it ends at line 109 with a closing `)`), and BEFORE `graph = augment_for_active_species(graph, drug)` (line 110), insert:

```python
    # CrCl renal individualization (covariate-fixed, applied once to the base
    # drug): scale the drug-level renal (glomerular-filtration) clearance. The
    # per-scale enzyme_affinity scaling below is orthogonal and unchanged.
    # F_engine is invariant to this (the engine is linear time-invariant), so the
    # f_engine column is unaffected (see the design spec, verification C2).
    if renal_factor != 1.0:
        drug = dataclasses.replace(
            drug,
            renal_clearance=Distribution(
                mean=drug.renal_clearance.mean * renal_factor,
                cv=drug.renal_clearance.cv,
            ),
        )
```

(`dataclasses` is already imported at grid.py:15; `Distribution` at grid.py:19.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mipd_grid.py -v`
Expected: PASS (all, including the 3 new tests and the pre-existing grid tests).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/mipd/grid.py tests/unit/test_mipd_grid.py
git commit -m "feat(mipd): build_cl_grid renal_factor scales drug renal_clearance"
```

---

## Task 5: Wire `covariates` into `predict_posterior` (dispatch + warnings + docstring)

**Files:**
- Modify: `src/sisyphus/mipd/api.py` (`predict_posterior`, lines 55-148; imports near line 17-26)
- Test: `tests/unit/test_mipd_api.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_mipd_api.py` (it already imports `predict_posterior`, `MIDAZOLAM`, `DOSE`, `predict`, `pytest`, `numpy as np` — reuse; if a symbol is missing, add the import):

```python
from sisyphus.mipd.covariates import Covariates

ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"  # high-fup, renally-cleared


def test_covariate_none_is_bit_identical_to_no_covariate():
    a = predict_posterior(MIDAZOLAM, DOSE, seed=0)
    b = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(), seed=0)
    assert np.array_equal(a.cmax.samples, b.cmax.samples)


def test_crcl_only_uses_single_solve_not_2latent_grid():
    post = predict_posterior(
        MIDAZOLAM, DOSE, covariates=Covariates(crcl_ml_min=25), seed=0
    )
    assert post.cl_scale is None  # clint latent NOT freed (no MeasuredConc)
    assert post.cmax.point > 0


def test_renal_impairment_raises_exposure_for_renal_drug():
    healthy = predict_posterior(ATENOLOL, DOSE, covariates=Covariates(crcl_ml_min=120), seed=0)
    impaired = predict_posterior(ATENOLOL, DOSE, covariates=Covariates(crcl_ml_min=20), seed=0)
    assert impaired.auc.point > healthy.auc.point


def test_extreme_crcl_emits_warning_normal_does_not():
    low = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(crcl_ml_min=3), seed=0)
    assert any("crcl" in w.lower() for w in low.warnings)
    normal = predict_posterior(MIDAZOLAM, DOSE, covariates=Covariates(crcl_ml_min=90), seed=0)
    assert normal.warnings == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_mipd_api.py -k "covariate or crcl or renal_impairment" -v`
Expected: FAIL — `TypeError: predict_posterior() got an unexpected keyword argument 'covariates'`.

- [ ] **Step 3: Edit the signature and add the import**

In `src/sisyphus/mipd/api.py`, add to the imports block (after line 25, `from sisyphus.mipd.core import APrioriPK, Posterior, PosteriorPK`):

```python
from sisyphus.mipd.covariates import Covariates
```

Change the `predict_posterior` signature (line 55) to add `covariates` as a keyword-only arg — insert it right after `cl_latent: bool = False,` / before `n_grid: int = 13,`:

```python
    cl_latent: bool = False,
    covariates: Covariates | None = None,
    n_grid: int = 13,
```

Also add `import dataclasses` at the top of `api.py` if not present (it IS present — `import dataclasses` is already imported in this module).

- [ ] **Step 4: Replace the dispatch body**

Replace the block from line 97 (`observations = list(observations)`) through line 148 (`return _attach_meta_and_interval(post, smiles, dose_mg, ap)`) with:

```python
    observations = list(observations)
    needs_grid = cl_latent or any(isinstance(o, MeasuredConc) for o in observations)
    renal_factor = covariates.renal_factor() if covariates is not None else 1.0
    individualized = renal_factor != 1.0
    rng = np.random.default_rng(seed)

    warnings_list: list[str] = []
    if covariates is not None and covariates.crcl_ml_min is not None:
        if not (5.0 <= covariates.crcl_ml_min <= 200.0):
            warnings_list.append(
                f"crcl:extreme:{covariates.crcl_ml_min}: the engine renal model is "
                "glomerular-filtration-only and least reliable outside [5, 200] mL/min"
            )

    # ap is needed for the (covariate-blind) meta tracks regardless. engine_f is
    # only consumed by the reference F-only branch, so the IV-reference solve is
    # requested only there.
    ap = predict(
        smiles, dose_mg, route=route,
        compute_f_engine=(not needs_grid and not individualized),
        **predict_kwargs,
    )
    if ap.engine_pk is None or ap.engine_pk.cmax is None:
        raise ValueError("engine produced no Cmax for this input; cannot build a posterior")

    if needs_grid:
        # CL-grid 2-latent (F, clint-scale) path — handles MeasuredConc. CrCl
        # individualizes renal CL via renal_factor; the clint latent is freed.
        from sisyphus.mipd.clgrid import CLGridForward, CLPrior, sir_posterior_2d
        from sisyphus.mipd.core import FPrior
        from sisyphus.mipd.grid import build_cl_grid

        grid = build_cl_grid(
            smiles, dose_mg, route=route, n_grid=n_grid,
            kp_method=predict_kwargs.get("kp_method", "rodgers_rowland"),
            renal_factor=renal_factor,
        )
        i1 = int(np.argmin(np.abs(np.log(grid.s_grid))))
        f_engine0 = float(min(max(grid.f_engine[i1], 1e-4), 1.0))
        f_prior = FPrior(f_engine0, prior_cv)
        cl_prior = CLPrior(
            cv=cl_prior_cv, s_min=float(grid.s_grid[0]), s_max=float(grid.s_grid[-1])
        )
        post = sir_posterior_2d(
            f_prior, cl_prior, CLGridForward(grid), observations,
            n_samples=n_samples, rng=rng,
        )
    elif individualized:
        # CrCl-only (no curve-shape obs): a single renal-scaled engine solve at
        # clint-scale s=1 (clint latent stays FIXED). Reuses build_cl_grid with a
        # 1-point grid; conc_at fragility does not bite (no MeasuredConc here).
        from sisyphus.mipd.grid import build_cl_grid

        g1 = build_cl_grid(
            smiles, dose_mg, route=route, n_grid=1, s_range=(1.0, 1.0),
            kp_method=predict_kwargs.get("kp_method", "rodgers_rowland"),
            renal_factor=renal_factor,
        )
        cmax0 = float(g1.cmax[0])
        auc0 = float(g1.auc[0])
        f_engine = float(min(max(g1.f_engine[0], 1e-4), 1.0))
        apriori = APrioriPK(cmax0=cmax0, auc0=auc0, f_engine=f_engine)
        post = SIRAmortizer(prior_cv=prior_cv, n_samples=n_samples).posterior(
            apriori, observations, rng=rng
        )
    else:
        # Reference F-only analytic path (unchanged): cmax0 off the a-priori call.
        cmax0 = ap.engine_pk.cmax.mean
        auc0 = ap.engine_pk.auc_0t.mean if ap.engine_pk.auc_0t is not None else 0.0
        if ap.engine_f is None:
            raise ValueError(
                "engine F-reference solve unavailable; cannot build the F posterior "
                "(use cl_latent=True for the CL-grid path, which derives F_engine "
                "from the engine grid)"
            )
        f_engine = min(max(ap.engine_f, 1e-4), 1.0)
        apriori = APrioriPK(cmax0=cmax0, auc0=auc0, f_engine=f_engine)
        post = SIRAmortizer(prior_cv=prior_cv, n_samples=n_samples).posterior(
            apriori, observations, rng=rng
        )

    post = _attach_meta_and_interval(post, smiles, dose_mg, ap)
    return dataclasses.replace(post, warnings=tuple(warnings_list))
```

- [ ] **Step 5: Update the docstring to surface `post.cmax` as the TDM primary**

In `predict_posterior`'s docstring (the `Args:`/`Returns` body), add a paragraph near the top of the docstring body:

```python
    """Posterior PK for ``smiles`` at ``dose_mg`` given ``observations``.

    Output fields (all on the returned PosteriorPK):
      - ``cmax`` (+ ``cmax.ci90``): the engine-track posterior, fully conditioned
        and CrCl-individualized — the PRIMARY estimate for TDM / patient
        individualization. Its ci90 is a parameter-uncertainty band (under-covers
        structural error).
      - ``meta_cmax``: the production population blend (covariate-blind ML/CLF/VDss
        mixed in; damped under conditioning, DE-43) — the SMILES-anchor product.
      - ``cmax_90ci``: the train-calibrated conformal predictive band around the
        meta point — the only coverage-validated interval (conservative under
        conditioning, review #6).
      - ``warnings``: structured non-fatal flags (e.g. extreme CrCl).

    Args:
        covariates: patient covariates (v1: measured CrCl) that deterministically
            individualize the engine's renal clearance (CrCl/125). Routes through a
            single renal-scaled engine solve when no MeasuredConc is present, so the
            metabolic clint latent stays fixed.
    ... (keep the existing Args entries below) ...
    """
```

- [ ] **Step 6: Run the new + existing api tests**

Run: `python -m pytest tests/unit/test_mipd_api.py -v`
Expected: PASS (new covariate tests + all pre-existing api tests).

- [ ] **Step 7: Commit**

```bash
git add src/sisyphus/mipd/api.py tests/unit/test_mipd_api.py
git commit -m "feat(mipd): predict_posterior covariates(CrCl) dispatch + warnings + post.cmax docs"
```

---

## Task 6: Clinical-sanity test — renal individualization matters MORE for renal drugs

**Files:**
- Test: `tests/unit/test_mipd_api.py` (append)

- [ ] **Step 1: Write the test**

```python
def test_renal_individualization_larger_for_high_renal_fraction_drug():
    """The CrCl effect on AUC is larger for a renally-cleared drug (atenolol)
    than for a hepatically-cleared one (midazolam), since renal_cl = 7.5*fup is a
    larger fraction of total CL when fup is high and metabolism is low."""
    def auc_ratio(smiles):
        hi = predict_posterior(smiles, DOSE, covariates=Covariates(crcl_ml_min=120), seed=0)
        lo = predict_posterior(smiles, DOSE, covariates=Covariates(crcl_ml_min=20), seed=0)
        return lo.auc.point / hi.auc.point

    assert auc_ratio(ATENOLOL) > auc_ratio(MIDAZOLAM)
    assert auc_ratio(ATENOLOL) > 1.0  # impairment raises exposure for the renal drug
```

- [ ] **Step 2: Run to verify it passes (implementation already exists from Task 5)**

Run: `python -m pytest tests/unit/test_mipd_api.py::test_renal_individualization_larger_for_high_renal_fraction_drug -v`
Expected: PASS. If `auc_ratio(ATENOLOL) <= auc_ratio(MIDAZOLAM)` (engine predicts an unexpectedly low atenolol fup), STOP and investigate the predicted renal fraction (`predict(ATENOLOL, DOSE).engine_pk` + `predict_adme`); do not weaken the assertion without understanding why.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_mipd_api.py
git commit -m "test(mipd): CrCl effect is larger for renally-cleared drugs"
```

---

## Task 7: Full regression + lint + graph update

**Files:** none (verification only)

- [ ] **Step 1: Run the full mipd + regression suites**

Run: `python -m pytest tests/unit/test_mipd_api.py tests/unit/test_mipd_core.py tests/unit/test_mipd_grid.py tests/unit/test_mipd_meta.py tests/unit/test_mipd_clgrid.py tests/unit/test_mipd_covariates.py tests/integration/test_holdout_regression.py -v`
Expected: PASS — all mipd tests (including the four conditioned-output tests in the spec §2.1, unchanged) and the holdout cache pin `test_cached_holdout_aafe_is_2p784` (predict() and the headline are untouched).

- [ ] **Step 2: Lint**

Run: `ruff check src/sisyphus/mipd/ tests/unit/test_mipd_covariates.py`
Expected: no errors (line length 100). Fix any reported issues and re-run.

- [ ] **Step 3: Update the knowledge graph**

Run: `graphify update .`
Expected: completes (AST-only, no API cost).

- [ ] **Step 4: Commit any lint/graph changes**

```bash
git add -A src/sisyphus/mipd/ tests/ graphify-out/
git commit -m "chore(mipd): ruff + graphify after CrCl individualization" || echo "nothing to commit"
```

---

## Self-Review (planner checklist — completed)

**1. Spec coverage:**
- §4.1 CrCl→renal physics → Task 4 (build_cl_grid renal_factor) + Task 5 (dispatch applies it). ✓
- §4.2 dispatch (clint latent not freed by CrCl alone) → Task 5 (`elif individualized` single-solve; `test_crcl_only_uses_single_solve_not_2latent_grid`). ✓
- §4.3 output surfacing (post.cmax primary, meta/conformal unchanged, non-breaking) → Task 5 docstring + `test_covariate_none_is_bit_identical`; existing conditioned tests unchanged → Task 7 regression. ✓
- §4.5 `Covariates` + `_REFERENCE_GFR_ML_MIN` → Task 3; `warnings` field → Task 1, populated in Task 5. ✓
- §4.4 floor → Task 2 `ci_floor` helper. **Deviation:** not wired into `predict_posterior` (helper-only) — flagged in File Structure + handoff.
- §5 invariants (bit-identity, headline untouched) → `test_covariate_none_is_bit_identical` + Task 7 holdout pin. ✓
- §7 tests (renal_factor math, F_engine invariance, directional, warnings) → Tasks 2-6. ✓

**2. Placeholder scan:** none — every step has concrete code/commands.

**3. Type consistency:** `Covariates.renal_factor()` (Task 3) used in Task 5; `renal_factor: float` param (Task 4) matches Task 5 call sites; `ci_floor(ci, mean, frac)` (Task 2) signature matches its tests; `PosteriorPK.warnings: tuple[str,...]` (Task 1) matches `dataclasses.replace(post, warnings=tuple(...))` (Task 5). ✓
