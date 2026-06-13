# Oral Steady-State TDM — Design

**Date:** 2026-06-12
**Status:** Approved (design); pending implementation plan
**Modules:** `src/sisyphus/mipd/oral_grid.py` (new), `mipd/tdm.py` + `mipd/dosing.py` (route dispatch)
**Builds on:** `mipd/clgrid.py` (2-latent F+clint forward, `sir_posterior_2d`), `mipd/renal_grid.py` (multi-dose grid template), `mipd/tdm.py`/`dosing.py` (IV TDM + dose recommendation), `regimen.solver.solve_regimen`

Grounded by a 6-agent understanding sweep (workflow `wf_7bc3305e-551`).

---

## 1. Purpose

Extend the MIPD TDM stack from **IV-only** to **oral multi-dose**: condition the engine-as-prior on a steady-state **oral** trough (and optionally a peak, or a measured F) to individualize an oral patient, and recommend a target-attainment oral dose. The IV path (`predict_tdm`, `recommend_dose`) frees a single renal-CL latent because for IV F≡1; for oral, **F re-enters as the dominant latent** (F is the engine's dominant structural error — DE-41/43→FLUX-1 — while CLint, R²~0.24, is already its best estimate).

This is a **measured-input individualization** feature, like the shipped measured-F routing — **not a 107-holdout headline lever** (DE-43: the fixed-weight meta damps engine moves). With zero measured data the path reduces to the a-priori prediction; the 2.731 headline is untouched.

## 2. The identifiability crux + the adaptive stance

At oral steady state `Css ≈ F·Dose/(CL·τ)`. A single trough is a **magnitude** observation: it constrains only the ridge `F/CL`, leaving F and clint marginals prior-wide if both are freed (`clgrid.py` identifiability note: *"with magnitude-only data, prefer the F-only path"*). The ridge is broken only by **shape** (a second conc at a *different within-interval phase* pins ke=CL/V independent of F) or by a **measured F** (anchors the F axis).

**Adaptive stance (v1):**
- **free-both (F + metabolic clint)** iff the observations contain shape/anchor info: **a `MeasuredF` is present**, **OR** the `MeasuredConc` set has **≥2 distinct within-interval phases** (`t mod τ` spread beyond a tolerance — a peak+trough). Runs `sir_posterior_2d` over the full oral grid; both latents are *identified by construction*, so the marginals are honest.
- **free-F-only** otherwise (the common single-trough case, or repeated same-phase troughs): one effective latent (F) over the grid's **s=1 slice**. Same-phase troughs still tighten the F/exposure posterior — not wasted. Returns `cl_scale=None`.

**Critical honesty point.** Even on the ridge, **predicted exposure (~F/CL) — the load-bearing output for dose recommendation — is exactly what the trough identifies**. So free-F-only and free-both give the *same* exposure/Cmax posterior for a single trough; free-F-only just omits two ridge-wide marginals. The F-only posterior is "the trough discrepancy attributed to F, holding CL at the engine estimate"; the **exposure posterior is attribution-independent** and is what `recommend_dose` consumes. State this in the docstrings.

## 3. New: `build_oral_cl_grid` (`oral_grid.py`)

A sibling of `build_renal_cl_grid` that returns the **existing `clgrid.CLGrid`** type (no new grid/forward types — `CLGridForward` + `sir_posterior_2d` work on it unchanged).

```python
def build_oral_cl_grid(
    smiles: str, regimen, *, n_grid: int = 13, s_range: tuple[float, float] = (0.05, 20.0),
    renal_factor: float = 1.0, body_weight_kg: float | None = None,
    age_years: float | None = None, kp_method: str = "rodgers_rowland",
    dt_output: float = 0.1,
) -> CLGrid
```

Per grid point `s` (metabolic clint-scale on `enzyme_affinity`, like `build_cl_grid`; renal/biliary fixed):
- **SS exposures** from `solve_regimen` over the oral `regimen`, cmax/auc/conc extracted on the **final interval** `[last, last+τ]` (the renal-grid mask, route-agnostic). `t_total = last + max(τ, 24)` so the trough at `last+τ` is in-window.
- **`f_engine[s]`** = the engine's emergent oral bioavailability at scale `s`, from a **single-dose** oral/IV reference (the existing `_engine_oral_bioavailability` machinery `build_cl_grid` uses). **F is a per-dose property, regimen-independent under linear PK** — so the single-dose F reference is the correct `f_engine` for the SS grid, and `CLGridForward`'s `scale = F/f_engine` exactly rescales SS exposure (SS exposure ∝ F).

Reuses `_build_grid_engine(smiles, dose, "oral", renal_factor, kp_method, body_weight_kg, age_years)` (the route param, already a string), `solve_regimen`, and the NaN-backfill helpers — verbatim except route and the metabolic-`s` scaling.

## 4. Oral `predict_tdm` (route dispatch in `tdm.py`)

`predict_tdm` reads the regimen route (`_regimen_route(regimen)`: all events at `DEFAULT_IV_NODE` → `"iv"`; all at `DEFAULT_ORAL_NODE` → `"oral"`; mixed/other → `ValueError`).

- **IV** → the existing renal-CL path, **byte-for-byte unchanged** (the dispatch is a pre-check; the IV branch is the current code). Existing IV tests pass unmodified.
- **Oral** →
  1. detect shape (§2): `MeasuredF` present OR ≥2 `MeasuredConc` with distinct phases.
  2. **free-both**: `build_oral_cl_grid(n_grid, s_range)` → `sir_posterior_2d(FPrior, CLPrior, CLGridForward(grid), observations)`. Returns `f`, `cl_scale`.
  3. **free-F-only**: `build_oral_cl_grid(n_grid=1, s_range=(1.0, 1.0))` (one solve, s≡1) → `sir_posterior_2d` (the forward clips s to 1, so F is the only effective latent). Returns `f`, `cl_scale=None`.
  4. Always: `renal_scale=None` (oral frees no renal latent; renal CL stays fixed via `covariates.renal_factor()`). **`meta_cmax=None`, `cmax_90ci=None`** — the conformal q90 is calibrated on **single-dose** holdout Cmax and is invalid for a **steady-state** Cmax (same omission as the IV path, different reason). `warnings` carries the covariate warnings.

`covariates` thread exactly as IV: `renal_factor()` (fixed renal CL), weight/age → `generate_physiology`. `renal_prior_cv` is **IV-only**; oral uses the principled wide `FPrior`/`CLPrior` defaults (cv=1.0 — do **not** tighten, per the DE-41/43 thread); document, and warn if `renal_prior_cv` is set non-default on an oral call.

## 5. Oral `recommend_dose` (route dispatch in `dosing.py`)

The dose-solve **core is route-agnostic and reused verbatim**: oral SS exposure is exactly linear in dose (first-order `ka` input into an LTI system), so `_sample_m_intervals`/`_max_overlap_region`/`_center_m`/`_attainment`/winner/`DoseRecommendation` all apply unchanged.

What changes (oral is a genuine **sibling**, because it threads 1–2 latents vs the IV path's 1):
- Route dispatch (oral requires all-oral events; mixed → `ValueError`).
- Infer the oral posterior once via the oral `predict_tdm` path → latent samples (`f` [+ `cl_scale`]).
- An oral `_interval_reference` sibling: per candidate interval τ_k, `build_oral_cl_grid` at τ_k, `CLGridForward`, evaluate the reference exposures at the posterior `(f, s)` samples (s≡1 when F-only) → `q_ref` {trough, cmax, auc24}. The `trough = grid.conc_at(latents, last+τ_k)`, `auc24 = auc·24/τ_k`, and all downstream LTI algebra are unchanged.
- Candidate regimens via `DosingRegimen.oral_repeated(dose_mg, interval_h, n_doses)` (no `duration_h`).

## 6. Error handling
`ValueError` on: mixed-route regimen (events spanning oral and IV nodes); an oral `recommend_dose`/`predict_tdm` with a non-oral, non-IV node. `build_oral_cl_grid` all-grid-points-fail → propagates `ValueError`. Trough past horizon → `conc_at` raises (horizon is sized to prevent it). Degenerate free-both (e.g. two near-coincident-phase concs that slipped the shape filter) → flagged by the existing `n_eff` diagnostic, not raised.

**Shape tolerance.** Within-interval phase `φ(t) = t mod τ`; concs have "distinct phases" iff `max(φ) − min(φ) > _SHAPE_PHASE_TOL` (default `0.1·τ`). Documented minor limitation: the `0`/`τ` wraparound (a conc just after a dose vs one at end-of-interval) are both trough-like; if such a pair is the only "shape," the free-both clint marginal will be wide and `n_eff` will flag it.

## 7. Testing (TDD; load-bearing first)
1. **Oral LTI exactness** (load-bearing): `build_oral_cl_grid` at `2·D` == the `D` grid ×2 on SS cmax/auc — the premise the oral dose-solve rests on.
2. **Grid faithfulness**: at `s=1`, `build_oral_cl_grid`'s SS trough matches a direct `solve_regimen` oral SS solve; its `f_engine[s≈1]` matches `build_cl_grid`'s `f_engine` at `s=1` (regimen-independence of F).
3. **Adaptive routing**: one `MeasuredConc` → `cl_scale is None` (F-only); two `MeasuredConc` at distinct phases → `cl_scale is not None` (free-both); **two same-phase troughs → `cl_scale is None`** (still F-only — the shape filter); one conc + `MeasuredF` → free-both.
4. **Inference direction** (stack-independent): an oral trough below the engine's s=1 prediction shifts the **F** posterior down (anchored to the engine's own prediction).
5. **Route-dispatch invariance**: an IV regimen through `predict_tdm`/`recommend_dose` is **bit-identical** to today (existing IV tests pass unchanged; add an explicit IV-still-works assertion).
6. **Oral recommend_dose**: feasible-window hit, longer-interval tie-break (the dosing-test patterns, oral regimen).
7. Mixed-route regimen → `ValueError`.
8. **No SS conformal**: oral `predict_tdm` returns `meta_cmax is None` and `cmax_90ci is None`.

Directional tests anchor to the engine's own s=1 prediction (no magic numbers) — stack-independent per the `df4492c` discipline.

## 8. Invariants
`engine/` & `predict()` untouched (reuse `solve_regimen`, `_build_grid_engine`, `_engine_oral_bioavailability`); `PosteriorPK` contract unchanged (`f`/`cl_scale`/`renal_scale`/`warnings` all already exist — additive use); **IV path bit-identical** → holdout & 2.731 headline untouched; identity-blind (operates on engine outputs only); all PK quantities are `Posterior`/`Distribution`; measured-input feature (not benchmarked on the 107-holdout).

## 9. Scope (decomposes into ~5–6 implementation tasks)
oral grid → route helper + oral `predict_tdm` free-F-only → adaptive (shape detect + free-both) → oral `_interval_reference` + oral `recommend_dose` → exports/integration. Bundled per the brainstorming decision (the dose-solve core is verbatim-reusable).

## 10. Out of scope (future)
- Non-steady-state oral titration / loading-dose schedules.
- Absorption-latent (`ka`/lag) freeing — oral frees F + metabolic clint only.
- SI (µmol/L) units; non-uniform oral regimens (assumed uniform, like the IV path).
- A conditioned-case SS conformal recalibration (the single-dose conformal is correctly omitted, not re-derived).
