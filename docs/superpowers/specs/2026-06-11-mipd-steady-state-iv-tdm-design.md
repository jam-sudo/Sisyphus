# MIPD Steady-State IV TDM — Renal-CL Latent (v1)

**Date:** 2026-06-11
**Author:** Hypatia (with Jae Min Yoon)
**Status:** Design — approved to write the implementation plan. No implementation until the plan is reviewed.
**One-line:** Condition the engine-as-prior on a **steady-state IV trough** to individualize a patient's **renal clearance** — a free renal-CL latent whose prior is centered on the CrCl-implied value and updated by the measured trough, over a multi-dose engine solve (`regimen.solver.solve_regimen`, reused).

This is the deferred steady-state/multi-dose TDM item from the CrCl spec (`docs/superpowers/specs/2026-06-11-mipd-crcl-renal-individualization-design.md` §8). It brings the just-shipped CrCl renal individualization + the engine-as-prior framing to the **clinically dominant** TDM regime: IV steady-state troughs (vancomycin, aminoglycosides). v1 is IV / single-trough / renal-drug scoped; oral steady-state, multi-trough, and a metabolic latent for IV hepatic drugs are deferred (§9).

---

## 1. Problem

The MIPD module conditions on a **single-dose, 0–24 h** concentration via the CL-grid (verified C3 of the CrCl spec: `grid.py` solves one bolus over `t_span=(0,24)`; `conc_at` raises beyond 24 h). Real TDM is **steady-state**: a vancomycin trough after ~5 doses q12 h (≈48–60 h) is structurally unrepresentable — past the horizon AND a multi-dose accumulated concentration the single-bolus curve never contains.

The clinically dominant TDM drugs are **IV and renally cleared** (vancomycin, aminoglycosides). For these:
- **F ≡ 1** (IV → no bioavailability latent; the engine-as-prior's dominant structural error, F, is absent — the cleanest engine-as-prior regime).
- The trough constrains **clearance**, and for renal drugs the metabolic clint latent has ~no leverage — the trough must update **renal CL**.
- **CrCl** (the v1 covariate) provides the renal-CL *prior*; the measured trough provides the *posterior* — the classic Bayesian MIPD structure.

The mipd module is currently **oral-only** (`predict_posterior` raises for non-oral). v1 adds an IV steady-state path.

---

## 2. Decisions (from brainstorming)

- **Route: IV-first.** IV infusion multi-dose; F ≡ 1 (no F latent). (Oral steady-state deferred.)
- **Latent: free renal-CL scale `r`.** Prior lognormal centered on 1.0 (= the CrCl-implied renal CL), wide cv (renal CL is uncertain); the trough updates it. The metabolic clint latent is held fixed (engine value); F is absent.
- **Approach A: a mipd low-D renal-CL grid built via `solve_regimen` (reused).** The multi-dose *solve* is `regimen.solver.solve_regimen` (event-driven, accumulation-faithful, horizon auto-extends to `last_dose + 24 h`); the *inference* is the mipd low-D SIR (≈`n_grid` solves, not the 2000-sample full-param IS of `regimen.tdm.bayesian_update`). This honors the CrCl spec §8 ("reuse `solve_regimen`") and extends the v1 CrCl + engine-as-prior + conformal framing to IV steady-state. (Coexistence with `regimen.tdm` is an explicit boundary — §8.)

---

## 3. Scope

**In v1:**
- A new entry point `mipd.tdm.predict_tdm(smiles, regimen, observations, covariates=None)` for IV steady-state TDM.
- A renal-CL steady-state grid (`build_renal_cl_grid`) that re-solves `solve_regimen` across a renal-CL scale `r`, scaling `drug.renal_clearance` per `r` (CrCl sets the prior center via `renal_factor`).
- A free renal-CL latent updated by a steady-state `MeasuredConc` trough (reuse the existing observation type).
- Honest IV-TDM output: the conditioned engine-track posterior is primary; the oral-calibrated `meta_cmax`/`cmax_90ci` are **not attached** for IV (they are oral-only artifacts).

**Deferred (§9):** oral steady-state (`oral_repeated`, F latent retained); multiple troughs / full concentration-time TDM; a metabolic-CL latent for IV hepatically-cleared drugs; a coverage-validated conditioned interval (data-blocked); a vancomycin renal-impairment benchmark (data acquisition).

### 3.1 Honest framing — what v1 is and is NOT

v1 builds and directionally validates the **mechanism** (engine-prior + CrCl + steady-state trough → renal-CL posterior). It is **not** a clinically-deployable drug-specific tool, for two structural reasons:

- **The engine ADME prior may be weak for the motivating drugs.** Vancomycin (MW 1449 glycopeptide) and aminoglycosides are large, hydrophilic, atypical molecules — exactly where the small-molecule `predict_adme` baseline (fup, Vd, the GFR×fup renal estimate) is least reliable. The engine-as-prior is only as good as its prior; for these molecules it may be poor, so the trough does heavy lifting.
- **A single trough + a CL-only latent corrects CL, not V.** The renal-CL latent `r` matches the trough by scaling clearance, but the volume of distribution (→ the peak, the loading) is taken from the engine and is **not** updated. If the engine's Vd is wrong, the predicted peak can be off even when the trough is matched. Separating CL and V needs a second observation (peak+trough) — deferred (§9).

So v1's claim is: *the conditioning loop is correct and moves in the right direction on a measured trough*. Drug-specific clinical accuracy (e.g. vancomycin AUC24-guided dosing) additionally requires the engine ADME to be adequate for that molecule (a separate validation) and the V/multi-trough follow-up. This follows the correctness-over-benchmark discipline — ship a correct mechanism, do not overclaim drug-level accuracy.

---

## 4. Design

### 4.1 The renal-CL steady-state grid (`build_renal_cl_grid`)

Mirrors `build_cl_grid` but (a) the grid axis is a **renal-CL scale `r`** (scales `drug.renal_clearance`, not `enzyme_affinity`), and (b) the solve is **multi-dose** via `solve_regimen`.

```
build_renal_cl_grid(smiles, regimen: DosingRegimen, *, n_grid=13,
                    r_range=(0.2, 5.0), renal_factor=1.0, kp_method=...)
  -> RenalCLGrid(r_grid, t_grid, conc, cmax, auc)
```

- Build `profile/adme/graph/drug` exactly as `build_cl_grid` does (reuse `detect_disposition` etc.), then apply the **CrCl** scale once: `drug.renal_clearance *= renal_factor` (the v1 mechanism — sets the prior center).
- Compile once; for each `r` in a geomspace `r_grid`: `drug_r = replace(drug, renal_clearance=Distribution(mean*r, cv))`; `solve_regimen(compiled, params_r, regimen, t_total_h=last_dose_time + max(interval_h, 24))` — the horizon must cover the **full final dosing interval** (the default `last_dose + 24 h` truncates a q24 h+ interval's AUC_τ,ss). Record the venous concentration curve re-gridded onto a common `t_grid`, plus `cmax` and `auc`.
- **`cmax`/`auc` are the final-dosing-interval quantities** (clinical TDM standard, approximate steady state): `cmax` = peak over `[last_dose_time, last_dose_time + τ]` (τ = `interval_h`), `auc` = AUC over that final interval (AUC_τ,ss). These approximate steady state **only if the regimen ran long enough to converge** (≳4–5 half-lives) — the caller sizes `n_doses`; the spec does not assert convergence. The **conditioning is independent of this**: the trough likelihood uses `conc_at(t)` at the observation's *actual* time on the simulated curve, not a steady-state assumption.
- No `f_engine` column — F ≡ 1 for IV (the F latent does not exist on this path).
- `t_grid` spans `[0, last_dose + 24 h]`; `conc_at(r, t)` interpolates in time then across `r` in log-log (mirror `CLGrid.conc_at`), and **raises** if `t` is outside the grid horizon (an explicit precondition — a trough time must lie within the simulated regimen).

The grid lives in a new `mipd/renal_grid.py` (the multi-dose solve differs enough from the single-bolus `grid.py` that a separate module is cleaner than overloading `build_cl_grid`).

### 4.2 The renal-CL latent + CrCl prior

```
RenalCLPrior(cv=1.0, r_min, r_max):  r ~ lognormal(median=1.0, cv), clipped to the grid range
```

`r` multiplies the renal CL **on top of** the CrCl-set value, so the prior is centered on the CrCl-implied clearance (`r=1.0`) with wide cv (renal CL is the engine's individual-level unknown). The CrCl covariate sets the *center*; the trough sets the *posterior*. Metabolic clint is fixed (engine value); F is absent.

### 4.3 Forward + SIR (no F latent)

`RenalCLForward(grid)` maps `r -> {r, cmax, auc, conc_at}` by log-log interpolation over `r_grid` (mirror `CLGridForward`, minus the F scaling). `sir_posterior_renal(prior, forward, observations, n_samples, rng)`: draw `r` from the prior, weight by the joint observation likelihood (`MeasuredConc.log_likelihood` uses `conc_at(t)` — reused unchanged), resample via `_softmax_resample` (reused from `core`), report `n_eff`. The trough on the elimination phase of the steady-state curve identifies `r` (= the patient's renal CL relative to the CrCl estimate).

### 4.4 Entry point — `mipd/tdm.py::predict_tdm`

```python
def predict_tdm(
    smiles: str,
    regimen: DosingRegimen,            # iv_infusion(dose, duration_h, interval_h, n_doses)
    observations,                      # MeasuredConc troughs (within the regimen horizon)
    *,
    covariates: Covariates | None = None,   # v1: CrCl sets the renal-CL prior center
    renal_prior_cv: float = 1.0,
    n_samples: int = 20000,
    n_grid: int = 13,
    seed: int = 0,
    kp_method: str = "rodgers_rowland",
) -> PosteriorPK
```

- Validates the regimen is IV (all events at the IV node `venous_blood`); raises `ValueError` otherwise (oral steady-state is deferred).
- `renal_factor = covariates.renal_factor() if covariates else 1.0`; warns on extreme CrCl (reuse the v1 rule).
- Builds the renal-CL grid, runs `sir_posterior_renal`, returns a `PosteriorPK` whose `cmax`/`auc` are the conditioned steady-state engine posterior and whose new `renal_scale` field carries the `r` posterior. `f` is a degenerate F≡1 Posterior (IV).

A **new entry point** (not an overload of `predict_posterior`) keeps the oral-F logic and the IV/F≡1/multi-dose/renal-CL logic cleanly separated.

### 4.5 Output (honest for the IV regime)

The v1 oral output rule kept `meta_cmax`/`cmax_90ci` because they are oral-train-calibrated. For **IV steady-state** neither is valid (the ML/CLF/VDss tracks and the conformal q90 are oral-Cmax artifacts). So `predict_tdm`:
- **Primary:** `post.cmax` (+ `post.cmax.ci90`) — the conditioned steady-state engine posterior. `post.renal_scale` carries the individualized renal-CL factor.
- **Does NOT attach** `meta_cmax`/`cmax_90ci` (left `None`) — no oral-calibrated population blend or conformal band is claimed for IV.
- `post.cmax.ci90` is documented as a **parameter-uncertainty** band (renal-CL latent only) that does **not** carry calibrated structural coverage — same honesty caveat as v1; a calibrated conditioned interval is deferred (data-blocked).

### 4.6 Observation

A steady-state trough is the existing `MeasuredConc(value, t, cv)` at a time `t` within the regimen horizon (e.g. just before the 5th dose). No new observation type. `conc_at` raises if `t` exceeds the simulated horizon (explicit precondition; the caller sizes the regimen to cover the trough).

---

## 5. Invariants

1. **Engine identity-blind (Inv 1):** all scaling is `dataclasses.replace` on `drug` at the predict/grid layer; no `engine/` change.
2. **Distributions (Inv 2):** `renal_clearance` stays a `Distribution`.
3. **Compile-once (Inv 3):** the grid compiles once and re-solves `solve_regimen` per `r`.
4. **Reuse, don't duplicate:** the multi-dose solve is `regimen.solver.solve_regimen`; the regimen is `regimen.types.DosingRegimen`/`iv_infusion`; the resample is `core._softmax_resample`; the likelihood is the existing `MeasuredConc`. New code is the renal-CL grid/forward/prior + the `predict_tdm` orchestration.
5. **Existing contracts untouched:** `predict()`, `predict_posterior`, and the oral CL-grid path are unmodified. `PosteriorPK` gains one additive field `renal_scale: Posterior | None = None` (default None → non-breaking).
6. **Headline untouched:** no change to `predict()` or any holdout artifact.

---

## 6. Error handling

- Non-IV regimen (any event not at `venous_blood`) → `ValueError` (oral steady-state deferred).
- Empty `regimen` / invalid dose/interval → reuse `DosingRegimen`/`iv_infusion` validation (`ValueError`).
- Trough time outside the grid horizon → `ValueError` (existing `conc_at` guard; explicit precondition).
- `crcl_ml_min <= 0` → `ValueError` (reuse `Covariates`). Extreme CrCl → `PosteriorPK.warnings` entry.
- Engine fails at all grid points → `ValueError` (mirror `build_cl_grid`).

---

## 7. Testing & validation

Honest scope: **no renal-impairment steady-state ground truth in the repo** → mechanism + directional validation only; a vancomycin benchmark is a data-acquisition effort (§9).

- `RenalCLPrior` math: median 1.0, clip to grid range.
- **Grid via solve_regimen:** `build_renal_cl_grid` with an `iv_infusion` regimen produces a steady-state curve whose horizon spans `last_dose + 24 h`; `conc_at` raises beyond it.
- **Renal scaling monotonicity:** lower `r` (lower renal CL) → higher steady-state AUC/trough at every grid point.
- **CrCl prior center:** lower CrCl (`renal_factor < 1`) shifts the grid to higher exposure (the prior center moves), independently of the latent.
- **Trough conditioning (directional MIPD):** at steady state `Css ∝ 1/CL`, so a **low** measured trough implies **faster** clearance → posterior `r` **> 1** → **lower** predicted steady-state exposure; a **high** trough → `r` **< 1** → **higher** exposure. The `renal_scale` posterior moves toward the trough-implied clearance. (Sign matters — the earlier draft inverted this.)
- **Identifiability:** a trough on the elimination phase yields `n_eff` well above the degeneracy floor; document that a peak-only sample is weakly informative.
- **Reuse faithfulness:** the grid's per-`r` solve equals a direct `solve_regimen` call at that `r` (pins the grid to the reused solver).
- **Output honesty:** `predict_tdm` returns `meta_cmax is None` and `cmax_90ci is None`; `post.cmax.point > 0`; `post.renal_scale is not None`.
- **IV-only guard:** an oral regimen → `ValueError`.

---

## 8. Coexistence with `regimen/tdm` (explicit boundary)

`regimen.tdm.bayesian_update(method="ibis"/"sbi")` already performs multi-dose IV TDM via full-parameter importance sampling (fup/peff/total_enzyme_affinity sampled, ~2000 `solve_regimen` simulations, multi-method routing). `predict_tdm` is the **low-D engine-as-prior alternative**: one renal-CL latent, a ~`n_grid`-solve grid surrogate, CrCl-as-prior, and the mipd output framing. They solve the same clinical problem by different means and **coexist**: `predict_tdm` reuses `solve_regimen`/`DosingRegimen` (not the inference). This is the C5 boundary, made explicit — not a unification (out of scope). When the two should merge is a future architectural decision.

---

## 9. Deferred (not in v1)

- **Oral steady-state** (`oral_repeated`): retains the F latent alongside renal-CL; needs the joint F/CL identifiability handling.
- **Multiple troughs / full concentration-time TDM** (peak+trough → CL and V separately).
- **Metabolic-CL latent for IV hepatically-cleared drugs** (the trough updates metabolic clint instead of renal).
- **Coverage-validated conditioned interval** (review #6; data-blocked).
- **Vancomycin/aminoglycoside renal-impairment benchmark** (data acquisition).
- **Merge with `regimen.tdm`** (unify the low-D and full-param TDM paths).

---

## 10. File-level change list

- **NEW** `src/sisyphus/mipd/renal_grid.py` — `RenalCLGrid`, `build_renal_cl_grid` (via `solve_regimen`), `RenalCLForward`, `RenalCLPrior`, `sir_posterior_renal`.
- **NEW** `src/sisyphus/mipd/tdm.py` — `predict_tdm` (IV steady-state entry; regimen validation; CrCl prior; output).
- **MODIFY** `src/sisyphus/mipd/core.py` — add `PosteriorPK.renal_scale: Posterior | None = None` (additive); docstring.
- **NEW** `tests/unit/test_mipd_renal_grid.py`, `tests/unit/test_mipd_tdm.py`.

No changes to `engine/`, `predict/predict()`, `predict_posterior`, the oral CL-grid path, the holdout, or any headline artifact. `mipd/` goes from 8 to 10 files (within the 20-file ceiling).
