# MIPD CrCl Renal Individualization + Conditioned-Output Surfacing (v1)

**Date:** 2026-06-11
**Author:** Hypatia (with Jae Min Yoon)
**Status:** Design — approved to write the implementation plan. No implementation until the plan is reviewed.
**One-line:** Individualize the engine-as-prior posterior's renal elimination from a measured creatinine clearance (CrCl → `drug.renal_clearance × CrCl/125`), and surface the un-damped individualized engine posterior (`post.cmax`) as the primary TDM estimate — while keeping the validated population blend + conformal band intact.

This is the first concrete slice of the MIPD "TDM individualization" next layer. It was hardened against five adversarial-review / self-review passes including a 5-agent code-level verification (§2) and a spec self-review that **corrected an over-confident output claim** (§2.1). It is a **foundational** increment — correct renal physics + surfacing the individualized estimate — not a clinically-complete TDM product. Steady-state dosing, weight/age covariates, a *calibrated* conditioned predictive interval, and any renal-impairment benchmark are explicitly **deferred** (§8).

---

## 1. Problem

The `sisyphus.mipd` module (PR #69, merged 2026-06-10) turns the mechanistic engine into a *structural prior* that sparse measured data updates: an F (bioavailability) latent + a metabolic clint-scale latent, conditioned via SIR, with the engine solved once on a clint-scale grid (compile-once / parameterize-many).

Two gaps block it from serving real TDM individualization:

1. **No renal-function covariate.** `build_cl_grid` scales only the drug's *metabolic* intrinsic clearance (`enzyme_affinity`); **renal/biliary clearance is held fixed** (`clgrid.py` module docstring). The dominant individualizing covariate for renally-cleared drugs (vancomycin, aminoglycosides, many antibiotics) — the patient's measured creatinine clearance — has no path into the prediction. Confirmed absent across the whole codebase (no `crcl`/`creatinine`/`egfr` references anywhere).

2. **The individualized estimate is not the surfaced primary.** `mipd.api._attach_meta_and_interval` makes the production **meta blend** (`meta_cmax`) the headline point and a split-conformal band (`cmax_90ci`) the headline interval — for every prediction, including conditioned ones. The meta blend re-mixes three **covariate-blind** SMILES-only tracks (ML/CLF/VDss; 0.72 of the blend weight for a non-base drug with VDss active), so a 4× engine individualization is pulled back to ≈1.47× and a 10× move to ≈1.91× (DE-43). The fully-individualized, conditioned estimate `post.cmax` already exists on the result object but is not surfaced as the primary TDM output. v1 surfaces it.

> **What v1 does NOT do (see §2.1):** it does **not** remove or replace `meta_cmax`/`cmax_90ci`. The original module deliberately chose the conservative conformal band as the user-facing predictive interval (it accounts for structural error; the narrow parameter-uncertainty band under-covers). A *calibrated individualized* interval is data-blocked and deferred (§2.1, §8).

---

## 2. Verification that anchors this design

A 5-agent adversarial verification (workflow `verify-mipd-tdm-design`, 2026-06-11) checked the load-bearing claims against the code. Verdicts:

- **C2 — renal scaling / F_engine invariance: SUPPORTED (high).** Renal CL enters the ODE linearly (`rhs_jax.py:326-339`, mirror `flux.py:285-294`; `compiler.py:103-104` resolves it to `drug.renal_clearance.mean`). The reference renal model is `CL_renal = GFR·fup` with `_GFR_L_PER_H = 7.5` (~125 mL/min) (`ivive.py:42-43`, `_estimate_renal_clearance` `ivive.py:557`). The full engine is **linear time-invariant** in the reference physiology — the only nonlinear flux (`ActiveTransportFluxSpec`, Michaelis-Menten, `rhs_jax.py:398-415`) is **not instantiated** (`flux.py:507` docstring; no `active_transport` edge in `reference_man.yaml`). So scaling `renal_clearance` multiplies the oral **and** the IV-reference AUC by the same factor, leaving the emergent **`F_engine = AUC_oral/AUC_iv` invariant** — the grid's `f_engine` column does not shift. The renal model is **glomerular-filtration-only** (`gfr_filtration`, `reference_man.yaml:346`; no secretion/reabsorption term).

- **C3 — single-dose grid: SUPPORTED (high).** The grid is a single bolus over `t_span=(0,24)` (`grid.py:138-140`), horizon exactly [0,24] h (`_default_t_grid` `grid.py:23-27`), and `conc_at` raises beyond it (`clgrid.py:64-68`). It **cannot** represent a steady-state trough. The clean steady-state path is the **existing** `regimen.solver.solve_regimen` (event-driven multi-dose, horizon auto-extends) — not grid superposition (blocked for accumulating drugs). → steady-state deferred to a regimen-backed follow-up (§8).

- **C1 — `generate_physiology` (weight/age) drop-in: PARTIAL (high).** It IS a structural drop-in for `build_cl_grid`'s `build_from_yaml` (augment→expand→compile→solve runs, `validate()` passes), but it does **not** scale renal CL, and `generate_physiology(70,30)` is **not** bit-identical to the reference (enzyme maturation never reaches 1.0; UGT1A1 differs ~0.1%). → weight/age deferred (§8); v1 touches the reference graph only.

- **C5 — coexistence with `regimen/tdm`: PARTIAL (high).** Real redundancy at the inference-kernel layer; each system is irreplaceable in its lane. `apply_ci_floor` takes a `TDMResult` (`tdm.py:112-146`) so it is **not** a drop-in for a mipd `Posterior` — only the floor *formula* is shared. The charter mandates reusing regimen kernels for the multi-dose/steady-state work (honored by deferring steady-state to `solve_regimen`).

### 2.1 Self-review correction — the C4 output claim was over-stated

C4 verified that the meta blend **damps** the engine individualization (4×→1.47×, 10×→1.91×; ML/CLF/VDss covariate-blind, weights `ensemble.py:29-44`, `_W_VDSS=0.20`, disagreement threshold 10× `_DISAGREEMENT_THRESHOLD_LOG10=1.0`) and that the conformal q90 is a-priori-calibrated (`conformal_calibration.json` meta `0.1=1.1113`, `calibration_set="train"`; `_conformal_q90_meta` `predict.py:30-42`). The **point** finding stands: for TDM, the un-damped `post.cmax` is the individualized estimate.

But the spec's first draft over-reached by proposing to **null `cmax_90ci` and report the SIR parameter-uncertainty band** under conditioning. A spec self-review of the *actual tests* refuted this:

- `test_mipd_meta.py:66` (under `MeasuredF`): *"The product output carries the validated measured-F improvement"* — `meta_cmax` is deliberately the product, conditioned.
- `test_mipd_meta.py:138-149` (under `MeasuredF`): *"`cmax_90ci` is the calibrated conformal band; `meta_cmax.ci90` is the (narrow) F-parameter-uncertainty band. **The predictive interval must be the wider one** … structural error … dominates."*
- `test_mipd_grid.py:65-74` (under `MeasuredConc`): asserts `meta_cmax is not None` and `cmax_90ci is not None  # calibrated PI attached`.

So the original design **already** evaluated the narrow parameter-uncertainty band and **deliberately rejected it as under-covering** — exactly the holdout-measured 29.9%-coverage failure of the parameter-only MC interval (CLAUDE.md / `holdout_pi_coverage_2026-04-24.json`). The conservative conformal is honest-but-wide (review finding #6), never anti-conservative. Nulling it would **reintroduce under-coverage** and break four tests that encode a coherent, deliberate design.

**A *calibrated* individualized conditioned interval is data-blocked in v1:**
- **CrCl path:** no renal-impairment Cmax ground truth exists in the repo (the same wall that limits v1 validation to mechanism + direction). Cannot calibrate.
- **Measured-observation path:** calibrating on self-simulated observations is circular (conditioning on a noised copy of the truth → optimistically narrow); calibrating on independent observations (literature F/AUC) is real but underpowered (N≈10 per regime).
- **Re-centering trap:** placing the meta-calibrated conformal *width* around the engine-track `post.cmax` center has unknown coverage (the validation was for conformal-around-`meta`), so it is not "safe."

Therefore v1 **keeps** the validated `meta_cmax` + `cmax_90ci` and **surfaces** `post.cmax` as the documented individualized primary, labeling each band honestly. A coverage-validated conditioned interval is deferred (§8) — shipping an unvalidated predictive interval in a clinical tool would violate the project's conformal-coverage discipline.

---

## 3. Scope

**In v1:**
- A `Covariates(crcl_ml_min)` input that scales `drug.renal_clearance` by `CrCl/125` before the engine solve.
- A dispatch fix so CrCl individualizes the engine solve **without** spuriously freeing the metabolic clint latent.
- **Surface the individualized estimate:** document `post.cmax` (+ `post.cmax.ci90`) as the patient-individualized engine posterior — the primary output for TDM/covariate use — while `meta_cmax` (population blend, damped under conditioning) and `cmax_90ci` (validated conservative conformal) are retained unchanged.
- A structured `warnings` field on `PosteriorPK` (additive) for flags such as extreme CrCl.

**Deferred (§8):** weight/age covariates (`generate_physiology`); steady-state / multi-dose (regimen-backed); a **coverage-validated conditioned predictive interval** (review #6 — data-blocked); CrCl-from-serum-creatinine (Cockcroft-Gault); any renal-impairment quantitative benchmark.

---

## 4. Design

### 4.1 CrCl → renal clearance (the new physics)

Renal clearance is individualized by scaling the drug-level `renal_clearance` Distribution **once** on the base drug (renal CL is covariate-fixed, not a per-grid-point latent), in the predict/grid layer — the engine stays identity-blind (Invariant 1). The per-scale metabolic-clint scaling (`enzyme_affinity`) then happens on top, unchanged.

```
_REFERENCE_GFR_ML_MIN = 125.0          # = _GFR_L_PER_H 7.5 L/h; ivive.py:42-43
renal_factor = crcl_ml_min / _REFERENCE_GFR_ML_MIN     # 1.0 when crcl == 125
# applied ONCE to the base drug, before the grid loop / before the single solve:
drug = replace(drug, renal_clearance=Distribution(
    mean=drug.renal_clearance.mean * renal_factor, cv=drug.renal_clearance.cv))
```

Same `dataclasses.replace` pattern the grid already uses for `enzyme_affinity` (`grid.py:128-134`). The renal factor is **deterministic** (a covariate, not a free latent) and **orthogonal** to the metabolic clint latent: renal CL is fixed by CrCl; the clint latent is what a measured concentration updates.

Because `F_engine` is invariant to renal scaling (C2), the renal individualization enters purely through `cmax0`/`auc0`/curve shape — never through the F prior. The engine solve captures Cmax and AUC responding to renal CL **differently** (no proportionality assumption is baked in).

**Applicability boundary (documented, not a defect):** the engine renal model is glomerular-filtration-only. CrCl-scaling is exact for filtration-cleared drugs and stays self-consistent within the engine, but the engine itself under-models drugs with major active tubular **secretion/reabsorption** (organic-anion/cation substrates). `F_engine` invariance is exact at infinite-time AUC and approximate on the 24 h truncation (negligible for typical TDM drugs). CrCl is treated as a point covariate in v1 (its own measurement uncertainty is not propagated — deferred).

### 4.2 Dispatch (correctness fix)

The metabolic clint latent must be freed **only** when a curve-shape observation can identify it (`MeasuredConc`), exactly as today. CrCl must individualize the engine solve **without** forcing the 2-latent grid (which would leave the clint latent prior-wide and over-widen `post.cmax` when no concentration was measured), and CrCl must never be silently dropped by the reference-physiology F-only path.

```
renal_factor      = covariates.renal_factor() if covariates else 1.0
needs_clint_grid  = cl_latent or any(isinstance(o, MeasuredConc) for o in observations)

if needs_clint_grid:
    grid = build_cl_grid(..., renal_factor=renal_factor)      # 13-pt grid; clint freed
    post = sir_posterior_2d(...)
elif renal_factor != 1.0:
    # individualized single engine solve at clint-scale s=1, clint FIXED
    c0, a0, fe = <single renal-scaled solve>                  # see note
    post = SIRAmortizer(...).posterior(APrioriPK(c0, a0, fe), observations)
else:
    # current reference F-only inference path via predict() (UNCHANGED, bit-identical)
    ap = predict(..., compute_f_engine=True); post = SIRAmortizer(...).posterior(...)
```

**Single renal-scaled solve — plan-level choice (do not over-specify):** the simplest reuse is `build_cl_grid(n_grid=1, s_range=(1.0, 1.0), renal_factor=renal_factor)` and read the s=1 scalars `cmax[0]/auc[0]/f_engine[0]` (the `conc_at` interpolation fragility at `n_grid=1` does not bite because the F-only path never calls `conc_at`). Alternatively, extract the per-scale solve body from `build_cl_grid` into a shared helper. Either keeps a single solve code path; the plan picks. The faithfulness pin (`test_cl_grid_at_unit_scale_reproduces_predict_engine_pk`) applies at `renal_factor=1`.

Note: the individualized path runs two solve sessions — `predict()` for the covariate-blind meta tracks (needed regardless) and the renal-scaled solve for the engine track. Acceptable; the meta tracks are SMILES-only by construction.

### 4.3 Output surfacing (the converged design)

`predict_posterior` returns one `PosteriorPK` carrying **all** of: `post.cmax` (engine-track, conditioned + covariate-individualized), `meta_cmax` (population blend), and `cmax_90ci` (calibrated conformal). v1 changes **which is documented as primary**, not which fields exist:

| Field | Meaning | Primary for |
|---|---|---|
| `post.cmax` (+ `.ci90`) | engine-track posterior, fully conditioned + CrCl-individualized | **TDM / patient individualization** |
| `meta_cmax` (+ `.ci90`) | production population blend (covariate-blind ML/CLF/VDss mixed in; **damped** under conditioning, DE-43) | SMILES-anchor / population product |
| `cmax_90ci` | train-calibrated split-conformal predictive band around the meta point (**only** coverage-validated band; conservative under conditioning, review #6) | the honest user-facing predictive interval |

`post.cmax.ci90` is documented as a **parameter-uncertainty** band (F + clint latents) that **under-covers** structural error — it is the optimistic bracket, not a calibrated predictive interval. `cmax_90ci` is the conservative (validated) bracket. A *calibrated individualized* interval that sits honestly between them is deferred (§2.1, §8).

**This is non-breaking:** `meta_cmax`/`cmax_90ci` are unchanged, so the four tests in §2.1 pass as-is. The only output-structure change is the additive `warnings` field (§4.5). The new behavior is: when CrCl is supplied, `post.cmax` (and, damped, `meta_cmax`) shift to the individualized values; when nothing is supplied, output is bit-identical to today. New tests assert `post.cmax`'s primacy/individualization (§7); no existing test is rewritten. *(An optional machine-readable `primary`/`regime` marker on `PosteriorPK` is a nice-to-have — deferred unless the plan finds it cheap.)*

### 4.4 Opt-in CI floor

`min_ci_half_width_fraction: float = 0.0` (default **off**) on `predict_posterior`. When `> 0`, widen `post.cmax.ci90` to half-width `frac × post.cmax.mean` if narrower (the rivaroxaban over-tight-posterior guard). The floor **formula** mirrors `regimen.tdm._apply_ci_floor` (`tdm.py:123-146`); since `apply_ci_floor` itself takes a `TDMResult`, v1 uses a small pure helper `_ci_floor(ci, mean, frac) -> ci` (cited to the regimen original; prefer extracting a shared pure helper over duplication). Default off avoids silently baking in regimen's un-revalidated `0.20`. Applies to `post.cmax.ci90` only (the param-uncertainty bracket); the conformal `cmax_90ci` is not floored.

### 4.5 API

```python
# mipd/covariates.py  (NEW)
_REFERENCE_GFR_ML_MIN = 125.0          # = _GFR_L_PER_H (7.5 L/h); ivive.py:42-43

@dataclass(frozen=True)
class Covariates:
    """Patient covariates that deterministically individualize the engine prior.

    v1: renal function only (measured creatinine clearance). Weight/age are a
    documented future extension (generate_physiology) — see the design spec.
    """
    crcl_ml_min: float | None = None      # measured creatinine clearance, mL/min

    def __post_init__(self) -> None:
        if self.crcl_ml_min is not None and self.crcl_ml_min <= 0:
            raise ValueError(f"crcl_ml_min must be > 0, got {self.crcl_ml_min}")

    def renal_factor(self) -> float:
        if self.crcl_ml_min is None:
            return 1.0
        return self.crcl_ml_min / _REFERENCE_GFR_ML_MIN
```

`predict_posterior(..., covariates: Covariates | None = None, min_ci_half_width_fraction: float = 0.0)`. `build_cl_grid(..., renal_factor: float = 1.0)`.

**`PosteriorPK` gains** `warnings: tuple[str, ...] = ()` (additive, default empty → non-breaking; aligns with the project's structured-result error doctrine — *never silently drop*). A `warnings` entry (not an error) is emitted for physiologically extreme CrCl (outside ~[5, 200] mL/min), where the filtration-only model is least trustworthy; the prediction still proceeds.

---

## 5. Invariants & faithfulness guards

1. **Engine identity-blind (Inv 1):** individualization is a `replace` on `drug.renal_clearance` at the predict/grid layer; no `engine/` change.
2. **Distributions (Inv 2):** `renal_clearance` stays a `Distribution` (mean scaled, cv preserved).
3. **Compile-once (Inv 3):** the single renal-scaled solve reuses the grid's compile/parameterize pattern.
4. **Not a fudge (Inv 8):** CrCl→GFR→renal CL is a mechanistic covariate, not a fit-to-Cmax-loss tuning knob.
5. **Headline + conditioned contract untouched:** `predict()` is not modified. `predict_posterior(covariates=None, observations=())` is **bit-identical** to today. `meta_cmax`/`cmax_90ci` are unchanged for all paths, so the existing conditioned tests (§2.1) pass unmodified. The 2.731/2.784 holdout headline is unaffected.

---

## 6. Error handling

- `crcl_ml_min <= 0` → `ValueError` (in `Covariates.__post_init__`).
- Observation time beyond the 24 h single-dose grid horizon → `ValueError` (existing `conc_at`, surfaced as an explicit single-dose precondition).
- Steady-state / multi-dose troughs are **out of scope** — documented precondition, not silently mis-handled.
- Extreme CrCl (~outside [5, 200] mL/min) → a `PosteriorPK.warnings` entry; the prediction proceeds.

---

## 7. Testing & validation

Honest scope: there is **no renal-impairment PK benchmark in the repository** (the holdout is healthy-volunteer Cmax). v1 is validated **mechanistically and directionally only**; a quantitative renal-impairment benchmark is a separate data-acquisition effort (§8). This follows the correctness-over-benchmark discipline — ship correct physics, do not imply a benchmark move.

Unit / integration tests (test-first):
- `Covariates.renal_factor()`: `crcl=125 → 1.0`; `crcl=62.5 → 0.5`; `crcl<=0 → ValueError`.
- **Bit-identity guard:** `predict_posterior(covariates=None, observations=())` equals the current output (post samples + `meta_cmax` + `cmax_90ci`) — pins Invariant 5.
- **Existing conditioned tests still pass** (the four in §2.1) — pins that v1 is non-breaking.
- **F_engine invariance:** `build_cl_grid(renal_factor=0.4)` has the same `f_engine` column as `renal_factor=1.0` (rel tol ~1e-3, allowing the 24 h-truncation residual) — pins C2.
- **Dispatch:** `covariates=Covariates(crcl=25)` with no observations routes through the single renal-scaled solve (clint latent **not** freed; `post.cl_scale is None`), not the 2-latent grid.
- **Directional clinical sanity:** for a **high renal-fraction drug** (high `fup`, low metabolic clearance — so `renal_cl = 7.5·fup` is a large fraction of total CL), low CrCl → higher AUC/Cmax in `post.cmax`; for a hepatically-cleared drug (e.g. midazolam, `fup≈0.03` → renal fraction negligible) ≈ no change. (The plan selects concrete drugs by computed renal fraction.)
- **Output surfacing:** with CrCl supplied, `post.cmax` reflects the individualization (un-damped) while `meta_cmax` moves less (damped); `cmax_90ci` remains the conformal band.
- **Floor:** `min_ci_half_width_fraction=0.20` widens an over-tight conditioned `post.cmax.ci90`; default `0.0` leaves it unchanged; `cmax_90ci` is never floored.
- **Warnings:** extreme CrCl (e.g. 3 mL/min) produces a `PosteriorPK.warnings` entry; normal CrCl produces none.

---

## 8. Explicitly deferred (not in v1)

- **Coverage-validated conditioned predictive interval** — recalibrate a conditioned/individualized conformal quantile so the band is both individualized and ≥0.90-covering (review finding #6). **Data-blocked** in v1: no conditioned/renal-impairment ground truth; self-simulated-observation calibration is circular; independent-observation calibration is underpowered (§2.1). This is a data-acquisition + calibration effort, not a code change.
- **Steady-state / multi-dose TDM** — via `regimen.solver.solve_regimen` (event-driven, accumulation-faithful), re-solved across the clint-scale grid; reuse `regimen.types.DosingRegimen`. NOT grid superposition (blocked for accumulating drugs).
- **Weight/age covariates** — via `sbi.physiology_generator.generate_physiology` (verified drop-in but needs its own integration test + accepts the ~0.1% non-identity at the reference; gate the swap on weight/age being supplied).
- **CrCl from serum creatinine** (Cockcroft-Gault: + age/weight/sex).
- **CrCl measurement-uncertainty propagation** (CrCl as a Distribution, not a point).
- **Renal-impairment quantitative benchmark** (data acquisition).
- **Regimen/mipd observation-type reconciliation** — `mipd.MeasuredConc {value,t,cv,lognormal}` vs `regimen.Observation {time_h,concentration,node,cv,normal}`; converge or document the divergence so assay-error semantics don't silently fork.
- **Optional `PosteriorPK.primary`/`regime` marker** — machine-readable signal of which point field is primary, if the plan finds it cheap.

---

## 9. File-level change list

- **NEW** `src/sisyphus/mipd/covariates.py` — `Covariates` dataclass, `_REFERENCE_GFR_ML_MIN`.
- **MODIFIED** `src/sisyphus/mipd/grid.py` — `renal_factor` param applied once to the base drug; (optionally) factor the per-scale solve for reuse by the single-solve path.
- **MODIFIED** `src/sisyphus/mipd/api.py` — `covariates` + `min_ci_half_width_fraction` params; dispatch branch (single renal-scaled solve); `_ci_floor` helper applied to `post.cmax.ci90`; populate `warnings`; docstrings surfacing `post.cmax` as the TDM primary.
- **MODIFIED** `src/sisyphus/mipd/core.py` — add `PosteriorPK.warnings: tuple[str, ...] = ()`; docstring clarifying the three outputs (§4.3) and that `post.cmax.ci90` under-covers while `cmax_90ci` is the validated conservative band. The `_ci_floor` pure helper (§4.4) lives here or in `mipd/api.py` — a plan-level placement detail.
- **NEW** `tests/unit/test_mipd_covariates.py`; additions to `tests/unit/test_mipd_api.py`.

No changes to `engine/`, `predict/predict()`, the holdout, any headline artifact, or the existing `meta_cmax`/`cmax_90ci` behavior.
