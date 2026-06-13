# Oral Steady-State TDM — Design

**Date:** 2026-06-12
**Status:** Approved (design); revised after adversarial go-over (2026-06-12); pending implementation plan
**Modules:** `src/sisyphus/mipd/oral_grid.py` (new), `mipd/tdm.py` + `mipd/dosing.py` (route dispatch), `mipd/clgrid.py`/`dosing.py` (two additive `| None` dataclass relaxations)
**Builds on:** `mipd/clgrid.py` (2-latent F+clint forward, `sir_posterior_2d`, `MeasuredConc`), `mipd/renal_grid.py` (multi-dose grid template), `mipd/tdm.py`/`dosing.py` (IV TDM + dose recommendation), `regimen.solver.solve_regimen`, `regimen.profile.compute_steady_state_metrics`

Grounded by a 6-agent understanding sweep (`wf_7bc3305e-551`) and an 8-defect adversarial go-over (`wf_51960a4c-334`, 2026-06-12: verdict **sound-after-edits**; 3 blockers B1–B3 + 2 important I1–I2 + 7 minor M1–M7 folded in below).

---

## 1. Purpose

Extend the MIPD TDM stack from **IV-only** to **oral multi-dose**: condition the engine-as-prior on a steady-state **oral** trough (and optionally a peak, or a measured F) to individualize an oral patient, and recommend a target-attainment oral dose. The IV path (`predict_tdm`, `recommend_dose`) frees a single renal-CL latent because for IV F≡1; for oral, **F re-enters as the dominant latent** (F is the engine's dominant structural error — DE-41/43→FLUX-1 — while CLint, R²~0.24, is already its best estimate).

This is a **measured-input individualization** feature, like the shipped measured-F routing — **not a 107-holdout headline lever** (DE-43: the fixed-weight meta damps engine moves). With zero measured data the path reduces to the a-priori prediction; the 2.731 headline is untouched.

## 2. The identifiability crux + the adaptive stance

At oral steady state, the textbook shorthand is `Css ≈ F·Dose/(CL·τ)` — a single trough is a **magnitude** observation. **This straight `F/CL = const` ridge is the low-extraction (E_h≪1) picture only** (M1). For the high-extraction DE-41/43 population this feature targets, the engine's `cl_int_h` saturates (extended ECM, `flux.py:320-347`) and the emergent `f_engine[s]` co-varies with the metabolic scale `s`, so the iso-trough indeterminate set is a **curved manifold**, not a line. The engine-faithful statement: a single trough pins the curved manifold `(f/f_engine[s])·conc(s; t_trough) = trough` in the grid's two free coordinates `(F_rescale, s)`. The second axis maps **non-proportionally** to physical hepatic CL and is conditional on the engine's fixed V/Kp (there is no volume latent — V error is absorbed into the CL attribution). The freed "F" marginal is the rescale factor `F_rescale = F/f_engine`, **not** physical `fa·Fg·Fh`. The SIR runs on the real grid (not the textbook formula), so this curvature is handled exactly; only the §2 prose simplifies it.

The ridge is broken only by **shape** — a second conc at a *different within-interval phase* adds tail-curvature (`Δlog c` over time) that tracks `ke=CL/V` independently of F — or by a **measured F** (anchors the F axis).

**Adaptive stance (v1):**
- **free-both (F + metabolic clint)** iff the observations contain shape/anchor info: **a `MeasuredF` is present**, **OR** the `MeasuredConc` set has **≥2 distinct within-interval phases** (`t mod τ` spread beyond a tolerance — a peak+trough). Runs `sir_posterior_2d` over the full oral grid; reports both `f` and `cl_scale`.
- **free-F-only** otherwise (the common single-trough case, or repeated same-phase troughs): one effective latent (F) over the grid's **s=1 slice**. Same-phase troughs still tighten the F/exposure posterior — not wasted. Returns `cl_scale=None`.

The phase-gap is a **heuristic proxy** for tail-curvature informativeness, **not** an identifiability guarantee (M2): two flat-tail troughs at different phases carry little shape info. So "free-both" does not mean "identified by construction"; it means "enough shape was supplied that freeing `s` is defensible." When it isn't (a wraparound or flat-tail pair), the `cl_scale` marginal can be silently ridge-wide — see §6.

**Critical honesty point (I1).** Only the **concentration at the observed trough time** is attribution-independent: free-F-only and free-both give the *same* posterior for *that* quantity from a single trough. **Cmax and AUC are NOT attribution-independent** — `Cmax = (F/CL)·(peak/trough ratio)` and the peak/trough ratio is the *strongest* s-dependence (it tracks `ke`), so free-both marginalizes over differently-shaped curves all passing through the trough → a materially wider, shifted Cmax/AUC band than free-F-only, which freezes the shape at the engine's s=1 estimate and therefore **understates exposure uncertainty**. Consequences:
- For `recommend_dose` (which consumes `cmax`/`auc24` targets), single-trough input is **not** exposure-neutral for Cmax-/AUC-targeted recommendations. Default a single-trough recommendation to a **trough target** (the invariant quantity); if a Cmax/AUC target is requested with single-trough input, run **free-both** (the honest wider band) and surface the s=1 shape-approximation caveat in `warnings`.
- State this precisely in the docstrings (§6 references): "the trough-time concentration posterior is attribution-independent; Cmax/AUC are not — free-F-only holds the curve shape at the engine's s=1 estimate."

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
- **`f_engine[s]`** = the engine's emergent oral bioavailability at scale `s`, from a **single-dose** oral/IV reference (the existing `_engine_oral_bioavailability` machinery `build_cl_grid` uses, via `grid._build_grid_engine`). **F is a per-dose property, regimen-independent under linear PK** — so the single-dose F reference is the correct `f_engine` for the SS grid, and `CLGridForward`'s `scale = F/f_engine` (clgrid.py:100) exactly rescales SS exposure (SS exposure ∝ F).

Reuses `grid._build_grid_engine(smiles, dose, "oral", renal_factor, kp_method, body_weight_kg, age_years)` (the route param is already a string), `solve_regimen`, and the NaN-backfill helpers — verbatim except route and the metabolic-`s` scaling.

**Steady-state convergence check (I2).** After the final-interval extraction, verify the solve actually reached steady state via the existing `regimen.profile.compute_steady_state_metrics` (consecutive-Cmax within 5%; **reuse, do not reinvent**). The grid does **not** raise on non-convergence — it records a structured flag the callers turn into a warning (the renal template trusts the caller's `n_doses`; oral must not silently consume a sub-SS trough). `predict_tdm`/`recommend_dose` surface this on `PosteriorPK.warnings` / `DoseRecommendation.warnings`.

## 4. Oral `predict_tdm` (route dispatch in `tdm.py`)

`predict_tdm` reads the regimen route via a shared `_regimen_route(regimen)`: all events at `DEFAULT_IV_NODE` → `"iv"`; all at `DEFAULT_ORAL_NODE` → `"oral"`; mixed/other → `ValueError`.

- **IV** → the existing renal-CL path, with the IV **numerical** path unchanged (the dispatch is a pre-check; the IV branch is the current code — same renal-grid build, same `sir_posterior_renal`, identical default args → the 2.731 guarantee). See M7 for the one cosmetic delta.
- **Oral** →
  1. detect shape (§2): `MeasuredF` present OR ≥2 `MeasuredConc` with distinct phases.
  2. **free-both**: `build_oral_cl_grid(n_grid, s_range)` → `sir_posterior_2d(FPrior(f_engine0, prior_cv), CLPrior(cv=1.0, s_min, s_max), CLGridForward(grid), observations)`. Returns `f`, `cl_scale`.
  3. **free-F-only** (B3 — corrected): `build_oral_cl_grid(n_grid=1, s_range=(1.0, 1.0))` → `sir_posterior_2d(FPrior(f_engine0, prior_cv), CLPrior(cv=1.0, s_min=1.0, s_max=1.0), CLGridForward(grid_s1), observations)`, then **explicitly `dataclasses.replace(post, cl_scale=None)`**.
     - **Why `sir_posterior_2d` and NOT the `APrioriPK`/`SIRAmortizer` F-only path** (the go-over's first instinct, overturned by verification): the oral free-F-only observation is a **steady-state `MeasuredConc` (a trough at time t)**, whose `log_likelihood` needs `state["conc_at"](t)` (clgrid.py:142). Only `CLGridForward` emits `conc_at`; `AnalyticForward` (the `SIRAmortizer` backend) returns only `{f, cmax, auc}` (core.py:60-64) and **cannot evaluate a `MeasuredConc`**. So the time-resolved grid forward is required.
     - **Why it is genuinely F-only and reports `cl_scale=None`**, not the garbage the original spec would have shipped: with `s_grid=[1.0]`, `CLGridForward._interp` clips every draw's `s` to `[1.0, 1.0]`, so `cmax/auc/conc_at` all come from the s=1 row → the likelihood depends only on `f` (no `s` leak). Pinning `CLPrior(s_min=1.0, s_max=1.0)` additionally clips the *sampled* `s` to exactly 1.0, so `state["cl_scale"]` is a degenerate point mass (not a wide resampled prior) and `n_eff` stays meaningful. The explicit `dataclasses.replace(post, cl_scale=None)` then honors the "F-only ⇒ no clint latent reported" contract (`PosteriorPK.cl_scale` is `| None`, core.py:215) and satisfies the §7.3/§8 `cl_scale is None` assertions.
  4. Always: `renal_scale=None` (oral frees no renal latent; renal CL stays fixed via `covariates.renal_factor()`). **`meta_cmax=None`, `cmax_90ci=None`** — the conformal q90 is calibrated on **single-dose** holdout Cmax and is invalid for a **steady-state** Cmax (M5: same omission as the IV path, *different* reason). Direct the user to **`cmax.ci90`** (the F[+clint] parameter-uncertainty band) as the available SS interval, and add a `warnings` note that no calibrated SS conformal exists — mirroring the IV docstring (tdm.py:7-9). `warnings` also carries the covariate warnings and the §3 sub-SS flag.

`covariates` thread exactly as IV: `renal_factor()` (fixed renal CL), weight/age → `generate_physiology`. **`renal_prior_cv` mechanics (M3):** change the oral-reachable signature to `renal_prior_cv: float | None = None` (the IV branch falls back to `1.0` internally → byte-identical); the oral branch **warns iff `renal_prior_cv is not None`** (a plain `float = 1.0` default makes an explicit `1.0` indistinguishable from the default, so an `is None` sentinel is required for a correct warning), carried on `PosteriorPK.warnings`. Oral otherwise uses the principled wide `FPrior`/`CLPrior` defaults (cv=1.0 — do **not** tighten, per the DE-41/43 thread).

## 5. Oral `recommend_dose` (route dispatch in `dosing.py`)

The dose-solve **LTI core is route-agnostic and reused verbatim**: oral SS exposure is exactly linear in dose (first-order `ka` input into an LTI system), so `_sample_m_intervals`/`_max_overlap_region`/`_center_m`/`_attainment`/winner all apply unchanged. **"Verbatim" is scoped to the post-`q_ref` LTI algebra only** (B2) — the orchestrator body is a *sibling*, not a verbatim copy, because it threads 1–2 latents (`f` [+ `cl_scale`]) vs the IV path's single `renal_scale`.

What the oral sibling does:
- Route dispatch via `_regimen_route` (oral requires all-oral events; mixed → `ValueError`).
- Infer the oral posterior once via the oral `predict_tdm` path → latent samples (`f` [+ `cl_scale`]).
- An oral `_interval_reference` sibling: per candidate interval τ_k, `build_oral_cl_grid` at τ_k, then **build the forward once and read all three reference quantities off the same F-scaled state** (B1):
  ```python
  state  = CLGridForward(grid)(f, s)            # s ≡ ones(len(f)) when F-only
  trough = state["conc_at"](last + τ_k)         # carries F/f_engine
  cmax   = state["cmax"]                         # carries F/f_engine
  auc24  = state["auc"] * 24.0 / τ_k
  ```
  **Do NOT copy the IV `dosing.py:196` pattern `grid.conc_at(latents, last+τ)`** — `grid.conc_at` returns the *raw, unscaled* engine concentration (clgrid.py:56-76); the `F/f_engine` vertical scale lives **only** inside `CLGridForward.__call__`'s `conc_at` closure (clgrid.py:100-101). The IV read is correct solely because IV has F≡1; transcribed to oral it would drop exactly the patient-individualization factor and produce a trough mis-scaled against the (correctly-scaled) `cmax`/`auc`.
- The oral sibling reads `post.f.samples` (and `post.cl_scale.samples` when free-both), **never `post.renal_scale`** (which is `None` for oral and would `AttributeError` at the IV body's `dosing.py:272` read), and constructs `DoseRecommendation(..., renal_scale=None, f=post.f, cl_scale=post.cl_scale)`.
- Candidate regimens via `DosingRegimen.oral_repeated(dose_mg, interval_h, n_doses)` (no `duration_h`).
- **Half-life-aware `n_doses` (I2):** the IV heuristic `n_doses = max(2, round(cur_last/τ)+1)` (dosing.py:292) keys candidate length to the *current regimen's span*, not half-life — a long-τ candidate for a long-t½ drug can be modeled with as few as 2 doses, far below SS, biasing the recommended dose high. Size candidates by the engine's effective terminal t½ (from the a-priori solve's `PKEndpoints.t_half`): `n_doses = max(N_min, ceil(k · t_half_eff / τ_k))`, k≈5. Per candidate, run the §3 `compute_steady_state_metrics` check and attach a sub-SS `warning` if not converged. (Time-to-SS is set by terminal t½ and is route-independent — the IV path has the same latent gap and should get the same guard so the siblings stay consistent.)

## 6. Error handling

`ValueError` on: mixed-route regimen (events spanning oral and IV nodes); an oral `recommend_dose`/`predict_tdm` with a non-oral, non-IV node. `build_oral_cl_grid` all-grid-points-fail → propagates `ValueError`.

**Uniform-regimen precondition (M4 — enforced, not assumed).** `predict_tdm`/`recommend_dose` accept an arbitrary hand-built `DosingRegimen`, and `_regimen_interval_h` currently reads `events[1]-events[0]` (the *first* interval). A clinician reconstructing a real irregular dosing history would get a silently wrong SS mask/trough — violating the "never silently drop" doctrine. Add a shared uniformity precondition called from both entrypoints: compute `diff(event_times)`; if the spread exceeds ~1% of the median, **raise `ValueError`** (non-uniform regimens are §10 out-of-scope, so reject rather than mis-model). Have the interval helper return the **final** interval (`events[-1]-events[-2]`) so τ is physically correct even within tolerance.

**Steady-state non-convergence** → structured `warning` (§3/§5), never raised. **Trough past horizon** → `conc_at` raises (the §3 horizon is sized to prevent it).

**Shape tolerance + its honest limits (M2).** Within-interval phase `φ(t) = t mod τ`; concs have "distinct phases" iff `max(φ) − min(φ) > _SHAPE_PHASE_TOL` (default `0.1·τ`). This is a heuristic; two failure modes are documented, **not** silently handled:
- the `0`/`τ` wraparound (a conc just after a dose vs one at end-of-interval) are both trough-like yet read as distinct-phase;
- two concs both on the flat elimination tail (genuinely different phases) carry little curvature.
In both, free-both runs but the `cl_scale` marginal is ridge-wide. **The original "`n_eff` flags it" mitigation is wrong** and is struck: `n_eff = 1/Σw²` flags prior/data *weight*-degeneracy, not a wide-but-supported ridge marginal (a same-phase pair yields *high* `n_eff` with a ridge-wide `cl_scale`). State honestly that this degrades only the reported `cl_scale` marginal — the exposure/dose output is attribution-independent (§2) and unaffected. Optional cheap hardening (defer to plan): gate free-both on `|Δlog c|` between the two sample times at s=1 exceeding a tolerance.

## 7. Testing (TDD; load-bearing first)

Directional tests anchor to the engine's own s=1 prediction (no magic numbers) — stack-independent per the `df4492c` discipline.

1. **Oral LTI exactness** (load-bearing): `build_oral_cl_grid` at `2·D` == the `D` grid ×2 on SS cmax/auc — the premise the oral dose-solve rests on.
2. **Grid faithfulness**: at `s=1`, `build_oral_cl_grid`'s SS trough matches a direct `solve_regimen` oral SS solve; its `f_engine[s≈1]` matches `build_cl_grid`'s `f_engine` at `s=1` (regimen-independence of F).
3. **Adaptive routing**: one `MeasuredConc` → `cl_scale is None` (F-only); two `MeasuredConc` at distinct phases → `cl_scale is not None` (free-both); **two same-phase troughs → `cl_scale is None`**; **a `0`/`τ` wraparound same-phase pair → `cl_scale is None`** (M2 — both routed to F-only by the shape filter); one conc + `MeasuredF` → free-both.
4. **B1 — oral trough carries the F-scale** (load-bearing): the oral `_interval_reference` trough at a posterior with `F ≠ f_engine` equals `(F/f_engine) × grid.conc_at(s, last+τ)` (catches a dropped-scale verbatim copy); anchor directionally to the engine's s=1 prediction.
5. **I1 — attribution honesty**: on one real engine oral grid with a single trough, the **AUC / average-exposure** posterior median + ci90 agree across free-F-only and free-both within MC tolerance (the genuinely-identified quantity, engine-anchored); for **Cmax** assert only that free-both's band is *wider* (a documented bound), **not** equality. The **trough-time** concentration posterior agrees across both paths.
6. **Inference direction** (stack-independent): an oral trough below the engine's s=1 prediction shifts the **F** posterior down (anchored to the engine's own prediction).
7. **B3 — free-F-only is clean**: free-F-only returns `cl_scale is None`; the F posterior is non-degenerate; assert the path does **not** surface a wide resampled `s` (e.g. the internal pre-replace `cl_scale` samples are all ≈1.0).
8. **Route-dispatch invariance** (B2/M7): an IV regimen through `predict_tdm`/`recommend_dose` is **numerically bit-identical** to today (existing IV tests pass unchanged; `test_recommend_renal_scale_shifts_with_observation` still passes; add an explicit IV-still-works assertion).
9. **B2 — oral `DoseRecommendation` contract**: oral recommendation has `renal_scale is None` and `f is not None`; IV recommendation unchanged.
10. **I2 — sub-SS guard**: a long-t½ oral (and IV) 2-dose long-interval regimen yields the sub-SS `warning`.
11. **M3 — renal_prior_cv warning**: present for an oral call with `renal_prior_cv` set non-`None`, absent otherwise.
12. **M4 — non-uniform rejection**: a non-uniform oral (and IV) regimen → `ValueError`.
13. **M6 — auc24 factor**: assert the factor identity directly against the grid's own `auc` — at τ=8 `auc24 == auc·3`, at τ=12 `auc24 == auc·2` (oral; add the same one-line assertion to the existing IV `test_interval_reference_returns_quantities_at_reference_dose`). Do **not** assert τ-invariance of `auc24` (that conflates with I2).
14. **Oral recommend_dose**: feasible-window hit, longer-interval tie-break (the dosing-test patterns, oral regimen).
15. **Mixed-route regimen → `ValueError`.**
16. **No SS conformal (M5)**: oral `predict_tdm` returns `meta_cmax is None` and `cmax_90ci is None`, and **`cmax.ci90` is populated** (assert populated, not "non-degenerate" — `FPrior`'s F→1 one-sided clip can make it degenerate).

## 8. Invariants

`engine/` & `predict()` untouched (reuse `solve_regimen`, `grid._build_grid_engine`, `_engine_oral_bioavailability`, `compute_steady_state_metrics` — import/call only, no mutation); **IV numerical path bit-identical** → holdout & 2.731 headline untouched; identity-blind (operates on engine outputs only); all PK quantities are `Posterior`/`Distribution`; measured-input feature (not benchmarked on the 107-holdout).

**Contract changes — two additive `| None` relaxations:**
- `PosteriorPK` (core.py): **unchanged** — `f`/`cl_scale`/`renal_scale`/`meta_cmax`/`cmax_90ci`/`warnings` are all already `| None` (core.py:211-223); oral's `cl_scale=None`/`renal_scale=None`/`meta_cmax=None` uses are additive.
- `DoseRecommendation` (dosing.py, B2): a **separate** output contract whose `renal_scale: Posterior` is currently non-Optional. Relax to `renal_scale: Posterior | None = None`, and add `f: Posterior | None = None` and `cl_scale: Posterior | None = None`. **Reorder** so the three defaulted fields trail the non-default fields (`n_eff`, `warnings` currently follow `renal_scale`); all construction is keyword-based (dosing.py:337-348), so IV construction stays equivalent.

**M7 — scope of "byte-for-byte".** "Unchanged" is scoped to the IV *numerical* path (renal-grid build + `sir_posterior_renal`, identical default args). One cosmetic delta: today's single IV guard (tdm.py:50) emits "IV regimens only" for *both* oral and mixed regimens; with `_regimen_route`, a mixed regimen's error **text** changes (same `ValueError` type). The two existing oral-rejection tests feed pure-oral regimens and must be **updated** (oral is now supported, not rejected). No numerical change.

## 9. Scope (decomposes into ~6 implementation tasks)

oral grid (+SS-convergence flag) → shared route/uniformity helpers → oral `predict_tdm` free-F-only (B3) → adaptive free-both (shape detect, M2) → `DoseRecommendation` contract relax (B2) + oral `_interval_reference` (B1) + oral `recommend_dose` (I1/I2) → exports/integration. Bundled per the brainstorming decision (the dose-solve LTI core is verbatim-reusable; the orchestrator is a sibling).

## 10. Out of scope (future)
- Non-steady-state oral titration / loading-dose schedules.
- Absorption-latent (`ka`/lag) freeing — oral frees F + metabolic clint only.
- SI (µmol/L) units; **non-uniform oral regimens** (now explicitly *rejected* with `ValueError` per §6, not silently mis-modeled).
- A conditioned-case SS conformal recalibration (the single-dose conformal is correctly omitted, not re-derived; users get `cmax.ci90` as the parameter-uncertainty fallback).
