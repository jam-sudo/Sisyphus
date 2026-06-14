# Oral Steady-State TDM — Design

**Date:** 2026-06-12
**Status:** Approved (design); revised after two adversarial go-overs (2026-06-12); ready for implementation plan
**Modules:** `src/sisyphus/mipd/oral_grid.py` (new), `src/sisyphus/mipd/_regimen.py` (new — shared route/uniformity/interval/shape helpers), `mipd/tdm.py` + `mipd/dosing.py` (route dispatch), `mipd/clgrid.py`/`dosing.py` (two additive `| None` dataclass relaxations)
**Builds on:** `mipd/clgrid.py` (2-latent F+clint forward, `sir_posterior_2d`, `MeasuredConc`), `mipd/renal_grid.py` (multi-dose grid template), `mipd/grid.py` (`_build_grid_engine`, `build_cl_grid`, NaN-backfill helpers), `mipd/tdm.py`/`dosing.py` (IV TDM + dose recommendation), `regimen.solver.solve_regimen`, `regimen.profile.compute_steady_state_metrics`, `pk.nca.terminal_half_life`

Grounded by a 6-agent understanding sweep (`wf_7bc3305e-551`) and two adversarial go-overs (`wf_51960a4c-334` → B1–B3/I1–I2/M1–M7; `wf_63081dc1-1c2` → Bx1–Bx6/Mx1–Mx3, verdict **yes-after-edits**). All findings folded below.

---

## 1. Purpose

Extend the MIPD TDM stack from **IV-only** to **oral multi-dose**: condition the engine-as-prior on a steady-state **oral** trough (and optionally a peak, or a measured F) to individualize an oral patient, and recommend a target-attainment oral dose. The IV path (`predict_tdm`, `recommend_dose`) frees a single renal-CL latent because for IV F≡1; for oral, **F re-enters as the dominant latent** (F is the engine's dominant structural error — DE-41/43→FLUX-1 — while CLint, R²~0.24, is already its best estimate).

This is a **measured-input individualization** feature, like the shipped measured-F routing — **not a 107-holdout headline lever** (DE-43: the fixed-weight meta damps engine moves). With zero measured data the path reduces to the a-priori prediction; the 2.731 headline is untouched.

## 2. The identifiability crux + the adaptive stance

At oral steady state, the textbook shorthand is `Css ≈ F·Dose/(CL·τ)` — a single trough is a **magnitude** observation. **This straight `F/CL = const` ridge is the low-extraction (E_h≪1) picture only** (M1). For the high-extraction DE-41/43 population this feature targets, the engine's `cl_int_h` saturates (extended ECM, `flux.py:320-347`) and the emergent `f_engine[s]` co-varies with the metabolic scale `s`, so the iso-trough indeterminate set is a **curved manifold**, not a line. The engine-faithful statement: a single trough pins the curved manifold `(f/f_engine[s])·conc(s; t_trough) = trough` in the grid's two free coordinates `(F_rescale, s)`. The second axis maps **non-proportionally** to physical hepatic CL and is conditional on the engine's fixed V/Kp (there is no volume latent — V error is absorbed into the CL attribution). The freed "F" marginal is the rescale factor `F_rescale = F/f_engine`, **not** physical `fa·Fg·Fh`. The SIR runs on the real grid (not the textbook formula), so this curvature is handled exactly; only the §2 prose simplifies it.

The ridge is broken only by **shape** — a second conc at a *different within-interval phase* adds tail-curvature (`Δlog c` over time) that tracks `ke=CL/V` independently of F — or by a **measured F** (anchors the F axis).

**Adaptive stance (v1):**
- **free-both (F + metabolic clint)** iff the observations contain shape/anchor info: **a `MeasuredF` is present**, **OR** the `MeasuredConc` set has **distinct within-interval phases** (the §6 wraparound-aware circular-phase test). Runs `sir_posterior_2d` over the full oral grid; reports both `f` and `cl_scale`.
- **free-F-only** otherwise (the common single-trough case, or repeated same-phase troughs): one effective latent (F) over the grid's **s=1 slice**. Same-phase troughs still tighten the F/exposure posterior — not wasted. Returns `cl_scale=None`.

The phase test is a **heuristic proxy** for tail-curvature informativeness, **not** an identifiability guarantee (M2): two flat-tail troughs at different phases carry little shape info. So "free-both" does not mean "identified by construction"; it means "enough shape was supplied that freeing `s` is defensible." When it isn't (a flat-tail pair), the `cl_scale` marginal can be prior-ridge-wide — handled by the **required** `|Δlog c|` gate + warning in §6 (Mx2), not left silent.

**Critical honesty point (I1 / Bx2 / Bx3).** Only the **concentration at the observed trough time, evaluated at the observed interval τ_observed**, is attribution-independent: free-F-only and free-both give the *same* posterior for *that* quantity from a single trough. Everything else depends on curve shape (`ke`), which a single same-phase trough does not identify:
- **Cmax and AUC are NOT attribution-independent.** `Cmax = (F/CL)·(peak/trough ratio)` and `AUC_τ = trough·(auc_grid[s]/conc_grid(s;t))`; both ratios move with `s`. free-both marginalizes over differently-shaped curves all passing through the trough → a materially wider, shifted band; free-F-only freezes the shape at the engine's s=1 estimate and therefore **understates exposure uncertainty**.
- **A trough at a *different* interval τ_k ≠ τ_observed is also NOT attribution-independent** — within-interval accumulation + decay over the new interval depend on `ke=f(s)`. This is exactly the sweep `recommend_dose` performs, so the "trough is the invariant fallback" guarantee holds *only at the observed interval* (Bx3).

Consequences for `recommend_dose` (which consumes `trough`/`cmax`/`auc24` targets): a single-trough recommendation is exposure-neutral **only** for a trough target at τ_observed (and dose-scaling of it, by LTI). For a Cmax/AUC target **or any candidate interval τ_k ≠ τ_observed**, the free-F-only recommendation is a frozen-shape (s=1) approximation; surface that caveat in `DoseRecommendation.warnings`, and prefer free-both (the honest wider band) when shape data is available. State this precisely in the docstrings.

## 3. New: `build_oral_cl_grid` (`oral_grid.py`)

A sibling of `build_renal_cl_grid` that populates the **existing `clgrid.CLGrid`** type. It returns a **3-tuple** — the grid plus two scalars that ride *alongside* it, so `CLGrid`/`CLGridForward`/`sir_posterior_2d`/`RenalCLGrid` types are all unchanged (no grid-type drift, Bx1):

```python
def build_oral_cl_grid(
    smiles: str, regimen, *, n_grid: int = 13, s_range: tuple[float, float] = (0.05, 20.0),
    renal_factor: float = 1.0, body_weight_kg: float | None = None,
    age_years: float | None = None, kp_method: str = "rodgers_rowland",
    dt_output: float = 0.1,
) -> tuple[CLGrid, bool, float | None]   # (grid, is_steady_state, t_half_h)
```

**Three engine solves per grid point `s`** (α / Finding #1) — `build_cl_grid` is single-dose (gives `f_engine`, no SS) and `build_renal_cl_grid` is multi-dose (gives SS, no `f_engine`); the oral grid composes both, scaling **metabolic** clint (`enzyme_affinity`, like `build_cl_grid`; renal/biliary fixed):
1. a **single-dose oral bolus** solve (`t_span=(0,24)`, `y0[admin]=dose`) → single-dose oral AUC + `compute_endpoints`;
2. a **single-dose IV reference** solve *internal to* `_engine_oral_bioavailability(compiled, params_s, drug_s, oral_auc, obs_node)` → `f_engine[s]`. **It is the single-dose oral AUC, not the SS-interval AUC, that feeds `f_engine`** (F is a per-dose first-pass property);
3. a **multi-dose `solve_regimen`** over the oral `regimen` → SS `conc`/`cmax`/`auc` extracted on the **final interval** `[last, last+τ]` (the renal-grid mask, route-agnostic), `t_total = last + max(τ, 24)` so the trough at `last+τ` is in-window.

`CLGridForward`'s `scale = F/f_engine` (clgrid.py:100) then exactly rescales the SS exposure: SS exposure ∝ F (linear), and mixing the single-dose `f_engine` with the multi-dose SS `cmax/auc/conc` is **correct** precisely because the per-dose F is the constant of proportionality (verified: `cmax_out = (F/f_engine[s])·cmax_ss[s]` maps the engine's emergent F to the patient's F at SS).

Reuses `grid._build_grid_engine(smiles, regimen.events[0].dose_mg, "oral", renal_factor, kp_method, body_weight_kg, age_years)`, `solve_regimen`, `_fill_nan_log_s`, `_nearest_finite_backfill` — verbatim except route, the metabolic-`s` scaling, and the added single-dose `f_engine` solve.

**`is_steady_state` + `t_half_h` (Bx1/Bx4).** At the grid point **nearest `s=1`** (`i1 = argmin|log(s_grid)|`, the a-priori central scale), before that point's `SimResult`s are discarded: (a) `is_steady_state = compute_steady_state_metrics(sim_multidose_s1, regimen, node=obs_node).is_steady_state` — passed `node=obs_node` (from `_resolve_observation_node`), **not** the `venous_blood` default, so SS is checked on the same curve the grid observes (matters for prodrug/active-species nodes); (b) `t_half_h = pk.t_half.mean` from the **single-dose** `compute_endpoints` at `s1` (`None` when `terminal_half_life` returns `None`). The grid does **not** raise on non-convergence; the callers turn `is_steady_state=False` into a warning.

## 4. Oral `predict_tdm` (route dispatch in `tdm.py`)

A new shared module **`src/sisyphus/mipd/_regimen.py`** (Bx5/Bx6) owns the route/uniformity/interval/shape helpers (so both `tdm.py` and `dosing.py` import one home, and `renal_grid._regimen_interval_h` is re-pointed to it):
- `_regimen_route(regimen) -> str`: all events at `DEFAULT_IV_NODE` → `"iv"`; all at `DEFAULT_ORAL_NODE` → `"oral"`; mixed/other → `ValueError`.
- `_require_uniform_regimen(regimen) -> None`: raises `ValueError` if `diff(event_times)` spread > ~1% of the median (non-uniform regimens are §10 out-of-scope — reject, don't mis-model).
- `_regimen_interval_h(regimen) -> float`: returns the **final** interval `events[-1].time_h - events[-2].time_h` (24.0 for a single dose). Under the uniformity gate this equals `events[1]-events[0]` for every admissible regimen, so re-pointing the renal helper is **value-identical** for uniform IV regimens → IV path & 2.731 untouched.
- `_distinct_phases(observations, tau) -> bool` + `_SHAPE_PHASE_TOL` (default `0.1·τ`): the §6 wraparound-aware shape test.

`predict_tdm` reads `_regimen_route(regimen)`:
- **IV** → the existing renal-CL path, IV **numerical** path unchanged (the dispatch is a pre-check; the IV branch is the current code — same renal-grid build, same `sir_posterior_renal`, identical default args after the M3 coalesce → the 2.731 guarantee). See M7 for the one cosmetic delta.
- **Oral** → call `_require_uniform_regimen`, then:
  1. detect shape (§2/§6): `MeasuredF` present OR `_distinct_phases(observations, τ)` where `τ = _regimen_interval_h(regimen)`.
  2. **free-both**: `grid, ss, _ = build_oral_cl_grid(n_grid, s_range)` → `sir_posterior_2d(FPrior(f_engine0, prior_cv), CLPrior(cv=1.0, s_min=grid.s_grid[0], s_max=grid.s_grid[-1]), CLGridForward(grid), observations)`. Returns `f`, `cl_scale`. Apply the Mx2 `|Δlog c|` warning if the supplied phases are flat-tail.
  3. **free-F-only** (B3 — corrected): `grid1, ss, _ = build_oral_cl_grid(n_grid=1, s_range=(1.0, 1.0))` → `sir_posterior_2d(FPrior(f_engine0, prior_cv), CLPrior(cv=1.0, s_min=1.0, s_max=1.0), CLGridForward(grid1), observations)`, then **explicitly `dataclasses.replace(post, cl_scale=None)`**.
     - **Why `sir_posterior_2d`, NOT the `APrioriPK`/`SIRAmortizer` F-only path** (the first go-over's instinct, overturned by verification): the oral free-F-only observation is a **steady-state `MeasuredConc` (a trough at time t)**, whose `log_likelihood` needs `state["conc_at"](t)` (clgrid.py:142). Only `CLGridForward` emits `conc_at`; `AnalyticForward` (the `SIRAmortizer` backend) returns only `{f, cmax, auc}` (core.py:60-64) and **cannot evaluate a `MeasuredConc`**.
     - **Why it is genuinely F-only and reports `cl_scale=None`**: with `s_grid=[1.0]`, `CLGridForward._interp` clips every draw's `s` to `[1.0,1.0]`, so `cmax/auc/conc_at` all come from the s=1 row → the likelihood depends only on `f`. Pinning `CLPrior(s_min=1.0, s_max=1.0)` clips the *sampled* `s` to exactly 1.0 so `n_eff` stays meaningful; the explicit `dataclasses.replace(post, cl_scale=None)` honors the "F-only ⇒ no clint latent" contract (`PosteriorPK.cl_scale` is `| None`, core.py:215) and satisfies the §7.3/§8 `cl_scale is None` assertions. (`CLGrid.conc_at` on a 1-point grid is valid — numpy's single-`xp` `np.interp` edge-clamps; `build_cl_grid` already exercises `n_grid=1`.)
  4. Always: `renal_scale=None`; **`meta_cmax=None`, `cmax_90ci=None`** — the conformal q90 is single-dose-Cmax-calibrated, invalid for SS Cmax (M5; same omission as IV, different reason). Direct the user to **`cmax.ci90`** (the F[+clint] parameter-uncertainty band) as the available SS interval, with a `warnings` note that no calibrated SS conformal exists (mirroring tdm.py:7-9). `warnings` also carries the covariate warnings and, when `ss is False`, a sub-SS flag.

`covariates` thread exactly as IV: `renal_factor()` (fixed renal CL), weight/age → `generate_physiology`. **`renal_prior_cv` (M3/Mx1):** flip **both** `predict_tdm` (tdm.py:28) and `recommend_dose` (dosing.py:232) signatures to `renal_prior_cv: float | None = None`; `recommend_dose` forwards the sentinel verbatim; coalesce `None→1.0` at the single IV-branch `RenalCLPrior` construction (byte-identical IV — and a default oral `recommend_dose` no longer forwards an explicit `1.0` that would trip a spurious warning). The oral branch **warns iff `renal_prior_cv is not None`**, carried on `PosteriorPK.warnings`. Oral otherwise uses the wide `FPrior`/`CLPrior` defaults (cv=1.0 — do **not** tighten, per DE-41/43).

## 5. Oral `recommend_dose` (route dispatch in `dosing.py`)

The reference is the MIPD **`mipd.dosing.DoseRecommendation`** (ε — *not* the legacy `regimen.dosing.DoseRecommendation`, a different class). The dose-solve **LTI core is route-agnostic and reused verbatim** — `_sample_m_intervals`/`_max_overlap_region`/`_center_m`/`_attainment`/winner all apply unchanged because oral SS exposure is exactly linear in dose. **"Verbatim" is scoped to the post-`q_ref` LTI algebra only** (B2); the orchestrator body is a *sibling* (it threads 1–2 latents `f` [+`cl_scale`] vs the IV path's single `renal_scale`).

What the oral sibling does:
- `_regimen_route` (oral requires all-oral events; mixed → `ValueError`) + `_require_uniform_regimen`.
- Infer the oral posterior once via the oral `predict_tdm` path → latent samples (`f` [+`cl_scale`]).
- An oral `_interval_reference` sibling: per candidate interval τ_k, `grid_k, _, _ = build_oral_cl_grid` at τ_k, then **build the forward once and read all three reference quantities off the same F-scaled state** (B1):
  ```python
  state  = CLGridForward(grid_k)(f, s)          # s ≡ ones(len(f)) when F-only
  trough = state["conc_at"](last + τ_k)         # carries F/f_engine
  cmax   = state["cmax"]                         # carries F/f_engine
  auc24  = state["auc"] * 24.0 / τ_k
  ```
  **Do NOT copy the IV `dosing.py:196` pattern `grid.conc_at(latents, last+τ)`** — `grid.conc_at` returns the *raw, unscaled* engine concentration (clgrid.py:56-76); the `F/f_engine` scale lives **only** inside `CLGridForward.__call__`'s `conc_at` closure (clgrid.py:100-101). The IV read is correct solely because IV has F≡1; transcribed to oral it would drop exactly the patient-individualization factor and mis-scale the trough against the (correctly-scaled) `cmax`/`auc`. `post.f` (absolute F from `FPrior`) is correctly consumed by a *different* candidate grid's forward because `f_engine` is regimen-independent, so `f_engine(grid_k) == f_engine(observed grid)` and `scale = post.f/f_engine(grid_k)` is consistent.
- Reads `post.f.samples` (and `post.cl_scale.samples` when free-both), **never `post.renal_scale`** (None for oral; would `AttributeError` at the IV body's `dosing.py:272` read), and constructs `DoseRecommendation(..., renal_scale=None, f=post.f, cl_scale=post.cl_scale)`.
- Candidate regimens via `DosingRegimen.oral_repeated(dose_mg, interval_h, n_doses)` (no `duration_h`).
- **Half-life-aware `n_doses` (I2/Bx4):** the IV heuristic `n_doses=max(2, round(cur_last/τ)+1)` (dosing.py:292) under-doses long-τ candidates for long-t½ drugs → sub-SS candidate grids bias the recommended dose high. Size by the grid's `t_half_h` (the 3rd tuple element, captured at s≈1): `n_doses = max(N_min, ceil(k · t_half_h / τ_k))` with **`N_min = 5`, `k = 5`**, `τ_k` in hours from the candidate-intervals tuple. **When `t_half_h is None`, fall back to the IV heuristic.** Per candidate, attach the §3 `is_steady_state=False` sub-SS `warning` if it still hasn't converged. (Route-independent — the IV path has the same latent gap and should get the same guard so the siblings stay consistent.)
- **Shape caveat (Bx3):** for any candidate `τ_k ≠ τ_observed`, or a Cmax/AUC target under free-F-only, append the s=1 shape-approximation caveat to `DoseRecommendation.warnings` (the trough invariance holds only at τ_observed).

## 6. Error handling

`ValueError` on: mixed-route regimen (`_regimen_route`); **non-uniform regimen** (`_require_uniform_regimen`, spread > ~1% of median — both entrypoints, both routes; only non-uniform inputs are affected, which no existing IV test exercises → IV numerical path unchanged); an oral call with a non-oral, non-IV node; `build_oral_cl_grid` all-grid-points-fail (propagates the existing `build_cl_grid` `ValueError`). **Trough past horizon** → `CLGrid.conc_at` raises (the §3 horizon is sized to prevent it). **Steady-state non-convergence** → structured `warning` (§3/§5), never raised.

**Shape detection — wraparound-aware (Bx6).** `MeasuredConc.t` is absolute; phase `φ(t) = t mod τ` with `τ = _regimen_interval_h(regimen)` (single regimen-level τ, valid because the uniformity gate rejects non-uniform). The set has "distinct phases" iff the **maximum pairwise circular distance** `max_{i,j} min(|φ_i−φ_j|, τ−|φ_i−φ_j|) > _SHAPE_PHASE_TOL`. The circular distance makes a `0`/`τ` pair (both trough-like) → `d≈0` → **F-only** (so §7.3 holds; there is no "wraparound runs free-both" case — that prose is struck). The one genuinely-unhandled heuristic gap is a **flat-tail** distinct-phase pair (real phase separation, little curvature): free-both still runs (exposure stays honest), but the `|Δlog c|`-at-`s=1` gate (Mx2) fires a `PosteriorPK.warning` — *"cl_scale marginal may be prior-ridge-wide; supplied phases do not identify the metabolic clint axis."* (The struck `n_eff` mitigation does not catch this — `n_eff` flags weight-degeneracy, not a wide-but-supported marginal.)

## 7. Testing (TDD; load-bearing first)

Directional tests anchor to the engine's own s=1 prediction (no magic numbers) — stack-independent per the `df4492c` discipline.

1. **Oral LTI exactness** (load-bearing): `build_oral_cl_grid` at `2·D` == the `D` grid ×2 on SS cmax/auc.
2. **Grid faithfulness**: at `s=1`, the SS trough matches a direct `solve_regimen` oral SS solve; `f_engine[s≈1]` matches `build_cl_grid`'s `f_engine` at `s=1` (regimen-independence of F). Include an explicit **`n_grid=1`** forward/grid-faithfulness case and the all-fail-at-`s=1` → `ValueError`.
3. **Adaptive routing**: one `MeasuredConc` → `cl_scale is None`; two distinct-phase concs → `cl_scale is not None`; two same-phase troughs → `cl_scale is None`; a `0`/`τ` wraparound pair → `cl_scale is None` (circular-distance filter); one conc + `MeasuredF` → free-both.
4. **B1 — oral trough carries the F-scale** (load-bearing): the oral `_interval_reference` trough at a posterior with `F ≠ f_engine` equals `(F/f_engine) × grid.conc_at(s, last+τ)`; anchor directionally to the engine's s=1 prediction.
5. **I1 — attribution honesty (Bx2)**: on one real engine oral grid with a single trough, the **trough-time concentration** posterior median + ci90 agree across free-F-only and free-both within MC tolerance (the one attribution-independent quantity, engine-anchored); for **both Cmax and AUC** assert only that free-both's band is wider/shifted (the documented direction), **NOT** equality.
6. **Inference direction**: an oral trough below the engine's s=1 prediction shifts the **F** posterior down.
7. **B3 — free-F-only is clean**: returns `cl_scale is None`; the F posterior is non-degenerate; the internal pre-replace `cl_scale` samples are all ≈1.0.
8. **Route-dispatch invariance (B2/M7)**: an IV regimen through `predict_tdm`/`recommend_dose` is **numerically bit-identical** to today (existing IV tests pass unchanged; `test_recommend_renal_scale_shifts_with_observation` still passes; explicit IV-still-works assertion).
9. **B2 — oral `DoseRecommendation` contract**: oral recommendation has `renal_scale is None` and `f is not None`; IV unchanged.
10. **I2/Bx1 — sub-SS guard**: a long-t½ oral (and IV) 2-dose long-interval regimen yields the sub-SS `warning`, computed at `obs_node` and surfaced on `warnings`.
11. **M3/Mx1 — renal_prior_cv warning**: a **default** oral `predict_tdm`/`recommend_dose` emits **NO** `renal_prior_cv` warning; set-non-`None` emits it.
12. **M4 — non-uniform rejection**: a non-uniform oral (and IV) regimen → `ValueError`.
13. **M6 — auc24 factor**: at τ=8 `auc24 == auc·3`, τ=12 `auc24 == auc·2` (oral; same one-line assertion added to the IV `test_interval_reference_returns_quantities_at_reference_dose`). Do **not** assert τ-invariance of `auc24`.
14. **Bx3 — τ-sweep shape caveat**: single-trough oral + trough target + a candidate `τ_k ≠ τ_observed` → free-both gives a wider/shifted τ_k-trough band than free-F-only, and the s=1 shape-approximation `warning` is present.
15. **Mx2 — flat-tail warning**: a distinct-phase but flat-tail conc pair routes to free-both **and** emits the "cl_scale may be prior-ridge-wide" warning.
16. **Oral recommend_dose**: feasible-window hit, longer-interval tie-break.
17. **Mixed-route regimen → `ValueError`.**
18. **No SS conformal (M5/Mx3)**: oral `predict_tdm` returns `meta_cmax is None`, `cmax_90ci is None`, and `cmax` is a `Posterior` whose `samples` is non-empty (so `ci90` returns a finite tuple); two-sidedness is **not** asserted (the `FPrior` (0,1] clip can legitimately make the upper bound one-sided near F→1).

## 8. Invariants

`engine/` & `predict()` untouched (reuse `solve_regimen`, `grid._build_grid_engine`, `_engine_oral_bioavailability`, `compute_steady_state_metrics`, `terminal_half_life` — import/call only); **IV numerical path bit-identical** → holdout & 2.731 headline untouched; identity-blind; all PK quantities are `Posterior`/`Distribution`; measured-input feature (not benchmarked on the 107-holdout).

**Contract changes (all additive):**
- `PosteriorPK` (core.py): **unchanged** — `f`/`cl_scale`/`renal_scale`/`meta_cmax`/`cmax_90ci`/`warnings` already `| None` (core.py:211-223); oral's `None` uses are additive.
- `mipd.dosing.DoseRecommendation` (B2): relax `renal_scale: Posterior` → `renal_scale: Posterior | None = None`; add `f: Posterior | None = None`, `cl_scale: Posterior | None = None`. **Reorder** so the three defaulted fields trail the non-default fields (`n_eff`, `warnings` follow `renal_scale` today); both construction sites (dosing.py:337) are keyword-only and no test constructs it positionally, so IV construction stays equivalent. (The legacy `regimen.dosing.DoseRecommendation` is a different class and is untouched.)
- **No new grid types** (Bx1): the SS bool and `t_half_h` ride in `build_oral_cl_grid`'s return *tuple*, not inside the frozen `CLGrid`; `CLGrid`/`CLGridForward`/`RenalCLGrid` are unchanged.

**M7 — scope of "byte-for-byte".** Scoped to the IV *numerical* path (renal-grid build + `sir_posterior_renal`, identical default args after the M3 None→1.0 coalesce). One cosmetic delta: today's IV guard (tdm.py:50) emits "IV regimens only" for both oral and mixed regimens; with `_regimen_route`, a mixed regimen's error **text** changes (same `ValueError`). The two existing oral-rejection tests feed pure-oral regimens and must be **updated** (oral is now supported).

## 9. Scope (decomposes into ~6 implementation tasks)

`_regimen.py` shared helpers (route/uniformity/interval/shape) → `oral_grid.build_oral_cl_grid` (3-solve, tuple return, SS+t_half) → oral `predict_tdm` free-F-only (B3) → adaptive free-both (shape detect + Mx2 gate) → `DoseRecommendation` relax (B2) + oral `_interval_reference` (B1) + oral `recommend_dose` (I1/I2/Bx3) → exports/integration. Bundled per the brainstorming decision.

## 10. Out of scope (future)
- Non-steady-state oral titration / loading-dose schedules.
- Absorption-latent (`ka`/lag) freeing — oral frees F + metabolic clint only.
- SI (µmol/L) units; **non-uniform oral regimens** (explicitly *rejected* with `ValueError`, §6).
- A conditioned-case SS conformal recalibration (single-dose conformal omitted; users get `cmax.ci90` as the parameter-uncertainty fallback).
- A flat-tail `|Δlog c|` *hard route-to-F-only* (v1 runs free-both + warns; auto-downgrade is a future refinement).
