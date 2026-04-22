# V3 IV-Cmax Observation Methodology — Design Spec

**Date:** 2026-04-22
**Status:** Design only — no code changes
**Author:** Hypatia (session)
**Triggered by:** ECM generalization test Mode C (commit `4fb6d38`)

---

## Investigation Log

### Step 1 — `src/sisyphus/engine/uncertainty.py`

`propagate_fast` (scipy backend, lines 265–304) iterates MC samples and calls `solve_mc()` per sample, collecting `(cmax, tmax, auc, success)` tuples. The Cmax value is extracted as `max(conc)` over the full time series returned by the ODE solver. The observation node defaults to `"venous_blood"`. No time filter or delay is applied before the max.

### Step 2 — `src/sisyphus/engine/solver.py`

`solve()` (full solve, line 16) creates a `t_eval = np.linspace(t_span[0], t_span[1], 500)`. The first point is always `t=0.0`. `solve_mc()` (fast solve, line 88) does NOT pass a `t_eval` — the LSODA adaptive integrator chooses its own output grid. In both cases, `cmax = float(np.max(conc))` at line 140, with no time conditioning.

For IV bolus, `y0[venous_blood_idx] = dose_mg` and all other states = 0. At `t=0`, the concentration is `dose_mg / V_venous_blood = dose_mg / 3.7`. This is the global maximum over the time series for any monotonically-declining distribution process.

### Step 3 — `data/physiology/reference_man.yaml`

Node volume declarations:

```yaml
- name: venous_blood
  type: blood_pool
  volume: 3.7          # bare float -> Distribution(mean=3.7, cv=0.0)

- name: arterial_blood
  type: blood_pool
  volume: 1.5          # bare float -> Distribution(mean=1.5, cv=0.0)
```

`global_params` has `cardiac_output: {mean: 390.0, cv: 0.10}`. All node volumes are bare scalars — parsed by `builder.py:_parse_distribution()` (line 249) as `Distribution(mean=float, cv=0.0)`. No other graph parameter carries non-zero CV except cardiac output.

**Confirmed:** `V_venous_blood = Distribution(3.7, cv=0.0)`. Every MC sample draws an identical value of 3.7 L.

### Step 4 — `data/reference/clinical_pk.json` + `data/reference/holdout.json`

All 331 drugs in `clinical_pk.json` are `"route": "oral"` (326) or `"route": "unknown"` (5). Zero IV entries.

All 107 holdout drugs are confirmed oral (manual inspection of `holdout.json` against `clinical_pk.json`). The routes confirmed to be oral for all 107 drugs. **No IV drugs exist in the 107-holdout set.**

Impact conclusion: any change to IV-Cmax handling has zero direct exposure to Meta AAFE 2.695.

### Step 5 — `src/sisyphus/pipeline/predict.py`

The production pipeline (line 121) calls `solve(compiled, params, y0, t_span=(0, 24))` for the single deterministic run, then `compute_endpoints(sim_result)`. `compute_endpoints` (`pk/endpoints.py` line 27) applies `cmax = float(np.max(conc))` — same unfiltered max as MC. The MC path (line 141) calls `ue.propagate_fast(compiled, graph, drug, n_samples=n_mc_samples)` which reaches `solve_mc()` with the same unfiltered max.

### Step 6 — `scripts/run_engine_benchmark.py`

The holdout benchmark (line 49) calls `predict(ref.smiles, ref.dose_mg, ref.route)` where all `ref.route` values are oral. The script does not use MC (`n_mc_samples` defaults to 0). PI degeneracy is therefore invisible in the benchmark — the issue only manifests when MC is invoked for IV drugs.

---

## §1. Problem Statement

### Observed Data

From `data/validation/oatp_generalization_result.json` (commit `4fb6d38`), ECM generalization test N=2:

| Drug | Observed Cmax (mg/L) | Engine Point Est (mg/L) | PI 90% Low | PI 90% High | FE | Pass |
|---|---|---|---|---|---|---|
| Glimepiride | 0.243 | **0.27027** | **0.27027** | **0.27027** | 1.11× | fail |
| Valsartan | 4.02 | **5.40541** | **5.40541** | **5.40541** | 1.35× | fail |

Zero-width 90% PI: all 1000 MC samples returned the identical Cmax value for both drugs.

### Mathematical Root Cause

**Initial conditions for IV bolus:**

```
y0[venous_blood] = dose_mg
y0[all_other_nodes] = 0
```

At `t=0`, the venous blood concentration is:

```
C_venous(t=0) = y0[venous_blood] / V_venous_blood = dose_mg / 3.7
```

Numeric verification against result JSON:

- Glimepiride: `1.0 / 3.7 = 0.27027027…` — matches `pi_90_low_mg_l` exactly to 15 decimal places.
- Valsartan: `20.0 / 3.7 = 5.40540540…` — matches `pi_90_low_mg_l` exactly to 15 decimal places.

**Why MC variance cannot reach this value:**

`V_venous_blood` is declared as a bare float (3.7) in `reference_man.yaml`. `builder.py:_parse_distribution()` converts this to `Distribution(mean=3.7, cv=0.0)`. The `BodyGraph.sample()` method (line 138 in `body.py`) draws from this distribution per MC sample — but with `cv=0.0`, every draw returns exactly 3.7.

Therefore:

```
C_venous(t=0) = dose_mg / sample(V_venous_blood) = dose_mg / 3.7   (identical for all 1000 samples)
```

`solve_mc()` returns `cmax = float(np.max(conc))`. For IV bolus, the venous concentration monotonically decreases from `t=0` as drug distributes to tissues. The global maximum over the time series is always the `t=0` value. All parameters that carry genuine uncertainty — cardiac output (CV=0.10), drug fup, drug Kp, enzyme affinity — influence concentrations only at `t > 0`. They are downstream of the maximum.

**Why this does not affect oral drugs:**

For oral dosing (`administration_node = "stomach_lumen"`), drug is deposited in the gut at `t=0`. Absorption and gut transit require time. Venous blood concentration builds from zero, reaches a peak at `Tmax ≈ 1–4h`, then declines. The maximum occurs post-distribution, where cardiac output, fup, Kp, and Peff all affect the concentration. All uncertainty sources propagate. This is why the 107-holdout benchmark (all oral) produces valid, non-degenerate PIs.

### Why the Point Estimates Are Close to Observed

The engine point estimates (1.11× for glimepiride, 1.35× for valsartan) are **not evidence of ECM failure.** They are a consequence of semantic mismatch:

- **Engine Cmax:** instantaneous concentration at `t=0` in venous blood, before any distribution occurs. Equivalent to `dose / V_venous_blood_only`.
- **Clinical Cmax:** concentration in blood at the first post-dose draw, typically 2–5 min after injection. By this time, the bolus has partially mixed through the central blood pool (venous + arterial, connected through the pulmonary circuit).

The PBPK pulmonary transit time is `V_lung / Q_CO = 0.50 L / 390 L/h ≈ 0.001h ≈ 4.6 sec`. After one recirculation (~1 min), arterial and venous blood concentrations are nearly equalized. The effective central compartment volume at 5 min post-dose is approximately `V_venous + V_arterial = 3.7 + 1.5 = 5.2 L`.

Implied effective Vc from clinical observations:

- Glimepiride: `1.0 / 0.243 = 4.12 L` (between 3.7 and 5.2 L)
- Valsartan: `20.0 / 4.02 = 4.97 L` (between 3.7 and 5.2 L)

Both values are consistent with rapid venous-arterial equilibration within the sampling interval. The 1.11–1.35× FE is a **systematic semantic bias** of using `V_venous` as the central compartment for IV drug observation, not an ECM calibration error.

---

## §2. Clinical Semantic Analysis

### What Clinical Labs Measure

Clinical IV bolus pharmacokinetic Cmax is defined as the highest measured plasma concentration following the dose. For standard IV bolus studies:

- First blood draw: 2–5 minutes post-injection (institutional protocol dependent).
- Draw site: peripheral venous (arm) or central venous catheter.
- At 5 min post-bolus: the bolus has completed one full systemic circulation (~1 min for CO=6.5 L/min through blood volume 5.2 L). Venous and arterial concentrations have partially but not completely equilibrated with the distribution phase.

The reported Cmax from a typical IV PK study is therefore the **post-first-circulation** concentration, not the instantaneous post-injection concentration.

### What the Engine Measures

The engine observes `max(concentrations["venous_blood"])` over `t ∈ [0, 24h]`. For IV bolus:

- `t=0`: `C_venous = dose / 3.7 L`. This is the **pre-distribution** concentration — drug has not moved anywhere.
- `t > 0`: drug distributes to arterial blood (through lung), then to organs, driven by blood flow, Kp, and fup. `C_venous` declines monotonically.
- `max(C_venous)` = the `t=0` value = `dose / 3.7`, always.

### Ratio Quantification

For a two-compartment approximation with valsartan-like distribution kinetics (`t½_alpha ≈ 30 min`), the venous concentration at 5 minutes post-bolus is approximately 85–90% of the `t=0` value. The engine-to-clinical ratio for the simple `V_venous`-only observation is:

```
ratio = C_engine(t=0) / C_clinical(t=5min) = V_eff_central / V_venous ≈ 5.2 / 3.7 ≈ 1.40
```

This range (1.1–1.4×) brackets both observed engine FEs (1.11× and 1.35×), confirming the semantic interpretation.

**Conclusion:** The engine-to-clinical discrepancy for IV bolus Cmax is a structural semantic mismatch, not a model calibration failure. ECM generalization cannot be formally assessed until this mismatch is resolved.

---

## §3. Design Alternatives

### Alternative A: Time-Windowed Max (`t ≥ t_min`)

**Mechanism:** Replace `max(conc)` with `max(conc[t >= t_min])` where `t_min = 0.083h` (5 minutes). This skips the pre-distribution spike at `t=0` and defines Cmax as the maximum concentration over the post-first-circulation interval.

**Code change scope:**

- `src/sisyphus/engine/solver.py` — `solve_mc()`: replace `cmax = float(np.max(conc))` with `mask = sol.t >= t_min; cmax = float(np.max(conc[mask])) if mask.any() else 0.0`. Also update `tmax` index lookup. ~4 lines.
- `src/sisyphus/pk/endpoints.py` — `compute_endpoints()`: add optional `t_min_h=0.0` parameter. Route-conditional default (`t_min_h=0.083 if route=="iv" else 0.0`). ~5 lines.
- `src/sisyphus/engine/uncertainty.py` — `propagate_fast()`: pass `t_min` to `solve_mc`. ~1 line.
- Total: ~10 lines across 3 files.

**Compatibility with 107-holdout test:** Zero impact. All 107 holdout drugs are oral. Oral Cmax occurs at Tmax ≈ 1–4h, far above the 5-minute threshold. `max(conc[t >= 0.083])` is identical to `max(conc)` for any drug with monotonically-rising venous concentration pre-Tmax (which is true for all oral drugs absorbed over hours).

**Additional compatibility check:** The `solve_mc()` adaptive grid (no `t_eval`) includes `t=0` as the first output point. For IV bolus with LSODA, the adaptive steps begin at very small intervals (< 0.001h). At threshold 0.083h, hundreds of solver steps will already have occurred, giving a well-resolved maximum.

For `solve()` (full MC path), `t_eval = linspace(0, 24, 500)` produces `t[1] = 0.048h` (first non-zero point). The first point satisfying `t >= 0.083h` is `t[2] = 0.096h = 5.77 min`. This is within the clinically-standard 5-minute sampling window.

**Variance propagation after fix:** At `t = 0.083h`, venous blood concentration is:

```
C_venous(0.083h) = f(cardiac_output, drug_fup, drug_Kp, enzyme_affinity, organ_flows, ...)
```

All these parameters carry non-zero CV (cardiac_output: CV=0.10; drug fup, Kp, enzyme_affinity: CV from XGBoost prediction uncertainty). The distribution machinery at `t > 0` activates all uncertainty sources. MC variance is fully propagated.

**Pros:**
- Minimal code change (10 lines).
- Clinically motivated: matches the 5-minute sampling convention.
- Zero regression risk for oral drugs (proven by construction).
- No new parameters, no YAML changes, no physiology changes.
- Variance sources already present (cardiac output CV=0.10, drug distribution parameters).

**Cons:**
- `t_min = 0.083h` is a convention, not a derived parameter. Different studies use 2–10 minute sampling windows.
- For extremely fast-distributing drugs (e.g., small MW, high logP, t½_alpha < 2 min), the windowed Cmax at 5 min may already be well below the `t=0` spike. The FE improvement is drug-dependent.
- Does not address the underlying semantic mismatch (engine still measuring venous, not central compartment).
- `solve_mc()` with no `t_eval` may not have output points densely sampled at exactly `t_min`. The max near `t_min` is solver-step-density-dependent. Recommendation: add a `t_min` to `t_eval` in `solve()` to guarantee the threshold is exactly captured.

**Verdict: Primary recommendation** — see §5.

---

### Alternative B: Alternative Observation Node (`arterial_blood`)

**Mechanism:** Change the observation node from `venous_blood` to `arterial_blood`. For IV bolus, `y0[arterial_blood] = 0` initially. Arterial blood receives drug only after transit through the lung. The arterial Cmax occurs at `t > 0`, enabling variance propagation.

**Code change scope:**

- `src/sisyphus/engine/uncertainty.py` — `propagate_fast()`: change default `observation_node="venous_blood"` to `"arterial_blood"` when `route == "iv"`. Requires passing `route` into the function (currently not a parameter). ~5 lines.
- `src/sisyphus/pk/endpoints.py` — `compute_endpoints()`: add `observation_node` parameter (currently defaults to `"venous_blood"`). ~2 lines.
- `src/sisyphus/pipeline/predict.py` — route-conditional node selection. ~3 lines.
- Total: ~10 lines across 3 files.

**Compatibility with 107-holdout test:** Zero direct impact (all oral, all use venous_blood). Requires care in the pipeline to avoid changing the node for oral drugs.

**Semantic analysis:**

In the PBPK model, `arterial_blood` starts at zero for IV bolus. Drug reaches arterial blood after one pulmonary transit (~4.6 sec). By `t = 0.1h` (6 min), arterial and venous concentrations are nearly equal. For drugs with significant first-pass effects or slow distribution, the arterial peak at a short lag time is clinically meaningful.

However, clinical Cmax is measured from venous blood draws. Arterial blood concentrations differ from venous during the distribution phase (arterial is lower than venous immediately after bolus, then they equilibrate). The direction of bias from switching to arterial is drug-specific and may improve or worsen accuracy.

**Additional concern:** `arterial_blood` in the model has volume 1.5 L (bare float, CV=0). The peak arterial concentration at first circulation is approximately `dose / (V_venous + V_arterial) ≈ dose / 5.2`. This is a better approximation to clinical Cmax than `dose / 3.7`, but it is still deterministic if volumes carry no CV. The PI degeneracy issue would only be partially resolved — the first arterial peak is also largely deterministic given CV=0 volumes.

**Variance propagation after fix:** Unlike Alternative A, the arterial peak concentration depends on cardiac output (CV=0.10) for its timing and on fup/Kp for the distribution kinetics. However, the cardiac output primarily affects timing not magnitude (the total amount in system is conserved). For a slow-distributing drug, the arterial peak may be meaningfully variance-propagated; for a fast-distributing drug, it may still be near-deterministic.

**Pros:**
- No threshold parameter required.
- May better represent drugs where arterial sampling is clinically used.

**Cons:**
- Clinical measurement is venous (peripheral), not arterial — semantic mismatch persists.
- PI degeneracy may remain (arterial_blood volume also CV=0; fast equilibration produces near-deterministic arterial peak).
- Requires route-aware node selection logic in pipeline.
- May directionally worsen point estimates for slow-distributing drugs where arterial < venous during distribution phase.

**Verdict: Not recommended as primary fix.** The semantic mismatch is not resolved, and partial variance improvement is not guaranteed.

---

### Alternative C: Physiological CV on `V_venous_blood`

**Mechanism:** Add `cv: 0.15` to the `venous_blood` volume declaration in `reference_man.yaml`:

```yaml
- name: venous_blood
  type: blood_pool
  volume: {mean: 3.7, cv: 0.15}
```

**Physiological basis:** ICRP Publication 89 (2002) and population PK literature report inter-individual variability in total blood volume of CV ≈ 10–20%, with 15% (SD ~0.55 L for 3.7 L) being a reasonable consensus estimate. Venous blood is approximately 70% of total blood volume; its variability scales similarly.

**Code change scope:**

- `data/physiology/reference_man.yaml` — 1 line change (bare float `3.7` → `{mean: 3.7, cv: 0.15}`).
- No code changes required.

**Compatibility with 107-holdout test:** This change affects ALL simulations — oral and IV. The oral Cmax is determined at `Tmax >> 0`, where venous blood volume participates in distribution equilibrium. Adding CV=0.15 on `V_venous_blood` adds a modest noise source to all oral predictions.

Quantitative impact (oral): The venous blood pool acts as the observation compartment for oral Cmax. A CV=0.15 on its volume adds CV ≈ 0.10–0.15 to the Cmax distribution (attenuated by the fact that venous volume is one of many parameters). The effect on point estimates is near-zero (sampled volumes are unbiased around mean 3.7). The effect on PI width is modest.

**Effect on IV degeneracy:** With CV=0.15 on `V_venous_blood`:

```
C_venous(t=0) = dose / sample(V_venous_blood)   ~ LognNormal(mean=dose/3.7, cv≈0.15)
```

For valsartan (dose=20 mg): simulated 5th–95th percentile of `C_t0` = `[4.28, 7.02] mg/L`. The observed 4.02 mg/L falls **just below** the 5th percentile (PI does not contain the observation). Glimepiride (dose=1 mg): `[0.214, 0.351] mg/L`. The observed 0.243 mg/L **is** contained.

Critical flaw: This approach makes the PI non-degenerate but **centers it on the wrong value** (`dose / V_venous`). The point estimate is still the pre-distribution concentration. The PI containment is achieved by making the PI wide enough to extend down to the clinical observation, not by correcting the semantic mismatch. For valsartan, even with CV=0.15, the PI fails to contain the observation.

Furthermore, this change adds spurious variability to 107 oral holdout drugs. The CV=0.15 noise source on venous blood volume has no clinical precedent in the Cmax of oral drugs where the liver, gut wall, and absorption rate are the dominant variance sources.

**Pros:**
- Physiologically real parameter (blood volume does vary inter-individually).
- Minimal implementation (1 YAML line).
- Fixes PI degeneracy without any code changes.

**Cons:**
- Does not address semantic mismatch — point estimate remains `dose / V_venous`.
- For valsartan, PI still fails to contain observed value.
- Adds noise to 107 oral holdout drugs (regression risk).
- Philosophically wrong: adding variance to V_venous to produce the appearance of uncertainty propagation, when the real issue is observation timing.

**Verdict: Not recommended as primary fix.** Addresses symptom not cause; creates regression risk.

---

### Alternative D: Vc-Weighted Central Compartment Observation

**Mechanism:** For IV drugs, observe Cmax from a weighted average of central compartment concentrations:

```
C_central(t) = (A_venous + A_arterial + A_lung * f_blood_lung) / V_central_blood
```

where `V_central_blood = V_venous + V_arterial + V_lung_blood_fraction ≈ 3.7 + 1.5 + 0.45 = 5.65 L`.

This is the classical "central compartment" of two-compartment PK models: the blood pools and rapidly-equilibrating tissues (lung, heart) that are sampled during the initial distribution phase.

**Code change scope:**

- `src/sisyphus/engine/solver.py` — `solve_mc()`: add weighted sum over multiple nodes. Requires knowing which nodes constitute the "central compartment" — a drug- or model-specific definition.
- `src/sisyphus/pipeline/predict.py` — must define and pass the central compartment node list.
- `src/sisyphus/engine/uncertainty.py` — `propagate_fast()`: pass node list.
- `data/physiology/reference_man.yaml` or a config file — define "central compartment nodes".
- Total: ~25 lines across 4 files + a new config entry.

**Semantic analysis:**

The central compartment at `t=0` for IV bolus has:
```
A_central(t=0) = dose_mg (in venous_blood)
A_arterial(t=0) = 0
A_lung(t=0) = 0
```

Therefore `C_central(t=0) = dose_mg / V_central = dose_mg / 5.65 ≈ 0.177× (dose/3.7)`. For valsartan: `20/5.65 = 3.54 mg/L` vs observed 4.02 — now under-predicts. The weights and compartment membership would require calibration.

At `t > 0` (after one recirculation), the central compartment concentrations equilibrate. The weighting becomes less critical. But the max-over-time still picks `t=0` as the global maximum since the weighted sum at t=0 includes all the dose in venous and zero elsewhere, then drops.

**Critical observation:** Unless V_central_blood has a non-zero CV, the same degeneracy problem recurs at `t=0`. The weighted sum at `t=0` is `dose_mg / V_central_blood = dose_mg / 5.65` — deterministic if all component volumes are CV=0.

**Pros:**
- Physiologically motivated — matches 2-comp model "Vc".
- May improve point estimate accuracy if Vc is calibrated correctly.

**Cons:**
- Drug-specific: different drugs have different central compartments (e.g., highly lipophilic drugs distribute fast to lung tissue, effectively increasing Vc).
- Still degenerate at t=0 unless CV is added to volumes.
- Requires new "central compartment" definition infrastructure.
- ~25 lines of code + new configuration.
- Breaks engine identity-blind invariant (must enumerate specific node names for central compartment).

**Verdict: Not recommended.** Adds complexity without fully resolving the root cause; breaks architecture invariants.

---

### Alternative E: Route-Aware Minimum Observation Time + Adaptive t_eval

**Mechanism:** A hybrid of Alternative A plus architectural improvement: inject `t_min` points into the solver's `t_eval` to guarantee exact threshold resolution, and make the minimum time route-aware rather than hardcoded.

For IV bolus: `t_eval = np.concatenate([[0.0, t_min], np.linspace(t_min, 24, 498)])` where `t_min = 0.083h`. Cmax is then `max(conc[1:])` (skip index 0 = t=0 exactly).

For oral: `t_eval = np.linspace(0, 24, 500)` (unchanged).

**Code change scope:**

- `src/sisyphus/engine/solver.py` — `solve()`: modify `t_eval` construction to accept and embed `t_min`. ~8 lines.
- `src/sisyphus/engine/solver.py` — `solve_mc()`: add `t_eval` anchor at `t_min` (currently uses no `t_eval`). Change: add `dense_output=True` or explicit `t_eval`. ~5 lines.
- `src/sisyphus/pipeline/predict.py` — pass `route` to solver calls. ~3 lines.
- Total: ~16 lines across 2 files.

**Additional property:** By ensuring `t_min = 0.083h` appears in `t_eval`, the Cmax value at the threshold is exactly evaluated (not interpolated from nearby steps). This improves precision of the windowed max for `solve_mc()`.

**Pros:**
- Exact threshold enforcement (no step-density ambiguity from adaptive grid).
- Route-aware: explicitly parameterized.
- Clinically grounded.
- Zero regression for oral drugs (t_eval is unchanged for oral routes).

**Cons:**
- Slightly more complex than pure Alternative A (16 vs 10 lines).
- `solve_mc()` currently has no `t_eval` for speed (by design). Adding any `t_eval` may increase memory and time. Mitigation: only enforce 2 early anchor points, not full 500-point grid.

**Verdict: Recommended as the V3 implementation** — see §5 for full recommendation.

---

## §4. 107-Holdout Impact Analysis

**Data source:** `data/reference/holdout.json` (107 drugs), `data/reference/clinical_pk.json` (331 drugs with route metadata).

**Route distribution in full reference set:**
- Oral: 326 / 331
- Unknown: 5 / 331
- Intravenous: 0 / 331

**Route distribution in 107-holdout:**
- Oral: 107 / 107 (all)
- Intravenous: 0 / 107

**Conclusion:** The 107-holdout set is entirely oral. Any change to IV-Cmax observation methodology — whether time-windowed max, observation node, or volume CV — has zero direct impact on the Meta AAFE 2.695 headline metric. The ECM generalization test and any future IV validation work constitute a parallel IV-specific benchmark track.

**Indirect impact risk:** Alternative C (adding CV to `V_venous_blood`) is the only alternative that modifies physiology parameters affecting oral simulations. Alternatives A and E modify only the max-extraction logic and are route-conditioned to IV. Alternatives B and D require route-conditional observation logic.

**Safe change classification:**

| Alternative | Oral holdout risk | Regression test required |
|---|---|---|
| A (time-windowed max, route-conditional) | None | No |
| B (arterial_blood node, route-conditional) | None | Yes (verify oral unchanged) |
| C (V_venous CV) | Low but real | Yes (full 107-holdout re-run) |
| D (central compartment) | None | Yes (verify oral unchanged) |
| E (adaptive t_eval, route-conditional) | None | No |

---

## §5. Recommended Path Forward

### Primary Recommendation: Alternative E (route-aware adaptive t_eval) + Alternative A fallback

**V3 methodology:** Time-windowed maximum with guaranteed threshold point in `t_eval`, implemented as a route-conditional modification.

**Implementation summary:**

1. Add `t_min_h` parameter to `solve()` and `solve_mc()` (default `0.0` = backward-compatible).
2. When `t_min_h > 0`: insert `t_min_h` as an explicit anchor in `t_eval` for `solve()`. For `solve_mc()`, insert `t_min_h` as the first non-zero output point.
3. Cmax extraction: `max(conc[sol.t >= t_min_h])` — excludes `t=0` point.
4. In `pipeline/predict.py`: pass `t_min_h = 0.083` when `route == "iv"`, `t_min_h = 0.0` otherwise.
5. `t_min = 0.083h` (5 minutes) declared as a named constant `_IV_CMAX_DELAY_H = 5.0/60.0` in `solver.py`.

**Why 0.083h (5 min):**

- Clinically motivated: 5 minutes is the standard earliest blood draw in IV PK studies.
- Within the standard PBPK accuracy range: for `t½_alpha ≈ 15–30 min` (most drugs), `C(5min)/C(0) ≈ 0.75–0.90`, which matches the observed engine-to-clinical ratio (1.11–1.35×).
- Stable: small perturbations in `t_min` (e.g., 3–8 min) do not materially change results since the decline from t=0 is monotonic.

**Why not Alternative C (V_venous CV):**

Adding volume CV to resolve degeneracy without fixing the semantic mismatch produces non-degenerate PIs centered on the wrong point estimate. For valsartan with CV=0.15, the PI `[4.28, 7.02]` still fails to contain 4.02. The apparent fix is illusory.

**Why not Alternative B or D:**

Arterial blood observation and weighted central compartment both require architectural changes (route-aware node selection, central compartment definition) that add complexity disproportionate to the benefit. They also do not fully resolve degeneracy unless volumes carry CV.

### Secondary Recommendation: Add Physiological CV to Blood Volumes (orthogonal)

Adding `cv: 0.10` to `V_venous_blood` and `V_arterial_blood` (not `cv: 0.15`) is physiologically correct regardless of IV vs oral:

- Real inter-individual blood volume variation exists (ICRP 2002: CV ≈ 12–15%).
- This would improve the model's biological realism for the minority of future IV drugs and multi-dose TDM scenarios.
- However, this is a **separate** concern from the IV-Cmax observation fix and should be evaluated independently with a full 107-holdout re-run.

**Minimum-risk migration path:**

```
Phase 1 (minimum change): Implement Alternative E
  - 16 lines of code, route-conditional, zero holdout regression
  - Enables immediate re-run of ECM generalization test with valid PI
  - No YAML changes, no physiology changes

Phase 2 (optional, separate spec): Add physiological CV to blood volumes
  - YAML change: V_venous_blood, V_arterial_blood gain cv: 0.10
  - Full 107-holdout re-run required
  - Expected impact: minimal (oral Cmax driven by absorption, not blood volume)
```

### Spec Amendment Required

The ECM generalization test spec (`docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md`) declares:

> "Single engine run per drug. [...] No parameter adjustment between run and report."

A V3 re-run would constitute a new execution under a corrected observation methodology, not a post-hoc parameter adjustment. The appropriate procedure:

1. Implement Phase 1 (Alternative E) under a separate spec with pre-registration.
2. The ECM generalization test re-run is a **new test execution** under V3, not an amendment to the frozen `4fb6d38` result.
3. The `4fb6d38` result remains valid as "Mode C under V2 observation methodology."
4. The V3 re-run result is classified independently using the same 4-mode taxonomy.

---

## §6. Connection to the N=2 Generalization Test

### Re-interpretation of `4fb6d38`

The Mode C result is valid as-is. The PI degeneracy is not a test protocol error — it was discovered *through* the generalization test, which is the correct outcome of a pre-registered test revealing an infrastructure limitation.

Specifically:
- Both drugs **failed the PI-containment criterion** because zero-width PI fails by construction (the pass gate requires observed Cmax to be inside `[PI_low, PI_high]`; when both equal the point estimate and the point estimate differs from the observed by any amount, containment is impossible).
- Both drugs **would have passed the FE gate** under the `|log10 FE| ≤ 0.48` criterion alone (FE 1.11× and 1.35× both within 3-fold).
- Mode C (inconclusive) is the correct classification.

### V3 Re-run Protocol

Upon implementing V3 observation methodology:

1. A new execution spec is written and committed (pre-registration).
2. The frozen artifacts from the original test (`data/transporters/oatp1b1.json`, `data/validation/oatp_generalization_drugs.json`, per-drug doses and Cmax observations) are **reused unchanged** — they are not the problem.
3. Single MC run (N=1000, seed=42) under V3 methodology.
4. Result file: `data/validation/oatp_generalization_result_v3.json`.
5. Mode classification uses identical 4-mode taxonomy (A/B/C/D).

**Expected V3 outcome:**

Under V3:
- Point estimates will **decrease** relative to `4fb6d38` (because windowed Cmax at 5 min < Cmax at t=0).
- For valsartan: estimated V3 point estimate ≈ `5.405 × (C(5min)/C(0)) ≈ 5.405 × 0.85 ≈ 4.59 mg/L` (rough 2-comp approximation). FE ≈ 1.14×.
- For glimepiride: similar correction, estimated point estimate ≈ `0.270 × 0.87 ≈ 0.235 mg/L`. FE ≈ 0.97× (possible near-pass).
- PI width will be non-degenerate. The PI for the windowed Cmax includes variance from cardiac output (CV=0.10), fup, Kp, and enzyme affinity. Estimated CV of windowed Cmax ≈ 10–20% → 90% PI spans roughly ±35% of median.
- Whether V3 outcome is Mode A or Mode C depends on whether the PI containment gate passes. This cannot be determined analytically — engine execution required.

**Diagnostic value of V3 even in Mode C:** A V3 Mode C result with non-degenerate PI provides:
- A genuine FE estimate under the corrected observation methodology.
- An estimate of the systematic ECM bias (if any) for non-statin OATP1B1 substrates.
- A valid starting point for any follow-up spec.

---

## Appendix: Key File References

| File | Role in issue |
|---|---|
| `src/sisyphus/engine/solver.py` | `solve_mc()` line 140: `cmax = float(np.max(conc))` — the bug |
| `src/sisyphus/pk/endpoints.py` | `compute_endpoints()` line 27: same unfiltered max |
| `data/physiology/reference_man.yaml` | lines 17–19: `venous_blood volume: 3.7` (bare float, CV=0) |
| `data/validation/oatp_generalization_result.json` | PI degeneracy data |
| `src/sisyphus/engine/uncertainty.py` | `propagate_fast()`: observation_node pipeline |
| `src/sisyphus/pipeline/predict.py` | Production pipeline, no route-conditional Cmax logic |
| `data/reference/holdout.json` | All 107 holdout drugs confirmed oral |
