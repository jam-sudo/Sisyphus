# Bridge B / B1.x Phase-0 — zonal GSH-pool depletion

> **Context.** First **B1.x** sub-project, deepening the B1 zonal reactive-metabolite hazard probe (PR #82) from a *static* detox capacity to a finite, consumable, slowly-regenerating **GSH pool** — the actual acetaminophen mechanism the B1 static threshold was an explicit proxy for (B1 spec §1 non-goal, §6). Phase-0 stays **mechanistic and qualitative**, **harness-isolated**, headline **2.731 untouched**, post-processor only (GSH and the reactive metabolite are NOT engine species). Decided fidelity (brainstorm): a per-zone GSH-pool ODE driven by the merged axial parent profile; the toxicity endpoint is the **escaped covalent-binding flux**; gates carry B1's G2/G3 forward and add two signatures the static model **structurally cannot** produce — a depletion **cliff** and **path-dependence**.

## 1. Purpose & non-goals

**Purpose.** Demonstrate, on the real axial engine + a per-zone GSH-pool post-processor, that a *dynamic* finite detox pool produces three things the B1 static capacity cannot:
1. a **sharp depletion cliff** in the dose–hazard curve (vs B1's soft static threshold) — once local NAPQI formation outruns local GSH synthesis, the pool crashes and escaped covalent binding jumps nonlinearly;
2. **path-dependence** — the same total dose given as one bolus vs two spaced half-doses yields *different* hazard, because the pool partially regenerates between doses; the static (memoryless) model gives identical hazard for both;
3. an **NAC-precursor protective lever** — raising the baseline pool `GSH0` monotonically lowers hazard / shifts the cliff dose up (the clinical antidote analog);

all while reproducing the acetaminophen zonal pattern (pericentral bioactivation-high + pericentral GSH-low → zone-3/centrilobular vulnerability) and confirming the B1 orthogonality result (bulk parent PK invariant to the zonation that moves the hazard, DE-50).

**Non-goals (explicit).**
- **Mechanistic/qualitative demonstration, NOT a calibrated toxicity number.** No DILI probability, no quantitative threshold dose vs clinical data. An honest-negative (no cliff sharpening, or paths don't separate) is a first-class outcome (cf. DE-49/DE-50).
- **Harness-isolated.** No `predict()` / `reference_man.yaml` / holdout / `src/sisyphus/engine/` / `expand_axial` change. Headline **2.731 bit-identical**. Built on the synthetic axial skeleton.
- **GSH and the reactive metabolite are NOT engine species.** Phase-0 computes per-zone formation + pool depletion as a **post-processor** on the parent profile. Modeling the reactive metabolite as a transported (convected) species is a *separate* B1.x feasibility spike (B1 spec §6), out of scope here.
- **Single bioactivation enzyme + single GSH pool** on the synthetic skeleton (not the real multi-enzyme ECM liver).
- **No parameter tuned to manufacture a cliff or a path-separation.** Synthetic-param selection is for mechanism *visibility* on synthetic skeletons (documented), in the discipline of B1/zonation/PGx — never a fit to clinical data.

## 2. Background: from static capacity to a depleting pool

B1 modeled per-zone detox as a **constant** capacity: `hazard_i = ∫ max(0, R_form,i(t) − Vmax_detox,i) dt`. That is memoryless — the instantaneous escaped flux depends only on the instantaneous formation rate, so (a) the dose threshold is soft (a smooth ramp as `R_form` rises past the fixed capacity) and (b) two dosing paths with the same `C_u,i(t)` envelope integrate to the same hazard.

The real glutathione mechanism is a **finite, consumable pool with slow resynthesis**. Per zone i (inlet→outlet = periportal→pericentral):

- **formation** (reused from B1): `R_form,i(t) = Vmax_bio,i · C_u,i(t) / (Km_bio + C_u,i(t))`, `Vmax_bio,i ∝ local CYP abundance` (zonatable, pericentral-high);
- **pool dynamics**: `dGSH_i/dt = k_syn·(GSH0_i − GSH_i) − R_form,i(t)·GSH_i/(Kg + GSH_i)` — first-order resynthesis toward the baseline `GSH0_i` (zonatable, pericentral-**low**) minus saturable scavenging consumption (a unit of NAPQI consumes a unit of GSH; the `GSH/(Kg+GSH)` factor makes scavenging efficiency collapse as the pool empties);
- **hazard** (escaped covalent-binding flux): `hazard_i = ∫ R_form,i(t)·(1 − GSH_i(t)/(Kg + GSH_i(t))) dt` — the fraction of formed NAPQI that escapes scavenging because the pool is depleted, the covalent-binding/toxicity proxy. Same units/meaning as the B1 hazard (covalent binding beyond detox), so the B1 gates carry over and the upgrade is direct (static capacity → dynamic pool).

**Why the dynamics matter.** When `R_form,i` stays below `k_syn·GSH0_i` (the max sustainable consumption), the pool holds near `GSH0_i`, the escape factor `(1 − GSH/(Kg+GSH))` stays small, hazard ≈ 0. Once `R_form,i` outruns resynthesis the pool drains toward 0, the escape factor → 1, and hazard rises sharply — a **cliff**, not a ramp. And because depletion is **cumulative**, the same dose spread over time (allowing partial regeneration) depletes less — **path-dependence**. Neither is reachable by the static model.

**Orthogonality (DE-50, carried).** Total parent bioactivated = total hepatic extraction, invariant to the CYP spatial distribution (plug-flow, DE-50). The redistribution that leaves bulk parent Cmax/AUC unchanged still moves the per-zone GSH-hazard profile — the Bridge-B orthogonal-information claim, re-confirmed here for the dynamic endpoint.

## 3. Method

### 3.1 Per-zone parent profile (reuse)
Reuse B1's `_parent_profile_by_zone` (importlib-load `scripts/probe_zonal_hazard.py`, the established pattern it already uses for `probe_liver_zonation`): build the synthetic axial liver (`_axial_graph`, PR #79), solve a single oral dose, return per-sub-tank unbound parent `C_u,i(t)` (= `fup · c_node`) inlet→outlet plus the time grid. Bulk parent `E` via B1's `bulk_E` (for the G2 arm).

**Divided-dose profile (new, for G-time).** `_divided_dose_profile_by_zone(..., n_splits, tau_h)`: solve the axial ODE in segments — administer `dose/n_splits` at t=0, integrate to `tau_h`, **add** the next `dose/n_splits` to the administration-node state, re-solve from the carried state, repeat; concatenate the per-sub-tank `C_u,i(t)` across segments onto one monotonic time grid. Pure harness-level segmentation of the existing compiled ODE solve — no `regimen`/engine surface, no new engine species. Equal total dose (mass) to the bolus arm; the parent exposure envelopes may differ slightly under saturable first-pass (lower peaks → less saturation) — the **static-model control** in G-time absorbs that confound, isolating the pool-dynamics contribution.

### 3.2 The GSH-pool hazard post-processor (pure, tested)
`gsh_pool_hazard(c_u_by_zone, vmax_bio_by_zone, km_bio, gsh0_by_zone, k_syn, kg, time) -> list[float]` in `src/sisyphus/validation/pgx_metrics.py`. Per zone: integrate the §2 pool ODE with a **self-contained fixed-step RK4** (linear interpolation of `C_u,i(t)` between grid points; an internal refined step independent of the engine's `t_eval` density for integration accuracy), then trapezoid-integrate the escaped-flux hazard. Numpy-only, deterministic, no scipy — so it is trivially unit-testable against a hand-worked constant-`C_u` steady state. `vmax_bio_by_zone` and `gsh0_by_zone` come from `zonation_weights × total` (independently zonatable). B1's static `zonal_hazard` is **left untouched** — the probe contrasts dynamic vs static.

### 3.3 The sweeps (`scripts/probe_gsh_depletion.py`)
- **Zonation sweep (G1/G2):** vary bioactivation and `GSH0` zonation (uniform / pericentral / periportal × ratio), totals fixed; record per-zone GSH-hazard profile (peak zone + magnitude) AND bulk parent `E`.
- **Dose sweep (G3):** for the APAP config (bioactivation pericentral-high, `GSH0` pericentral-low), vary dose; record per-zone hazard vs dose for **both** the dynamic pool and B1's static `zonal_hazard` on the same config; compute each curve's sharpness (max local log-log slope of `maxH` vs dose); the dynamic cliff must exceed the static ramp by a pinned margin.
- **Path sweep (G-time):** at a fixed total dose above the cliff, compare bolus vs `n_splits=2`, `tau_h` spaced; record dynamic hazard (divided < bolus) and the static control (≈equal). Also report the per-zone min-GSH for both paths (the depletion-depth secondary readout).
- **Protective-lever sweep (G-NAC):** scale `GSH0` up (e.g. ×1.5, ×3) on the APAP config; record cliff-dose shift / hazard reduction (monotone).

## 4. Metric & pre-registered gates

- **G1 — localization (sanity).** APAP config → hazard peaks at the outlet zone (zone 3, pericentral) and that zone's GSH depletes deepest; flipping the bioactivation/`GSH0` gradients moves the peak. Near-trivial sanity check, not the result.
- **G2 — local matters, bulk doesn't (DE-50 closure, carried).** Holding total bioactivation enzyme fixed, varying its zonation leaves bulk parent `E` ~invariant (`|ΔE|` below a pinned small tolerance, reusing the DE-50 result) **while** the per-zone GSH-hazard peak-zone and/or peak magnitude change materially (above a pinned threshold).
- **G3 — dynamic cliff > static soft threshold (centerpiece).** On the same APAP config and dose grid, the dynamic-pool `maxH(dose)` curve has a strictly **larger** max local log-log slope than B1's static `zonal_hazard` `maxH(dose)` curve, by a pinned margin — the threshold sharpens from a ramp to a cliff. Below the cliff dose the dynamic hazard is ≈0 (pool sustains); above it the pool crashes and hazard jumps.
- **G-time — path-dependence (uniquely dynamic).** At equal total dose (above the cliff), the dynamic hazard for two spaced half-doses is **strictly less** than for the single bolus (pinned relative margin), and the divided arm's min-GSH is higher (less depleted). **Control:** B1's static hazard for the two paths differs by less than a pinned tight tolerance (memoryless ⇒ ≈equal). Only the dynamic model separates the paths.
- **G-NAC — protective lever.** Raising `GSH0` monotonically lowers `maxH` at a fixed supra-threshold dose (and raises the cliff dose). Correctly signed.
- **Honest-negative path:** if the dynamic cliff does not out-sharpen the static ramp (G3), or the paths do not separate beyond the static control (G-time), report it as-is. No parameter tuned to manufacture either effect.

## 5. Components
- **Extend** `src/sisyphus/validation/pgx_metrics.py`: `gsh_pool_hazard(...)` (self-contained RK4 pool integrator + escaped-flux trapezoid), pure, tested. B1's `zonal_hazard`/`mm_rate`/`zonation_weights` reused unchanged.
- **New** `scripts/probe_gsh_depletion.py`: importlib-reuses B1's `_parent_profile_by_zone` + `bulk_E`; adds `_divided_dose_profile_by_zone` (two-segment solve); the four sweeps; G1/G2/G3/G-time/G-NAC scoring; report writer. No hand-written result values.
- **New** `tests/unit/test_gsh_pool_hazard.py`: constant-`C_u` steady-state worked example (`k_syn·(GSH0−GSH*) = R_form·GSH*/(Kg+GSH*)`); zero-formation → GSH stays at `GSH0`, hazard = 0; saturating-formation → GSH → 0, hazard → `∫R_form`; monotone-in-`GSH0` (larger pool → smaller hazard); deterministic/reproducible.
- **New** `tests/integration/test_gsh_depletion_probe.py`: G1 localization, G2 bulk-`E`-invariant-while-hazard-variant, G3 dynamic-cliff-sharper-than-static, G-time divided<bolus (+ static≈equal control), G-NAC protective-monotone, per-zone `C_u` inlet→outlet sanity, headline-isolation guard (`4track_holdout_predictions.json` untouched; `test_mm_headline_bit_identity` + `test_cached_holdout_aafe_is_2p731` pass).
- **New** `data/validation/gsh_depletion_2026-06-18.{json,md}`: sweep tables + G1/G2/G3/G-time/G-NAC verdicts + the explicit conclusion (a dynamic GSH pool yields a depletion cliff + path-dependence + a precursor lever the static model cannot, orthogonal to bulk PK, qualitatively the acetaminophen mechanism).

## 6. Out of scope (→ later B1.x / Bridge B)
Transported reactive-metabolite as a convected engine species (downstream-detox fidelity; its own feasibility spike). Quantitative toxicity threshold/PoD vs clinical data (DOSE-L1000 calibration). Productization into `pkpd.py` / a tox module / `reference_man`. Multi-enzyme real liver. Mitochondrial/ROS amplification beyond the GSH pool. Tumor/efficacy PD (B4).

## 7. Constraints (operational)
Harness-isolated; NO `predict()` / `reference_man.yaml` / `src/sisyphus/engine/` change; headline **2.731 bit-identical** (v2.2a empty-`enzyme_km` path + `test_cached_holdout_aafe_is_2p731` unchanged). NO fitting to clinical data; NO cherry-picking (gates pre-registered; honest-negative path explicit). Commit as `jam-sudo <jam-sudo@users.noreply.github.com>`, NO Claude/AI trailer, `git commit --no-verify`. Tests with `/opt/miniconda3/bin/python -m pytest`. `ruff check src tests` line-length 100. Reuses the merged axial machinery (PR #79), v2.2a saturable flux, `zonation_weights` (DE-50), and the B1 post-processor harness (PR #82).
