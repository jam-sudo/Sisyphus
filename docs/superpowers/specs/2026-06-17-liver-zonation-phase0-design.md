# Phase-0 liver-zonation invariance probe — zonation does NOT change bulk first-pass

> **Context & course-correction (2026-06-17 ultrathink).** This started as a Bridge-A probe asking "does zonating a hepatic enzyme along the axial liver (`liver__ax1..N`, PR #79) change first-pass / Cmax?". An ultrathink review **refuted the premise analytically**: in the continuous (plug-flow) limit, hepatic extraction is **invariant to the spatial distribution** of the enzyme — it depends only on the *total*. So zonation is **not** a bulk-first-pass lever; any effect in the finite-N axial model is a **discretization artifact** that vanishes as `N→∞`. The probe is reframed to **demonstrate that invariance** (and, as a bonus, validate that the axial CSTR cascade converges to the correct plug-flow continuum). Zonation's real pharmacological value is **local/zonal** (pericentral bioactivation → zone-3 toxicity), i.e. **Bridge B (PD/toxicity)**, not Bridge A.

## 1. Purpose & non-goals

**Purpose.** Demonstrate, on the real engine, that **first-pass extraction `E` / Cmax is invariant to the axial spatial distribution of a hepatic enzyme** (total abundance preserved): the difference between a zonated and a uniform axial liver, `ΔE(N) = E_zonated − E_uniform`, **decays toward 0 as the sub-tank count `N` grows** (convergence to the plug-flow continuum). Quantify the `N=10` discretization artifact so future axial work knows its size, and validate the axial cascade against the closed-form plug-flow extraction. **Conclusion to establish:** zonation does not refine bulk PK; redirect its modeling value to Bridge B.

**Non-goals (explicit).**
- **A negative-result / model-validation demonstration, not an accuracy claim.** We are *disproving* a Cmax lever, not adding one. Headline **2.731 bit-identical**.
- **Harness-isolated.** No `predict()` / `reference_man.yaml` / holdout / `src/sisyphus/engine/` / `expand_axial` change. Zonation is applied to a **synthetic** axial skeleton via `dataclasses.replace` (the PGx-harness pattern).
- **No productization.** A YAML `zonation_profile` is explicitly **not warranted** for bulk PK (this probe's finding); any future zonation work belongs to Bridge B (zonal toxicity), separately specced.
- **Single-enzyme synthetic skeleton**, not the real multi-enzyme ECM liver.

## 2. The analytic result the probe demonstrates

The axial liver is a finite-`N` CSTR-cascade discretization of a continuous sinusoid (plug-flow tube), flow `Q`, dual inflow into tank 1 (periportal, zone 1), outflow from tank N (pericentral, zone 3 → venous). With the v2.2a intrinsic-clearance flux, the continuum steady-state profile obeys
```
dc/dx = −(1/Q)·v(x)·c/(Km+c)   ⟹   ∫(Km/c + 1) dc = −(1/Q)∫ v(x) dx
⟹   Km·ln(c_out/c_in) + (c_out − c_in) = −V_total / Q
```
The right-hand side depends **only on `V_total = ∫v(x)dx`**; the spatial density `v(x)` integrates out. So **`c_out` (hence `E`) is independent of the enzyme's spatial distribution** — for Michaelis–Menten *and*, as `Km→∞`, for linear clearance (`E_lin = 1 − exp(−fu·CLint_total/Q)`, the axial machinery's existing convergence target).

**Two consequences the probe shows numerically:**
1. **Invariance in the limit:** `ΔE(N) = E_zonated − E_uniform → 0` as `N→∞`, for both regimes and any zonation direction/steepness.
2. **Finite-N artifact structure:** at finite `N` the CSTR cascade is *not* plug-flow, so a residual `ΔE(N) ≠ 0` exists — **linear**: a direction-*symmetric* convexity effect (uniform maximizes `E`; pericentral ≈ periportal); **saturable**: a direction-*asymmetric* effect (the inlet tank sees higher `C_u`, so `E_periportal > E_pericentral` — note this is the *opposite* of the naive "pericentral is more efficient" intuition, and is itself an artifact that vanishes with `N`).

## 3. Method

### 3.1 Zonation weight profile (pure, tested)
`zonation_weights(n, ratio, direction, shape="linear") -> list[float]`: `n` weights summing to **1.0**; `direction="pericentral"` increases toward tank N, `"periportal"` decreases, `"uniform"` = `1/n`; `ratio = w_max/w_min` (`ratio=1` ⇒ uniform). In `src/sisyphus/validation/pgx_metrics.py`, unit-tested.

### 3.2 Total-preserving application
`apply_zonation(axial_graph, gene_tag, weights)` sets sub-tank `i` abundance `= V_total × weights[i]` via `dataclasses.replace`. **Invariant: `Σ abundance_i` equals the uniform graph's total exactly** (pure redistribution — isolates the spatial effect from any abundance change).

### 3.3 The convergence sweep
For each **regime** ∈ {linear, saturable} and **direction** ∈ {pericentral, periportal} (plus the uniform baseline), over **N** ∈ {5, 10, 20, 40, 80} × a physiological **ratio** ∈ {2, 3} (extended {1.5, 4} for robustness) × (saturable) **Km** spanning the engaging range:
- build the synthetic axial skeleton at `N` (reuse `_axial_graph`); anchor `cltot` to a target first-pass `E` (`_engine_e_h` / `anchor_em`);
- apply zonation; measure `E` (`_engine_e_h`) and Cmax (`_cmax_auc_tmax`);
- record `ΔE(N) = E_zonated − E_uniform` and relative `ΔCmax(N)`.

## 4. Metric & pre-registered gates

- **G1 — invariance (the headline).** `|ΔE(N)|` (and `|relative ΔCmax(N)|`) **decreases monotonically toward 0 as `N` grows**: specifically `|ΔE(80)| < |ΔE(10)|` and `|ΔE(80)|` below a small pinned tolerance (plan-pinned, e.g. < 0.5% absolute), for **both** regimes and **both** directions, across the ratio grid. This is the demonstration that zonation does not change bulk first-pass.
- **G2 — analytic convergence (oracle).** **Linear regime (clean, dose-independent):** as `N→∞`, the engine's AUC-based `E` (uniform *and* zonated) converges to `1 − exp(−fu·CLint_total/Q)` to a pinned tolerance — validating the axial cascade against the plug-flow continuum *independent of distribution* (and doubling as a correctness check on the PR-#79 axial machinery). **Saturable regime:** invariance is shown by **G1** (`ΔE(N)→0`); the closed form `Km·ln(c_out/c_in)+(c_out−c_in)=−V/Q` is *steady-state*, so it is checked only against a **steady-state** `c_out` (via the `regimen` solver / `_steady_state_exposure`, NOT the dynamic single-dose AUC `E`, which is dose-dependent for MM) as a secondary, distribution-independent check.
- **G3 — artifact characterization.** Report the `N=10` artifact magnitude `ΔE(10)` and confirm its structure: linear is direction-symmetric (`|E_pericentral − E_periportal|` ≈ 0 at fixed N), saturable is direction-asymmetric (`E_periportal > E_pericentral` at fixed N) — and **both** asymmetry and convexity decay with `N`. A *non-decaying* `ΔE(N)` would falsify the invariance (a real finding to investigate, not forced).
- **Ratio-1 oracle:** uniform weights reproduce the unmodified axial `E` bit-identically (redistribution is a no-op at `ratio=1`).
- **Honest framing:** the deliverable is the *negative* (no bulk-PK lever) + the convergence validation; report as-is.

## 5. Components
- **Extend** `src/sisyphus/validation/pgx_metrics.py`: `zonation_weights` (pure, tested); `plugflow_E_linear(fu, clint_total, q)` and `plugflow_cout_mm(c_in, km, v_total, q)` (the closed forms, for G2).
- **New** `scripts/probe_liver_zonation.py`: `apply_zonation`, the N-convergence sweep, G1/G2/G3 scoring, report writer. Reuses the synthetic-engine helpers from `scripts/validate_pgx_cmax_v2b.py` (`_axial_graph`, `_well_stirred_graph`, `_drug`, `_sat_drug`, `_engine_e_h`, `_cmax_auc_tmax`, `anchor_em`) via `importlib` (the pattern the PGx tests use) — no edit to the merged harness.
- **New** `tests/unit/test_zonation_weights.py`: sum-to-1, monotonic per direction, correct `ratio`, uniform at `ratio=1`; closed-form helpers vs hand-worked values.
- **New** `tests/integration/test_liver_zonation_invariance.py`: total-abundance preserved (exact); `|ΔE(N)|` decreasing in `N` toward ~0 (G1); engine `E` → plug-flow closed form as `N` grows, uniform and zonated alike (G2); finite-N artifact structure (linear symmetric, saturable asymmetric, G3); ratio-1 no-op; headline-isolation guard (holdout cache + `test_mm_headline_bit_identity` + `test_cached_holdout_aafe_is_2p731`).
- **New** `data/validation/liver_zonation_invariance_2026-06-17.{json,md}`: the `ΔE(N)` convergence table, G1/G2/G3 verdicts, and the explicit conclusion (zonation is not a bulk-PK lever → Bridge B).

## 6. Out of scope (→ Bridge B, separately)
The zonal/local surface where zonation *does* matter: per-zone metabolite / reactive-metabolite formation (pericentral CYP3A4 bioactivation → zone-3 exposure), zonal toxicity / DILI, consuming the engine's per-sub-tank concentrations. Plus: productized `zonation_profile`, real-substrate zonation, non-CYP zonation, sourcing zone-resolved proteomics.
