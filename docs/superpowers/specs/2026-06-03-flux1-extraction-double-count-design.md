# FLUX-1: Flow-Limitation Double-Count Fix + Gut Re-Anchor — Design

**Date:** 2026-06-03
**Status:** approved (correctness-first, honest-report acceptance criterion)
**Branch:** `fix/flux1-extraction-double-count`

## 1. Problem (triple-verified)

The engine models clearing organs (liver, gut_wall) as **perfusion compartments**: each has an
explicit convective outflow `FlowEdge` carrying `Q·c_out`, **and** a `ClearanceEdge` whose sink
applies the *whole-organ* clearance `CL_h · c_out`. But `CL_h = Q·fu·CLint/(Q+fu·CLint)` already
embeds the flow limitation `Q`. Applying it to the post-washout outlet concentration `c_out` inside a
compartment that *separately* washes out `Q·c_out` **double-counts the flow term**.

Steady-state mass balance `Q·C_in = Q·c_out + CL_h·c_out` gives realized extraction
`E = CL_h/(Q+CL_h)`. Substituting `CL_h`:

```
E_coded = fu·CLint / (Q + 2·fu·CLint)   →  0.5   as fu·CLint → ∞
E_canon = fu·CLint / (Q +   fu·CLint)   →  1.0   (correct)
```

A literal extra factor of 2 on `fu·CLint`. **The engine structurally cannot extract > 50% of
liver/gut inflow**, flooring oral first-pass `F = (1−E_gut)(1−E_liver)` near 0.25 regardless of CLint.

**Verification (3 independent methods):**
1. Topology — `reference_man.yaml`: liver has inflows arterial(0.065)+portal(0.19)=0.255·CO and a
   separate `liver→venous_blood` (0.255·CO) outflow + `liver→metabolized_hepatic` (extended) clearance.
   Same dual structure for gut_wall (`well_stirred`). `total_inflow` == the convective Q exactly.
2. Algebra — `E_coded = x/(Q+2x)` reproduced to 8 digits.
3. Empirical — probing the real flux code (both `well_stirred` and `extended`): at `fu·CLint=5548`,
   E_engine = **0.496 / 0.495** vs canonical 0.982. Caps at 0.5 on both production paths.

This is the documented root cause of the first-pass-F under-prediction (DE-41/42/43) — previously
mis-attributed to an irreducible calibration floor; it is a correctable formulation error.

## 2. Fix — apply the intrinsic (flow-unlimited) clearance to `c_out`

In a perfusion compartment the metabolic sink must be the **intrinsic** clearance; the separate
`Q·c_out` edge then *emerges* the canonical `E → 1.0`.

| Model (`engine/flux.py`) | current sink | corrected sink |
|---|---|---|
| `well_stirred` | `clh·c_out`, `clh=Q·fup·CLint/(Q+fup·CLint)` | `fup·CLint·c_out` |
| `extended`/ECM | `clh·c_out`, `clh=Q·fup·ps_inf·cl_int_h/[Q·(ps_eff+cl_int_h)+fup·ps_inf·cl_int_h]` | `CL_int,hep·c_out`, `CL_int,hep = fup·ps_inf·cl_int_h/(ps_eff+cl_int_h)` |
| `prodrug_activation` | `cl_organ·c_out` (WS form) | `fup·CLint·c_out` |
| `parallel_tube` (unused in production) | `Q·(1−e^(−fup·CLint/Q))·c_out` | `fup·CLint·c_out` + comment: true PT needs an axial-gradient liver (out of scope) |

The ECM `clh` is exactly the well-stirred *wrap* of `CL_int,hep` (`clh = Q·CL_int,hep/(Q+CL_int,hep)`),
so dropping the wrap is the minimal correct fix. The `gfr_filtration` branch is **not** a
perfusion-compartment double-count (kidney has no convective drug-return modeled the same way) and is
out of scope here (its separate low-severity RBP-basis nit is a different follow-up).

**Rejected — form (b)** `CL_h·C_in` (DESIGN.md spec form): also correct, but needs the mixed inflow
concentration `C_in`, breaking the engine's per-edge locality. Form (a) is standard perfusion-limited
PBPK and a minimal local change.

**Direction:** `fup·CLint ≥ clh` always ⇒ larger sink ⇒ **more first-pass extraction ⇒ lower Cmax/F**,
and *only* for high-extraction drugs (at low `fu·CLint`, both forms ≈ `x/Q`, unchanged). Monotonic
toward correct.

## 3. JAX parity

`engine/rhs_jax.py` mirrors the broken form for all four edge types. Apply the identical intrinsic-
clearance change there; assert numpy↔JAX agreement post-fix.

## 4. Recalibration — hold midazolam gut `E_gut` invariant

Liver enzyme affinities are XGBoost-decomposed (`_decompose_clint`: `abundance×affinity×ivive =
CLint_hepatic`, the true in-vitro intrinsic clearance) ⇒ **no liver recalibration**. Only the
`gut_wall` CYP3A4 abundance (`reference_man.yaml:92`, the explicit midazolam back-fit) was tuned
against the depressed ceiling.

**Strategy:** scale gut CYP3A4 abundance by `k_gut = Q_gut/(Q_gut + fup·CLint_gut)` evaluated at
midazolam, so midazolam's gut extraction is identical pre/post fix.

**Measured anchor (midazolam, oral 2 mg, fup=0.030, Q_gut=58.5):** `fup·CLint_gut = 31.23` ⇒
`k_gut = 58.5/(58.5+31.23) = 0.652` ⇒ abundance `21224338 → 13837729` (≈1.384e7). Verified: under the
fixed formula this reproduces `E_gut = 0.2582` exactly (= pre-fix value).

Midazolam is in `train`, **not** `holdout` (authoritatively checked: `holdout.json` `train` list, 76
drugs) — invariant #5 respected. Anchoring to a physiological reference, not to Cmax loss — invariant
#8 respected. Midazolam's *liver* E corrects to physiological (0.124→0.141) — intended; its overall
first-pass F moves only 0.650→~0.637.

**Note (follow-up, out of scope):** even after `k_gut`, the gut abundance remains well above the
physiological gut:liver ratio — it lumps real first-pass geometry, not only the bug. A fully
physiological gut-CYP3A4 model is a separate effort.

## 5. Validation, regen, honest report

1. **Failing test first:** `tests/unit/test_extraction_ceiling.py` — high `fu·CLint` ⇒ engine
   `E→1.0` (>0.9), not 0.5, for `well_stirred` and `extended`. (The probe becomes a regression test.)
2. Update hand-computed references in `test_flux.py` / `test_ecm_flux.py` to the intrinsic formula;
   confirm the ECM→well-stirred degeneracy still holds (both now intrinsic); mass-balance preserved.
3. Apply `k_gut` to the YAML; verify midazolam `E_gut` unchanged empirically.
4. Run full `pytest`; triage golden-value failures (distinguish "test encoded the bug" from real
   regression — verify each updated golden is physiologically sound before changing it).
5. **Regenerate** `4track_holdout_predictions.json` (`run_engine_benchmark.py`), refresh CI
   (`bootstrap_4track_ci.py`), re-run prospective N=28.
6. Report the **true** new Meta / Engine / In-domain / Prospective AAFE — whatever it is. Reconcile the
   CLAUDE.md metrics block against the regenerated cache.
7. Document: `experiment-log.md` (numeric outcome), `diagnosis.md §8` (FLUX-1 = root cause of
   DE-41/42/43 first-pass-F under-prediction — the headline finding regardless of the AAFE number),
   and remove/qualify the DE-41/42/43 "engine F is not a lever / irreducible floor" framing where the
   double-count supersedes it.

**Baseline (this environment, pre-fix, reproduced exactly):** Engine 3.831 / ML 3.010 / Meta 2.698,
N=107. Benchmark runtime ~13 s.

## 6. Risks

- **Headline may regress** (accepted; documented honestly). Engine-only AAFE *will* move materially
  (its over-prediction was the error-cancellation counterweight); meta damps to ~18%.
- Larger clearance ⇒ stiffer ODE for very-high-CLint drugs; watch `solver_success` + mass-balance in
  the regen.
- Tests that hard-code midazolam-class Cmax against Omega golden values may need updating; verify each
  is closer-to-correct, not blindly rewritten.

## 7. Scope

- **In:** the 4 clearance models' intrinsic-clearance fix (numpy + JAX), gut re-anchor, regen, docs.
- **Out:** the other 39 audit findings (Berezhkovskiy, zwitterion, RBP-basis, GFR `c_plasma` var,
  B-11 ECM dead-code, etc.) — separate follow-ups. Optional ride-along: `MeasuredADMEInput.fup ≤ 1.0`
  guard (low-risk) — excluded unless requested.

## 8. Hard-constraint check

- #5 holdout inviolable — re-anchor uses midazolam (train), not holdout. ✓
- #6 no drug-specific branches — fix is identity-blind math; re-anchor is a single global abundance. ✓
- #8 hard no-touch — touches `engine/flux.py` + `engine/rhs_jax.py` (allowed); NOT compiler/solver,
  NOT DrugOnGraph fields, NOT holdout list; abundance re-anchor is a physiological anchor, not a
  Cmax-loss fudge. ✓
