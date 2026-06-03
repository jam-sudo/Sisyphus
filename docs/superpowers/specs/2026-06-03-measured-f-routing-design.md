---
date: 2026-06-03
status: approved
parent: ../../claude/diagnosis.md
charter: Measured oral bioavailability (F) input channel for predict() — the one un-foreclosed F lever
---

# Measured-F routing — design

## Motivation

Three measurement-only investigations (2026-06-03; dead-ends.md DE-42, DE-43) foreclosed
every *engine-recalibration* route to the systematic bioavailability-F under-call: the
absorption knob is linear (a flat scalar that cannot reduce per-drug dispersion), the
residual is bidirectional first-pass, and the fixed-weight meta damps any engine change to
~18% pass-through on **both** the retrospective and prospective sets. The diagnosis names
exactly one un-foreclosed F lever: **per-drug measured-F routing** — let a caller who knows
a drug's true oral bioavailability supply it, instead of trusting the engine's emergent F.

This extends the SP1 measured-ADME channel (`MeasuredADMEInput`, 2026-06-02) with an
`f_bioavail` field. It is a **capability**, not a headline move: it lands on the engine
track only, is reported via the separate measured-input benchmark, and is **bit-identical to
the SMILES-only path when unused** (headline 2.698 untouched).

## The core problem

F (oral bioavailability) is a scalar the caller knows, but in the PBPK engine F is
**emergent**: `F = fa · Fg · Fh` (fraction absorbed × gut availability × hepatic
availability). There is no "F input" to set. Injection requires a method.

## Approach (approved): exposure-scaling

F sets the systemic **exposure scale**; the engine supplies the kinetic **shape**.

1. Run the engine oral as usual → `engine_pk` (Cmax, AUC, …).
2. Compute the engine's own emergent oral F via one **IV-reference solve** at matched dose:
   `F_engine = AUC_oral / AUC_iv`. Both are 0–24h truncated AUCs, so clearance cancels only
   approximately (exactly at infinite time) — `F_engine` carries a mild truncation bias for
   drugs whose t½ approaches the 24h window. Because the IV reference re-uses the same compiled
   graph and params as the oral solve, the scaling stays self-consistent and **target-hitting
   (corrected oral AUC / IV AUC == F_measured) is exact regardless**.
3. `k = clamp(F_measured / F_engine, 0.05, 50)`.
4. Scale `engine_pk.cmax`, `auc_0t`, `auc_0inf` by `k`; fold `f_bioavail_cv` into the Cmax/AUC
   CV in quadrature. `tmax`, `t_half` unchanged (shape preserved).

**Why this method** (vs absorption back-solve): F is a *scalar* — it can only constrain the
1-dof exposure scale, not the absorption-rate shape. Exposure-scaling is exact for AUC and
exact for Cmax *when the absorption rate is right*; for slow-absorbers the residual Cmax
shape error is corrected by the separate, composable measured-`peff` input (SP1). The
back-solve method conflates F (an amount) with `ka` (a rate the caller did not supply) and is
infeasible whenever `Fg·Fh < F_measured`. Exposure-scaling keeps the engine identity-blind
(correction lives in the pipeline) and the mechanistic profile intact.

## Contract

`MeasuredADMEInput` gains:
- `f_bioavail: float | None = None` — measured oral bioavailability, `0 < F ≤ 1`.
- `f_bioavail_cv: float = 0.15` — measurement CV, `≥ 0.10` (same floor rationale as siblings).

`f_bioavail` is **independent** (not paired with `fup`/`clint`); it may be supplied alone or
with any other measured field.

### Behavior
- **Oral route only.** For non-oral routes F=1 by definition; `f_bioavail` is ignored with a
  `measured_adme:f_bioavail ignored for non-oral route` warning.
- Correction lands on **`result.engine_pk`** (and therefore flows into the meta blend, exactly
  as the other measured overrides do). Reported via the separate measured-input benchmark.
- **Bit-identity invariant (load-bearing):** `predict(s, d, measured_adme=None)` and any
  `MeasuredADMEInput` with `f_bioavail=None` are byte-for-byte identical to the SMILES-only
  path. The IV-reference solve and scaling run **only** when `f_bioavail is not None`.
- Warning tag on application: `measured_adme:f_bioavail=<F> f_engine=<F_eng> k=<k>` (+ `(clamped)`).
- Degenerate guards: if the IV-reference solve fails or `F_engine ≤ 0`, skip the correction
  with a warning (engine_pk left as-is). `k` clamped to `[0.05, 50]` to bound numerical
  absurdity when the engine catastrophically mis-calls F.
- Deterministic path (default `n_mc_samples=0`) is fully corrected. When MC runs, the same
  `k` scales `cmax_90ci` so the interval stays consistent with the corrected point estimate.

## Architecture / invariant compliance

- **Engine identity-blind (#1):** F-correction is pipeline-layer post-processing of endpoints
  plus one extra engine *solve* (no engine code changed, no string matching).
- **No drug-specific branches (#6):** the correction is a generic `k`-scaling driven solely by
  caller-supplied numbers; no `if drug == X`.
- **Additive / no-touch (#8):** no change to `compiler.py`, `solver.py`, `DrugOnGraph` fields,
  the holdout list, or any Cmax loss. The headline path is untouched.

## Files

- `src/sisyphus/predict/adme.py` — add `f_bioavail` / `f_bioavail_cv` fields + validation.
- `src/sisyphus/pipeline/predict.py` — `_engine_oral_bioavailability(...)` helper (IV-ref
  solve), `_apply_measured_f(...)` helper (k-scaling), and the guarded call after `engine_pk`.
- `tests/unit/test_measured_adme_input.py` — f_bioavail validation cases.
- `tests/regression/test_measured_adme_passthrough.py` — bit-identity (None), behavior
  (F changes Cmax in the right direction), target-hitting (corrected engine F ≈ F_measured),
  non-oral ignore.
- `scripts/run_measured_adme_benchmark.py` — add a measured-F column on the PoC drugs (lit-F
  ballparks, clearly labeled).

## Out of scope (follow-ups)

- Full MC propagation of `f_bioavail` uncertainty beyond the CI rescale.
- Measured `fa`/`Fg`/`Fh` component routing (this ships only end-to-end F).
- Curated, citation-grade verified-F dataset (the benchmark uses approximate literature F).

## Validation

Correctness is proven by the test suite (bit-identity, direction, target-hitting). The
benchmark is **illustrative** (literature F values are approximate ballparks, per
diagnosis.md §3 / `run_f_decomposition.py`) and is reported separately — never blended into
the 2.698 headline.
