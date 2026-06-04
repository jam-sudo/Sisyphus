# Conformal Prediction Intervals — calibrated Cmax PI (fixes the 29.9%@90% under-coverage) — Design

**Date:** 2026-06-04
**Status:** CORE IMPLEMENTED + real-data proof (branch `fix/rbp-concentration-basis`); deployment (train/CV+ calibration artifact + pipeline wiring) is the next increment.
**Branch (proposed):** `feat/conformal-pi`

**Outcome (2026-06-04, core increment).** `src/sisyphus/validation/conformal.py` ships the split-conformal
core (finite-sample `conformal_quantile`, multiplicative `conformal_interval`, `empirical_coverage`,
`nonconformity_scores`), unit-tested (`tests/unit/test_conformal.py`, incl. the marginal-coverage
guarantee on synthetic biased/heavy-tailed residuals). **Real-data proof** (`tests/integration/
test_conformal_coverage.py`, LOO cross-conformal on the committed holdout cache — *measures* the
method's coverage, does NOT tune a deployed constant on the holdout): meta-track coverage **0.907 at
nominal 90%** (vs the documented MC **0.299**), and calibrated across levels (50%→0.505, 80%→0.804,
95%→0.953); 90% half-width **/÷8.88-fold** (the honest, wide price of structural error). **Remaining
(next increment):** generate `data/validation/conformal_calibration.json` from **train/CV+** residuals
(Invariant #5 — never holdout), then wire `PredictionResult.cmax_90ci` to the conformal interval while
relabelling the MC interval as the parameter-uncertainty component.

## 1. Problem

The production MC prediction interval has **empirical coverage 29.9% at nominal 90%** (N=107 holdout,
`data/validation/holdout_pi_coverage_2026-04-24.json`). The MC propagates **parameter uncertainty
only**; ~60 percentage points of the residual spread is **structural model-form error** not represented
in the MC. CLAUDE.md already flags H3 PI as "diagnostic, not calibrated — do not quote as a user-facing
interval." This makes the platform's native output format (uncertainty-bearing predictions) unusable as
a real interval.

External evidence (deep-research 2026-06-04): a parameter-only Bayesian/MC interval is **"confidently
wrong"** under model-form error — its width does not reflect the true error and shrinks to zero around a
biased point as data grows (Kennedy-O'Hagan; O'Hagan simmach). **Conformal prediction gives valid
marginal coverage regardless of base-model structural error** — the bias is absorbed into a wider
interval (CQR's coverage guarantee is distribution-free and independent of base-model correctness;
split-conformal on QSAR regression shows marginal coverage errors ~0.01–0.04). This is precisely the
30%→90% fix.

## 2. Approach — split / cross-conformal on log-Cmax residuals

Nonconformity score on the meta (production) track: `s_i = |log10(Cmax_pred,i) − log10(Cmax_obs,i)|`.
The conformal half-width is the `⌈(n+1)(1−α)⌉/n` empirical quantile `q_α` of the calibration scores;
the interval is `Cmax_pred ×÷ 10^{q_α}` (multiplicative, symmetric in log space). Coverage ≥ 1−α holds
marginally by exchangeability.

**Illustrative achievable widths** (computed live from the 107 holdout residuals — see §3 for why these
are NOT the deployed calibration): meta 90% PI = pred **/÷ 8.2-fold**; 50% = /÷ 2.28; 95% = /÷ 11.8.
Wide — that is the **honest price** of a calibrated interval given the structural error. The engine/ML
tracks are wider (90%: /÷16.4 / /÷11.9).

**Conditioning is NOT free:** Mondrian (group-conditional) conformal split on the AD flag is **inert**
here — in_ad 90% /÷8.18 vs out_ad /÷9.26 (the AD flag does not separate easy from hard, consistent with
DE-41). So do NOT condition on AD. If conditional (per-molecule) sharpness is wanted, use **CQR**
(conformalized quantile regression) or a heteroscedastic nonconformity score — marginal validity does
not imply conditional coverage (non-adaptive intervals over-cover easy molecules, under-cover hard ones).

## 3. ⚠ Invariant #5 — calibrate on TRAIN / CV+, validate on HOLDOUT

The holdout is inviolable: it must **never** be used in tuning — and fitting the deployed PI width on
holdout residuals IS tuning on the holdout. Therefore:
- **Deployed half-width** `q_α` is computed on a calibration set the base models did NOT train on:
  either a dedicated calibration split, or — since train is only 76 drugs — **cross-conformal (CV+ /
  Jackknife+)** over the train set so every calibration residual is out-of-fold (avoids in-sample
  optimism of train residuals).
- **Holdout coverage is the honest out-of-sample validation**, never the calibration source.
- The §2 "/÷8.2 from 107 holdout" is an **illustration of achievable width only**; the production
  interval comes from train/CV+ and may be modestly wider. Document this distinction in the artifact.

## 4. Where it lands

- `PredictionResult.cmax_90ci` currently comes from the MC path (`pipeline.predict`, `uncertainty.py`).
  Add a conformal interval as the **calibrated** total-predictive interval, computed at the pipeline
  layer from a shipped calibration artifact `data/validation/conformal_calibration.json` (the train/CV+
  `q_α` per nominal level, per track).
- **Keep the MC interval** as the explicit *parameter-uncertainty* component (relabel it as such); the
  conformal interval is the user-facing one. Do not conflate — they answer different questions.
- Engine stays untouched (Invariant #1); this is pipeline + validation only.

## 5. Validation (TDD)

1. **Failing test:** on a held-out validation split (NOT the deployed calibration set), assert conformal
   90% interval empirical coverage ≥ ~0.85 (target 0.90), where the MC interval gives ~0.30. Pins the
   fix.
2. Report sharpness: median interval width (fold). Honesty gate: coverage AND width both reported.
3. Coverage by nominal level (50/80/90/95) — reliability check (PIT / calibration curve).
4. Confirm production point predictions and the headline AAFE are **bit-identical** (conformal touches
   only the interval, never the point estimate → cannot move 2.784).

## 6. Risks / honesty

- **Wide intervals** (/÷8.2 at 90%) — real, and the honest consequence of structural error; not a bug.
- **Marginal ≠ conditional** — the interval is valid on average, not per-molecule; state this in the
  PredictionResult/warnings.
- **Exchangeability** — holdout/prospective drugs are a covariate shift vs train; marginal coverage may
  degrade out-of-distribution (weighted conformal is the documented remedy but needs a density-ratio
  estimate — out of scope v1; flag it).
- N is small (train 76 / holdout 107) → the conformal quantile itself is uncertain; report it as an
  estimate.

## 7. Hard-constraint check

- #1 identity-blind — pipeline/validation layer only, no engine change. ✓
- #5 holdout inviolable — **calibrate on train/CV+, validate on holdout** (§3); the holdout is never the
  calibration source. ✓
- #8 hard no-touch — no engine/compiler/solver/DrugOnGraph/holdout-list edits; adds a calibration
  artifact + pipeline interval; not a Cmax-loss fudge (touches the interval, not the point). ✓
