# P7 Ketorolac IVIVE fup Mismatch — Decision Package

**Status:** awaiting user decision
**Context:** P7 was flagged as a HIGH RISK backlog item. Investigation shows the fix was already tried and failed in production (2026-04-11, 35th error cancellation). This package re-states the evidence and proposes closing the task.

## The problem in one paragraph

Ketorolac's XGBoost-predicted fup (0.069) disagrees with DrugBank measured (0.010) by 6.9×. The current gate prefers **XGBoost** when disagreement >5× — so ketorolac flows through the engine at fup=0.069 (underbound). This contributes to the engine's ketorolac Cmax landing at 0.8× observed (FE=3.25) and TDM posterior bias of -31% (SBI) / -46% (IS). All three TDM methods cannot reach coverage on ketorolac — `docs/tdm_ci_calibration.md` confirms this is engine-level, not a TDM calibration issue.

## What was already tried (2026-04-11)

Inverted the gate: prefer DrugBank measured fup even at >5× disagreement (the "ketorolac fix").

**Results on 107-holdout:**
- Engine AAFE: 3.421 → **3.726 (+0.306, regression)**
- Meta AAFE: 2.695 → 2.728 (+0.033, noise)
- Ketorolac engine fold: 0.259× → 0.534× (directionally better, still well under truth 0.80)
- Ketorolac TDM CI: still doesn't cover even with floor=0.5

**Root cause:** Ketorolac's failure is not just fup. CLint also contributes. A partial fix (fup only) breaks error cancellation on 100+ other drugs for ~zero coverage gain on ketorolac. This is the 35th consecutive error-cancellation failure. Reverted; documented in `adme.py:236-250`.

## Options

| # | Name | Code LOC | Risk | Holdout AAFE impact | Ketorolac impact |
|---|------|:--:|:--:|:--:|---|
| 1 | Status quo | 0 | None | 0 | unchanged |
| 2 | AD flag "high-acid-low-fup" | ~10 | Low | 0 | informational warning |
| 3 | Retry engine-level override | 0 (revert) | **High** | +0.03 meta, +0.31 engine | minor improvement |
| 4 | Class-aware fup model | ~30+training | Medium-High | Unknown, likely negative | Unknown |

---

## Option 1 — Status quo

**Change:** None. Ketorolac remains on XGBoost fup.

**State:** Known structural limitation. Engine predicts ketorolac Cmax at ~0.8× observed. TDM posterior bias -18% to -46% depending on method. Cannot be fixed under current architecture + data.

**Why this is correct:**
- The fix was already tried empirically with rigorous measurement.
- Result: net negative system-wide (+0.306 engine AAFE) for near-zero ketorolac gain.
- Invariant #6 prohibits drug-specific branches. A "ketorolac-only" fix violates the architecture principle that has protected the system from 35+ prior attempts.
- Meta AAFE 2.695 is the documented ceiling. Known outliers are an expected feature, not a bug.

---

## Option 2 — Add AD flag for high-acid-low-fup drugs

**Change** (`src/sisyphus/validation/applicability.py`, ~10 LOC):

Flag drugs with DrugBank measured fup < 0.02 AND pKa < 5 (acidic, highly protein-bound). Ketorolac, indomethacin, diclofenac, warfarin would all flag.

**Effect:** `PredictionResult.ad_flags` contains `"high_acid_low_fup"`. User-facing: `confidence="medium"` downgrade. No numerical change to predictions.

**Why:** Informational transparency — users see the flag in output and interpret ketorolac-class predictions with caution. Lowest-risk path to "handling" the issue without trying a fix that's empirically known to fail.

**Test plan:** unit test on ketorolac/indomethacin → flag set; on morphine (base, fup=0.35) → flag absent.

---

## Option 3 — Retry engine-level override

**Explicitly DO NOT do this.** The 2026-04-11 attempt measured:
- Engine AAFE +0.306 (matching the 34-prior pattern of error-cancellation failures)
- Meta AAFE +0.033 (noise)
- Ketorolac TDM coverage: still failed

Listed for completeness. Any future attempt must include a pre-registered hypothesis about *why this time will differ* and a fallback plan.

---

## Option 4 — Class-aware fup model for acids

**Change:** Train separate XGBoost fup model on acid subset (pKa < 5). Replace generic fup for those drugs.

**Priors against:**
- Kinase class-aware weights (2026-04-07) failed — negative.
- F% bioavailability predictor (2026-04-07) failed — negative.
- Direct CL/F, t½ predictors (post-VDss) failed 6× — negative.
- ~40 total negative results under this broader pattern.

**Why it might still be worth considering:** Acid-specific chemistry is a genuinely orthogonal channel (protein binding physics differs from clearance). A well-curated acid-specific dataset + scaffold-CV rigor could plausibly succeed where others failed.

**Why I don't recommend starting now:** No acid-specific fup dataset is currently curated. Would need 2-4 weeks of data collection, then training + error-cancellation test. Low ROI given 35 prior failures under similar premises.

---

## My recommendation: **Option 1 (status quo) + optional Option 2 (AD flag)**

**Reasoning:**
1. Option 3 is a known failure — repeating it is negligent.
2. Option 4 has weak priors and high opportunity cost.
3. Option 1 is the honest state: the limitation is real, documented, and architecturally necessary.
4. Option 2 is the smallest-risk transparency improvement (~10 LOC, zero model changes, improves user-facing honesty).

**Proposed closing action:** Mark P7 as "resolved — documented structural limitation." Optionally add AD flag (Option 2) as a 10-LOC PR.

---

## Decision

Please pick one:

- [ ] **Option 1** (status quo) — close P7 as resolved-structural-limitation, no code change
- [ ] **Option 1 + Option 2** (status quo + AD flag) — close P7, add ~10 LOC AD flag + unit test
- [ ] **Option 3** (retry override) — explicitly override my recommendation; I will run the experiment but expect it to fail as before
- [ ] **Option 4** (class-aware model) — multi-week effort, defer to a milestone planning discussion
