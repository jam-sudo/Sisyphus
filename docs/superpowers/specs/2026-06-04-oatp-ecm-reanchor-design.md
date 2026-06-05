# OATP1B1 Re-Anchor (post-FLUX-1) + optional +BSA PSu,inf — Design

**Date:** 2026-06-04
**Status:** IMPLEMENTED (branch `fix/rbp-concentration-basis`); anchor decision resolved → pitavastatin.
**Depends on:** FLUX-1 (`2026-06-03-flux1-extraction-double-count-design.md`) — this closes the FLUX-1-deferred xfail.
**Branch:** `fix/rbp-concentration-basis`

**Outcome (2026-06-04).** Anchor = **pitavastatin** (§3 recommended). Sweep (`realize_means`, post-FLUX-1+RBP-2):
pitavastatin |ln FE| optimum = **1.3e5** (FE 1.000); `reference_man.yaml` OATP1B1 abundance re-anchored
**5.0e5 → 1.3e5**. **pravastatin (holdout) is now validation-only** and improves to **FE 1.40** (was 2.28
@ 5e5) as a clean consequence — its test gate relaxed from the holdout-tuned **1.6 → 3.0** (general bar).
**Headline-neutral**: isolated on one stack, Meta **2.625 → 2.633 (+0.008)** — confirms the spec's ΔMeta≈0
(the −0.15 vs the committed 2.784 was macOS-py3.13 vs CI-py3.10-lock **stack drift**, not the re-anchor;
the committed CI cache is NOT regenerated from this non-canonical stack — CLAUDE.md developer-state trap —
and a CI regen lands within the cache-pin ±0.020 tolerance, FLUX-1 pattern). `test_oatp_ecm_statins`
[pravastatin, pitavastatin] **un-xfailed → PASS** (strict); rosuvastatin/atorvastatin stay xfail (Peff);
fluvastatin stays xfail (issue #21, OATP not rate-limiting). Invariant #5 **un-eroded** (abundance no
longer tuned to a holdout drug). +BSA PSu,inf was NOT adopted (minimal clean re-anchor; deferred).

## 1. Problem — two coupled defects in the OATP/ECM path

1. **Invariant-#5 erosion (pre-existing, audit-flagged).** The hepatic OATP1B1 uptake abundance was
   calibrated on **pravastatin**, which is in the **holdout** (`oatp1b1.json` notes: "abundance
   constant … calibrated on pravastatin in Phase 1"; `test_oatp_ecm_statins` docstring: pravastatin
   40 mg / 0.045 mg/L is the T7 anchor). Tuning a production constant to a holdout drug's Cmax violates
   "holdout is inviolable." The post-FLUX-1 audit confirmed pravastatin's ECM path **activates in the
   production benchmark**, so 1/107 holdout drugs is currently scored through a constant fitted to its
   own observation (near-AAFE-neutral, but a real erosion).
2. **FLUX-1 broke the calibration.** FLUX-1's intrinsic-clearance fix increased ECM hepatic extraction
   (`extended` model: dropped the `Q`-wrap), so the OATP1B1 abundance — fitted against the OLD
   flow-double-counted wrap — now **over-extracts**. `test_oatp_ecm_statins[pravastatin, pitavastatin]`
   were `xfail(strict=False)` deferred by FLUX-1 (`_FLUX1_ECM_RECAL_FAILS`).

The fix must re-anchor the abundance to the corrected physics **without** using a holdout drug as the
anchor — resolving both at once.

## 2. The accuracy lever this rides on (external evidence)

Deep-research 2026-06-04 ([[external-pbpk-benchmark-bar]]) found the **one concrete accuracy lever**:
+BSA / extended-clearance **albumin-mediated uptake** — adding 4% BSA to hepatocyte uptake assays
raises measured unbound uptake PSu,inf, and direct extrapolation from PSu,inf(+BSA) reconciles in-vitro
with in-vivo human hepatic CL at **~1.9–2.0 fold across all 16 OATP substrates with no empirical
scaling factor** (Li/Benet 2020 AAPS J, PMID 33196949). Sisyphus's ECM already has the machinery
(`hepatic_ecm.json` PS_passive/PS_eff/CL_int_bile + `oatp1b1.json` Jmax/Km saturable uptake → ps_inf in
`flux.py:304-347`). So this is a parameterization upgrade, not new architecture. (Mechanism — albumin-
mediated vs in-vitro-binding artifact — is still debated; the ~2-fold empirical result is accepted by
both camps.)

## 3. ⚠ OPEN DECISION — the re-anchor substrate

The anchor **must be non-holdout** (Invariant #5). Candidates and trade-offs:

| Anchor | non-holdout? | OATP-rate-limited? | confound |
|---|---|---|---|
| **pitavastatin** (recommended) | yes | yes (clean OATP substrate) | currently xfailed *only* for the FLUX-1 recal (not Peff) — it is the natural anchor |
| rosuvastatin | yes | yes | Peff XGBoost over-predicts absorption (`_KNOWN_PEFF_FAILS`) → confounds the Cmax anchor |
| valsartan / glimepiride | yes | yes (non-statin OATP, Amendment v2 verified Km/Jmax) | smaller clinical anchor base; different chemotype |
| ~~pravastatin~~ | **NO (holdout)** | — | the current (invalid) anchor |

**Recommendation:** anchor on **pitavastatin** (non-holdout, OATP-rate-limited, verified Niemi-2009 Km),
optionally cross-checked against valsartan/glimepiride. Then **pravastatin becomes a *validation*
read-out** (its FE should improve as a *consequence*, never as the target) — which both un-erodes
Invariant #5 and demonstrates generalization. Confirm before implementing.

## 4. Fix

1. **Re-anchor** the OATP1B1 abundance constant so that, under the FLUX-1-corrected `extended` flux,
   the chosen non-holdout anchor hits its FDA-label Cmax (calibration via the existing
   `scripts/calibrate_oatp_abundance_ecm.py`, retargeted off pravastatin).
2. **Optional (recommended) +BSA PSu,inf:** replace the seed PS_passive/PS_eff midpoints in
   `hepatic_ecm.json` with the +BSA-derived PSu,inf parameterization for the OATP substrate set, per
   Li/Benet 2020, so the abundance constant absorbs *less* residual bias (more mechanistic, less
   fitted). If adopted, re-derive the abundance after the PS update.
3. Keep `ecm_applicable` gating unchanged (pravastatin-only auto-activation in production stays; this
   spec changes the *constant*, not the routing).

## 5. Validation

1. **Un-xfail** `test_oatp_ecm_statins[pravastatin, pitavastatin]` (remove from `_FLUX1_ECM_RECAL_FAILS`);
   they must PASS their FE gates under the re-anchored constant. (`rosuvastatin`/`atorvastatin`/
   `fluvastatin` stay xfailed — documented Peff/applicability reasons, unchanged.)
2. `test_predict_auto_ecm` (the auto-ECM pravastatin path) un-xfailed and green.
3. **Holdout regen:** re-run `run_engine_benchmark.py`. Acceptance: the headline is **bit-identical for
   all non-ECM drugs** (ECM activates for pravastatin only in production); report pravastatin's Cmax
   shift. Because only 1 holdout drug's ECM path is touched and it is near-AAFE-neutral, the Meta 2.784
   headline should be unchanged to 3 sig figs — verify, don't assume.
4. Schema regression test (`test_oatp_registry_schema`) still green.
5. Document: experiment-log entry; note the Invariant-#5 un-erosion explicitly.

## 6. Risks

- **Anchor generalization:** a constant fitted to pitavastatin may not hold pravastatin/rosuvastatin
  simultaneously (the historical pravastatin↔fluvastatin tension, issue #21). Accept that the gate is
  "OATP-rate-limited statins," not all statins; fluvastatin stays excluded (CYP2C9-dominant).
- **+BSA scope creep:** the +BSA PS values exist for the OATP-substrate set; do not extrapolate to
  non-OATP drugs. Keep `ecm_applicable` the gate.
- Near-zero headline movement is the *expected* outcome (ECM is niche in production) — this is a
  **correctness + test-debt** fix, not a headline lever. Frame it honestly.

## 7. Hard-constraint check

- #1 identity-blind — engine unchanged; this edits data (`oatp1b1.json`/`hepatic_ecm.json`) + a
  calibration script + test markers. ✓
- #5 holdout inviolable — **the entire point**: re-anchor OFF the holdout (pravastatin) ONTO a
  non-holdout substrate; pravastatin demoted to validation-only. ✓ (strictly improved)
- #6 no drug-specific branches — a single global abundance constant + per-substrate literature PS
  values (data, not code branches). ✓
- #8 hard no-touch — no engine/compiler/solver edits; abundance/PS are physiological/literature
  anchors, not Cmax-loss fudge (the calibration targets a non-holdout FDA-label Cmax, the same class of
  physiological anchor FLUX-1's gut re-anchor used). ✓
