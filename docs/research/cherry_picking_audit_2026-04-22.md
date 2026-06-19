---
last_updated: 2026-04-22
author: the author
type: audit
charter: Quantitative retrospective assessment of cherry-picking risk in Sisyphus
---

# Cherry-Picking Risk Audit — 2026-04-22

**Summary:** Aggregate risk score **4.65/10** (moderate). Dominant risk factor is
holdout exposure count (§3, score 7/10): ~47 distinct algorithmic configurations
evaluated against the same N=107 since 2026-03-23. Mitigating: 4 of 5 production
weight changes were LOOCV-validated (unbiased estimator), prospective AAFE is
*better* than retrospective (wrong direction for overfit), and dead-ends
documentation is comprehensive. The key honest admission: LOOCV on N=107 is still
feedback from the same 107 drugs; the "no touch" guarantee applies to explicit
optimization but not to implicit signal propagation.

---

## §1. Method Routing Overrides (`data/sbi/method_routing.json`)

### What the file contains

13 drugs in the routing table (as of 2026-04-19, production P6):

| Method  | Count | Notes |
|---------|-------|-------|
| SBI     | 12    | includes morphine with `sbi_reweight=True` |
| IBIS    | 1     | pravastatin (coverage hard fail) |
| IS      | 0     | was 1 (morphine) from 2026-04-14 to 2026-04-19; retired |

### Git history of the file

| Commit | Date | Change |
|--------|------|--------|
| `cad7526` | 2026-04-10 | File created: 11 SBI / 1 IS / 1 IBIS (initial Track B routing) |
| `0673817` | 2026-04-12 | Phase 2.0.5 promotion: 12 SBI / 1 IS / 1 IBIS (diclofenac + posaconazole recovered) |
| `02808a3` | 2026-04-14 | Morphine override SBI → IS (TDM bias: SBI +52% vs IS +3%) |
| `06d24a8` | 2026-04-19 | Retire morphine IS override → SBI + `sbi_reweight=True` |

### Override classification

All routing decisions are gated by the **Simulation-Based Calibration (SBC) coverage
test**, not by holdout AAFE:

- **Hard gate**: `coverage_max_deviation` compared to thresholds (≤10pp → SBI,
  ≤15pp → borderline, >15pp → IBIS). These thresholds were pre-specified.
- **Morphine `sbi_reweight=True`**: set by TDM *bias* comparison (the posterior point
  estimate quality: +52.3% SBI vs +2.1% after reweighting), not by Cmax AAFE on
  the 107-holdout benchmark.
- **Pravastatin IBIS**: `coverage_max_deviation = 0.223`, mechanistically explained
  by OATP1B1 transporter OOD (training set had 0 OATP substrates before Phase 2.0.5).

**Critical distinction:** the routing table controls **TDM inference** (per-patient
Bayesian update). It does **not** affect the population-level Cmax AAFE on the
107-holdout benchmark. The benchmark runs via `pipeline/predict.py` which is
routing-table-agnostic.

### Per-drug override assessment

One historical concern: morphine was forced IS (2026-04-14) based on TDM accuracy,
then reverted to SBI+reweight (2026-04-19) when a principled reweighting method was
found. This is NOT cherry-picking in the Cmax-AAFE sense — it is TDM quality
improvement for a single patient-facing clinical use case.

**§1 verdict: LOW RISK (score 2/10).** Routing is mechanistically gated by SBC
coverage, independent of the holdout AAFE benchmark.

---

## §2. Track Weight Evolution

### Current production weights (committed 2026-04-05, unchanged since)

```python
_W_ENGINE_BASE  = 0.60   # oral base drugs
_W_ML_BASE      = 0.40
_W_CLF_BASE     = 0.00
_W_ENGINE_OTHER = 0.35   # all other compound types
_W_ML_OTHER     = 0.50
_W_CLF_OTHER    = 0.15
_W_VDSS         = 0.20   # scales all three 3-track weights by 0.80 when VDss available
```

### Full weight change timeline

| Date | Commit | Configuration | Validation method | Holdout-driven? |
|------|--------|---------------|-------------------|-----------------|
| 2026-03-23 | `e5e8319` | `w_engine=0.17, w_ml=0.83` (initial) | Direct N=38 holdout | **Yes** — explicitly calibrated against AAFE target ≤2.5 |
| 2026-03-24 | `935d789` | Adaptive `base=0.60, other=0.00` | LOOCV N=51 | Indirect |
| 2026-03-24 | `ea931a0` | `base=0.65, other=0.00` | LOOCV-B N=51 (94% fold stability) | Indirect |
| 2026-03-26 | `6af1779` | `base=0.45, other=0.00` | LOOCV N=107 (82% stability) | Indirect |
| 2026-04-04 | `5e5a3d0` | ML retrained (leakage fix), no weight change | Holdout retest | **Yes** — this was a correctness fix, not optimization |
| 2026-04-05 | `201f37e` | `_W_VDSS=0.20` added | LOOCV N=107 (100% stability) | Indirect |

**Classification of validation methods:**

- **Direct holdout**: 1 weight change (initial 2026-03-23). Explicit statement: "calibrated at 0.17 based on holdout validation." This is the cleanest cherry-pick: a free parameter set to minimize AAFE on the same set used for reporting.
- **LOOCV (indirect)**: 4 weight changes. LOOCV on N=107 is theoretically unbiased — each drug appears as test exactly once, never used for its own weight selection. The caveat: grid-search over LOOCV chooses the best of ~100 grid cells; this introduces an expected bias of ~0.04–0.07 log10 AAFE units on the test set.
- **Correctness fix**: 1 (contamination fix — not optimization).

**Weight regime count:** 2 distinct regimes (base vs other), 1 VDss scalar. Grid-searched over O(10²) combinations via LOOCV.

**§2 verdict: MODERATE RISK (score 5/10).** LOOCV is the principled choice and 4/5
changes were LOOCV-validated. However: (a) the initial calibration was direct
holdout, and (b) LOOCV on N=107 still exposes the holdout drugs as implicit signal
via repeated grid search. The weight space (2D × ~10 grid values + VDss scalar) is
small enough that overfit is bounded.

---

## §3. Holdout Intervention Count

### Holdout freeze date

`data/reference/holdout.json` first appeared in commit `beca502` (2026-03-23).
The dataset was expanded N=61 → N=107 in commit `6af1779` (2026-03-26). The N=107
set has been frozen since 2026-03-26 (approximately 27 days as of this audit).

### Commits mentioning AAFE or holdout since freeze

```
$ git log --all --oneline --grep="AAFE|holdout|Meta|benchmark" | wc -l
128
```

128 commits. Note: many are documentation, validation artifacts, and retests of the
same configuration. The more meaningful count is **distinct algorithmic
configurations evaluated**:

| Category | Count |
|----------|-------|
| Dead-end experiments (reverted or negative result) | 32 |
| Production weight/parameter changes | 6 |
| Reference data corrections and enrichments | 5 |
| SBI routing decisions (TDM, not AAFE-driven) | 4 |
| **Total distinct configurations tested against holdout** | **~47** |

### Experiments in experiment-log.md that report Meta AAFE

The experiment log contains 25 experiment sections, 12 distinct AAFE values
(ranging from 2.043 to 3.074). Every AAFE value represents a holdout probe.

### Statistical implication of 47 configurations

Given:
- N=107 holdout, SD(log10 fold error) = 0.378, SE = 0.037 log10 AAFE
- Expected best-of-K overfit = SE × √(2 · ln K)

| Scenario | K | Expected AAFE inflation |
|----------|---|------------------------|
| All 47 configs independent, best selected | 47 | 0.102 log10 units = **1.27×** |
| 8 production-changing configs only | 8 | 0.075 log10 units = **1.19×** |
| LOOCV-validated changes | — | ~0 (theoretically unbiased) |

**Implied true AAFE range:**
- If LOOCV is fully valid: reported 2.695 ≈ true AAFE
- If 8 production changes behave as direct tuning: true AAFE ≈ 3.20
- If all 47 configs contributed independently: true AAFE ≈ 3.41 (upper bound)
- **Best estimate: 2.85–3.10**, assuming LOOCV has residual grid-search bias

This is the **most concerning quantitative finding** of this audit: if LOOCV
validation is not perfectly unbiased (grid search over LOOCV objectives still
selects based on N=107 distribution), the reported headline AAFE 2.695 could
overstate performance by 10–25%.

**§3 verdict: HIGH RISK (score 7/10).** N=107 is small for 47+ configuration
tests. Statistical inflation is real and bounded ~1.19–1.27× under worst-case
direct-tuning assumptions, ~1.05× under LOOCV-validity assumptions. Prospective
validation partially mitigates (§5).

---

## §4. Dead-Ends Documentation Discipline

### Enumerated dead-ends: 32 entries across 11 categories

| Category | Dead-ends | Representative entries |
|----------|-----------|----------------------|
| Post-hoc meta-learner variants | 5 | DE-23, DE-24, DE-25, DE-26, DE-27 |
| CLint R² improvement | 6 | DE-08, DE-11, DE-13, DE-16, DE-17, DE-20 |
| Full/partial ADME replacement | 4 | DE-01, DE-21, DE-31, DE-03 |
| Foundation models / E2E Neural PK | 3 | DE-05, DE-14, DE-19 |
| UDE / gradient-through-solver | 2 | DE-29, DE-30 |
| Training data expansion | 3 | DE-09, DE-11, DE-16 |
| Bioavailability / CL / t½ predictors | 4 | DE-15, DE-26, DE-27, DE-28, DE-29 |
| CLint features (docking, BDE) | 2 | DE-13, DE-18 |
| Kp correction methods | 2 | DE-09, DE-10 |
| SBI-specific | 1 | DE-32 |
| Reference-level overrides | 1 | DE-31 |

### Formal revert vs silent documentation

Of 32 dead-ends:
- **8 explicitly noted as "Revert"** in their dead-ends entry (DE-04, DE-08, DE-09,
  DE-11, DE-12, DE-16, DE-31, DE-32)
- **24 documented as negative results without formal git revert** — these were mostly
  branch-only explorations never merged to main, consistent with the `eval(...)` commit
  prefix convention

### Gaps in documentation

1. **Pre-leakage contamination window**: 6+ commits between the leakage introduction
   and its discovery (2026-04-04). During this window, AAFE 2.283 was reported as
   valid; all experiments in that window were evaluated against contaminated metrics.
   This is an honesty-gap, though it was self-discovered and publicly documented.

2. **Informal DE numbering inconsistency**: Commit messages and docs reference "#35
   error cancellation" but the canonical table has 32 entries. The discrepancy is
   explained (informal counting includes pre-Sisyphus Omega attempts) but is
   confusing to auditors.

3. **Missing numeric outcomes in early entries**: DE-01, DE-02, DE-03 lack explicit
   ΔAAFE numbers. All others have numeric outcomes.

4. **"Accepted" experiments not symmetrically documented**: The dead-ends list covers
   failures comprehensively, but there is no symmetric "successes" register (only the
   experiment log). A companion "what-worked" list would strengthen the audit trail.

### Success/failure ratio

Across the project timeline:
- **Successes (retained in production and improved holdout AAFE)**: VDss 4th track
  (−0.113 AAFE), adaptive weights (−0.31 AAFE from initial), contamination fix
  (correctness). Approximately 3 genuine improvements.
- **Failures (reverted or w=0.00)**: 32 documented. All SBI routing except
  production defaults ultimately landed mechanistically.

**Discipline score: 3/10** (low = better discipline). Documentation is
comprehensive and consistently maintained. The pre-leakage window and informal
numbering are minor gaps. No evidence of silently-dropped experiments that escaped
documentation.

---

## §5. Prospective vs Retrospective AAFE

### Source data verification

**Retrospective (N=107 holdout):** Computed from `data/training/4track_holdout_predictions.json`:
- Stored AAFE = 2.6946 (matches computed from raw folds to 4 decimal places ✓)
- 95% CI: [2.285, 3.178] (from SE = 0.037 log10 units)

**Prospective (N=15, FDA 2024-25 NMEs):** Computed from `data/validation/prospective_N15_4track.json`:
- Stored AAFE = 2.3613 (matches computed from raw folds ✓)
- In-domain AAFE = 2.0434 (N=13)
- 95% CI: [1.615, 3.451] (wide, SE = 0.084 log10 units on N=15)

**Prospective/Retrospective ratio: 0.876** (prospective is BETTER than retrospective)

### Direction analysis

For an overfit system, the expected pattern is: prospective AAFE > retrospective AAFE
(new drugs reveal the overfit). The observed pattern is the *opposite*: prospective
< retrospective. This is a reassuring signal, but:

1. **N=15 has very wide CIs** (95% CI range from 1.615 to 3.451). The
   prospective AAFE 2.361 is statistically indistinguishable from the holdout 2.695
   (overlap of 95% CIs is substantial).

2. **Prospective set methodology is honest but imperfect**: the original N=6→N=10
   cohort was subsequently found to have selection bias toward "PK-easy" drugs
   (commit `2a7a281` retracted an N=7 contaminated claim; `39b1de5` expanded to N=15
   and acknowledged selection bias, with resulting AAFE increase from 1.959→2.361).
   This self-correction is a genuine mark of rigor.

3. **No prospective-specific pipeline tuning found**: Search of `src/` for
   "prospective" / "NME" / "FDA NME" returns zero hits. The prospective drugs were
   run through the same production pipeline without modification. The VDss track
   weight `_W_VDSS=0.20` was selected via LOOCV on the 107-holdout, not the
   prospective set (confirmed by commit `201f37e` methodology).

4. **Prospective set timing**: N=15 was available from 2026-04-05 onward. The class-
   aware meta-benchmark script (`DE-27`) did sweep the prospective set alongside
   holdout (see `scripts/class_aware_meta_benchmark.py:129-189`), but DE-27 was a
   negative result that changed no production weights. No evidence that the
   prospective data was used to *select* production configurations.

**§5 verdict: LOW RISK (score 2/10).** Favorable direction, honest self-correction
on selection bias, no prospective-specific tuning detected. However, N=15 is
statistically insufficient to be confident — the true prospective AAFE 95% CI
spans from 1.62 to 3.45.

---

## §6. Quantitative Cherry-Picking Score

### Factor decomposition

| Factor | Score | Weight | Contribution | Notes |
|--------|-------|--------|-------------|-------|
| §1 Per-drug routing overrides | 2/10 | 15% | 0.30 | Gated by SBC coverage, not AAFE |
| §2 Track weight evolution | 5/10 | 25% | 1.25 | LOOCV principled but implicit holdout feedback |
| §3 Holdout exposure count | 7/10 | 35% | 2.45 | ~47 configs, N=107 small; statistical inflation 1.19–1.27× |
| §4 Dead-ends discipline | 3/10 | 15% | 0.45 | Comprehensive docs; 3 gaps; early contamination miss |
| §5 Prospective/retrospective | 2/10 | 10% | 0.20 | Favorable direction; N=15 underpowered |

**Aggregate: 4.65/10** — moderate cherry-picking risk.

### What the score means

The score of 4.65 sits at the high end of "moderate." It does not indicate
deliberate fraud or intentional data manipulation. It indicates a system where:

- The same N=107 holdout has been examined ~47 times with different configurations,
  creating **implicit selection pressure** even when explicit optimization was avoided.
- The LOOCV protocol partially breaks this linkage but cannot eliminate it entirely.
- The system's reported AAFE 2.695 is most defensibly interpreted as an **optimistic
  estimate** of true prospective performance, potentially by 0.10–0.20 AAFE units.

**True AAFE estimate (best guess): 2.85–3.10**, with 2.695 as the lower bound
(valid only if LOOCV is fully unbiased) and 3.41 as the upper bound (valid only
if all 47 configs were independent direct-tuning attempts).

---

## §7. Recommendations

### Immediate (score = 4.65, moderate)

**1. Hold out a secondary validation set permanently.**
The 107-drug holdout has been used for feedback ~47 times. A secondary "never-touch"
set of 30–50 drugs should be created, drawn from different sources (e.g., EUDRAMED,
published EU clinical studies), and used for first-pass prospective validation only —
never for weight selection, routing decisions, or experiment feedback.

**2. Pre-register all future experiments.**
Before any experiment that could change production weights or routing, write a
pre-registration document (spec file) that specifies:
- Hypothesis and expected direction
- Success/failure criteria with pre-specified threshold
- Which data set will be used for evaluation
This is already partially in place for TDM work (P6, P7 decision packages). Extend
it to all Cmax-pipeline changes.

**3. Report AAFE with 95% CI in all claims.**
Replace "Meta AAFE 2.695" with "Meta AAFE 2.695 [95% CI: 2.29–3.18, N=107]" in
the project README and papers. This forces honest communication of the statistical
precision of the holdout benchmark.

**4. Distinguish "LOOCV-validated" from "holdout-validated" in commit messages.**
Track weight changes validated by LOOCV should be tagged `LOOCV-validated` and
those from direct holdout `holdout-tuned`. The audit record shows only 1 of 6
weight changes was direct holdout tuning (the initial setup), but this distinction
was not consistently visible in commit messages.

### Confirmatory checks (score ≤ 5 is not automatic clearance)

**1. The prospective N=15 favorable direction is reassuring but statistically
underpowered.** The 95% CI [1.62, 3.45] spans the entire range of concern. A
prospective N of 50+ would be needed for a confident statement.

**2. Verify LOOCV stability is not inflated by the grid-search procedure.**
The LOOCV scripts (e.g., `scripts/run_loocv_validation.py`) grid-search over
(w_base, w_other) and report the *best* LOOCV AAFE. This is not fully unbiased —
the best-of-K grid cells inflates the LOOCV estimate by ~1/sqrt(N_holdout) ×
log(K_grid). For a 10×10 grid and N=107: expected bias ≈ 0.020 log10 units.
This is small but nonzero.

**3. The "error cancellation defense" is a real mechanism, not a rationalization.**
The measured ADME PoC (Pattern C, 2026-03-26) directly tested the error cancellation
hypothesis: with measured ADME, AAFE improved from 2.329 → 1.980 on N=12. This
confirms the pipeline architecture is sound. The ceiling is not artificially low from
cherry-picking — it reflects real CLint prediction noise (R²=0.24 intrinsic to
hepatocyte assay). However, the *exact level* of the ceiling (2.695 vs, say, 3.10)
remains uncertain due to the holdout exposure problem.

---

## Appendix: Evidence Sources

| Claim | Source |
|-------|--------|
| Routing override count | `data/sbi/method_routing.json` (read directly) |
| Routing git history | `git log --follow -- data/sbi/method_routing.json` (4 commits) |
| Weight change timeline | `git log -- src/sisyphus/ml/ensemble.py` (12 commits) |
| LOOCV validation | `git show ea931a0`, `git show 201f37e`, `scripts/run_loocv_validation.py` |
| Dead-end count | `docs/research/dead-ends.md` (32 enumerated) |
| Holdout AAFE (verified) | `data/training/4track_holdout_predictions.json` (computed = 2.6946 ✓) |
| Prospective AAFE (verified) | `data/validation/prospective_N15_4track.json` (computed = 2.3613 ✓) |
| Contamination episode | `git show 5e5a3d0`, `docs/holdout_contamination_audit.md` |
| Prospective self-correction | `git show 2a7a281` (retraction), `git show 39b1de5` (expansion) |
| Error cancellation PoC | `docs/research/experiment-log.md` §Measured ADME PoC |

---

*Audit conducted by: the author, 2026-04-22. Read-only
investigation — no code, weights, or routing changed. All numbers independently
computed from raw data files and cross-checked against stored values.*
