---
last_updated: 2026-04-22
parent: ../../README.md
charter: Process document for cherry-picking reduction. Derived from cherry_picking_audit_2026-04-22.md §7. Binding for all future accuracy-changing work.
---

# Cherry-Picking Reduction Process V1

Derived from `docs/research/cherry_picking_audit_2026-04-22.md` §7. Active from `3e38210`.

The 107-holdout is exposed to 47+ config decisions (track weights, routing overrides, meta-learner choices). Audit aggregate score 4.65/10 (moderate). This document codifies four safeguards to stop the score from drifting higher.

---

## §1. Secondary Permanent Holdout Set ("N50")

**Purpose:** Break the holdout/training dual-role. The 107-drug holdout has been the *only* feedback signal for all track-weight and routing changes since 2026-04-05. Adding a secondary holdout that is genuinely never-touch gives us an unbiased estimator of true generalization.

**Design:**
- **Size:** 50 drugs (N50), curated before any accuracy-changing work resumes.
- **Sources:** FDA 2024–2026 NMEs (distinct from the 15 already used in prospective), EU EPAR labels, published Clin Pharmacokinet retrospective reviews from 2023–2025. Exclude any drug in the existing 107 or in `data/training/` MMPK/TDC/DrugBank.
- **Storage:** `data/reference/holdout_n50.json` (new file, schema mirrors `holdout.json`).
- **Read access:** Unrestricted (read is fine — the risk is *writing back* against the output).
- **Write access against this set:** Forbidden. No track weight tuning, no routing entries, no experiment-log AAFE tracking, no meta-learner retraining. One-time usage: at the end of a major release cycle, run the benchmark once, publish `N50 AAFE ± 95% CI` as the next release headline, freeze that result, and retire the set.
- **Replacement cadence:** Retire N50 after one measurement cycle. Curate a successor N50' for the next cycle.

**Constraint satisfaction:** N=50 gives bootstrap 95% CI ~ ±20% on AAFE (vs ±18% at N=107), so statistical precision is comparable while cherry-picking exposure is zero.

**Blocker:** Curation cost. Estimated ~4 hours of literature work per cycle. Worth it — without it, every future AAFE claim is contaminated.

**Action to take:** Before the next accuracy-changing commit touching `method_routing.json`, `meta_learner.py`, or track weights, curate N50 first. If this process cost is unacceptable, the answer is NOT to skip — it is to stop making accuracy-changing commits until someone has time to curate.

---

## §2. Pre-Registration Template

**Scope:** Any change that can affect `data/training/4track_holdout_predictions.json` (= the file backing the README headline). Examples: track weight edits, routing additions, meta-learner changes, new tracks, calibration constant edits.

**Required artifact:** A spec file at `docs/_internal/specs/YYYY-MM-DD-<change>-design.md` containing these sections BEFORE any code execution touches the holdout benchmark:

1. **Hypothesis** — what the change is expected to do in one sentence.
2. **Pre-specified direction** — which direction is "success" (e.g., "AAFE decrease by ≥ 2% on 107-holdout").
3. **Pre-specified failure threshold** — when to revert (e.g., "AAFE increase by any amount, OR 2-fold% decrease by > 1%").
4. **Evaluation set** — 107-holdout, N50, or both. If both, specify which is primary.
5. **Error decorrelation requirement** — for any NEW track proposal (not weight adjustment), pre-declare the maximum allowed Pearson r between the new track's residuals and existing meta residuals. Reject at > 0.90 before running holdout.
6. **Commit/revert plan** — what branch, who reviews, what commit message prefix.

**Discipline:** The spec must be committed before the experiment runs. Post-hoc rationalization of a positive AAFE swing is the failure mode we are preventing. If a spec is written after the engine has been run, the experiment is a **negative** by construction regardless of the numeric outcome.

**Existing partial compliance:** TDM work (P6, P7) and OATP ECM (2026-04-20) followed this pattern. Track-weight edits historically did not. This doc extends the discipline to all Cmax-pipeline changes.

---

## §3. 95% CI Reporting in All AAFE Claims

**Rule:** Every AAFE number in the project README, experiment-log.md, commit messages, PR descriptions, and external papers must be accompanied by a bootstrap 95% CI and the N.

**Bootstrap method:** 10,000 resamples with replacement on `abs(log10(fold))`, report `10^percentile(bootstrap, [2.5, 97.5])`.

**Format:** `AAFE X.XXX [95% CI: Y.YYY, Z.ZZZ, N=NN]`

**Current headline (computed 2026-04-22, commit `3e38210`):**

- Meta (production): **AAFE 2.695 [95% CI: 2.302, 3.197, N=107]**
- pct_2fold: 47.7%; pct_3fold: 65.4%

The CI upper bound (3.20) brackets the audit's true-AAFE estimate (2.85–3.10), meaning we cannot statistically reject the null hypothesis that retrospective tuning inflated the point estimate. This is the most important finding in this process doc — *the point estimate alone is not a defensible claim*.

**Action:** Update the README headline table to include CI column. Do this in the same commit as this process doc.

---

## §4. Routing-Change Decorrelation Gate

**Problem:** `data/sbi/method_routing.json` entries (per-drug SBI/IS/IBIS/override assignments) are a latent cherry-picking surface. Each entry is effectively an `if drug == X` branch (Invariant #6 violation in spirit, not letter, since it lives in a JSON).

**Gate requirements** for any new entry in `method_routing.json`:

1. **SBC pass** (existing — do not bypass).
2. **Not AAFE-gated** — do not add a routing entry "because it reduces holdout AAFE." The holdout is not the selection signal here; SBC is.
3. **Decorrelation declaration** — before committing the routing entry, compute the Pearson r between the proposed routing's residuals (on 107-holdout) and the current meta residuals. If r > 0.90, the new routing is not orthogonal — REJECT as cherry-picking regardless of AAFE delta.
4. **Effect reporting** — commit message must include: before/after 107-holdout AAFE with CI, SBC cov_dev, decorrelation r, N50 AAFE (if N50 exists at that cycle).

**Concrete procedure** (to be codified as `scripts/routing_decorrelation_gate.py` in a future task):

```
python3 scripts/routing_decorrelation_gate.py \
    --drug DRUG_NAME --proposed-method {SBI,IS,IBIS,override} \
    --report-only
```

Outputs a go/no-go verdict. Commit only on go.

**Retroactive:** The 13 existing entries (12 SBI + 1 IBIS) predate this gate. They are not audited retroactively — the gate is forward-looking only. If any of them is touched again, they must pass the gate on re-touch.

---

## §5. Commit-Tagging Discipline

**Rule:** Every commit whose diff touches the Cmax pipeline end-to-end (i.e., can change the holdout benchmark) must use one of these prefix tags in the commit message body:

- `[LOOCV-validated]` — the change was validated on leave-one-out cross-validation against 107-holdout with the held-out drug actually out. This is the preferred path.
- `[holdout-tuned]` — the change was selected by direct 107-holdout AAFE optimization. This is the high-cherry-picking path and must justify why LOOCV was not feasible.
- `[N50-gated]` — the change was evaluated on N50 (not the selection signal but the pre-committed direction was confirmed). Safest.
- `[no-AAFE-impact]` — the change demonstrably cannot affect `4track_holdout_predictions.json` (pure infrastructure, docs, tests). No evaluation needed.

A commit without one of these tags touching Cmax pipeline code will fail a future pre-commit hook (to be implemented).

**Example for today's V3 work:** the 7-commit V3 chain (`4630b0b..4e10ad2`) is `[no-AAFE-impact]` — V3 is route-conditional, 107-holdout is all oral, structurally byte-identical. The regression guard test pins this invariant.

---

## §6. Process Governance

**Authority:** This document is binding on all future Sisyphus commits that touch accuracy. Violations are reverts, not "we'll fix it next time."

**Review cadence:** Every 6 months or when a new cherry-picking audit is run. Next audit scheduled after N50 first measurement.

**Amendment:** Must be via a spec under `docs/_internal/specs/`. This document cannot be amended by direct edit except for the headline AAFE/CI numbers in §3, which are updated with each production benchmark re-run.

**Escape hatch:** None. A process that can be bypassed "in an emergency" is not a process. If a genuine emergency requires bypassing these gates, the bypass itself is a negative experiment and must be documented in `dead-ends.md` alongside the numerical outcome.

---

## Appendix: Action Items from This Doc

| # | Action | Owner | Timing |
|---|---|---|---|
| 1 | Update the README headline with 95% CI | the author | same commit as this doc |
| 2 | Curate N50 (50 drugs, new holdout file) | the author | before next accuracy commit |
| 3 | Write `scripts/routing_decorrelation_gate.py` | the author | when next routing entry is proposed |
| 4 | Pre-commit hook for tag enforcement | deferred | after N50 lands |
| 5 | First N50 benchmark run + CI | deferred | after N50 lands + cycle freeze |
