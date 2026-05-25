---
last_updated: 2026-05-25
parent: ../../CLAUDE.md
charter: Deferred work items with effort/value/risk tags. Promote to a spec cycle, GitHub Issue, dead-ends.md, or experiment-log.md when the disposition is decided. Items here have NOT been triaged through brainstorming yet — they are candidates, not commitments.
---

# Backlog

Reverse-chronological by **when the item was identified**, grouped by tier. Tier reflects effort × value × risk, not priority — priority is decided at session start.

## How to use

1. When a new candidate surfaces, add it as `B-NN` with the next free id. Include effort estimate, value rationale, risk class, and any dependency.
2. When work begins, **promote**:
   - **Spec cycle** (`docs/superpowers/specs/...`) for non-trivial implementations.
   - **GitHub Issue** when external discussion or async tracking is needed.
   - **Dead-ends entry** (`dead-ends.md` DE-NN) if disproven.
   - **Experiment-log entry** (`experiment-log.md`) when shipped.
3. Once promoted, **delete the backlog entry** (don't leave duplicates). The promoted document is the new SSOT.
4. If priorities change such that an item is no longer expected to ship, move it to a `## Won't Do` section with a one-line reason — don't delete (preserve the audit trail).

## Tier 1 — Substantive accuracy levers (large, high-value, requires spec cycle)

### ~~B-11~~ — Hepatic intracellular fu correction (Phase A shipped, Phase B closed as DE-37 2026-05-22)

**Status:** Closed. Phase A infrastructure shipped to main 2026-05-21 (commit `a0c90f8`, no headline shift by design). Phase B literature search closed as **DE-37** 2026-05-22 — primary corpus (Watanabe 2009 / Yamazaki 2010 / Riccardi 2017 / Patilea-Vrana 2017) is paywall-only via public web tools; 0 of 4 PPB candidates yielded a usable `fu_inc/fu_p` ratio. 19 audit rows kept as documentation trail (4 ceiling_accepted, 15 not_applicable). All entries `mean=1.0` → 107-holdout cache bit-identical post-Phase-B vs post-Phase-A (Meta AAFE 2.7715238009, delta 0.0).

**Future-iteration unlock:** subscription access to one of the 4 primary papers OR an independent hepatocyte-uptake assay providing `fu_inc/fu_p` for ≥1 of {paroxetine, oxybutynin, abiraterone, progesterone, similar high-PPB CYP-substrate}. Infrastructure is ready to receive new entries; loader anti-fudge guard still enforces `>= 1.0`.

**Spec:** `docs/superpowers/specs/2026-05-21-B11-hepatic-fu-correction-design.md`
**Plan:** `docs/superpowers/plans/2026-05-21-B11-hepatic-fu-correction.md`
**Phase B curation log:** `docs/superpowers/specs/2026-05-22-B11-Phase-B-curation-log.md`
**Dead-end entry:** `docs/claude/dead-ends.md` §DE-37

### B-01 — DE-33 / Jmax-PS architectural recalibration
**Effort**: 1+ day (full spec cycle). **Value**: highest ceiling lever in the project (could move Meta AAFE materially on OATP1B1 non-statin substrates). **Risk**: HIGH — per dead-ends.md, ECM tunes have a strong prior of breaking statin balance (DE-08–DE-18 error-cancellation family).

**Blocker**: N50 secondary holdout is currently frozen (per `project_n50_curation.md` memory). N=2 (valsartan + glimepiride) from the V3 generalization test isn't enough to falsify per `cherry_picking_audit_2026-04-22.md`. **Requires N50 unfreeze + 3–5 additional OATP1B1 non-statin substrate curation** before the spec cycle can responsibly run.

**Trigger to revisit**: when N50 is unfrozen for any reason, fold this in.

### B-02 — UGT path Phase 2 (public-clone reproducible)
**Effort**: 1–2 days. **Value**: Engine AAFE −0.029 demonstrated (DE-36 in `dead-ends.md`), Meta currently neutral via error cancellation. Phase 2 = activate in production AND make the gain reproducible on a fresh clone.

**Two threads, either of which would unlock the path:**

1. **Public UGT substrate registry build** — curate `data/enzymes/ugt2b7_substrates.json` (and ugt1a9, ugt1a4) following the v0.3.2 NAT2 / UGT1A1 pattern (`data/enzymes/{nat2,ugt1a1}_substrates.json`). Seed list: morphine, codeine, dapagliflozin, etodolac, ketorolac, metronidazole, indomethacin, bexagliflozin, glasdegib (the 9 drugs DE-36 showed Engine FE improvement on). Each entry requires a literature anchor — Niemi/Lautens UGT substrate reviews + DrugMetabRev.
2. **Meta-learner re-evaluation with error-decorrelation gate** — per `diagnosis.md` §4. If the registry build is real, re-train meta weights with the UGT path active and verify the gain isn't cancelled.

**Trigger to revisit**: capability completeness becomes a priority, or someone wants a real DrugBank-free reproducibility story.

## Tier 3 — Small items (trivial effort, narrow value)

### B-08 — DrugBank-free reproducibility alternatives
**Effort**: variable. **Value**: closes the audit-cycle reproducibility gap (PR #43) more thoroughly than artifact gates alone. Three alternatives:

1. **DrugBank substrate stub** — extract ONLY the substrate annotations + fup + pKa + logP for the ~50 most-cited drugs as a synthetic stub. Distribute as `data/drugbank_stub.json` (transformative use, not bulk redistribution). 4–6h.
2. **Per-property literature registries** — replicate B-02 pattern for fup overrides, pKa overrides, etc. Bigger curation effort but cleaner schema.
3. **Document DrugBank as optional power-user feature** — explicitly accept the +2.7% local-vs-CI gap. Cheapest, but doesn't move the public headline.

Pick one when there's actual external interest in reproducing the headline.

### B-09 — numpy 2.x + rdkit 2026.x migration audit
**Effort**: 1+ day. **Value**: forward-compatibility with current ecosystem; numpy 1.26 is LTS until 2027 but rdkit 2022.09 is several years stale. **Risk**: medium — version migrations historically shift Cmax (PR #42 hypothesis disproven only because the drift driver was elsewhere, but newer libs WILL produce some drift). Headline regen required.

**Deferred from**: PR #42 close note ("numpy 2.x migration deserves its own spec cycle").

### B-12 — GitHub Actions Node.js 20 → 24 migration (hard deadline 2026-09-16)
**Effort**: 1–2 h (action pin bumps + CI dry-run). **Value**: keeps CI green past the GitHub-runner Node 20 removal. **Risk**: low — pure infra, action maintainers handle the Node 24 compat internally. **Deadline**: hard — Node 20 binary removed from runner 2026-09-16; soft transition 2026-06-02 (runner default → Node 24, env opt-out exists until 09-16).

**Source**: CI run `26415053877` (2026-05-25) annotation: *"Node.js 20 actions are deprecated. … `actions/checkout@v4`, `actions/setup-python@v5` … forced to Node 24 starting June 2nd, 2026. Node 20 removed September 16th, 2026."*

**Action**:
1. Around 2026-06-15 (2 weeks after soft transition), check for newer Node-24-native action versions (`actions/checkout@v5`, `actions/setup-python@v6` or equivalent).
2. Bump pins in `.github/workflows/ci.yml`. Verify CI still green.
3. If newer versions not yet released, set `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` env in workflow as interim measure.

**Synergy**: natural pairing with [[B-09]] (numpy 2.x + rdkit migration) — both are "ecosystem version refresh" infra work. If bundling, [[B-09]] is the headline (Cmax-affecting); B-12 is the freebie addendum.

### ~~B-10~~ — Pitavastatin/rosuvastatin/atorvastatin metabolic_fraction curation (closed 2026-05-25)

**Status:** Closed. Shipped in 2026-05-24 doctrine completion sprint Phase A. atorvastatin + rosuvastatin promoted with literature-curated metabolic_fraction entries (Kantola 1998 / Martin 2003 anchors); ecm_applicable=true flipped. v0.3 ECM doctrine complete for all 4 statin substrates. Pitavastatin was already promoted in v0.3.1 (PR #30) — README §463 was stale.

**Spec:** `docs/superpowers/specs/2026-05-24-doctrine-completion-sprint-design.md`
**Plan:** `docs/superpowers/plans/2026-05-24-doctrine-completion-sprint.md`
**Commit:** `1cd6ff1`

---

## Won't Do (preserved for audit trail)

*(empty — no items moved here yet)*
