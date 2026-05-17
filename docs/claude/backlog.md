---
last_updated: 2026-05-15
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

## Tier 2 — Capability extensions (moderate, single-issue closeout)

### B-04 — Multi-enzyme prodrug conversion schema (per-enzyme yield)
**Effort**: 4–6h (schema + builder + tests; engine already supports per-edge yield via `params.edge_param`). **Value**: unlocks B-03, prasugrel, ticagrelor, and other dual-fate prodrugs. **Risk**: low — backward-compatible (optional per-enzyme `yield` field with entry-level fallback); existing 6 entries bit-identical post-migration.

**Spec**: `docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md`.

**Prerequisite for B-03** (re-ordered 2026-05-17 after a structural-blocker analysis showed clopidogrel's CES1-dead-end + CYP2C19-active dual fate cannot be expressed with a single entry-level yield without violating mass balance or mechanistic-A doctrine; see experiment-log 2026-05-17 entry).

### B-03 — Clopidogrel (#11 잔여)
**Effort**: 2–3h **after B-04 lands**. **Value**: closes the remaining 1/3 of issue #11. **Risk**: medium — clopidogrel is a 107-holdout member, so the addition shifts headline AAFE and requires regen + delta documentation. Disposition expected **ceiling_accepted** per v3 mechanistic-A doctrine (R-130964 active thiol PK poorly characterized in primary literature; rapid covalent P2Y12 binding sink prevents conventional CL/Vd measurement).

**Implementation outline** (see §8 of the B-04 spec for the full path):
- Registry entry with per-enzyme yields: `CES1{yield=0}` (dead-end) + `CYP2C19{yield≈1}` (active fraction); overall systemic yield ~15% emerges from the combination, not a global scalar.
- `observation_species="parent"` — 107-holdout reference is parent clopidogrel Cmax; mismatch with "active" would inject a deliberate 5–20× species-mismatch fold error.
- `cyp_clearance_overrides.json` entry with `metabolic_fraction=0` to route 100% of hepatic CL through the two ProdrugActivationEdges (CES1 + CYP2C19), preventing double-count with XGBoost-derived CL.
- 107-holdout regen + bootstrap CIs refreshed + AAFE delta documented.

**Blocked by**: B-04.

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

### B-10 — Pitavastatin/rosuvastatin/atorvastatin metabolic_fraction curation
**Effort**: 4–6h. **Value**: complete the v0.3 ECM auto-activation rollout. Currently only pravastatin has a literature-anchored `metabolic_fraction` in `data/transporters/cyp_clearance_overrides.json`; pitavastatin was promoted in v0.3.1 (PR #30) with mf=0 mechanistic-only, rosuvastatin/atorvastatin not yet promoted. Each requires Niemi/Lautens-style fm curation per statin.

**Deferred from**: README §463 note ("Pitavastatin/rosuvastatin/atorvastatin remain unflagged pending literature-curated metabolic_fraction entries (deferred to v0.3.x follow-up commits)").

---

## Won't Do (preserved for audit trail)

*(empty — no items moved here yet)*
