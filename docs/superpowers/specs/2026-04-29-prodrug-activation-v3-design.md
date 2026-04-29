# Prodrug Activation v3 — Input-Data Quality Refresh

**Date:** 2026-04-29
**Branch (planned):** `feat/prodrug-activation-v3` (branched from `main` after v2 PR #7 merge)
**Predecessor:** v2 spec `docs/superpowers/specs/2026-04-27-prodrug-activation-v2-design.md`, T1 literature deliverable `docs/superpowers/specs/2026-04-27-prodrug-v2-task1-literature.md`
**Successor of:** v1 spec `docs/superpowers/specs/2026-04-24-prodrug-activation-design.md` (merged 2026-04-26 `af9d2be`), v2 (PR #7, 2026-04-28)

---

## 1. Goal

v2 produced architecturally correct prodrug activation infrastructure (well-stirred extraction, identity-blind multi-site discovery, mass balance verified, DDI-free) but did not move 4-drug 3-fold validation gate. v1 → v2 fold-error change marginal because v2 retained v1 active species CL/Vd values (D1 deferred per attribution rationale) and physiology enzyme abundances remained class-extrapolated.

**v3 closes the input-data quality gap that v2 explicitly deferred.** Each T1 caution flag receives a per-item acceptance gate decision (literature_applied / interpretation_resolved / ceiling_accepted). Architecture from v2 is unchanged.

## 2. Success Criterion

Per-item acceptance gates apply across all 6 T1-flagged items. v3 success ≠ gate-pass; v3 success = each T1 flag explicitly resolved with documented disposition state.

| State | Meaning |
|---|---|
| `literature_applied` | Primary source found in exhaustive search (§4.3); doctrine (§4) applied; value updated |
| `interpretation_resolved` | Primary source exists but requires explicit interpretation decision (e.g., BH4 central-vs-apparent); decision made and documented; value updated |
| `ceiling_accepted` | Search exhaustive (§4.3) but no primary source meeting doctrine criteria; class-extrapolation retained; ceiling explicitly documented |

The 3-fold validation gate is an outcome, not a stopping criterion. Gate-pass for any drug results in xfail removal (§6.4); gate-fail with improvement is acceptable; gate-regression triggers red-flag review (§6.4). **Mechanistic-A promise (v2 spec §3.3) is preserved**: no clinical-fit knobs.

## 3. Scope

**Single combined v3 PR** containing all 6 T1-flagged items. Sub-items are coupled through shared drugs (sepiapterin ← BH4 CL/Vd + SPR abundance; tebipenem ← tebipenem CL/Vd + CES2/tebipenem affinity), making split PRs introduce sequencing complexity without reducing review burden.

### 3.1 Items

| # | Item | Affects drug | Type |
|---|---|---|---|
| 1 | BH4 active CL/Vd | sepiapterin | popPK literature |
| 2 | GS-441524 active CL/Vd | remdesivir | popPK literature |
| 3 | R406 active CL/Vd | fostamatinib | popPK literature |
| 4 | tebipenem active CL/Vd | tebipenem_pivoxil | popPK literature (T1: "mostly OK") |
| 5 | SPR primary proteomic abundance | sepiapterin | enzyme abundance (physiology) |
| 6 | CES2/tebipenem direct Vmax/Km | tebipenem_pivoxil | enzyme kinetics (drug affinity) |

### 3.2 Out of Scope (v3)

The following are NOT v3 work; they are v4 candidates or separate projects:

- CES1 abundance refresh (no T1 caution flag; class-extrapolated retained)
- ALPI abundance refresh (no T1 caution flag)
- Affinity refresh for non-T1-flagged drug-enzyme pairs (sepiapterin-SPR affinity, fostamatinib-ALPI affinity, etc.)
- New prodrug drug registry additions (4 drugs only; new drugs are v4+)
- 1-comp → 2-comp active species elimination model change (architectural)
- SBI model retraining (12 SBI drugs — staleness handled in §6, separate follow-up)
- PI coverage recalibration (29.9% baseline structural error dominated)
- Meta-learner gate evaluation (Engine gate is sole acceptance gate)
- DDI clinical scenarios beyond v2 smoke test (v4)

Opportunistic in-scope expansion (e.g., refreshing CES1 abundance because data is encountered during SPR search) is **prohibited** — violates (B-refined) discipline and muddles attribution.

## 4. Doctrines

### 4.1 Mean-Value Doctrine (popPK, items 1-4)

| Parameter | Rule |
|---|---|
| Vd | **Vss** (steady-state Vd) from 2-comp popPK; 1-comp Vd accepted as Vss-equivalent if model fit reasonable |
| CL | Total CL from popPK (1-comp or 2-comp; total CL is unambiguous across models) |
| Source drug | **Same chemical entity only** (salt forms accepted: sapropterin = BH4) |
| Route | Direct IV preferred. **Oral popPK accepted iff F is separable** (V/F → V via explicit F division) |
| F application | **Decoupled from conversion**: in PBPK the active species is generated intra-systemically, so oral F applies only to prodrug absorption; active-species V/F must be divided by F to recover central V |

**Same-entity strict interpretation** (Gap 1 closure): surrogate-species data (rat, monkey) is rejected — species extrapolation violates mechanistic-A. Same-entity-human-only or `ceiling_accepted`.

**Multi-source model conflict** (Gap 2 closure): when literature studies use different model dimensionality:
- Prefer 2-comp Vss
- Accept 1-comp Vd as Vss-equivalent if model fit reasonable, document
- If parallel 1-comp and 2-comp sources: geometric mean of point estimates; CV via §4.2 max() rule

**Non-canonical sources** (Gap 3 closure): primary source must be peer-reviewed journal article OR FDA/EMA review document OR bioRxiv preprint. Conference abstracts and dissertations alone are not primary; if a preprint is later published, use the published version.

### 4.2 CV Doctrine

| Priority | Source |
|---|---|
| 1st | **BSV** (between-subject variability) reported by popPK source |
| 2nd | **Inter-study geometric SD** (n ≥ 3 sources) |
| 3rd | **Class default**: CL/Vd CV=0.30; enzyme abundance CV=0.5; in vitro CLint CV=0.5 |

**Multi-source mean**: geometric mean of point estimates (lognormal assumption for popPK CL/Vd).

**BSV + inter-study GSD both available**: `max(BSV, inter-study GSD)` — conservative; captures within-population BSV + cross-population/cross-study uncertainty.

**Single source with no BSV reported** (Gap 4 closure): class-default CV applied; document explicitly as "single source, BSV not reported, class default applied".

### 4.3 Source Exhaustiveness

**Search corpus**:
- PubMed
- Google Scholar
- FDA review documents
- EMA assessment reports
- ChEMBL / DrugBank canonical references
- bioRxiv preprints

**Stopping rule**: predefined source set fully searched, OR 30 candidate abstracts/documents reviewed (whichever first).

**Per-item documentation**:
- Search terms used
- Databases searched
- N candidates reviewed
- Selected source(s) with full citation
- Disposition state

### 4.4 Per-Item Documentation Template

Each item in v3 literature deliverable (§9.3) follows this template:

```
### Item N: <name>
- v1/v2 state: <values, source>
- T1 flag: <rationale>
- Search:
  - terms: [...]
  - databases: [...]
  - N candidates reviewed: <int>
- Selected source(s): [<full citation with DOI>] OR null
- Doctrine application:
  - Mean rule: <Vss / Vc / V÷F + F=...>
  - CV rule: <BSV / inter-study GSD / class default>
  - Same-entity check: <pass/fail + rationale>
- Sub-decisions resolved: [...]
- Final values: mean=<X>, cv=<Y>
- Disposition: literature_applied | interpretation_resolved | ceiling_accepted
- IF ceiling: rationale + retained class-estimate value
```

### 4.5 Items 5-6 Parallel Doctrine (non-popPK)

Items 5 (SPR primary proteomic) and 6 (CES2/tebipenem direct Vmax/Km) are not popPK; doctrines §4.1 do not directly apply. Parallel patterns:

**Item 5 — SPR abundance**:
- Mean: primary proteomic measurement (quantitative MS-based, e.g., Wegler et al.-style enzyme atlas)
- Unit conversion: spec implementation must verify primary source units (pmol/mg microsomal protein vs pmol/g organ) match physiology yaml convention; conversion factor cited if used
- CV: inter-individual variability if reported, else inter-study GSD, else class default 0.5
- Same-entity check: human SPR isoform; species extrapolation rejected

**Item 6 — CES2/tebipenem CLint**:
- Mean: CLint = Vmax/Km from in vitro CES2 incubation (recombinant or human liver CES2); IVIVE per-pmol-enzyme scaling
- Unit chain: spec implementation must verify Vmax/Km units → CLint per pmol enzyme conversion → consistency with v2 `enzyme_affinity_for_conversion` units
- CV: inter-experiment GSD if multiple studies, else class default 0.5
- Same-entity check: human CES2 isoform (not CES1; not animal)

Source exhaustiveness (§4.3) and documentation template (§4.4) apply identically to items 5-6 (item-specific search corpus may differ — proteomic atlas journals vs drug metabolism journals).

## 5. Per-Item Acceptance Gates

### 5.1 Item 1 — BH4 active CL/Vd (sepiapterin)

- **v1/v2 state**: Vd = 150 L (T1 flagged: 1.5-50× off literature)
- **Target source(s)**: sapropterin Feillet 2008 popPK (oral) + IV BH4 disposition studies if found
- **Doctrine path**: §4.1 oral acceptance, V/F division by F, 2-comp Vss extraction, same-entity (sapropterin = BH4 salt form OK)
- **Sub-decisions**:
  - F_sapropterin: literature primary citation required
  - 2-comp model selection: Vss vs Vc
- **Sub-decision fallback chain** (Gap 5 closure):
  1. Primary F citation found → use directly
  2. Primary F not found → **strict downgrade to `ceiling_accepted`**, retain v1 Vd=150, document "F primary uncertain, ceiling reached"
  3. F geometric mean over reported ranges OR CV inflation: **rejected** (ad-hoc, violates Q4 doctrine)
- **Expected disposition**: `interpretation_resolved` (if F primary found) OR `ceiling_accepted` (if F primary not found)

### 5.2 Item 2 — GS-441524 active CL/Vd (remdesivir)

- **v1/v2 state**: T1 flagged 2.5× off
- **Target source(s)**: Sukeishi 2022 popPK (T1 reference); IV remdesivir / GS-441524 human disposition studies
- **Doctrine path**: §4.1 2-comp Vss; oral V/F division if applicable; **same-entity strict — human only** (species extrapolation rejected)
- **Sub-decisions**:
  - Verify Sukeishi 2022 species/route during search (T1 cited but doctrine eligibility unverified)
  - If Sukeishi 2022 is non-human or uses non-eligible route, fall back to other human GS-441524 popPK sources
- **Expected disposition**: `literature_applied` if eligible human GS-441524 popPK found; `ceiling_accepted` otherwise

### 5.3 Item 3 — R406 active CL/Vd (fostamatinib)

- **v1/v2 state**: T1 flagged 1.8× off
- **Target source(s)**: PMC9250994 IV micro-dose
- **Doctrine path**: §4.1 IV direct → central Vd; if 2-comp reported, Vss
- **Sub-decisions**: none significant (cleanest item)
- **Expected disposition**: `literature_applied`

### 5.4 Item 4 — tebipenem active CL/Vd

- **v1/v2 state**: T1 noted "mostly OK"
- **Target source(s)**: existing tebipenem oral popPK + any IV human studies
- **Doctrine path**: §4.1 oral V/F if applicable
- **Sub-decisions**: F_tebipenem decision if oral; "mostly OK" baseline does NOT permit lazy ceiling — full source exhaustiveness (§4.3) applies
- **Expected disposition**: `literature_applied` (small adjustment) OR `ceiling_accepted`

### 5.5 Item 5 — SPR primary proteomic abundance

- **v1/v2 state**: liver SPR=1e5, gut_wall SPR=3e3, kidney SPR=3e4 (class-estimated); T1 caution
- **Target source(s)**: quantitative MS-based proteomic studies of human SPR (e.g., enzyme atlas literature)
- **Doctrine path**: §4.5 parallel doctrine; unit conversion check mandatory
- **Sub-decisions**: unit conversion (per mg microsomal vs per g organ); inter-individual CV vs inter-study GSD
- **Expected disposition**: `literature_applied` (if primary proteomic exists) OR `ceiling_accepted` (if not)

### 5.6 Item 6 — CES2/tebipenem direct Vmax/Km

- **v1/v2 state**: class-extrapolated CES2 affinity in registry
- **Target source(s)**: in vitro tebipenem-pivoxil hydrolysis with recombinant CES2 or human liver CES2 microsomes
- **Doctrine path**: §4.5 parallel doctrine; IVIVE chain validation
- **Sub-decisions**: Vmax/Km units → CLint per pmol enzyme; isoform-specificity check (CES2 not CES1)
- **Expected disposition**: `literature_applied` (if direct in vitro exists) OR `ceiling_accepted`

## 6. Validation Strategy

### 6.1 Existing v2 Test Reuse

| Test | v3 disposition |
|---|---|
| `tests/regression/test_prodrug_v2_validation_gate.py` | **Update**: per-drug parametrize retained; xfail decorator removed for drugs newly passing 3-fold |
| `tests/regression/test_prodrug_v2_identity_blind.py` | **Re-execute (no change)**: enzyme abundance value changes do not violate tag-blindness |
| `tests/integration/test_prodrug_v2_mass_balance.py` | **Re-execute (no change)**: well-stirred equation invariant |
| `tests/integration/test_prodrug_v2_pipeline_smoke.py` | **Refactor to functional-only**: remove hardcoded Cmax expected values; assert pipeline executes without crash, returns valid `PredictionResult`, `Cmax > 0`. Numerical regression handled by snapshot test (separate concern). |
| `tests/integration/test_prodrug_v2_ddi_smoke.py` | **Re-execute + ±5% tolerance verification**: if DDI ratio shifts outside ±5% (likely if conversion saturating), investigate non-linearity, widen tolerance to ±10% with rationale OR update expected ratio |
| `tests/regression/test_prodrug_v2_snapshot.py` | **Update**: regenerate per-prodrug ±5% snapshots |

### 6.2 New v3 Tests

#### `tests/regression/test_prodrug_v3_enzyme_leak_audit.py`

**Purpose**: verify v3 changes affect only intended drugs.

**Logic**:
```
CHANGED_ENZYMES = set of enzymes with v3 abundance/affinity changes
                 (e.g., {"SPR", "CES2"} per items 5-6 if literature_applied; empty if D1-only)

D1_AFFECTED_DRUGS = set of prodrug drugs whose active species CL/Vd changed
                   (subset of {sepiapterin, remdesivir, fostamatinib, tebipenem_pivoxil})

for drug in 107_holdout:
    drug_graph = load deterministic point-estimate (cv=0)
    enzymes_used = drug_graph.enzyme_affinity ∪ drug_graph.enzyme_affinity_for_conversion
    if enzymes_used ∩ CHANGED_ENZYMES OR drug.name in D1_AFFECTED_DRUGS:
        expected_changed.append(drug)
    else:
        expected_unchanged.append(drug)

# For each unchanged drug:
assert Cmax_v3(drug, deterministic) == Cmax_pre_v3(drug, deterministic)
```

**Notes**:
- Deterministic point-estimate (cv=0) eliminates MC non-determinism for byte-identical comparison
- `@pytest.mark.slow`: excluded from default CI; run on prodrug-label PR or nightly
- D1 changes are drug-side (active species elimination edges) and are captured by `D1_AFFECTED_DRUGS` not by `CHANGED_ENZYMES` — both dimensions enumerated explicitly to avoid false alarms
- If `CHANGED_ENZYMES = ∅` AND `D1_AFFECTED_DRUGS = ∅` (all-ceiling scenario), audit asserts byte-identical for all 107 holdout drugs (sanity)

#### `tests/integration/test_prodrug_v3_registry_schema.py`

**Purpose**: structural validation of `data/sbi/prodrug_activation_registry.json` v3 entries.

**Assertions per entry**:
- Required fields present: `citation`, `doctrine_path`, `disposition_state`, `source_dbs_searched`, `n_candidates_reviewed`
- `citation` matches strict format regex: `"<Author> <Year> doi:<DOI>"` OR `null` (for ceiling_accepted)
- `disposition_state` ∈ `{literature_applied, interpretation_resolved, ceiling_accepted}`
- IF `disposition_state == ceiling_accepted`: `ceiling_rationale` field non-empty
- IF `disposition_state == interpretation_resolved`: `interpretation_decision` field non-empty

This is structural validation, not value comparison — avoids tautology.

### 6.3 Benchmark Protocol

```
v3 implementation complete (items 1-6 disposition assigned)
  ↓
scripts/run_engine_benchmark.py (107 drugs, default MC sample count from script)
  ↓
data/training/4track_holdout_predictions.json regenerated
  ↓
Diff analysis (PR body):
  - 4 prodrug drugs: v1 / v2 / v3 fold-error table
  - Drugs in `expected_unchanged` set (§6.2): byte-identical assertion
  - Drugs in `expected_changed` set (prodrugs + any enzyme-cross-leaked drug): per-drug Cmax delta tabulated; cross-leak drugs flagged for review
  - AAFE delta (Engine track, Meta track) — material delta triggers CLAUDE.md table update
  ↓
CLAUDE.md top metrics table reconciled against new predictions JSON (per CLAUDE.md self-maintenance §1)
```

**Skip condition**: if all 6 items resolve to `ceiling_accepted` (no value changes), benchmark re-run is unnecessary. v3 PR is then documentation-only and per §7.2 may close as "T1 ceiling reached" without a code merge.

**Archive**: pre-v3 predictions JSON state available via git history (`git show HEAD~1:data/training/4track_holdout_predictions.json`); no separate archive directory created.

### 6.4 xfail Removal Procedure

```
Run gate test on v3 implementation.

For each of 4 prodrug drugs:
    IF v3 fold-error ≤ 3.0:
        Remove @pytest.mark.xfail decorator
        Update reason comment: "v3 <disposition> → passes 3-fold"

    ELIF v3 fold-error > 3.0 AND v3 fold-error ≤ v2 fold-error:
        Keep xfail; update reason: "v3 <new_value>×, improvement vs v2 <old>×, ceiling"

    ELIF v3 fold-error > v2 fold-error:
        ⚠ REGRESSION RED FLAG
        Implementer analyzes:
          - Intended doctrine consequence → document in spec/PR body, proceed
          - Unknown bug → block PR, debug
```

Regression detection prevents silent quality degradation across the 4 prodrug drugs.

### 6.5 AAFE Table Update (CLAUDE.md self-maintenance)

Per CLAUDE.md §self-maintenance: top metrics table updated only after `4track_holdout_predictions.json` regeneration. v3 PR diff includes both regenerated JSON and reconciled CLAUDE.md table — synchronous update prevents drift.

## 7. Risks and Contingencies

### 7.1 Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | F_sapropterin primary not found → BH4 ceiling | Medium | Item 1 closes without value change | §5.1 fallback chain |
| R2 | SPR primary proteomic does not exist | Medium-High | sepiapterin Cmax not improved | Acceptable per (B-refined); ceiling_accepted closes T1 caution |
| R3 | CES2/tebipenem direct in vitro does not exist | Medium-High | tebipenem not improved | Same as R2 |
| R4 | Some prodrug drug regresses (v3 > v2 fold-error) | Low-Medium | Silent quality drop | §6.4 regression red-flag gate |
| R5 | Enzyme leak to non-prodrug drugs | Low | Headline AAFE perturbation | §6.2 leak audit (deterministic) |
| R6 | All 6 items ceiling_accepted | Low | No code PR; docs-only outcome | §7.2 contingency flow |
| R7 | v2 PR #7 architectural changes during review | Low | v3 base shifts | §8 precondition: v3 implementation starts only after v2 merge |
| R8 | Doctrine inconsistency across items | Medium | Schema test passes but values incoherent | Per-item doctrine path explicit (§5); reviewer cross-check |
| R9 | SBI priors stale post-v3 (12 SBI drugs) | High (if prodrugs in SBI set) | TDM routing accuracy | Out-of-scope (separate retrain follow-up); §8 precondition: enumerate prodrug-SBI intersection; PR body warning if non-empty |
| R10 | Headline AAFE marginally worse | Low (4/107 drugs only) | Narrative concern | Acceptable; §7.3 sanity bound |

### 7.2 Contingency Flows

**Per-item ceiling fallbacks** (Section 5):
- F primary not found (BH4) → ceiling_accepted, Vd=150 retained, "F primary uncertain" documented
- SPR primary proteomic not found → ceiling_accepted, class-estimate retained, T1 caution closure documented
- CES2 direct in vitro not found → ceiling_accepted, class-extrapolation retained
- Regression detected → §6.4 procedure (intended vs unknown-bug analysis)

**All-items-ceiling scenario**:
- No code PR created (no value changes to merge)
- Documentation: `docs/claude/diagnosis.md` updated with "T1 ceiling reached, v4 hypothesis required" entry; v3 design spec (this file) and v3 literature deliverable retained as historical record
- v2 PR #7 unchanged; mechanistic-A consistency preserved
- v4 follow-up requires new hypothesis (active uptake, 2-comp active, etc.)

### 7.3 R10 AAFE Sensitivity Sanity Bound

Headline AAFE = 2.695 (geometric mean across 107 drugs). log(2.695) ≈ 0.991. If 4 prodrug drugs uniformly worsen by factor of 2× (extreme case), shift = log(2) × 4/107 ≈ 0.026 in log space, ≈ +0.07 in AAFE → ≈ 2.77. Within margin of MC noise + bootstrap CI [2.30, 3.20]. Marginal impact on headline narrative — acceptable.

## 8. Preconditions

Before v3 implementation begins (writing-plans skill spawned):

### 8.1 v2-v3 Sequencing

v3 implementation **starts only after v2 PR #7 merged to main**. Before merge:
- Design and research phases (this spec, literature deliverable drafting) are permitted
- No code branch created from v2 branch (`feat/prodrug-activation-v2`)
- v2 review-time changes incorporated into v3 base

After v2 merge:
- v3 branch from main HEAD (post-v2-merge)
- v3 base includes any review-time fixes to v2

### 8.2 SBI Routing Intersection Check

Before v3 implementation, confirm prodrug-SBI intersection:

```bash
jq '.[] | select(.method == "SBI")' data/sbi/method_routing.json | grep -E "sepiapterin|remdesivir|fostamatinib|tebipenem"
```

- IF intersection non-empty: PR body must include explicit SBI staleness warning per affected drug
- IF intersection empty: SBI not directly affected; warning unnecessary

This determines R9 severity.

## 9. Deliverables

### 9.1 Spec Files (this brainstorming phase)

```
docs/superpowers/specs/2026-04-29-prodrug-activation-v3-design.md  (this design spec)
```

### 9.2 Plan File (writing-plans phase, follows this spec)

```
docs/superpowers/plans/2026-04-29-prodrug-activation-v3.md
```

### 9.3 Implementation Deliverables (v3 PR diff)

```
docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md          (NEW — v3 literature deliverable, T1 pattern)
data/sbi/prodrug_activation_registry.json                            (UPDATED — items 1-4 active CL/Vd, item 6 affinity if applicable)
data/physiology/reference_man.yaml                                   (UPDATED — item 5 SPR abundance if applicable)
tests/regression/test_prodrug_v2_validation_gate.py                  (UPDATED — xfail removal per §6.4)
tests/regression/test_prodrug_v2_snapshot.py                         (UPDATED — snapshot regeneration)
tests/integration/test_prodrug_v2_pipeline_smoke.py                  (REFACTORED — functional-only per §6.1)
tests/integration/test_prodrug_v2_ddi_smoke.py                       (UPDATED — tolerance verification per §6.1)
tests/regression/test_prodrug_v3_enzyme_leak_audit.py                (NEW — §6.2)
tests/integration/test_prodrug_v3_registry_schema.py                 (NEW — §6.2)
data/training/4track_holdout_predictions.json                        (REGENERATED — §6.3)
CLAUDE.md                                                             (UPDATED — top metrics table per §6.5)
CHANGELOG.md                                                          (UPDATED — v3 entry)
```

## 10. Success Criteria

v3 PR is mergeable iff all of the following hold:

1. **Minimum scope**: ≥1 of 6 items has disposition state `literature_applied` or `interpretation_resolved` (R6 prevention — all-ceiling closes documentation-only per §7.2).
2. **Explicit disposition**: every item has documented disposition state, search documentation, and (for non-ceiling) primary citation.
3. **v2 tests pass post-v3-modification**: all 6 v2 tests (per §6.1 dispositions — re-executed, refactored, or updated) pass. 5 pre-existing main failures remain (separate triage, not v3-caused).
4. **xfail flip discipline**: drugs newly passing 3-fold have xfail decorator removed; CI green on all xfail-removed drugs.
5. **New v3 tests pass**: enzyme_leak_audit, registry_schema both green.
6. **Invariance verified**: drugs in `expected_unchanged` set (§6.2) Cmax byte-identical (deterministic point-estimate).
7. **No silent regression**: regression red-flag (§6.4) absent OR documented as intended doctrine consequence.
8. **PR diff completeness**: registry JSON + predictions JSON + CLAUDE.md table + CHANGELOG entry + all test changes present in single PR.

## 11. References

- v3 design spec: `docs/superpowers/specs/2026-04-29-prodrug-activation-v3-design.md` (this file)
- v2 design spec: `docs/superpowers/specs/2026-04-27-prodrug-activation-v2-design.md`
- v2 plan: `docs/superpowers/plans/2026-04-27-prodrug-activation-v2.md`
- T1 literature deliverable (v2): `docs/superpowers/specs/2026-04-27-prodrug-v2-task1-literature.md`
- v1 design spec: `docs/superpowers/specs/2026-04-24-prodrug-activation-design.md`
- v1 plan: `docs/superpowers/plans/2026-04-25-prodrug-activation.md`
- v1 merge to main: `af9d2be` (2026-04-26)
- v2 push: `aef6f8e` (2026-04-28) on `feat/prodrug-activation-v2`
- v2 PR: https://github.com/jam-sudo/Sisyphus/pull/7
- Sisyphus design: `DESIGN.md`
- CLAUDE.md self-maintenance: `CLAUDE.md` §self-maintenance
