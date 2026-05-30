---
date: 2026-05-30
spec: B-14 — Hepatic UGT IVIVE differential (per-substrate scaling factor registry)
status: design (awaiting user review)
parent: ../../claude/dead-ends.md §DE-38 / §DE-39 (morphine/codeine over-prediction lineage)
related:
  - docs/claude/dead-ends.md §DE-36 (UGT fm redistribution — headline-neutral)
  - docs/claude/dead-ends.md §DE-38 (B-02: morphine/codeine worsen under UGT activation)
  - docs/claude/dead-ends.md §DE-39 (B-13: gut UGT cannot fix morphine; gut = 0.15% of hepatic)
  - data/enzymes/ugt2b7_substrates.json, data/enzymes/ugt1a9_substrates.json (fm registries)
  - src/sisyphus/predict/ivive.py (_decompose_clint — the hook site)
  - src/sisyphus/predict/non_cyp_substrates.py (registry-loading pattern)
---

# B-14 — Hepatic UGT IVIVE Differential

## Goal

Correct the systematic **UGT in-vitro→in-vivo under-prediction** by multiplying each drug's
UGT-routed intrinsic clearance by a literature-verified, per-substrate scaling factor (SF). The
proximate target is the morphine/codeine over-prediction documented in **DE-38** (worsened under
B-02's correct UGT activation) and confirmed in **DE-39** as *not* fixable by gut UGT (gut UGT2B7
is ~0.15% of hepatic). This is the hepatic lever DE-39 named as "the only plausible remaining one."

**This is a mechanism-correctness ship with measured acceptance, NOT a guaranteed accuracy win.**
The honest precedent is B-02/B-13: apply literature-verified values, measure the 107-holdout NET,
and accept the result — shipping if it helps or is neutral-without-catastrophe, retiring to a new
dead-end (DE-40) if it net-regresses. No value is fit to any holdout drug's observed Cmax.

## Background

### The clearance mechanism (verified from code, 2026-05-30)

`_decompose_clint` (`ivive.py:305`) back-derives each per-enzyme affinity so the engine
reconstructs the ML CLint:

```
affinity[enzyme] = (CLint_hepatic × fm[enzyme]) / (abundance × ivive_scaling)
engine:  CLint_organ = Σ (abundance × affinity × ivive_scaling) = CLint_hepatic × Σ fm = CLint_hepatic
```

The ML CLint predictor is trained on **in-vitro** hepatocyte/microsome data (TDC Hepatocyte_AZ).
UGT glucuronidation is systematically **under-predicted** by in-vitro assays (the "UGT IVIVE
under-prediction" problem), so the engine's UGT-routed clearance inherits that under-prediction.
A per-substrate SF on the UGT-routed affinity is the mechanistically-correct correction.

### Why this is not DE-36 / DE-38 / DE-39 redux

- **DE-36** (UGT fm redistribution) and **DE-38** (B-02 activation) changed *which* enzyme carries
  the clearance, not its *magnitude*. They were headline-neutral because UGT effective CL was
  *lower* than the CYP-default it replaced. B-14 raises the UGT-routed magnitude — the missing piece.
- **DE-39** (gut UGT) is extra-hepatic and negligible (0.15% of hepatic). B-14 is hepatic.
- **The DE-39 lesson is load-bearing here:** the B-13 spec confabulated literature citations twice.
  B-14 therefore requires primary-source verification of *every* SF before it is committed (§5).

### Viability findings (literature check, 2026-05-30)

A clean "bases over-predicted / acids under-predicted" separation does **not** exist — the
albumin-mediated UGT IVIVE under-prediction is general (literature: 1% BSA for acids, 2% for
bases/neutrals; up to 16× CLint increase for morphine, Gill/Galetin 2012 PMC3310423). The approach
is nonetheless viable because of a **three-way natural protection** rooted in per-substrate facts:

| drug | UGT | fm | engine dir | expected SF disposition | net effect |
|---|---|---|---|---|---|
| morphine | UGT2B7 | **0.85** | over 3.38× | large verified SF (albumin ↑16×; renal glucuronidation, Knights 2016) | **big improvement** |
| codeine | UGT2B7 | (reg) | over 1.78× | verified SF (O-glucuronidation) | improvement |
| indomethacin | UGT2B7 | **0.15** | under | SF may exist but fm low (CYP2C9-dominated) → tiny effect | ~unchanged |
| ketorolac | UGT2B7 | (reg) | under | AD-flagged `HIGH_ACID_LOW_FUP` → out of in-domain | excluded from in-domain |
| dapagliflozin | UGT1A9 | 0.50 | under | PBPK well-behaved → SF ~1.0 | unchanged |
| bexagliflozin | UGT1A9 | (reg) | under | likely SF ~1.0 | unchanged |
| etodolac | UGT1A9 | (reg) | under | **verified** no SF (Gill 2012 / Rowland albumin set exclude it) → 1.0; AD-flagged | unchanged |
| glasdegib | UGT1A9 | (reg) | under | **verified** UGT minor (~7%, CYP3A4-dominated) → not_applicable | unchanged |

The drugs that need help (morphine/codeine) have high fm + large documented under-prediction; the
drugs that would be hurt are protected by **low fm** (indomethacin 0.15), **well-behaved IVIVE**
(gliflozins → SF≈1), **AD-flag exclusion** (ketorolac, etodolac), or **minor UGT** (glasdegib).
This protection is read from per-substrate literature, not chosen to flatter morphine (§7 guard).

**Caveat:** the parallel verification workflow stalled after fully verifying only 2 of 8 substrates
(etodolac, glasdegib). The remaining six — including morphine and codeine — MUST be verified
sequentially at implementation (§5) before any SF is committed.

## Architecture (predict-side; zero engine change)

The SF multiplies the UGT-tagged affinity in `_decompose_clint`. `predict/` is permitted to know
enzyme identity (the identity-blind invariant binds only `engine/`); the engine receives a larger
affinity number and is untouched.

```
build_drug_on_graph (ivive.py ~654)
  ├─ get_non_cyp_fractions(smiles)         # existing: fm[UGT2B7]=0.85 etc.
  ├─ get_ugt_ivive_sf(smiles)   ── NEW     # InChIKey → per-substrate scalar SF (default 1.0)
  └─ _decompose_clint(..., ugt_ivive_sf=<float>)

# inside _decompose_clint, per enzyme:
affinity = (clint_hepatic_l_per_h * fraction) / (abundance * _IVIVE_SCALING)
scaled   = max(affinity, 0.0) * metabolic_fraction
if enzyme in ugt_enzymes:                  # predict knows identity; legal
    scaled *= ugt_ivive_sf                 # default 1.0 → bit-identical no-op
enzyme_affinity[enzyme] = Distribution(mean=scaled, cv=clint.cv)
```

**Effect:** `CLint_organ = CLint_hepatic × (fm_UGT × SF + fm_residual)` → UGT path amplified →
extraction ↑ → Cmax ↓. The SF, being a per-pmol-enzyme correction, also applies wherever the
enzyme acts (gut/kidney) since the same drug-level affinity is reused — mechanistically coherent
and numerically negligible at the gut (B-13: 0.15% of hepatic).

**Invariant preservation:**
- Engine identity-blind — **zero lines changed** in `engine/`; it receives a scalar affinity. ✓
- All parameters Distribution — affinity stays a `Distribution`. ✓
- SF = 1.0 ⇒ bit-identical to current cache ⇒ Gate-D shows only the seeded SF>1 drugs shift. ✓
- Holdout inviolable — SFs are literature in-vivo/in-vitro ratios, never fit to observed Cmax. ✓
- No drug-specific branch — a literature registry keyed by InChIKey, identical in kind to the
  B-02 fm registry and B-03.x CES1 kinetics (`if drug==X` is never written). ✓

## The SF registry

New file `data/enzymes/ugt_ivive_sf.json`, InChIKey-keyed, following the
`cyp_clearance_overrides.json` / substrate-registry pattern:

```json
{
  "version": 1,
  "description": "Per-substrate UGT in-vitro→in-vivo scaling factors (B-14). Default 1.0.",
  "entries": {
    "<full_rdkit_inchikey>": {
      "drug": "morphine",
      "ivive_sf": 5.0,
      "basis": "HLM+albumin",
      "disposition": "literature_applied",
      "literature": [{"citation": "...", "pmid_or_doi": "...", "reported_value": "...", "verified": true}]
    }
  }
}
```

The `5.0` above is **illustrative schema only** — every actual `ivive_sf` is set by the §5
verification against primary sources and is never pre-decided in this spec.

- Loader `get_ugt_ivive_sf(smiles) -> float` co-located in `predict/non_cyp_substrates.py`
  (no new file → respects the 20-files-per-directory ceiling). Returns the entry's `ivive_sf`
  by full InChIKey, else **1.0**.
- The SF is a per-substrate scalar applied to **all** of that drug's UGT-tagged affinities (the
  under-prediction is a substrate-level glucuronidation property, not enzyme-pair-specific).
- `disposition` follows the B-02/B-03.x doctrine: `literature_applied` (verified SF>1),
  `ceiling_accepted` (under-prediction documented but no usable per-drug number → 1.0 with a note),
  `not_applicable` (UGT not rate-limiting → 1.0), `default_1.0` (no evidence → 1.0).

## SF derivation & verification (anti-confabulation)

**The DE-39 lesson is mandatory here.** Every SF and every citation is verified against primary
sources before it enters the registry.

1. **Sequential verification, not 8-way parallel.** The 8-parallel viability workflow stalled
   (heavy WebFetch). Verify in batches of ≤3 agents, or inline, with a per-drug WebFetch budget.
2. **Per substrate**, find the in-vivo/in-vitro UGT CLint ratio (or the albumin-effect fold, or the
   PBPK-fitted glucuronidation SF), with a resolvable PMID/DOI and the value confirmed in the paper.
3. **Adversarial check on the consequential SFs** (morphine, codeine): an independent skeptic must
   confirm the citation is real and the value correctly extracted (refute-by-default).
4. **No verifiable value ⇒ SF = 1.0** (`default_1.0` or `ceiling_accepted`). Never invent.
5. Candidate anchors already surfaced (to be re-verified, not trusted): Gill/Galetin 2012
   (PMC3310423, albumin effect, up to 16× for morphine), Knights 2016 (PMID 26808419, renal
   glucuronidation IVIVE), Kilford 2009 DMD, Rowland albumin-effect series. **Confabulation guard:
   none of these is "applied" until its PMID resolves and the number is seen in the source.**

## Acceptance gates

Mirrors B-02/B-13 Gate-D, on the same numerics stack (`/opt/miniconda3/bin/python3`):

- **D1 — invariance:** every non-seeded holdout drug bit-identical to the pre-B-14 cache (SF=1.0 is
  a no-op). Only drugs with an `ivive_sf ≠ 1.0` may shift.
- **D2 — direction:** every shifted drug's Cmax moves **down** (SF raises clearance). Verify.
- **D3 — NET honesty:** regenerate the 4-track cache; report Meta AAFE Δ against the B-13 cache
  (2.69825) and the bootstrap CI half-width. **Accept the literature-driven NET**, whatever it is.
- **D4 — record:** Gate results + per-drug shifts logged to experiment-log; CLAUDE.md top metrics
  touched **only** if a new 3-sig-fig headline is promoted (reconciled against the regenerated cache).

**Ship / retire decision (honest, pre-committed, aligned with the B-02 amended Gate-A criterion):**
- NET Meta improves, or is statistically unchanged (|Δ| within the bootstrap-noise floor, ~±0.02 —
  the same threshold pinned in `test_cached_holdout_aafe_is_2p698`) → **ship** as a
  mechanism-correctness and (possible) accuracy improvement.
- NET Meta regresses beyond that noise floor → **retire to DE-40** with the finding, registry kept
  as an audited artifact (B-11/DE-37 precedent), SFs reverted to 1.0.

## Scope

**In:** UGT2B7 + UGT1A9 substrates already in the B-02 registries (8 seeds). Predict-side SF hook.
A separate `ugt_ivive_sf.json` registry. Sequential literature verification. 4-track regen + gates.

**Out (YAGNI / deferred):**
- **Renal UGT glucuronidation** (Knights 2016 shows morphine's under-prediction is *partly* missing
  renal UGT2B7/UGT1A9). A kidney-node UGT abundance is a *separate* structural cycle (like B-13 but
  renal) and is **not** in B-14 — flagged as a possible Phase 2 if the hepatic SF under-delivers.
- New UGT substrates beyond the 8 seeds (separate registry-expansion cycle).
- CYP / CES IVIVE differentials (UGT is the documented under-prediction; others out of scope).
- Any engine change (the design explicitly avoids one).

## Testing

- **Unit:** `get_ugt_ivive_sf` returns the registry SF by InChIKey, 1.0 for unlisted, 1.0 for
  invalid SMILES (no raise).
- **Unit:** `_decompose_clint` with `ugt_ivive_sf=1.0` reproduces current affinities bit-identically;
  with SF=k, the UGT-tagged affinity scales by exactly k and non-UGT affinities are unchanged.
- **Regression:** `test_ugt_ivive_sf_registry_schema` — every `literature_applied` entry has a
  `verified: true` citation with a PMID/DOI; no entry has `ivive_sf < 1.0` (anti-fudge: the
  correction only ever *raises* UGT clearance).
- **Integration:** Gate-D spot-check (only seeded drugs shift; directions down).
- **Identity-blind invariance:** the existing random-rename engine test must still pass (engine
  untouched, so it will — assert it as a guard).

## Risks & open questions

1. **DE-40 risk (primary).** The albumin-effect SF is general; if morphine/codeine's verified SFs
   are small, or if indomethacin's fm-0.15 worsening + any gliflozin SF outweigh the morphine gain,
   NET is neutral/negative. Mitigated by honest acceptance (ship-or-retire pre-committed above).
2. **Cherry-pick guard.** Every registry substrate is audited; SFs come only from verified
   literature; the NET is measured on the full holdout. We do not seed only morphine. If only
   morphine yields a verified SF, that is an honest literature outcome, not a selection.
3. **fm dependence.** The protection of indomethacin rests on its fm[UGT2B7]=0.15. If a future fm
   re-curation raises it, B-14's safety margin shrinks — note the coupling in the registry comment.
4. **Renal alternative.** If the hepatic SF under-delivers on morphine (because its under-prediction
   is substantially renal), Phase 2 = kidney UGT node. Documented, not started.
