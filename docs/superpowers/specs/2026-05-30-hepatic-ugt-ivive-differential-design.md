---
date: 2026-05-30
spec: B-14 — Hepatic UGT IVIVE differential (bounded blind decisive experiment)
status: design v2 (awaiting user review; v1 superseded after adversarial review)
parent: ../../claude/dead-ends.md §DE-38 / §DE-39 (morphine/codeine over-prediction lineage)
related:
  - docs/claude/dead-ends.md §DE-36 (UGT fm redistribution — headline-neutral)
  - docs/claude/dead-ends.md §DE-38 (B-02: morphine/codeine worsen; root cause = CYP-route over-extraction revealed)
  - docs/claude/dead-ends.md §DE-39 (B-13: gut UGT cannot fix morphine; names *this* lever as the only remaining one)
  - docs/claude/cherry_picking_audit_2026-04-22.md (the integrity bar this spec must clear)
  - data/enzymes/ugt2b7_substrates.json, data/enzymes/ugt1a9_substrates.json (fm registries)
  - src/sisyphus/predict/ivive.py (_decompose_clint — the hook site)
---

# B-14 — Hepatic UGT IVIVE Differential (Bounded Blind Decisive Experiment)

> **v2 (2026-05-30).** v1 was rewritten after a 3-critic adversarial review found it (a) carried a
> cherry-picking signature, (b) mis-specified the SF basis (HLM vs hepatocyte) and mis-attributed
> renal clearance to hepatic first-pass, and (c) under-weighted the DE-36/38/39 prior. v2 reframes
> B-14 from a "fix morphine" build into a **bounded, blind, decisive experiment** whose primary
> deliverable is an *answer*, with **DE-40 (no-op) as an expected, first-class terminal state.**

## Goal

Answer one question, honestly and cheaply: **does a mechanism-correct UGT IVIVE scaling factor —
hepatocyte-basis, hepatic-fraction-only, derived blind to holdout residuals — exist for morphine/
codeine, and is it large enough to matter?**

- If **yes** (a verified SF predicts a per-drug improvement beyond noise): apply it, measure, and
  ship only if the *realized* improvement matches the *pre-registered literature prediction*.
- If **no** (SF small, unverifiable, or dominated by the renal/CYP fractions B-14 excludes): ship a
  bit-identical no-op and **retire to DE-40** with the evidence, closing the lever DE-39 named.

This is **not** a guaranteed accuracy play. Given three prior UGT dead-ends (DE-36/38/39) and the
mechanistic caveats below, **the prior favors DE-40.** The value of B-14 is a *definitive, honest
answer* to "is the hepatic UGT IVIVE differential a real lever," produced cheaply (bounded) and
without tuning to the holdout (blind).

## Background

### The clearance mechanism (verified from code, 2026-05-30)

`_decompose_clint` (`ivive.py:305`) back-derives each per-enzyme affinity so the engine reconstructs
the ML CLint; the engine applies a well-stirred extraction `CL = Q·fup·CLint/(Q+fup·CLint)`:

```
affinity[enzyme] = (CLint_hepatic × fm[enzyme]) / (abundance × ivive_scaling)
engine:  CLint_organ = Σ (abundance × affinity × ivive_scaling) = CLint_hepatic
```

The hook: multiply the UGT-tagged affinity by a per-substrate, per-enzyme SF. `predict/` may know
enzyme identity (the identity-blind invariant binds only `engine/`); the engine is untouched.

### The prior is DE-40 — three honest reasons (do not under-weight)

1. **Lineage.** DE-36 (UGT fm redistribution) was headline-neutral. DE-38 (B-02 UGT activation)
   *worsened* morphine (eng FE 1.90→2.94) and codeine (1.98→2.71); net Meta +0.0067 (within noise).
   DE-39 (gut UGT) moved morphine −0.112% (Meta Δ −2.7e-05). **DE-39 explicitly names "the hepatic
   UGT2B7 IVIVE differential" as the only remaining lever — B-14 is that telltale.** Three UGT
   interventions, all neutral/negative.
2. **DE-38's root cause is a CYP-route problem, not (only) a UGT one.** DE-38 diagnosed morphine's
   pre-B-02 FE 1.90 as a *coincidental cancellation* — CYP-default **over-extraction** plus a
   missing UGT path. Activating UGT revealed the CYP-default routing was already over-extracting.
   DE-38's prescribed remedy is "UGT IVIVE recalibration **with a CYP-route rebalance alongside**."
   **B-14 supplies only the UGT half.** It may move morphine, but it does not address the diagnosed
   CYP-route imbalance; a UGT-only correction can re-reveal it from the other side (over-correction).
3. **The honest hepatic SF is probably small.** Morphine's famous under-prediction is largely (a) an
   **HLM** albumin artifact (up to 16× in microsomes, Gill/Galetin 2012 PMC3310423) — but the ML is
   trained on **intact-hepatocyte** CLint (TDC Hepatocyte_AZ), which already recovers much of that
   correction, so the *hepatocyte* fold is materially smaller; and (b) **renal** glucuronidation
   (Knights 2016 PMID 26808419) — which B-14 excludes (renal clearance does not cause hepatic
   first-pass). The hepatic-attributable, hepatocyte-basis fraction of morphine's deficit — the only
   part B-14 may honestly apply — is plausibly ≈1.5–3×, not 16×. The decisive experiment measures it.

### Why the naïve "fix morphine" framing fails integrity

The 8 candidate substrates are exactly the holdout drugs whose over/under directions we already know
(morphine/codeine over; the other six under). A sign-restricted SF≥1 lever (raises clearance, lowers
Cmax) can *only* help the two over-predicted drugs. That joint structure — seed set co-extensive
with known residual directions + sign-restricted lever + a range-valued anchor ("up to 16×") — is
indistinguishable from fitting to holdout residuals, even with "literature" values and no
`if drug==X`. **v2 breaks this dependence** (blind derivation + pre-registered prediction-match gate,
below) instead of asserting a post-hoc "natural protection" table.

## Phase 0 — the decisive experiment (blind SF derivation) — RUNS FIRST

This phase produces the SFs and the go/no-go decision **before any code path changes the cache.**

1. **Blind to holdout folds.** SFs are extracted from literature by verifiers who are **not** given
   any holdout Cmax/fold and **not** permitted to tune toward 2-fold. We cannot un-know that morphine
   is over-predicted; we *can* ensure the SF *value* is literature-derived, not fold-derived.
2. **Correct basis, single point.** Each SF must be a **hepatocyte-basis** in-vivo/in-vitro ratio (or
   an HLM ratio explicitly scaled to hepatocyte-equivalent, with the scaling documented), tied to a
   **single named primary-source quantity** — never a range endpoint, never a midpoint guess.
3. **Hepatic fraction only.** For substrates with documented extrahepatic (renal/intestinal)
   glucuronidation (morphine, codeine), partition the deficit and apply **only the hepatic fraction**
   (Knights 2016 / fraction-excreted data). Record the withheld renal fraction in the registry so a
   future kidney-UGT node cannot double-count it.
4. **Bounded.** Hard cap **2 sequential agents per substrate**, **≤12 WebFetch calls per substrate**,
   single implementation session. The earlier 8-parallel workflow stalled; sequential + capped.
5. **Anti-confabulation (DE-39 lesson, mandatory).** DE-39 — dated *one day before this spec* — found
   the prior cycle's UGT anchors were confabulated (claimed 15 vs real 0.60 pmol/mg; a PMID that
   resolved to an unrelated paper; a citation that did not exist). Every SF citation here must
   resolve to a real PMID/DOI with the number visible in the source; morphine/codeine SFs get an
   independent adversarial skeptic confirming the fold is hepatic-only and not renal/albumin-
   contaminated. **No verifiable hepatocyte-basis hepatic-fraction number within budget ⇒ SF = 1.0,
   `disposition: ceiling_accepted`.** Never invent.
6. **Pre-register the prediction.** Freeze the registry, then compute — from the frozen SFs and each
   drug's actual well-stirred E and `fup·CLint/Q` — the **predicted per-drug ΔCmax and Meta Δ**, and
   write them into the spec/experiment-log *before* regenerating the cache.
7. **Go / no-go (pre-committed):**
   - Predicted morphine+codeine Meta Δ **below the ±0.02 worth-building threshold** (the
     regression-tripwire magnitude — cf. D4; note a *full* morphine 3.38→2.0 + codeine 1.78→1.3 fix
     is only ≈ −0.021, so a realistic partial fix is sub-threshold) ⇒ **stop. Ship the SF=1.0 no-op,
     retire to DE-40** ("hepatic UGT IVIVE differential, done honestly, is too small to matter —
     morphine's deficit is renal + CYP-route, not hepatic UGT IVIVE"). The expected, complete result.
   - Predicted Δ beyond noise ⇒ proceed to build + the acceptance gates below.

## Architecture (predict-side; zero engine change)

```
build_drug_on_graph (ivive.py ~654)
  ├─ get_non_cyp_fractions(smiles)         # existing fm routing
  ├─ get_ugt_ivive_sf(smiles) -> dict      # NEW: InChIKey → {UGT2B7: k1, UGT1A9: k2}; {} if unlisted
  └─ _decompose_clint(..., ugt_ivive_sf=<dict>)

# inside _decompose_clint, per enzyme:
affinity = (clint_hepatic_l_per_h * fraction) / (abundance * _IVIVE_SCALING)
scaled   = max(affinity, 0.0) * metabolic_fraction
if enzyme in ugt_enzymes:                       # predict knows identity; legal
    scaled *= ugt_ivive_sf.get(enzyme, 1.0)     # per-enzyme; default 1.0 → bit-identical no-op
enzyme_affinity[enzyme] = Distribution(mean=scaled, cv=clint.cv)
```

**Per-enzyme map, not a scalar** (review flaw 8): a single scalar cannot represent a multi-UGT
substrate with one under-predicted and one well-behaved isoform (the albumin effect itself differs by
isoform: 1% BSA acids vs 2% bases/neutrals). The 8 seeds are single-UGT today, but the map is
correct from day one and costs nothing.

**Node-blind reuse (review flaw, documented):** `flux.py:227 drug_enzyme_affinity(tag)` is
node-independent, so this SF also acts at **gut UGT2B7** (B-13, 3.6e3 — ~0.15% of hepatic, negligible
even ×SF). A registry comment must record: *this SF acts at every node carrying the enzyme; today
liver + gut only. If a kidney-UGT node is ever added (the deferred renal option), the SF basis MUST
be re-derived to avoid double-counting the renal fraction withheld in Phase 0.*

**Invariant preservation:** engine identity-blind (0 lines changed) ✓; all params Distribution ✓;
SF=1.0 ⇒ per-drug bit-identical ✓; holdout inviolable — SFs literature-derived, blind, never fit to
Cmax ✓; no drug-specific branch (InChIKey registry, B-02/B-03.x pattern) ✓.

## The SF registry

New file `data/enzymes/ugt_ivive_sf.json`, InChIKey-keyed:

```json
{
  "version": 1,
  "description": "Per-substrate, per-enzyme UGT hepatocyte-basis in-vitro→in-vivo SFs (B-14). Default 1.0.",
  "entries": {
    "<full_rdkit_inchikey>": {
      "drug": "<name>",
      "ivive_sf": {"UGT2B7": 1.0},
      "basis": "hepatocyte | hepatocyte_scaled",
      "hepatic_fraction_of_deficit": 1.0,
      "renal_fraction_withheld": 0.0,
      "disposition": "literature_applied | ceiling_accepted | not_applicable | default_1.0",
      "literature": [{"citation": "...", "pmid_or_doi": "...", "reported_value": "...", "basis": "...", "verified": true}]
    }
  }
}
```

All numeric values above are **illustrative schema only**; every real value is set by Phase 0.

- Loader `get_ugt_ivive_sf(smiles) -> dict[str, float]` co-located in `non_cyp_substrates.py` with a
  module section-comment delineating *fm routing* (existing) from *IVIVE magnitude* (new). Contract:
  returns the entry's `ivive_sf` map by full InChIKey, else `{}`; **never raises** (invalid SMILES →
  `{}`), unlike the `None`-returning `lookup_*` neighbors — stated in the docstring and unit-tested.
- **No hard SF≥1 floor** (review flaw 9). The schema test asserts *unverified entries are exactly
  1.0* and *every non-1.0 entry has `verified: true` + a PMID/DOI + `basis ∈ {hepatocyte,
  hepatocyte_scaled}`*. A verified SF<1 (rare: substrate-depletion over-read) is allowed if sourced;
  this removes the directional asymmetry the review flagged as a covert morphine-bias.
- **Upper sanity bound:** any `ivive_sf > 5` requires a recorded second-source adversarial
  re-verification (guards against an over-sized headline-drug fold).

## Acceptance gates (only if Phase 0 says "proceed")

Same numerics stack (`/opt/miniconda3/bin/python3`):

- **D1 — per-drug bit-identity:** every non-seeded holdout drug's predicted Cmax is **exactly** equal
  (float compare against `4track_holdout_predictions.json`) to the pre-B-14 value. SF=1.0 guarantees
  this; assert it per-drug, **not** via aggregate AAFE.
- **D2a — engine-track direction (mechanistic):** each seeded SF>1 drug's **engine** Cmax strictly
  decreases. Asserted.
- **D2b — meta-track direction (observed):** report each seeded drug's **meta** Cmax shift; do not
  assert monotonicity (the meta-learner is a learned blend). **Guard:** no seeded drug's `meta_fold`
  may cross 1.0 (over→under flip). Codeine (meta_fold 1.78, ML already well-calibrated at 0.95) is the
  flip risk; if it flips, the SF is over-sized → reduce or set 1.0.
- **D3 — prediction-match (the real honesty gate):** the *realized* per-drug ΔCmax for morphine/
  codeine must match the **Phase 0 pre-registered literature prediction** within tolerance. A drug
  landing near its literature-predicted fold = mechanism; a drug landing suspiciously near 2.0-on-the
  -nose while the predicted value was elsewhere = tuning → reject.
- **D4 — NET, recorded:** report Meta AAFE Δ vs the B-13 cache (2.69825). Note the headline floor is
  the bootstrap CI half-width (~0.43), so the **headline cannot detect a 2-drug change** — D3, not the
  headline, is the gate that constrains integrity. Resolve the v1 ±0.02-vs-~0.43 ambiguity: ±0.02 is
  the regression tripwire (`test_cached_holdout_aafe_is_2p698`); ~0.43 is statistical noise; D3 is the
  per-drug honesty gate. Ship requires D1∧D2a∧D3 and no D2b flip.

**Ship / retire (pre-committed):** D1–D3 pass and morphine/codeine realized ≈ predicted → **ship**.
Prediction-match fails, or codeine flips, or Phase 0 returned all-1.0 → **DE-40**, registry kept as an
audited all-1.0 artifact (B-11/DE-37 precedent), zero behavioral change.

## Scope

**In:** Phase 0 blind SF derivation; predict-side per-enzyme SF hook; `ugt_ivive_sf.json`; gates.
Substrate universe = UGT2B7 + UGT1A9 substrates **sourced by substrate membership, not holdout
membership** (the 8 B-02 seeds plus any literature UGT substrates the blind search surfaces — so the
registry is not co-extensive with the known-direction holdout set).

**Out (deferred / explicitly excluded):**
- **Renal UGT** (the withheld fraction). A kidney-UGT node is a separate cycle and would require SF
  basis re-derivation (double-count guard above).
- **CYP-route IVIVE rebalance** — DE-38's *other* prescribed half. If B-14 ships and morphine remains
  materially over, the DE-38-complete fix (UGT + CYP rebalance together) is the documented next cycle.
- CYP/CES IVIVE differentials; new substrates beyond what the blind search surfaces; any engine change.

## Testing

- **Unit:** `get_ugt_ivive_sf` → SF map by InChIKey; `{}` for unlisted; `{}` for invalid SMILES
  (no raise). `_decompose_clint` with empty/1.0 map reproduces current affinities bit-identically;
  with `{UGT2B7: k}`, the UGT2B7 affinity scales ×k and all other affinities (incl. UGT1A9) unchanged.
- **Unit (multi-UGT landmine):** a synthetic two-UGT drug with `{UGT2B7: k1, UGT1A9: k2}` scales each
  tag independently.
- **Regression `test_ugt_ivive_sf_registry_schema`:** unverified entries are exactly 1.0; every
  non-1.0 entry has `verified:true` + PMID/DOI + `basis ∈ {hepatocyte, hepatocyte_scaled}`; any
  `>5` carries a second-source note; morphine/codeine entries carry `hepatic_fraction_of_deficit`.
- **Integration:** Gate D1 per-drug bit-identity harness; D2a engine direction.
- **Identity-blind invariance:** the existing random-rename engine test still passes (engine untouched).

## Risks & open questions

1. **Expected DE-40 (primary, accepted).** The honest hepatocyte-basis hepatic-fraction SF is likely
   small; Phase 0 may return all-1.0. That is a *complete result*, not a failure — it closes DE-39's
   "only remaining lever" with evidence.
2. **Cherry-picking — addressed at the set level (not the strawman).** The risk is the *joint* of
   (seed set ≈ known-direction holdout drugs) + (sign-restricted lever) + (range anchor). v2 mitigates
   by: blind literature derivation; single-point hepatocyte-basis values (no range); substrate-
   membership sourcing (not holdout membership); and the D3 pre-registered prediction-match gate that
   a headline-only NET cannot launder. We cannot un-know morphine's direction, but we *can* make the
   SF value and its realized effect falsifiably literature-predicted.
3. **DE-38 incompleteness.** B-14 is the UGT half only. If it ships and under-delivers, the diagnosed
   CYP-route imbalance remains — the DE-38-complete cycle (UGT+CYP) is the documented successor.
4. **Ratio-of-ratios fragility.** The ML CLint (R²≈0.24) is not the in-vitro value the literature
   ratio was paired with. Phase 0 must record, per seeded drug, the ratio of the engine's baseline
   UGT-routed CLint to the literature in-vitro CLint; if they diverge >~2×, re-anchor to the absolute
   in-vivo glucuronidation CLint (back out the implied SF for *this* engine's baseline) or set 1.0.
5. **indomethacin (honest, not "tiny").** fm[UGT2B7]=0.15, in-AD, meta_fold 0.281. At a morphine-class
   SF~5 its total CLint ×(0.15·5+0.85)=1.6× → ~60% more clearance → further under-prediction, counted
   in in-domain Meta. Phase 0 verifies indomethacin's *own* hepatocyte SF (apply it if real, even
   though it worsens the fold — mechanism-honesty) or records it as a counted casualty with its NET
   contribution quantified. No reliance on "low-fm protection" without the arithmetic.
