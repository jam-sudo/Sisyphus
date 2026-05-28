---
date: 2026-05-27
spec: B-13 — Gut UGT2B7 + UGT1A9 abundance expansion (B-02 Phase 2.x follow-up)
status: design (awaiting user review)
parent: ../../claude/backlog.md §B-13
related:
  - docs/claude/dead-ends.md §DE-38 (B-02 secondary finding: morphine/codeine over-prediction worsening)
  - docs/superpowers/specs/2026-05-26-B02-ugt-public-registry-design.md §"Out-of-scope" (gut UGT explicitly deferred)
  - data/physiology/reference_man.yaml (target file)
---

# B-13 — Gut UGT2B7 + UGT1A9 Abundance Expansion

## Goal

Add literature-anchored UGT2B7 + UGT1A9 abundance to the gut_wall node of `reference_man.yaml`, completing the physiologically correct extra-hepatic UGT representation that B-02 deferred. Re-measure the 107-holdout cache. Accept the resulting Meta AAFE and per-drug FE shifts as-is, provided gates D1/D2/D3 hold. The morphine/codeine FE worsening flagged in DE-38 may or may not improve under this expansion; literature analysis (ultrathink, 2026-05-27) showed that the pre-B-02 morphine engine FE 1.90 was likely a **phantom coincidence** built on inflated gut CYP3A4 extraction (CYP3A4 fm × gut_CYP3A4_abundance produced ~0.78× CLint of false extraction for a drug that is not clinically a CYP3A4 substrate). Even literature-correct gut UGT addition restores only ~0.4% of that magnitude. The cycle's success therefore cannot be tied to "morphine FE ≤ pre-B-02"; it is a **mechanism-correctness ship**, not an accuracy ship.

## Background

### DE-38 (the proximate trigger)

B-02 Phase 2 (2026-05-27, commit `dd197a5`) activated the UGT path via literature-curated UGT2B7 + UGT1A9 substrate registries (8 seed drugs). Same-numerics-stack measurement showed:
- 6 of 8 seeds improved (under-predicted drugs moved toward observation: ketorolac, indomethacin, dapagliflozin, etodolac, bexagliflozin, glasdegib)
- 2 of 8 seeds worsened (over-predicted drugs moved further over: morphine FE 1.90 → 2.94, codeine 1.98 → 2.71)
- Net Meta Δ = +0.0067 (within bootstrap noise; Gate-A under amended bootstrap-noise criterion PASS)

The DE-38 finding documents that morphine/codeine worsening is mechanistic — UGT2B7 effective hepatic CL (abundance × literature-fm × XGBoost CLint) is lower than the pre-B-02 CYP-default-route allocation it replaced, producing higher Cmax.

### Ultrathink-surfaced refinement (2026-05-27, this spec)

A pre-design ultrathink traced the code mass-balance and identified that **the dominant pre-B-02 morphine extraction was happening in gut_wall, not liver**:

```
gut_CYP3A4_CL = gut_CYP3A4_abundance × affinity[CYP3A4] × ivive_scaling
              = 21M × (CLint_h × fm_pre[CYP3A4]) / (liver_CYP3A4 × IVIVE) × gut_ivive
              ≈ CLint_h × fm_pre[CYP3A4] × 2.59
```

For morphine ("base" compound), `_FM_ADJUSTMENTS["base"]` allocates fm[CYP3A4] ≈ 0.3 by default. Pre-B-02 gut extraction ≈ 0.78 × CLint_h — substantial first-pass via an enzyme morphine does not clinically use (Coffman 1997, Court 2003). Post-B-02 (fm[UGT2B7]=0.85, fm[CYP3A4]=0.045 residual), the gut CYP3A4 path drops to 0.045 × 2.59 ≈ 0.12 × CLint_h. Liver UGT2B7 picks up 0.85 × CLint_h, but there is no gut UGT2B7 to replace the lost gut CYP3A4 contribution. Net first-pass drops → Cmax rises.

The B-02 spec §"Out-of-scope" explicitly deferred gut UGT:
> "Gut wall UGT abundance. UGT2B7 in particular has substantial gut expression (relevant for morphine first-pass), but DE-36's measurement was liver-only. Gut UGT expansion is a separate cycle..."

B-13 IS that separate cycle.

### Why this is NOT an abundance/IVIVE recalibration despite the backlog framing

The original B-13 backlog entry framed the work as "UGT2B7 abundance + IVIVE recalibration" with 3 investigation surfaces (gut, liver, IVIVE). The ultrathink reduced scope to **gut UGT only**:

- **Liver UGT2B7 abundance audit** (current 2.43e6 = lower-bound 36 pmol/mg, mid-point would be 3.04e6 = 45): the `_decompose_clint` math derives affinity as `(CLint × fm) / (abundance × IVIVE)`. Increasing liver abundance INCREASES the denominator, decreasing per-pmol affinity, but the product `abundance × affinity = CLint × fm` is UNCHANGED. So liver abundance audit alone has no effective-CL impact for the activated UGT path. Removed from scope.
- **IVIVE differential** (UGT-specific vs CYP-specific scaling, supported by Cubitt 2009 / Riley 2005 systematic UGT under-prediction literature): requires extending the `ivive_scaling` field from organ-level to enzyme-level in the engine. This is an architectural change. Deferred to a hypothetical B-13.x with its own spec.

What remains: **gut UGT abundance addition**. Effective CL impact is direct (gut abundance × affinity is a new clearance contribution). This is the smallest scope that addresses the mechanistic gap.

## Scope

### In-scope

1. Two new abundance entries in `data/physiology/reference_man.yaml` gut_wall enzymes block: UGT2B7 + UGT1A9.
2. Cache regeneration via `scripts/run_engine_benchmark.py`.
3. Bootstrap CI re-computation via `scripts/bootstrap_4track_ci.py`.
4. T4 cached AAFE pin update (rename + value).
5. Docs updates (README, experiment-log, dead-ends DE-38 closure or DE-39 new, backlog B-13 closure, landmarks).

### Out-of-scope

- **Liver UGT2B7/UGT1A9 abundance audit** (analytically null for effective CL given the fm-normalized affinity derivation; documented in §Background above).
- **IVIVE differential / enzyme-level scaling** (requires engine architectural change; deferred to B-13.x with its own spec).
- **Phenotype scaling for UGT2B7/UGT1A9** (B-02 Phase 2.y backlog item).
- **Gut UGT1A1 / UGT1A4 / UGT1A6 / UGT1A8 / UGT2B15 / UGT2B17** addition (no holdout drug currently has affinity for these; adding them would be inert until a registry pairs them).
- **Per-drug fm tuning** (registry values are literature mid-points and remain verbatim).

## Per-enzyme values

### gut_wall UGT2B7 abundance

```yaml
UGT2B7: {mean: 9.0e4, cv: 0.6}
```

Derivation: 15 pmol/mg × 6000 mg mucosal protein = 9.0e4 pmol total in gut wall.

**Anchor:** Bhatt 2019 DMD 47:498 ("Comparative Quantitative Proteomic Analysis of Hepatic and Extrahepatic UGT and CYP Enzymes in Human Tissues") — Table 2 reports intestinal microsomal UGT2B7 5-30 pmol/mg with median ~15 across donors. Open-access via PMC.

Mucosal protein 6000 mg is the established value used by existing gut wall entries (CES2 500 pmol/mg × 6000 mg = 3.0e6, ALPI 3.89 pmol/mg × 6000 mg = 2.3e4 — both per Al-Majdoub 2020 PMC8048492).

cv = 0.6 (matches the existing CES2 cv = 0.6 in the same block, slightly higher than liver UGT cv=0.5 reflecting greater inter-donor variability in intestinal proteomics).

### gut_wall UGT1A9 abundance

```yaml
UGT1A9: {mean: 1.2e4, cv: 0.6}
```

Derivation: 2 pmol/mg × 6000 mg = 1.2e4 pmol.

**Anchor:** Bhatt 2019 DMD 47:498 reports intestinal UGT1A9 as **hepatic-dominant**, intestinal mucosa 0-5 pmol/mg (some donors below detection). Mid-point 2 pmol/mg used. The value is small but non-zero is mechanistically defensible: dapagliflozin and other gliflozins do show modest gut UGT1A9 metabolism in pharmacokinetic studies.

If Bhatt 2019 detail is not retrievable at implementation, fallback: 1.5 pmol/mg from Akabane 2012 DMD 40:1310 (intestinal UGT1A9 quantitative proteomics).

### No fallback values below the literature lower bound

Anti-fudge: if the cited paper's lower bound is X, B-13 must not use a value below X. The mid-points selected (15, 2 pmol/mg) are inside the published ranges. The implementer must verify against the paper at PR review.

## Acceptance gates

### Gate D1 (required) — Literature-anchored values verbatim

Each added abundance entry must:
- Cite a specific paper (Bhatt 2019 DMD 47:498 or equivalent open-access PMC source)
- Use a value within the cited paper's reported range
- Document derivation (`pmol/mg × mucosal mass = pmol total`) in the YAML comment

Reviewer at implementation-PR time checks the cited paper. No FE-driven adjustment permitted.

### Gate D2 (required) — Gate-D (8 seeds only shift)

Compare post-B-13 cache vs B-02 baseline cache (`data/training/4track_holdout_predictions.json` at commit `91a359e`, generated on miniconda Python 3.13.13 + numpy 2.2.6 stack on macOS arm64).

- The 8 UGT seed drugs MAY shift.
- The 99 non-seed drugs MUST be bit-identical (`|Cmax_post − Cmax_pre| < 1e-8 mg/L`).
- Any unexpected non-seed shift indicates a wiring bug (e.g., RNG-order regression, enzyme-block parsing change, accidental edit). Stop and investigate.

Compare on the SAME numerics stack used for B-02 cache generation. Cross-stack comparison invalidates the gate (per the B-02 numerics-stack incident, §experiment-log 2026-05-27).

### Gate D3 (required) — Bootstrap noise

Run `scripts/bootstrap_4track_ci.py --tag B13 --out data/validation/4track_ci_2026-05-27_B13.json` (or 2026-05-28 if date rolls). The new CI half-width is half of (upper − lower). |ΔMeta vs B-02 cache| must be less than the new CI half-width.

The B-02 CI half-width was 0.43. B-13 changes are smaller-scoped than B-02 (only YAML, no new code paths), so the CI is expected to be similar. Δ Meta is expected to be small (gut UGT addition contributes modest extraction).

### Gate D4 (informational only — NOT a pass/fail criterion)

Record post-B-13 morphine and codeine engine FE. Compare to B-02 baseline (FE 2.94 / 2.71) and pre-B-02 (FE 1.90 / 1.98). Possible outcomes:

- **morphine FE improves from 2.94 toward 1.90**: gut UGT path was the dominant gap; DE-38 productively closed.
- **morphine FE shifts marginally**: gut UGT contribution exists but is small relative to total; DE-38 partially closed, residual deferred to B-13.x (IVIVE differential) or another cycle.
- **morphine FE unchanged or worsens**: gut UGT abundance addition is mechanistically correct but does not address the residual over-prediction. DE-38 deepened — over-prediction stems from compound_type fm defaults, first-pass model, or another layer.

Each outcome is informative. The cycle ships in any case provided D1/D2/D3 pass.

### Failure response (anti-fudge)

- **D1 fail** (abundance outside literature range, or no citation): stop immediately. Re-anchor to literature. Do not iterate on values.
- **D2 fail** (non-seed drugs shifted): stop immediately. Wiring bug. Investigate enzyme block parsing, RNG order, accidental YAML edit. Do not proceed until isolated.
- **D3 fail** (|ΔMeta| > CI half-width): stop. **Do not adjust abundance to reduce the delta** (CLAUDE.md invariant #8). Two paths:
  - If failure is mild (e.g., 1.2× CI half-width), document as "B-13 ships with Meta drift slightly exceeding bootstrap noise; literature mid-point retained; user-disclosed in README"
  - If failure is severe (e.g., 2× CI half-width), retire B-13 to a new DE-39 entry: "literature-anchored gut UGT addition produces Meta drift beyond bootstrap noise; mechanism is correct but global re-calibration of other layers may be needed to absorb the shift". Mechanism finding is preserved, capability not shipped.

## Tests

### T1 — schema invariant (no change needed)

`tests/regression/test_oatp_registry_schema.py` and `tests/regression/test_ugt_registry_schema.py` are registry-schema tests; they don't validate physiology YAML enzyme blocks. No T1 work for B-13.

### T2 — `data/physiology/reference_man.yaml` parse + abundance present

Add to existing `tests/unit/test_yaml_transporters.py` (or `test_physiology_yaml.py` if it exists; verify at impl time):

```python
def test_gut_wall_has_ugt2b7_ugt1a9():
    """B-13: gut wall enzymes must include UGT2B7 + UGT1A9 (literature-anchored)."""
    import yaml
    with open("data/physiology/reference_man.yaml") as f:
        data = yaml.safe_load(f)
    gut = [n for n in data["nodes"] if n["name"] == "gut_wall"][0]
    enz = gut["enzymes"]
    assert "UGT2B7" in enz, "gut_wall missing UGT2B7 abundance (B-13)"
    assert "UGT1A9" in enz, "gut_wall missing UGT1A9 abundance (B-13)"
```

### T3 — T4 cached AAFE pin update

`tests/integration/test_holdout_regression.py::test_cached_holdout_aafe_is_2p698` → renamed to `test_cached_holdout_aafe_is_2pXXX` where `XXX` is the new post-B-13 Meta value. Tolerance retained at 0.020 (matches B-02 amendment criterion).

### T4 — existing test_prodrug_v3_enzyme_leak_audit

DRUG_SPECIFIC_CHANGES already includes the 8 UGT seed drugs (B-02 cycle update). B-13 doesn't add new drugs, so no leak-audit update needed.

### T5 — existing test_ecm_holdout_spot_check_10_drugs

Tolerance already 35% (B-02 cycle update). No change for B-13.

## Implementation Verification Gate

Before merge:
1. **D1**: every YAML entry cites a paper, value is within paper's reported range, comment documents derivation
2. **D2**: same-numerics-stack Gate-D verified — 99 of 107 holdout drugs bit-identical
3. **D3**: bootstrap CI recomputed; |ΔMeta| < CI half-width
4. **D4 recorded** in experiment-log entry (informational, not gating)
5. **Tests T2 + T3 pass** (existing T4 + T5 unchanged)

## Atomicity / Rollback

Single-commit shipping (4-5 files: YAML, cache, CI artifact, test pin, docs). `git revert <merge-commit>` cleanly undoes B-13 entirely. No external dependencies.

## Self-Maintenance Order

Per CLAUDE.md §Self-maintenance:
1. YAML edit
2. Cache regen
3. Bootstrap CI re-compute
4. T3 pin update
5. README headline + reproducibility note (only if |ΔMeta| > 0.005, which is likely given B-13's small scope; prose update regardless)
6. experiment-log entry (top, 2026-05-27)
7. dead-ends.md DE-38 closure note ("B-13 added gut UGT, [outcome]") OR new DE-39 if retired
8. backlog.md B-13 strikethrough with closure note
9. landmarks.md (new CI artifact + paper reference)

## Open Questions for Implementation

1. **Bhatt 2019 PMC accessibility**: confirm via WebFetch at impl time. If paywalled or only abstract returned, fall back to Akabane 2012 (UGT1A9) and Cubitt 2006 DMD 34:434 (UGT2B7). Both are open-access PMC.
2. **gut UGT2B7 5-30 mid-point**: 15 is the median, but the distribution skews log-normal. Use median (15) per Bhatt 2019 Table 2 column reporting.
3. **gut UGT1A9 below-detection donors**: Bhatt 2019 reports some intestinal UGT1A9 below LOD. Use the mean of detected donors as the mid-point, not the population mean including zeros (which would bias low).
