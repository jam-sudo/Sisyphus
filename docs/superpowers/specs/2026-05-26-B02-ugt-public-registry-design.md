---
date: 2026-05-26
spec: B-02 Phase 2 — UGT public substrate registry
status: design (awaiting user review)
parent: ../../claude/backlog.md §B-02
related:
  - docs/claude/dead-ends.md §DE-36 (UGT path Meta-invariance measurement, 2026-05-13)
  - data/enzymes/nat2_substrates.json (schema template)
  - data/enzymes/ugt1a1_substrates.json (schema template)
  - src/sisyphus/predict/non_cyp_substrates.py (loader pattern)
  - src/sisyphus/predict/ivive.py:649-665 (currently-disabled UGT path)
---

# B-02 Phase 2 — UGT Public Substrate Registry

## Goal

Build a public, literature-anchored substrate registry for UGT2B7 and UGT1A9, activating the currently-disabled UGT path in `ivive.py` without DrugBank dependency. The headline Meta AAFE is expected to remain within ±0.005 of the current cache (2.769); Engine track may improve marginally (DE-36 prior: −0.029).

This is a **capability + reproducibility** project, not an accuracy project. DE-36 unambiguously measured Meta-invariance for the UGT path. We accept zero headline AAFE gain and ship for:

1. **Capability**: 8 holdout drugs (morphine, codeine, ketorolac, indomethacin, dapagliflozin, etodolac, bexagliflozin, glasdegib) gain proper UGT enzyme attribution in their `enzyme_affinity` dict — currently they have no UGT path at all.
2. **Reproducibility**: removes the last DrugBank-derived enzyme-attribution surface that DE-36's measurement relied on. Public-clone state now has a literature-curated substitute.

## Background

### DE-36 prior (2026-05-13)

A sensitivity test under the current pipeline (post-v0.3.2 NAT2/UGT1A1, public-clone-augmented-with-local-DrugBank) measured:

- Engine (overall N=107): 3.791 → 3.762 (Δ = −0.029, marginal improvement)
- **Meta (overall N=107): 2.679 → 2.679 (Δ = +0.0002, invariant)**

The Meta-learner's track weights absorbed the Engine improvement entirely. Mechanism: the meta-learner's compound-type-adaptive weights are calibrated against a specific Engine error profile; improving Engine on one subset shifts the error profile and the weights re-balance to neutralize the gain. This is the DE-08~DE-18 error-cancellation family.

Why DE-36 was a dead-end (not B-02 Phase 2):

1. Zero net Meta benefit at current weights
2. UGT data sourced from DrugBank only — public-clone reproducibility would require a curated literature registry (this spec)

Phase 2 addresses (2). We accept (1).

### Backlog charter (B-02)

> Phase 2 = activate in production AND make the gain reproducible on a fresh clone.
> Trigger to revisit: capability completeness becomes a priority, or someone wants a real DrugBank-free reproducibility story.

The recent public-clone reproducibility cycle (PR #43, B-03, B-03.x, 2026-05-09 → 2026-05-25) establishes the trigger via DrugBank-free narrative continuity.

## Scope

### In-scope (Phase 2)

1. Two new substrate registries: `data/enzymes/{ugt2b7,ugt1a9}_substrates.json` following the NAT2/UGT1A1 schema exactly.
2. Two new abundance declarations in `data/physiology/reference_man.yaml` for UGT2B7 and UGT1A9 — **liver node only**.
3. Two new loader functions + extended aggregator in `src/sisyphus/predict/non_cyp_substrates.py`.
4. One-line activation edit in `src/sisyphus/predict/ivive.py` (remove `ugt_enzymes = None`, derive from registry).
5. Three new tests (schema, unit lookup, integration mechanism).
6. One updated test (cached holdout AAFE pin renamed to new cache value).
7. Cache regeneration (`scripts/run_engine_benchmark.py` → `data/training/4track_holdout_predictions.json`).
8. README + CLAUDE.md + backlog.md + experiment-log.md updates per CLAUDE.md self-maintenance order.

### Out-of-scope (deferred)

- **Gut wall UGT abundance.** UGT2B7 in particular has substantial gut expression (relevant for morphine first-pass), but DE-36's measurement was liver-only. Gut UGT expansion is a separate cycle; activating it without re-anchoring the meta-learner risks under-prediction of orally-dosed UGT2B7 substrates.
- **UGT2B7/UGT1A9 phenotype scaling.** UGT2B7\*2, UGT1A9\*3, etc. allele-aware predict(phenotypes={...}) propagation is Phase 2.x.
- **UGT1A4 registry.** No seed drug in DE-36's 9-drug list is UGT1A4-dominant. Added in a future cycle when the first UGT1A4 substrate is needed.
- **Multi-enzyme attribution schema (Approach 2 from brainstorming).** Each drug appears in exactly one registry under its dominant UGT isoform. Minor isoforms documented in `notes` field for future Phase 2.x phenotype work.
- **Metronidazole.** Excluded from seed list despite appearing in DE-36's per-drug improvement set. Literature evidence (Lamp 1999) places UGT involvement at ~5-10%, below the noise floor. The DE-36 improvement is likely a collateral effect from co-activation with the other 8 drugs.

## File Inventory

| File | Action | Lines (approx) |
|---|---|---|
| `data/enzymes/ugt2b7_substrates.json` | create | ~60 (4 drugs + header) |
| `data/enzymes/ugt1a9_substrates.json` | create | ~60 (4 drugs + header) |
| `data/physiology/reference_man.yaml` | edit | +2 lines (UGT2B7, UGT1A9 abundance) |
| `src/sisyphus/predict/non_cyp_substrates.py` | edit | +30 lines (2 loader + 2 lookup + aggregator extension) |
| `src/sisyphus/predict/ivive.py` | edit | ~10 lines net (665 activation + 649-664 comment refresh) |
| `tests/regression/test_ugt_registry_schema.py` | create | ~80 lines |
| `tests/unit/test_non_cyp_substrates.py` | extend | +20 lines |
| `tests/integration/test_ugt_path_mechanism.py` | create | ~40 lines |
| `tests/integration/test_holdout_regression.py` | edit | rename 1 test + new pin value |

Total: 2 new JSON data files, 2 new test files, 4 file edits.

## Registry Schema

Each registry file follows the existing NAT2/UGT1A1 schema:

```json
{
  "version": 1,
  "description": "Substrates of hepatic UGT2B7 with explicit metabolic_fraction. ...",
  "rationale": "...",
  "schema": {
    "drug": "Lowercase common name (informational)",
    "smiles": "Canonical RDKit SMILES",
    "inchikey": "RDKit-derived InChIKey (primary lookup key, full 27-char)",
    "metabolic_fraction": "float in (0, 1]. Fraction of total hepatic CL allocated to this UGT.",
    "literature": "Refs supporting fm choice",
    "notes": "Free-form rationale, including minor-isoform context for Phase 2.x phenotype work"
  },
  "substrates": [ ... ]
}
```

**No schema deviation from NAT2/UGT1A1.** This is a deliberate choice (Approach 1 from brainstorming). Phase 2.x multi-enzyme attribution may extend the schema; Phase 2 does not.

## Per-Drug Allocation Table (Provisional)

The table below lists the 8 seed drugs with **provisional** dominant-UGT attribution and `metabolic_fraction`. Final values are verified against primary literature at implementation PR time (see §"Implementation Verification Gate").

| Drug | Registry | Provisional fm | Primary literature anchor | Minor isoforms (notes only) |
|---|---|:-:|---|---|
| morphine | `ugt2b7_substrates.json` | 0.85 | Coffman 1997 DMD 25:1-4; Court 2003 JPET 305:998 | UGT1A1 ~5% (M3G/M6G ratio) |
| codeine | `ugt2b7_substrates.json` | 0.70 | Court 2003 JPET 305:998 | CYP2D6 ~10% (O-demethyl to morphine) |
| ketorolac | `ugt2b7_substrates.json` | 0.75 | Jett 1999 Pharmacology 58:101 | p-hydroxyl CYP minor |
| indomethacin | `ugt2b7_substrates.json` | 0.15 | Mamiya 2000 DMD 28:1474; Vree 1993 BJCP 35:467 | CYP2C9 O-demethyl ~50% (residual XGBoost CL captures) |
| dapagliflozin | `ugt1a9_substrates.json` | 0.50 | Obermeier 2010 DMD 38:405 | UGT2B7 ~5% (Phase 2.x) |
| etodolac | `ugt1a9_substrates.json` | 0.40 | Tougou 2004 DMD 32:1037 | UGT2B7 ~40% (S-enantiomer; Phase 2.x multi-enzyme) |
| bexagliflozin | `ugt1a9_substrates.json` | 0.40 | Brenzavvy PI 2023 (Theracos/FDA); class-extrapolation from Devineni 2015 (canagliflozin) | UGT2B7 minor |
| glasdegib | `ugt1a9_substrates.json` | 0.15 | Daurismo PI 2018 (Pfizer/FDA) | CYP3A4 ~70% primary (residual XGBoost CL captures) |

### Notes on the table

- **fm values are anchored to literature mid-points**, not tuned. They may be revised to literature low/high bounds during implementation PR review, but only with explicit citation to the supporting paper section.
- **Minor isoforms** are documented in the `notes` field of each entry, NOT split into separate registries (Approach 1 single-source-of-truth).
- **Indomethacin and glasdegib fm < 0.2** are deliberately small. The non-UGT residual is handled by XGBoost CLint default routing (mostly CYP). These entries exist for capability completeness and phenotype-readiness.

## Physiology Abundance

```yaml
# data/physiology/reference_man.yaml — liver enzyme block
UGT2B7: {mean: 2.43e6, cv: 0.5}   # see "Abundance derivation" below
UGT1A9: {mean: 8.10e5, cv: 0.5}   # see "Abundance derivation" below
```

### Abundance derivation

Hepatic enzyme content per ICRP Reference Man liver mass (1,500 g):

```
abundance [pmol] = specific_content [pmol/mg microsomal protein]
                   × MPPGL [mg microsomal protein per g liver]
                   × liver_mass [g]
```

Provisional values:

- UGT2B7: 36 pmol/mg × 45 mg/g × 1,500 g = **2.43e6 pmol**
- UGT1A9: 12 pmol/mg × 45 mg/g × 1,500 g = **8.10e5 pmol**

MPPGL = 45 (Barter 2007, Cmax-relevant central value). Specific contents are provisional from quantitative proteomics literature (Sato 2014 DMD 42:885; Margaillan 2015 DMD 43:1532; Achour 2017 PMC5328673). Implementation verifies which open-access source provides the published values to ≥2 sig figs. If primary sources are paywalled (DE-37 precedent), fall back to class-default conservative values within published ranges (UGT2B7 30-60 pmol/mg, UGT1A9 10-20 pmol/mg).

cv = 0.5 (matches NAT2, UGT1A1, and other liver enzymes already in `reference_man.yaml`).

## Activation

### `src/sisyphus/predict/non_cyp_substrates.py` extension

Add:

```python
_UGT2B7_PATH = _REPO_ROOT / "data" / "enzymes" / "ugt2b7_substrates.json"
_UGT1A9_PATH = _REPO_ROOT / "data" / "enzymes" / "ugt1a9_substrates.json"

@lru_cache(maxsize=1)
def _load_ugt2b7_index() -> dict[str, dict]: ...

@lru_cache(maxsize=1)
def _load_ugt1a9_index() -> dict[str, dict]: ...

def lookup_ugt2b7_substrate(smiles: str) -> dict | None: ...
def lookup_ugt1a9_substrate(smiles: str) -> dict | None: ...
```

Extend `get_non_cyp_fractions`:

```python
def get_non_cyp_fractions(smiles: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for gene, lookup in [
        ("NAT2",    lookup_nat2_substrate),
        ("UGT1A1",  lookup_ugt1a1_substrate),
        ("UGT2B7",  lookup_ugt2b7_substrate),
        ("UGT1A9",  lookup_ugt1a9_substrate),
    ]:
        entry = lookup(smiles)
        if entry is not None:
            out[gene] = float(entry["metabolic_fraction"])
    # existing re-normalization logic unchanged
    return out
```

### `src/sisyphus/predict/ivive.py` edit

Replace lines 649-665 (the disabled-UGT comment block + `ugt_enzymes = None`) with a registry-driven activation:

- Derive a set of UGT tags by calling `get_non_cyp_fractions(profile.smiles)` and filtering to keys with the `UGT` prefix (e.g., `UGT2B7`, `UGT1A9`, and incidentally `UGT1A1` if already present).
- Assign the result to `ugt_enzymes`, or `None` if the set is empty.
- Update the comment to reference B-02 Phase 2 and `dead-ends.md §DE-36` as the baseline.

Implementer determines variable scoping at the call site (either invoke `get_non_cyp_fractions` locally or pass `non_cyp_fractions` deeper from `pipeline/predict.py`). The existing `ugt_enzymes` parameter contract on `build_drug_on_graph` and `_get_fm_fractions` is preserved.

## Tests

### T1 — `tests/regression/test_ugt_registry_schema.py` (new)

Pattern: parallel to `tests/regression/test_oatp_registry_schema.py`.

Three gates:

1. **Schema completeness**: every entry has `drug`, `smiles`, `inchikey`, `metabolic_fraction`, `literature` (non-empty), `notes`.
2. **InChIKey ↔ SMILES consistency**: `Chem.MolToInchiKey(MolFromSmiles(smiles))` equals registered `inchikey`.
3. **`metabolic_fraction` ∈ (0, 1]** per entry.
4. **Cross-registry duplicate check**: no inchikey appears in two or more of {nat2, ugt1a1, ugt2b7, ugt1a9} simultaneously.

### T2 — `tests/unit/test_non_cyp_substrates.py` (extend)

- `lookup_ugt2b7_substrate(morphine_smiles)` returns non-None with `metabolic_fraction == 0.85`.
- `lookup_ugt1a9_substrate(dapagliflozin_smiles)` returns non-None with `metabolic_fraction == 0.50`.
- `get_non_cyp_fractions(morphine_smiles)` returns `{"UGT2B7": 0.85}` (single key).
- `get_non_cyp_fractions(dapagliflozin_smiles)` returns `{"UGT1A9": 0.50}` (single key).
- Non-substrate SMILES (e.g., midazolam) returns `{}`.

### T3 — `tests/integration/test_ugt_path_mechanism.py` (new)

For each seed drug, verify the integrated path:

1. `predict(smiles, dose_mg=...)` succeeds (no exception).
2. The resulting `DrugOnGraph.enzyme_affinity` dict contains the expected UGT key (e.g., morphine has `"UGT2B7"`).
3. Solver completes; `solver_success == True`; `mass_balance_error < 1e-10`.

No specific Cmax pin — that lives in T4. T3 is a mechanism-correctness gate, not a numeric pin.

### T4 — `tests/integration/test_holdout_regression.py` (update)

Rename `test_cached_holdout_aafe_is_2p769` → `test_cached_holdout_aafe_is_2pXXX` where `XXX` is the new post-activation Meta cache value (e.g., `2p770`).

Tolerance stays 0.005 (matches Gate-A).

## Acceptance Gates

### Gate-A (required) — Meta AAFE

`|Meta_post − Meta_pre| < 0.005` where `Meta_pre = 2.7690` (current cache).

Rationale: B-03.x precedent. Bootstrap CI [2.37, 3.26] ⇒ this delta is well within sampling noise. Headline preservation is intentional given B-02 is a capability project.

### Gate-B (informational) — Engine AAFE

DE-36 measured Engine Δ = −0.029. Engine improvement is the affirmative signal that the registry is mechanistically working. **Not a hard gate** — if Engine moves up or stays flat, we proceed (and document) provided Gate-A passes. The asymmetry: Engine regression > 0.05 with Gate-A passing is a yellow flag worth investigating (likely points to abundance mis-calibration).

### Gate-C (required) — Per-drug Cmax variation

Max(|Cmax_post − Cmax_pre| / Cmax_pre) over 107 holdout drugs < 50%.

Rationale: a single drug shifting >50% indicates a wiring bug (e.g., abundance off by 10x, fm mis-assigned). Normal DE-36-class shifts are 5-30%.

### Gate-D (required) — 99-of-107 bit-identical invariance

Of 107 holdout drugs, **only the 8 seed drugs in the new UGT2B7/UGT1A9 registries** may have `|Cmax_post − Cmax_pre| > 1e-8` mg/L. The other 99 drugs MUST be bit-identical to the pre-B-02 cache.

Rationale: post-2026-05-01 Hardening, `realize_means()` is per-node deterministic; `get_non_cyp_fractions` returns `{}` for non-substrate drugs unchanged; adding YAML abundance entries for UGT2B7/UGT1A9 is silent for drugs whose `enzyme_affinity` lacks those tags. Non-UGT-registry drugs therefore MUST be bit-identical. Failure indicates either (a) RNG-order coupling regression (Hardening invariant broken), (b) aggregator wiring bug routing UGT path to non-seed drugs, or (c) YAML edit accidentally modifying a non-UGT enzyme. All are critical wiring bugs.

### Gate-E (required) — Atomic deployment

YAML abundance entries, registry files, `non_cyp_substrates.py` extension, `ivive.py` activation edit, and tests MUST merge in a single PR. Partial deployment is unsafe:

- Registry without YAML abundance → drugs gain `enzyme_affinity["UGT2B7"]` but `liver.enzymes` lacks the tag → engine `KeyError` or silent 0-clearance.
- YAML abundance without registry → harmless but pointless (no drugs use the new path).
- Activation code without registry → harmless (registry lookups return None).

The CI on the merge commit must show all gates A/B/C/D passing.

### Gate-A failure response (anti-fudge procedure)

CLAUDE.md invariant #8 forbids tuning fm to Cmax loss. If Gate-A fails:

1. **First — literature mid-point verification**. For each new registry entry, re-check the fm against the cited paper's reported value. If any fm is outside the paper's stated range, correct it to the mid-point (this is verification, not tuning). Commit message must cite the specific page/table that justifies the change.
2. **Second — drug exclusion, not fm tuning**. If all fms are at their literature mid-points and Gate-A still fails, drop the drug with the largest per-drug |Δ| from the registry. This removes the entry; the drug's UGT path returns to disabled (same as pre-B-02). Repeat until Gate-A passes.
3. **Third — retirement**. If exclusion of 2 or more drugs is needed, retire B-02 to a new DE entry. The finding is: "literature-anchored UGT activation conflicts with the meta-learner's current calibration in a way that single-drug exclusions cannot resolve." This is a real result; document it.

At no point may fm be adjusted to a non-literature value to make a gate pass.

## Implementation Verification Gate

Before merging the implementation PR:

1. **fm values verified** — each registry entry's `metabolic_fraction` is within the range stated in its cited literature anchor. Reviewer checks via the linked paper. If the paper is paywalled and only abstract/PMC is accessible, use the most-cited value from a secondary review (Niemi UGT review, Court 2010 reviews, Lautens DMR 2017). Documented in the `notes` field.
2. **Abundance values verified** — UGT2B7 and UGT1A9 abundance computed from at least one open-access quantitative proteomics paper. If primary sources are paywalled, conservative class-defaults within published ranges are acceptable (documented). "Conservative" here means the **lower bound** of the published specific-content range (smaller abundance ⇒ smaller UGT contribution ⇒ smaller risk of unintended headline AAFE shift; mirrors the anti-fudge bias toward not over-attributing clearance to a new path).
3. **Schema test T1 passes locally** — including cross-registry duplicate check.
4. **All gates (A, B-informational, C) passed** on the regenerated cache.

## Rollback

Phase 2 is atomic with respect to rollback:

- `git revert <merge-commit>` cleanly undoes registry creation, YAML abundance, code activation, and tests in one step.
- Cache file (`4track_holdout_predictions.json`) is restored to the prior canonical state by the revert.
- No external dependencies (DrugBank API, paywalled fetches) — pure local file operations.

## Self-Maintenance Order

Per CLAUDE.md §Self-maintenance:

1. Cache regen via `scripts/run_engine_benchmark.py`.
2. T4 cached AAFE test pin updated to new value.
3. CLAUDE.md headline metrics table — **not updated** if `|ΔMeta| < 0.005` (B-03.x threshold policy). The B-02 phase note may be added to the existing narrative.
4. README §Reproducibility note + §Limitations §UGT — narrative updated to reflect Phase 2 activation; headline table preserved per (3).
5. `docs/claude/experiment-log.md` — new entry at top with date, commit, numeric outcome (Engine Δ, Meta Δ).
6. `docs/claude/dead-ends.md` §DE-36 — append note pointing to this Phase 2 activation as the productive resolution.
7. `docs/claude/backlog.md` §B-02 — strikethrough with closure note.
8. `docs/claude/landmarks.md` — add the 2 new registry files to the file inventory.

If Gate-A fails and exclusion is needed (anti-fudge response step 2), document the excluded drug(s) in both `dead-ends.md` (new DE entry) and `experiment-log.md`.

## Open Questions for Implementation

None at design time. All design decisions are locked. Open questions during implementation are:

1. Which open-access source provides UGT2B7/UGT1A9 specific content values (Sato 2014 vs Margaillan 2015 vs Achour 2017 PMC5328673)?
2. For drugs with literature ranges (e.g., morphine fm 0.70-0.90 across sources), which mid-point to use?
3. If Gate-A fails and exclusion is triggered, which drug exits first?

These are deferred to implementation PR review.
