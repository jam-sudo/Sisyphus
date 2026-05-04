# NAT2 + UGT1A1 Phenotype Propagation (v0.3.2) — Design

**Date**: 2026-05-04
**Issue**: [#10](https://github.com/jam-sudo/Sisyphus/issues/10)
**Target version**: v0.3.2
**Branch**: `feat/nat2-ugt1a1-phenotype`
**Architecture pattern**: registry-keyed by full RDKit InChIKey + schema regression test (PR #22, PR #29, PR #30 lineage)

---

## 1. Goal

Add NAT2 and UGT1A1 to the phenotype scaling infrastructure so that
`predict(SMILES, dose, phenotypes={"NAT2": "PM"})` actually shifts Cmax for
NAT2-substrate drugs (and similarly for UGT1A1). This unblocks GenoADME's
deferred Tier 1 pairs (NAT2/isoniazid CPIC Level B; UGT1A1/raltegravir-class).

The issue scope is *infrastructure capability*, not per-drug PK calibration.
We restore mechanistic correctness of the NAT2/UGT1A1 path; absolute Cmax
fold-error tuning against clinical PK is downstream.

## 2. Background

### 2.1 What's missing today

`src/sisyphus/predict/phenotype.py:38-45` defines `PHENOTYPE_SCALES = {PM: 0.10, IM: 0.50, EM: 1.00, ...}` — gene-blind activity multipliers. `apply_phenotype_to_graph(graph, phenotypes)` walks the dict, looks up the tag in `graph.nodes[node].enzymes` or `graph.nodes[node].transporters`, and scales the abundance Distribution.

The mechanism works for any tag *that exists in the graph YAML AND that the drug has an `enzyme_affinity` entry for*. Today:

- `data/physiology/reference_man.yaml` `liver.enzymes` block contains: `CYP3A4 / CYP2D6 / CYP1A2 / CYP2C9 / CYP2E1 / SPR / CES1 / CES2`. **No NAT2, no UGT1A1.**
- `src/sisyphus/predict/ivive.py:49-60` `_LIVER_ENZYME_ABUNDANCE` constant contains `UGT1A1: 1.215e6` (used by `_decompose_clint`) but no NAT2.
- `_get_fm_fractions` (`ivive.py:148`) handles CYP fm + UGT fm via simple 0.30/0.90 partition based on DrugBank `ugt_enzymes` annotation. **No NAT2 path** — NAT2-substrate drugs (isoniazid, etc.) get all their fm allocated to CYPs (which is wrong).

Result: `apply_phenotype_to_graph(graph, {"NAT2": "PM"})` warns *"tag NAT2 not found"*, and even if the tag were registered, no drug has `enzyme_affinity[NAT2] > 0` so PM scaling has zero PK propagation.

### 2.2 Why this matters

GenoADME deferred two Tier 1 PGx pairs because Sisyphus doesn't represent the gene at all:

- **NAT2/isoniazid** (CPIC Level B). Slow vs rapid acetylator AUC ratio ~3-4× (Ellard 1976 BJCP).
- **UGT1A1/irinotecan** (CPIC Level A) — actually about *SN-38* (active metabolite of irinotecan via CES2) glucuronidation. Parent irinotecan PK is mostly CES2-driven, not UGT1A1. **This case is deferred to issue #11 (prodrug-metabolite phenotype work).**

Adding NAT2/UGT1A1 infrastructure unblocks at minimum the isoniazid pair plus several pure UGT1A1-substrate drugs (raltegravir, atazanavir, dolutegravir).

## 3. Architecture

**Engine: zero changes.** The engine is identity-blind (CLAUDE.md Invariant #1) — it iterates `node.enzymes` keys and calls `drug.enzyme_affinity[tag]`. Adding NAT2/UGT1A1 to YAML and providing matching `enzyme_affinity` entries from ivive is sufficient.

```
predict(SMILES, dose, phenotypes={"NAT2": "PM"})
  │
  ▼
predict.compute_profile(SMILES) → MolecularProfile
predict.predict_adme(profile)   → ADMEProperties (CLint, ...)
  │
  ▼
non_cyp_substrates.get_non_cyp_fractions(SMILES)
  → {"NAT2": 0.90}  (or {} if no match)
  │
  ▼
ivive.build_drug_on_graph(profile, adme, ..., non_cyp_fractions={"NAT2": 0.90})
  → ivive._get_fm_fractions(compound, cyp, ugt, non_cyp_fractions=...)
       → {"NAT2": 0.90, "CYP3A4": 0.05, "CYP2C9": 0.05}
  → ivive._decompose_clint(...)
       → enzyme_affinity = {"NAT2": ..., "CYP3A4": ..., "CYP2C9": ...}
  → DrugOnGraph with enzyme_affinity[NAT2] > 0
  │
  ▼
phenotype.apply_phenotype_to_graph(graph, {"NAT2": "PM"})
  → liver.enzymes["NAT2"].mean × 0.10
  │
  ▼
engine: rate_NAT2 = abundance × enzyme_affinity[NAT2] × ivive_scaling   # graph-blind
```

## 4. Data layer (new)

### 4.1 `data/enzymes/nat2_substrates.json`

```json
{
  "version": 1,
  "description": "Substrates of hepatic cytosolic NAT2 (N-acetyltransferase 2). Used by predict() to allocate metabolic_fraction of XGBoost CLint to NAT2; the residual is split among CYPs per compound_type defaults. NAT2 phenotype scaling (PM/IM/EM/RM via PHENOTYPE_SCALES) then propagates into Cmax via abundance × enzyme_affinity multiplication in the engine.",
  "rationale": "Without registry entry, drug has no NAT2 enzyme_affinity → phenotype scaling is a no-op. This registry is the curation surface for which drugs trigger NAT2 path.",
  "schema": {
    "drug": "Lowercase common name (informational)",
    "smiles": "Canonical SMILES (RDKit). Lookup matches by full InChIKey",
    "inchikey": "RDKit-derived InChIKey (primary lookup)",
    "metabolic_fraction": "float in [0, 1]. Fraction of total hepatic CL allocated to NAT2. Residual split among CYPs per compound_type defaults.",
    "literature": "Refs supporting fm choice",
    "notes": "Free-form rationale"
  },
  "substrates": [
    {
      "drug": "isoniazid",
      "smiles": "NNC(=O)c1ccncc1",
      "inchikey": "QRXWMOHMRWLFEY-UHFFFAOYSA-N",
      "metabolic_fraction": 0.90,
      "literature": [
        "Weber 1983 Drug Metab Rev 14:1163-1205",
        "Ellard 1976 Br J Clin Pharmacol 3:541-7"
      ],
      "notes": "Canonical NAT2 substrate. Slow acetylator t1/2 ~3h, rapid acetylator t1/2 ~1h (Ellard 1976). Minor CYP2E1 contribution to hepatotoxic metabolite hydrazine, not rate-limiting at the whole-organ CL level."
    },
    {
      "drug": "hydralazine",
      "smiles": "NNc1nncc2ccccc12",
      "inchikey": "RPTUSVTUFVMDQK-UHFFFAOYSA-N",
      "metabolic_fraction": 0.50,
      "literature": [
        "Reece 1981 Eur J Clin Pharmacol 19:79-85",
        "Reidenberg 1973 Clin Pharmacol Ther 14:970-7"
      ],
      "notes": "NAT2 ~50%, CYP3A4/2C9 ~50%. Mixed pathway."
    },
    {
      "drug": "procainamide",
      "smiles": "CCN(CC)CCNC(=O)c1ccc(N)cc1",
      "inchikey": "REQCZEXYDRLIBE-UHFFFAOYSA-N",
      "metabolic_fraction": 0.50,
      "literature": [
        "Drayer 1977 Clin Pharmacol Ther 22:14-22",
        "Reidenberg 1972 N Engl J Med 286:419-25"
      ],
      "notes": "NAT2 → N-acetylprocainamide ~50%; renal ~50% (parent + metabolite). PM/EM acetylator distinction primarily affects N-acetyl metabolite ratio."
    }
  ]
}
```

### 4.2 `data/enzymes/ugt1a1_substrates.json`

```json
{
  "version": 1,
  "description": "Substrates of hepatic UGT1A1 with explicit metabolic_fraction. Takes precedence over the existing DrugBank-driven UGT1A1 path in _get_fm_fractions (registry hit → UGT1A1 stripped from DrugBank ugt_enzymes set, fm replaced by registry value). Drugs not in this registry continue using existing DrugBank+0.30 default.",
  "schema": "(same as nat2)",
  "substrates": [
    {
      "drug": "raltegravir",
      "smiles": "Cc1noc(C(C)(C)NC(=O)c2nc(C(=O)NCc3cccc(F)c3)c(O)c(=O)n2C)n1",
      "inchikey": "YVCLRAGXKWQLPW-UHFFFAOYSA-N",
      "metabolic_fraction": 0.70,
      "literature": ["Iwamoto 2008 Clin Pharmacol Ther 83:293-9"],
      "notes": "~70% UGT1A1 glucuronidation, ~30% renal/other. UGT1A1*28 carriers show ~40% AUC increase."
    },
    {
      "drug": "atazanavir",
      "smiles": "<canonical RDKit SMILES from PubChem>",
      "inchikey": "<RDKit-derived>",
      "metabolic_fraction": 0.40,
      "literature": ["Lankisch 2006 Pharmacogenet Genomics 16:495-501"],
      "notes": "Atazanavir is a UGT1A1 INHIBITOR but is itself partially glucuronidated. Mixed CYP3A4 (primary) + UGT1A1 (~40%) metabolism."
    },
    {
      "drug": "dolutegravir",
      "smiles": "<canonical RDKit SMILES from PubChem>",
      "inchikey": "<RDKit-derived>",
      "metabolic_fraction": 0.50,
      "literature": ["Reese 2013 J Acquir Immune Defic Syndr 64:e35-6"],
      "notes": "~50% UGT1A1, ~30% CYP3A4, ~20% UGT1A9 (Reese 2013). UGT1A1*28 effect modest (~30% AUC increase)."
    }
  ]
}
```

**irinotecan EXCLUDED** — parent PK is CES2-driven; UGT1A1 phenotype effect is on SN-38 (active metabolite). Belongs in issue #11 prodrug-metabolite phenotype work, not v0.3.2.

### 4.3 SMILES placeholders

Atazanavir and dolutegravir SMILES strings shown as placeholders. Implementer must derive canonical RDKit SMILES + InChIKey from authoritative source (PubChem/DrugBank) and validate via Task 1 (registry creation) tests.

## 5. Physiology

`data/physiology/reference_man.yaml` `liver.enzymes` block — append at end (after CES2):

```yaml
    NAT2:    {mean: 1.0e7, cv: 0.6}    # Cytosolic. Calibration-arbitrary anchored to Grant 1991 cytosolic activity range; absolute value is back-solved by _decompose_clint such that abundance × affinity × ivive_scaling = NAT2 fm × CLint_hepatic. Independent lognormal (no Achour 2021 matrix entry).
    UGT1A1:  {mean: 1.215e6, cv: 0.5}  # 18 pmol/mg microsomal × 45 MPPGL × 1500g (Achour 2014 PMC4118705). Consistent with ivive.py _LIVER_ENZYME_ABUNDANCE. Independent lognormal.
```

**Position rationale**: end of `liver.enzymes` block — minimizes RNG-order disruption for `BodyGraph.sample(rng=42)` MC paths. Production `predict()` uses `realize_means()` (post-Hardening 2026-05-01) so deterministic path is invariant regardless. `gut_wall` and `kidney` enzyme blocks unchanged.

**No correlation_group**: NAT2 and UGT1A1 are not in Achour 2021 Table S7 covariance matrix. Independent lognormal sampling per H4-style Distribution.

`data/physiology/achour2021_correlation.json`: unchanged.

## 6. Code (predict layer)

### 6.1 New module `src/sisyphus/predict/non_cyp_substrates.py`

```python
"""Non-CYP enzyme substrate registries for NAT2 and UGT1A1 phenotype propagation."""

def lookup_nat2_substrate(smiles: str) -> dict | None:
    """Return registry entry if SMILES InChIKey matches; None otherwise."""

def lookup_ugt1a1_substrate(smiles: str) -> dict | None:
    """Return registry entry if SMILES InChIKey matches; None otherwise."""

def get_non_cyp_fractions(smiles: str) -> dict[str, float]:
    """Return {gene: metabolic_fraction} aggregated across NAT2 + UGT1A1 registries.
    Empty dict if no matches. Caller passes this to _get_fm_fractions."""
```

Module mirrors `transporter_db.py` (PR #29) pattern:
- File-anchored paths via `__file__.resolve().parent.parent.parent` to repo root
- `lru_cache` on JSON loaders
- Full InChIKey matching only (rejects block-1 truncation per spec lessons learned in issue #25)
- Returns None for missing — never raises

### 6.2 Modify `src/sisyphus/predict/ivive.py`

**Constant** `_LIVER_ENZYME_ABUNDANCE`:
```python
"NAT2": 1.0e7,  # consistent with reference_man.yaml NAT2 entry
# UGT1A1: 1_215_000.0  -- already present, no change
```

**Function** `_get_fm_fractions(compound_type, substrate_enzymes, ugt_enzymes, non_cyp_fractions=None)`:

New parameter `non_cyp_fractions: dict[str, float] | None`. Algorithm:

1. If `non_cyp_fractions` empty/None → existing behavior, no change.
2. Otherwise:
   a. `non_cyp_total = sum(non_cyp_fractions.values())`. Cap at 1.0; raise ValueError if any individual value not in [0, 1].
   b. **Strip UGT1A1 from `ugt_enzymes` set if present in `non_cyp_fractions`** (defect A1 fix — registry replaces DrugBank UGT1A1 path).
   c. Compute existing CYP+UGT fm with stripped `ugt_enzymes`, then scale all values by `(1 - non_cyp_total)`.
   d. Add `non_cyp_fractions` entries directly into result dict.
   e. Normalize via `_normalize_fm`.

**Function** `_decompose_clint`: no signature change. NAT2/UGT1A1 are handled automatically because `_get_fm_fractions` returns them in the fm dict, and the existing loop allocates affinity for every key in fm using `enzyme_abundances.get(enzyme, _LIVER_ENZYME_ABUNDANCE.get(enzyme, 1.0))`.

**Function** `build_drug_on_graph(profile, adme, dose_mg, route, ..., non_cyp_fractions=None)`:

Add `non_cyp_fractions` kwarg with default `None`. Forward to `_decompose_clint` via `_get_fm_fractions`. Backward-compatible — existing callers unaffected.

### 6.3 Modify `src/sisyphus/pipeline/predict.py`

After `compute_profile` and `predict_adme`, before `build_drug_on_graph`:

```python
from sisyphus.predict.non_cyp_substrates import get_non_cyp_fractions
non_cyp_fractions = get_non_cyp_fractions(profile.smiles)  # {} or {"NAT2": 0.90, ...}
```

Pass `non_cyp_fractions=non_cyp_fractions` to `build_drug_on_graph`.

## 7. Tests

### 7.1 Unit (`tests/unit/test_non_cyp_substrates.py`)

- `lookup_nat2_substrate("isoniazid SMILES")` returns dict with `metabolic_fraction == 0.90`
- `lookup_nat2_substrate("metoprolol SMILES")` returns None
- `lookup_ugt1a1_substrate("raltegravir SMILES")` returns dict with `metabolic_fraction == 0.70`
- `lookup_ugt1a1_substrate(invalid_smiles)` returns None (no exception)
- `get_non_cyp_fractions("isoniazid")` returns `{"NAT2": 0.90}`
- `get_non_cyp_fractions("raltegravir")` returns `{"UGT1A1": 0.70}`
- `get_non_cyp_fractions("metoprolol")` returns `{}`
- Empty / malformed SMILES → `{}`
- Cache behavior: repeat calls fast (no JSON re-read)

### 7.2 Schema regression (`tests/regression/test_non_cyp_registry_schema.py`)

Three gates per registry (NAT2 + UGT1A1):

1. **Seed list pinned**: matches `_EXPECTED_NAT2_SEED = frozenset({"isoniazid", "hydralazine", "procainamide"})` and `_EXPECTED_UGT1A1_SEED = frozenset({"raltegravir", "atazanavir", "dolutegravir"})`. Catches silent additions.
2. **InChIKey ↔ SMILES**: registered InChIKey matches RDKit canonicalization of SMILES.
3. **fm in [0, 1]**: literature-bounded.
4. **YAML enzyme present**: `liver.enzymes` block contains `NAT2` and `UGT1A1` keys.
5. **Holdout-disjoint gate**: no drug in either registry appears in `data/reference/holdout.json`. Failure message instructs caller to either remove from registry or run holdout regen with invariance check + update gate.

### 7.3 Integration

**`tests/integration/test_phenotype_nat2.py`**:
- `parse_phenotype_spec("NAT2:PM")` → `{"NAT2": "PM"}`
- `apply_phenotype_to_graph(graph, {"NAT2": "PM"})` warning-free; liver.NAT2 mean × 0.10
- `_decompose_clint(isoniazid_clint, ...)` returns dict with `enzyme_affinity["NAT2"] > 0`
- `_decompose_clint(metoprolol_clint, ...)` returns dict with no NAT2 key (or zero)
- End-to-end: `predict(isoniazid_SMILES, 300mg PO, phenotypes={"NAT2": "PM"}).cmax > predict(isoniazid_SMILES, 300mg PO).cmax × 1.3` (gate conservative — Ellard 1976 reports ~3× AUC).

**`tests/integration/test_phenotype_ugt1a1.py`**:
- Mirror NAT2 test for raltegravir
- Gate: `predict(raltegravir, 400mg PO, phenotypes={"UGT1A1": "PM"}).cmax > predict(...).cmax × 1.2` (UGT1A1*28 effect ~40% AUC; Cmax effect smaller)

### 7.4 Holdout invariance (`tests/integration/test_holdout_regression.py`)

No change to test code. Existing pin (Meta 2.679) must hold post-merge. Verified automatically: registry seed contains 0/107 holdout drugs (per Section 5 schema gate).

## 8. Failure modes / decision points

### 8.1 NAT2 abundance literature uncertainty
NAT2 is cytosolic; Grant 1991 + Aklillu 2006 report activity (V_max), not abundance (pmol). **Insight: abundance value is calibration-arbitrary at mean** — `_decompose_clint` back-solves affinity such that `abundance × affinity × ivive_scaling = fm × CLint_total`. Absolute value affects MC variance only via abundance CV propagation. CV=0.6 anchored to Boberg 2017-class hepatic CES uncertainty. 1.0e7 placeholder is order-of-magnitude defensible.

### 8.2 UGT1A1 double-counting (defect A1)
Existing `_get_fm_fractions` allocates UGT1A1 fm via DrugBank `ugt_enzymes` set + 0.30 default. New registry path overrides — **must strip UGT1A1 from ugt_enzymes when registry hit** to avoid double allocation. Test gate verifies: `_get_fm_fractions(compound, cyp_set, {"UGT1A1", "UGT2B7"}, non_cyp_fractions={"UGT1A1": 0.70})` returns `UGT1A1=0.70` (registry wins) and `UGT2B7=expected_default` (preserved).

### 8.3 RNG-order coupling
Adding NAT2/UGT1A1 to YAML shifts `BodyGraph.sample(rng=42)` enzyme iteration order. Production `predict()` uses `realize_means()` (deterministic, post-Hardening 2026-05-01) → bit-identical. MC tests with seed=42 may shift; affected tests should pin or be regenerated.

### 8.4 NAT2 CPIC label semantics
CPIC NAT2 uses "Slow Acetylator (SA) / Rapid Acetylator (RA)"; we map SA→PM, RA→EM/RM. CLI alias support deferred to follow-up. Document in `phenotype.py` docstring + experiment-log.

### 8.5 Phenotype on non-substrate drug
`predict(metoprolol, phenotypes={"NAT2": "PM"})` is safe — metoprolol has no `enzyme_affinity[NAT2]` (registry miss → fm has no NAT2 → no NAT2 affinity entry). `apply_phenotype_to_graph` scales NAT2 abundance by 0.10 but engine multiplies by zero affinity → silent zero. **Already correct via graph-blind invariant; no extra code needed.**

## 9. Scope / out of scope

### In scope (v0.3.2)

- 3 NAT2 substrates + 3 UGT1A1 substrates seeded
- `_get_fm_fractions` extension with `non_cyp_fractions` parameter
- `_LIVER_ENZYME_ABUNDANCE` adds NAT2; YAML adds NAT2 + UGT1A1
- New `non_cyp_substrates.py` module
- 5 test files (unit + schema regression + 2 integration + invariance)
- Engine: 0 line changes
- v0.3.2 / `feat/nat2-ugt1a1-phenotype` branch + PR

### Out of scope (deferred)

- **irinotecan/UGT1A1 PGx pair** — issue #11 prodrug-metabolite phenotype work
- **Per-drug isoniazid Cmax calibration vs Ellard 1976** — issue body explicitly says "calibrating PK effects against clinical data is downstream"
- **CPIC SA/RA → PM/EM CLI alias** — follow-up; not architectural blocker
- **NAT2 in gut/kidney** — minimal expression; liver-only suffices for first cut
- **Refactoring existing UGT2B7/UGT1A4/UGT1A9 to use registry** — keeps existing DrugBank+0.30 default; this PR scopes to NAT2 + UGT1A1 only
- **Validation against full GenoADME deferred-pair test set** — separate PR after infrastructure lands

## 10. Estimated breakdown

~10 tasks (subagent-driven pattern, mirror PR #29 structure):

1. Create `nat2_substrates.json` + tests (registry + schema validators only)
2. Create `ugt1a1_substrates.json` + tests
3. Create `non_cyp_substrates.py` loader module + unit tests
4. Add NAT2/UGT1A1 to `reference_man.yaml`; add NAT2 to `_LIVER_ENZYME_ABUNDANCE`
5. Extend `_get_fm_fractions` with `non_cyp_fractions` parameter (handle UGT1A1 strip)
6. Extend `build_drug_on_graph` and `pipeline.predict.predict()` to pass non_cyp_fractions through
7. Schema regression test (`test_non_cyp_registry_schema.py`)
8. Integration: NAT2 phenotype propagation end-to-end (isoniazid)
9. Integration: UGT1A1 phenotype propagation end-to-end (raltegravir)
10. Holdout invariance verification + experiment-log entry + branch push + PR

## 11. Acceptance criteria

- [ ] `apply_phenotype_to_graph(graph, {"NAT2": "PM"})` does NOT emit "tag not found" warning
- [ ] `predict(isoniazid, dose, phenotypes={"NAT2": "PM"}).cmax > predict(...).cmax × 1.3`
- [ ] `predict(raltegravir, dose, phenotypes={"UGT1A1": "PM"}).cmax > predict(...).cmax × 1.2`
- [ ] 107-holdout AAFE bit-identical to 2.679 (Meta), 3.791 (Engine), 3.012 (ML)
- [ ] All schema gates pass (3 per registry × 2 registries + 2 cross-cutting)
- [ ] Engine code: 0 line diff
- [ ] Existing `test_oatp_registry_schema.py` (PR #29 schema gate) untouched and passing
- [ ] CI green on all merged-to-main paths

## 12. References

- Issue [#10](https://github.com/jam-sudo/Sisyphus/issues/10)
- PR #22 (`598358b`) — pravastatin metabolic_fraction registry pattern (precedent)
- PR #29 (`88aa618`) — ECM auto-activation gating + schema regression test pattern
- PR #30 (`2f8bdc6`) — pitavastatin promotion + retroactive caveat documentation pattern
- CLAUDE.md Invariants 1, 2, 6 (engine identity-blind, distributions, no drug-specific branches)
- Caudle 2017 CPIC for PHENOTYPE_SCALES convention
- Cooper-DeHoff 2022 SLCO1B1 → PM/IM/EM mapping (existing precedent in `phenotype.py`)
- 2026-05-01 Hardening: `realize_means()` deterministic path (eliminates RNG-order coupling for production `predict()`)
