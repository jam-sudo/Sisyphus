# NAT2 + UGT1A1 Phenotype Propagation (v0.3.2) — Design v3

**Date**: 2026-05-04 (v3 revised post empirical defect discovery)
**Issue**: [#10](https://github.com/jam-sudo/Sisyphus/issues/10)
**Target version**: v0.3.2
**Branch**: `feat/nat2-ugt1a1-phenotype`
**Architecture pattern**: registry-keyed by full RDKit InChIKey + schema regression test (PR #22, PR #29, PR #30 lineage) + latent-bug fix in `pipeline/predict.py`

---

## 0. v3 revision summary (vs v2)

Empirical verification surfaced a critical defect: **the existing CYP/UGT/NAT phenotype scaling produces zero PK effect** because `_decompose_clint` back-solves affinity from abundance, and `pipeline/predict.py` rebuilds the drug *after* phenotype application — so the rebuild cancels the scaling exactly. SLCO1B1 escapes the cancellation only because OATP1B1 uses saturable Michaelis-Menten kinetics (no back-solve).

Empirical confirmation:
- caffeine + `phenotypes={"CYP1A2": "PM"}` → PM/EM Cmax ratio = **1.0000×** (broken)
- pravastatin + `phenotypes={"SLCO1B1": "PM"}` → PM/EM Cmax ratio = **3.034×** (works; transporter path bypasses back-solve)

Without fixing this, NAT2/UGT1A1 acceptance criteria §11 #2 #3 are mathematically impossible to satisfy.

v3 expands scope to v0.3.2 = **NAT2/UGT1A1 substrate registry + back-solve cancellation fix**. Tasks count: 12 (was 10).

Other v2→v3 deltas:
- §8.2 "UGT1A1 double-counting" deleted — current code hardcodes `ugt_enzymes = None` (`ivive.py:611`, "UGT fm redistribution disabled" sensitivity result), so the double-counting risk is non-existent.
- raltegravir SMILES added to placeholder list — my v2 string `Cc1noc(...)n1` is 1,2,4-oxadiazole, but PubChem canonical is 1,3,4-oxadiazole (`CC1=NN=C(O1)...`); InChIKey neither matched true raltegravir, so all 4 SMILES (raltegravir, atazanavir, dolutegravir, hydralazine) deserve Task 1 implementer re-derivation from authoritative source.
- §6.2 `non_cyp_total > 1` handling: re-normalize (each / total) so sum=1.0 exactly, log.INFO. Round-off tolerance, no ValueError.

---

## 1. Goal

Add NAT2 and UGT1A1 to the phenotype scaling infrastructure so that
`predict(SMILES, dose, phenotypes={"NAT2": "PM"})` actually shifts Cmax for
NAT2-substrate drugs (and similarly for UGT1A1). Fix the underlying back-solve
cancellation bug so that **all** CYP/UGT/NAT phenotypes propagate correctly,
not just SLCO1B1.

This unblocks GenoADME's deferred Tier 1 pairs (NAT2/isoniazid CPIC Level B;
UGT1A1/raltegravir-class) AND retroactively activates the previously-broken
CYP2D6/CYP2C9/CYP1A2/etc. phenotype paths.

The issue scope is *infrastructure capability*, not per-drug PK calibration.
Mechanistic correctness; absolute Cmax fold-error tuning is downstream.

## 2. Background

### 2.1 What's missing today (data layer)

`src/sisyphus/predict/phenotype.py:38-45` defines `PHENOTYPE_SCALES = {PM: 0.10, IM: 0.50, EM: 1.00, ...}` — gene-blind activity multipliers. `apply_phenotype_to_graph(graph, phenotypes)` walks the dict, looks up the tag in `graph.nodes[node].enzymes` or `graph.nodes[node].transporters`, and scales the abundance Distribution.

The mechanism works for any tag *that exists in the graph YAML AND that the drug has an `enzyme_affinity` entry for*. Today:

- `data/physiology/reference_man.yaml` `liver.enzymes` block: `CYP3A4 / CYP2D6 / CYP1A2 / CYP2C9 / CYP2E1 / SPR / CES1 / CES2`. **No NAT2, no UGT1A1.**
- `src/sisyphus/predict/ivive.py:49-60` `_LIVER_ENZYME_ABUNDANCE` constant has `UGT1A1: 1.215e6` but no NAT2.
- `_get_fm_fractions` (`ivive.py:148`): handles CYP fm + UGT fm via 0.30/0.90 partition based on DrugBank `ugt_enzymes` annotation. **No NAT2 path.**
- `build_drug_on_graph` line 611: `ugt_enzymes = None` hardcoded — the UGT path is currently dead code, disabled after a sensitivity test (commit not located; comment says "UGT fm redistribution disabled — engine AAFE degradation 2.861→3.090"). My new registry path operates independently; no double-counting risk.

### 2.2 What's broken today (latent bug)

**`pipeline/predict.py:251-269` cancels phenotype scaling for any enzyme that goes through `_decompose_clint`.** Trace:

```
1. graph built from YAML, liver.enzymes[CYP1A2].mean = 3.0375e6
2. apply_phenotype_to_graph(graph, {"CYP1A2": "PM"}) → mean *= 0.10 → 3.0375e5
3. Line 261: liver_enzymes = {"CYP1A2": 3.0375e5, ...}  ← reads SCALED abundance
4. Line 263: build_drug_on_graph(..., liver_enzymes=liver_enzymes)
5. Inside _decompose_clint:
   affinity_CYP1A2 = (drug.CLint × fm) / (3.0375e5 × ivive)   ← affinity 10× larger
6. Engine: rate = abundance × affinity × ivive
   = 3.0375e5 × [10× original affinity] × ivive
   = 3.0375e6 × [original affinity] × ivive   (mathematically identical to no-phenotype)
   = drug.CLint × fm   ← phenotype effect: ZERO
```

SLCO1B1 phenotype escapes this because OATP1B1's flux is `Vmax × C / (Km + C)` where Vmax = abundance × kcat (kcat is fixed, not back-solved). Scaled abundance → scaled Vmax → scaled rate. Empirical confirmation: pravastatin + SLCO1B1:PM gives 3.034× ratio matching Niemi 2009 PM/EM ~3×.

### 2.3 Why this matters

GenoADME deferred two Tier 1 PGx pairs because Sisyphus doesn't represent the gene at all:

- **NAT2/isoniazid** (CPIC Level B). Slow vs rapid acetylator AUC ratio ~3-4× (Ellard 1976 BJCP).
- **UGT1A1/irinotecan** (CPIC Level A) — actually about *SN-38* (active metabolite of irinotecan via CES2) glucuronidation. Parent irinotecan PK is mostly CES2-driven, not UGT1A1. **Deferred to issue #11 prodrug-metabolite phenotype work.**

Beyond the deferred list, fixing the back-solve bug retroactively activates:
- **CYP2D6/codeine** (CPIC Level A): codeine → morphine bioactivation requires CYP2D6
- **CYP2C19/voriconazole** (CPIC Level A)
- **CYP2C9/warfarin/phenytoin** (CPIC Level A)
- All other CYP-PGx pairs that previously appeared to "support phenotypes" but had silent zero effect

## 3. Architecture

**Engine: zero changes.** Engine is identity-blind (CLAUDE.md Invariant #1).

**Pipeline: refactor `predict()` to read pre-phenotype enzyme abundances** for affinity back-solve, then apply phenotype to graph. Engine multiplication then propagates phenotype as `scaled_abundance × pre_phenotype_affinity = scale × original_rate`.

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
graph = build_from_yaml(reference_man.yaml)
liver_enzymes_pre = {tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()}
                    # ← pre-phenotype snapshot
  │
  ▼ (only if phenotypes given)
graph = apply_phenotype_to_graph(graph, {"NAT2": "PM"})
  → liver.enzymes["NAT2"].mean × 0.10
  │
  ▼
ivive.build_drug_on_graph(profile, adme, ...,
                           liver_enzymes=liver_enzymes_pre,    # ← pre-phenotype
                           non_cyp_fractions={"NAT2": 0.90})
  → _decompose_clint(...) uses liver_enzymes_pre to back-solve affinity
  → drug.enzyme_affinity[NAT2] computed from PRE-phenotype abundance
  │
  ▼
engine: rate_NAT2 = scaled_abundance × pre_affinity × ivive_scaling
       = (0.10 × A_pre) × ((CLint × fm) / (A_pre × ivive)) × ivive
       = 0.10 × CLint × fm        ← phenotype propagates correctly
```

## 4. Data layer (new)

### 4.1 `data/enzymes/nat2_substrates.json`

```json
{
  "version": 1,
  "description": "Substrates of hepatic cytosolic NAT2 (N-acetyltransferase 2). Used by predict() to allocate metabolic_fraction of XGBoost CLint to NAT2; the residual is split among CYPs per compound_type defaults. NAT2 phenotype scaling propagates into Cmax via abundance × enzyme_affinity multiplication in the engine, predicated on the pipeline-level back-solve cancellation fix in v0.3.2.",
  "schema": {
    "drug": "Lowercase common name (informational)",
    "smiles": "Canonical RDKit SMILES (Task 1 implementer derives from authoritative source: PubChem CID or DrugBank)",
    "inchikey": "RDKit-derived InChIKey (primary lookup key)",
    "metabolic_fraction": "float in [0, 1]. Fraction of total hepatic CL allocated to NAT2.",
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
        "Ellard 1976 Br J Clin Pharmacol 3:541-7",
        "Peloquin 1999 Chest 115:12-8"
      ],
      "notes": "Canonical NAT2 substrate. Slow acetylator t1/2 ~3h, rapid acetylator t1/2 ~1h (Ellard 1976). Cmax PM/EM ~1.3-1.5× at 300mg PO (Peloquin 1999). Minor CYP2E1 contribution to hepatotoxic metabolite hydrazine, not rate-limiting at the whole-organ CL level."
    },
    {
      "drug": "hydralazine",
      "smiles": "<Task 1: derive canonical from PubChem CID 3637>",
      "inchikey": "<RDKit-derived>",
      "metabolic_fraction": 0.50,
      "literature": [
        "Reece 1981 Eur J Clin Pharmacol 19:79-85",
        "Reidenberg 1973 Clin Pharmacol Ther 14:970-7"
      ],
      "notes": "NAT2 ~50%, CYP3A4/2C9 ~50%. Mixed pathway. NB: my v2 spec used 'NNc1nncc2ccccc12' (RPTUSVTUFVMDQK-UHFFFAOYSA-N) — Task 1 must verify this against PubChem canonical."
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
  "description": "Substrates of hepatic UGT1A1 with explicit metabolic_fraction. The current build_drug_on_graph hardcodes ugt_enzymes=None (UGT path disabled per ivive.py:611 sensitivity result), so this registry is the only path that creates a non-zero enzyme_affinity[UGT1A1] for listed drugs.",
  "schema": "(same as nat2)",
  "substrates": [
    {
      "drug": "raltegravir",
      "smiles": "<Task 1: derive canonical from PubChem CID 54671008>",
      "inchikey": "<RDKit-derived>",
      "metabolic_fraction": 0.70,
      "literature": ["Iwamoto 2008 Clin Pharmacol Ther 83:293-9"],
      "notes": "~70% UGT1A1 glucuronidation, ~30% renal/other. UGT1A1*28 carriers ~40% AUC increase. NB: my v2 spec SMILES 'Cc1noc(...)n1' is 1,2,4-oxadiazole; PubChem canonical is 1,3,4-oxadiazole 'CC1=NN=C(O1)...'. Task 1 must verify."
    },
    {
      "drug": "atazanavir",
      "smiles": "<Task 1: derive canonical from PubChem CID 148192>",
      "inchikey": "<RDKit-derived>",
      "metabolic_fraction": 0.40,
      "literature": ["Lankisch 2006 Pharmacogenet Genomics 16:495-501"],
      "notes": "Atazanavir is a UGT1A1 INHIBITOR but is itself partially glucuronidated. Mixed CYP3A4 (primary) + UGT1A1 (~40%) metabolism."
    },
    {
      "drug": "dolutegravir",
      "smiles": "<Task 1: derive canonical from PubChem CID 54726191>",
      "inchikey": "<RDKit-derived>",
      "metabolic_fraction": 0.50,
      "literature": ["Reese 2013 J Acquir Immune Defic Syndr 64:e35-6"],
      "notes": "~50% UGT1A1, ~30% CYP3A4, ~20% UGT1A9 (Reese 2013). UGT1A1*28 effect modest (~30% AUC increase)."
    }
  ]
}
```

**irinotecan EXCLUDED** — parent PK is CES2-driven; UGT1A1 phenotype effect is on SN-38. Belongs in issue #11 prodrug-metabolite phenotype work.

### 4.3 SMILES placeholder policy

`<Task 1: derive ...>` placeholders for hydralazine, raltegravir, atazanavir, dolutegravir. **Implementer derives canonical RDKit SMILES + InChIKey from authoritative source (PubChem CID listed)** in Task 1, then schema regression test (§7.2 gate 2) validates `RDKit-canonicalize(SMILES) → InChIKey == registered InChIKey`. isoniazid and procainamide pre-validated by spec author (RDKit roundtrip OK).

## 5. Physiology

`data/physiology/reference_man.yaml` `liver.enzymes` block — append at end (after CES2):

```yaml
    NAT2:    {mean: 1.0e7, cv: 0.6}    # Cytosolic. Calibration-arbitrary anchored to Grant 1991 cytosolic activity range; absolute value is back-solved by _decompose_clint such that abundance × affinity × ivive_scaling = NAT2 fm × CLint_hepatic. Independent lognormal (no Achour 2021 matrix entry).
    UGT1A1:  {mean: 1.215e6, cv: 0.5}  # 18 pmol/mg microsomal × 45 MPPGL × 1500g (Achour 2014 PMC4118705). Consistent with ivive.py _LIVER_ENZYME_ABUNDANCE. Independent lognormal.
```

**Position rationale**: end of `liver.enzymes` block — minimizes RNG-order disruption for `BodyGraph.sample(rng=42)` MC paths. Production `predict()` uses `realize_means()` (post-Hardening 2026-05-01) so deterministic path is invariant regardless. `gut_wall` and `kidney` enzyme blocks unchanged.

**No correlation_group**: NAT2 and UGT1A1 not in Achour 2021 Table S7 covariance matrix. Independent lognormal sampling.

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
    """Return {gene: metabolic_fraction} aggregated across NAT2 + UGT1A1.
    Empty dict if no matches. If aggregated total > 1.0 (round-off or curation
    error), re-normalize so values sum to 1.0; emit logger.info note."""
```

Mirrors `transporter_db.py` (PR #29):
- File-anchored paths via `__file__.resolve().parent.parent.parent` to repo root
- `lru_cache` on JSON loaders
- Full InChIKey matching (rejects block-1 truncation per issue #25 lessons)
- Returns None for missing — never raises

### 6.2 Modify `src/sisyphus/predict/ivive.py`

**Constant** `_LIVER_ENZYME_ABUNDANCE`: add `"NAT2": 1.0e7` (consistent with YAML).

**Function** `_get_fm_fractions(compound_type, substrate_enzymes, ugt_enzymes, non_cyp_fractions=None)`:

New parameter `non_cyp_fractions: dict[str, float] | None`. Algorithm:

1. If `non_cyp_fractions` empty/None → existing behavior, no change.
2. Otherwise:
   a. Validate each value in [0, 1]; raise ValueError if not (curation enforces this).
   b. `non_cyp_total = sum(non_cyp_fractions.values())`.
   c. If `non_cyp_total > 1.0` (round-off or multi-substrate overlap): re-normalize → divide each by `non_cyp_total`; log.info.
   d. `remaining_for_cyp_ugt = max(1.0 - non_cyp_total, 0.0)`.
   e. Compute existing CYP+UGT fm via current logic. Scale all values by `remaining_for_cyp_ugt`.
   f. Add `non_cyp_fractions` entries directly into result dict.
   g. `_normalize_fm` (no-op if already normalized; defensive against round-off).

**Function** `_decompose_clint`: no signature change. NAT2/UGT1A1 are handled automatically via the fm dict + `enzyme_abundances.get(...)` lookup pattern.

**Function** `build_drug_on_graph(profile, adme, dose_mg, route, ..., non_cyp_fractions=None)`:

Add `non_cyp_fractions` kwarg with default `None`. Forward to `_decompose_clint` via `_get_fm_fractions`. Backward-compatible — existing callers unaffected.

### 6.3 Modify `src/sisyphus/pipeline/predict.py` (registry wiring)

After `compute_profile` and before initial `build_drug_on_graph`:

```python
from sisyphus.predict.non_cyp_substrates import get_non_cyp_fractions
non_cyp_fractions = get_non_cyp_fractions(profile.smiles)  # {} or {"NAT2": 0.90, ...}
```

Pass `non_cyp_fractions=non_cyp_fractions` to both `build_drug_on_graph` calls (line 202 initial + line 263 rebuild).

### 6.4 Modify `src/sisyphus/pipeline/predict.py` (back-solve cancellation fix — CRITICAL)

**Current (broken) flow**:
```python
graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")
if phenotypes:
    graph = apply_phenotype_to_graph(graph, phenotypes)
if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
    liver_enzymes = {tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(..., liver_enzymes=liver_enzymes, ...)  # uses scaled abundance
```

**Fixed flow**:
```python
graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")

# CRITICAL: snapshot pre-phenotype abundances. The IVIVE _decompose_clint back-solves
# affinity from abundance, so passing scaled abundances would cause phenotype scaling
# to cancel out at engine multiplication time. Snapshot BEFORE phenotype application
# so the affinity back-solve uses the unscaled baseline; phenotype then propagates
# through the engine as scaled_abundance × pre_affinity = scale × original_rate.
liver_enzymes_pre: dict[str, float] | None = None
if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
    liver_enzymes_pre = {tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()}

if phenotypes:
    graph = apply_phenotype_to_graph(graph, phenotypes)

# Rebuild drug with PRE-phenotype abundances. Phenotype's effect on graph remains
# (scaled abundances flow into the engine), but affinity is computed from the
# unscaled baseline so engine multiplication propagates the scaling.
if liver_enzymes_pre is not None:
    drug = build_drug_on_graph(..., liver_enzymes=liver_enzymes_pre, non_cyp_fractions=non_cyp_fractions, ...)
```

The diff is small (~5 lines moved) but the impact is large — it activates ALL CYP/UGT/NAT phenotype propagation, not just NAT2/UGT1A1.

## 7. Tests

### 7.1 Unit (`tests/unit/test_non_cyp_substrates.py`)

- `lookup_nat2_substrate("isoniazid SMILES")` returns dict with `metabolic_fraction == 0.90`
- `lookup_nat2_substrate("metoprolol SMILES")` returns None
- `lookup_ugt1a1_substrate("raltegravir SMILES post-Task1")` returns dict with `metabolic_fraction == 0.70`
- `lookup_ugt1a1_substrate(invalid_smiles)` returns None (no exception)
- `get_non_cyp_fractions("isoniazid")` returns `{"NAT2": 0.90}`
- `get_non_cyp_fractions("raltegravir")` returns `{"UGT1A1": 0.70}`
- `get_non_cyp_fractions("metoprolol")` returns `{}`
- Empty / malformed SMILES → `{}` no exception
- Cache behavior: repeat calls do not re-read JSON (lru_cache hit)
- Sum-overflow re-normalization: synthesize fixture where NAT2=0.7 + UGT1A1=0.5 totals 1.2 → expect normalized 0.583 / 0.417

### 7.2 Schema regression (`tests/regression/test_non_cyp_registry_schema.py`)

Per registry (NAT2 + UGT1A1):

1. **Seed list pinned**: `_EXPECTED_NAT2 = frozenset({"isoniazid", "hydralazine", "procainamide"})`, `_EXPECTED_UGT1A1 = frozenset({"raltegravir", "atazanavir", "dolutegravir"})`. Catches silent additions.
2. **InChIKey ↔ SMILES**: registered InChIKey matches RDKit canonicalization of SMILES (also catches Task 1 placeholder leaks if implementer forgets to fill in).
3. **fm in [0, 1]**: literature-bounded.
4. **YAML enzyme present**: `liver.enzymes` block contains `NAT2` and `UGT1A1`.
5. **Holdout-disjoint gate**: no drug in either registry appears in `data/reference/holdout.json`. Failure message instructs caller to either remove from registry OR run holdout regen with invariance check + update gate.

### 7.3 Integration

**`tests/integration/test_phenotype_nat2.py`**:
- `parse_phenotype_spec("NAT2:PM")` → `{"NAT2": "PM"}`
- `apply_phenotype_to_graph(graph, {"NAT2": "PM"})` warning-free; liver.NAT2 mean × 0.10
- `_decompose_clint(isoniazid_clint, ...)` returns dict with `enzyme_affinity["NAT2"] > 0`
- `_decompose_clint(metoprolol_clint, ...)` returns dict with no NAT2 key
- **End-to-end PM/EM gate**: `predict(isoniazid, 300mg PO, phenotypes={"NAT2": "PM"}).cmax > predict(...).cmax × 1.3` (Ellard 1976 ~3× AUC; Cmax effect smaller, conservative 1.3× gate).

**`tests/integration/test_phenotype_ugt1a1.py`**:
- Mirror NAT2 test for raltegravir
- Gate: `predict(raltegravir, 400mg PO, phenotypes={"UGT1A1": "PM"}).cmax > predict(...).cmax × 1.2` (UGT1A1*28 ~40% AUC; Cmax effect smaller)

**`tests/integration/test_phenotype_cyp_propagation.py` (NEW — back-solve fix regression)**:
Covers the previously-broken CYP path. At minimum:
- caffeine + `phenotypes={"CYP1A2": "PM"}` → Cmax_PM > Cmax_EM × 1.5 (CYP1A2 fm ~80% caffeine; PM scaling × 0.10 → total CL drops to ~30% → Cmax rises ~3×; gate conservative)
- midazolam + `phenotypes={"CYP3A5": "PM"}` → Cmax_PM ≥ Cmax_EM × 1.05 (3A5 minor; small expected effect, gate just above noise)
- S-warfarin + `phenotypes={"CYP2C9": "PM"}` → Cmax_PM > Cmax_EM × 1.3 (2C9 ~80% S-warfarin)
- pravastatin + `phenotypes={"SLCO1B1": "PM"}` → Cmax_PM > Cmax_EM × 2.5 (Niemi 2009 ~3×; existing path stays working — regression test for the transporter case)

This test file is the empirical evidence that v0.3.2 fixes the back-solve cancellation. **If this test suite passes, the latent bug is gone.**

### 7.4 Holdout invariance (`tests/integration/test_holdout_regression.py`)

No change to test code. Existing pin (Meta 2.679) holds post-merge.

Verified automatically:
- Holdout benchmark (`scripts/run_engine_benchmark.py`) does not pass `phenotypes` argument → default None → no behavior change from §6.4 fix.
- Registry seed contains 0/107 holdout drugs → no behavior change from §6.3 wiring.

Empirically run as the final task before PR push (regen → diff against current `data/training/4track_holdout_predictions.json` → bit-identical Meta/Engine/ML/InDomain).

## 8. Failure modes / decision points

### 8.1 NAT2 abundance literature uncertainty
NAT2 is cytosolic; Grant 1991 + Aklillu 2006 report activity (V_max), not abundance (pmol). **Insight: abundance value is calibration-arbitrary at mean** — `_decompose_clint` back-solves affinity such that `abundance × affinity × ivive_scaling = fm × CLint_total` (pre-phenotype, post-fix). Absolute value affects MC variance only. CV=0.6 anchored to Boberg 2017-class uncertainty. 1.0e7 placeholder is order-of-magnitude defensible.

### 8.2 ~~UGT1A1 double-counting~~ (deleted in v3)
Current code hardcodes `ugt_enzymes = None` (`build_drug_on_graph:611`) — the existing UGT path is dead. New registry path is the only UGT1A1 path. No double-counting risk.

### 8.3 RNG-order coupling
Adding NAT2/UGT1A1 to YAML shifts `BodyGraph.sample(rng=42)` enzyme iteration order. Production `predict()` uses `realize_means()` → bit-identical. MC tests with seed=42 may shift; affected tests should pin or be regenerated. Spec §A7 noted same risk; no test currently relies on seed=42 enzyme order with these tags absent.

### 8.4 NAT2 CPIC label semantics
CPIC NAT2 uses "Slow Acetylator (SA) / Rapid Acetylator (RA)"; we map SA→PM, RA→EM/RM. CLI alias deferred to follow-up. Document in `phenotype.py` docstring + experiment-log.

### 8.5 Phenotype on non-substrate drug
`predict(metoprolol, phenotypes={"NAT2": "PM"})`: metoprolol has no `enzyme_affinity[NAT2]` (registry miss → fm has no NAT2 → no NAT2 affinity entry). Engine multiplies scaled abundance by zero affinity → silent zero. **Already correct via graph-blind invariant.**

### 8.6 Back-solve fix scope (v3 critical)
The §6.4 fix changes behavior for ANY existing user passing CYP phenotypes. Risk:
- `tests/unit/test_pipeline_phenotypes.py` and `tests/unit/test_phenotype.py`: only test invocation patterns and graph-level scaling, NOT pipeline-level Cmax magnitude. No assertions on PM/EM ratio. **Safe.**
- `scripts/diagnose_pravastatin_ecm.py`: uses SLCO1B1 (transporter path, untouched by fix). **Safe.**
- 107-holdout benchmark: no phenotype usage. **Safe.**
- External users (none documented in repo): may observe phenotype actually has effect now. This is the *intended* behavior; the previous zero-effect was the bug.

### 8.7 Hardening interaction (`realize_means()`)
`realize_means()` operates on the graph after phenotype application. Pre-phenotype snapshot must happen before `apply_phenotype_to_graph`. The fix does not interact with `realize_means()` semantics.

## 9. Scope / out of scope

### In scope (v0.3.2)

- **Back-solve cancellation fix** in `pipeline/predict.py:251-269` (latent bug) — Task 4.5
- 3 NAT2 substrates + 3 UGT1A1 substrates seeded
- `_get_fm_fractions` extension with `non_cyp_fractions` parameter
- `_LIVER_ENZYME_ABUNDANCE` adds NAT2; YAML adds NAT2 + UGT1A1
- New `non_cyp_substrates.py` module
- 6 test files (unit + schema + 2 integration NAT2/UGT1A1 + 1 integration CYP propagation regression + invariance)
- Engine: 0 line changes
- v0.3.2 / `feat/nat2-ugt1a1-phenotype` branch + PR

### Out of scope (deferred)

- **irinotecan/UGT1A1 PGx pair** — issue #11 prodrug-metabolite phenotype work
- **Per-drug isoniazid Cmax calibration vs Ellard 1976** — issue body explicitly downstream
- **CPIC SA/RA → PM/EM CLI alias** — follow-up; not architectural blocker
- **NAT2 in gut/kidney** — minimal expression; liver-only suffices
- **Refactoring existing UGT2B7/UGT1A4/UGT1A9 to use registry** — keeps existing dead-code state
- **Reactivating `ugt_enzymes` DrugBank path** — separate decision, requires sensitivity rerun
- **Validation against full GenoADME deferred-pair test set** — separate PR after infrastructure lands

## 10. Estimated breakdown (v3: 12 tasks)

Subagent-driven pattern, mirror PR #29:

1. Create `nat2_substrates.json`: derive canonical SMILES + InChIKey for hydralazine via PubChem CID 3637; isoniazid + procainamide pre-validated. Validators tests (RDKit roundtrip, fm bounds).
2. Create `ugt1a1_substrates.json`: derive canonical SMILES + InChIKey for raltegravir/atazanavir/dolutegravir from PubChem CIDs. Validators tests.
3. Create `non_cyp_substrates.py` loader module + unit tests (lookup, get_non_cyp_fractions, sum-overflow normalization, lru_cache behavior).
4. Add NAT2/UGT1A1 to `reference_man.yaml`; add NAT2 to `_LIVER_ENZYME_ABUNDANCE` constant. Verify constant ↔ YAML consistency.
5. **Back-solve cancellation fix** in `pipeline/predict.py:251-269` — pre-phenotype enzyme snapshot. (Was: Task 4.5 in spec; promoted to dedicated task #5.)
6. Extend `_get_fm_fractions` with `non_cyp_fractions` parameter (sum-cap re-normalize, fm composition).
7. Extend `build_drug_on_graph` with `non_cyp_fractions` kwarg; thread through `_get_fm_fractions` → `_decompose_clint`.
8. Pipeline wiring in `pipeline.predict.predict()`: `get_non_cyp_fractions(profile.smiles)` → pass to both `build_drug_on_graph` calls (line 202 + line 263 in current code).
9. Schema regression test (`test_non_cyp_registry_schema.py`) — 5 gates per registry × 2 + holdout-disjoint cross-cutting.
10. Integration NAT2 (`test_phenotype_nat2.py`): Cmax PM/EM ratio gate via isoniazid.
11. Integration UGT1A1 (`test_phenotype_ugt1a1.py`): Cmax PM/EM ratio gate via raltegravir.
12. **Integration CYP propagation regression** (`test_phenotype_cyp_propagation.py`): caffeine/CYP1A2, midazolam/CYP3A5, S-warfarin/CYP2C9, pravastatin/SLCO1B1 (the latter as transporter-path regression backstop).

Holdout invariance verification + experiment-log entry + branch push + PR is the final session-level operation, not a numbered task.

## 11. Acceptance criteria

- [ ] `apply_phenotype_to_graph(graph, {"NAT2": "PM"})` does NOT emit "tag not found" warning
- [ ] `predict(isoniazid, dose, phenotypes={"NAT2": "PM"}).cmax > predict(...).cmax × 1.3`
- [ ] `predict(raltegravir, dose, phenotypes={"UGT1A1": "PM"}).cmax > predict(...).cmax × 1.2`
- [ ] **`predict(caffeine, dose, phenotypes={"CYP1A2": "PM"}).cmax > predict(...).cmax × 1.5`** (back-solve fix regression)
- [ ] `predict(pravastatin, dose, phenotypes={"SLCO1B1": "PM"}).cmax > predict(...).cmax × 2.5` (transporter path stays working)
- [ ] 107-holdout AAFE bit-identical to 2.679 (Meta), 3.791 (Engine), 3.012 (ML), 2.733 (In-domain Meta)
- [ ] All schema gates pass (5 per registry × 2 registries + 2 cross-cutting)
- [ ] Engine code: 0 line diff
- [ ] Existing `test_oatp_registry_schema.py` (PR #29 schema gate) untouched and passing
- [ ] CI green on all merged-to-main paths

## 12. References

- Issue [#10](https://github.com/jam-sudo/Sisyphus/issues/10)
- PR #22 (`598358b`) — pravastatin metabolic_fraction registry pattern (precedent)
- PR #29 (`88aa618`) — ECM auto-activation gating + schema regression test pattern
- PR #30 (`2f8bdc6`) — pitavastatin promotion + retroactive caveat documentation pattern
- CLAUDE.md Invariants 1, 2, 6 (engine identity-blind, distributions, no drug-specific branches)
- Caudle 2017 CPIC PHENOTYPE_SCALES convention
- Cooper-DeHoff 2022 SLCO1B1 → PM/IM/EM mapping
- 2026-05-01 Hardening: `realize_means()` deterministic path
- v3 empirical defect verification: caffeine/CYP1A2:PM/EM = 1.0000× (broken) vs pravastatin/SLCO1B1:PM/EM = 3.034× (works)

## 13. v3 self-review checklist

- ✅ Placeholder scan: SMILES `<Task 1: ...>` are documented intentional placeholders with explicit PubChem CID; no "TBD/TODO" leftovers
- ✅ Internal consistency: data + code + tests + acceptance align; back-solve fix referenced consistently across §3, §6.4, §7.3, §11
- ✅ Scope check: v0.3.2 = NAT2/UGT1A1 + back-solve fix as a coherent bundle; 12 tasks
- ✅ Ambiguity: fm priority (registry > existing dead-code DrugBank UGT path > default), back-solve fix mechanism, irinotecan deferral all unambiguous
- ✅ Empirical anchor: caffeine 1.0000× and pravastatin 3.034× ratios verified 2026-05-04 in current main
- ✅ Risk surface: §8.6 enumerates impact of back-solve fix on existing tests/scripts/users; all safe
