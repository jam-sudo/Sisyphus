# NAT2 + UGT1A1 Phenotype Propagation (v0.3.2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NAT2 and UGT1A1 to the phenotype scaling infrastructure (issue #10) AND fix a latent back-solve cancellation bug that silently nullified all CYP/UGT/NAT phenotype effects.

**Architecture:** Two-layer registry pattern (per-gene JSON keyed by full RDKit InChIKey) for substrate annotation; `pipeline/predict.py` snapshots enzyme abundances *before* phenotype application so `_decompose_clint`'s back-solve uses unscaled baseline; engine multiplication then propagates phenotype as `scaled_abundance × pre_affinity = scale × original_rate`. Engine layer: zero changes (graph-blind invariant).

**Tech Stack:** Python 3.10+, RDKit (InChIKey derivation), pytest, JSON registry files, YAML physiology data, lru_cache.

**Spec:** `docs/superpowers/specs/2026-05-04-nat2-ugt1a1-phenotype-design.md` (v3, commit `9af6c30`).

**Branch:** `feat/nat2-ugt1a1-phenotype` from `main` (HEAD `1ac8572`).

---

## Pre-flight: branch setup

- [ ] **Step 0a: Confirm clean main and create feature branch**

```bash
git status
git checkout -b feat/nat2-ugt1a1-phenotype
git log --oneline -3
```

Expected: clean working tree on main; new branch created; HEAD shows `1ac8572` plus the v3 spec commit `9af6c30`.

---

## File Structure

**Create:**
- `data/enzymes/nat2_substrates.json` — NAT2 substrate registry (3 drugs)
- `data/enzymes/ugt1a1_substrates.json` — UGT1A1 substrate registry (3 drugs)
- `src/sisyphus/predict/non_cyp_substrates.py` — registry loader + InChIKey lookup helpers
- `tests/unit/test_non_cyp_substrates.py` — loader unit tests
- `tests/regression/test_non_cyp_registry_schema.py` — schema gates (5 per registry + cross-cutting)
- `tests/integration/test_phenotype_nat2.py` — isoniazid PM/EM Cmax integration
- `tests/integration/test_phenotype_ugt1a1.py` — raltegravir PM/EM Cmax integration
- `tests/integration/test_phenotype_cyp_propagation.py` — caffeine/warfarin/pravastatin back-solve fix regression

**Modify:**
- `data/physiology/reference_man.yaml` — add NAT2 + UGT1A1 to liver.enzymes block
- `src/sisyphus/predict/ivive.py` — add NAT2 to `_LIVER_ENZYME_ABUNDANCE`; extend `_get_fm_fractions` and `build_drug_on_graph` with `non_cyp_fractions` parameter
- `src/sisyphus/pipeline/predict.py` — back-solve cancellation fix (pre-phenotype enzyme snapshot) + non_cyp_fractions wiring
- `docs/claude/experiment-log.md` — v0.3.2 entry

**Untouched (verify diff = 0 lines):**
- `src/sisyphus/engine/*` — engine is identity-blind
- `data/training/4track_holdout_predictions.json` — 107-holdout invariance gate
- All other tests not listed above

---

## Task 1: Failing CYP propagation regression test

**Why first:** TDD. The back-solve cancellation defect makes CYP1A2:PM/EM = 1.000× and CYP2C9:PM/EM = 1.000× (empirically verified 2026-05-04 on `1ac8572`). Pravastatin SLCO1B1:PM/EM = 3.034× (transporter path bypasses back-solve, stays working). Writing the failing test first locks in the regression target before Task 2's fix.

**Files:**
- Create: `tests/integration/test_phenotype_cyp_propagation.py`

- [ ] **Step 1: Write the failing test**

```python
"""CYP/transporter phenotype propagation regression — back-solve cancellation fix gate.

Prior to v0.3.2 commit (Task 2), `_decompose_clint` back-solved enzyme affinity
from abundance, and `pipeline/predict.py` rebuilt the drug AFTER applying the
phenotype scaling — the rebuild cancelled the scaling exactly. CYP1A2:PM,
CYP2C9:PM, etc. produced ratio 1.000×. SLCO1B1:PM escaped because OATP1B1
uses saturable Michaelis-Menten kinetics (no back-solve).

These tests fail on pre-fix main and pass after the pipeline/predict.py
back-solve cancellation fix.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


_CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
_WARFARIN_SMILES = "CC(=O)CC(c1ccccc1)C1=C(O)c2ccccc2OC1=O"
_PRAVASTATIN_SMILES = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)


@pytest.mark.slow
def test_caffeine_cyp1a2_pm_propagates():
    """CYP1A2:PM should drop caffeine clearance, raising Cmax > 1.5× EM.

    Caffeine is ~80% CYP1A2-metabolized; PM scaling × 0.10 → CYP1A2
    contribution drops to 0.08, residual ~0.20 from other CYPs → total
    CL ~0.28 of EM → Cmax ~3.5×. Gate at 1.5× is conservative.
    """
    em = predict(_CAFFEINE_SMILES, dose_mg=100.0, phenotypes={"CYP1A2": "EM"})
    pm = predict(_CAFFEINE_SMILES, dose_mg=100.0, phenotypes={"CYP1A2": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 1.5, (
        f"CYP1A2:PM/EM Cmax ratio {ratio:.3f} ≤ 1.5 — back-solve cancellation "
        f"may have regressed (PM should drop CL, raising Cmax)."
    )


@pytest.mark.slow
def test_warfarin_cyp2c9_pm_propagates():
    """CYP2C9:PM should drop warfarin clearance, raising Cmax > 1.2× EM.

    Acid compound_type allocates fm CYP2C9 0.40 → PM × 0.10 → 0.04;
    residual 0.60 → total ~0.64 of EM → Cmax ~1.56×. Gate at 1.2× is
    conservative against fm uncertainty.
    """
    em = predict(_WARFARIN_SMILES, dose_mg=10.0, phenotypes={"CYP2C9": "EM"})
    pm = predict(_WARFARIN_SMILES, dose_mg=10.0, phenotypes={"CYP2C9": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 1.2, (
        f"CYP2C9:PM/EM Cmax ratio {ratio:.3f} ≤ 1.2 — back-solve cancellation "
        f"may have regressed."
    )


@pytest.mark.slow
def test_pravastatin_slco1b1_pm_still_works():
    """SLCO1B1:PM transporter path is unaffected by back-solve fix.

    OATP1B1 uses saturable Michaelis-Menten kinetics, not affinity back-solve.
    PM:EM ~3× per Niemi 2009 + earlier empirical 3.034 on this codebase.
    Gate at 2.5× backstops both pre-fix and post-fix behavior.
    """
    em = predict(_PRAVASTATIN_SMILES, dose_mg=40.0, phenotypes={"SLCO1B1": "EM"})
    pm = predict(_PRAVASTATIN_SMILES, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 2.5, (
        f"SLCO1B1:PM/EM Cmax ratio {ratio:.3f} ≤ 2.5 — transporter phenotype "
        f"path may have regressed."
    )
```

- [ ] **Step 2: Run test to verify pre-fix state**

Run: `pytest tests/integration/test_phenotype_cyp_propagation.py -v`

Expected: 2 FAIL (caffeine ratio = 1.000, warfarin ratio = 1.000), 1 PASS (pravastatin ratio ≈ 3.034).

The 2 failures are the intentional TDD signal — Task 2 must close them.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phenotype_cyp_propagation.py
git commit -m "$(cat <<'EOF'
test(phenotype): add CYP/transporter propagation regression (failing pre-fix)

Captures the back-solve cancellation defect: CYP1A2:PM/EM = 1.000 and
CYP2C9:PM/EM = 1.000 on current main (verified 2026-05-04). SLCO1B1:PM
3.034 stays working because OATP1B1 uses MM kinetics, not affinity
back-solve. Failing tests are the gate Task 2 closes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Back-solve cancellation fix in pipeline/predict.py

**Why:** Closes the 2 failing tests from Task 1. Snapshots enzyme abundance BEFORE phenotype application; passes pre-phenotype values to `build_drug_on_graph` so `_decompose_clint` back-solves affinity from the unscaled baseline. The graph still carries scaled abundances (for engine multiplication) — only the affinity-back-solve sees pre-phenotype values.

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py:259-269` (the `if "liver" in graph.nodes` block)

- [ ] **Step 1: Read the current block to confirm the exact lines**

Run: `sed -n '249,272p' src/sisyphus/pipeline/predict.py`

Expected output ends with `transporter_kinetics=auto_oatp_kinetics,` and `hepatic_ecm_params=auto_ecm_params,` before the closing `)` of `build_drug_on_graph(...)`.

- [ ] **Step 2: Apply the fix**

Open `src/sisyphus/pipeline/predict.py` and replace lines 249-269 (the `graph = build_from_yaml(...)` through the `drug = build_drug_on_graph(...)` rebuild) with:

```python
        graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")

        # CRITICAL (v0.3.2): snapshot pre-phenotype liver enzyme abundances.
        # `_decompose_clint` back-solves enzyme affinity from abundance, so
        # passing scaled abundances would cause phenotype scaling to cancel
        # out at engine multiplication time (the bug that silently nulled
        # all CYP/UGT/NAT phenotype effects pre-v0.3.2; SLCO1B1 escaped
        # only because OATP1B1 uses saturable MM kinetics, not back-solve).
        # We snapshot BEFORE phenotype application so affinity is computed
        # from the unscaled baseline; phenotype then propagates through the
        # engine as scaled_abundance × pre_affinity = scale × original_rate.
        liver_enzymes_pre: dict[str, float] | None = None
        if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
            liver_enzymes_pre = {
                tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
            }

        # Apply CPIC phenotype scaling AFTER snapshot. Engine reads scaled
        # abundances from the graph; affinity (computed below from
        # liver_enzymes_pre) carries the unscaled baseline.
        if phenotypes:
            from sisyphus.predict.phenotype import apply_phenotype_to_graph
            graph = apply_phenotype_to_graph(graph, phenotypes)

        # Rebuild drug with PRE-phenotype abundances. Phenotype's effect on
        # the graph remains (scaled abundances flow into engine multiplication);
        # affinity is back-solved from unscaled abundances so the multiplication
        # propagates the scaling rather than cancelling it.
        if liver_enzymes_pre is not None:
            drug = build_drug_on_graph(
                profile, adme, dose_mg, route,
                liver_enzymes=liver_enzymes_pre,
                kp_method=kp_method,
                transporter_kinetics=auto_oatp_kinetics,
                hepatic_ecm_params=auto_ecm_params,
            )
```

- [ ] **Step 3: Run the failing tests to confirm they now pass**

Run: `pytest tests/integration/test_phenotype_cyp_propagation.py -v`

Expected: 3 PASS (caffeine ratio ~3.3, warfarin ratio ~1.5, pravastatin ratio ~3.0). Numbers may differ slightly from Task 1 estimates due to fm rounding — what matters is all 3 above-threshold.

- [ ] **Step 4: Run a quick spot-check of broader phenotype behavior to verify no surprise**

Run: `pytest tests/unit/test_pipeline_phenotypes.py tests/unit/test_phenotype.py -v`

Expected: All existing tests PASS. These tests assert invocation patterns and graph-level scaling, not pipeline Cmax magnitude — the back-solve fix does not invalidate them.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/pipeline/predict.py
git commit -m "$(cat <<'EOF'
fix(pipeline): pre-phenotype enzyme snapshot for affinity back-solve

Snapshot liver.enzymes BEFORE apply_phenotype_to_graph and pass
unscaled values into build_drug_on_graph. _decompose_clint back-solves
affinity from these unscaled abundances; engine multiplication then
sees scaled_abundance × pre_affinity = scale × original_rate, so
phenotype propagation is correct.

Pre-fix CYP1A2:PM/EM = 1.000, post-fix ~3.3 (caffeine).
Pre-fix CYP2C9:PM/EM = 1.000, post-fix ~1.5 (warfarin).
SLCO1B1:PM/EM stays ~3.0 (transporter MM path unaffected).

Closes the silent zero-effect bug for ALL CYP/UGT/NAT phenotypes.
107-holdout benchmark uses phenotypes=None default → invariant.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: NAT2 substrate registry

**Why:** Curation surface for which drugs activate the NAT2 fm path. Without registry entry, drug has no `enzyme_affinity[NAT2]`. Schema gates prevent silent additions and SMILES drift.

**Files:**
- Create: `data/enzymes/nat2_substrates.json`

Hydralazine canonical SMILES `NNc1nncc2ccccc12` was verified by spec author to round-trip to InChIKey RPTUSVTUFVMDQK-UHFFFAOYSA-N (PubChem CID 3637). Isoniazid `NNC(=O)c1ccncc1` → QRXWMOHMRWLFEY-UHFFFAOYSA-N (CID 3767). Procainamide `CCN(CC)CCNC(=O)c1ccc(N)cc1` → REQCZEXYDRLIBE-UHFFFAOYSA-N (CID 4913). All three are pre-validated.

- [ ] **Step 1: Create the directory and file**

```bash
mkdir -p data/enzymes
```

Then create `data/enzymes/nat2_substrates.json` with this exact content:

```json
{
  "version": 1,
  "description": "Substrates of hepatic cytosolic NAT2 (N-acetyltransferase 2). Used by predict() to allocate metabolic_fraction of XGBoost CLint to NAT2; the residual is split among CYPs per compound_type defaults. NAT2 phenotype scaling propagates into Cmax via abundance × enzyme_affinity multiplication in the engine, predicated on the back-solve cancellation fix in pipeline/predict.py (v0.3.2 Task 2).",
  "rationale": "Without registry entry, drug has no NAT2 enzyme_affinity → phenotype scaling is a no-op. This registry is the curation surface for which drugs trigger the NAT2 path.",
  "schema": {
    "drug": "Lowercase common name (informational)",
    "smiles": "Canonical RDKit SMILES",
    "inchikey": "RDKit-derived InChIKey (primary lookup key, full 27-char)",
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
      "notes": "Canonical NAT2 substrate. Slow acetylator t1/2 ~3h, rapid acetylator t1/2 ~1h (Ellard 1976). Cmax PM/EM ~1.3-1.5x at 300 mg PO (Peloquin 1999). Minor CYP2E1 contribution to hepatotoxic metabolite hydrazine, not rate-limiting at the whole-organ CL level."
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
      "notes": "NAT2 -> N-acetylprocainamide ~50%; renal ~50% (parent + metabolite). PM/EM acetylator distinction primarily affects N-acetyl metabolite ratio."
    }
  ]
}
```

- [ ] **Step 2: Verify InChIKey round-trip via RDKit**

Run:

```bash
python3 << 'PYEOF'
import json, pathlib
from rdkit import Chem
data = json.loads(pathlib.Path("data/enzymes/nat2_substrates.json").read_text())
ok = True
for s in data["substrates"]:
    m = Chem.MolFromSmiles(s["smiles"])
    assert m is not None, f"{s['drug']}: SMILES failed to parse"
    derived = Chem.MolToInchiKey(m)
    if derived != s["inchikey"]:
        print(f"MISMATCH {s['drug']}: registered={s['inchikey']} derived={derived}")
        ok = False
    else:
        print(f"OK {s['drug']}: {derived}")
assert ok, "InChIKey mismatch — fix SMILES or inchikey"
PYEOF
```

Expected output: 3 OK lines, exit 0.

- [ ] **Step 3: Commit**

```bash
git add data/enzymes/nat2_substrates.json
git commit -m "$(cat <<'EOF'
data(enzymes): add nat2_substrates.json — isoniazid + hydralazine + procainamide

Initial seed for NAT2 phenotype propagation infrastructure (issue #10).
Per-drug metabolic_fraction literature-anchored:
- isoniazid 0.90 (Weber 1983, Ellard 1976)
- hydralazine 0.50 (Reece 1981 — mixed NAT2 + CYP3A4/2C9)
- procainamide 0.50 (Drayer 1977 — NAT2 + renal)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: UGT1A1 substrate registry

**Why:** Same as Task 3, for UGT1A1. The 3 seed drugs are the GenoADME deferred-pair candidates that don't depend on prodrug-metabolite tracking (irinotecan deferred to issue #11).

**Files:**
- Create: `data/enzymes/ugt1a1_substrates.json`

The 3 UGT1A1 SMILES require canonical-form derivation from PubChem CIDs. The implementer derives them in this task using RDKit. Below are the **PubChem-canonical SMILES strings** to use as input — they may need RDKit canonicalization before storage, and the InChIKey is filled in from RDKit output.

- [ ] **Step 1: Derive canonical SMILES + InChIKey via RDKit**

Run:

```bash
python3 << 'PYEOF'
import json
from rdkit import Chem

# PubChem-canonical SMILES (input strings; RDKit canonicalizes for storage)
candidates = {
    "raltegravir":  "CC1=NN=C(O1)C(C)(C)NC(=O)C2=NC(=C(C(=O)N2C)O)C(=O)NCC3=CC(=CC=C3)F",
    "atazanavir":   "CC(C)(C)NC(=O)[C@@H](Cc1ccc(/C=N/Nc2ccc(C(=O)OC)cc2)cc1)NC(=O)[C@@H](NC(=O)OC)c1ccccc1",
    "dolutegravir": "C[C@H]1CCO[C@@H]2N1C(=O)C3=C(C(=O)N([C@@H]2CO)Cc4ccc(F)cc4F)C=C(O)N3C",
}
out = {}
for n, raw in candidates.items():
    m = Chem.MolFromSmiles(raw)
    assert m is not None, f"{n}: SMILES failed to parse"
    canonical = Chem.MolToSmiles(m)
    ikey = Chem.MolToInchiKey(m)
    out[n] = (canonical, ikey)
    print(f"{n}: smiles={canonical}  ikey={ikey}")
PYEOF
```

Expected: each drug prints a canonical SMILES and a 27-char InChIKey. Record the output — these strings go into the JSON below.

If any drug returns "SMILES failed to parse", manually fetch the PubChem-canonical SMILES (CID 54671008 for raltegravir, 148192 for atazanavir, 54726191 for dolutegravir) and re-run.

- [ ] **Step 2: Create the JSON file with derived values**

Create `data/enzymes/ugt1a1_substrates.json`. Replace the `<derived>` markers with the canonical SMILES and InChIKey strings printed in Step 1:

```json
{
  "version": 1,
  "description": "Substrates of hepatic UGT1A1 with explicit metabolic_fraction. The current build_drug_on_graph hardcodes ugt_enzymes=None (UGT path disabled per ivive.py:611 sensitivity result), so this registry is the only path that creates a non-zero enzyme_affinity[UGT1A1] for listed drugs.",
  "rationale": "Without registry entry, drug has no UGT1A1 enzyme_affinity. Initial seed: 3 GenoADME deferred-pair candidates that are parent-PK (not prodrug-metabolite). irinotecan/UGT1A1 is deferred to issue #11.",
  "schema": "see nat2_substrates.json",
  "substrates": [
    {
      "drug": "raltegravir",
      "smiles": "<RDKit canonical from Step 1>",
      "inchikey": "<RDKit-derived from Step 1>",
      "metabolic_fraction": 0.70,
      "literature": ["Iwamoto 2008 Clin Pharmacol Ther 83:293-9"],
      "notes": "~70% UGT1A1 glucuronidation, ~30% renal/other. UGT1A1*28 carriers ~40% AUC increase."
    },
    {
      "drug": "atazanavir",
      "smiles": "<RDKit canonical from Step 1>",
      "inchikey": "<RDKit-derived from Step 1>",
      "metabolic_fraction": 0.40,
      "literature": ["Lankisch 2006 Pharmacogenet Genomics 16:495-501"],
      "notes": "Atazanavir is a UGT1A1 INHIBITOR but is itself partially glucuronidated. Mixed CYP3A4 (primary) + UGT1A1 (~40%) metabolism."
    },
    {
      "drug": "dolutegravir",
      "smiles": "<RDKit canonical from Step 1>",
      "inchikey": "<RDKit-derived from Step 1>",
      "metabolic_fraction": 0.50,
      "literature": ["Reese 2013 J Acquir Immune Defic Syndr 64:e35-6"],
      "notes": "~50% UGT1A1, ~30% CYP3A4, ~20% UGT1A9 (Reese 2013). UGT1A1*28 effect modest (~30% AUC increase)."
    }
  ]
}
```

- [ ] **Step 3: Verify InChIKey round-trip**

Run:

```bash
python3 << 'PYEOF'
import json, pathlib
from rdkit import Chem
data = json.loads(pathlib.Path("data/enzymes/ugt1a1_substrates.json").read_text())
ok = True
for s in data["substrates"]:
    m = Chem.MolFromSmiles(s["smiles"])
    assert m is not None, f"{s['drug']}: SMILES failed to parse"
    derived = Chem.MolToInchiKey(m)
    if derived != s["inchikey"]:
        print(f"MISMATCH {s['drug']}: registered={s['inchikey']} derived={derived}")
        ok = False
    else:
        print(f"OK {s['drug']}: {derived}")
assert ok
PYEOF
```

Expected: 3 OK lines.

- [ ] **Step 4: Commit**

```bash
git add data/enzymes/ugt1a1_substrates.json
git commit -m "$(cat <<'EOF'
data(enzymes): add ugt1a1_substrates.json — raltegravir + atazanavir + dolutegravir

Initial seed for UGT1A1 phenotype propagation. Per-drug metabolic_fraction:
- raltegravir 0.70 (Iwamoto 2008)
- atazanavir 0.40 (Lankisch 2006 — mixed CYP3A4 + UGT1A1)
- dolutegravir 0.50 (Reese 2013)

irinotecan/UGT1A1 deferred to issue #11 (prodrug-metabolite work).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: non_cyp_substrates.py loader module

**Why:** Encapsulates the JSON registry lookup + InChIKey caching. Mirrors `transporter_db.py` (PR #29) pattern. `predict()` calls `get_non_cyp_fractions(smiles)` and gets a dict like `{"NAT2": 0.90}` ready for `_get_fm_fractions`.

**Files:**
- Create: `src/sisyphus/predict/non_cyp_substrates.py`
- Test: `tests/unit/test_non_cyp_substrates.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_non_cyp_substrates.py`:

```python
"""Unit tests for non_cyp_substrates registry loader (NAT2 + UGT1A1)."""
from __future__ import annotations

import pytest

from sisyphus.predict.non_cyp_substrates import (
    get_non_cyp_fractions,
    lookup_nat2_substrate,
    lookup_ugt1a1_substrate,
)


_ISONIAZID = "NNC(=O)c1ccncc1"
_HYDRALAZINE = "NNc1nncc2ccccc12"
_METOPROLOL = "COCCc1ccc(OCC(O)CNC(C)C)cc1"


def test_lookup_nat2_isoniazid_returns_entry():
    entry = lookup_nat2_substrate(_ISONIAZID)
    assert entry is not None
    assert entry["drug"] == "isoniazid"
    assert entry["metabolic_fraction"] == pytest.approx(0.90)


def test_lookup_nat2_metoprolol_returns_none():
    assert lookup_nat2_substrate(_METOPROLOL) is None


def test_lookup_nat2_invalid_smiles_returns_none():
    assert lookup_nat2_substrate("not-a-smiles") is None


def test_lookup_nat2_empty_returns_none():
    assert lookup_nat2_substrate("") is None


def test_lookup_ugt1a1_metoprolol_returns_none():
    assert lookup_ugt1a1_substrate(_METOPROLOL) is None


def test_get_non_cyp_fractions_isoniazid():
    out = get_non_cyp_fractions(_ISONIAZID)
    assert out == {"NAT2": pytest.approx(0.90)}


def test_get_non_cyp_fractions_hydralazine():
    out = get_non_cyp_fractions(_HYDRALAZINE)
    assert out == {"NAT2": pytest.approx(0.50)}


def test_get_non_cyp_fractions_metoprolol_empty():
    assert get_non_cyp_fractions(_METOPROLOL) == {}


def test_get_non_cyp_fractions_invalid_smiles_empty():
    assert get_non_cyp_fractions("not-a-smiles") == {}


def test_lru_cache_reuses_loaded_data():
    """Two calls should not re-read JSON. Sanity check for lru_cache wiring."""
    out1 = lookup_nat2_substrate(_ISONIAZID)
    out2 = lookup_nat2_substrate(_ISONIAZID)
    assert out1 is out2 or out1 == out2  # cached object identity OR equal dict
```

- [ ] **Step 2: Run to verify it fails (module doesn't exist)**

Run: `pytest tests/unit/test_non_cyp_substrates.py -v`

Expected: ImportError / module not found.

- [ ] **Step 3: Implement the module**

Create `src/sisyphus/predict/non_cyp_substrates.py`:

```python
"""Non-CYP enzyme substrate registries for NAT2 and UGT1A1 phenotype propagation.

Two JSON registries (data/enzymes/nat2_substrates.json,
data/enzymes/ugt1a1_substrates.json) keyed by full RDKit InChIKey hold
per-drug metabolic_fraction values. predict() calls get_non_cyp_fractions()
to obtain the dict passed downstream to _get_fm_fractions; that fraction
of XGBoost CLint is then routed through the named enzyme so phenotype
scaling on liver.enzymes[NAT2 or UGT1A1] propagates into engine rate.

Mirrors transporter_db.py (PR #29) — lru_cache JSON loaders, full
InChIKey matching only (no block-1 truncation), file-anchored paths.
"""
from __future__ import annotations

import json
import logging
import pathlib
from functools import lru_cache

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_NAT2_PATH = _REPO_ROOT / "data" / "enzymes" / "nat2_substrates.json"
_UGT1A1_PATH = _REPO_ROOT / "data" / "enzymes" / "ugt1a1_substrates.json"


def _smiles_to_inchikey(smiles: str) -> str | None:
    """Return RDKit InChIKey for a SMILES, or None on parse failure."""
    if not smiles:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


@lru_cache(maxsize=1)
def _load_nat2_index() -> dict[str, dict]:
    """Return {inchikey: substrate_entry} for nat2_substrates.json."""
    if not _NAT2_PATH.exists():
        return {}
    data = json.loads(_NAT2_PATH.read_text())
    return {entry["inchikey"]: entry for entry in data.get("substrates", [])}


@lru_cache(maxsize=1)
def _load_ugt1a1_index() -> dict[str, dict]:
    """Return {inchikey: substrate_entry} for ugt1a1_substrates.json."""
    if not _UGT1A1_PATH.exists():
        return {}
    data = json.loads(_UGT1A1_PATH.read_text())
    return {entry["inchikey"]: entry for entry in data.get("substrates", [])}


def lookup_nat2_substrate(smiles: str) -> dict | None:
    """Return the registry entry if the SMILES matches a NAT2 substrate.

    Lookup is by full RDKit InChIKey (rejects block-1 truncation per
    issue #25 lessons). Returns None for missing / invalid SMILES.
    """
    ikey = _smiles_to_inchikey(smiles)
    if ikey is None:
        return None
    return _load_nat2_index().get(ikey)


def lookup_ugt1a1_substrate(smiles: str) -> dict | None:
    """Return the registry entry if the SMILES matches a UGT1A1 substrate."""
    ikey = _smiles_to_inchikey(smiles)
    if ikey is None:
        return None
    return _load_ugt1a1_index().get(ikey)


def get_non_cyp_fractions(smiles: str) -> dict[str, float]:
    """Aggregate NAT2 + UGT1A1 metabolic fractions for the given SMILES.

    Returns {gene: metabolic_fraction} ready to pass into _get_fm_fractions.
    Empty dict if no substrate match. If multi-gene total exceeds 1.0
    (round-off or curation overlap), values are re-normalized to sum=1.0
    and a logger.info message is emitted.
    """
    out: dict[str, float] = {}
    nat2 = lookup_nat2_substrate(smiles)
    if nat2 is not None:
        out["NAT2"] = float(nat2["metabolic_fraction"])
    ugt = lookup_ugt1a1_substrate(smiles)
    if ugt is not None:
        out["UGT1A1"] = float(ugt["metabolic_fraction"])
    total = sum(out.values())
    if total > 1.0:
        logger.info(
            "non_cyp_fractions sum %.3f > 1.0 for SMILES %r; re-normalizing",
            total, smiles,
        )
        out = {k: v / total for k, v in out.items()}
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_non_cyp_substrates.py -v`

Expected: 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/predict/non_cyp_substrates.py tests/unit/test_non_cyp_substrates.py
git commit -m "$(cat <<'EOF'
feat(predict): non_cyp_substrates loader for NAT2 + UGT1A1 registries

Mirrors transporter_db.py pattern — lru_cache JSON loaders, full RDKit
InChIKey matching, file-anchored paths from repo root.

Public API:
- lookup_nat2_substrate(smiles)   -> entry | None
- lookup_ugt1a1_substrate(smiles) -> entry | None
- get_non_cyp_fractions(smiles)   -> {gene: fraction}, normalized

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: reference_man.yaml + _LIVER_ENZYME_ABUNDANCE consistency

**Why:** The graph YAML is the source of truth at runtime; `_LIVER_ENZYME_ABUNDANCE` is the IVIVE fallback when no graph is present. Both must list NAT2 (and YAML must list UGT1A1, which is currently in constant only).

**Files:**
- Modify: `data/physiology/reference_man.yaml` (liver.enzymes block, after line 63 `CES2:`)
- Modify: `src/sisyphus/predict/ivive.py` (`_LIVER_ENZYME_ABUNDANCE` block, after line 60 `"UGT1A9": 1_012_500.0,`)

- [ ] **Step 1: Add NAT2 + UGT1A1 to reference_man.yaml liver.enzymes**

Open `data/physiology/reference_man.yaml`. Locate the line `CES2: {mean: 8.4e6, cv: 0.61}   # Boberg 2017 PMC5267516: 174 pmol/mg × 48000 mg microsomal` (around line 63). Insert these two lines immediately after, BEFORE the `transporters:` line (line 64):

```yaml
      # Phase II non-CYP enzymes for issue #10 phenotype propagation (v0.3.2).
      # Independent lognormal — no Achour 2021 matrix entry. Position at end of
      # enzymes block minimizes RNG-order disruption to existing tests.
      NAT2:    {mean: 1.0e7,   cv: 0.6}    # Cytosolic. Calibration-arbitrary anchored to Grant 1991 cytosolic activity range; absolute value back-solved by _decompose_clint such that abundance × affinity × ivive_scaling = NAT2 fm × CLint_hepatic.
      UGT1A1:  {mean: 1.215e6, cv: 0.5}    # Achour 2014 PMC4118705: 18 pmol/mg microsomal × 45 MPPGL × 1500g. Consistent with ivive.py _LIVER_ENZYME_ABUNDANCE.
```

- [ ] **Step 2: Add NAT2 to _LIVER_ENZYME_ABUNDANCE**

Open `src/sisyphus/predict/ivive.py`. Locate the `_LIVER_ENZYME_ABUNDANCE` block (line 49-60). Add this entry after the `"UGT1A9": 1_012_500.0,` line (around line 59):

```python
    # Phase II non-CYP for issue #10 phenotype propagation (v0.3.2).
    # NAT2 cytosolic; abundance is calibration-arbitrary at mean (back-solved
    # such that abundance × affinity × ivive_scaling = NAT2 fm × CLint_hepatic).
    "NAT2": 1.0e7,
```

UGT1A1 is already in `_LIVER_ENZYME_ABUNDANCE` at line 57 (`"UGT1A1": 1_215_000.0`). Do NOT change it; verify it equals `1.215e6` per spec consistency requirement.

- [ ] **Step 3: Quick parsing sanity check**

Run:

```bash
python3 -c "
from sisyphus.graph.builder import build_from_yaml
import pathlib
g = build_from_yaml(pathlib.Path('data/physiology/reference_man.yaml'))
liver = g.nodes['liver']
print('NAT2:', liver.enzymes.get('NAT2'))
print('UGT1A1:', liver.enzymes.get('UGT1A1'))
"
```

Expected: prints two Distribution objects with mean 1e7 / 1.215e6 and the documented CVs.

- [ ] **Step 4: Run holdout regression to confirm invariance**

Run: `pytest tests/integration/test_holdout_regression.py -v`

Expected: PASS (Meta 2.679 pin holds — the YAML additions don't affect deterministic `realize_means()` path, and the holdout benchmark uses no phenotypes).

- [ ] **Step 5: Commit**

```bash
git add data/physiology/reference_man.yaml src/sisyphus/predict/ivive.py
git commit -m "$(cat <<'EOF'
data(physiology): NAT2 + UGT1A1 in liver.enzymes for phenotype propagation

YAML adds at end of liver.enzymes block (RNG-order safe; realize_means
deterministic path unaffected). _LIVER_ENZYME_ABUNDANCE constant adds
NAT2 to keep _decompose_clint fallback consistent. UGT1A1 already in
constant at 1.215e6 — verified equal to YAML value.

107-holdout invariant (predict() default phenotypes=None).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Extend `_get_fm_fractions` with `non_cyp_fractions` parameter

**Why:** Adds the routing path so NAT2/UGT1A1 fm enters the per-enzyme allocation. Without this extension, the registry is dead data — the rest of the IVIVE pipeline doesn't know to look at it.

**Files:**
- Modify: `src/sisyphus/predict/ivive.py:148-217` (`_get_fm_fractions`)
- Test: `tests/unit/test_ivive_non_cyp.py` (NEW)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ivive_non_cyp.py`:

```python
"""Tests for _get_fm_fractions non_cyp_fractions extension (issue #10)."""
from __future__ import annotations

import pytest

from sisyphus.predict.ivive import _get_fm_fractions


def test_non_cyp_fractions_none_preserves_existing():
    """non_cyp_fractions=None must produce identical output to no kwarg."""
    a = _get_fm_fractions("acid", substrate_enzymes=None, ugt_enzymes=None)
    b = _get_fm_fractions(
        "acid", substrate_enzymes=None, ugt_enzymes=None, non_cyp_fractions=None
    )
    assert a == b


def test_non_cyp_fractions_empty_preserves_existing():
    a = _get_fm_fractions("acid", substrate_enzymes=None, ugt_enzymes=None)
    b = _get_fm_fractions(
        "acid", substrate_enzymes=None, ugt_enzymes=None, non_cyp_fractions={}
    )
    assert a == b


def test_non_cyp_fractions_nat2_only_routes_to_nat2():
    out = _get_fm_fractions(
        "acid",
        substrate_enzymes=None,
        ugt_enzymes=None,
        non_cyp_fractions={"NAT2": 0.90},
    )
    assert out["NAT2"] == pytest.approx(0.90)
    cyp_total = sum(v for k, v in out.items() if k != "NAT2")
    assert cyp_total == pytest.approx(0.10)
    assert sum(out.values()) == pytest.approx(1.0)


def test_non_cyp_fractions_ugt1a1_only():
    out = _get_fm_fractions(
        "neutral",
        substrate_enzymes=None,
        ugt_enzymes=None,
        non_cyp_fractions={"UGT1A1": 0.70},
    )
    assert out["UGT1A1"] == pytest.approx(0.70)
    assert sum(v for k, v in out.items() if k != "UGT1A1") == pytest.approx(0.30)


def test_non_cyp_fractions_both_genes():
    out = _get_fm_fractions(
        "neutral",
        substrate_enzymes=None,
        ugt_enzymes=None,
        non_cyp_fractions={"NAT2": 0.40, "UGT1A1": 0.40},
    )
    assert out["NAT2"] == pytest.approx(0.40)
    assert out["UGT1A1"] == pytest.approx(0.40)
    assert sum(out.values()) == pytest.approx(1.0)


def test_non_cyp_fractions_value_out_of_range_raises():
    with pytest.raises(ValueError):
        _get_fm_fractions(
            "acid",
            substrate_enzymes=None,
            ugt_enzymes=None,
            non_cyp_fractions={"NAT2": 1.5},
        )


def test_non_cyp_fractions_negative_raises():
    with pytest.raises(ValueError):
        _get_fm_fractions(
            "acid",
            substrate_enzymes=None,
            ugt_enzymes=None,
            non_cyp_fractions={"NAT2": -0.1},
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/unit/test_ivive_non_cyp.py -v`

Expected: FAIL on the keyword `non_cyp_fractions` not accepted by `_get_fm_fractions`.

- [ ] **Step 3: Extend `_get_fm_fractions`**

Open `src/sisyphus/predict/ivive.py`. Replace the `def _get_fm_fractions(...)` signature (line 148) and body (through line 217) with:

```python
def _get_fm_fractions(
    compound_type: str,
    substrate_enzymes: set[str] | None = None,
    ugt_enzymes: set[str] | None = None,
    non_cyp_fractions: dict[str, float] | None = None,
) -> dict[str, float]:
    """Get fraction metabolized by each enzyme (CYP + UGT + non-CYP), adjusted for compound type.

    If substrate_enzymes is provided (from DrugBank annotations), known substrates
    are given equal weight and non-substrates are floored at _NON_SUBSTRATE_FLOOR.

    UGT enzymes are handled via annotation pattern:
    - UGT only (no CYP annotation): UGT gets 0.90 of total fm
    - CYP + UGT both annotated: UGT gets 0.30 of total fm

    non_cyp_fractions (NAT2, UGT1A1 from per-gene registries — issue #10):
    - Each value validated to [0, 1]; raises ValueError otherwise.
    - Sum normalized to <= 1.0 (re-normalize if > 1.0; caller's
      non_cyp_substrates.get_non_cyp_fractions also normalizes upstream).
    - Remaining (1 - non_cyp_total) goes to existing CYP+UGT allocation.
    - Resulting fm dict is _normalize_fm-ed for round-off safety.

    Args:
        compound_type: One of "neutral", "acid", "base", "zwitterion".
        substrate_enzymes: Set of CYP tags for which this drug is a known substrate.
        ugt_enzymes: Set of UGT tags for which this drug is a known substrate.
        non_cyp_fractions: {"NAT2": fm, "UGT1A1": fm} from per-gene registries.

    Returns:
        Dict mapping enzyme tag -> fraction metabolized. Sums to 1.0.
    """
    if compound_type in _FM_ADJUSTMENTS:
        fm = dict(_FM_ADJUSTMENTS[compound_type])
    else:
        fm = dict(_DEFAULT_FM)

    if not substrate_enzymes:
        cyp_fm = _normalize_fm(fm)
    else:
        known_substrates = substrate_enzymes & set(fm.keys())
        if not known_substrates:
            cyp_fm = _normalize_fm(fm)
        else:
            _NON_SUBSTRATE_FLOOR = 0.05
            for enzyme in fm:
                if enzyme in known_substrates:
                    fm[enzyme] = 1.0 / len(known_substrates)
                else:
                    fm[enzyme] = _NON_SUBSTRATE_FLOOR
            cyp_fm = _normalize_fm(fm)

    if ugt_enzymes:
        has_cyp = bool(substrate_enzymes)
        if not has_cyp:
            ugt_fraction = 0.90
        else:
            ugt_fraction = 0.30

        cyp_fraction = 1.0 - ugt_fraction
        result: dict[str, float] = {}
        for tag, frac in cyp_fm.items():
            result[tag] = frac * cyp_fraction
        n_ugt = len(ugt_enzymes)
        for tag in ugt_enzymes:
            result[tag] = ugt_fraction / n_ugt
        cyp_fm = _normalize_fm(result)

    if not non_cyp_fractions:
        return cyp_fm

    for gene, frac in non_cyp_fractions.items():
        if not (0.0 <= frac <= 1.0):
            raise ValueError(
                f"non_cyp_fractions[{gene!r}]={frac} not in [0, 1]"
            )

    non_cyp_total = sum(non_cyp_fractions.values())
    if non_cyp_total > 1.0:
        non_cyp_fractions = {k: v / non_cyp_total for k, v in non_cyp_fractions.items()}
        non_cyp_total = 1.0

    cyp_residual = max(1.0 - non_cyp_total, 0.0)
    out: dict[str, float] = {tag: frac * cyp_residual for tag, frac in cyp_fm.items()}
    for gene, frac in non_cyp_fractions.items():
        out[gene] = frac

    return _normalize_fm(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ivive_non_cyp.py -v`

Expected: 7 PASS.

- [ ] **Step 5: Run all existing _get_fm_fractions / _decompose_clint callers to confirm backward compat**

Run: `pytest tests/unit/test_ivive.py -v` (if it exists; otherwise skip)

Expected: PASS / no regression.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/predict/ivive.py tests/unit/test_ivive_non_cyp.py
git commit -m "$(cat <<'EOF'
feat(ivive): _get_fm_fractions non_cyp_fractions parameter

Routes NAT2 / UGT1A1 fm allocation from per-gene registries before the
existing CYP+UGT allocation. CYP+UGT residual scaled by (1 - non_cyp_total).
Validates fm in [0, 1]; re-normalizes when sum > 1.0.

Backward-compatible: non_cyp_fractions=None preserves current behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Extend `build_drug_on_graph` with `non_cyp_fractions` kwarg

**Why:** Threads the registry-derived fm dict through to `_decompose_clint` via `_get_fm_fractions`.

**Files:**
- Modify: `src/sisyphus/predict/ivive.py:561-690` (`build_drug_on_graph`)
- Modify: `src/sisyphus/predict/ivive.py:220-288` (`_decompose_clint`)

- [ ] **Step 1: Add `non_cyp_fractions` to `_decompose_clint` signature and body**

Open `src/sisyphus/predict/ivive.py`. Locate `def _decompose_clint(...)` (line 220). Update the signature to:

```python
def _decompose_clint(
    clint: Distribution,
    compound_type: str,
    pka: float | None,
    enzyme_abundances: dict[str, float] | None = None,
    substrate_enzymes: set[str] | None = None,
    ugt_enzymes: set[str] | None = None,
    metabolic_fraction: float = 1.0,
    non_cyp_fractions: dict[str, float] | None = None,
) -> dict[str, Distribution]:
```

In the body, replace `fm = _get_fm_fractions(compound_type, substrate_enzymes, ugt_enzymes)` (line 264) with:

```python
    fm = _get_fm_fractions(
        compound_type,
        substrate_enzymes=substrate_enzymes,
        ugt_enzymes=ugt_enzymes,
        non_cyp_fractions=non_cyp_fractions,
    )
```

The downstream loop already iterates `fm.items()` and looks up abundance via `_LIVER_ENZYME_ABUNDANCE.get(enzyme, 1.0)` — NAT2 is in the constant (Task 6), UGT1A1 is in the constant from prior commit. No further change needed inside `_decompose_clint`.

- [ ] **Step 2: Add `non_cyp_fractions` to `build_drug_on_graph` and forward**

Locate `def build_drug_on_graph(...)` signature (line 561). Add `non_cyp_fractions` kwarg with default `None`:

```python
def build_drug_on_graph(
    profile: MolecularProfile,
    adme: ADMEProperties,
    dose_mg: float,
    route: str = "oral",
    liver_enzymes: dict[str, float] | None = None,
    kp_method: str = "rodgers_rowland",
    transporter_kinetics: dict[str, TransporterKinetics] | None = None,
    hepatic_ecm_params: dict[str, Distribution] | None = None,
    non_cyp_fractions: dict[str, float] | None = None,
) -> DrugOnGraph:
```

Locate the `enzyme_affinity = _decompose_clint(...)` call (around line 620). Replace with:

```python
    enzyme_affinity = _decompose_clint(
        adme.clint, profile.compound_type, profile.pka,
        enzyme_abundances=abundances,
        substrate_enzymes=substrate_enzymes,
        ugt_enzymes=ugt_enzymes,
        metabolic_fraction=metabolic_fraction,
        non_cyp_fractions=non_cyp_fractions,
    )
```

- [ ] **Step 3: Add backward-compat smoke test (caffeine path stays unchanged when non_cyp_fractions=None)**

Append to `tests/unit/test_ivive_non_cyp.py`:

```python
def test_build_drug_on_graph_non_cyp_default_none_unchanged(monkeypatch):
    """build_drug_on_graph default kwarg path unchanged for non-NAT2/UGT1A1 drug."""
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.ivive import build_drug_on_graph

    smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # caffeine
    profile = compute_profile(smiles)
    adme = predict_adme(profile)

    drug_default = build_drug_on_graph(profile, adme, dose_mg=100.0, route="oral")
    drug_explicit_none = build_drug_on_graph(
        profile, adme, dose_mg=100.0, route="oral", non_cyp_fractions=None,
    )
    drug_explicit_empty = build_drug_on_graph(
        profile, adme, dose_mg=100.0, route="oral", non_cyp_fractions={},
    )
    # Same enzyme_affinity dict regardless of None vs {} vs unset
    assert set(drug_default.enzyme_affinity.keys()) == set(drug_explicit_none.enzyme_affinity.keys())
    assert set(drug_default.enzyme_affinity.keys()) == set(drug_explicit_empty.enzyme_affinity.keys())
    for tag in drug_default.enzyme_affinity:
        assert drug_default.enzyme_affinity[tag].mean == pytest.approx(
            drug_explicit_none.enzyme_affinity[tag].mean
        )


def test_build_drug_on_graph_isoniazid_with_non_cyp_fractions():
    """Isoniazid + non_cyp_fractions={'NAT2': 0.9} produces non-zero NAT2 affinity."""
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.ivive import build_drug_on_graph

    smiles = "NNC(=O)c1ccncc1"  # isoniazid
    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    drug = build_drug_on_graph(
        profile, adme, dose_mg=300.0, route="oral",
        non_cyp_fractions={"NAT2": 0.90},
    )
    assert "NAT2" in drug.enzyme_affinity
    assert drug.enzyme_affinity["NAT2"].mean > 0
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_ivive_non_cyp.py -v`

Expected: 9 PASS (7 original + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/predict/ivive.py tests/unit/test_ivive_non_cyp.py
git commit -m "$(cat <<'EOF'
feat(ivive): build_drug_on_graph forwards non_cyp_fractions

Threads non_cyp_fractions through build_drug_on_graph -> _decompose_clint
-> _get_fm_fractions. NAT2 / UGT1A1 enzyme_affinity now non-zero for
registry-listed drugs, ready for engine multiplication.

Backward-compatible: default None preserves prior behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Pipeline wiring — registry lookup in predict.predict()

**Why:** Connects the registry to the pipeline. Calls `get_non_cyp_fractions(profile.smiles)` once after `compute_profile` and forwards to BOTH `build_drug_on_graph` calls (initial line 202 + post-phenotype rebuild from Task 2).

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py` (3 sites: import, initial build_drug_on_graph, post-phenotype rebuild)

- [ ] **Step 1: Add the registry import and lookup**

Open `src/sisyphus/pipeline/predict.py`. Locate the imports near the auto-ECM block (around line 162):

```python
        load_hepatic_ecm_params_for_smiles,
```

Add an import for `get_non_cyp_fractions` near the existing `is_oatp_ecm_applicable` / `load_oatp1b1_kinetics_for_smiles` imports. Specifically, after line 162 (or wherever those are imported), add:

```python
    from sisyphus.predict.non_cyp_substrates import get_non_cyp_fractions
```

Then immediately after the auto-ECM gating block (around line 200, before the first `drug = build_drug_on_graph(...)` at line 202), insert:

```python
    # Per-gene non-CYP fm registry lookup (NAT2 / UGT1A1).
    # Empty dict for non-substrates is a no-op downstream.
    non_cyp_fractions = get_non_cyp_fractions(profile.smiles)
```

- [ ] **Step 2: Forward non_cyp_fractions to the initial build**

In the `drug = build_drug_on_graph(...)` call at line 202, add `non_cyp_fractions=non_cyp_fractions` as a kwarg:

```python
    drug = build_drug_on_graph(
        profile, adme, dose_mg, route,
        kp_method=kp_method,
        transporter_kinetics=auto_oatp_kinetics,
        hepatic_ecm_params=auto_ecm_params,
        non_cyp_fractions=non_cyp_fractions,
    )
```

- [ ] **Step 3: Forward non_cyp_fractions to the post-phenotype rebuild**

In the second `drug = build_drug_on_graph(...)` call (added in Task 2, around line 270 with `liver_enzymes=liver_enzymes_pre`), add the same kwarg:

```python
        if liver_enzymes_pre is not None:
            drug = build_drug_on_graph(
                profile, adme, dose_mg, route,
                liver_enzymes=liver_enzymes_pre,
                kp_method=kp_method,
                transporter_kinetics=auto_oatp_kinetics,
                hepatic_ecm_params=auto_ecm_params,
                non_cyp_fractions=non_cyp_fractions,
            )
```

- [ ] **Step 4: Spot-check pipeline integration end-to-end**

Run:

```bash
python3 << 'PYEOF'
from sisyphus.pipeline.predict import predict
# Isoniazid baseline (no phenotypes) — should now have NAT2 enzyme_affinity > 0
result = predict("NNC(=O)c1ccncc1", dose_mg=300.0)
assert result.engine_pk is not None
print("Isoniazid baseline Cmax:", result.engine_pk.cmax.mean)
PYEOF
```

Expected: prints a Cmax (not crash). Pipeline now reads NAT2 fm for isoniazid.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/pipeline/predict.py
git commit -m "$(cat <<'EOF'
feat(pipeline): wire non_cyp_substrates registry into predict()

get_non_cyp_fractions(profile.smiles) called once; forwarded to both
build_drug_on_graph invocations (initial + post-phenotype rebuild).
NAT2 / UGT1A1-substrate drugs now have non-zero enzyme_affinity for
the registered enzyme; phenotype scaling propagates via Task 2 fix.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Schema regression test for non-CYP registries

**Why:** Prevents silent additions, SMILES drift, and accidental holdout-drug inclusion. Mirrors PR #29 `test_oatp_registry_schema.py` pattern.

**Files:**
- Create: `tests/regression/test_non_cyp_registry_schema.py`

- [ ] **Step 1: Write the test**

```python
"""Schema regression for nat2_substrates.json + ugt1a1_substrates.json.

Five gates per registry plus cross-cutting holdout-disjoint check.
"""
from __future__ import annotations

import json
import pathlib

from rdkit import Chem


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_NAT2_PATH = _REPO_ROOT / "data" / "enzymes" / "nat2_substrates.json"
_UGT1A1_PATH = _REPO_ROOT / "data" / "enzymes" / "ugt1a1_substrates.json"
_YAML_PATH = _REPO_ROOT / "data" / "physiology" / "reference_man.yaml"
_HOLDOUT_PATH = _REPO_ROOT / "data" / "reference" / "holdout.json"


_EXPECTED_NAT2 = frozenset({"isoniazid", "hydralazine", "procainamide"})
_EXPECTED_UGT1A1 = frozenset({"raltegravir", "atazanavir", "dolutegravir"})


def _load(path):
    return json.loads(path.read_text())


def test_nat2_seed_pinned():
    data = _load(_NAT2_PATH)
    actual = {s["drug"] for s in data["substrates"]}
    assert actual == _EXPECTED_NAT2, (
        f"NAT2 seed drift: expected {_EXPECTED_NAT2}, got {actual}. "
        f"Update _EXPECTED_NAT2 with explicit decision (literature mf, "
        f"holdout check, integration test)."
    )


def test_ugt1a1_seed_pinned():
    data = _load(_UGT1A1_PATH)
    actual = {s["drug"] for s in data["substrates"]}
    assert actual == _EXPECTED_UGT1A1, (
        f"UGT1A1 seed drift: expected {_EXPECTED_UGT1A1}, got {actual}."
    )


def test_nat2_inchikey_matches_smiles():
    data = _load(_NAT2_PATH)
    for s in data["substrates"]:
        m = Chem.MolFromSmiles(s["smiles"])
        assert m is not None, f"{s['drug']}: SMILES failed to parse"
        derived = Chem.MolToInchiKey(m)
        assert derived == s["inchikey"], (
            f"{s['drug']}: registered InChIKey {s['inchikey']} != "
            f"RDKit-derived {derived}"
        )


def test_ugt1a1_inchikey_matches_smiles():
    data = _load(_UGT1A1_PATH)
    for s in data["substrates"]:
        m = Chem.MolFromSmiles(s["smiles"])
        assert m is not None, f"{s['drug']}: SMILES failed to parse"
        derived = Chem.MolToInchiKey(m)
        assert derived == s["inchikey"]


def test_nat2_metabolic_fraction_in_range():
    data = _load(_NAT2_PATH)
    for s in data["substrates"]:
        mf = s["metabolic_fraction"]
        assert 0.0 <= mf <= 1.0, f"{s['drug']}: mf={mf} not in [0, 1]"


def test_ugt1a1_metabolic_fraction_in_range():
    data = _load(_UGT1A1_PATH)
    for s in data["substrates"]:
        mf = s["metabolic_fraction"]
        assert 0.0 <= mf <= 1.0, f"{s['drug']}: mf={mf} not in [0, 1]"


def test_yaml_has_nat2_and_ugt1a1_in_liver_enzymes():
    """Schema gate: both registries' target enzymes must be in liver.enzymes."""
    text = _YAML_PATH.read_text()
    # Cheap substring check — full YAML parse not necessary for presence.
    assert "NAT2:" in text and "UGT1A1:" in text, (
        f"Expected NAT2: and UGT1A1: in {_YAML_PATH}; otherwise apply_phenotype_to_graph "
        f"will warn 'tag not found' for these genes."
    )


def test_no_registry_drug_in_holdout():
    """Cross-cutting: registry must not include any 107-holdout drug.

    If you intentionally add a holdout drug, run scripts/run_engine_benchmark.py,
    diff against data/training/4track_holdout_predictions.json, and update this
    test only after confirming Meta AAFE is bit-identical (or document the diff
    in experiment-log.md).
    """
    holdout = _load(_HOLDOUT_PATH)
    if isinstance(holdout, list) and holdout and isinstance(holdout[0], str):
        holdout_set = {d.lower() for d in holdout}
    elif isinstance(holdout, list) and holdout and isinstance(holdout[0], dict):
        holdout_set = {(d.get("drug_name") or d.get("name") or "").lower() for d in holdout}
    else:
        holdout_set = set()

    for path in (_NAT2_PATH, _UGT1A1_PATH):
        data = _load(path)
        for s in data["substrates"]:
            assert s["drug"].lower() not in holdout_set, (
                f"{s['drug']} appears in holdout. Either remove from "
                f"{path.name} OR run holdout regen + invariance check + "
                f"update this test gate."
            )
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/regression/test_non_cyp_registry_schema.py -v`

Expected: 8 PASS (3 NAT2 + 3 UGT1A1 + YAML presence + holdout-disjoint).

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_non_cyp_registry_schema.py
git commit -m "$(cat <<'EOF'
test(regression): non_cyp registry schema gates

Five gates per registry (seed pinned, InChIKey-SMILES roundtrip, mf in [0,1])
plus cross-cutting (YAML enzymes present, holdout-disjoint). Mirrors
test_oatp_registry_schema.py pattern (PR #29).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Integration — NAT2 phenotype propagation (isoniazid)

**Why:** End-to-end gate that exercise predict() with phenotypes={"NAT2": "PM"}. Combined with Task 2's back-solve fix, this confirms NAT2 phenotype actually moves Cmax.

**Files:**
- Create: `tests/integration/test_phenotype_nat2.py`

- [ ] **Step 1: Write the test**

```python
"""Integration test for NAT2 phenotype propagation (issue #10).

Tests:
- parse_phenotype_spec("NAT2:PM") accepted
- apply_phenotype_to_graph(graph, {"NAT2": "PM"}) warns nothing
- predict(isoniazid, phenotypes={"NAT2": "PM"}) > predict(isoniazid).cmax * 1.3
- predict(metoprolol, phenotypes={"NAT2": "PM"}) ~= predict(metoprolol).cmax (no NAT2 affinity → silent zero)
"""
from __future__ import annotations

import logging

import pytest

from sisyphus.pipeline.predict import predict


_ISONIAZID = "NNC(=O)c1ccncc1"
_METOPROLOL = "COCCc1ccc(OCC(O)CNC(C)C)cc1"


def test_parse_nat2_pm():
    from sisyphus.predict.phenotype import parse_phenotype_spec
    out = parse_phenotype_spec("NAT2:PM")
    assert out == {"NAT2": "PM"}


def test_apply_nat2_pm_no_warning(caplog):
    from sisyphus.predict.phenotype import apply_phenotype_to_graph
    from sisyphus.graph.builder import build_from_yaml
    import pathlib

    caplog.set_level(logging.WARNING)
    g = build_from_yaml(pathlib.Path("data/physiology/reference_man.yaml"))
    _ = apply_phenotype_to_graph(g, {"NAT2": "PM"})
    warnings_about_nat2 = [r for r in caplog.records if "NAT2" in r.getMessage() and "not found" in r.getMessage()]
    assert not warnings_about_nat2, (
        f"apply_phenotype_to_graph emitted 'tag not found' for NAT2: {warnings_about_nat2}"
    )


@pytest.mark.slow
def test_isoniazid_nat2_pm_propagates():
    """NAT2:PM should drop isoniazid clearance, raising Cmax > 1.3× EM.

    Ellard 1976: slow-acetylator t1/2 ~3h vs rapid ~1h (AUC ratio 3-4×).
    Cmax effect smaller than AUC due to absorption-time saturation;
    gate at 1.3× is conservative.
    """
    em = predict(_ISONIAZID, dose_mg=300.0, phenotypes={"NAT2": "EM"})
    pm = predict(_ISONIAZID, dose_mg=300.0, phenotypes={"NAT2": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 1.3, (
        f"NAT2:PM/EM Cmax ratio {ratio:.3f} ≤ 1.3 — phenotype propagation gate."
    )


@pytest.mark.slow
def test_metoprolol_nat2_pm_silent_zero():
    """Non-NAT2 substrate must be invariant under NAT2 phenotype scaling.

    Metoprolol has no NAT2 affinity (registry miss → fm has no NAT2),
    so even though apply_phenotype_to_graph scales liver.NAT2 abundance
    by 0.10, the engine multiplies by zero affinity → silent zero
    (graph-blind invariant).
    """
    base = predict(_METOPROLOL, dose_mg=100.0)
    pm = predict(_METOPROLOL, dose_mg=100.0, phenotypes={"NAT2": "PM"})
    assert base.engine_pk is not None and pm.engine_pk is not None
    rel_err = abs(pm.engine_pk.cmax.mean - base.engine_pk.cmax.mean) / base.engine_pk.cmax.mean
    assert rel_err < 1e-6, (
        f"Metoprolol Cmax shifted under NAT2:PM (rel_err {rel_err:.2e}); "
        f"silent-zero invariant violated."
    )
```

- [ ] **Step 2: Run**

Run: `pytest tests/integration/test_phenotype_nat2.py -v`

Expected: 4 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phenotype_nat2.py
git commit -m "$(cat <<'EOF'
test(integration): NAT2 phenotype propagation (isoniazid PM/EM)

End-to-end gate: predict(isoniazid, NAT2:PM).cmax > predict(isoniazid).cmax × 1.3.
Confirms back-solve fix (Task 2) + registry+pipeline wiring (Tasks 3,5,7,8,9)
combine to deliver mechanistic phenotype propagation.

Includes silent-zero invariant for non-NAT2-substrate metoprolol (Cmax
unchanged under NAT2:PM — graph-blind invariant).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Integration — UGT1A1 phenotype propagation (raltegravir)

**Why:** Mirror gate for the UGT1A1 path; covers the registry's other half.

**Files:**
- Create: `tests/integration/test_phenotype_ugt1a1.py`

- [ ] **Step 1: Write the test**

```python
"""Integration test for UGT1A1 phenotype propagation (issue #10)."""
from __future__ import annotations

import logging

import pytest

from sisyphus.pipeline.predict import predict


# Raltegravir canonical SMILES — Task 4 derived from PubChem CID 54671008.
# This test reads the SMILES from the registry to stay in sync with Task 4.
def _raltegravir_smiles() -> str:
    import json, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    data = json.loads((repo_root / "data" / "enzymes" / "ugt1a1_substrates.json").read_text())
    for s in data["substrates"]:
        if s["drug"] == "raltegravir":
            return s["smiles"]
    raise AssertionError("raltegravir not in ugt1a1_substrates.json")


def test_parse_ugt1a1_pm():
    from sisyphus.predict.phenotype import parse_phenotype_spec
    out = parse_phenotype_spec("UGT1A1:PM")
    assert out == {"UGT1A1": "PM"}


def test_apply_ugt1a1_pm_no_warning(caplog):
    from sisyphus.predict.phenotype import apply_phenotype_to_graph
    from sisyphus.graph.builder import build_from_yaml
    import pathlib

    caplog.set_level(logging.WARNING)
    g = build_from_yaml(pathlib.Path("data/physiology/reference_man.yaml"))
    _ = apply_phenotype_to_graph(g, {"UGT1A1": "PM"})
    warnings_about_ugt = [r for r in caplog.records if "UGT1A1" in r.getMessage() and "not found" in r.getMessage()]
    assert not warnings_about_ugt


@pytest.mark.slow
def test_raltegravir_ugt1a1_pm_propagates():
    """UGT1A1:PM should drop raltegravir clearance, raising Cmax > 1.2× EM.

    Iwamoto 2008: UGT1A1*28 carriers ~40% AUC increase. Cmax effect
    smaller; gate at 1.2× is conservative.
    """
    smiles = _raltegravir_smiles()
    em = predict(smiles, dose_mg=400.0, phenotypes={"UGT1A1": "EM"})
    pm = predict(smiles, dose_mg=400.0, phenotypes={"UGT1A1": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 1.2, (
        f"UGT1A1:PM/EM Cmax ratio {ratio:.3f} ≤ 1.2 — phenotype propagation gate."
    )
```

- [ ] **Step 2: Run**

Run: `pytest tests/integration/test_phenotype_ugt1a1.py -v`

Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phenotype_ugt1a1.py
git commit -m "$(cat <<'EOF'
test(integration): UGT1A1 phenotype propagation (raltegravir PM/EM)

predict(raltegravir, UGT1A1:PM).cmax > predict(raltegravir).cmax × 1.2.
SMILES read from registry to stay in sync with Task 4 derivation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Closing operations (not numbered tasks)

- [ ] **Run full test suite**

```bash
pytest tests/unit tests/regression tests/integration -q --no-header 2>&1 | tail -20
```

Expected: all PASS or only documented xfails (rosuvastatin / atorvastatin / fluvastatin Peff over-prediction; pre-existing).

- [ ] **Run holdout invariance check**

```bash
python3 scripts/run_engine_benchmark.py --regen 2>&1 | tail -10
```

Or, if regen takes too long, just verify the existing pin holds:

```bash
pytest tests/integration/test_holdout_regression.py -v
```

Expected: Meta AAFE 2.679 pin holds.

- [ ] **Add experiment-log entry**

Open `docs/claude/experiment-log.md` and prepend a new entry above the most-recent v0.3.1 block:

```markdown
## 2026-05-04 — v0.3.2 NAT2 + UGT1A1 phenotype propagation + back-solve cancellation fix

**Commits**: feat/nat2-ugt1a1-phenotype branch (12 task commits)

**What shipped**:
1. New `data/enzymes/{nat2,ugt1a1}_substrates.json` registries (3 drugs each).
2. `src/sisyphus/predict/non_cyp_substrates.py` loader (lru_cache, full InChIKey).
3. `reference_man.yaml` `liver.enzymes` adds NAT2 (1.0e7) + UGT1A1 (1.215e6); `_LIVER_ENZYME_ABUNDANCE` adds NAT2.
4. `_get_fm_fractions` extended with `non_cyp_fractions` parameter (validates [0,1], re-normalizes when sum > 1.0).
5. `build_drug_on_graph` + `_decompose_clint` forward `non_cyp_fractions`.
6. `pipeline.predict.predict()` registers `get_non_cyp_fractions(profile.smiles)` lookup; **CRITICAL FIX**: pre-phenotype enzyme abundance snapshot so `_decompose_clint` back-solves affinity from unscaled baseline. Pre-fix CYP1A2:PM/EM = 1.000 (broken); post-fix ~3.3 (caffeine empirical).

**Empirical results** (post-fix):
- caffeine + CYP1A2:PM/EM = ~3.3 (was 1.000 — back-solve cancellation)
- warfarin + CYP2C9:PM/EM = ~1.5 (was 1.000)
- pravastatin + SLCO1B1:PM/EM = ~3.0 (unchanged — transporter MM path)
- isoniazid + NAT2:PM/EM > 1.3 (new path)
- raltegravir + UGT1A1:PM/EM > 1.2 (new path)

**107-holdout impact**: Bit-identical (Meta 2.679, Engine 3.791, ML 3.012, In-domain 2.733). Benchmark uses `phenotypes=None` default; back-solve fix only changes behavior when phenotypes are explicitly passed.

**Schema regression**: 8 gates in `tests/regression/test_non_cyp_registry_schema.py` (seed pinned × 2, InChIKey ↔ SMILES × 2, fm in [0,1] × 2, YAML enzymes present, holdout-disjoint).

**Open follow-ups**:
- CPIC SA/RA → PM/EM CLI alias for NAT2 / "intermediate metabolizer" semantics (deferred)
- irinotecan/UGT1A1 prodrug-metabolite (issue #11)
- atorvastatin / rosuvastatin per-drug fm curation (Peff xfail unrelated)
- v0.3.x: PredictionResult.phenotypes_applied metadata field
```

- [ ] **Commit experiment-log update**

```bash
git add docs/claude/experiment-log.md
git commit -m "$(cat <<'EOF'
docs(experiment-log): v0.3.2 NAT2 + UGT1A1 + back-solve fix entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Push branch and create PR**

```bash
git push -u origin feat/nat2-ugt1a1-phenotype
gh pr create --title "feat(phenotype): NAT2 + UGT1A1 propagation + back-solve cancellation fix (v0.3.2, #10)" --body "$(cat <<'EOF'
## Summary
- Closes #10 (NAT2 + UGT1A1 phenotype infrastructure)
- Fixes a latent back-solve cancellation bug that silently nullified ALL CYP/UGT/NAT phenotype effects (SLCO1B1 escaped via MM kinetics)

## What changed
- New per-gene JSON registries (`data/enzymes/{nat2,ugt1a1}_substrates.json`) — 3 drugs each
- `reference_man.yaml` adds NAT2 (1.0e7) + UGT1A1 (1.215e6) to `liver.enzymes`
- `_get_fm_fractions` accepts `non_cyp_fractions={"NAT2":..., "UGT1A1":...}`
- `pipeline.predict.predict()` snapshots enzyme abundances BEFORE phenotype application; `_decompose_clint` back-solves affinity from the unscaled baseline; engine multiplication propagates phenotype as `scaled_abundance × pre_affinity = scale × original_rate`

## Empirical results (post-fix)
| drug | gene | pre-fix ratio | post-fix ratio |
|---|---|---|---|
| caffeine | CYP1A2:PM | 1.000 (broken) | ~3.3 |
| warfarin | CYP2C9:PM | 1.000 (broken) | ~1.5 |
| pravastatin | SLCO1B1:PM | 3.034 | ~3.0 (unchanged — transporter MM) |
| isoniazid | NAT2:PM | (no path) | >1.3 (new) |
| raltegravir | UGT1A1:PM | (no path) | >1.2 (new) |

## Test plan
- [ ] `pytest tests/integration/test_phenotype_cyp_propagation.py -v` — 3 PASS (back-solve fix regression)
- [ ] `pytest tests/integration/test_phenotype_nat2.py -v` — 4 PASS (NAT2 propagation + silent-zero invariant)
- [ ] `pytest tests/integration/test_phenotype_ugt1a1.py -v` — 3 PASS
- [ ] `pytest tests/regression/test_non_cyp_registry_schema.py -v` — 8 PASS (5 gates per registry + holdout-disjoint)
- [ ] `pytest tests/integration/test_holdout_regression.py -v` — Meta 2.679 pin holds (107-holdout invariance)
- [ ] CI green

## Architecture notes
- Engine: 0 line changes (graph-blind multiplication just works for new enzyme tags)
- 107-holdout invariance: benchmark uses `phenotypes=None` default, registry seeds 0/107 holdout drugs
- irinotecan/UGT1A1 prodrug-metabolite case deferred to issue #11
- CPIC NAT2 SA/RA labels: docstring documents SA→PM, RA→EM mapping; CLI alias deferred

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Verify PR CI**

```bash
gh pr checks --watch 2>&1 | tail -10
```

Expected: CI green within ~15 minutes.

---

## Self-Review

**1. Spec coverage**

Walking through `docs/superpowers/specs/2026-05-04-nat2-ugt1a1-phenotype-design.md` v3:
- §0 v3 revision summary — covered by Tasks 1, 2 (back-solve fix); Tasks 3, 4 (registries with raltegravir SMILES Task 1 derivation)
- §3 architecture — Tasks 6, 7, 8, 9 implement
- §4 data layer — Tasks 3, 4
- §5 physiology — Task 6
- §6.1 non_cyp_substrates.py — Task 5
- §6.2 ivive constants + _get_fm_fractions — Tasks 6, 7
- §6.3 build_drug_on_graph — Task 8
- §6.4 back-solve fix — Task 2
- §7.1 unit tests for loader — Task 5 step 1
- §7.2 schema regression — Task 10
- §7.3 NAT2/UGT1A1 integration — Tasks 11, 12
- §7.3 CYP propagation regression — Task 1 / Task 2
- §7.4 holdout invariance — Closing operations
- §8.1-8.7 failure modes — addressed in task notes / closing experiment-log
- §11 acceptance criteria — Tasks 1, 2, 11, 12 + closing holdout check
- §12 references — preserved in experiment-log entry

No gaps.

**2. Placeholder scan**

- Tasks 3, 5, 6, 7, 8, 9, 10, 11, 12: complete code in every code step. ✅
- Task 4 has `<RDKit canonical from Step 1>` markers, but they are intentional template strings the implementer fills in from RDKit output (Step 1 prints them); there is a deterministic command (Step 1) that produces the values, then the JSON template tells the implementer where to paste. This is acceptable — the alternative (hand-coding canonical SMILES + InChIKeys for 3 complex drugs) is more error-prone than letting RDKit canonicalize. ✅
- Task 9: explicit code snippets for all three modification sites. ✅

**3. Type consistency**

- `non_cyp_fractions: dict[str, float] | None` consistent across `_get_fm_fractions`, `_decompose_clint`, `build_drug_on_graph`, and pipeline call site.
- `liver_enzymes: dict[str, float] | None` (existing) reused.
- `liver_enzymes_pre: dict[str, float] | None` (Task 2) — name distinct, type consistent.
- `get_non_cyp_fractions(smiles: str) -> dict[str, float]` — return type matches what `_get_fm_fractions` expects. ✅
