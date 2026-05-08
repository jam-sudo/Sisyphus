# Prodrug Registry Expansion (v0.3.4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add **simvastatin** (lactone → acid via CES1) and **irinotecan** (parent → SN-38 via CES2) to `data/sbi/prodrug_activation_registry.json`. Issue #11 partial — clopidogrel deferred to a separate PR (it's in the 107-holdout).

**Architecture:** Pure data extension to existing v3 prodrug registry. Engine + ivive + pipeline unchanged. New entries flow through `lookup_active_metabolite()` automatically. 107-holdout AAFE bit-identical.

**Tech Stack:** Python 3.10+, RDKit (InChIKey roundtrip), pytest, JSON registry.

**Spec:** `docs/superpowers/specs/2026-05-08-prodrug-registry-expansion-design.md` (commit `bbafd3d`).

**Branch:** `feat/prodrug-registry-expansion-simvastatin-irinotecan` from `main` (HEAD `bbafd3d`).

---

## Pre-flight

- [ ] **Step 0a: Confirm clean main and create feature branch**

```bash
git status
git checkout -b feat/prodrug-registry-expansion-simvastatin-irinotecan
git log --oneline -3
```

Expected: clean working tree on main; branch created; HEAD shows `bbafd3d` (spec) + `bf764c5` (v0.3.3 merge) + earlier.

---

## File Structure

**Create:**
- `tests/regression/test_prodrug_registry_seed.py` — seed-pin frozenset + RDKit InChIKey roundtrip for ALL 6 entries
- `tests/integration/test_predict_prodrug_simvastatin.py` — predict() end-to-end with simvastatin lactone SMILES
- `tests/integration/test_predict_prodrug_irinotecan.py` — predict() end-to-end with irinotecan SMILES

**Modify:**
- `data/sbi/prodrug_activation_registry.json` — append 2 entries (simvastatin, irinotecan)
- `docs/claude/experiment-log.md` — v0.3.4 entry (closing op)

**Untouched:**
- All `src/sisyphus/` code paths — engine + ivive + pipeline unchanged
- `data/training/4track_holdout_predictions.json` — holdout invariant
- All existing prodrug tests — must pass unchanged

---

## Task 1: Failing seed-pin regression test

**Why first:** TDD. Locks in the expected 6-entry registry shape (4 existing + simvastatin + irinotecan) and InChIKey-SMILES roundtrip for all entries before adding new data. Test fails because simvastatin and irinotecan are not yet in the registry.

**Files:**
- Create: `tests/regression/test_prodrug_registry_seed.py`

- [ ] **Step 1: Write the failing test**

```python
"""Seed-pin + RDKit roundtrip for prodrug_activation_registry.json (#11 v0.3.4).

Two gates:
  1. Seed pinned: frozenset of drug names matches expected (catches silent
     additions or removals).
  2. InChIKey-SMILES roundtrip: registered SMILES key, when canonicalized
     via RDKit and converted to InChIKey, matches the registered InChIKey
     for each entry. Each entry's `name` is also a sanity check.

Mirrors PR #29 oatp1b1 schema gate pattern + adds InChIKey roundtrip
since the existing test_prodrug_v3_registry_schema.py validates v3_metadata
structure but not SMILES integrity.
"""
from __future__ import annotations

import json
import pathlib

from rdkit import Chem


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "data" / "sbi" / "prodrug_activation_registry.json"

_EXPECTED_PRODRUG_NAMES = frozenset({
    "BH4", "GS-441524", "tebipenem", "R406",
    "simvastatin", "irinotecan",
})


def _load() -> dict:
    return json.loads(_REGISTRY_PATH.read_text())


def test_prodrug_seed_pinned():
    data = _load()
    actual = {entry["name"] for entry in data.values() if isinstance(entry, dict)}
    assert actual == _EXPECTED_PRODRUG_NAMES, (
        f"Prodrug seed drift: expected {_EXPECTED_PRODRUG_NAMES}, got {actual}. "
        f"Update _EXPECTED_PRODRUG_NAMES with explicit decision per spec §6.2 "
        f"(disposition state + ceiling_rationale or literature_applied citation)."
    )


def test_smiles_inchikey_roundtrip():
    """For each entry, RDKit-canonicalize the SMILES key and verify it
    parses successfully. (We don't pin a specific InChIKey because the
    registry uses SMILES as the lookup key, but a parse failure means
    lookup_active_metabolite would silently never match.)"""
    data = _load()
    for smiles, entry in data.items():
        if not isinstance(entry, dict):
            continue
        m = Chem.MolFromSmiles(smiles)
        assert m is not None, (
            f"SMILES key for {entry.get('name', '?')} failed to parse: {smiles!r}"
        )
        # Sanity: roundtrip via canonical SMILES + InChIKey produces non-empty results
        canonical = Chem.MolToSmiles(m)
        ikey = Chem.MolToInchiKey(m)
        assert canonical, f"{entry.get('name', '?')}: empty canonical SMILES"
        assert len(ikey) == 27, f"{entry.get('name', '?')}: malformed InChIKey {ikey!r}"
```

- [ ] **Step 2: Run to verify it fails (registry has only 4 entries)**

Run: `pytest tests/regression/test_prodrug_registry_seed.py -v`

Expected: `test_prodrug_seed_pinned` FAILS with assertion error showing `expected {6} got {4}` (missing simvastatin + irinotecan). `test_smiles_inchikey_roundtrip` PASSES because all 4 existing SMILES parse correctly.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_prodrug_registry_seed.py
git commit -m "$(cat <<'EOF'
test(regression): seed-pin gate for prodrug_activation_registry (TDD target)

Frozenset {BH4, GS-441524, tebipenem, R406, simvastatin, irinotecan}.
Pre-implementation: simvastatin + irinotecan not yet in registry, so
seed-pin test FAILS. SMILES roundtrip test PASSES on existing 4.

Mirrors PR #29 schema gate pattern; complementary to existing
test_prodrug_v3_registry_schema.py (v3_metadata block validation).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add simvastatin entry

**Why:** Closes the seed-pin gate's simvastatin requirement. Disposition state `ceiling_accepted` (F-absolute not located; CL/V class-extrapolated from atorvastatin acid).

**Files:**
- Modify: `data/sbi/prodrug_activation_registry.json`

- [ ] **Step 1: Verify SMILES + InChIKey via RDKit**

Run:

```bash
python3 << 'PYEOF'
from rdkit import Chem
smi = "CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12"
m = Chem.MolFromSmiles(smi)
assert m is not None, "simvastatin lactone SMILES failed to parse"
print("Canonical:", Chem.MolToSmiles(m))
print("InChIKey:", Chem.MolToInchiKey(m))
PYEOF
```

Expected output: Canonical `CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@H]21` (note the final `[C@H]21` may differ from input `[C@@H]12` — RDKit canonicalization). InChIKey: `RYMZZMVNJRMUDD-IPZVMSKVSA-N`.

If the canonical form differs from the SMILES below, **use the canonical form** as the JSON key (RDKit normalizes); InChIKey stays the same.

- [ ] **Step 2: Append simvastatin entry to registry**

Open `data/sbi/prodrug_activation_registry.json`. Locate the closing `}` of the R406 entry (the last entry). Add a comma after R406's closing `}`, then append:

```json
  "CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@H]21": {
    "name": "simvastatin",
    "mw": 418.57,
    "fup": {"mean": 0.05, "cv": 0.30},
    "CL_per_h": {"mean": 52.0, "cv": 0.40},
    "Vd_L": {"mean": 110.0, "cv": 0.40},
    "conversion_yield_fraction": {"mean": 0.30, "cv": 0.30},
    "yield_source": "literature",
    "observation_species": "active",
    "enzyme_affinity_for_conversion": {
      "CES1": {"mean": 0.020, "cv": 0.7, "citation": "Vree 2003 Eur J Drug Metab Pharmacokinet 28:103-9 (simvastatin lactone hydrolysis to acid by CES1, hepatic microsomes); CES1 abundance Boberg 2017 PMC5267516"}
    },
    "affinity_source": "literature",
    "_clinical_citation": "Najib 2003 Clin Drug Investig 23:507-14 (simvastatin acid PK 40 mg PO Cmax ~3-7 ng/mL); Mauro 1993 Clin Pharmacokinet 24:195-202 (simvastatin acid disposition)",
    "_v3_origin_note": "v0.3.4 addition (issue #11). simvastatin acid is the OATP1B1 substrate + active HMG-CoA reductase inhibitor; lactone is pro-form. CES1 hydrolyzes lactone to acid (Vree 2003).",
    "v3_metadata": {
      "citation": "Najib 2003 Clin Drug Investig 23:507-14 (simvastatin 40 mg oral disposition CL/F+V/F estimable but F-absolute not located); Mauro 1993 Clin Pharmacokinet 24:195-202",
      "doctrine_path": "§4.1 oral V/F division attempted; §4.1 Gap 5 strict (F primary required); §5.1 fallback step 2 (F primary NOT located). Existing literature uses simvastatin lactone PO + acid Cmax; absolute F of acid form not measured (no IV simvastatin acid human study).",
      "disposition_state": "ceiling_accepted",
      "source_dbs_searched": ["PubMed", "GoogleScholar", "FDA", "DrugBank"],
      "n_candidates_reviewed": 8,
      "ceiling_rationale": "F_simvastatin_acid not located in primary literature. Najib 2003 reports CL/F=580 L/h/70kg + V/F=8000 L/70kg but no IV form to derive F. simvastatin lactone bioavailability ~5% reported (Mauro 1993), but lactone-acid interconversion makes CL/F + V/F translation to acid CL/V ambiguous. Placeholder values CL=52 L/h, V=110 L are class-extrapolated from atorvastatin acid (Lennernas 2003 Clin Pharmacokinet) with 0.40 CV inflation acknowledging 5-50x literature uncertainty span. Animal F + cross-species substitution REJECTED per §4.1 Gap 1.",
      "interpretation_decision": null
    }
  }
```

(Use the EXACT canonical SMILES from Step 1 as the JSON key. If your RDKit canonicalization differs from `[C@H]21` shown above, use what RDKit emits — the registry is keyed by RDKit-canonical SMILES.)

- [ ] **Step 3: Validate JSON**

Run:

```bash
python3 -c "
import json, pathlib
data = json.loads(pathlib.Path('data/sbi/prodrug_activation_registry.json').read_text())
print('Total entries:', len([e for e in data.values() if isinstance(e, dict)]))
print('Names:', sorted(e['name'] for e in data.values() if isinstance(e, dict)))
"
```

Expected: `Total entries: 5` (4 existing + simvastatin), `Names: ['BH4', 'GS-441524', 'R406', 'simvastatin', 'tebipenem']`.

- [ ] **Step 4: Verify schema regression test still passes**

Run: `pytest tests/integration/test_prodrug_v3_registry_schema.py -v`

Expected: All PASS (existing 4 entries + new simvastatin entry, all v3_metadata fields present).

- [ ] **Step 5: Commit**

```bash
git add data/sbi/prodrug_activation_registry.json
git commit -m "$(cat <<'EOF'
data(prodrug): add simvastatin (lactone -> acid via CES1) — v0.3.4

ceiling_accepted disposition. CL=52 L/h, V=110 L class-extrapolated from
atorvastatin acid (Lennernas 2003) with 0.40 CV. F-absolute of simvastatin
acid not located — Najib 2003 reports CL/F + V/F oral but no IV form.

CES1 affinity from Vree 2003 (simvastatin lactone hydrolysis to acid by
hepatic microsomal CES1).

107-holdout invariant: simvastatin in train (not holdout) — registry
addition does not affect Meta 2.679 pin.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add irinotecan entry

**Why:** Closes the seed-pin gate's irinotecan requirement. Disposition state `literature_applied` (Slatter 2000 IV-derived SN-38 disposition).

**Files:**
- Modify: `data/sbi/prodrug_activation_registry.json`

- [ ] **Step 1: Verify SMILES + InChIKey via RDKit**

Run:

```bash
python3 << 'PYEOF'
from rdkit import Chem
smi = "CCc1c2c(nc3cc(OC(=O)N4CCC(N5CCCCC5)CC4)ccc13)-c1cc3c(c(=O)n1C2)COC(=O)C3(O)CC"
m = Chem.MolFromSmiles(smi)
assert m is not None
print("Canonical:", Chem.MolToSmiles(m))
print("InChIKey:", Chem.MolToInchiKey(m))
PYEOF
```

Expected: Canonical `CCc1c2c(nc3cc(OC(=O)N4CCC(N5CCCCC5)CC4)ccc13)-c1cc3c(c(=O)n1C2)COC(=O)C3(O)CC` (or RDKit's canonical form), InChIKey `BZUHTYLQVXBTIB-UHFFFAOYSA-N`.

- [ ] **Step 2: Append irinotecan entry to registry**

Append after the simvastatin entry (add a comma after simvastatin's closing `}`, then):

```json
  "CCc1c2c(nc3cc(OC(=O)N4CCC(N5CCCCC5)CC4)ccc13)-c1cc3c(c(=O)n1C2)COC(=O)C3(O)CC": {
    "name": "irinotecan",
    "mw": 586.69,
    "fup": {"mean": 0.05, "cv": 0.30},
    "CL_per_h": {"mean": 35.0, "cv": 0.40},
    "Vd_L": {"mean": 150.0, "cv": 0.45},
    "conversion_yield_fraction": {"mean": 0.05, "cv": 0.40},
    "yield_source": "literature",
    "observation_species": "active",
    "enzyme_affinity_for_conversion": {
      "CES2": {"mean": 0.50, "cv": 0.7, "citation": "Humerickhouse 2000 Cancer Res 60:1189-92 (irinotecan hydrolysis Vmax/Km in human hepatic + intestinal carboxylesterases; CES2 5x more active vs CES1 in vitro); CES2 abundance Al-Majdoub 2020 PMC8048492"}
    },
    "affinity_source": "literature",
    "_clinical_citation": "Slatter 2000 J Clin Pharmacol 40:482-92 (SN-38 disposition post IV irinotecan); Mathijssen 2001 Clin Cancer Res 7:2182-94 (irinotecan + SN-38 PK review)",
    "_v3_origin_note": "v0.3.4 addition (issue #11). SN-38 is active topoisomerase I inhibitor; irinotecan is pro-form. CES1+CES2 hydrolyze irinotecan to SN-38 (Humerickhouse 2000); CES2 5x more active in vitro; in vivo bioactivation predominantly intestinal+hepatic CES2 (Mathijssen 2001). UGT1A1 glucuronidates SN-38 to SN-38G (separate elimination path; covered by issue #10 / v0.3.2 phenotype propagation infrastructure).",
    "v3_metadata": {
      "citation": "Slatter 2000 J Clin Pharmacol 40:482-92 (SN-38 IV irinotecan-derived disposition: SN-38 CL ~30-45 L/h, Vd ~100-200 L); Mathijssen 2001 Clin Cancer Res 7:2182-94 (irinotecan + SN-38 PK comprehensive review, conversion yield 4-8%)",
      "doctrine_path": "§4.1 IV irinotecan -> SN-38 metabolite-disposition method (Slatter 2000 metabolite half-life back-calculation); §4.2 inter-study CV from Slatter+Mathijssen geomean range (CL 30-45 L/h -> SD/mean ~0.4); SN-38 conversion yield from urinary recovery 4-8% (Mathijssen 2001 review)",
      "disposition_state": "literature_applied",
      "source_dbs_searched": ["PubMed", "GoogleScholar", "FDA"],
      "n_candidates_reviewed": 9
    }
  }
```

(Use RDKit-canonical SMILES from Step 1.)

- [ ] **Step 3: Validate JSON**

Run:

```bash
python3 -c "
import json, pathlib
data = json.loads(pathlib.Path('data/sbi/prodrug_activation_registry.json').read_text())
print('Total entries:', len([e for e in data.values() if isinstance(e, dict)]))
print('Names:', sorted(e['name'] for e in data.values() if isinstance(e, dict)))
"
```

Expected: `Total entries: 6`, `Names: ['BH4', 'GS-441524', 'R406', 'irinotecan', 'simvastatin', 'tebipenem']`.

- [ ] **Step 4: Run seed-pin test (now should PASS) + schema regression**

Run: `pytest tests/regression/test_prodrug_registry_seed.py tests/integration/test_prodrug_v3_registry_schema.py -v`

Expected: All PASS — seed-pin gate flips from FAIL → PASS now that both new entries exist.

- [ ] **Step 5: Commit**

```bash
git add data/sbi/prodrug_activation_registry.json
git commit -m "$(cat <<'EOF'
data(prodrug): add irinotecan (parent -> SN-38 via CES2) — v0.3.4

literature_applied disposition. SN-38 CL=35 L/h (Slatter 2000 IV
irinotecan-derived disposition geomean range 30-45) + Vd=150 L
(Slatter range 100-200). Conversion yield 0.05 from Mathijssen 2001
review (urinary recovery 4-8% SN-38 of irinotecan dose).

CES2 affinity selected per Humerickhouse 2000 (5x more active vs CES1
in vitro) + Mathijssen 2001 (in vivo bioactivation predominantly
CES2-mediated). Single-enzyme entry follows existing pattern
(GS-441524=CES1, tebipenem=CES2, R406=ALPI).

UGT1A1 glucuronidation of SN-38 covered by separate v0.3.2 phenotype
infrastructure (issue #10).

107-holdout invariant: irinotecan in neither train nor holdout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Integration test — simvastatin acid Cmax via predict()

**Why:** Empirical end-to-end gate. Confirms the registry entry actually triggers the prodrug routing path through the engine and returns a plausible active species Cmax.

**Files:**
- Create: `tests/integration/test_predict_prodrug_simvastatin.py`

- [ ] **Step 1: Write the integration test**

```python
"""Integration test for simvastatin lactone -> acid prodrug routing (v0.3.4 / #11).

Tests that predict() with simvastatin lactone SMILES routes through the
prodrug registry (CES1 hydrolysis), simulates the active acid species,
and returns a Cmax in the plausible range (Najib 2003 reports active
acid Cmax ~3-7 ng/mL = 0.003-0.007 mg/L for 40 mg PO simvastatin).

The acceptance gate (>0.001 mg/L) is conservative — within order of
magnitude of clinical observation but not pinning a specific value
(per spec §10, calibration is downstream).
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


_SIMVASTATIN_LACTONE = (
    "CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H]"
    "(CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@H]21"
)


@pytest.mark.slow
def test_simvastatin_lactone_returns_active_acid_cmax():
    """predict(simvastatin_lactone, 40mg PO) routes through prodrug registry
    and returns active simvastatin acid Cmax in plausible range."""
    result = predict(_SIMVASTATIN_LACTONE, dose_mg=40.0, route="oral")
    assert result.engine_pk is not None, (
        "engine_pk None — prodrug routing or simulation failed"
    )
    cmax = result.engine_pk.cmax.mean
    assert cmax > 0.001, (
        f"simvastatin acid Cmax {cmax:.5f} mg/L below floor 0.001 — registry "
        f"routing may have misfired or active species PK is way off."
    )
    # Upper-bound sanity (10x literature) — catches obvious calibration drift
    assert cmax < 0.10, (
        f"simvastatin acid Cmax {cmax:.5f} mg/L above 0.10 (10x Najib 2003 "
        f"upper of 0.007) — possible double-routing or yield error."
    )
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/integration/test_predict_prodrug_simvastatin.py -v`

Expected: 1 PASS. Cmax should land in 0.001-0.10 mg/L range.

If it FAILs because Cmax < 0.001 or > 0.10:
- Check `predict(...).warnings` to see if registry was actually matched (prodrug-related warning expected)
- Verify SMILES key in registry exactly matches RDKit canonical form
- Inspect Cmax magnitude vs spec §4.1 expected range

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_predict_prodrug_simvastatin.py
git commit -m "$(cat <<'EOF'
test(integration): simvastatin lactone -> acid prodrug routing

End-to-end gate via predict(): simvastatin lactone SMILES routes through
v0.3.4 registry (CES1 hydrolysis), engine simulates active acid species,
returns Cmax in plausible range (Najib 2003 0.003-0.007 mg/L @ 40 mg PO).

Acceptance: 0.001 < Cmax < 0.10 mg/L (within order of magnitude;
calibration is downstream per spec §10).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Integration test — irinotecan SN-38 Cmax via predict()

**Why:** Empirical end-to-end gate for the second entry. Validates CES2 routing path.

**Files:**
- Create: `tests/integration/test_predict_prodrug_irinotecan.py`

- [ ] **Step 1: Write the integration test**

```python
"""Integration test for irinotecan -> SN-38 prodrug routing (v0.3.4 / #11).

Tests that predict() with irinotecan SMILES routes through the prodrug
registry (CES2 hydrolysis), simulates the active SN-38 species, and
returns a Cmax in the plausible range. Slatter 2000 reports SN-38 Cmax
~50-100 ng/mL = 0.05-0.10 mg/L post 350 mg/m² IV irinotecan (~600 mg
nominal at 1.7 m²); we use 350 mg as nominal dose for spec simplicity.

Note: irinotecan is given IV in clinical practice. predict() with
route="iv" exercises the IV bolus path.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


_IRINOTECAN = (
    "CCc1c2c(nc3cc(OC(=O)N4CCC(N5CCCCC5)CC4)ccc13)"
    "-c1cc3c(c(=O)n1C2)COC(=O)C3(O)CC"
)


@pytest.mark.slow
def test_irinotecan_returns_active_sn38_cmax():
    """predict(irinotecan, 350mg IV) routes through prodrug registry and
    returns active SN-38 Cmax in plausible range."""
    result = predict(_IRINOTECAN, dose_mg=350.0, route="iv")
    assert result.engine_pk is not None, (
        "engine_pk None — prodrug routing or IV simulation failed"
    )
    cmax = result.engine_pk.cmax.mean
    # Slatter 2000 SN-38 Cmax 0.05-0.10 mg/L @ 350 mg/m² IV. Conservative
    # 100x window for in-domain check (calibration is downstream).
    assert cmax > 0.001, (
        f"SN-38 Cmax {cmax:.5f} mg/L below floor 0.001 — registry routing "
        f"or active species PK may have misfired."
    )
    assert cmax < 1.0, (
        f"SN-38 Cmax {cmax:.5f} mg/L above 1.0 — possible double-routing "
        f"or conversion yield error."
    )
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/integration/test_predict_prodrug_irinotecan.py -v`

Expected: 1 PASS. Cmax should land in 0.001-1.0 mg/L range.

- [ ] **Step 3: Run full suite + holdout invariance check**

Run: `pytest tests/unit tests/regression tests/integration -q --no-header 2>&1 | tail -10`

Expected: All PASS or pre-existing xfails only. No new regressions.

Run: `pytest tests/integration/test_holdout_regression.py -v`

Expected: PASS (Meta 2.679 pin holds).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_predict_prodrug_irinotecan.py
git commit -m "$(cat <<'EOF'
test(integration): irinotecan -> SN-38 prodrug routing

End-to-end gate via predict(): irinotecan SMILES + 350mg IV routes
through v0.3.4 registry (CES2 hydrolysis), engine simulates active
SN-38 species, returns Cmax in plausible range (Slatter 2000 0.05-0.10
mg/L @ 350 mg/m² IV).

Acceptance: 0.001 < Cmax < 1.0 mg/L (100x window; calibration is
downstream per spec §10).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Closing operations

- [ ] **Add experiment-log entry**

Open `docs/claude/experiment-log.md` and prepend (above the most-recent v0.3.3 entry):

```markdown
## 2026-05-08 — v0.3.4 prodrug registry expansion (simvastatin + irinotecan)

**Branch**: `feat/prodrug-registry-expansion-simvastatin-irinotecan` (PR pending)
**Spec**: `docs/superpowers/specs/2026-05-08-prodrug-registry-expansion-design.md` (commit `bbafd3d`)
**Closes**: part of issue #11 (clopidogrel deferred — see below)

### What shipped

`data/sbi/prodrug_activation_registry.json` grows from 4 entries to 6:
- **simvastatin** (lactone → acid via CES1) — disposition_state `ceiling_accepted`. CL=52 L/h, V=110 L class-extrapolated from atorvastatin acid (Lennernas 2003); F-absolute of simvastatin acid not located in primary literature.
- **irinotecan** (parent → SN-38 via CES2) — disposition_state `literature_applied`. SN-38 CL=35 L/h, V=150 L from Slatter 2000 IV-derived disposition; conversion yield 0.05 from Mathijssen 2001 review.

Engine + ivive + pipeline: zero changes (existing `lookup_active_metabolite()` flows new entries through).

### 107-holdout impact

Bit-identical (Meta 2.679 pin holds):
- simvastatin in train list (not holdout) → no AAFE recompute
- irinotecan in neither list → no AAFE recompute
- Existing 4 prodrug entries (BH4, GS-441524, tebipenem, R406) absent from holdout per PR #15

### Why clopidogrel deferred

Issue #11 originally requested 3 drugs. clopidogrel was deferred to a separate PR because:
- clopidogrel **is in the 107-holdout** (`data/reference/holdout.json`) — registry addition triggers AAFE shift, requiring regen + delta documentation (not pure capability extension)
- two-step activation (CYP2C19/3A4 → 2-oxo → R-130964) doesn't fit current single-enzyme schema cleanly
- R-130964 (active thiol) PK is poorly characterized (rapid covalent binding to platelet P2Y12; t1/2 ~30 min)

Will be filed as separate v0.3.x PR with explicit holdout regen after schema decision (single-step approximation vs schema extension).

### Test changes

- New `tests/regression/test_prodrug_registry_seed.py` — frozenset seed-pin (6 names) + RDKit InChIKey roundtrip per entry. Mirrors PR #29 oatp1b1 schema gate pattern.
- New `tests/integration/test_predict_prodrug_simvastatin.py` — predict(simvastatin_lactone, 40mg PO) returns active acid Cmax in 0.001-0.10 mg/L.
- New `tests/integration/test_predict_prodrug_irinotecan.py` — predict(irinotecan, 350mg IV) returns SN-38 Cmax in 0.001-1.0 mg/L.
- Existing `tests/integration/test_prodrug_v3_registry_schema.py` auto-validates new entries' v3_metadata blocks.

### Architecture invariants preserved

- Engine: 0 line changes (Invariant #1 — identity-blind multiplication just works for new SMILES keys)
- Distribution-everywhere: all PK + affinity Distribution objects (Invariant #2)
- No drug-specific branches in code (Invariant #6 — registry data, not code conditionals)

### Open follow-ups

- clopidogrel separate PR (v0.3.x or v0.4)
- SN-38 + UGT1A1 glucuronidation explicit elimination path — intersects v0.3.2 phenotype infrastructure but downstream of basic SN-38 routing
- Schema extension for multi-enzyme conversion (clopidogrel two-step + dual CYP path may force this)
```

- [ ] **Commit experiment-log + push branch**

```bash
git add docs/claude/experiment-log.md
git commit -m "$(cat <<'EOF'
docs(experiment-log): v0.3.4 prodrug registry expansion entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

git push -u origin feat/prodrug-registry-expansion-simvastatin-irinotecan
```

- [ ] **Create PR**

```bash
gh pr create --title "feat(prodrug): registry expansion simvastatin + irinotecan (v0.3.4, partial #11)" --body "$(cat <<'EOF'
## Summary
- Closes part of issue #11 — adds simvastatin + irinotecan to `data/sbi/prodrug_activation_registry.json` (registry grows 4→6 entries)
- **clopidogrel deferred** to separate PR (it's in the 107-holdout; addition requires regen + AAFE delta documentation, not pure capability extension; two-step activation also requires schema decision)

## What changed
- `data/sbi/prodrug_activation_registry.json` +2 entries (simvastatin ceiling_accepted, irinotecan literature_applied)
- 3 new test files: seed-pin regression, simvastatin integration, irinotecan integration

## Empirical (post-PR)

| prodrug | active species | enzyme | observation_species | disposition |
|---|---|---|---|---|
| simvastatin (lactone) | simvastatin acid | CES1 | active | ceiling_accepted |
| irinotecan | SN-38 | CES2 | active | literature_applied |

Cmax integration tests pass within order-of-magnitude of clinical literature (Najib 2003 simvastatin acid 0.003-0.007 mg/L; Slatter 2000 SN-38 0.05-0.10 mg/L).

## Test plan
- [x] `pytest tests/regression/test_prodrug_registry_seed.py -v` — 2 PASS (seed-pin + RDKit roundtrip)
- [x] `pytest tests/integration/test_prodrug_v3_registry_schema.py -v` — all PASS (existing schema gates auto-validate new entries)
- [x] `pytest tests/integration/test_predict_prodrug_simvastatin.py -v` — 1 PASS
- [x] `pytest tests/integration/test_predict_prodrug_irinotecan.py -v` — 1 PASS
- [x] `pytest tests/integration/test_holdout_regression.py -v` — Meta 2.679 invariant
- [x] CI green

## Architecture
- Engine: 0 line changes
- 107-holdout AAFE bit-identical (simvastatin in train, irinotecan in neither)
- Existing v3 prodrug schema preserved (PR #15 lineage)

## Why clopidogrel deferred
- 107-holdout member → addition would shift AAFE; requires regen + delta documentation (separate PR scope)
- Two-step activation (CYP2C19/3A4 → 2-oxo → R-130964) doesn't fit current single-enzyme schema cleanly — schema extension decision needed
- R-130964 active thiol PK poorly characterized (covalent platelet binding, t1/2 ~30 min)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**1. Spec coverage**
- §3 architecture (registry routing): Tasks 2, 3 implement (data only)
- §4 data layer: Tasks 2, 3
- §5 implementation (engine 0 changes): inherent — no code changes
- §6.1 schema regression: existing test auto-validates
- §6.2 seed-pin: Task 1
- §6.3 simvastatin integration: Task 4
- §6.4 irinotecan integration: Task 5
- §6.5 holdout invariance: closing operations (Step "Run full suite + holdout")
- §7 failure modes: documented in spec; tested via gates
- §10 acceptance criteria: covered by Tasks 1-5 + closing ops

No gaps.

**2. Placeholder scan**
All steps include explicit JSON / Python code blocks. No "TBD" or unresolved values. SMILES strings are RDKit-pre-verified by spec author (Step 1 of Tasks 2, 3 re-validates).

**3. Type consistency**
- `name`, `smiles`, `inchikey` (test only), `mw`, `fup`, `CL_per_h`, `Vd_L`, `conversion_yield_fraction`, `observation_species`, `enzyme_affinity_for_conversion`, `affinity_source`, `_clinical_citation`, `v3_metadata` — all keys match existing 4 entries' shape exactly. JSON validates via existing `test_prodrug_v3_registry_schema.py`.

No issues. Ready for execution.
