# Doctrine Completion Sprint — B-10 + B-03.x Design Spec

**Date:** 2026-05-24
**Cycle:** Doctrine completion sprint (post-B-11)
**Goal:** Close two explicitly flagged TODOs from prior cycles via literature-IVIVE only (no Cmax tuning).

---

## 1. Sprint Structure

Two independent phases. Phase A is guaranteed completion; Phase B is probabilistic with honest DE branch.

### Phase A — B-10: atorvastatin + rosuvastatin `metabolic_fraction` curation
- **Status:** v0.3 ECM doctrine completion (PR #30 deferred follow-up).
- **Files:** `data/transporters/cyp_clearance_overrides.json` (add entries), `data/transporters/oatp1b1.json` (flip `ecm_applicable: true`).
- **Headline effect:** zero (atorvastatin and rosuvastatin are NOT in the 107-holdout; only pravastatin is).
- **Production effect:** correct ECM routing for these statins; eliminates known double-counting (per pravastatin/pitavastatin precedent).
- **Test gate:** `tests/integration/test_oatp_ecm_statins` FE ≤ 1.3 must hold post-flip.
- **Out of scope:** fluvastatin (separate issue #21 — under-prediction direction conflicts with single mf curation, requires Jmax/Km recalibration not covered here).

### Phase B — B-03.x: clopidogrel CES1/CYP literature-IVIVE
- **Status:** B-03 doctrine completion (explicit TODO in CLAUDE.md).
- **File:** `data/sbi/prodrug_activation_registry.json` (clopidogrel entry by SMILES key `COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1`).
- **Current state:** CES1/CYP3A4/CYP2C9 = 0.030/0.030/0.030 (`disposition_state: ceiling_accepted`); calibrated to ~85/15 fate split, NOT absolute parent extraction.
- **Symptom:** clopidogrel parent Meta fold 2.56× → 5.15× post-B-03 (under-extraction → over-predicts Cmax).
- **Headline effect target:** AAFE −0.01~0.02 if successful; FE 5.15× → ~2~3×.
- **Probabilistic outcome:**
  - ~55% literature data extractable AND IVIVE closes the gap → `disposition_state: literature_applied`
  - ~25% data extractable but IVIVE doesn't close (in-vitro→in-vivo scaling fails) → DE-38 entry, `ceiling_accepted` maintained
  - ~20% data not extractable (paywall / contradictory) → DE-38 entry, `ceiling_accepted` maintained

---

## 2. Doctrine Compliance

### Invariant #8 ("no Cmax loss fudging")

This sprint MUST use literature-derived numbers only. The following rules enforce that:

1. **Pre-committed sources** (§3). Implementer cannot substitute alternative sources without spec amendment.
2. **No iterative tuning**: compute IVIVE value once from literature, commit. Do NOT iterate on holdout AAFE to "find" a better number.
3. **DE-38 branch is acceptable**: if literature IVIVE does not close the gap, the honest answer is to maintain `ceiling_accepted` and log the dead-end. We do NOT then back-calculate a "matching" affinity.

### Invariant #1 (identity-blind engine)

Phase A and Phase B touch data registries only. No engine code changes. No new node/edge types. Invariant #1 untouched.

### Invariant #5 (holdout inviolable)

Clopidogrel IS in the 107-holdout (parent observation). Atorvastatin and rosuvastatin are NOT. The 107-holdout regen after Phase B will report a genuine post-update AAFE — this is not optimization on holdout, it is mechanism-driven update with downstream headline disclosure.

---

## 3. Pre-Committed Literature Sources

Implementer MUST use these sources. Alternative sources require spec amendment.

| Parameter | Primary source | Fallback | Value (if known) |
|---|---|---|---|
| **CES1 hepatic abundance** | Pharmaceutics 2025 (PMC12735975) | Sato 2012 | 1664.4 ± 781.7 pmol/mg microsomal protein |
| **CES1 clopidogrel Vmax/Km** | Mol Pharm 2025 PMC12673578 ("Confounding Effect of Hepatic CES1 Variability on Clopidogrel Oxidation") | PMC5369137 (MDPI 2017 review) | TBD by implementer |
| **CYP3A4/2C19 clopidogrel contribution ratio** | Kazui 2010 DMD 38:92-99 | Zahno 2010 Br J Pharmacol 161:393 | Relative %, already cited in B-03 |
| **Atorvastatin CYP3A4 CL_int** | Park 2008 (primary; cited via ResearchGate aggregator) | Reactome R-HSA-9754706 | 1.22 mL/min/nmol (para- + ortho-OH sum) |
| **Atorvastatin CYP3A5 CL_int** | Park 2008 | — | 0.37 mL/min/nmol |
| **Rosuvastatin fm partition** | Niemi 2009 Pharmacol Ther 125:84 (PMC2765590) | Martin 2003 review | TBD by implementer (expected ~0.0~0.1 CYP, rest OATP1B1+biliary) |

**Source fallback rule:** if primary unavailable in 30 minutes of search, drop to fallback. If fallback also unavailable, log as DE-38 (Phase B only) or skip-with-rationale (Phase A — atorvastatin/rosuvastatin only).

---

## 4. Architecture (where things live)

### Phase A: `data/transporters/cyp_clearance_overrides.json`

Append two entries to `overrides[]`, following existing pravastatin/pitavastatin schema:

```json
{
  "drug": "atorvastatin",
  "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
  "inchikey": "XUKUURHRXDUEBC-KAYWLYCHSA-N",
  "metabolic_fraction": <literature-derived>,
  "literature": ["Park 2008 ...", "Niemi 2009 ...", ...],
  "notes": "<mechanism rationale, mf basis>"
}
```

And in `data/transporters/oatp1b1.json`, set `ecm_applicable: true` for atorvastatin and rosuvastatin entries.

### Phase B: `data/sbi/prodrug_activation_registry.json`

Update `enzyme_affinity_for_conversion` in clopidogrel entry. Replace placeholder 0.030 values with IVIVE-derived numbers. Update `affinity_source: "literature"` → `"literature_ivive"`, `disposition_state: "ceiling_accepted"` → `"literature_applied"` (if successful) or keep `ceiling_accepted` + add DE-38 note (if failed).

**No new files.** No new engine code. No new test infrastructure.

---

## 5. Phase A Detail (B-10)

### A.1 Atorvastatin

**Mechanism:** mixed CYP3A4 + OATP1B1. fm_CYP3A4 partition is non-trivial.

**Approach:**
1. Extract atorvastatin total in-vivo hepatic CL from FDA label or PK reference.
2. Estimate CYP3A4 + CYP3A5 contribution from Park 2008 CL_int × CYP3A4 abundance.
3. Compute fm_CYP = CL_CYP / CL_total → `metabolic_fraction`.
4. If fm_CYP = 0.7, `metabolic_fraction = 0.7` (XGBoost CL × 0.7 reflects CYP fraction; remaining 30% routed through ECM OATP1B1).

### A.2 Rosuvastatin

**Mechanism:** mostly biliary + OATP1B1, minimal CYP.

**Approach:**
1. Niemi 2009 review reports rosuvastatin CYP-mediated metabolism as <10% of total CL.
2. `metabolic_fraction = 0.0` if CYP contribution is negligible (similar to pravastatin/pitavastatin pattern).
3. If small but non-zero CYP contribution exists (e.g., 0.05), use that.

### A.3 ecm_applicable flip

In `data/transporters/oatp1b1.json`:
- `drugs.atorvastatin.ecm_applicable: true`
- `drugs.rosuvastatin.ecm_applicable: true`

Already exists for pravastatin and pitavastatin.

### A.4 Verification

Run `pytest tests/integration/test_oatp_ecm_statins.py` and verify all FE gates (≤ 1.3) hold for both new statins. If any FE exceeds 1.3, revert that statin's flip and log as Phase A partial completion.

---

## 6. Phase B Detail (B-03.x)

### B.1 Literature extraction

1. Fetch PMC12673578 (Mol Pharm 2025) and search for: Vmax, Km, k_cat, intrinsic clearance for clopidogrel hydrolysis.
2. If primary missing kinetics: fetch PMC5369137 (MDPI 2017 review).
3. If both unavailable: DE-38 path.

### B.2 IVIVE computation

Given Tang/Mol-Pharm Vmax (nmol/min/mg HLM) and Km (μM):

```
CL_int_per_mg_HLM = Vmax / Km   [mL/min/mg]
CL_int_per_pmol_CES1 = CL_int_per_mg_HLM / 1664.4 pmol/mg   [mL/min/pmol]
```

Convert to Sisyphus `enzyme_affinity` unit (CL_int per unit enzyme abundance). Compare to current placeholder 0.030. Replace if literature-derived.

### B.3 CYP3A4 / CYP2C9 update

Kazui 2010 reports CYP3A4 and CYP2C19 (mapped to Sisyphus CYP2C9 surrogate) relative contributions to clopidogrel bioactivation. Given CES1 absolute affinity (from B.2), derive CYP3A4 and CYP2C9 absolute affinities to preserve the 85/15 inactive/active fate split.

If CES1 IVIVE gives e.g. 0.15 (5× scale-up):
- CES1: 0.15 (yield=0, dead-end)
- CYP3A4 + CYP2C9: combined ~0.15 × (15/85) = 0.026 each (assuming equal partition; Kazui ratio for refinement)

### B.4 Verification

Run `scripts/run_engine_benchmark.py` on 107-holdout. Report:
- Clopidogrel parent FE: 5.15× → ?×
- 107-holdout Meta AAFE: 2.772 → ?

If clopidogrel FE improves AND Meta AAFE ≤ 2.772: SUCCESS, commit + update CLAUDE.md headline.
If clopidogrel FE improves but Meta AAFE > 2.772: partial — discuss with user.
If clopidogrel FE worsens or unchanged: revert, log DE-38.

---

## 7. Testing

### Phase A tests
- `tests/integration/test_oatp_ecm_statins.py` — existing; verify atorvastatin/rosuvastatin FE gates.
- `tests/regression/test_oatp_registry_schema.py` — existing; verify schema integrity (every `ecm_applicable=true` has paired `metabolic_fraction` entry).

### Phase B tests
- `tests/integration/test_holdout_regression.py::test_cached_holdout_aafe_is_2p772` — update AAFE pin if SUCCESS.
- `tests/regression/test_prodrug_v3_enzyme_leak_audit.py` — verify clopidogrel still in DRUG_SPECIFIC_CHANGES allowlist.
- New test: `tests/regression/test_clopidogrel_ces1_literature_applied.py` — verify `disposition_state == "literature_applied"` and `affinity_source == "literature_ivive"` post-update (Phase B SUCCESS only).

---

## 8. Risk Register

| Risk | Mitigation |
|---|---|
| Mol Pharm 2025 (PMC12673578) doesn't actually contain Vmax/Km | Fallback PMC5369137; if both fail → DE-38 |
| Park 2008 CL_int 1.22 attribution wrong (could be derived value) | Verify via Reactome ADME source list; cross-check with Lennernäs 2003 |
| Rosuvastatin fm CYP > 0.1 (Niemi review contradicted) | Use Niemi consensus; spec amendment if alternative source justifies |
| Clopidogrel CES1 IVIVE 5× over-scales (overcorrects 5.15× to <1×) | Honest outcome — DE-38 (in-vitro→in-vivo scaling failure); document direction |
| Phase A breaks atorvastatin or rosuvastatin FE gate | Revert that statin only; partial Phase A completion |
| Headline Meta AAFE regresses on non-clopidogrel drugs (RNG-order-style) | Should not occur (registry change is local); investigate if observed |

---

## 9. Execution

- **Workflow:** subagent-driven-development (B-11 pattern), opus across implementer + spec reviewer + code quality reviewer.
- **Branch:** direct work on main per CLAUDE.md repo state pattern; per-task commits.
- **Cache:** invalidate 107-holdout cache once at Phase B end (Phase A doesn't touch holdout).
- **Plan:** to be written via writing-plans skill after this spec is approved.
- **Duration:** ~1~1.5 days estimated.

---

## 10. Self-Maintenance (post-sprint)

### If Phase A SUCCESS only (Phase B DE-38)
1. Update `CLAUDE.md` "B-10 atorvastatin + rosuvastatin promoted" entry under v0.3 status.
2. Append `docs/claude/experiment-log.md` entry.
3. Append `docs/claude/dead-ends.md` DE-38 (Phase B closure).
4. Strike B-10 from `docs/claude/backlog.md` (~~B-10~~).

### If Phase A + Phase B SUCCESS
All of above, plus:
1. Update `CLAUDE.md` headline metrics table (Meta AAFE, In-domain Meta).
2. Regenerate `data/training/4track_holdout_predictions.json`.
3. Optionally bootstrap new 95% CI (10k resamples, seed=20260524) → `data/validation/4track_ci_2026-05-24.json`.

### If Phase A partial + Phase B DE-38
1. Document atorvastatin/rosuvastatin individually (whichever flipped successfully).
2. Phase B DE-38 entry.
3. Leave failed statin in backlog with reason.
