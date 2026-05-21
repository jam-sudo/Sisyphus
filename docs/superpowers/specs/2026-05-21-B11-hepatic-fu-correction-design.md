# B-11 — Hepatic intracellular fu correction for highly-bound drugs

**Date**: 2026-05-21
**Backlog**: B-11 (renamed from B-03.x clopidogrel CES1 calibration after Broad-scope decision)
**Predecessor**: B-03 clopidogrel prodrug registry (2026-05-20)

## 1. Goal

Reduce systematic over-prediction of plasma C<sub>max</sub> for highly protein-bound drugs in the 107-holdout by introducing a per-drug `fu_correction_liver` Distribution that scales the unbound fraction inside the well-stirred / parallel-tube hepatic extraction formula. Default value `1.0` preserves current behavior for non-curated drugs.

## 2. Mechanistic basis

Well-stirred and parallel-tube hepatic extraction models use plasma `fup` directly:

```
WS: CL = (Q × fup × CLint) / (Q + fup × CLint)
PT: CL = Q × (1 − exp(−fup × CLint / Q))
```

For highly-bound drugs (fup < 0.1) the `fup × CLint` term is small relative to Q, so hepatic CL is rate-limited by `fup × CLint`. This systematically under-predicts in-vivo hepatic CL for drugs with substantial albumin-facilitated hepatocyte uptake or active intracellular protein binding that raises the effective intracellular unbound fraction above plasma `fup`.

Primary literature corpus:

- Watanabe et al. 2009, *DMD* 37:1471–1480 — hepatic uptake transporters + intracellular fu for statins / OATP substrates
- Yamazaki et al. 2010, *DMD* 38:998–1005 — albumin-facilitated hepatocyte uptake
- Riccardi et al. 2017, *DMD* 45:781–790 — albumin-facilitated uptake review with measured ratios
- Patilea-Vrana & Unadkat 2017, *Clin Pharmacokinet* — fu,liver vs fu,plasma review

The mechanism: when `fu_inc / fu_plasma > 1`, replacing `fup` with `fup × fu_correction_liver` (with `fu_correction_liver = fu_inc / fu_plasma`) in the WS/PT formula recovers the literature in-vivo extraction.

## 3. Scope

### 3.1 In-scope

- **Audit** all 19 holdout drugs with `meta_fold > 3` (over-prediction signature). Triage into PPB-related vs other-mechanism.
- **Curate** the PPB-related subset (estimated 5–7 drugs) via primary-literature search. Drugs with measured `fu_inc / fu_plasma` ratios receive `literature_applied`; chemically similar siblings receive `class_extrapolated`; PPB candidates without data receive `ceiling_accepted` (value remains 1.0); other-mechanism drugs receive `not_applicable`.
- **Infrastructure**: registry file, loader module, DrugOnGraph field, ivive.py wiring, ClearanceFluxSpec + ProdrugActivationFluxSpec apply changes, physiology YAML liver-node flag.
- **Tests**: 8 new tests in unit/regression covering schema, lookup, anti-fudge guard, identity-blindness, and flux integration.

### 3.2 Out-of-scope

- Gut-wall fu correction (different biology; intestinal permeability dominates).
- Renal fu correction (GFR does not use organ-level WS).
- Calibrating CYP / non-CYP affinity values (orthogonal to fu correction).
- Reverse-direction correction (`fu_correction < 1.0`): banned by anti-fudge guard for B-11, can be revisited.
- Under-prediction fixes (20 drugs with `meta_fold < 1/3`); separate initiative if pursued.

### 3.3 Audit-but-do-not-curate handling

The 19-drug audit produces a row in `hepatic_fu_correction.json` for every drug, even when the value stays 1.0. The disposition field documents why. This makes the audit conclusion legible in source control and prevents future re-investigation without prior context.

## 4. Architecture

### 4.1 Data flow

```
SMILES → MolecularProfile → ADMEProperties
                                  ↓
                          lookup_hepatic_fu_correction(smiles) → Distribution
                                  ↓
                          build_drug_on_graph(profile, adme, ...)
                                  ↓
                          DrugOnGraph.fu_correction_liver
                                  ↓
                          ResolvedParams.drug_param("fu_correction_liver")
                                  ↓
                          ClearanceFluxSpec.apply / ProdrugActivationFluxSpec.apply:
                            if params.node_param(source, "fu_correction_applicable"):
                                fup_effective = fup × fu_correction_liver
                            else:
                                fup_effective = fup
```

### 4.2 Files

| Path | Action | Purpose |
|---|---|---|
| `data/transporters/hepatic_fu_correction.json` | NEW | InChIKey-keyed registry; 19 audit rows after Phase B |
| `src/sisyphus/predict/hepatic_fu_correction.py` | NEW | Loader, `lookup_hepatic_fu_correction(smiles) → Distribution`, lru_cache, full-InChIKey + connectivity fallback |
| `src/sisyphus/core.py` | MODIFY | Add `DrugOnGraph.fu_correction_liver: Distribution = Distribution(mean=1.0, cv=0.0)`. Propagate through `realize_means()` and `sample(rng)` |
| `src/sisyphus/predict/ivive.py` | MODIFY | `build_drug_on_graph` calls lookup and threads value into DrugOnGraph |
| `src/sisyphus/engine/flux.py` | MODIFY | `ClearanceFluxSpec.apply` (well_stirred + parallel_tube branches) + `ProdrugActivationFluxSpec.apply`: gated correction |
| `data/physiology/reference_man.yaml` | MODIFY | Liver node gets `fu_correction_applicable: true` parameter |

`src/sisyphus/engine/compiler.py` and `solver.py` remain untouched (CLAUDE.md invariant #8 no-touch list).

### 4.3 Identity-blind preservation

The engine never matches the string `"liver"`. It reads `params.node_param(<source_name>, "fu_correction_applicable")`, which is declared in the physiology YAML. Renaming every organ to a random string still produces bit-identical numerical results, provided the flag travels with the renamed liver. CLAUDE.md invariant #1 preserved.

### 4.4 Backward compatibility

- Empty registry (Phase A end state) → every lookup returns the default `Distribution(mean=1.0, cv=0.0)` → `fup_effective = fup × 1.0 = fup` → bit-identical to current 107-holdout cache.
- Non-empty registry (Phase B end state) → only drugs with `mean > 1.0` shift; `mean = 1.0` entries are no-op audits.

### 4.5 ProdrugActivationFluxSpec inclusion

Phase A applies the correction to both `ClearanceFluxSpec` and `ProdrugActivationFluxSpec`. The literature for fu,inc/fu,p is dominated by hepatic uptake for elimination, not bioactivation; transferring the same ratio to bioactivation enzymes is an architectural assumption. For clopidogrel the assumption is benign (same enzymes, same hepatocyte environment); for other prodrugs revisit if curation expands.

## 5. Mechanism & curation protocol

### 5.1 Anti-fudge constraint

Loader-level validation: `fu_correction_liver.mean >= 1.0`. Sub-1.0 entries are rejected.

- `> 1.0`: physiologically motivated (albumin-facilitated uptake, intracellular protein binding) — permits raising effective CL → lowering Cmax → fixing over-prediction.
- `< 1.0`: would lower CL → raise Cmax → equivalent to fitting to under-prediction loss. Violates CLAUDE.md invariant #8 (`fudge parameters to Cmax loss (any form)`).

This is a B-11 scope constraint, not a permanent rule. A future iteration with a literature-driven `fu_inc < fu_plasma` case (e.g., lysosomal trapping data) can revisit.

### 5.2 Per-drug curation steps (1.5–3h each)

For each of 19 over-predict candidates:

1. **Mechanism triage**: PPB-related (fup < 0.1 AND hepatic-CL-dominant) → continue. Otherwise → `not_applicable` with one-line mechanism explanation (e.g., `renal excretion`, `CYP3A4 induction`, `unknown novel`).
2. **Literature search**:
   - Primary corpus: Watanabe 2009 Table 1–3, Yamazaki 2010 Table 1, Riccardi 2017 Table 2, Patilea-Vrana 2017 supplemental table.
   - Secondary: PubMed query `<drug> hepatic uptake intracellular`, `<drug> albumin-facilitated`, `<drug> fu liver`.
   - Target datum: explicit `fu_inc / fu_p` ratio OR hepatocyte uptake CL_int that resolves to a ratio.
3. **Disposition**:

| State | Trigger | Value | Documentation |
|---|---|---|---|
| `literature_applied` | Direct measurement | from paper | citation + DOI |
| `class_extrapolated` | Sibling drug measured | parent value, broader CV | both citations |
| `ceiling_accepted` | PPB candidate, no data | 1.0, cv=0 | search corpus list + `n_candidates_reviewed` |
| `not_applicable` | Non-PPB over-prediction | 1.0, cv=0 | mechanism note |

### 5.3 Registry shape

The clopidogrel entry below is illustrative shape only. The `mean = 8.5` value is a placeholder pending Phase B literature search; the final committed value will be whatever the cited paper reports.

```json
{
  "overrides": [
    {
      "drug": "clopidogrel",
      "smiles": "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
      "inchikey": "GKTWGGQPFAXNFI-HNNXBMFYSA-N",
      "fu_correction_liver": {"mean": <from-paper>, "cv": <from-paper-or-default-0.50>},
      "disposition": "literature_applied",
      "literature": [
        "<full citation with DOI>"
      ],
      "notes": "<rationale, derivation if not direct ratio>",
      "n_candidates_reviewed": <integer>,
      "source_dbs_searched": ["PubMed", "Watanabe 2009 Table N", "..."]
    },
    {
      "drug": "acamprosate",
      "smiles": "...",
      "inchikey": "...",
      "fu_correction_liver": {"mean": 1.0, "cv": 0.0},
      "disposition": "not_applicable",
      "notes": "Renal excretion dominates; no hepatic CL pathway to correct.",
      "literature": [],
      "n_candidates_reviewed": 0,
      "source_dbs_searched": []
    }
  ]
}
```

`fu_correction_liver.mean == 1.0` entries are no-ops at runtime; their value is the audit trail.

## 6. Phases

### 6.1 Phase A — Infrastructure (1–2 days, headline invariant)

Deliverables:

1. `data/transporters/hepatic_fu_correction.json` committed with `{"overrides": []}` (explicit empty array; the file exists from Phase A so test artifacts can be added by Phase B with a single edit rather than a file-create event).
2. `src/sisyphus/predict/hepatic_fu_correction.py` complete with lookup + connectivity fallback.
3. `src/sisyphus/core.py` extended with the new DrugOnGraph field + propagation.
4. `src/sisyphus/predict/ivive.py` wiring.
5. `src/sisyphus/engine/flux.py` gated correction in both flux specs.
6. `data/physiology/reference_man.yaml` flag.
7. 8 tests passing.

**Acceptance gate**:
- `data/training/4track_holdout_predictions.json` Meta AAFE bit-identical to current 2.7715 (registry empty → no behavior change).
- All existing tests pass (including `test_cached_holdout_aafe_is_2p772` and `test_enzyme_leak_audit`).
- Identity-blind test passes (random-rename invariance).

### 6.2 Phase B — Curation cycle (2–5 days, headline may shift)

Deliverables:

1. 19 audit rows in `hepatic_fu_correction.json`.
2. Per-drug literature search log.
3. Regenerated holdout cache + bootstrap CI bundle.
4. Per-drug pre/post Δfold table.
5. Headline ΔAAFE documented in `docs/claude/experiment-log.md`.

**Acceptance gate** (one of):

| Outcome | Meta AAFE shift | Action |
|---|---|---|
| **Success** | Improvement ≥ 1% | Ship B-11 (infra + curation), update headline, mark literature_applied entries shipped |
| **DE-37** | Shift < 0.5% | Ship infra only; curation rows kept as audit trail; B-11 reclassified as DE-37 (literature too thin); literature_applied entries kept if any |
| **Failure** | Worse by any amount | Revert curation; keep infra; per-drug RCA in retrospective |

### 6.3 Out-of-band rollback

- Empty `hepatic_fu_correction.json` → instant baseline.
- Delete one drug row → that drug only reverts.
- `git revert <commit-range>` → full rollback (engine/flux.py + DrugOnGraph field included).

## 7. Tests

### 7.1 New tests (Phase A)

```
tests/unit/test_hepatic_fu_correction.py:
  test_default_returns_one_for_unregistered
  test_inchikey_full_match
  test_inchikey_connectivity_fallback_for_stereo_variant
  test_loader_rejects_value_below_one   ← anti-fudge guard
  test_loader_rejects_missing_disposition
  test_loader_rejects_unknown_disposition

tests/regression/test_hepatic_fu_correction_schema.py:
  test_literature_applied_requires_citation
  test_disposition_in_allowed_set

tests/unit/test_flux_fu_correction_integration.py:
  test_clearance_flux_applies_correction_at_flagged_node_only
  test_prodrug_flux_applies_correction_at_flagged_node_only
  test_identity_blind_rename_invariant
```

### 7.2 Existing tests to update (Phase B only)

```
tests/integration/test_holdout_regression.py:
  test_cached_holdout_aafe_is_2p772 → pin to new value after Phase B
  (Phase A end: still 2.772 ± tolerance)

tests/regression/test_prodrug_v3_enzyme_leak_audit.py:
  DRUG_SPECIFIC_CHANGES expanded with any Phase B curated drugs

CLAUDE.md (gitignored) headline table refreshed
README.md headline table + reproducibility note refreshed
docs/claude/experiment-log.md new entry
docs/claude/backlog.md remove B-11 (or mark DE-37)
```

## 8. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Literature data sparse → mostly `ceiling_accepted` | Medium | DE-37 outcome | Phase B escape clause, infra still shipped |
| In-domain Meta AAFE worsens (curation hits wrong drug class) | Low | Headline narrative damage | Per-drug Δfold gate: any drug ≥ 50% worse → revert that row |
| Identity-blind invariant violated | Low | Architecture broken | Test #11 (random-rename invariance) is mandatory |
| Anti-fudge bypass via `< 1.0` value sneaking in | Low | CLAUDE.md #8 breach | Loader-level validation + test #4 |
| Engine flux change regresses other drugs | Low | Cross-drug regression | Full unit + regression + integration must pass; cache bit-identity gate in Phase A |
| ProdrugActivationFlux correction is wrong for non-clopidogrel prodrugs | Low | Wrong CL on simvastatin/irinotecan/etc | Phase A: no curation; bit-identical cache verifies no perturbation. Phase B: only clopidogrel is currently a prodrug-and-PPB candidate |

## 9. Open questions

- For `class_extrapolated` entries, how do we choose the sibling? (Same scaffold? Same enzyme system? Same fup tier?) → Decide per-case in Phase B; document choice in `notes`.
- Should non-PPB over-prediction drugs (e.g., acamprosate renal) trigger separate fixes? → **No for B-11.** Tracked in backlog as separate initiatives if pursued.
- Should we apply gut-wall fu correction for highly-bound drugs with first-pass gut metabolism? → **Out of scope for B-11.** Revisit if Phase B identifies a clear case.

## 10. Status

- Spec: draft 2026-05-21.
- Approval: pending user review.
- Implementation plan: pending writing-plans skill invocation after user approval.
