# N50 Secondary Permanent Holdout — Design Spec

**Date:** 2026-04-22
**Status:** Pre-registration (spec written before any data is admitted to N50)
**Author:** Hypatia (session a3cb907a continuation)
**Binding on:** all future Sisyphus accuracy-changing commits per `docs/claude/cherry_picking_process_v1.md` §1

---

## §1. Charter

The 107-drug retrospective holdout (`data/reference/holdout.json`) has been exposed to ~47 config feedback cycles (track weights, routing overrides, meta-learner variants) since its 2026-04-05 freeze. The cherry-picking audit (`docs/claude/cherry_picking_audit_2026-04-22.md`) scores aggregate risk 4.65/10 (moderate), with holdout dual-role as the top concern (7/10). The 95% CI for Meta AAFE 2.695 is [2.30, 3.20] — upper bound overlaps the audit's retrospective-contamination estimate (2.85–3.10), meaning the headline point estimate cannot be statistically distinguished from a tuning-inflated value.

**N50 is the measurement instrument that resolves this ambiguity.** It is a permanent, write-once holdout set used for unbiased generalization measurement at the end of each release cycle. N50 never feeds back into track weights, routing, meta-learner architecture, or any other modeling decision.

**Bi-purpose (D1 ∩ D6):** The N50 also serves as the independent substrate set required to falsify the ECM OATP1B1 non-statin underprediction observed in the V3 generalization test (2026-04-22, `7aa49ae`). Per `docs/claude/dead-ends.md` DE-33 and `project_ecm_generalization_test.md`, the N=2 result (valsartan + glimepiride, both underpredicting 2.5×) cannot distinguish between Jmax calibration error, Vss/Kp over-distribution, and ECM architectural limit without independent substrates. The N50 curation adds 3–5 OATP1B1 non-statin substrates drawn from sources outside any Sisyphus training corpus.

---

## §2. Size and Statistical Rationale

**N = 50 drugs.**

At N=50, bootstrap 95% CI on AAFE spans roughly ±20% of the point estimate, vs ±18% at N=107. Statistical precision is comparable while cherry-picking exposure drops to zero.

Below N=30, CI widens to ~±30% (underpowered for definitive comparison with the 107-holdout AAFE). Above N=100, curation cost scales super-linearly in the available-drug pool (remaining FDA NMEs 2024–2026 that are not already in MMPK/TDC/DrugBank training number roughly 80–120 candidates after verification).

**N=50 is the pragmatic lower bound satisfying: (a) 95% CI precision ≤ ±25%; (b) budget ≤ 1 session of literature curation work per cycle; (c) permit retirement after a single measurement and replacement with N50' for the next cycle.**

---

## §3. Exclusion Rules

A drug is admitted to N50 only if it satisfies **all** of:

**E1. Not in the 107-drug primary holdout.**
Cross-check `data/reference/holdout.json::holdout[]` (case-insensitive exact match).

**E2. Not in MMPK training corpus.**
Cross-check `data/training/mmpk_expanded_full.csv`, `mmpk_expanded_v2.csv`, `mmpk_pbpk_features.csv` (case-insensitive).

**E3. Not in TDC CLint training (Hepatocyte_AZ).**
Cross-check `data/training/clearance_hepatocyte_az.tab` (case-insensitive).

**E4. Not in DrugBank enrichment pool.**
Any drug whose DrugBank entry was used by `src/sisyphus/predict/drugbank.py` to fill fup/logP must be excluded. (Practically: if DrugBank has a measured ADME value, assume the drug has been seen. Conservative.)

**E5. Not present in any existing Sisyphus validation data file.**
Cross-check everything under `data/validation/` including prospective_* sets. (This avoids overlap with the prospective N=15 2024-25 NME set.)

A precomputed exclusion inventory is at `/tmp/n50_exclusion.txt` (2405 names, union of E1–E3, built by `scripts/build_n50_exclusion.py`). E4 and E5 are applied via runtime cross-check during curation.

---

## §4. Admission Criteria (positive requirements)

A candidate drug is admitted only if it satisfies **all** of:

**A1. Small-molecule, oral or IV, clinical PK dataset available.**
No biologics, no prodrugs requiring active metabolite measurement (unless the parent is what is dosed and the parent Cmax is clinically observed).

**A2. Primary-source citation for observed Cmax.**
Either:
- The primary PK paper reports Cmax in a table (preferred), OR
- An open-access systematic review tabulates Cmax and cites the primary paper (per spec discipline set in `2026-04-21-ecm-generalization-test-design.md` §Verification Requirements, acceptable when the primary is paywalled).
- Back-calculation from AUC + half-life is **forbidden** (parameter-choice degrees of freedom).

**A3. Structurally canonical SMILES.**
Resolved via PubChem canonical SMILES (CID-verified). No stereochemistry ambiguity relative to what was clinically dosed.

**A4. Administration specification.**
Route (oral, iv_bolus, iv_infusion_XYZ_min) and exact dose (mg, not mg/m²; convert using body surface area BSA=1.8 m² if the primary paper reports mg/m²; document the conversion).

**A5. Source DOI/URL + page/table reference.**
Every entry must cite the primary reference with DOI, page number, and table/figure identifier.

---

## §5. Subset Constraint: OATP1B1 Non-Statin Substrates

**Minimum 3, target 5, drawn from the N50 total (not in addition).**

An OATP1B1 substrate subset within N50 is flagged by `oatp1b1_substrate: true` in the JSON schema and supports independent ECM replication. To count toward this subset, a drug must:

- Be a documented OATP1B1 substrate in a peer-reviewed transporter review or primary paper (e.g., Niemi 2011 Pharmacol Rev, Kalliokoski 2009 BJP, Huang 2018 PMC6054689).
- Not be a statin (the ECM was calibrated on statins; including more statins would not be independent).
- Satisfy all E1–E5 and A1–A5.

**Pre-identified candidates from the 2026-04-22 gap audit (clean vs `/tmp/n50_exclusion.txt`):**

| Candidate | OATP1B1 evidence | Exclusion status | Notes |
|---|---|---|---|
| rifampin (low dose) | Tirona 2003 JPET | CLEAN | Dual substrate + inhibitor; must use single-dose IV PK (not steady state) to avoid induction confound |
| irinotecan | Nozawa 2005 DMD | CLEAN | Prodrug of SN-38 — dose irinotecan, measure irinotecan Cmax (not SN-38) |
| paclitaxel | Smith 2005 ChemBiolInteract | CLEAN | Primary OATP1B3, secondary OATP1B1 — acceptable as non-statin substrate |
| docetaxel | Smith 2005 | CLEAN | Similar to paclitaxel |
| caspofungin | Sandhu 2005 AAC | CLEAN | IV antifungal, OATP1B1 role reported |
| micafungin | Sandhu 2005 AAC | CLEAN | IV antifungal |
| temocaprilat | Ishizuka 1998 | CLEAN | Active metabolite — only if parent temocapril is NOT in training (verify) |
| benzylpenicillin | Hirano 2006 DMD | CLEAN | Weak OATP1B1 substrate |

Curation targets 5 of these, with preference for rifampin + irinotecan + paclitaxel (strongest evidence + richest PK literature).

---

## §6. Schema

File: `data/reference/holdout_n50.json` (to be created on first successful curation batch).

```json
{
  "description": "N50 secondary permanent holdout. Per docs/superpowers/specs/2026-04-22-n50-secondary-holdout-design.md. Never-touch for weight tuning, routing, or meta-learner feedback. One-time measurement per release cycle, then retired.",
  "spec_commit": "<sha at first commit>",
  "cycle_id": "2026Q2",
  "curation_date": "<YYYY-MM-DD>",
  "n_target": 50,
  "n_admitted": <int>,
  "oatp1b1_substrate_count": <int, must be >= 3>,
  "drugs": {
    "<drug_name>": {
      "status": "VERIFIED | BLOCKED",
      "smiles": "<PubChem CID canonical>",
      "pubchem_cid": <int>,
      "dose_mg": <float>,
      "route": "oral | iv_bolus | iv_infusion",
      "infusion_duration_min": <float or null>,
      "observed_cmax_mg_l": <float>,
      "observed_cmax_sd_mg_l": <float or null>,
      "patient_n": <int>,
      "oatp1b1_substrate": <bool>,
      "mechanism_tags": ["CYP3A4", "OATP1B1", "renal", ...],
      "source": {
        "citation": "<Author Year Journal>",
        "doi": "<10.xxxx/...>",
        "pmid": "<optional>",
        "table_ref": "<Table N, page X>",
        "access_path": "primary_paper | review_with_table",
        "review_doi": "<if access_path=review_with_table>"
      },
      "exclusion_checks": {
        "not_in_107": true,
        "not_in_mmpk": true,
        "not_in_tdc": true,
        "not_in_drugbank": true,
        "not_in_validation": true
      },
      "notes": "<free text>"
    }
  },
  "exclusion_inventory_hash": "<sha256 of /tmp/n50_exclusion.txt used at curation>",
  "methodology_notes": "mg/m² conversions used BSA=1.8; individual Cmax N=... averaged to mean when available."
}
```

---

## §7. Curation Workflow

1. **Spec commit** (this document) → establishes pre-registration.
2. **Exclusion inventory build** → `scripts/build_n50_exclusion.py` produces `/tmp/n50_exclusion.txt` (done 2026-04-22).
3. **Seed batch (this session)** → 5–10 drugs, of which ≥3 OATP1B1 substrates.
4. **Full curation** → subsequent sessions. Target 50.
5. **Freeze** → commit `data/reference/holdout_n50.json` with `n_admitted == 50`.
6. **Benchmark run** → a new benchmark script that loads `holdout_n50.json` and runs predictions once.
7. **CI reporting** → bootstrap 95% CI on N50 AAFE, published in CLAUDE.md headline.
8. **Retirement** → after measurement, mark cycle closed. Future cycles start a fresh N50' from different sources.

**Forbidden at any step:**
- Running the benchmark on a partial N50 and using the result to tune weights.
- Adding a drug after curation because "it would improve AAFE."
- Dropping a drug after curation because "its observed value looks wrong."
- Modifying admitted Cmax values (a BLOCKED drug stays BLOCKED; a VERIFIED drug stays VERIFIED).

Violations are reverts.

---

## §8. Retirement and Renewal

**After one benchmark measurement:** the N50 is retired. The AAFE result is published and frozen. The file is renamed `data/reference/holdout_n50_2026Q2_retired.json` and cannot be re-run.

**Next cycle (N50'):** fresh curation from sources that did not contribute to N50. FDA NMEs approved in the next cycle window; EU EPAR drugs not previously touched; published Clin Pharmacokinet retrospective reviews post-2026Q2.

**Why retire?** Using the same N50 twice re-establishes the cherry-picking risk we are trying to eliminate. The instrument must be single-use.

---

## §9. Open Questions (to resolve before freeze)

1. **BSA convention.** Is BSA=1.8 m² the right default, or should we use 1.73 m² (older standard)? ICRP reference man is 70 kg × 1.74 m height → BSA ≈ 1.85 m². Decision: **1.85 m²** (ICRP-consistent).
2. **Infusion Cmax observation timing.** For IV infusion drugs, clinical Cmax is often at end-of-infusion. Does V3 `t_min_h=5 min` apply, or should we use route-specific timing? Decision: **for `iv_infusion`, V3 `t_min_h` is extended to infusion_duration + 5 min.** Requires a minor V3.1 methodology amendment before the N50 benchmark run. Tracked as separate spec.
3. **Prodrug handling.** Irinotecan → SN-38: we dose irinotecan and measure irinotecan Cmax. The SN-38 measurement is a DDI/metabolism question, out of scope. Consistent with spec §A1.

---

## §10. Connection to Existing Docs

- `docs/claude/cherry_picking_process_v1.md` §1 — requires this N50.
- `docs/claude/cherry_picking_audit_2026-04-22.md` §7 item 1 — recommends this N50.
- `docs/claude/dead-ends.md` DE-33 — establishes ECM OATP non-statin underprediction as open question.
- `docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md` — methodology inheritance for OATP-specific admission.
- `docs/superpowers/specs/2026-04-22-iv-cmax-observation-design.md` — V3 methodology applied at N50 benchmark.

**Any future modification of this spec is a pre-registration violation** except the three Open Questions in §9 (resolved via separate spec amendment with dated commit before N50 freeze).
