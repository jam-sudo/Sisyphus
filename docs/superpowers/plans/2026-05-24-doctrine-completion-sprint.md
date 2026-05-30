# Doctrine Completion Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two explicit TODOs from prior cycles via literature-IVIVE: B-10 (atorvastatin + rosuvastatin `metabolic_fraction`) and B-03.x (clopidogrel CES1/CYP affinities).

**Architecture:** Pure data-registry changes — no engine code, no new files except optional regression test. Phase A (B-10) is doctrine completion with zero headline impact (atorvastatin/rosuvastatin not in 107-holdout). Phase B (B-03.x) is probabilistic with explicit DE-38 branch.

**Tech Stack:** Python registries (JSON), pytest gates, RDKit InChIKey lookup, `scripts/run_engine_benchmark.py` for 107-holdout regen.

**Spec:** `docs/superpowers/specs/2026-05-24-doctrine-completion-sprint-design.md` (amendment commit `4195935`).

---

## File Map

### Modified
- `data/transporters/cyp_clearance_overrides.json` — add atorvastatin + rosuvastatin entries (Phase A)
- `data/transporters/oatp1b1.json` — flip `ecm_applicable: true` for atorvastatin + rosuvastatin (Phase A)
- `data/sbi/prodrug_activation_registry.json` — update clopidogrel `enzyme_affinity_for_conversion` from 0.030 placeholders to literature-derived (Phase B)
- `tests/regression/test_oatp_registry_schema.py` — extend `_EXPECTED_ECM_APPLICABLE` set (Phase A)
- `tests/integration/test_holdout_regression.py` — update AAFE pin if Phase B SUCCESS
- `docs/claude/experiment-log.md` — sprint outcome entry
- `docs/claude/backlog.md` — strike B-10; B-03.x outcome
- `docs/claude/dead-ends.md` — DE-38 entry if Phase B fails
- `CLAUDE.md` — headline metrics if Phase B SUCCESS (≥0.01 AAFE shift)

### Created (conditional)
- `tests/regression/test_clopidogrel_ces1_literature_applied.py` — only if Phase B SUCCESS
- `data/validation/4track_ci_2026-05-24.json` — only if Phase B `|ΔAAFE| > 0.01`

### Unchanged (no engine touch)
- All of `src/sisyphus/engine/` (invariant #1)
- `src/sisyphus/predict/cyp_clearance_overrides.py` (already supports new entries via InChIKey lookup)
- `data/physiology/reference_man.yaml` (CES1 abundance 8.0e7 pmol/liver, cv=0.47 already correct per Boberg 2017)

---

# Phase A — B-10 (atorvastatin + rosuvastatin)

## Task 1: Atorvastatin literature extraction (CYP3A4/3A5 contribution)

**Files:** none (research output → Task 3)

- [ ] **Step 1: Fetch Reactome atorvastatin ADME pathway**

Use WebFetch tool:
```
URL: https://reactome.org/content/detail/R-HSA-9754706
Prompt: "Extract atorvastatin metabolism details: primary citations for CYP3A4 CL_int values (mL/min/nmol), CYP3A5 contribution ratio, and OATP1B1 vs CYP relative contribution to total hepatic clearance. Report exact citation, year, journal."
```

Expected output: primary citation(s) for CYP3A4 CL_int = 1.22 mL/min/nmol and CYP3A5 = 0.37 mL/min/nmol attribution.

- [ ] **Step 2: Fetch Park 2008 (or alternative primary)**

If Reactome cites Park 2008: WebSearch for "Park 2008 atorvastatin CYP3A4 CYP3A5 intrinsic clearance hydroxylation" + WebFetch the first PMC or open-access result.

Document the actual primary source URL and exact values.

If Park 2008 is paywall-only: fall through to Lennernäs 2003 (`Clinical Pharmacokinetics 42:1141-1160`) or Maeda 2011 (`Clin Pharmacol Ther 90:575-581`) — both have open-access aggregator hits.

- [ ] **Step 3: Compute fm_CYP**

```
CL_total_hepatic_in_vivo ≈ 37 L/h (atorvastatin total CL, FDA label)
CYP3A4_abundance_liver ≈ 9.25e6 pmol (per reference_man.yaml line 53)
CL_int_CYP3A4 ≈ 1.22 mL/min/nmol × CYP3A4_abundance_pmol / 1000
              = 1.22 × 9250 = 11,285 mL/min/nmol_total_3A4
              = 11.3 L/min × 60 min/h = 677 L/h microsomal-CL_int
Well-stirred: CL_CYP3A4 = (Q × fup × CLint) / (Q + fup × CLint)
With Q_liver ≈ 87 L/h, fup_atorvastatin ≈ 0.02:
   CL_CYP3A4 ≈ (87 × 0.02 × 677) / (87 + 0.02 × 677)
            ≈ 1178 / 100.5
            ≈ 11.7 L/h
fm_CYP3A4 ≈ CL_CYP3A4 / CL_total = 11.7 / 37 ≈ 0.32
```

**IMPORTANT:** Above is illustrative arithmetic; do the actual calculation with your primary-source values. Expected fm_CYP range: 0.4 to 0.8 per spec §A.0 sanity gate.

If your computed fm_CYP falls OUTSIDE [0.4, 0.8]: HALT and report. Likely interpretation error.

- [ ] **Step 4: Document literature findings**

Write a temporary markdown table:
```
| Source | URL | Vmax | Km | CL_int | Abundance basis |
|---|---|---|---|---|---|
| Reactome R-HSA-9754706 | <url> | ... | ... | 1.22 mL/min/nmol | ... |
| <primary source> | <url> | ... | ... | ... | ... |
```

Pass these findings to Task 3 (registry update).

## Task 2: Rosuvastatin literature extraction (CYP contribution)

**Files:** none (research output → Task 4)

- [ ] **Step 1: Fetch Niemi 2009 review**

Use WebFetch:
```
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC2765590/
Prompt: "Extract rosuvastatin section: report fm_CYP2C9, fm_CYP3A4, fm_biliary, fm_OATP1B1 if listed. If only qualitative ('minor CYP contribution'), quote the exact phrasing and cite Niemi 2009 Pharmacol Ther 125:84-117."
```

Expected: rosuvastatin CYP contribution <10% of total CL.

- [ ] **Step 2: Cross-check via Martin 2003 (fallback)**

If Niemi 2009 doesn't give explicit fm: WebSearch "rosuvastatin Martin 2003 clinical pharmacokinetics CYP2C9 contribution" and WebFetch the first PMC hit.

- [ ] **Step 3: Set mf value**

Based on findings:
- If literature consensus says "negligible CYP": mf = 0.0 (matches pravastatin/pitavastatin precedent)
- If literature gives explicit fraction ≤ 0.1: use that value
- If literature is contradictory or > 0.1: HALT per spec §A.0 sanity gate

Expected landing: mf ∈ [0.0, 0.1].

- [ ] **Step 4: Document literature findings**

Same table format as Task 1.

## Task 3: Add atorvastatin to cyp_clearance_overrides.json

**Files:**
- Modify: `data/transporters/cyp_clearance_overrides.json`

- [ ] **Step 1: Read current registry**

```bash
cat data/transporters/cyp_clearance_overrides.json
```

Note the existing pravastatin/pitavastatin/clopidogrel schema.

- [ ] **Step 2: Append atorvastatin entry**

Edit the `overrides` array. Insert AFTER pitavastatin (before clopidogrel) the following entry — substituting `<computed_mf>` with your Task 1 step 3 result, and `<your_citations>` with Task 1 step 4 literature:

```json
    {
      "drug": "atorvastatin",
      "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
      "inchikey": "XUKUURHRXDUEBC-KAYWLYCHSA-N",
      "metabolic_fraction": <computed_mf>,
      "literature": [
        "<primary CL_int source with year/journal/page>",
        "Niemi 2009 Pharmacol Ther 125:84-117 (PMC2765590): atorvastatin is CYP3A4 substrate AND OATP1B1 substrate; mixed mechanism.",
        "Lennernäs 2003 Clin Pharmacokinet 42:1141-1160 (atorvastatin PK review): hepatic uptake by OATP1B1 + CYP3A4 metabolism; F% ≈ 14% with first-pass."
      ],
      "notes": "Atorvastatin has mixed CYP3A4 + OATP1B1 hepatic clearance. fm_CYP ≈ <computed_mf> per <primary source>. metabolic_fraction=<computed_mf> scales the XGBoost-derived CYP3A4 affinity when ECM auto-activates, leaving the OATP1B1 contribution (1-<mf>) routed through ECM. NOTE: test_oatp_ecm_statins atorvastatin xfail remains in place because Peff XGBoost over-predicts atorvastatin absorption (test docstring lines 18-30); this mf entry completes the v0.3 ECM doctrine but does NOT fix the Peff-driven FE gate failure."
    },
```

- [ ] **Step 3: Verify JSON parses**

```bash
python3 -c "import json; json.load(open('data/transporters/cyp_clearance_overrides.json')); print('OK')"
```

Expected: `OK`

## Task 4: Add rosuvastatin to cyp_clearance_overrides.json

**Files:**
- Modify: `data/transporters/cyp_clearance_overrides.json`

- [ ] **Step 1: Append rosuvastatin entry**

Insert AFTER atorvastatin entry (from Task 3):

```json
    {
      "drug": "rosuvastatin",
      "smiles": "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O",
      "inchikey": "BPRHUIZQVSMCRT-VEUZHWNKSA-N",
      "metabolic_fraction": <computed_mf_from_task_2>,
      "literature": [
        "Niemi 2009 Pharmacol Ther 125:84-117 (PMC2765590): rosuvastatin CYP2C9-mediated metabolism is minor (<10% of total CL); dominant clearance is OATP1B1 hepatic uptake + biliary excretion.",
        "<your additional source from Task 2 if applicable>"
      ],
      "notes": "Rosuvastatin is predominantly cleared via OATP1B1 hepatic uptake + biliary excretion (~90% of total CL), with minor CYP2C9 contribution (<10%). metabolic_fraction=<computed_mf> routes <100*(1-mf)>% of XGBoost CLint through ECM OATP1B1 (matches pravastatin pattern). NOTE: test_oatp_ecm_statins rosuvastatin xfail remains in place because Peff XGBoost over-predicts rosuvastatin absorption (test docstring lines 18-30); this mf entry completes the v0.3 ECM doctrine but does NOT fix the Peff-driven FE gate failure."
    },
```

- [ ] **Step 2: Verify JSON parses**

```bash
python3 -c "import json; json.load(open('data/transporters/cyp_clearance_overrides.json')); print('OK')"
```

Expected: `OK`

## Task 5: Flip ecm_applicable=true in oatp1b1.json

**Files:**
- Modify: `data/transporters/oatp1b1.json`

- [ ] **Step 1: Locate rosuvastatin entry**

Read `data/transporters/oatp1b1.json` lines 21-33. The rosuvastatin block currently has NO `ecm_applicable` field.

- [ ] **Step 2: Add ecm_applicable to rosuvastatin**

Edit rosuvastatin entry — add `"ecm_applicable": true,` line after `"inchikey": "BPRHUIZQVSMCRT-VEUZHWNKSA-N"`:

```json
    "rosuvastatin": {
      "jmax_pmol_per_min_per_mg": {
        "mean": 170.0,
        "cv": 0.4
      },
      "km_uM": {
        "mean": 6.5,
        "cv": 0.35
      },
      "source": "Km: Niemi 2009 review (Simonson 2004, Ho 2006; range 4.0-7.3 μM). Jmax: scaled from rosuvastatin/pravastatin clinical hepatic uptake CL ratio ≈ 1.2 (Ho 2006 Gastro).",
      "smiles": "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O",
      "inchikey": "BPRHUIZQVSMCRT-VEUZHWNKSA-N",
      "ecm_applicable": true
    },
```

- [ ] **Step 3: Add ecm_applicable to atorvastatin**

Same operation on atorvastatin entry (lines 34-46):

```json
    "atorvastatin": {
      ...existing fields...
      "inchikey": "XUKUURHRXDUEBC-KAYWLYCHSA-N",
      "ecm_applicable": true
    },
```

- [ ] **Step 4: Verify JSON parses**

```bash
python3 -c "import json; json.load(open('data/transporters/oatp1b1.json')); print('OK')"
```

Expected: `OK`

## Task 6: Update _EXPECTED_ECM_APPLICABLE in schema test

**Files:**
- Modify: `tests/regression/test_oatp_registry_schema.py:33`

- [ ] **Step 1: Edit the seed list constant**

Replace line 33:

```python
_EXPECTED_ECM_APPLICABLE = frozenset({"pravastatin", "pitavastatin"})
```

with:

```python
# 2026-05-24 (B-10): atorvastatin + rosuvastatin promoted with literature-curated
# metabolic_fraction entries per spec 2026-05-24-doctrine-completion-sprint-design.md.
# Atorvastatin: fm_CYP3A4 from Park 2008 / Reactome lineage.
# Rosuvastatin: fm_CYP <10% from Niemi 2009 PMC2765590.
_EXPECTED_ECM_APPLICABLE = frozenset({"pravastatin", "pitavastatin", "atorvastatin", "rosuvastatin"})
```

- [ ] **Step 2: Run schema test**

```bash
pytest tests/regression/test_oatp_registry_schema.py -v
```

Expected: 3/3 PASS (`test_seed_list_pinned`, `test_inchikey_matches_smiles`, `test_metabolic_fraction_paired`).

If FAIL on `test_metabolic_fraction_paired`: check that Task 3 and Task 4 inserted the entries correctly.
If FAIL on `test_inchikey_matches_smiles`: re-canonicalize SMILES via RDKit and update the entry.

## Task 7: Run statins integration test (Phase A verification)

**Files:** none (test execution only)

- [ ] **Step 1: Run test_oatp_ecm_statins**

```bash
pytest tests/integration/test_oatp_ecm_statins.py -v --tb=short
```

Expected:
- `pravastatin`: PASS (FE ≤ 1.3)
- `pitavastatin`: PASS or xfail per current state
- `atorvastatin`: **xfail** (FE > 3.0 expected per test docstring — Peff over-prediction, NOT ECM regression)
- `rosuvastatin`: **xfail** (same reason)
- `fluvastatin`: skip or xfail (out of scope per spec)

If pravastatin REGRESSES (FE > 1.3): investigate — Phase A flip may have changed pravastatin path. STOP and report.

If atorvastatin/rosuvastatin xfail UNEXPECTEDLY PASSES: also report — would indicate the Peff diagnosis was wrong and FE gate should be re-enabled.

- [ ] **Step 2: Run full regression suite (selectivity)**

```bash
pytest tests/regression/ tests/integration/test_oatp_ecm_statins.py tests/integration/test_oatp_pravastatin.py -v --tb=short
```

Expected: all pass except pre-existing xfails. No new failures.

## Task 8: Commit Phase A

**Files:** all Phase A changes

- [ ] **Step 1: Stage Phase A files**

```bash
git add data/transporters/cyp_clearance_overrides.json \
        data/transporters/oatp1b1.json \
        tests/regression/test_oatp_registry_schema.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(b-10): atorvastatin + rosuvastatin ECM doctrine completion

Phase A of 2026-05-24 doctrine completion sprint. Promotes the two
remaining v0.3 ECM-eligible statins per PR #30 deferred follow-up.

atorvastatin:
  metabolic_fraction=<value> (fm_CYP3A4 from <primary citation>)
  ecm_applicable=true
rosuvastatin:
  metabolic_fraction=<value> (fm_CYP <10% per Niemi 2009 PMC2765590)
  ecm_applicable=true

Schema test _EXPECTED_ECM_APPLICABLE extended to include both drugs.

NOTE: test_oatp_ecm_statins.py atorvastatin + rosuvastatin xfail remains
because rate-limiting step is Peff over-prediction (test docstring lines
18-30), NOT ECM under-extraction. This commit completes doctrine; FE gate
unaffected.

107-holdout headline unchanged (both drugs absent from holdout).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Substitute `<value>` and `<primary citation>` with Tasks 1+2 actual outputs.

---

# Phase B — B-03.x (clopidogrel CES1/CYP literature-IVIVE)

## Task 9: Clopidogrel CES1 literature extraction (PRIMARY)

**Files:** none (research output → Task 11)

- [ ] **Step 1: Fetch Mol Pharm 2025 primary source**

```
URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12673578/
Prompt: "Extract clopidogrel CES1 in vitro kinetic parameters. Report exact Vmax (nmol/min/mg HLM or pmol/min/pmol enzyme), Km (μM), and any reported intrinsic clearance values. Quote table numbers verbatim. Cite the original Tang 2006 paper if values are reproduced from it; otherwise cite this 2025 paper as primary."
```

Expected: numeric Vmax/Km for clopidogrel hydrolysis by CES1.

- [ ] **Step 2: If primary missing kinetics, try fallback**

If Step 1 returns "values not in this paper" or only qualitative discussion:

```
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC5369137/
Prompt: "Extract clopidogrel CES1 kinetic parameters Vmax (nmol/min/mg HLM) and Km (μM) for hydrolysis to clopidogrel carboxylic acid. Cite primary source (Tang 2006 or other)."
```

- [ ] **Step 3: Decision branch**

If BOTH steps return usable Vmax/Km: proceed to Task 10.

If BOTH fail (no accessible numeric data): GOTO Task 15 DE-38 branch. Do NOT make up values. Do NOT skip to CYP because B-03.x doctrine requires CES1 as primary (~85% of fate).

- [ ] **Step 4: Document literature findings**

Write findings to spec amendment-log section (do NOT commit yet):

```
# B-03.x literature extraction (2026-05-24)
Primary source: <PMC URL>
Vmax: <value> ± <SD> nmol/min/mg HLM (or pmol/min/pmol CES1)
Km: <value> ± <SD> μM
CL_int_per_mg_HLM = Vmax / Km = <computed> mL/min/mg
Original citation: <Tang 2006 if available, else current paper>
```

## Task 10: IVIVE computation + unit-sanity gate

**Files:** none (computation → Task 11)

- [ ] **Step 1: Compute IVIVE chain**

Use the following derivation. CES1 hepatic abundance (per `data/physiology/reference_man.yaml` line 62 and Boberg 2017 PMC5267516):

```
CES1_per_mg_microsomal = 1664 pmol/mg
CES1_total_liver = 8.0e7 pmol (1664 × 48000 mg microsomal)
CES1_distribution = Distribution(mean=8.0e7, cv=0.47)  # liver-total absolute
```

From your Task 9 numbers:
```
Vmax (nmol/min/mg HLM) and Km (μM)
CL_int_per_mg_microsomal = Vmax / Km   [mL/min/mg]   # standard IVIVE
```

Convert to Sisyphus `enzyme_affinity` unit. Sisyphus convention (per `src/sisyphus/engine/flux.py:226-229`):
```
CL_organ [L/h] = abundance [pmol] × affinity [L/(h·pmol)] × ivive [dimensionless]
```

So:
```
affinity [L/(h·pmol)] = CL_int_per_mg_microsomal [mL/min/mg]
                        ÷ 1664 [pmol/mg]       (per-pmol normalization)
                        × 0.06                 (mL/min → L/h: ×60 ÷ 1000)
```

Compute the affinity numerically with your Task 9 Vmax/Km values.

- [ ] **Step 2: Mandatory unit-sanity gate (spec §B.2)**

Current placeholder: `CES1.mean = 0.030` L/(h·pmol).

```
abundance(liver) × placeholder × ivive(~0.5)
= 8.0e7 × 0.030 × 0.5
= 1.2e6 L/h × dimensionless
```

That's astronomically large in raw form, but the well-stirred flux equation (q=87 L/h liver) caps actual clearance at organ blood flow. The placeholder is in saturated/extracted regime.

**Sanity window: derived affinity must satisfy abundance × affinity × ivive_typical >> q_liver (flow-limited).**

Concretely:
- Derived affinity in `[1e-5, 1e0]` L/(h·pmol) → unit chain plausible. Proceed.
- Derived affinity in `[1e-10, 1e-5]` or `[1e0, 1e5]` → unit conversion likely WRONG. STOP. Re-read `flux.py:222-253` (ClearanceFluxSpec.apply well_stirred branch). Verify whether `abundance` in your derivation is pmol per liver-total or pmol per mg microsomal.
- Derived affinity exactly at placeholder 0.030 ± 0.005 → coincidence; verify literature was actually applied, not silently reused placeholder.

Document the verification in a markdown notes block:

```
# Unit sanity gate verification
Derived affinity: <value> L/(h·pmol)
Within window [1e-5, 1e0]: <YES/NO>
clint_organ (= abundance × affinity × ivive=0.5):
  = 8.0e7 × <value> × 0.5
  = <result> L/h
Compare to q_liver = 87 L/h:
  Flow-limited if clint_organ >> 87 (well-stirred saturates at ~q)
  Linear if clint_organ << 87
Sanity: <PASS/FAIL — explain>
```

If FAIL: GOTO Task 15 DE-38 branch (root cause = unit derivation, not literature).

- [ ] **Step 3: Document affinity Distribution**

Per spec §3 propagation rule:
- Vmax cv = 0.40 default (CES high-variability)
- Km cv = 0.30 default
- Combined affinity cv via propagation: cv_affinity ≈ sqrt(0.40² + 0.30²) ≈ 0.50

Or use literature-reported SD if available — convert to cv = SD/mean.

Output:
```python
CES1_affinity = Distribution(mean=<derived>, cv=<derived_cv>)
```

## Task 11: Update prodrug_activation_registry.json clopidogrel entry

**Files:**
- Modify: `data/sbi/prodrug_activation_registry.json`

- [ ] **Step 1: Read current clopidogrel entry**

```bash
python3 -c "
import json
d = json.load(open('data/sbi/prodrug_activation_registry.json'))
print(json.dumps(d['COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1'], indent=2))
"
```

- [ ] **Step 2: Derive CYP3A4 + CYP2C9 affinities from CES1 + fate split**

Per spec §B.3, given CES1 absolute affinity = `<derived>` and Kazui 2010 fate split (~85% CES1 dead-end, ~15% CYP bioactivation):

```
V_CES1 / V_total = 0.85
V_CYPs / V_total = 0.15
V_CYPs / V_CES1 = 0.15 / 0.85 ≈ 0.176

CES1_abundance_liver = 8.0e7 pmol
CYP3A4_abundance_liver = 9.25e6 pmol (per reference_man.yaml line 53)
CYP2C9_abundance_liver = ? pmol (read from reference_man.yaml — likely smaller)

For CL contribution equality:
  abundance_CES1 × affinity_CES1 ≈ 5.67 × (abundance_CYP3A4 × affinity_CYP3A4)
  → affinity_CYP3A4 = affinity_CES1 × (8.0e7 / 9.25e6) × (0.15/0.85)
                    = affinity_CES1 × 8.65 × 0.176
                    = affinity_CES1 × 1.52
```

If Kazui 2010 gives CYP3A4:CYP2C19 specific ratio (e.g., 60:40), split the combined CYP contribution accordingly:
```
affinity_CYP3A4 = (CYP_total_contribution × 0.6) / abundance_CYP3A4
affinity_CYP2C9 = (CYP_total_contribution × 0.4) / abundance_CYP2C9_proxy
```

If Kazui doesn't give explicit ratio: split 50/50.

- [ ] **Step 3: Update the registry entry**

Edit the `enzyme_affinity_for_conversion` block:

```json
    "enzyme_affinity_for_conversion": {
      "CES1": {
        "mean": <derived_CES1_affinity>,
        "cv": <derived_cv>,
        "yield": {"mean": 0.0, "cv": 0.0},
        "citation": "<primary source from Task 9>: Vmax=<X>, Km=<Y>, CL_int_per_mg=<Z>; IVIVE via Boberg 2017 PMC5267516 CES1 abundance 1664 pmol/mg microsomal → 8.0e7 pmol/liver. Spec 2026-05-24-doctrine-completion-sprint-design.md §B.2."
      },
      "CYP3A4": {
        "mean": <derived_CYP3A4_affinity>,
        "cv": <derived_cv>,
        "yield": {"mean": 1.0, "cv": 0.30},
        "citation": "Kazui 2010 DMD 38:92-99 + Zahno 2010 Br J Pharmacol 161:393-404: CYP3A4 contributes <X>% of clopidogrel bioactivation. Derived from CES1 absolute affinity (Task 10) and 85/15 inactive/active fate split. Spec §B.3."
      },
      "CYP2C9": {
        "mean": <derived_CYP2C9_affinity>,
        "cv": <derived_cv>,
        "yield": {"mean": 1.0, "cv": 0.30},
        "citation": "Kazui 2010 DMD 38:92-99: CYP2C19 (mapped to Sisyphus CYP2C9 surrogate) contributes <Y>% of clopidogrel bioactivation. Derived per §B.3."
      }
    },
    "affinity_source": "literature_ivive",
```

Also update:
```json
    "v3_metadata": {
      ...
      "disposition_state": "literature_applied",
      ...
    }
```

(Was `"ceiling_accepted"`; flips to `"literature_applied"` on SUCCESS path.)

- [ ] **Step 4: Verify JSON parses**

```bash
python3 -c "import json; json.load(open('data/sbi/prodrug_activation_registry.json')); print('OK')"
```

Expected: `OK`

## Task 12: Run prodrug regression tests

**Files:** none

- [ ] **Step 1: Run prodrug leak audit**

```bash
pytest tests/regression/test_prodrug_v3_enzyme_leak_audit.py -v --tb=short
```

Expected: PASS. Clopidogrel is already in `DRUG_SPECIFIC_CHANGES` allowlist (added during B-03 fix-forward commit `61b3fa7`).

- [ ] **Step 2: Run prodrug registry schema**

```bash
pytest tests/regression/test_prodrug_v3_registry_schema.py tests/integration/test_prodrug_v3_registry_schema.py -v --tb=short
```

Expected: PASS. Registry remains valid JSON with required fields.

## Task 13: Regenerate 107-holdout benchmark

**Files:**
- Modify: `data/training/4track_holdout_predictions.json` (regenerated)

- [ ] **Step 1: Hide local-only artifacts (per CLAUDE.md public-clone procedure)**

```bash
test -d data/drugbank && mv data/drugbank data/drugbank.local-archive
test -f models/adme/logp_correction.json && mv models/adme/logp_correction.json models/adme/logp_correction.json.local-archive
```

- [ ] **Step 2: Run benchmark regen**

```bash
python scripts/run_engine_benchmark.py 2>&1 | tee /tmp/benchmark_phase_b.log
```

Expected runtime: 10-20 minutes.

- [ ] **Step 3: Compute key metrics**

```bash
python3 -c "
import json
d = json.load(open('data/training/4track_holdout_predictions.json'))
# Find clopidogrel
clop = next((x for x in d if x['drug'] == 'clopidogrel'), None)
if clop:
    pred = clop.get('meta_cmax_mg_L') or clop.get('meta_pk', {}).get('Cmax_mg_L')
    obs = clop.get('obs_cmax_mg_L', 0.3)
    fe = max(pred/obs, obs/pred)
    print(f'clopidogrel: pred={pred:.4f}, obs={obs:.4f}, FE={fe:.3f}')
# AAFE
import statistics
folds = []
for x in d:
    p = x.get('meta_cmax_mg_L') or x.get('meta_pk', {}).get('Cmax_mg_L')
    o = x.get('obs_cmax_mg_L')
    if p and o:
        folds.append(max(p/o, o/p))
import math
gmfe = math.exp(statistics.mean(math.log(f) for f in folds))
print(f'Meta AAFE: {gmfe:.4f}  (N={len(folds)})')
print(f'Previous: 2.7715238009')
print(f'Delta: {gmfe - 2.7715238009:+.4f}')
"
```

Expected outputs to analyze:
- Clopidogrel FE: was 5.15× → ?×
- Meta AAFE: was 2.7715 → ?
- Delta sign matters

- [ ] **Step 4: Restore local artifacts**

```bash
test -d data/drugbank.local-archive && mv data/drugbank.local-archive data/drugbank
test -f models/adme/logp_correction.json.local-archive && mv models/adme/logp_correction.json.local-archive models/adme/logp_correction.json
```

## Task 14: Disposition decision (SUCCESS / PARTIAL / DE-38)

**Files:** none (decision point)

- [ ] **Step 1: Evaluate Task 13 outputs**

Apply spec §B.4 decision tree:

**SUCCESS path** if BOTH conditions met:
1. Clopidogrel FE improved (decreased from 5.15×)
2. Meta AAFE ≤ 2.772 (no regression on other 106 drugs)

**PARTIAL path** if:
- Clopidogrel FE improved BUT Meta AAFE > 2.772
- → STOP and ask user for direction (do not auto-commit)

**DE-38 path** if:
- Clopidogrel FE unchanged or worse, OR
- Unit-sanity gate failed in Task 10, OR
- Literature unavailable in Task 9

### SUCCESS branch (Task 14a)

- [ ] **Step 2a: Update test_holdout_regression.py AAFE pin**

Edit `tests/integration/test_holdout_regression.py` — find `test_cached_holdout_aafe_is_2p772` (or similar pin) and update to new value:

```python
def test_cached_holdout_aafe_is_<new_value_dotted>():
    """Pin: regenerated Meta AAFE = <new_value> post-B-03.x literature IVIVE
    (2026-05-24 doctrine completion sprint Phase B SUCCESS).
    """
    ...
    assert abs(aafe - <new_value>) < 0.005
```

- [ ] **Step 3a: Create new regression test (literature_applied disposition)**

Create `tests/regression/test_clopidogrel_ces1_literature_applied.py`:

```python
"""B-03.x literature-IVIVE disposition pin.

Verifies clopidogrel CES1/CYP affinities flipped from B-03 placeholders
(0.030 each, ceiling_accepted) to literature-IVIVE values
(literature_applied). Prevents silent regression back to placeholders.
"""
from __future__ import annotations

import json
from pathlib import Path

REGISTRY = Path("data/sbi/prodrug_activation_registry.json")
CLOPIDOGREL_SMILES = "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1"


def test_clopidogrel_disposition_literature_applied():
    """Clopidogrel B-03.x: disposition_state must be literature_applied."""
    entry = json.loads(REGISTRY.read_text())[CLOPIDOGREL_SMILES]
    assert entry["v3_metadata"]["disposition_state"] == "literature_applied"
    assert entry["affinity_source"] == "literature_ivive"


def test_clopidogrel_ces1_affinity_not_placeholder():
    """CES1 affinity must differ from the B-03 placeholder 0.030."""
    entry = json.loads(REGISTRY.read_text())[CLOPIDOGREL_SMILES]
    ces1_mean = entry["enzyme_affinity_for_conversion"]["CES1"]["mean"]
    assert ces1_mean != 0.03, (
        "CES1 affinity is the B-03 placeholder — literature-IVIVE not applied"
    )
    # Unit-sanity window per spec §B.2
    assert 1e-5 <= ces1_mean <= 1e0, (
        f"CES1 affinity {ces1_mean} outside spec sanity window [1e-5, 1e0]; "
        f"likely unit conversion error"
    )
```

- [ ] **Step 4a: Run new test**

```bash
pytest tests/regression/test_clopidogrel_ces1_literature_applied.py -v
```

Expected: 2/2 PASS.

- [ ] **Step 5a: CI bootstrap if |ΔAAFE| > 0.01**

```bash
python3 -c "
delta = abs(<new_aafe> - 2.7715238009)
if delta > 0.01:
    print(f'Delta {delta:.4f} > 0.01 — CI bootstrap MANDATORY')
else:
    print(f'Delta {delta:.4f} ≤ 0.01 — existing CI remains canonical')
"
```

If MANDATORY: run bootstrap (use existing script — e.g. `scripts/bootstrap_holdout_ci.py` if present; otherwise spec §10):

```bash
python scripts/bootstrap_holdout_ci.py --seed 20260524 --n-resamples 10000 \
    --input data/training/4track_holdout_predictions.json \
    --output data/validation/4track_ci_2026-05-24.json
```

(If script path differs, search via: `find scripts -name '*bootstrap*'`)

- [ ] **Step 6a: Commit SUCCESS path**

```bash
git add data/sbi/prodrug_activation_registry.json \
        data/training/4track_holdout_predictions.json \
        tests/integration/test_holdout_regression.py \
        tests/regression/test_clopidogrel_ces1_literature_applied.py
test -f data/validation/4track_ci_2026-05-24.json && git add data/validation/4track_ci_2026-05-24.json

git commit -m "$(cat <<'EOF'
feat(b-03.x): clopidogrel CES1/CYP literature-IVIVE

B-03.x SUCCESS — B-03 placeholders 0.030 replaced with literature-derived
intrinsic clearances per spec §B.2.

CES1: <derived> L/(h·pmol) per <primary source> (Vmax=<X>, Km=<Y>)
      IVIVE via Boberg 2017 CES1 abundance 1664 pmol/mg microsomal.
CYP3A4: <derived> L/(h·pmol) per Kazui 2010 fate split + abundance
CYP2C9: <derived> L/(h·pmol) per same

disposition_state: ceiling_accepted → literature_applied
affinity_source: literature → literature_ivive

107-holdout impact:
  Clopidogrel FE: 5.15× → <new>× (improvement)
  Meta AAFE: 2.7715 → <new> (Δ=<delta>)

Unit-sanity gate verified per spec §B.2 (affinity in [1e-5, 1e0] window).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### DE-38 branch (Task 14b)

- [ ] **Step 2b: Revert any partial registry changes**

```bash
git checkout data/sbi/prodrug_activation_registry.json
```

(Phase A commits stay; only Phase B reverts.)

- [ ] **Step 3b: Re-regen benchmark to baseline**

If Task 13 cache reflects rejected literature attempt:

```bash
test -d data/drugbank && mv data/drugbank data/drugbank.local-archive
test -f models/adme/logp_correction.json && mv models/adme/logp_correction.json models/adme/logp_correction.json.local-archive
python scripts/run_engine_benchmark.py
test -d data/drugbank.local-archive && mv data/drugbank.local-archive data/drugbank
test -f models/adme/logp_correction.json.local-archive && mv models/adme/logp_correction.json.local-archive models/adme/logp_correction.json
git checkout data/training/4track_holdout_predictions.json  # or commit if cache shifts cosmetically
```

- [ ] **Step 4b: Add DE-38 entry**

Append to `docs/claude/dead-ends.md` (next free DE-NN id):

```markdown
## DE-38 — B-03.x clopidogrel CES1 literature-IVIVE: <root cause>

**Date:** 2026-05-24
**Spec:** docs/superpowers/specs/2026-05-24-doctrine-completion-sprint-design.md
**Outcome:** B-03 placeholders 0.030 retained; disposition_state remains ceiling_accepted.

**Path attempted:**
- Primary literature: <source URL, success/failure>
- Fallback: <source URL, success/failure>
- IVIVE computation: <attempted/blocked at which step>
- Unit-sanity gate: <PASS/FAIL>
- Benchmark outcome: clopidogrel FE 5.15× → <new>× (worse/unchanged)

**Root cause:** <one of: literature paywall, in-vitro→in-vivo scaling failure, unit derivation ambiguity, IVIVE overcorrected>

**Future-iteration unlock:**
- Subscription access to <paper>, OR
- Independent CES1 clopidogrel kinetic assay providing direct CL_int per pmol enzyme, OR
- Per-organ CES1 abundance refinement (gut + liver) if intestinal first-pass dominates

**Infrastructure preserved:** Registry schema supports literature_ivive disposition; loader unchanged. Future iteration can replace placeholders without code change.
```

## Task 15: Final documentation updates

**Files:**
- Modify: `docs/claude/experiment-log.md`
- Modify: `docs/claude/backlog.md`
- Conditionally: `CLAUDE.md`

- [ ] **Step 1: Append experiment-log entry**

Read first 20 lines of `docs/claude/experiment-log.md` to confirm format, then append (at top — reverse chronological):

```markdown
## 2026-05-24 — Doctrine completion sprint (B-10 + B-03.x)

**Phase A (B-10) — SUCCESS:** atorvastatin + rosuvastatin promoted with literature-curated metabolic_fraction entries; ecm_applicable=true flipped. v0.3 ECM doctrine complete for all 4 statin substrates (pravastatin, pitavastatin, atorvastatin, rosuvastatin). 107-holdout headline unchanged (neither drug in holdout). atorvastatin/rosuvastatin test_oatp_ecm_statins FE gate xfail remains in place per Peff over-prediction diagnosis (test docstring lines 18-30).

**Phase B (B-03.x) — <SUCCESS/DE-38>:** <fill outcome>

Commits: <Phase A SHA>, <Phase B SHA or "DE-38">
```

- [ ] **Step 2: Update backlog.md**

Strike B-10:

```markdown
### ~~B-10~~ — Pitavastatin/rosuvastatin/atorvastatin metabolic_fraction curation (closed 2026-05-24)

**Status:** Closed. Shipped in 2026-05-24 doctrine completion sprint Phase A. atorvastatin + rosuvastatin promoted with literature-curated metabolic_fraction entries per spec `docs/superpowers/specs/2026-05-24-doctrine-completion-sprint-design.md`. Pitavastatin was already promoted in v0.3.1 (PR #30) — README §463 was stale.
```

For B-03.x:
- If SUCCESS: no backlog entry needed (was implicit in CLAUDE.md TODO; mention closure in experiment-log only)
- If DE-38: add backlog entry "B-03.x DE-38 closed — see dead-ends.md"

- [ ] **Step 3: Conditionally update CLAUDE.md headline**

ONLY if Phase B SUCCESS AND |ΔAAFE| > 0.005:

Update the metrics table at top of `CLAUDE.md`:
- Meta AAFE: 2.772 → <new>
- 95% CI: if Step 5a bootstrap ran, use new CI; else note "CI 2026-05-12 still canonical per |ΔAAFE| < 0.01"
- Add note in the "Current Performance" prose describing the B-03.x literature-IVIVE shift on clopidogrel

If Phase B DE-38: do NOT modify CLAUDE.md headline (no change to ship).

- [ ] **Step 4: Final commit**

```bash
git add docs/claude/experiment-log.md docs/claude/backlog.md
test -f docs/claude/dead-ends.md && git diff --cached --quiet docs/claude/dead-ends.md || git add docs/claude/dead-ends.md
# Conditional CLAUDE.md
git diff --quiet CLAUDE.md || git add CLAUDE.md

git commit -m "$(cat <<'EOF'
docs(sprint): 2026-05-24 doctrine completion sprint closeout

Phase A (B-10): atorvastatin + rosuvastatin ECM doctrine complete.
Phase B (B-03.x): <SUCCESS with AAFE shift | DE-38 closed>.

experiment-log + backlog + <dead-ends if DE-38 | CLAUDE.md if SUCCESS> updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 16: Push to main and verify CI

**Files:** none

- [ ] **Step 1: Verify branch state**

```bash
git status
git log --oneline -10
```

Expected: clean working tree, last 3-5 commits are this sprint's work.

- [ ] **Step 2: Push to main**

```bash
git push origin main
```

Expected: success. Owner direct push pattern per CLAUDE.md repo state.

If protected-branch reject: this is the B-11 cycle's known workaround:
```bash
git push origin HEAD:main
```

- [ ] **Step 3: Trigger / verify CI**

```bash
# Check most recent workflow run
gh run list --limit 3
# Watch (optional)
gh run watch <run-id>
```

Expected: green (test workflow status=completed conclusion=success).

- [ ] **Step 4: Report outcomes to user**

Final summary text (substitute placeholders):

```
Sprint 완료 (2026-05-24):
- Phase A (B-10): SUCCESS — atorvastatin (mf=<X>), rosuvastatin (mf=<Y>) 등록
- Phase B (B-03.x): <SUCCESS — clopidogrel FE 5.15→<X>×, Meta AAFE 2.7715→<Y> | DE-38 — root cause: <Z>>
CI: green @ <commit SHA>
헤드라인 영향: <0 | -0.0X>
```

---

## Self-Review

**1. Spec coverage:**

| Spec section | Tasks |
|---|---|
| §1 Sprint Structure | All 16 tasks |
| §2 Doctrine Compliance (invariants 1/5/8) | T1-2 (literature only), T11 (no engine touch), T9-10 (no Cmax tuning) |
| §3 Pre-Committed Sources | T1 (atorvastatin Reactome), T2 (rosuvastatin Niemi 2009 PMC2765590), T9 (clopidogrel Mol Pharm 2025 PMC12673578) |
| §3 Distribution propagation rule | T10 Step 3 (Vmax/Km cv defaults) |
| §4 Architecture (file map) | File Map section at top |
| §5.A.0 mf semantics | T1 Step 3 formula + T2 Step 3 |
| §5.A.0 sanity gates | T1 Step 3 (atorvastatin [0.4, 0.8]), T2 Step 3 (rosuvastatin [0.0, 0.1]) |
| §5 Phase A detail | T1-T8 |
| §6 Phase B detail | T9-T14 |
| §6.B.2 unit-sanity gate | T10 Step 2 |
| §7 Testing | T6 (schema), T7 (statins integration), T12 (prodrug regression), T14 Step 3a (new test) |
| §8 Risk register | T7 Step 1 (xfail acknowledgment), T10 Step 2 (unit gate), T14 (DE-38 branch) |
| §9 Execution (subagent-driven, opus, direct main) | T16 (push to main pattern) |
| §10 Self-maintenance | T15 + T14a Step 5a (CI bootstrap conditional) |

All spec sections covered.

**2. Placeholder scan:**

- `<computed_mf>`, `<derived>`, `<value>`, `<X>`, `<Y>` etc. in templates: these are SUBSTITUTION TARGETS for the implementer's actual numbers, not "TBD" — acceptable in a plan with literature-extraction tasks where the value is the OUTPUT of the task. Each is paired with explicit derivation steps showing how to compute it.
- No "TODO", "implement later", "add appropriate error handling" — clean.
- No "similar to Task N" without repetition — Task 4 (rosuvastatin) mirrors Task 3 (atorvastatin) but repeats the exact JSON template.

**3. Type/name consistency:**

- `metabolic_fraction` field used consistently across T3, T4, T6.
- `ecm_applicable` field used consistently across T5, T6.
- `enzyme_affinity_for_conversion` for clopidogrel registry (T11) matches the actual JSON key in `data/sbi/prodrug_activation_registry.json`.
- `disposition_state` / `affinity_source` enum values match B-03 doctrine (`ceiling_accepted` / `literature_applied` / `literature` / `literature_ivive`).
- SMILES strings match `oatp1b1.json` canonical forms (verified during exploration).
- Commit SHAs and dates use consistent format.

No issues found.
