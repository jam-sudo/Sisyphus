# Sisyphus P2 label-blind diagnostic acquisition — summary (2026-09-24)

AI-assisted diagnostic acquisition; **not** two-human V1 validation. No Sisyphus predictions were run and no model output was inspected.

## Result

- Eligible distinct compounds: **0** (target 12). Eligible arms: 0. The 2015–2025 window was screened to exhaustion.

## Method

- Source: FDA *Compilation of CDER NME and New Biologic Approvals 1985–2025* XLSX (SHA256 `0dd3e4683901e3b8c41ee6bd0f54bac1c33eaae95f1fae8679e6b3b0b45502cd`, matches the V1 protocol).
- Enumeration: NDA rows, approval year 2015–2025, any route field containing "oral"; ordered by approval year descending, then normalized ingredient name (lowercase, non-alphanumerics removed). Every row was processed.
- Exclusion: the repository union from `scripts/audit_external_holdout_manifest.py` at commit `ddbfdf2` (fitted corpora, runtime registries, prior validation/P0/P1/modeler-seen JSON indices, compound YAMLs; names plus InChIKey-14), with the P1 prediction file skipped unread. Names were matched as full strings, after salt stripping, and per combination component. As an extra modeler-seen screen, any ingredient token of five or more characters mentioned in repository text also excluded the entry. Remaining name-clean parents were resolved through PubChem and IK14-screened, including a known conjugate relative.
- Strict arm criteria (never relaxed): healthy adults; single unboosted oral dose of a formulation explicitly stated to be immediate release; parent plasma Cmax; fasted, with a stated post-dose food-free interval of at least 4 h; a source-supported parent active-moiety dose basis; an exact original FDA clinical-pharmacology review or original publication table/page.

## Ordered attrition (N=248 oral NDA rows)

| Step / status | n |
|---|---|
| excluded_repository_name_hit | 218 |
| excluded_component_name_hit | 17 |
| excluded_modeler_seen | 1 |
| excluded_combination_and_related_structure | 1 |
| excluded_component_hits_or_outside_window | 1 |
| excluded_not_small_molecule_parent | 3 |
| excluded_endogenous | 1 |
| excluded_no_parent_plasma_cmax | 3 |
| excluded_no_quantifiable_parent_cmax | 1 |
| excluded_criteria_not_met | 2 |
| eligible | 0 |

Two structure-clean parents reached arm-level source review. Both failed because the examined sources did not state a post-dose food-free interval for the single-agent fasted arms. A later source check found one explicitly documented oral-solution arm, but no qualifying meal-timing evidence for it.

## Follow-up source check

A second AI-assisted pass re-examined exactly these two entries against the original [NDA 209195 clinical-pharmacology review](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2017/209195Orig1s000ClinPharmR.pdf), [NDA 206038 clinical-pharmacology review](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2015/206038Orig1s000ClinPharmR.pdf), matching EMA reports, ClinicalTrials.gov records, and the original lumacaftor PK publication. It found **0 additional eligible arms**. The searched public sources did not establish a ≥4 h post-dose food-free interval for either single-agent arm; searches by the cited study identifiers found no matching ClinicalTrials.gov registration. This is an absence of qualifying public evidence, not proof that the studies lacked a post-dose fast.

The pass corrected two details in the raw screening log without changing either exclusion: the voxilaprevir first-in-human study is `GS-US-338-1120` (not `GS-US-338-1118`), and its Part B Cohort 9 Day 8 arm used an explicit **100 mL oral solution** (review p. 69), satisfying the formulation limb. The 4 h post-dose fast on pp. 117–118 belongs to a fixed-dose-combination arm and cannot be transferred to that single-agent solution arm. For lumacaftor, the cited single-agent arm used an aqueous **suspension**, not a solution (review p. 117), and neither it nor the alternative tablet publication states the required post-dose interval.

The follow-up is still single-curator AI evidence, not External Holdout V1. Its full source-location report and empty manifests remain under `/tmp/sisyphus_p2_followup_20260924/`; SHA256: `report.md` `5b2e702c9e25bdb15f31a91a51e7514452f1a074078b2b249cf37dd9e47a4e3e`, `inputs.json` `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`, `labels.json` `f98e0e513381cd8ed5bfc02ae018718b88eac424c70246a142a030107812996d`. No prediction was run. Both supervised worker terminals were released after completion.

## Source-window coverage

| Approval year | rows screened |
|---|---|
| 2025 | 25 |
| 2024 | 22 |
| 2023 | 24 |
| 2022 | 13 |
| 2021 | 24 |
| 2020 | 24 |
| 2019 | 26 |
| 2018 | 33 |
| 2017 | 24 |
| 2016 | 7 |
| 2015 | 26 |

## File hashes (SHA256)

- inputs.json: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- labels.json: `82a993cb10927fc80cbab13cd5ebb41035c568654f01efaa8d89d4c50b11558d`
- screening_log.json: `f14211b4d6f5b7f54bcc5de18d0479f9b56e9edf8242d34abd7748782f2c51ab`

## Limitations

- 235 of 248 rows (95%) were removed by repository name collision alone, so this FDA window is effectively exhausted for this repository.
- Exact-name matching can over-exclude through shared tokens and under-exclude aliases. Structure-level IK14 screening was applied only to the name-clean residue.
- Seven class-based exclusions (non-absorbed, inorganic, peptide, endogenous, and prodrug/metabolite-only products) were not checked against original documents. None could plausibly give an eligible parent small-molecule plasma Cmax arm.
- Arm-level review used FDA review text and PubMed full text. Image-only tables were not OCR-read. ClinicalTrials.gov protocols were not used to fill missing post-dose meal timing, because they are outside the permitted source types.
- This was one AI curator, with no independent second verification.

## Coordinator check

After the worker reported completion, the coordinator independently recomputed
all four file hashes, counted 248 screening entries and their status totals,
confirmed descending approval years, and found empty input and label-arm lists.
The raw screening log remains at `/tmp/sisyphus_p2_20260924/screening_log.json`
with the hash above; it is not a durable external custody record. The completed
worker terminal was released and its transcript archived by Orca.
