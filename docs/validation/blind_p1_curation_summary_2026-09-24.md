# Blind Cmax pilot P1 — label-free curation summary (2026-09-24)

## Status and scope

This is an **AI-assisted diagnostic pilot**, produced by a single automated agent.
It is **not** the two-human independently curated External Holdout V1 final cohort
defined in `docs/validation/external_holdout_v1_protocol.md`. No curator registry,
no sealed label envelope, no external timestamp, and no second verifier exist for
these records. Every arm below was verified once, by one agent, against the
original open-access clinical report. Treat the cohort as a development-grade
diagnostic instrument only; using it as an independent external test would
overstate independence.

This file contains **no drug names and no observed outcome values**. Identities,
Cmax values, statistics, subject counts and source citations live only in
`labels.json`.

## Eligibility criteria applied

Each included arm had to satisfy, from the original report's own text:

1. Healthy adult participants (no patient cohorts, no organ impairment, no paediatric).
2. Single oral dose (no steady-state or multiple-dose arm).
3. Fasted state with an explicitly stated post-dose food-free interval of **>= 4 h**
   (protocol/ICH M13A rule; an unqualified word "fasted" was not accepted).
4. Immediate-release conventional oral dosage form, or an oral solution
   (no extended/modified/delayed release).
5. Parent drug measured in plasma (no metabolite-only or total-radioactivity labels).
6. Unboosted monotherapy arm (no ritonavir/cobicistat, no combination product).
7. Dose expressed in parent active-moiety mass, with per-arm evidence recorded.
8. No name or structure collision with the repository exclusion union.

## Search and retrieval

Discovery ran over two axes, both label-blind with respect to the model:

* **Identity axis.** A hand-assembled inventory of 2020–2026 oral small-molecule
  approvals plus late-phase investigational compounds, enumerated before any
  source was opened. 170 distinct identity strings were screened, plus 8
  synonym strings for candidates that survived.
* **Literature axis.** Four bounded queries against Europe PMC / PMC E-utilities
  for first-in-human, single-ascending-dose, single-dose and phase-1 reports in
  healthy participants with oral administration (2015–2026, open access with full
  text), plus per-name title searches for the exclusion-clean identity pool.
  653 full-text XML documents were retrieved locally.

Full texts were triaged automatically for an explicit post-dose fasting phrase
(a >= 4 h food-free interval tied to dosing). 75 documents passed that filter
across the four retrieval passes; the remainder were dropped without arm review
because no source-stated post-dose food-free interval could be found.

## Exclusion screening

Name and structure collisions were computed with the repository's own code, not a
re-implementation: `scripts/audit_external_holdout_manifest.py::_repository_exclusions`
(which loads `scripts/build_n50_exclusion.py` for hard-corpus InChIKey-14
ingestion), against the working tree at the session's HEAD. That union covers
the fitted training corpora, clinical reference and registry JSON under
`data/reference`, `data/validation`, `data/enzymes`, `data/transporters`,
`data/sbi`, `data/training`, and the curated compound YAML — including
`data/validation/ctgov_modeler_seen_identities_2026-09-24.json` and
`data/validation/frdb_modeler_seen_identities_2026-09-23.json`. The union held
9,046 InChIKey-14 structures and 5,542 normalized names. Screening was applied to
each candidate's INN, development code(s) and canonical structure
(largest-fragment InChIKey-14), and additionally to a text scan of tracked
`docs/` and `data/` files to catch modeler-seen identities recorded in prose.

Screening result: **100 of 170** identity strings hit the exclusion union and were
dropped; 70 strings (68 distinct compounds, after collapsing two code/INN
duplicates) were provisionally clean. A final re-check of the 8 included
compounds — INN, development codes and canonical SMILES — returned zero hits.

## Attrition

| Stage | Count |
|---|---|
| Identity strings screened | 170 |
| Dropped: name or structure collision with exclusion union | 100 |
| Provisionally clean distinct compounds | 68 |
| Full texts retrieved | 653 |
| Documents passing post-dose fasting-phrase triage | 75 |
| Compounds adjudicated arm-by-arm against the original report | 26 |
| Omitted at adjudication | 18 |
| **Included** | **8 compounds / 8 arms** |

Omission reasons at adjudication (aggregate only):

* 6 — dose stated on a salt/solvate basis, or a salt form exists and the report
  gives no active-moiety conversion. Salt mass alone is not primary-eligible.
* 6 — no source-stated post-dose food-free interval (pre-dose fast stated only).
* 4 — administered dosage form never stated, so immediate release could not be
  adjudicated.
* 2 — no verifiable single-dose fasted parent-plasma arm in the report's own
  tables (multiple-dose design, or fasting state of the tabulated cohorts unstated).

No candidate was replaced after a verification failure, and no criterion was
relaxed to reach a target count.

## Verification performed per included arm

For each of the 8 arms the agent recorded, from the primary report: the exact
table and column quoted; the central-statistic type as the source labels it
(arithmetic mean, geometric mean); subject count for that arm; dose and route;
dosage form with the verbatim basis for treating it as immediate release;
the pre-dose and post-dose fasting statements; the analyte (parent in plasma);
the population description; trial registration identifier where given; and an
explicit dose-basis evidence statement. Structures were resolved from PubChem by
name and cross-checked against ChEMBL where a record existed (all 8 parsed under
RDKit; InChIKey-14 values are mutually distinct).

## Counts and hashes

* `inputs.json` — 8 anonymous records `{candidate_id, arm_id, smiles, dose_mg, route}`
  SHA256 `562d4dfebb9fbee96b9219cb35505527f032b7e2a74daaa14fa002614eaba1fe`
* `labels.json` — 8 label records, same IDs
  SHA256 `942fcc598b8749a5869a52d607d272e9be4b65cee7e89a74c70b1295bf9c6936`

Candidate IDs `cand_01`–`cand_08`; arm IDs `arm_01a`–`arm_08a`. The ID sets in the
two files are identical (asserted at build time).

## Unresolved issues and limitations

1. **Single-agent verification.** No independent second curator; no custodian
   separation. The V1 protocol's two-person requirement is unmet by construction.
2. **Immediate release is inferred, not stated.** Seven of eight arms use a
   conventional capsule, tablet or aqueous suspension with no modified-release
   language anywhere in the report; one uses an oral solution, where the question
   does not arise. No report explicitly labels its formulation "immediate
   release". Each arm's `release_evidence` field states exactly what the source
   says, so a reviewer can overturn any single inference.
3. **Dose basis rests on absence of a salt.** For all eight compounds the report
   states the dose with no salt designation, and the public structure records
   (PubChem, ChEMBL) list no salt, hydrate or solvate form. That is a negative
   argument, not a sponsor-stated free-base equivalence; a regulatory document
   could still reveal a salt-basis dose. One arm's source additionally states a
   unit strength inconsistent with the administered dose, and that inconsistency
   is recorded verbatim in its label record.
4. **Phase-1 investigational compounds dominate.** These are early-development
   molecules, so the cohort carries no regulatory clinical-pharmacology quota,
   under-represents marketed drugs, and is chemically narrow relative to the
   N=120/N=260 designs.
5. **Discovery was not a consecutive enumeration.** The literature axis depended
   on open-access full text and on a phrase-level fasting filter, both of which
   bias toward reports that happen to state meal timing precisely. Eligible arms
   in paywalled or terser reports were systematically missed, so the attrition
   rates here do not estimate the true eligible-case yield of any source family.
6. **N=8 has no inferential power.** This cohort cannot test Meta-versus-ML
   superiority, nor support any release gate. It is a diagnostic probe.
7. **No exclusion-union hash is frozen here.** The screen ran against the working
   tree at session HEAD rather than a pinned, hash-recorded union, so the
   contamination check is reproducible only against that same checkout.
