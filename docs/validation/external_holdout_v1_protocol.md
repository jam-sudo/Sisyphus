# External Holdout V1 Protocol

## Decision objective

Measure the frozen system's independent structure-only Cmax accuracy and decide
whether the PBPK/meta stack adds enough value over the direct ML track to justify
its production complexity. N=107 is development data for this purpose; N=28 and
the AI-assisted blind P1 N=8 diagnostic are already consumed; the 2026Q2 N50
is invalidated. P1 identities and structures are indexed in
`data/validation/blind_p1_labels_2026-09-24.json` and
`data/validation/blind_p1_inputs_2026-09-24.json` for final-cohort exclusion.

## Frozen primary estimand

The primary cohort contains healthy adults in single-dose, oral immediate-release, fasted,
unboosted studies reporting parent-drug plasma Cmax. Inputs available to the model
are canonical parent SMILES and oral dose in mg; route is fixed to oral. Extended/modified release,
fed studies, combination/boosted regimens, active-metabolite observations, and
multiple-dose steady-state records are excluded from the primary estimand and may
appear only in predeclared challenge strata.

The statistical unit is the compound. If a compound has multiple eligible arms,
its absolute log errors are averaged before averaging across compounds, so a drug
with many studies cannot dominate the result.

For compound `i`, arm `j`:

```
e_ij = abs(ln(predicted_cmax_ij / observed_cmax_ij))
e_i  = mean_j(e_ij)
AAFE = exp(mean_i(e_i))
```

Confidence intervals use a compound-cluster bootstrap with 100,000 resamples and
a seed fixed in `manifest.freeze.random_seed`; the scorer has no seed or resample
override. Report observation-uncertainty propagation separately when the
source supplies Cmax CV/SD; do not mix it into the sampling CI.

## Size and power

The observed paired Meta-versus-ML absolute-log-error SD is 0.24–0.26 log10 units.
For the CI-only superiority test at two-sided alpha 0.05 and 80% power,
this implies approximately:

- N=90–110 compounds to detect a 15% AAFE improvement.
- N=210–260 compounds to detect a 10% AAFE improvement.

The recommended target is **N=260 compounds**. A resource-limited design may fix
N=120 before curation begins, but it cannot interpret a nonsignificant 5–10%
difference as equivalence.
The release gate also requires the point estimate to pass a stricter margin
(`R <= 0.90` at N=260 or `R <= 0.85` at N=120). At a true effect exactly on
either margin, the approximate probability of passing that point-estimate gate
is only 50%, regardless of N. With paired SD 0.24–0.26 log10, a normal
approximation gives about 80% probability of passing the *combined* point and
CI gate at true `R ≈ 0.873` for N=260 or `R ≈ 0.814` for N=120. The size
figures above must not be presented as 80% power for the full release gate.

## Source and sampling strategy

1. Use an independent curation team or data custodian. The modeling team must not
   see observed Cmax until code, models, registries, dependencies, and the cohort
   manifest are frozen.
2. Define source windows and enumerate all eligible records consecutively. Do not
   select candidates based on prediction availability or expected difficulty.
3. Prefer primary PK papers, regulatory clinical-pharmacology reviews, and label
   tables. Each value requires two-person verification against the cited table.
4. Record parent/analyte, salt, formulation, IR/ER/MR, fed/fasted, dose form,
   route, single/multiple dose, population, co-medication, matrix, units, study N,
   Cmax central-statistic type (`arithmetic_mean`, `geometric_mean`,
   `geometric_lsmean`, or `median`), and source table/page. The frozen `dose_mg`
   must be in parent active-moiety
   units matching the model input. Record `dose_basis` and a source-specific
   `dose_basis_evidence` statement. For a salt, cocrystal, or solvate, an
   explicit original or regulatory equivalence may support conversion to
   parent active-moiety mg; record the reported mass and conversion in that
   evidence. Salt/solvate mass alone or an unknown basis is not primary-eligible.
   Both curators verify this mapping before the prediction manifest is frozen.
   Record the first permitted post-dose meal time as `postdose_fast_h`; primary
   fasted arms require at least four food-free hours after dosing. Unknown
   timing stays non-primary, even if a source calls the pre-dose state fasted.
   This threshold follows [ICH M13A's fasting-study meal standard](https://www.ema.europa.eu/system/files/documents/scientific-guideline/ich-m13a-guideline-bioequivalence-immediate-release-solid-oral-dosage-forms_step-5_en.pdf)
   and the diagnostic P0 adjudication rule.
5. Store labels encrypted or in a separate access-controlled repository. The
   custodian retains both identities and outcomes until the container is frozen.
   The evaluator receives the frozen container plus a label-free arm manifest
   (SMILES, dose, route, arm ID); the modeling team receives identities only after
   the one-time run is committed. Before freezing the manifest, the custodian
   computes `label_content_sha256(labels)` from the complete label records and
   cycle ID, and places that digest in the source plan. The helper excludes
   `manifest_sha256` and `predictions_sha256`, which are not yet known. The manifest binds the source-plan
   hash, and scoring checks the revealed labels against the earlier digest.
   This commits the numeric Cmax outcomes before prediction without exposing them.
   After the one-time prediction, the custodian records its file SHA256 in the
   sealed label envelope as `predictions_sha256` and archives the envelope's
   SHA256 in an independent timestamped record before revealing any Cmax values.
   Scoring checks the custodian's prediction digest. The external timestamp is
   an operational custody requirement; local hashes alone cannot prove chronology.

The label-free, blinded-label, and frozen-prediction contracts are pinned in
`data/reference/external_holdout_v1_manifest.schema.json` and
`data/reference/external_holdout_v1_labels.schema.json`, and
`data/reference/external_holdout_v1_predictions.schema.json`. Every executable
stage validates these schemas. Primary eligibility is recomputed from the label
metadata at scoring; the manifest boolean is never trusted by itself.
The manifest candidate name must exactly match the hashed verified shortlist,
so changing it to an alias cannot bypass name-only prior-use exclusions.
The dose-basis and post-dose meal requirements were added on 2026-09-23 after
diagnostic P0 exposed a possible verlukast-sodium ambiguity and fasted-study
meal-timing ambiguity. No V1 cohort or predictions had been frozen; P0 remains
development evidence and is not a V1 test.
The source Cmax statistic is also required and bound into the blinded
source-record hash. The primary comparison accepts all four labelled central
estimates. As a descriptive sensitivity, report compound-level Meta and ML AAFE,
their paired ratio and 95% compound-bootstrap CI for each statistic type with
at least 20 compounds whose every primary arm has that type. Report counts only
below 20; count mixed-statistic compounds separately. This sensitivity never
changes the primary cohort or release gate.

### Operational acquisition plan

N=260 cannot be obtained from the current repository: its clean internal candidate
pool is zero after structure-level exclusion. The set therefore requires a fresh
external acquisition campaign. Enumerate approximately 900 candidates before any
prediction is run, targeting at least 550 verified compounds after attrition.
An acquisition feasibility check on the FDA's [1985–2025 NME compilation](https://www.fda.gov/drugs/drug-approvals-and-databases/compilation-cder-new-molecular-entity-nme-drug-and-new-biologic-approvals)
(retrieved 2026-09-23) found 682 NDA rows with any route field marked oral, of
which 132 were approved in 2020–2025. These are raw product counts, not unique,
eligible, decontaminated compounds; they are an upper bound for that FDA source.
In a label-blind exact-name screen of the same XLSX (SHA256
`0dd3e4683901e3b8c41ee6bd0f54bac1c33eaae95f1fae8679e6b3b0b45502cd`)
against the repository exclusion-name union at `ed0380a`, all 682 rows had
distinct active-ingredient/moiety strings after lowercasing and removing
non-alphanumeric characters. Of these, 473 matched an existing name and 209
did not: 12 of 132 approved in 2020–2025 and 197 of 550 approved earlier.
The route screen includes one mixed injection/oral/rectal entry, and 36
ingredient strings contain a comma or `and`, so some rows may represent
combinations. The 209 nonmatches are only an optimistic discovery pool:
synonyms, salts, combinations, structure-level collisions, non-IR products,
and missing eligible fasted single-dose Cmax arms still need review. FDA's
recent-approval subset is especially small; even the full 1985–2025 source
cannot establish the N=260 test cohort from this name screen alone. No Cmax
values were read.
A later [P2 source-window screen](blind_p2_fda_acquisition_2026-09-24.md)
consecutively reviewed all 248 oral NDA rows approved in 2015–2025 and found
zero arms meeting the strict primary criteria. Its two source-reviewed parent
identities are now development-only and pinned in
`data/validation/p2_modeler_seen_identities_2026-09-24.json`; the exclusion
audit checks their names, study-code aliases, and parent structures. P2 was
AI-assisted and does not satisfy the independent-curation requirement.
The 900-identity inventory therefore needs older and investigational drugs plus
non-FDA sources, with duplicates across agencies collapsed before allocation.
EMA's [Article 57 product data](https://www.ema.europa.eu/en/human-regulatory-overview/post-authorisation/data-medicines-iso-idmp-standards-post-authorisation/public-data-article-57-database)
(Rev. 88, snapshot dated 2026-08-05; XLSX SHA256
`57f71f38b20f87693b36a8c5e397883746dcde4d4be8f6761945972b30692102`)
contains 163,791 product rows. Of these, 97,719 have a route containing `Oral
Use`, representing 6,696 distinct active-substance *strings* after lowercasing
and removal of non-alphanumeric characters. This is a broad identity-discovery
reservoir, not 6,696 eligible compounds: products repeat across countries,
the substance field can group multiple products, and the file does not establish
SMILES, IR formulation, fasted single-dose Cmax, or study provenance. Use it to
enumerate candidates, then verify each against original clinical sources.

EMA's [medicine-page JSON](https://www.ema.europa.eu/en/about-us/about-website/download-website-data-json-data-format)
(snapshot 2026-09-23 06:01:57 UTC; SHA256
`402ba8031d383bc1959d4933a845a6055fa906406ae234162353aae756e87d33`)
has 319 human, authorised, non-generic, non-biosimilar, non-advanced-therapy
product records with a `marketing_authorisation_date` in 2020–2025, covering
310 nonempty active-substance strings. These records also include nonoral drugs
and biologics. Use `marketing_authorisation_date` for approval-window screening;
`european_commission_decision_date` can reflect a later procedure. Neither EMA
table establishes the required Cmax arm. The 900-name inventory is therefore
plausible from public catalogues, while the ≥550 source-verified, eligible
compounds remain an unproven acquisition milestone.

The NIH NCATS [Inxight Drugs FRDB download](https://drugs.ncats.io/downloads-public)
(v. 2024-12-30 ZIP SHA256
`647b80d9cdac4a0517ce649570517dc1aff40545bdc84617ec9f1a56405f9c7a`)
offers a more specific Cmax discovery pool: 13,455 human PK rows across 4,051
drug records. A label-blind screen of its metadata found 1,965 rows for 823
compound IDs with adult, healthy, fasted, single-dose oral administration,
nonempty dose and Cmax fields, and directly convertible mass units. Excluding
combination applications and non-plasma analytes leaves 1,748 rows for 728
compound IDs. Among those, 190 IDs (187 distinct analyte InChIKey-14 keys)
have one parseable analyte structure per compound and no hit in the repository's
fitted-corpus or previously used reference structure/name union. Only 29 of
these 190 IDs have a row linked to an FDA, EMA, PMDA, or MHRA regulatory domain.
These are **provisional discovery counts**, not verified eligible cases: the
table has no clinical source date or reliable IR/parent-drug adjudication, and
each original report, formulation, source agency, structure, and collision must
still be checked by two curators. FRDB alone cannot provide the N=260 final
cohort or the ≥70% regulatory-source quota. The screen neither read nor used
observed Cmax values for model assessment.

A stricter metadata-only re-screen on 2026-09-23 required an exact adult,
healthy, fasted, single-dose, oral, plasma record with nonempty dose/Cmax
fields and excluded currently indexed repository names and structures. It left
74 rows for 28 FRDB compound IDs; only imidapril had a regulatory-domain
source URL, and several other rows still name metabolites rather than the
administered parent. The model-development agent saw these 28 identities during
triage, so they are **development-only** and excluded from the final-test and
reserve pools by the 28 names and 40 analyte/administered structures in
`data/validation/frdb_modeler_seen_identities_2026-09-23.json`.
No numeric Cmax outcomes were inspected. This narrower count does not revise
the earlier 190-ID optimistic discovery count because the screens impose
different metadata conditions and repository snapshots.

A second [EPA publication data extract](https://github.com/USEPA/clinical-nonclinical-concordance)
(`cmax_auc_dataextraction.xlsx`, SHA256
`2600cc6014911097d53c086341279f14b27c267b56b4e6bf8edb36a644ddb950`)
has 405 human Cmax/dose rows for 122 named drugs, mostly sourced from FDA
documents. Only 51 names avoid an exact normalized-name hit in the repository
union, and only nine of those explicitly say both healthy and fasted in the
status field. Route, release type, parent analyte, and structural identity
require original-source review; this is a small supplementary discovery pool,
not a shortcut to an eligible cohort. The 2026 open [PK-DataBase](https://github.com/ClickFF/PK-DataBase)
contains human PK descriptors but no Cmax field in its principal tables, so
it cannot supply outcome labels for this protocol.

The [PKRxiv source paper](https://doi.org/10.1002/cpt.70206) reports 11
available study datasets as of September 2025, covering only five distinct
drugs (efavirenz, nitazoxanide/tizoxanide, dolutegravir, nevirapine, and
rilpivirine). Its Table 2 predominantly describes pregnant or postpartum
populations; the one adult-only study concerns nitazoxanide/tizoxanide.
Dataset access also requires registration and approval of a request. These
metadata alone cannot establish an eligible parent-plasma, healthy-adult,
fasted single-dose arm, and the five-drug ceiling rules out PKRxiv as a
standalone N=120 or N=260 source. No concentration outcomes were inspected.

The separate [PK-DB](https://pk-db.com/api/v1/swagger/) REST service was screened
without reading outcome values on 2026-09-23. Its statistics endpoint reported
819 studies and 138,411 outputs (snapshot SHA256
`f84432ad22afac3c9eab398224bc42344384fa05b4d4970507b50740d925483e`),
while the public studies endpoint returned 803 records, only 88 marked with an
`open` licence (studies snapshot SHA256
`f1a308a686d1acd39a73bcc3f47dada70b4bfa45a221b5eec3a9a4d867c5b979`).
The substance-statistics endpoint returned 800 names; 136 had both a nonzero
intervention count and a nonzero output count. Only 37 of those 136 avoided an
exact normalized-name hit in the repository exclusion union produced by
`scripts/audit_external_holdout_manifest.py` (substance snapshot SHA256
`abb6310c681a5574d9bb80a72168aefa2d96eb36c817b5d291da731421337842`).
Normalization lowercased names and removed non-alphanumeric characters. These
37 are an optimistic **discovery count**, not verified novel compounds: aliases,
metabolites, isotopic tracers, nonoral routes, and structure-level collisions
remain unresolved, and aggregate substance counts do not establish a matching
Cmax arm. Moreover, `GET /outputs/?page_size=1` and
`GET /pkdata/timecourses/?page_size=1` both returned HTTP 200 with zero rows on
that date, despite the nonzero statistics; see the related
[PK-DB API issue](https://github.com/matthiaskoenig/pkdb/issues/758).
Original-source licensing and Cmax values would need separate verification.
PK-DB therefore cannot currently supply the primary N=120 or N=260 cohort by
itself.

The [Open Systems Pharmacology observed-data workbook](https://github.com/Open-Systems-Pharmacology/Database-for-observed-data/blob/master/ObsDataPK_OSP.xlsx)
(retrieved 2026-09-23; XLSX SHA256
`e77e3c99c059f97ea45785508f7b0cbb52382306b8257737c9cd1ac83d83e86b`)
was also screened using only Cmax-field presence and study metadata, without
inspecting numeric outcomes. Its `PK-Parameter` sheet has 895 rows with a Cmax
field across 34 distinct normalized analyte names. Joining to `Studies` by ID,
then requiring human, oral, plasma, fasted, a single administration at time zero,
and a recorded post-dose fast of at least four hours leaves 122 rows across 19
names. Only eight rows across **two** names avoid an exact normalized-name hit
in the repository exclusion union. This case-insensitive screen accepts `PO`
and `po` as oral and `Fasted` and `fasted` as fasted. The two nonmatches are
unverified leads, not eligible compounds: structure, parent analyte, dose
basis, formulation, and original-source evidence remain unchecked. Even the
34-name pre-exclusion ceiling is below N=120, so this workbook can only be a
supplementary discovery source.

The separate [Geci et al. high-throughput PBK study](https://link.springer.com/article/10.1007/s00204-024-03764-9)
reports 2,235 human concentration-time profiles for 210 compounds. Its
[profile-source supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs00204-024-03764-9/MediaObjects/204_2024_3764_MOESM3_ESM.xlsx)
(SHA256 `1981f68dfbb8cd38687fe6d0aa8f94a31589b174b0f3b3c06abbd5c690745788`)
lists 1,486 rows marked oral (`PO`) across 192 normalized compound names.
Twenty-two of these names avoid an exact repository exclusion-name hit. The
[compound-structure supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs00204-024-03764-9/MediaObjects/204_2024_3764_MOESM2_ESM.xlsx)
(SHA256 `09ce7f809c18ff2caaa9d27ea3bf4a9372c0677ad22c8ae05c555bb8efcd06b6`)
has 209 compound rows; 176 oral names join exactly after normalization. Only
**nine** joined oral compounds avoid both a name and InChIKey-14 structure hit
in the repository exclusion union. These are provisional discovery leads,
not verified primary cases. The supplements are an identity/source index and
compound-property table, not a ready-to-score dose/Cmax table; original
studies, eligibility, and numeric outcomes would need separate curation. No
numeric Cmax outcomes were inspected in this screen. These aggregate counts
were obtained from the supplements retrieved 2026-09-23 and cannot establish
the N=120 cohort.

The independently maintained [e-Drug3D collection](https://chemoinfo.ipmc.cnrs.fr/edrug3d.html)
has a downloadable [PK table](https://chemoinfo.ipmc.cnrs.fr/DOWNLOAD/MOLDB/e-Drug3D_2197_PK.txt)
(SHA256 `39a89a1e49f91397773275699757933f0470563f6d137301e8397b82e98e766c`)
and [structure SDF](https://chemoinfo.ipmc.cnrs.fr/DOWNLOAD/MOLDB/e-Drug3D_2197.sdf)
(SHA256 `ca26479ed6bf2dfd5fd7ffb5cfcdccdb76f429acaee5bff8641c513134f7a264`).
A 2026-09-23 metadata screen found 990 rows with a finite numeric Cmax field,
683 of them marked oral, 632 also not marked as metabolites, and 36 of those
without an exact normalized-name hit in the repository exclusions. Joining
the PK ID to SDF structure and excluding InChIKey-14 collisions leaves **21
distinct provisional structures**. The table does not provide the dose,
fasting state, formulation, sampling matrix, or a Cmax-specific source link;
its route field may list several routes for one compound. Thus these 21 are
not verified oral Cmax arms, and the collection cannot by itself supply the
N=120 cohort. A web preview exposed numeric Cmax values for some records
among PK IDs 1–61 to the model-development agent before any freeze. Exclude
all 1–61 from any future blinded cohort drawn from this source.

The repository's DrugBank `pk_data.csv` is narrative PK text, not the separately
licensed structured Cmax table. A value-blind screen of its absorption records
(`pk_data.csv` SHA256 `8a0c7d11da7bd1e91cddc4fb2bb6291f584b5481f0d3949ff22949f71ade9d5f`;
`drugs.csv` SHA256 `70852f7db2d7a50935cc027681402322da6a9a40fff8fe7ca0459c1268670882`)
found 292 records mentioning `Cmax`, 175 also mentioning a numeric mass dose,
40 also mentioning fasting, and only two with a parseable structure and no
structure/name collision in the repository exclusion union at `bc9ed5f`.
The text export has no per-record original-source URL. Missing narrative detail
does not prove ineligibility, so these are triage counts, not verified cases or
an upper bound on DrugBank's separately available structured records. This local
export cannot supply the external primary cohort.

Health Canada's official [Summary Reports API](https://health-products.canada.ca/api/documentation/summary-report-documentation-en.html)
(`basisdecision/?lang=en&type=json`, snapshot SHA256
`a796f3fb403c8362514e2d2ffb107f73967a9df3437891137deae8e33eb7db7b`)
returns 619 drug SBD records, including 129 with an authorization date in
2020–2026. Only 25 drug records mention `Cmax` anywhere in their summary text,
and seven of the 2020–2026 records do. The SBD feed is suitable for discovering
drug identities and linking to monographs, but direct Cmax table acquisition
still requires following the primary product documents. A keyword mention is
not an eligible PK observation.

A cross-source identity check on the snapshots above (FDA XLSX SHA256
`0dd3e4683901e3b8c41ee6bd0f54bac1c33eaae95f1fae8679e6b3b0b45502cd`)
found 93 of the 310 EMA 2020–2025 active-substance strings with an exact
normalized-name match to an Article 57 product whose comma-separated route
list contains the `Oral Use` token. The FDA oral-NDA set and those EMA names
yield 711 distinct strings after lowercasing and removing non-alphanumeric
characters. Adding Health Canada's 124 distinct 2020–2026 medicinal-ingredient
strings yields 808. This is a mechanical discovery count, not an eligible or
decontaminated cohort: exact names miss salt/synonym matches, EMA rows can be
biologics, and the Canada subset was not route-filtered. These three bounded
subsets alone do not establish the 900-name inventory; older Article 57 products,
investigational compounds, or other agency sources must be enumerated under a
predeclared source window. No observed Cmax was read for this count.

The official [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-about-studies/learn-about-api)
provides another discovery route. A metadata-only pull on 2026-09-23 for
`AREA[OutcomeMeasureTitle]Cmax` returned 5,868 result-posted studies. Among
them, 4,799 had a Cmax-titled result with a mass-concentration unit, 1,991
also marked healthy volunteers, 1,982 also included adults, and 918 also had
results first posted in 2020–2026. Requiring the arm/intervention text to
mention fasting, single dosing, oral administration, and a mass dose reduced
that cross-field screen to 75 study records. These mentions can refer to
different arms, repeat one compound across studies, or name an investigational
code with no resolved structure. The registry query therefore supplies
source-discovery leads, not 75 verified primary arms, and cannot replace
arm-level two-person adjudication against posted results and original reports.
In a narrower automated triage of those 75 records, 24 had exactly one
non-placebo intervention name. The [PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
name-to-structure lookup resolved 16; 11 of those hit the repository's existing
structure/name exclusion union, leaving five provisional structure-clean
leads. This is not a verified eligibility count, and an unresolved name does
not prove that its structure is novel.

A separate metadata-only cross-check on 2026-09-24 queried result-posted,
healthy-volunteer records with a `Cmax` outcome title and a `fasted` text hit.
The API query was `AREA[OutcomeMeasureTitle](Cmax) AND
AREA[HealthyVolunteers](true) AND AREA[HasResults](true) AND fasted`.
The API's field projection included titles, intervention names/types,
eligibility text, summary text, and results-posting date, but no numeric outcome
fields. Of 379 records, 200 first posted results in 2020–2026; 85 of those had
exactly one non-placebo drug intervention. Requiring oral, single-dose, and
fasting mentions in the protocol text left seven records. PubChem resolved five
names; the production-corpus name/structure screen found one hit and four
provisional nonhits, while two names did not resolve. These seven are only
development source leads: the text mentions may describe different arms,
PubChem's first name match needs identity review, and overlap with the earlier
75-record screen was not established. They must not be counted as additional
eligible or blinded final-test compounds.
Protocol text supports fasted single oral-dose investigation for
`NCT05635461` (CVN424), `NCT04504435` (GSK3494245), and `NCT04208321`
(VT-1598). A [registry-results review](https://clinicaltrials.gov/study/NCT05635461)
on 2026-09-24 confirmed two distinct 150 mg, fasted, single-dose CVN424 arms:
geometric-mean parent-plasma Cmax was 812.9 ng/mL for suspension (N=30) and
231.1 ng/mL for tablet (N=26). The [posted protocol](https://cdn.clinicaltrials.gov/large-docs/61/NCT05635461/Prot_000.pdf)
specifies at least 10 hours of fasting before dosing and four hours afterward;
it does not explicitly establish immediate release or the active-moiety dose
basis. These are two arms of **one compound**, not two independent cohort
members. Because the model-development agent inspected the numeric results,
CVN424 is permanently development-only, regardless of later eligibility review.
The other two named studies still need arm-level source review. The four named
leads and 20 more intervention names displayed in a broader reproducibility
screen are indexed in `data/validation/ctgov_modeler_seen_identities_2026-09-24.json`
for final-test and reserve exclusion. The fourth provisional nonhit,
`NCT05627518` (linaprazan glurate), reports only Cmax *ratios* in its
outcome titles, so it does not establish an absolute Cmax label for this
benchmark.

The PMDA [English review-report index](https://www.pmda.go.jp/english/review-services/reviews/approved-information/drugs/0001.html)
(snapshot SHA256
`11568ed9c0bafb86d94fcc2959a4a9694556cc694a174db6cc10157b1a111e59`)
lists 174 product rows approved in 2020–2025, with 152 distinct non-proprietary
name strings and English report links. A first PubChem lookup resolved 53 of
those strings; 44 hit the repository's exclusion union and nine were
provisionally clean. Another 49 lookups returned HTTP 503, so this is an
**incomplete lower-bound screen**, not a reliable eligibility or attrition
rate. Route, formulation, fasted single-dose parent Cmax, and report dates
still require verification in the individual PMDA reviews. PMDA also notes
that its Japanese originals prevail over the English translations.

Allocate those compounds without outcome-based replacement to three disjoint
roles: 120–150 calibration-development compounds, N=260 final external-test
compounds, and a sealed reserve cohort. Calibration labels may be opened before
the final freeze and immediately become development data; final-test and reserve
labels remain inaccessible. The prediction manifest contains only final-test
compound IDs, including any predeclared challenge arms on those same compounds:

- regulatory clinical-pharmacology packages from FDA, EMA, PMDA, Health Canada,
  and TGA for 2020–2026 novel oral small molecules not already consumed;
- peer-reviewed first-in-human/SAD studies for development compounds with an
  unambiguous structure and directly tabulated Cmax;
- older approved oral drugs absent from fitted target corpora; DrugBank catalog
  membership alone is allowed under the public profile after the fup v2 model's
  public-only retrain, but runtime clinical-registry use is not;
- ideally, an independent sponsor or consortium dataset held by a data custodian,
  which gives the strongest source independence.

Before enumeration, freeze source windows and quotas: at least 70% regulatory
packages, no more than 30% from any one agency, no more than one primary compound
per closely related stereoisomer/salt family, and no retrospective replacement of
a verification failure. A failed candidate remains in the CONSORT-style exclusion
flow with its reason; it is not silently replaced based on prediction quality.

Delivery milestones are: (1) 900-row identity-only inventory; (2) structure/name
contamination audit; (3) at least 550 source-verified compounds; (4) frozen,
structure-stratified calibration/final/reserve allocation; (5) calibration release
and final model freeze; (6) frozen N=260 manifest and encrypted label store;
(7) custodian-run one-time prediction with no modeling-team access to identities;
(8) two-person label unblinding and locked scoring; (9) publication
and retirement. If only N=120 can be funded, that smaller target is fixed at
milestone 1 and the superiority margin remains 15%.

The source plan names four label-free JSON files and their SHA256 hashes:
`inventory` is an array of candidate ID, name, source family/date/reference;
`verified_shortlist` is an array of candidate ID, the same name, and SMILES;
`allocation` maps `calibration`, `final_test`, and `reserve` to disjoint candidate-ID
arrays; `exclusion_flow` gives every inventoried ID a `verified` or `excluded`
decision, with a reason for each exclusion. Source families and dates must fall
inside the frozen windows. The blinded label metadata also records the date and
family of the actual clinical Cmax report or study, separately from any catalogue
snapshot date; scoring rejects a clinical source outside those windows. The audit
checks file hashes, declared counts, full allocation and exclusion coverage,
final-test membership, and InChIKey-14
uniqueness across the verified shortlist. These files contain no observed Cmax.
The shared source-plan gate rejects a registered compound if the runtime
prediction observes its active metabolite, since primary labels require parent
Cmax.
Each label arm's verifier list is included in its outcome-free source-record
hash, and scoring requires every verifier to appear in the source plan's frozen
curator registry.

## Contamination gate

Before model freeze, canonicalize both candidate and corpus structures using:

- largest organic fragment after counterion removal;
- canonical isomeric and non-isomeric SMILES;
- full InChIKey and 14-character connectivity block;
- normalized generic names and synonyms;
- explicit parent/prodrug/active-metabolite relations.

Each verified-shortlist record must include `synonyms` and
`related_structures` (each relation has a type, SMILES, and source citation;
empty arrays require curator review). The audit rejects a declared synonym or
related structure that collides with the frozen exclusion union. This check
cannot discover an omitted relation: both independent curators must search and
verify these fields against the original drug and metabolite records before
the shortlist is frozen.

Reject collisions with every fitted model target corpus, clinical reference used
by runtime registries, previous validation set, manual per-drug override, and
meta-weight/routing cache. The current fup v2 artifact was retrained on the
hash-pinned public TDC human subset; DrugBank catalog membership alone is not
a fitted-target collision for this artifact. The historical DrugBank-trained
fup artifact remains explicitly barred from the public profile. Inference-time
DrugBank enrichment also remains disabled in that profile.

The exclusion-union SHA256 emitted by the audit becomes part of the frozen
manifest. Re-running the audit must reproduce that hash; the audit report itself
is separately archived to avoid a circular manifest/report hash dependency.
During freeze, run the audit once to obtain the union hash, insert that value into
the manifest, then run it again; only the second report is the passing freeze gate.

For the public profile, `freeze.training_membership_path` must point to
`data/validation/training_membership_sources_v1.json` and its SHA256 must be
recorded in `freeze.training_membership_sha256`. That file pins 18 conservative
corpus inputs used by `scripts/audit_external_holdout_manifest.py`; the audit and
prediction runner verify every listed source hash before continuing. The ignored
N50 convenience inventory is not a freeze dependency. Raw licensed DrugBank
exports are not pinned or used by the current public-only fup artifact. Its
1,557-row filtered human TDC dataset and model artifact are both SHA-pinned;
`scripts/retrain_fup_public.py` rebuilds them from the committed PPBR_AZ input.
The Peff model now uses the SHA-pinned 874-row filtered Caco2_Wang dataset;
`scripts/train_peff.py` rebuilds it from the committed TDC tab file.
The older `scripts/train_fup_v2.py` is a historical DrugBank-dependent recipe
and must not be used to regenerate the public artifact.

The inventory includes the [Omega `mmpk_clean.csv` source at commit
`08a45047`](https://github.com/jam-sudo/Omega/blob/08a45047a2b5dcdca8c9a8f36ff1fe3b50ed3d6d/data/ml/clinical/mmpk_clean.csv),
SHA256 `e7228d14bdfdfc6c790177207779630c1e5655c19d451528c87b80e2e9de9c3d`.
Its 1,128 rows yield the documented 100 exclusions and 1,028 remaining rows
under the current 107-compound holdout and three-key matching. Retraining
with the recorded hyperparameters and current feature code did not reproduce
the former shipped model's trees or predictions. The public Cmax replacement
is now trained from the exact 1,028-row fitted CSV, SHA256
`5668d3747e11d03d8c3d57c03ff0842ed3aa330f4c938386e47c05e1e318c919`,
and its model SHA256 is
`14391eb0881cb3ec83ab2f81592c5fe75da7c949290ec438fddc2d7136f792a0`.
The single-assay TDC hepatocyte CLint replacement is pinned to its 995-row
fitted CSV (SHA256
`dbf2b750b58a68af02b01cfe90a370630908c311c8436f65818bbc08fcfaaa94`)
and model artifact (SHA256
`0ca4ee7e88367dfb3ad55f94adc8eaa025cdd40b5a308e584b89e05555d3ff42`).
Its previous 996-row fitted snapshot contained `CHEMBL1144`, which matches the
corrected pravastatin reference structure; that row was removed before this
retrain.
The public VDss replacement uses [TDC's Lombardo data file](https://dataverse.harvard.edu/api/access/datafile/4267387)
(raw SHA256 `00bb7e0dea19f78c4c1887e27ecf476135d9de2ef5ecc26d7c645c37bd9cb7af`),
with 1,055 fitted rows (CSV SHA256
`778851b9dad82c2eb3d948b7ffb3ae829ce5fd529395b7d0599d4c86e02e5e54`)
and model SHA256
`29f84cbff97da197ae516fccb2d91aff972f9bae661a8b2bde39786a984c1c35`.
All seven active fitted models now name repository-relative, hash-matched
training snapshots in the public inventory. The source-membership prerequisite
is met; an external V1 freeze still requires an untouched candidate cohort,
independent source adjudication, a sealed manifest, and one-time outcome opening.

## Freeze and one-time execution

The release manifest must include git SHA, source-tree hash, model artifact hashes,
training-membership hash, feature schema, resource profile, dependency lock hash,
container digest, random seeds, and output-grid/solver settings. The independent
evaluator/custodian runs the public profile once in the canonical container and
verifies that candidate IDs, arm IDs, doses, routes, and eligibility flags match
the hashed manifest exactly. After labels are opened,
the set is consumed for that and all later model versions.

No compound-specific registry or mechanism change may be made after candidate
identities are disclosed. Correctness bugs found during evaluation are reported;
they do not authorize rerunning the same holdout as a new independent result.

The locked execution sequence is:

Build `scripts/Dockerfile.holdout` on Linux x86_64 from the frozen release
checkout. It installs `requirements-lock.txt`; the custodian mounts that same
clean checkout read-only at `/repo` and the private holdout directory at
`/holdout`. Record `docker image inspect --format '{{.Id}}'` as the local
`sha256:` container identifier in the manifest, verify it again immediately
before each run, and pass that value as `SISYPHUS_CONTAINER_DIGEST`. A published
image may instead be pinned and verified by its registry digest. The runner
also checks the mounted checkout's git SHA and source-tree hash.

```bash
docker build -f scripts/Dockerfile.holdout -t sisyphus-holdout:v1 .
image_id="$(docker image inspect --format '{{.Id}}' sisyphus-holdout:v1)"
docker run --rm -v "$PWD:/repo:ro" -v "$HOLDOUT_DIR:/holdout" \
  -e SISYPHUS_CONTAINER_DIGEST="$image_id" sisyphus-holdout:v1 \
  python scripts/audit_external_holdout_manifest.py /holdout/manifest.json \
  --out /holdout/audit.json
```

Use the same verified image and read-only checkout for prediction and scoring.
For prediction, `/holdout` must contain only the manifest, source plan, audit,
and prediction output; mount decrypted labels from a separate `/labels` volume
only for scoring.

```bash
python scripts/audit_external_holdout_manifest.py /holdout/manifest.json \
  --out /holdout/audit.json
python scripts/predict_external_holdout_manifest.py /holdout/manifest.json \
  --manifest-sha256 <sha256> \
  --audit-report /holdout/audit.json --audit-report-sha256 <audit_sha256> \
  --out /holdout/blinded_predictions.json
python scripts/score_external_holdout.py /holdout/blinded_predictions.json \
  --labels /labels/unblinded.json \
  --manifest /holdout/manifest.json --manifest-sha256 <sha256> \
  --audit-report /holdout/audit.json --audit-report-sha256 <audit_sha256> \
  --predictions-sha256 <predictions_sha256> --labels-sha256 <labels_sha256> \
  --out /holdout/final_score.json
```

The prediction runner refuses a non-public profile, dirty worktree, git/source
tree mismatch, dependency-lock mismatch, artifact-inventory mismatch, training-
membership mismatch, feature-schema mismatch, solver-settings mismatch, audit
failure, container-digest mismatch, stale development-residual interval sources,
or resources outside the frozen checkout.
The scorer refuses any candidate, arm,
dose, route, derived eligibility, source-record hash, source quota, execution
status, interval source, cycle, freeze field, or precommitted file-hash mismatch
before reading the estimand.

## Primary comparison and release gate

For each compound compute:

```
d_i = abs(ln(meta_i / obs_i)) - abs(ln(ml_i / obs_i))
R   = AAFE_meta / AAFE_ml = exp(mean_i(d_i))
```

Retain the meta/PBPK production path only when all primary requirements hold:

- point estimate `R <= 0.90` for N=260, or `R <= 0.85` for the
  preregistered N=120 fallback;
- paired compound-bootstrap 95% CI upper bound `< 1.0`;
- geometric prediction/observation bias in `[0.8, 1.25]`;
- at least 50% within two-fold;
- 90th-percentile fold error below 6;
- nominal 90% residual interval coverage in `[85%, 97.5%]`, with a
  compound-bootstrap coverage CI and median multiplicative half-width reported;
  the median half-width must not exceed `13.0×`.

If superiority fails, direct ML becomes the production default and the PBPK engine
remains a mechanistic research/simulation output. Engine, CL/F, and VDss tracks are
secondary comparisons and cannot replace the predeclared Meta-versus-ML decision.

## Reporting format

Report the primary result exactly as:

> Frozen release `<version>`, independent external primary cohort: Meta AAFE
> `<point>` (compound-cluster bootstrap 95% CI `<low>–<high>`; N=`<compounds>`,
> M=`<arms>`), Meta/ML AAFE ratio `<R>` (paired 95% CI `<low>–<high>`).

Report all eligible compounds as primary. Any applicability-domain or mechanism
subgroup is secondary, even when predeclared. Publish the full exclusion flow,
missingness, per-compound errors, protocol deviations, and challenge strata.
