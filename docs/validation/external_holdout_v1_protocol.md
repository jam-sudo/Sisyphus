# External Holdout V1 Protocol

## Decision objective

Measure the frozen system's independent structure-only Cmax accuracy and decide
whether the PBPK/meta stack adds enough value over the direct ML track to justify
its production complexity. N=107 is development data for this purpose; N=28 is
already consumed; the 2026Q2 N50 is invalidated.

## Frozen primary estimand

The primary cohort contains adult, single-dose, oral immediate-release, fasted,
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
At two-sided alpha 0.05 and 80% power this implies approximately:

- N=90–110 compounds to detect a 15% AAFE improvement.
- N=210–260 compounds to detect a 10% AAFE improvement.

The recommended target is **N=260 compounds**. A resource-limited design may fix
N=120 before curation begins, but must then define the relevant effect as at least
15%; it cannot interpret a nonsignificant 5–10% difference as equivalence.

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
   and source table/page.
5. Store labels encrypted or in a separate access-controlled repository. The
   custodian retains both identities and outcomes until the container is frozen.
   The evaluator receives the frozen container plus a label-free arm manifest
   (SMILES, dose, route, arm ID); the modeling team receives identities only after
   the one-time run is committed.

The label-free, blinded-label, and frozen-prediction contracts are pinned in
`data/reference/external_holdout_v1_manifest.schema.json` and
`data/reference/external_holdout_v1_labels.schema.json`, and
`data/reference/external_holdout_v1_predictions.schema.json`. Every executable
stage validates these schemas. Primary eligibility is recomputed from the label
metadata at scoring; the manifest boolean is never trusted by itself.

### Operational acquisition plan

N=260 cannot be obtained from the current repository: its clean internal candidate
pool is zero after structure-level exclusion. The set therefore requires a fresh
external acquisition campaign. Enumerate approximately 900 candidates before any
prediction is run, targeting at least 550 verified compounds after attrition.
An acquisition feasibility check on the FDA's [1985–2025 NME compilation](https://www.fda.gov/drugs/drug-approvals-and-databases/compilation-cder-new-molecular-entity-nme-drug-and-new-biologic-approvals)
(retrieved 2026-09-23) found 682 NDA rows with any route field marked oral, of
which 132 were approved in 2020–2025. These are raw product counts, not unique,
eligible, decontaminated compounds; they are an upper bound for that FDA source.
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

Allocate those compounds without outcome-based replacement to three disjoint
roles: 120–150 calibration-development compounds, N=260 final external-test
compounds, and a sealed reserve cohort. Calibration labels may be opened before
the final freeze and immediately become development data; final-test and reserve
labels remain inaccessible:

- regulatory clinical-pharmacology packages from FDA, EMA, PMDA, Health Canada,
  and TGA for 2020–2026 novel oral small molecules not already consumed;
- peer-reviewed first-in-human/SAD studies for development compounds with an
  unambiguous structure and directly tabulated Cmax;
- older approved oral drugs absent from fitted target corpora; DrugBank catalog
  membership is allowed under the frozen public profile, but use in a runtime
  clinical registry is not;
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

## Contamination gate

Before model freeze, canonicalize both candidate and corpus structures using:

- largest organic fragment after counterion removal;
- canonical isomeric and non-isomeric SMILES;
- full InChIKey and 14-character connectivity block;
- normalized generic names and synonyms;
- explicit parent/prodrug/active-metabolite relations.

Reject collisions with every fitted model target corpus, clinical reference used
by runtime registries, previous validation set, manual per-drug override, and
meta-weight/routing cache. DrugBank membership alone is not contamination under
the public profile if DrugBank enrichment is disabled and its values were not
used to fit an artifact.

The exclusion-union SHA256 emitted by the audit becomes part of the frozen
manifest. Re-running the audit must reproduce that hash; the audit report itself
is separately archived to avoid a circular manifest/report hash dependency.
During freeze, run the audit once to obtain the union hash, insert that value into
the manifest, then run it again; only the second report is the passing freeze gate.

For the public profile, `freeze.training_membership_path` must point to
`data/validation/training_membership_sources_v1.json` and its SHA256 must be
recorded in `freeze.training_membership_sha256`. That file pins the 11 fitted-target
corpus inputs used by `scripts/audit_external_holdout_manifest.py`; the audit and
prediction runner verify every listed source hash before continuing. The ignored
N50 convenience inventory and
licensed DrugBank exports are not freeze dependencies.

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
  --predictions-sha256 <predictions_sha256> --labels-sha256 <labels_sha256> \
  --out /holdout/final_score.json
```

The prediction runner refuses a non-public profile, dirty worktree, git/source
tree mismatch, dependency-lock mismatch, artifact-inventory mismatch, training-
membership mismatch, feature-schema mismatch, solver-settings mismatch, audit
failure, or container-digest mismatch. The scorer refuses any candidate, arm,
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

- point estimate `R <= 0.90`;
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
