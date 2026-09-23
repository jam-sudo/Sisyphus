# Self-run external-data pilot P0

This is a diagnostic pilot, not the independent External Holdout V1. It asks
whether the frozen Cmax model degrades badly on previously unused compounds.
It cannot establish clinical fitness, calibrated interval coverage, or the
Meta-versus-ML superiority gate in `external_holdout_v1_protocol.md`.

## Freeze before outcomes

- Model/code: Git commit `618106b53b0c9ce3c5b8a8fe62c5adf8f02b2308`, public
  resource profile, with no retraining, weight changes, or compound overrides.
- Discovery snapshot: [NCATS Inxight FRDB](https://drugs.ncats.io/downloads-public),
  2024-12-30 ZIP SHA256
  `647b80d9cdac4a0517ce649570517dc1aff40545bdc84617ec9f1a56405f9c7a`.
- Candidate ordering: ascending SHA256 of `sisyphus-p0-2026-09-23:` followed by
  the FRDB `compound_id`. Within a compound, order rows by numeric FRDB `id`.
- Select rows using metadata only: `ADULT`, `HEALTHY`, `FASTED`, `Oral`, `SINGLE`,
  `PLASMA`, `pkappcombo=false`, a parseable analyte structure, a positive dose
  in mg, μg, or g, and a Cmax unit
  in {ng/mL, μg/mL, μg/L, mg/L, pg/mL, ng/L, mg/mL}. Do not inspect
  `pk_cmax_value` during selection.
- Exclude compound identities and connectivity keys found in fitted training,
  clinical registries, N=107 development, N=28 temporal, or invalidated N=50
  sources using the repository's holdout exclusion logic. Record every exclusion.

Commit the label-free candidate manifest and prediction script before running
the model. Predict **all** selected candidates with the frozen release and
record the prediction-file SHA256 in a local commit before opening any Cmax value or
original-source PK table. No candidate may be added after that commitment.

## Extract and score once

Walk the frozen candidate order. For each candidate, check the original cited
source for parent-drug plasma Cmax after a single immediate-release oral dose
in fasted healthy adults; record dose, units, analyte, formulation, page/table,
source URL, and reason for any exclusion. An AI second pass may flag extraction
disagreements but does not count as an independent human verifier. Use every
eligible arm of a compound; never choose an arm by prediction error. Stop after
50 source-verified compounds or exhaustion of the committed candidate list.

Report compound-cluster AAFE for Meta and direct ML, their paired ratio and
bootstrap CI, within-twofold rate, all exclusions, and the actual sample size.
The ratio is descriptive: N=50 is not powered for the V1 superiority margin.
Do not tune on these outcomes or report the pilot as an independent blinded
holdout. Once opened, these compounds are development data and are excluded
from any future final test.

Before any outcome value or original PK table was opened, the application/analyte
connectivity-match filter was removed: the FRDB application-SMILES field is
empty in all 1,750 rows that pass the other metadata filters. Parent-analyte
identity remains a mandatory original-source check after predictions are
committed. This amendment was made from field completeness alone.
