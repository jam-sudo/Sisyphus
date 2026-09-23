# Self-run P0 result (2026-09-23)

The source adjudication was committed as `d9c4e4c` before the sealed prediction
file was opened. Its SHA256 is
`cb964fc5a30ad140df54a9d175082f154038312c2d8337cbdc87c5c142673812`.
All 186 frozen candidates and 452 arms were assigned; 18 compounds and 58 arms
qualified, 47 compounds were excluded, and 121 remain unresolved. The sample
is smaller than the planned maximum of 50 because the committed list was
exhausted under the source rules.

| Diagnostic P0 measure | Meta | Direct ML |
| --- | ---: | ---: |
| Compound-cluster AAFE | 3.34 (95% bootstrap CI 2.43–4.79) | 3.01 (2.26–4.18) |
| Compound-cluster within twofold | 33.3% | — |

The paired Meta/direct-ML AAFE ratio is **1.11** (95% cluster-bootstrap CI
**0.93–1.34**, 100,000 resamples, seed 20260923). A ratio below 1 would favor
Meta. This pilot does not show Meta superiority, and its wide interval is
consistent with either direction. Meta geometric bias is 0.68, indicating
underprediction on average in this selected cohort. The complete result and
reproducible scorer are `data/validation/self_run_p0_results.json` and
`scripts/score_self_run_p0.py`.

Large compound-level errors are diagnostic leads, not tuning targets:
verlukast is underpredicted about 25-fold by Meta across two doses; oxatomide
is overpredicted about 9-fold across two formulations; teneligliptin and
ipragliflozin are underpredicted about 4-fold across seven and nine arms.
Those mechanisms and source mappings need investigation before any model
change. Re-scoring this same P0 cohort after changes would be development
testing, not an independent confirmation.

Track-level reruns of seven compounds reproduced their sealed Meta values.
For example, at 20 mg teneligliptin the observed Cmax was 0.236 mg/L, while
the engine, direct ML, and Meta values were 0.0074, 0.142, and 0.046 mg/L.
The engine's large error pulls the blend away from the more accurate ML
estimate. At 75 mg verlukast, all four tracks were below the observed 6.7
mg/L (engine 0.096, ML 0.284, CL/F 0.122, VDss 1.285 mg/L); at 60 mg
oxatomide, all four were above the observed 0.0136 mg/L. Thus a change to
blend weights alone cannot resolve the broader errors. These reruns used a
temporary macOS environment with SciPy 1.16.3 because the pinned 1.15.3 wheel
failed to load locally; all seven Meta outputs matched the sealed values.

The engine's optional 24-hour-truncated oral/IV AUC ratio was 0.051 for
teneligliptin, 0.075 for ipragliflozin, and 0.179 for garenoxacin in these
same reruns. This ratio is a diagnostic approximation, not a measured absolute
bioavailability. For context, the [EMA's garenoxacin assessment](https://www.ema.europa.eu/en/documents/withdrawal-report/withdrawal-assessment-report-garenoxacin-mesylate_en.pdf)
reports 92% absolute oral bioavailability in fasted healthy subjects. The
repository's prior [DE-42 analysis](../research/dead-ends.md) already tested
uniform absorption-rate increases: they improved some engine predictions but
increased the opposite-error tail and did not improve the final Meta model.
P0 therefore supports investigating compound-specific input and disposition
errors, not repeating a global absorption multiplier.

As a post-unseal source concern, [FDA GSRS identifies verlukast sodium](https://precision.fda.gov/uniisearch/srs/unii/Q8W8588793),
while the [clinical paper](https://pubmed.ncbi.nlm.nih.gov/12959296/) names
verlukast tablets without establishing in the reviewed record whether its
75/500 mg doses are salt mass or active-moiety mass. The primary cohort was
not changed after unsealing. Excluding this compound as a *post-hoc
sensitivity* gives Meta AAFE 2.97, direct ML AAFE 2.67, and ratio 1.11 across
17 compounds; the substantive conclusion is unchanged. This uncertainty
still weakens confidence in P0 source eligibility.

The development-residual 90% band covered 55/58 included arms, but its fixed
half-width is a factor of 12.9 on either side of the point prediction. This
pilot cannot establish independent interval calibration.

P0 used AI-assisted original-source extraction with coordinator checks. The
source eligibility clarification was committed after first-pass source Cmax
had been viewed but before predictions were unsealed. No independent human
curators verified the labels. P0 therefore cannot establish clinical fitness,
interval calibration, or the formal External Holdout V1 superiority gate.
