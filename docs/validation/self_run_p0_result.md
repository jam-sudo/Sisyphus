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

P0 used AI-assisted original-source extraction with coordinator checks. The
source eligibility clarification was committed after first-pass source Cmax
had been viewed but before predictions were unsealed. No independent human
curators verified the labels. P0 therefore cannot establish clinical fitness,
interval calibration, or the formal External Holdout V1 superiority gate.
