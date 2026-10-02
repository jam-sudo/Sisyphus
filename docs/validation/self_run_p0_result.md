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

A post-hoc fitted-set audit found that VERLUKAST and GARENOXACIN, two of the
18 scored compounds (8 arms), occur in the current TDC Lombardo VDss fitted
snapshot (SHA256
`778851b9dad82c2eb3d948b7ffb3ae829ce5fd529395b7d0599d4c86e02e5e54`).
The historical VDss artifact at `618106b` used to score P0 describes its source
as `TDC VDss_Lombardo` but records the fitted-row SHA256 as `unknown_legacy`;
its exact training membership cannot be reconstructed. This is **possible,
not proven, training overlap** for P0. Excluding both compounds after unsealing
gives Meta AAFE 2.90 (95% bootstrap CI 2.20–3.86), direct-ML AAFE 2.61, and
a paired ratio of 1.11 (0.91–1.37) across 16 compounds and 50 arms. Both
tracks improve on this selected subset, while the relative conclusion stays
unchanged. This sensitivity does not replace the original 18-compound result
or establish an independent cohort.

The pre-unseal protocol also requested a same-source-statistic sensitivity.
Only compounds whose *every included arm* has the same Cmax statistic enter
each group; the primary 18-compound result above remains unchanged.

| Source Cmax statistic | Compounds / arms | Meta AAFE | Direct-ML AAFE | Paired ratio (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic mean | 10 / 37 | 3.82 | 3.13 | 1.22 (0.97–1.58) |
| Geometric mean | 4 / 12 | 2.48 | 2.53 | 0.98 (0.67–1.36) |
| Geometric LSmean | 2 / 2 | — | — | — |
| Median | 1 / 1 | — | — | — |

Garenoxacin's six arms mix arithmetic and geometric means, so it is omitted
from these homogeneous-statistic groups only. Groups below four compounds
are counted but not scored. Among final eligible compounds whose *scored
arms* were all labelled `verified` in the original AI first pass, six
compounds / 31 arms have Meta AAFE 4.11, direct-ML AAFE 2.93, and paired
ratio 1.40 (95% bootstrap CI 1.04–1.96). The original `verified` +
`verified_with_caveat` labels are only
an inclusive screening ceiling; many such arms failed mandatory source
rules and cannot be scored as a valid holdout. These small, selected subsets
show sensitivity to source composition, not comparative efficacy.

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
bioavailability. A separate [PMDA ipragliflozin review](https://www.pmda.go.jp/files/000206796.pdf)
reports measured absolute bioavailability **90.2% ± 5.3%** in 14 healthy adults
after 100 mg oral versus 25 mg IV dosing (Study CL-0057, pp. 31–32). At the
same 100 mg oral dose, a post-unseal engine rerun gives a 24-hour AUC ratio
of **7.47%** and Cmax **0.0795 mg/L**, versus the study's observed
**1.406 mg/L**. Supplying the *measured* F to the existing conditional route
raises engine Cmax to **0.961 mg/L** and Meta Cmax to **0.737 mg/L**. This
exploratory intervention uses a clinical input unavailable in structure-only
prediction; it supports a large exposure error in the engine but does not
establish which absorption or first-pass component causes it. The deterministic
100 mg engine mass balance at 24 h places **68.85 mg** in the fecal sink and
**18.82 mg** still in the colon lumen; these two unabsorbed compartments alone
account for **87.67%** of the simulated dose. This localizes much of the
discrepancy to simulated uptake from the gut, while the actual tablet's
formulation is not an input to the structure-only model. The rerun used a
temporary macOS environment with NumPy 2.5.3 and SciPy 1.16.3; its baseline
100 mg Meta Cmax matches the sealed 100 mg prediction to numerical precision.
For context, the [EMA's garenoxacin assessment](https://www.ema.europa.eu/en/documents/withdrawal-report/withdrawal-assessment-report-garenoxacin-mesylate_en.pdf)
reports 92% absolute oral bioavailability in fasted healthy subjects. The
repository's prior [DE-42 analysis](../research/dead-ends.md) already tested
uniform absorption-rate increases: they improved some engine predictions but
increased the opposite-error tail and did not improve the final Meta model.
P0 therefore supports investigating compound-specific input and disposition
errors, not repeating a global absorption multiplier.

The structure-only ADME path predicts ipragliflozin Peff **0.488 × 10⁻⁴
cm/s** and aqueous solubility **0.0226 mg/mL**. The latter selects a 35 µm
effective particle radius, giving proximal `ka = 0.0401 h⁻¹` from
`2.88 × Peff / radius`, versus duodenal transit **3.846 h⁻¹**. A separate
[experimental patent comparison](https://patents.google.com/patent/KR102097250B1/en)
reports **0.554 mg/mL** at pH 6.8 for an ipragliflozin L-proline co-crystal
(Table 2); this is a different solid form from the free-molecule SMILES input
and is not a measurement of the clinical tablet's in-vivo dissolution. As a
post-unseal diagnostic, supplying only this solubility value through
`MeasuredADMEInput` changes the 100 mg engine Cmax **0.0795 → 0.1095 mg/L**
and Meta Cmax **0.3670 → 0.4014 mg/L**, still far below the observed
**1.406 mg/L**. The model uses solubility only to select a coarse particle-
radius bucket, so this experiment cannot isolate permeability from
formulation effects. No coefficient or benchmark label was changed.

As a post-unseal source concern, [FDA GSRS identifies verlukast sodium](https://precision.fda.gov/uniisearch/srs/unii/Q8W8588793),
while the [clinical paper](https://pubmed.ncbi.nlm.nih.gov/12959296/) names
verlukast tablets without establishing in the reviewed record whether its
75/500 mg doses are salt mass or active-moiety mass. The
[scanned original Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC1364621/)
confirms **6.7 and 41.3 μg/mL** for those fasted arms; it does not resolve the
dose basis. The primary cohort was
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
