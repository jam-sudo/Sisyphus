# AI-assisted blind P1 diagnostic — 2026-09-24

This is a **consumed development-grade diagnostic**, not the independently
curated External Holdout V1. One Claude Code worker collected eight previously
unseen compound arms while the modeling agent remained blind to their Cmax
values. The label-free [inputs](../../data/validation/blind_p1_inputs_2026-09-24.json),
[predictions](../../data/validation/blind_p1_predictions_2026-09-24.json), and
[curation summary](blind_p1_curation_summary_2026-09-24.md) were committed as
`40d222e` before the separately held [labels](../../data/validation/blind_p1_labels_2026-09-24.json)
were opened. The prediction artifact records model commit `84c0d3c`, public
resource hashes, dependency versions, input SHA256, and the prior label-content
SHA256. All eight strict predictions completed; no fitted model or routing rule
was changed. The label SHA256 still matches the pre-prediction commitment.

| Source arm | Observed Cmax (mg/L) | Meta (mg/L) | Direct ML (mg/L) |
| --- | ---: | ---: | ---: |
| [Usnoflast / ZYIL1, 100 mg](https://doi.org/10.1002/cpdd.1162) | 11.5 | 0.347 | 0.200 |
| [DFV890 crystalline suspension, 100 mg](https://doi.org/10.1111/cts.13789) | 4.670 | 0.335 | 0.352 |
| [EC5026 fasted HME tablet, 8 mg](https://doi.org/10.1111/cts.70033) | 0.0350 | 0.0284 | 0.0446 |
| [Fazamorexant, 40 mg](https://doi.org/10.2147/DDDT.S501111) | 1.570 | 0.158 | 0.186 |
| [Cabamiquine / M5717, 400 mg](https://doi.org/10.1016/S1473-3099(21)00252-8) | 0.146 | 0.937 | 0.529 |
| [Flizasertib / GDC-8264, 75 mg](https://doi.org/10.1111/cts.13607) | 0.783 | 0.301 | 0.498 |
| [ALG-055009 oral solution, 4 mg](https://doi.org/10.1002/cpdd.1606) | 0.0922 | 0.0169 | 0.0299 |
| [Culmerciclib / TQB3616, 180 mg](https://doi.org/10.3389/fphar.2025.1586368) | 0.06039 | 0.425 | 0.166 |

The modeling agent rechecked each selected dose/Cmax cell against the original
paper's table in the Europe PMC full-text XML, including the μg/mL-to-mg/L
and ng/mL-to-mg/L conversions. All eight copied central values matched. Input
and label SMILES have identical full InChIKeys. A repeat of the repository's
name/connectivity exclusion lookup found no prior hit after excluding this
newly committed P1 artifact itself. The eight compounds are now modeler-seen
and must not enter a future blinded final or reserve cohort.

| Descriptive diagnostic | All eight | Without source-inconsistent TQB3616 |
| --- | ---: | ---: |
| Compounds / arms | 8 / 8 | 7 / 7 |
| Meta AAFE (compound bootstrap 95% CI) | 6.60 (3.40–12.54) | 6.54 (3.09–13.62) |
| Direct-ML AAFE (95% CI) | 5.01 (2.37–11.96) | 5.46 (2.34–14.40) |
| Paired Meta/ML AAFE ratio (95% CI) | 1.32 (0.96–1.76) | 1.20 (0.89–1.55) |

The scorer uses absolute natural-log error per compound and 100,000 paired
compound-bootstrap resamples (seed `20260924`). This one-arm-per-compound pilot
shows large errors on several investigational drugs, notably a roughly 33-fold
Meta underprediction for usnoflast and 14-fold for DFV890. Its paired interval
crosses 1; the pilot establishes neither Meta superiority nor inferiority in a
target population. The open-access/fasting-phrase discovery process is
nonconsecutive and chemically selective. The cohort is small and mixes
arithmetic and geometric source means.
All eight predictions were marked structurally in-domain; that flag did not
identify the large errors in this pilot. The model's nominal 90% development-
residual interval covered six of eight observations, missing usnoflast and
DFV890. Eight observations are too few to estimate interval coverage reliably.

A post-label structural-neighbor check used Morgan radius-2, 2,048-bit
fingerprints against the 1,028-row public Cmax training snapshot. Six P1
compounds had nearest-neighbor Tanimoto similarity ≤0.30, but splitting at
this exploratory cutoff gave nearly identical Meta AAFE: 6.56 for those six
versus 6.72 for the other two. The same split on the repeatedly used 73-drug
development set gave 4.18 (21 drugs) versus 2.43 (52 drugs). Thus structural
novelty may contribute to development-set error, but this cutoff does not
explain the P1 failures and is not a justified new applicability gate. It was
examined after seeing P1 labels and provides no fresh validation evidence.

The two largest underpredictions also have source-reported exposure data.
[Usnoflast Table 1](https://doi.org/10.1002/cpdd.1162) gives 100 mg
AUC0–t **92.3 mg·h/L** and median Tmax **1 h**; the frozen engine predicts
AUC0–24 **1.301 mg·h/L** and Tmax **2.16 h**. [DFV890 Table 3](https://doi.org/10.1111/cts.13789)
gives 100 mg crystalline-suspension AUC0–last **72.2 mg·h/L** and median Tmax
**2 h**; the engine predicts AUC0–24 **0.358 mg·h/L** and Tmax **1.35 h**.
The AUC observation horizons differ from 24 h, so these are directional
comparisons, not formal AUC fold errors. They show a broader exposure deficit
alongside the Cmax deficit; changing peak timing alone cannot explain it.
These post-label observations must not be used to fit a P1-specific correction.

**No arm supplies protocol-compliant V1 final-test evidence.** The source
review was AI-assisted and single-agent; none has two independent human checks.
Immediate release was inferred for seven conventional tablets, capsules, or
suspension; the eighth was an oral solution. The reports do not affirm
active-moiety dose equivalence to a
named free form or rule out all salt formulations. The TQB3616 paper states a
180 mg administered dose but also calls the article 5 mg/capsule; the
[trial registry](https://clinicaltrials.gov/study/NCT05344729) repeats the
180 mg total dose without resolving the unit-strength discrepancy. The
seven-compound analysis was specified from this source defect before inspecting
the prediction errors and is only a sensitivity. There is also no regulatory
source quota or externally timestamped custody. Further tuning on these eight
would convert their diagnostic errors into development feedback, not independent
validation. An N=120/260 protocol-compliant test is still required for an
85/100 quality claim.
