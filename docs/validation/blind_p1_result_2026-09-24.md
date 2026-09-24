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
