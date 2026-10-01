# AI-assisted development reference cross-check — 2026-09-25

One AI-assisted worker, blind to predictions and prior audit conclusions, compared five `clinical_pk.json` Cmax arms with their cited primary sources. I separately checked the [XYZAL label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=1673f7ff-0c7c-4403-86cf-c05eb1475222). The worker's detailed read-only record is local at `/tmp/sisyphus_claude_reference_review_20260925.md`. This is development data QA by AI, not independent external validation or a new holdout score.

| Arm | Source check | Remaining limitation |
| --- | --- | --- |
| [Clopidogrel](https://pmc.ncbi.nlm.nih.gov/articles/PMC3690105/), 300 mg / 14.5 ng/mL | Reported parent Cmax agrees. | 24 analysed patients were women with stable coronary disease; sampling ended at 4 h. |
| [Lenacapavir](https://pmc.ncbi.nlm.nih.gov/articles/PMC10994821/), 300 mg / 23.4 ng/mL | Agrees with the geometric mean for the 10 normal-hepatic-function controls. | A separate 10-person normal-renal-function control arm reports 19.7 ng/mL; meal state and exact formulation are unspecified. |
| [Budesonide](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=4a8e79df-3946-4e9a-b214-e9fdd824b565&version=4), 9 mg / 1.50 ng/mL | Agrees with the fasting single-dose mean. | Enteric-coated, delayed-release capsule. |
| [Levocetirizine](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=1673f7ff-0c7c-4403-86cf-c05eb1475222), 5 mg dihydrochloride / 270 ng/mL | Agrees with the adult single-dose tablet label and parent-equivalent model dose. | Label says only "typically"; statistic, sample size, and meal state are unspecified. Downgraded gold → silver. |
| [Paroxetine](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=027c7db2-2a36-477f-90bf-608ff3a0e14e&type=display), 25 mg / 5.5 ng/mL | Agrees with the mean for 23 normal volunteers. | Controlled-release, enteric-coated tablet. |

No Cmax value changed. These repeated development checks do not supply the untouched, independently curated external cohort required for an 85/100 validation claim.
