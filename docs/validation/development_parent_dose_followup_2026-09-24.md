# Development reference parent-dose follow-up — 2026-09-24

The model takes parent-molecule SMILES and a parent-equivalent dose. Three
scored single-dose rows still used salt mass or an unmatched observed arm.
The original labels and the corrected model inputs are:

| Drug | Source arm and parent Cmax | Parent-equivalent model dose | Change |
| --- | --- | ---: | --- |
| Methylphenidate | 20 mg hydrochloride oral solution, fasted healthy volunteers; 9.1 ng/mL | 20 × 233.31 / 269.77 = 17.296957 mg | Convert salt mass; keep Cmax 0.0091 mg/L. |
| Levocetirizine | 5 mg dihydrochloride tablet, adult single dose; 270 ng/mL | 5 × 388.9 / 461.8 = 4.210697 mg | Convert salt mass; keep Cmax 0.27 mg/L. |
| Quinine | 648 mg sulfate (two capsules), 23 healthy subjects, single oral dose; 3.2 ± 0.7 mcg/mL | 2 × 269 = 538 mg | Replace the unverified 600 mg / 5.4 mg/L pair with the label's matched arm. |

Sources: [methylphenidate hydrochloride DailyMed label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=57fd619e-687d-403c-a77a-3dc3a1bab65a), [methylphenidate parent MW](https://pubchem.ncbi.nlm.nih.gov/compound/Methylphenidate), [hydrochloride MW](https://pubchem.ncbi.nlm.nih.gov/compound/Methylphenidate-Hydrochloride); [XYZAL DailyMed label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=1673f7ff-0c7c-4403-86cf-c05eb1475222), [levocetirizine parent MW](https://pubchem.ncbi.nlm.nih.gov/compound/1549000), [dihydrochloride MW](https://pubchem.ncbi.nlm.nih.gov/compound/Levocetirizine-Dihydrochloride); [QUALAQUIN DailyMed label, section 12.3, Table 1](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=1f66fba7-4026-4504-918d-4c88f2835cc0). The quinine label itself states 324 mg sulfate per capsule equals 269 mg free base. Its Table 1 AUC is AUC0–12, so it was not entered as generic AUC. The 648 mg arm's meal state is not specified in that table. The old extracted quinine 6.8 mcg/mL belongs to a **seven-day, three-times-daily steady-state** arm, not the single-dose arm.

The public-profile, repeatedly accessed N=79 development benchmark now gives
Meta AAFE **2.9348** (conditional bootstrap 95% CI **2.3989–3.6527**), Engine
**3.8747**, direct ML **3.2882**, and descriptive in-domain Meta **3.0194**
(N=65). The paired Meta/ML AAFE ratio is **0.8925** (CI **0.7890–1.0048**).
The nominal 90% development-residual half-width remains 10.24× and covers
72/79. Fitted models did not change. The movement from 2.9492 is a reference
correction, not evidence of improved model generalization. These rows and the
cohort have already informed development, so no independent accuracy claim
follows from the re-score.
