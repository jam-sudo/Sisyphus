# Development training dose and specimen screen — 2026-09-24

The subsequent [additional arm screen](development_reference_arm_rescreen_2026-09-24.md)
supersedes the interval figures below; the reference and training counts remain
127 and 38.

This pass selected eight remaining training records by drug name, without
consulting their model residuals. It checked dose, formulation, specimen,
analyte, population, and dosing schedule against primary labels. Six records
needed action; aspirin and ezetimibe were left unchanged pending exact
arm-level confirmation.

| Record | Primary-source finding | Action |
| --- | --- | --- |
| Hydroxychloroquine | [DailyMed](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=006801ed-d93f-4346-863a-cc03e6712520): single 200 mg sulfate tablet equals 155 mg parent base; 129.6 ng/mL is whole blood, whereas parent plasma Cmax is 50.3 ng/mL in healthy males. The same study gives a plasma terminal half-life of 2963 h. | Retained with 155 mg parent dose and 0.0503 mg/L plasma Cmax; removed synthetic curve and unsupported bioavailability. |
| Isotretinoin | [DailyMed Table 2](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=c2917c3d-3499-48a0-ba53-120cb979195d): single fasted 80 mg in 74 healthy adults gives parent Cmax 301 ng/mL, AUC0–∞ 3703 ng·h/mL, and half-life 21 h. The old 573.25 ng/mL was a pediatric arm; 90 h referred to radiolabel activity in blood. | Re-sourced adult parent plasma arm; removed synthetic curve. |
| Tranexamic acid | [DailyMed Table 3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=af95be04-a6ea-4789-8e79-7dcad332c640): the 13.83 μg/mL Cmax belongs to a single fasted **1300 mg** oral dose in 19 healthy women, not to one 650 mg tablet. Its AUC0–∞ is 80.19 μg·h/mL and half-life 11.08 h. | Corrected dose and same-arm PK parameters; removed synthetic curve. |
| Paricalcitol | [DailyMed Table 7](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=1b27c026-2d60-47b4-2a9c-ded1818f21b5): the 4 μg, 0.11 ng/mL Cmax arm is in adult stage 3 CKD patients, outside the healthy-adult target. Its half-life is 16.8 h rather than the row's 4 h. | Quarantined. |
| Vorasidenib | [DailyMed §12.3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=31405fee-55b7-4857-987e-2724ee76be84): 133 ng/mL is a **40 mg daily steady-state** mean, not a 200 mg single-dose Cmax. | Quarantined. |
| Entecavir | [DailyMed §12.3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=92a5bebd-6408-463d-933c-bab8338a677e): 4.2 ng/mL belongs to **0.5 mg at steady state**; 1 mg steady-state Cmax is 8.2 ng/mL. Tablet-versus-solution bioavailability is relative, not absolute. | Quarantined. |

The reference now contains **127** usable Cmax rows. The development-residual
calibration uses **38** training records; the repeatedly used development
benchmark remains **86** scored compounds. The nominal 90% Meta half-width is
**5.45×**, with only **75.58%** observed coverage on that consumed benchmark.
The latter is diagnostic, not an independent coverage estimate or a calibrated
90% guarantee. No new blinded external cohort was created; see the
[V1 protocol](external_holdout_v1_protocol.md).
