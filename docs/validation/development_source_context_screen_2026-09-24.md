# Development training source-context screen — 2026-09-24

The subsequent [dose and specimen screen](development_matrix_dose_screen_2026-09-24.md)
supersedes the counts and interval figures below.

After the residual-led audit, this screen checked training records by their
**names, administered drugs, formulations, analytes, and cited label contexts**,
without selecting them by prediction error. The following eleven records had
source-context mismatches or needed exact re-sourcing:

| Record | Primary-source finding | Action |
| --- | --- | --- |
| Benzhydrocodone | [APADAZ Table 4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=3a790a49-df42-4577-914c-6ee07a40b60f) gives 16.04 ng/mL **hydrocodone** after **6.12 mg benzhydrocodone + acetaminophen, fed**; prodrug exposure was not measurable. | Quarantined the 4.08 mg parent row. |
| “Dimethyl” | The [TECFIDERA label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=665d7e74-036c-5f68-5b67-ab84b9b49151) gives 1.87 mg/L **monomethyl fumarate** after **240 mg dimethyl fumarate twice daily with food**; parent is not quantifiable. The row's SMILES `CC` encodes ethane. | Quarantined. |
| Guanfacine ER | [Table 15 and adjacent text](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=187bd24b-d69b-4cdf-be4a-c89e9b131b81) give 1 mg adult ER Cmax **1.0 ng/mL**, whereas **10 ng/mL and 162 ng·h/mL** concern **4 mg multiple-dose pediatric ER** exposure. | Quarantined the 1 mg row; ER is also outside the primary IR estimand. |
| Naproxen | The [delayed-release label](https://www.dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=14e4b008-ae31-4b40-a227-1445367e09ed) gives **94.9 μg/mL after 500 mg twice daily for one week**, not a single IR dose. | Quarantined. |
| Oseltamivir | [TAMIFLU Table 6](https://www.dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=247fb507-b3a1-4005-abb9-d17774fa2996) gives **65 ng/mL parent** after **75 mg twice-daily multiple dosing**, not 30 mg single dose. | Quarantined. |
| Naproxen oral suspension | The [suspension label](https://www.dailymed.nlm.nih.gov/dailymed/getFile.cfm?setid=570974d4-0d5b-4df2-b307-37380511835d&type=pdf) reports **64.3 μg/mL parent** after a **500 mg single oral dose** in a fasted crossover of 12 subjects. | Retained, corrected half-life to 16.8 h, removed synthetic curve. |
| Atazanavir | The [REYATAZ label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=165cff62-b284-4a27-a65d-9ec8a5bfcdd8) assigns **3190 ng/mL / 34459 ng·h/mL** to atazanavir given with **ritonavir and tenofovir**. | Quarantined the unboosted single-dose row. |
| Butalbital | The [combination label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ac6ceec2-72d8-4901-9d30-278965a79049) gives **8.8 μg/mL after 650 mg aspirin** under the aspirin subsection; butalbital has a much longer half-life. | Quarantined the mislabeled butalbital row. |
| Cefixime | The [cefixime label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=b5a3c547-3bc7-4856-af9c-1bead294f5d9) gives **3.7 μg/mL serum** after a **400 mg** tablet; the row was 200 mg and the model target is plasma. | Quarantined. |
| Efavirenz | [Table 6](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=c5eedb5d-4cab-4056-a14a-ffdf67701355) gives **6.57 μg/mL as a predicted pediatric steady-state** Cmax for the **600 mg** weight band. | Quarantined the 300 mg measured-single-dose claim. |
| Clarithromycin | The [Health Canada monograph, Table 23](https://pdf.hres.ca/dpd_pm/00050373.PDF) gives **1.77 mg/L parent Cmax** after **single fasted 500 mg** oral dosing. | Re-sourced the unsupported 0.8 mg/L row. |

The available Cmax reference count is now **130** and the diagnostic training
calibration set **41**; the scored development benchmark remains **86**.
The recalculated nominal 90% Meta half-width is **5.34×**, but coverage on the
already-used development cohort is only **73.26%**. This is not a calibrated
90% prediction interval, and neither this audit nor the development benchmark
can serve as the blinded external test required by the
[V1 protocol](external_holdout_v1_protocol.md).
