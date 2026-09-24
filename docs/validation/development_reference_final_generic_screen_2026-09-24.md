# Development reference generic-source screen — 2026-09-24

Four training Cmax rows were checked against original study tables or regulatory
labels without selecting by model residual. This repairs reference data, not
fitted model weights or independent validation.

| Record | Source finding and correction |
| --- | --- |
| Ibuprofen | The [oral-suspension label, Table 1](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=d40a898e-524a-4f6e-83a7-bab4ce5218d1&type=display) assigns the old **19 µg/mL** to **200 mg suspension in adults**, not 400 mg. [Bramlage and Goldis 2008, Table 2](https://link.springer.com/article/10.1186/1471-2210-8-18) reports parent plasma Cmax **32.92 µg/mL**, AUC0–∞ **117.38 µg·h/mL**, and t½ **2.52 h** after a single fasted **400 mg Nurofen Forte coated tablet**. |
| Metoclopramide | The [ODT label, Table 6](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=7cd1dc35-2fb2-4f1d-8769-24c3c3aa34fa) assigns the old **44 ng/mL** to **20 mg metoclopramide alone** in a fluoxetine interaction study. Table 5 instead gives parent plasma Cmax **28 ng/mL** and AUC0–∞ **268 ng·h/mL** in **41 healthy adults** after one fasted **10 mg ODT**. The label expresses ODT strength as parent equivalent. The former half-life and bioavailability came from other arms and were removed. |
| Furosemide | The former **1.11 mg/L** was approximately right but misattributed to Mehanna et al. The matching [Najib et al. 2003 study, Table 1](https://www.ammanu.edu.jo/english/pdf/StaffResearch/Pharmacy/10086/Bioequivalence%20evaluation%20of%20two%20brands%20of%20furosemide%2040mg%20tablets%20(Salurin%20and%20Lasix)%20in%20healthy%20human%20volunteers.pdf) reports the single fasted **40 mg Lasix reference tablet** parent plasma Cmax **1109.71 ng/mL** (table rounded to 1109), AUC0–∞ **2609 ng·h/mL**, and t½ **2.47 h** in 24 healthy men. |
| Terbinafine | The [tablet label, §12.3](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=e0b309a5-bebe-c6da-2847-bda21aa488b1&type=display) confirms approximately **1 µg/mL** parent plasma Cmax after one **250 mg parent-equivalent** tablet and approximately **40%** oral bioavailability. Its **36 h** is an *effective accumulation half-life*, not a measured single-dose terminal half-life; AUC **4.56 µg·h/mL** has no specified integration horizon. Both ambiguous fields were removed. The input structure now encodes the marketed **(E)** geometry ([PubChem CID 1549008](https://pubchem.ncbi.nlm.nih.gov/compound/Terbinafine)). |

All four synthetic concentration curves were removed; they were not measured
time-series data. A stale non-isomeric 250 mg/1 µg/mL terbinafine duplicate was
removed from each exploratory MMPK CSV and the SHA inventory was refreshed.
The reference pool and calibration train set remain 127 and 38. Recalculation
leaves the nominal 90% Meta residual half-width at **10.24×** and coverage on
the repeatedly used development cohort at **79/86 = 91.86%**. Neither is an
external result under the [V1 protocol](external_holdout_v1_protocol.md).
