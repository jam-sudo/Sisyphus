# Development generic-label arm follow-up — 2026-09-24

Eight scored rows had the generic provenance “FDA label (analytical model from
PK params).” Their original labels were reviewed by drug name and source text,
before inspecting prediction error. This is a reference-data repair, not an
external validation or fitted-model change.

| Record | Original label finding | Action |
| --- | --- | --- |
| Azithromycin | The [tablet label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=7e7c2f49-449c-4ba2-98ae-0ff0c040614c) reports Cmax **0.5 mcg/mL** and AUC0–72 **4.3 mcg·h/mL** after one fasted 500 mg oral dose in 36 healthy men. | Keep Cmax; correct the unsupported AUC **17.4→4.3 mg·h/L** and remove the synthetic curve. |
| Ciprofloxacin | [Tables 12–13](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=888dc7f9-ad9c-4c00-8d50-8ddfd9bd27c0) give single oral 500 mg Cmax **2.4 mcg/mL** and AUC **11.6 mcg·h/mL**. The former **2.97 mcg/mL** is after 500 mg every 12 hours at steady state. | Correct Cmax **2.97→2.4 mg/L** and AUC **250→11.6 mg·h/L**; remove the synthetic curve. |
| Colchicine | [Table 5](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=c7d013b7-0a67-0953-e053-2a95a90ac47c&type=display) confirms **2.5 ng/mL** after one fasted 0.6 mg tablet in 13 healthy adults. Its single-dose half-life cell is blank; the local **24.92 h** came from a different cohort. | Keep Cmax; remove the unmatched half-life and synthetic curve. |
| Diclofenac | The [sodium delayed-release tablet label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=a4fd2d68-019f-4c89-8923-4e61262f6eee) gives approximately **1.0 mcg/mL** after fasted 25 mg and approximately **2 h** parent half-life. PubChem gives [296.1 g/mol parent](https://pubchem.ncbi.nlm.nih.gov/compound/diclofenac) and [318.1 g/mol sodium salt](https://pubchem.ncbi.nlm.nih.gov/compound/5018304). | Correct Cmax **0.5→1.0 mg/L**, half-life **1→2 h**, and input dose to **23.27 mg parent equivalent**; record delayed-release context and remove the synthetic curve. |
| Isosorbide mononitrate | The [label table](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=897a0327-25d4-4769-bd6a-0e674264017f) gives **1242–1534 ng/mL** as a range across single-dose 60 mg oral-solution studies; a 60 mg extended-release tablet gives only **424–541 ng/mL**. The local 1.242 mg/L selected the range's lower endpoint without an identified study arm. | Quarantine the Cmax and synthetic curve pending a specific original arm. |
| Losartan | [Table 2](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=d67926b0-c22b-4803-b55d-c98a81389449) assigns **224 ng/mL** parent Cmax to 50 mg losartan potassium **once daily for 7 days** in 12 hypertensive adults. | Quarantine this repeated-dose patient arm and its synthetic curve from single-dose scoring. |
| Moxifloxacin | [Table 7](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=45ce14e5-98b8-296b-e063-6294a90a769b) gives **3.1 ± 1 mg/L** after one oral 400 mg dose in 372 healthy subjects. **4.1 mcg/mL** describes a separate Japanese steady-state cohort. | Correct Cmax **4.1→3.1 mg/L**; remove the unmatched 14 h point and synthetic curve. |
| Zolpidem | The [extended-release label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=35f6c1ab-a61c-4625-b932-11a7483f1d84) confirms single 12.5 mg tartrate-tablet parent Cmax **134 ng/mL**, AUC **740 ng·h/mL**, and half-life **2.8 h** in healthy men. It explicitly equates 12.5 mg tartrate with **10 mg zolpidem base**. | Keep observed PK, correct model input dose **12.5→10 mg parent**, record extended-release context, and remove the synthetic curve. |

After regeneration, the repeatedly used development benchmark scores **N=79**:
Meta AAFE **2.9320** (conditional compound-bootstrap 95% CI
**2.3897–3.6737**), Engine **3.9716**, and direct ML **3.2605**. The descriptive
in-domain Meta slice is **2.9497** (N=65). The paired Meta/ML AAFE ratio is
**0.8992** (95% conditional CI **0.7975–1.0106**), which includes 1. The
development-residual half-width is **10.24×** and covers **72/79 (91.1%)** on
this consumed cohort. None of these estimates establishes independent
generalization or superiority.
