# Additional development training arm screen — 2026-09-24

Five remaining training names were checked against original studies or regulatory
labels without selecting by model error. These changes repair dose, formulation,
population, and analyte provenance; fitted production models and the 86-compound
development benchmark were not changed.

| Record | Source finding | Correction |
| --- | --- | --- |
| Aspirin | [Voelker and Hammer, Table 2](https://doi.org/10.1007/s10787-011-0099-z) reports mean **4.4 μg/mL parent ASA** after a single **500 mg regular tablet** in healthy adults (26 study completers), with AUC0–∞ 6.5 μg·h/mL and t½ 0.54 h. The former 1.98 μg/mL had no traceable same-arm source. Meal timing is not stated. | Re-sourced; downgraded to silver for unknown meal timing and removed synthetic curve. |
| Metaxalone | [DailyMed Table 1](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=168c539d-3637-4a43-a8b6-cd8e82cf6342) gives **983 ng/mL** after a single fasted **400 mg** tablet in 42 healthy volunteers. The old **1816 ng/mL** is the **800 mg** arm. | Corrected Cmax, same-arm AUC and half-life; removed synthetic curve. |
| Pregabalin | [Filipe et al., Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC4488182/) gives reference Lyrica **300 mg** immediate-release capsule under fasting conditions in 39 healthy adults: geometric mean parent Cmax **7420.08 ng/mL**, AUC0–∞ **55236.37 ng·h/mL**, t½ **6.49 h**. The previous 2.0 μg/mL has no matching 300 mg single-dose label arm. | Re-sourced and corrected the input's pregabalin stereochemistry; removed synthetic curve. |
| Sertraline | A [DailyMed label](https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?setid=59287bc1-ac6f-47d2-84ba-b52f32f5894f&type=pdf) identifies the former **165 ng/mL** with pediatric **200 mg/day repeated dosing**. [Abbas et al., Table 3](https://doi.org/10.1002/cpdd.749) reports geometric mean parent plasma Cmax **11.39 ng/mL** in 52 healthy adults after the first **50 mg** dose. Regulatory [tablet strength](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=61835f25-bfe4-49ac-9dbf-3d6387866e78) is expressed as sertraline base equivalent. Pre-dose fast is not established; lunch was served about 4 h after morning dosing. | Re-sourced; downgraded to silver, corrected stereochemistry, removed unrelated half-life and synthetic curve. Removed one unsupported 0.165 mg/L duplicate from each exploratory MMPK CSV; the separate 50 mg / 0.0106 mg/L row remains. |
| Tramadol | [DailyMed Table 1](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=8fbcb289-57fd-42ba-a146-25ea55a80a08) gives racemic parent Cmax **308 ng/mL** after a single **100 mg tramadol HCl** oral dose in healthy adults. The prior 64.3 ng/mL was from a 37.5 mg tramadol/acetaminophen combination enantiomer arm. The [label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=8fbcb289-57fd-42ba-a146-25ea55a80a08) gives HCl molecular weight 299.8; parent is 263.38. | Converted the administered HCl dose to **87.852 mg parent base**, corrected Cmax and t½, retained label bioavailability, removed synthetic curve. |

There are still **127** usable Cmax reference rows and **38** training records
in the development residual calculation. The nominal 90% Meta interval now has
a **6.94×** half-width and covers **71/86 = 82.56%** of the repeatedly used
development benchmark. This is descriptive and below the V1 lower coverage
gate even before accounting for dependence. The interval is partly in sample,
and no new blinded external evaluation occurred. See the
[V1 protocol](external_holdout_v1_protocol.md).
