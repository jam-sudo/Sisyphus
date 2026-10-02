# Generic-label training reference follow-up — 2026-09-24

Three more training Cmax records were adjudicated by original or regulatory
sources without selecting by model error. No fitted model weights or development
benchmark labels changed.

| Record | Evidence and correction |
| --- | --- |
| Abacavir | The previous **3.67 μg/mL** is the [steady-state dolutegravir Cmax](https://www.medicines.org.uk/emc/product/3318/smpc) listed next to abacavir in combination-product information. The [abacavir tablet label](https://www.dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=5409c8ed-17c3-454f-bb88-d597ba8f84ec&type=display) instead gives parent Cmax **4.26 μg/mL**, AUC0–∞ **11.95 μg·h/mL**, and single-dose t½ **1.54 h** after **600 mg** abacavir. The sulfate tablet strength is expressed as parent equivalent. Corrected the two wrong PK fields, and restored the marketed **(1S,4R)** structure using [PubChem CID 441300](https://pubchem.ncbi.nlm.nih.gov/compound/Abacavir). Removed the spurious 600 mg/3.67 mg/L duplicate from both exploratory MMPK CSVs; the separate sourced 600 mg row remains. |
| Pantoprazole | The [delayed-release tablet label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=491102c1-f04b-4b7b-e054-00144ff8d46c) supports parent **2.5 μg/mL** Cmax and **4.8 μg·h/mL** AUC at **40 mg parent-equivalent** in extensive metabolizers. Its terminal half-life is approximately **1 h** there; **3.5 h** is at the lower edge of the poor-metabolizer range. Corrected the half-life in the reference and exploratory CSVs and labeled the latter's formulation as enteric-coated. |
| Propranolol | The former **80 ng/mL** lacked a matching single-dose label arm. [Australian TGA study 1012A, Table 2](https://www.tga.gov.au/sites/default/files/auspar-propranolol-hydrochloride-150819-cer.pdf) reports arithmetic mean parent plasma Cmax **49.5 ng/mL**, AUC0–∞ **328.2 ng·h/mL**, and t½ **4.41 h** after a single fasted **80 mg base-equivalent oral solution** in 12 healthy men. Dose basis and formulation are explicit. Removed unsupported **90% bioavailability**; the [tablet label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=1791ef4d-b609-41c8-b4ef-15bff752ddd7&type=display) instead says systemic availability averages approximately **25%**, while **90%** refers to plasma protein binding. |

All three synthetic concentration curves were removed because they are not
observed measurements. Both corrected exploratory CSV hashes were updated in
`training_membership_sources_v1.json` to keep the provenance contract current.
The recalculated development residual interval remains partially in sample and
does not satisfy [V1 external validation](external_holdout_v1_protocol.md).
