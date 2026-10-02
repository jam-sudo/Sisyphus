# Three development-reference arm checks (2026-09-24)

These rows were selected for source review from incomplete arm descriptions, before inspecting prediction errors. They remain part of a repeatedly accessed development set, not independent validation.

| Drug | Primary evidence and decision |
| --- | --- |
| Alvimopan | The [DailyMed label, §12.3](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=59ef406b-79db-4efd-b993-08b937ac682b) assigns parent-plasma Cmax **10.98 ± 6.43 ng/mL** to **12 mg twice daily for five days**. The scored 12 mg single-dose label was therefore misclassified. Quarantined from the scored benchmark and cleared the unsupported 10.5 ng/mL “single-dose” FDA extraction. A separately verified, training-disjoint single-dose arm would be needed to restore it. |
| Donepezil | [Tiseo et al. 1998](https://pubmed.ncbi.nlm.nih.gov/9839768/) report parent-plasma Cmax **7.7 ± 1.2 ng/mL** after one oral **5 mg donepezil HCl** dose in **11 age/sex-matched healthy controls**. The [ARICEPT 5 mg SmPC, §2](https://www.medicines.org.uk/emc/product/3776/smpc) states that each 5 mg hydrochloride tablet provides **4.56 mg donepezil free base**. Retained the observed peak and changed the model input to 4.56 mg because its SMILES is the free base. |
| Fruquintinib | The [original mass-balance study](https://pubmed.ncbi.nlm.nih.gov/28730290/) reports parent-plasma mean Cmax **113 ng/mL** after one **5 mg radiolabeled oral suspension** dose in **six fasted healthy Chinese men**. Retained the value and identified the original formulation in the source field; it is not the marketed capsule arm. |

With fitted models unchanged, the scored development set moves **N=75 → 74**. Meta AAFE **2.8626 → 2.8657** (conditional bootstrap CI **2.3255–3.5723**), Engine **3.8887**, direct ML **3.2511**, and paired Meta/ML ratio **0.8815** (CI **0.7686–1.0043**, crossing 1). The residual band remains 10.24-fold and covers **68/74** (91.9%) on this consumed set. These changes correct reference integrity; they do not measure an independently validated accuracy gain.
