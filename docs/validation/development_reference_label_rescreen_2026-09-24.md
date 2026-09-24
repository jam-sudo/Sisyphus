# Development training reference label rescreen — 2026-09-24

Six `clinical_pk.json` training names with generic “FDA label” provenance were
checked against original studies or regulatory labels before inspecting model
residuals. These are reference-data corrections, not a fitted-model improvement
or independent validation.

| Record | Source finding and disposition |
| --- | --- |
| Cyclobenzaprine | The [tablet label](https://www.dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=592871f6-0eb2-4624-9e10-bd8bf767c002) identifies **25.9 ng/mL** as the peak after **10 mg three times daily at steady state**, not a single dose. [Brioschi et al., reference arm](https://doi.org/10.1155/2013/281392) measured **7.0 ng/mL** parent Cmax and **199.4 ng·h/mL** AUC after one fasted 10 mg immediate-release tablet in 23 healthy adults. The paper does not state the salt; applying the marketed [hydrochloride tablet](https://www.dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=592871f6-0eb2-4624-9e10-bd8bf767c002) basis is an **inference**. On that basis, parent/salt molecular weights [275.39](https://www.deadiversion.usdoj.gov/drug_chem_info/cyclobenzaprine.pdf)/311.9 give **8.829433 mg** base. |
| Desloratadine | The [label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=2a90b899-7746-43dc-ac8a-e754428eb30c) reports the former **4 ng/mL** after repeated 5 mg dosing. [Ponnuru et al., Table 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC5760887/) gives reference-product parent Cmax **2058.1 pg/mL = 0.0020581 mg/L** after one fasted 5 mg tablet in 35 healthy men, with AUC0–∞ **0.0540256 mg·h/L**. |
| Entacapone | The [label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ee02a04a-4b42-4cfa-8d9d-b1459c0ee9fd) confirms the existing approximate parent Cmax **1.2 μg/mL** after a single 200 mg dose. Its 0.4–0.7 h beta-phase half-life is a range, with a longer 2.4 h gamma phase, so the old 0.4 h point value was removed. |
| Fluconazole | The [label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=5cbd78cd-2cd7-4882-9902-bdb7812fbc6b) gives **6.72 μg/mL** after a single fasted **400 mg** dose, not 200 mg. Corrected dose. Deleted the old **76.4 μg·h/mL AUC**, which belongs to an elderly **50 mg** study in the same label. |
| Gabapentin | The previous **2.99 mg/L** had no traceable matching label arm. An [original 300 mg single-dose crossover study, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC4159312/) reports reference-product parent Cmax **3223.69 ng/mL = 3.22369 mg/L** and AUC0–∞ **23.57403 mg·h/L** in 37 fasted healthy adults. |
| Glycopyrrolate | The [FDA clinical review, Table 2](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2010/022571Orig1s000MedR.pdf) associates **0.318 ng/mL** with **10 mL of 1 mg/5 mL oral solution = 2 mg** under fasting, not 1 mg. The [label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=7049feaf-004a-481f-b129-996b21c3170a) identifies glycopyrrolate as the bromide, matching the stored salt SMILES. Corrected dose and unqualified AUC to the same-arm AUC0–∞ **1.81 ng·h/mL**. |

Synthetic analytical concentration curves were removed from all six entries;
they were not observed time-series data. The pool remains 127 Cmax rows with
38 training references. Recalibration on this partially in-sample training
set yields a nominal 90% Meta half-width of **10.24×**; the repeatedly used
86-compound development benchmark shows **79/86 = 91.86%** coverage. Neither
number establishes prospective coverage. The [V1 independent validation
protocol](external_holdout_v1_protocol.md) remains unmet because no untouched,
source-adjudicated external cohort exists.

A supervised read-only Claude Code source re-read (`run_049f1a65ffe4`,
`ctx_6b934194c915`) confirmed the six Cmax values and found the author-name
error corrected above, the cyclobenzaprine salt-basis inference, and the
entacapone half-life range issue. This is a second AI check, not the protocol's
independent two-person human source verification.
