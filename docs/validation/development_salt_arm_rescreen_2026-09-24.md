# Development salt and source-arm rescreen — 2026-09-24

Six of the 79 scored development references were rechecked against primary
product labels or study reports. The benchmark still uses parent SMILES and
parent-equivalent input dose. No fitted model was changed.

| Drug | Source finding | Scored reference |
| --- | --- | --- |
| Carbinoxamine | The [DailyMed tablet/solution label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=9d11b197-cfdc-4b9b-9935-2920ccb6e522) specifies 4 mg **maleate** per 5 mL and describes a single 8 mg solution dose with approximate parent-plasma Cmax 24 ng/mL. Interpreting the study's 8 mg as the marketed maleate strength is an inference; exact N and meal state are not given. Parent/salt masses are [290.79](https://pubchem.ncbi.nlm.nih.gov/compound/2564)/406.86 g/mol. | 5.717741 mg parent; 0.024 mg/L. Tier reduced from gold to silver. |
| Montelukast | [Cheng et al. 1996](https://pubmed.ncbi.nlm.nih.gov/8692739/) report 350 ng/mL for the **female** 10 mg oral arm (N=6); males reached 385 ng/mL; female meal state is not specified in the abstract. The [SINGULAIR label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=482dcc92-b47f-4ea6-854a-f5ac2aea7842) states a 10 mg tablet contains 10.4 mg sodium salt equivalent to **10 mg parent**. | 10 mg parent; 0.350 mg/L. Numeric reference unchanged; arm identity corrected. |
| Pravastatin | The [pravastatin sodium label](https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?setid=cf6f7031-33b9-40d3-bc87-32f1b6af98f2&type=pdf) explicitly equates a 20 mg sodium tablet to **19.01 mg parent** and reports **26.5 ng/mL geometric mean** parent Cmax after a fasted 20 mg dose, rather than the previous untraceable 25 ng/mL. | 19.01 mg parent; 0.0265 mg/L. |
| Quizartinib | [Li et al. 2020, Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7027461/) give a fasted N=34 single 30 mg **dihydrochloride** tablet equivalent to **26.5 mg free base**. Parent-plasma Cmax is **102.0 ng/mL arithmetic mean**; the former 99.3 ng/mL is the geometric mean. The prior author attribution to Faucette was wrong. | 26.5 mg parent; 0.102 mg/L. |
| Ranitidine | The [DailyMed label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=a4be3f71-31f6-4559-82f0-aa11168af779&type=display) gives only a **440–545 ng/mL range** after 150 mg, making the former 440 ng/mL an arbitrary lower-endpoint point label. [Gschwend et al. 2007](https://pubmed.ncbi.nlm.nih.gov/17688076/) instead report **450.6 ng/mL** parent-plasma Cmax for the 150 mg reference HCl film-coated tablet in healthy men in a single-dose crossover. The tablet's 168 mg HCl is equivalent to 150 mg parent. Meal state and N are absent from the abstract. | 150 mg parent; 0.4506 mg/L. Generic half-life and bioavailability values not linked to this arm were removed. |
| Selegiline | The [FDA Zelapar clinical pharmacology review, study Z/SEL/96/008](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2006/021479s000_ClinPharmR_P2.pdf) reports **3.093 ng/mL mean parent-plasma Cmax** after one fasted 10 mg swallowed Eldepryl dose (two 5 mg HCl tablets) in 12 healthy volunteers. The previous approximately 1 ng/mL claim was not traceable to that arm. Parent/HCl masses are [187.28](https://pubchem.ncbi.nlm.nih.gov/compound/26757)/[223.74](https://pubchem.ncbi.nlm.nih.gov/compound/26758) g/mol. | 8.370430 mg parent; 0.003093 mg/L. |

The carbinoxamine and selegiline upstream curated/extraction rows were aligned
with these corrections. The incompatible selegiline extraction was marked
superseded. The repeatedly used N=79 public-profile development cache now
reports Meta AAFE **2.9040** (conditional compound-bootstrap 95% CI
**2.3784–3.6157**), Engine **3.8150**, direct ML **3.2381**, and in-domain
Meta **2.9741** (N=65). The paired Meta/ML ratio remains **0.8968** (CI
**0.7922–1.0106**). The nominal 90% development-residual interval still has
10.24-fold half-width and covers 72/79. The change from 2.9312 is reference
correction, not model improvement or independent validation.
