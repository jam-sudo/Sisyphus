# Development salt-equivalent reference follow-up — 2026-09-24

Two scored development references still mixed administered salt mass with the
canonical parent SMILES dose. Their source evidence was checked before using
prediction error to decide the correction.

| Drug | Source finding | Reference action |
| --- | --- | --- |
| Acamprosate | The [DailyMed label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ae6e79c0-a307-4888-b647-1b7be4cb9127) states that each 333 mg calcium-salt tablet contains the equivalent of 300 mg acamprosate. A [2010 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC2853976/) reports mean parent Cmax 180 ng/mL after one oral dose of two 333 mg tablets in healthy volunteers. The exact original numeric study arm and meal state remain unidentified. | Use **600 mg parent-equivalent** model input instead of 666 mg salt, retain **0.18 mg/L** observed Cmax, and mark the reference **silver** because its numeric peak is review-reported. |
| Phenytoin | [Gogtay et al. 2013, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3825999/) directly reports an Eptoin reference arm: single 300 mg oral phenytoin sodium in 20 fasting healthy men, parent Cmax **2.32 ± 1.11 µg/mL** and AUC0–∞ **108.99 ± 91.45 µg·h/mL**. [PubChem parent](https://pubchem.ncbi.nlm.nih.gov/compound/1775) and [sodium salt](https://pubchem.ncbi.nlm.nih.gov/compound/Phenytoin-Sodium) molecular weights are 252.27 and 274.25 g/mol. | Use **275.956 mg parent-equivalent** model input (`300 × 252.27 / 274.25`), **2.32 mg/L** observed Cmax, and **108.99 mg·h/L** AUC. Remove the previous untraceable 5 mg/L range midpoint, unmatched generic half-life, and synthetic concentration curve. |

The public-profile cache was regenerated without changing fitted models.
On the repeatedly accessed **N=79 development** cohort, Meta AAFE is
**2.9492** (conditional compound-bootstrap 95% CI **2.4077–3.6756**),
Engine **3.8794**, direct ML **3.3044**, and descriptive in-domain Meta
**3.0375** (N=65). The paired Meta/ML ratio is **0.8925**
(95% CI **0.7890–1.0048**), including 1. The nominal 90% development-residual
interval has a **10.24×** half-width and covers **72/79 (91.1%)** of the
consumed cohort. The previous Meta AAFE was 2.9276; the worsening is a
reference correction, not a model change. Neither score nor interval is an
independent external validation result.
