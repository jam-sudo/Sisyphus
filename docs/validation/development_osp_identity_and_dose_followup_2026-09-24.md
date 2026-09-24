# OSP observation identity and dose audit — 2026-09-24

An audit of all 21 selected OSP source rows against their original model JSON
found that the old extractor filtered species, route, food state and plasma
compartment, but did **not** check the observed molecule or whether the profile
followed a single administration. OSP PBPK repos include co-modeled drug and
metabolite profiles. The model repo name is not an observed-analyte identifier.

| Selected row | Original JSON finding | Scored-reference action |
| --- | --- | --- |
| Cabozantinib 600 mg / 9.8 mg/L | [Cabozantinib-Model JSON](https://github.com/Open-Systems-Pharmacology/Cabozantinib-Model/blob/main/Cabozantinib.json): the Eon Labs 1997 record names **rifampicin**, 600 mg, in its `Molecule` and profile title. | Replace with [FDA NDA 208692 clinical-pharmacology review Table 12](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2016/208692Orig1s000ClinPharmR.pdf), study XL184-010: single oral 140 mg cabozantinib free-base-equivalent **capsule**, mean parent plasma Cmax 554 ng/mL and AUC0–∞ 58,300 ng·h/mL (N=72). The [published study](https://pubmed.ncbi.nlm.nih.gov/27139820/) confirms tablet and capsule are not Cmax-bioequivalent, so formulation remains explicit. |
| Ruxolitinib 15 mg / 64.889 ng/mL | [Ruxolitinib-Model JSON](https://github.com/Open-Systems-Pharmacology/Ruxolitinib-Model/blob/master/Ruxolitinib.json): Bornemann 1986 is a **midazolam** 15 mg fasted profile. | Replace with [FDA NDA 202192 clinical-pharmacology review Table 6](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2011/202192Orig1s000ClinPharmR.pdf), study INCB 18424-138: one oral 25 mg tablet in 47 healthy adults, parent mean Cmax 1510 nM, AUC0–∞ 5320 nM·h, half-life 2.6 h. Using [parent molecular weight 306.4 g/mol](https://pubchem.ncbi.nlm.nih.gov/compound/Ruxolitinib), these are 0.4627 mg/L and 1.630 mg·h/L. The [tablet label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=f1c82580-87ae-11e0-bc84-0002a5d5c51b) states that 25 mg is free-base-equivalent. |
| Erythromycin 500 mg / 1.7864 mg/L | [Erythromycin-Model JSON](https://github.com/Open-Systems-Pharmacology/Erythromycin-Model/blob/master/Erythromycin-Model.json): the selected Olkkola 1993 individual profile followed **500 mg every 8 h** across multiple days. | Replace with the same OSP repo's DiSanto 1981 Figure 1 aggregate, single fasted 500 mg enteric-coated erythromycin-base tablet in 21 adults, digitized mean-profile maximum **1.014211 mg/L**. The [original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC181662/) confirms the single-dose arm. Retain silver tier because a profile maximum is not a mean of individual Cmax values. |

The source-catalog rows for cabozantinib and ruxolitinib were removed from
`osp_observed.json`; erythromycin was updated. Three other **unscored**
source-catalog rows were also removed: felodipine selected a DDI treatment
profile at hour 73, itraconazole selected **hydroxy-itraconazole** after repeated
dosing, and verapamil selected **norverapamil**. Their main reference records
were already unverified. The extractor now requires exact observed parent name
and a single numeric administration time. A focused test checks both
exclusion conditions.

After regenerating the public-profile **development** cache, N=79:
Meta AAFE **2.9276** (conditional compound-bootstrap 95% CI
**2.3845–3.6603**), Engine **3.9184**, direct ML **3.2802**.
The descriptive in-domain Meta slice is **3.0105** (N=65).
The paired Meta/ML AAFE ratio is **0.8925** (conditional 95% CI
**0.7890–1.0048**), still including 1. The nominal 90% development-residual
half-width stays **10.24×**, covering **72/79 (91.1%)** of this consumed
cohort. Fitted models are unchanged; none of these estimates is an
independent external accuracy result.
