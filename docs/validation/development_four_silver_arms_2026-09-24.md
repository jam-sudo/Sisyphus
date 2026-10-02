# Four legacy silver PK arms: original-source check — 2026-09-24

These rows were selected for their generic source strings before inspecting model
errors. All conclusions concern the repeatedly accessed development set.

| Arm | Original-source finding | Action |
|---|---|---|
| Penicillamine, 250 mg | The [Cuprimine label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=80e736d3-2017-4d68-94b4-38255c3c59c6) and [Netter et al. review](https://pubmed.ncbi.nlm.nih.gov/3319347/) give an approximate **1–2 mg/L** peak range, not a 1.5 mg/L cohort mean. The separate [Osman et al. 1983](https://doi.org/10.1038/clpt.1983.63) fasted 500 mg arm reports a directly observed mean 3.05 mg/L, but that arm is already in the exploratory MMPK training CSV. | Quarantine the invented midpoint from the scored set and the curated source. Quarantine the FDA extraction's 2.0 mg/L range endpoint. Do not substitute the training arm as a nominal holdout. |
| Lenacapavir, 300 mg | [Antimicrobial Agents and Chemotherapy 2024, Table 2](https://doi.org/10.1128/aac.01344-23) reports geometric mean **23.4 ng/mL** in **10 normal-hepatic-function matched controls** after one oral 300 mg dose. The separate normal-renal-function control arm is 19.7 ng/mL. The former FDA extraction's **73.8 ng/mL** is unsupported as a 300 mg single oral Cmax; that number appears in the [Yeztugo label, Table 9](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=1c241af1-ce62-4b0a-9eb7-f6b626174f01&type=display) as a population-PK exposure over the first six months of a combined oral/subcutaneous regimen. | Keep the scored 0.0234 mg/L arm, identify the exact control group, and quarantine the mismatched FDA extraction. |
| Lorlatinib, 100 mg | [Hibma et al., Cancer Chemotherapy and Pharmacology 2022, Table 2](https://doi.org/10.1007/s00280-021-04368-1) gives geometric mean **501.3 ng/mL** in **11 healthy men** after one oral 100 mg dose as four 25 mg development tablets, not the final commercial formulation. The prior Chen/CPT Pharmacometrics attribution is wrong. The separate [Lorbrena label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=004f93d7-a1cd-4b67-9207-31cdcb5c5976) gives **577 ng/mL at steady state**; it is not this arm. | Keep 0.5013 mg/L, correct the original-study citation and formulation. |
| Vonoprazan, 20 mg | [Sakurai et al., Clinical and Translational Gastroenterology 2015, Table 2](https://doi.org/10.1038/ctg.2015.18) reports arithmetic mean **25.0 ± 5.6 ng/mL** for **7 Japanese healthy men** after one 20 mg oral dose. The prior Clinical Pharmacology & Therapeutics journal attribution is wrong. | Keep 0.025 mg/L, correct the citation and arm details. |

The historical FDA extraction builder contains other superseded source rows. It
now refuses to overwrite the newer adjudicated JSON. After this reference-only
change, the scored development cohort is **N=76**. Meta AAFE is **2.9539**
(conditional compound-bootstrap 95% CI **2.3914–3.7036**), Engine **3.8928**,
direct ML **3.3483**, and in-domain Meta **3.0559** (N=62). The paired
Meta/direct-ML ratio is **0.8822** (95% CI **0.7730–1.0041**). None of these
numbers is an independent external validation result.
