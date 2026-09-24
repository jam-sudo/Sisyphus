# DrugBank-arm development reference follow-up — 2026-09-24

This review checked eight scored oral parent-drug references against original
regulatory documents before looking at prediction error. It changes labels,
not fitted model weights. The 107-compound development split has already been
used for system selection, so these results are not external validation.

| Compound | Primary-source finding | Decision |
| --- | --- | --- |
| Clonidine | The [US clonidine hydrochloride tablet label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=26a63aea-3e7f-554d-e063-6394a90a8502&type=display) says a 0.1 mg tablet contains 0.087 mg clonidine free base and gives terminal elimination half-life 12–16 h; about 20 min is the IV distribution half-life. The DrugBank-cited 400.72 pg/mL after 100 mcg could not be traced to an exact original PK arm. | Convert recorded dose to 0.087 mg parent, remove the incorrect 0.33 h point and synthetic curve, and quarantine Cmax until the original arm is recovered. |
| Pindolol | The [Health Canada APO-PINDOL monograph](https://pdf.hres.ca/dpd_pm/00047827.PDF) reports mean parent Cmax 33.1 ± 5.2 ng/mL after one 5 mg dose. Parent half-life is 3–4 h; 8 h describes inactive polar metabolites. | Keep 5 mg / 0.0331 mg/L parent Cmax; remove the wrong half-life and synthetic curve. |
| Sumatriptan | The [US tablet label](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=284ee7ac-759e-b592-e063-6394a90a78ba) expresses the tablet strength as sumatriptan base and reports mean Cmax 18 ng/mL (range 7–47) after oral 25 mg. The label-wide half-life of about 2.5 h and bioavailability of about 15% are general approximations. | Replace secondary 16.5 ng/mL with 0.018 mg/L; retain the explicitly qualified general values and remove the synthetic curve. |
| Bexagliflozin | The [FDA NDA 214373 integrated review](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2023/214373Orig1s000IntegratedR.pdf) identifies a fasted **single 20 mg tablet** with geometric mean parent Cmax **134 ng/mL** (CV 43%). The [tablet label](https://www.dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=7f0ad5d2-3509-4057-904c-d27993de7408&type=display) gives about 12 h terminal half-life generally. | Keep the exact scored Cmax, specify the primary source, and remove the synthetic curve. |
| Indomethacin | DrugBank reports 1.54 ± 0.76 mcg/mL after a single 25 mg oral dose in fasting subjects, but the original study could not be retrieved. The [US capsule label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=5b05b8bc-4df0-4175-bf53-72129fb21b50) independently gives **about 1 mcg/mL** after a single 25 mg capsule. | Retain 1.54 mg/L as a lower-confidence **silver** reference, record the primary-label discrepancy, and remove the synthetic curve. Original-study retrieval remains open. |
| Ketoconazole | The [US tablet label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=369f31da-d3c3-49b8-956f-f70598760164) reports mean Cmax **about 3.5 mcg/mL** after a single **200 mg oral tablet with a meal**. | Replace unsourced 3.0 with 3.5 mg/L, record fed context, and remove the synthetic curve. |
| Levofloxacin | [Table 10 of the US tablet label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=f14cf651-a213-4a4f-adb9-f628562af27c&type=display) gives **5.1 ± 0.8 mcg/mL** after a single **500 mg oral tablet** in healthy males. Its **6.2 ± 1.0 mcg/mL** value is after a **500 mg IV infusion**. | Correct the route collision from 6.2 to 5.1 mg/L and remove the synthetic curve. |
| Metronidazole | The [Pfizer Canada product monograph](https://webfiles.pfizer.com/file/a1878dae-540b-4649-8912-3273916d4577) reports peak parent plasma concentration **about 13 mg/L** after a single **500 mg oral dose**; Figure 1 uses nine female subjects. | Keep 13 mg/L, specify original source and population, and remove the synthetic curve. |

After regeneration, the scored development cohort is **N=85**: Meta AAFE
**2.9079** (conditional compound-bootstrap 95% CI **2.3839–3.5930**), Engine
**3.9551**, direct ML **3.3040**. The descriptive in-domain slice is N=69,
Meta AAFE **2.8981**. The nominal 90% development-residual half-width is
**10.24×**, with **78/85 (91.8%)** coverage on this consumed cohort. These
conditional statistics do not measure performance on new compounds or support
an 85/100 quality judgment.
