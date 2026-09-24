# DrugBank-arm development reference follow-up — 2026-09-24

This review checked four scored oral parent-drug references against original
regulatory documents before looking at prediction error. It changes labels,
not fitted model weights. The 107-compound development split has already been
used for system selection, so these results are not external validation.

| Compound | Primary-source finding | Decision |
| --- | --- | --- |
| Clonidine | The [US clonidine hydrochloride tablet label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=26a63aea-3e7f-554d-e063-6394a90a8502&type=display) says a 0.1 mg tablet contains 0.087 mg clonidine free base and gives terminal elimination half-life 12–16 h; about 20 min is the IV distribution half-life. The DrugBank-cited 400.72 pg/mL after 100 mcg could not be traced to an exact original PK arm. | Convert recorded dose to 0.087 mg parent, remove the incorrect 0.33 h point and synthetic curve, and quarantine Cmax until the original arm is recovered. |
| Pindolol | The [Health Canada APO-PINDOL monograph](https://pdf.hres.ca/dpd_pm/00047827.PDF) reports mean parent Cmax 33.1 ± 5.2 ng/mL after one 5 mg dose. Parent half-life is 3–4 h; 8 h describes inactive polar metabolites. | Keep 5 mg / 0.0331 mg/L parent Cmax; remove the wrong half-life and synthetic curve. |
| Sumatriptan | The [US tablet label](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=284ee7ac-759e-b592-e063-6394a90a78ba) expresses the tablet strength as sumatriptan base and reports mean Cmax 18 ng/mL (range 7–47) after oral 25 mg. The label-wide half-life of about 2.5 h and bioavailability of about 15% are general approximations. | Replace secondary 16.5 ng/mL with 0.018 mg/L; retain the explicitly qualified general values and remove the synthetic curve. |
| Bexagliflozin | The [FDA NDA 214373 integrated review](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2023/214373Orig1s000IntegratedR.pdf) identifies a fasted **single 20 mg tablet** with geometric mean parent Cmax **134 ng/mL** (CV 43%). The [tablet label](https://www.dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=7f0ad5d2-3509-4057-904c-d27993de7408&type=display) gives about 12 h terminal half-life generally. | Keep the exact scored Cmax, specify the primary source, and remove the synthetic curve. |

After regeneration, the scored development cohort is **N=85**: Meta AAFE
**2.9093** (conditional compound-bootstrap 95% CI **2.3885–3.5929**), Engine
**3.9571**, direct ML **3.3056**. The descriptive in-domain slice is N=69,
Meta AAFE **2.8998**. The nominal 90% development-residual half-width is
**10.24×**, with **78/85 (91.8%)** coverage on this consumed cohort. These
conditional statistics do not measure performance on new compounds or support
an 85/100 quality judgment.
