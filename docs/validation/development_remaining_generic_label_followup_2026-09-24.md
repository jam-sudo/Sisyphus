# Remaining generic-label development arms — 2026-09-24

The last six scored references whose provenance included “FDA label
(analytical model from PK params)” were checked by source arm before comparing
prediction errors. The reference file had attached identical normalized
exponential concentration curves to all six; these were model-generated, not
observed, so they were removed. This is a development-label repair, not an
independent validation or fitted-model change.

| Record | Primary-source finding | Correction |
| --- | --- | --- |
| Budesonide | The [enteric-coated capsule label](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=4a8e79df-3946-4e9a-b214-e9fdd824b565&version=4) reports parent plasma Cmax 1.50 ng/mL and AUC 14.13 ng·h/mL after one fasted oral 9 mg dose in healthy adults. The old AUC 17.78 belongs to pediatric patients after seven days; 1.9 h is pediatric intravenous half-life. | Retain Cmax, correct AUC to 0.01413 mg·h/L, remove unmatched half-life. |
| Dalfampridine | The [extended-release tablet label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=302c8270-4711-4ca4-bee8-be0738fbe094&type=display) reports parent Cmax 42.7 ng/mL for a 10 mg **oral solution** comparator. The old 7.6 h half-life belongs to a sulfate metabolite. The 96% bioavailability is relative tablet/solution exposure, not absolute solution bioavailability. | Retain and explicitly identify the solution Cmax; remove unrelated half-life and bioavailability. This formulation differs from the marketed extended-release tablet. |
| Dapagliflozin | The local [OSP extraction](../../data/reference/osp_observed.json) traces 121.809 ng/mL to the digitized mean-profile maximum of a fasted, single-dose 10 mg healthy-adult arm attributed to Kasichayanula 2011a. The [original study abstract](https://pubmed.ncbi.nlm.nih.gov/21435141/) confirms the 14-subject crossover design, but does not establish that this profile maximum equals the mean of individual Cmax values. The 12.9 h half-life was a generic label value, not traced to this OSP arm. | Retain the explicitly profile-derived silver development peak and mark its estimator limitation; remove unmatched half-life. Confirm against the original study table before any higher-confidence promotion. |
| Etodolac | The [capsule label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=526a2ad8-547c-480f-e054-00144ff88e88&type=display) gives approximately 14 ± 4 µg/mL **plasma** Cmax after a single 200 mg capsule, and pooled healthy-adult half-life 6.4 h. The old 15.9 µg/mL came from a [serum study](https://pubmed.ncbi.nlm.nih.gov/2525981/). Label “100%” is relative to oral solution. | Replace Cmax 15.9→about 14 mg/L; retain pooled half-life, remove miscast relative bioavailability. Approximate label Cmax remains silver. |
| Ramelteon | The [Rozerem label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=63823b05-b54b-4221-9be9-5865d9221ee1) reports Cmax 11.6 ng/mL, AUC0–∞ 18.7 ng·h/mL, and half-life 2.6 h for one oral 16 mg dose in 24 elderly adults. | Retain Cmax and half-life; add matching AUC 0.0187 mg·h/L. The elderly cohort is explicit. |
| Rivaroxaban | [Zhao et al. Table 2/5](https://pmc.ncbi.nlm.nih.gov/articles/PMC2732942/) reports Cmax 143.2 µg/L and half-life 7.57 h for one fasted 10 mg oral dose in eight healthy Chinese men. The old 66% bioavailability belongs to a separate fasted 20 mg context in the [tablet label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=38b11f5d-f0fc-46dc-aef1-ea31872bb621). | Retain Cmax; correct half-life 5→7.57 h and remove the mismatched bioavailability. |

The regenerated public-profile **development** benchmark still scores N=79:
Meta AAFE **2.9272** (conditional compound-bootstrap 95% CI
**2.3869–3.6662**), Engine **3.9652**, direct ML **3.2553**, and descriptive
in-domain Meta **2.9439** (N=65). The paired Meta/ML AAFE ratio is **0.8992**
(95% conditional CI **0.7975–1.0106**), which includes 1. The nominal 90%
development-residual interval remains ±10.24-fold, covering **72/79 (91.1%)**
on this consumed cohort. None of these values measures independent
generalization.
