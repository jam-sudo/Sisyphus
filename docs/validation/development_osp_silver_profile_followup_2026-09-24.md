# Five OSP silver Cmax arms: source and estimator check — 2026-09-24

The five scored OSP rows below carried the generic source `OSP observed data`.
The extraction script takes the maximum of a digitized concentration–time curve.
For an aggregate curve, this is **not** the mean of the subjects' individual
Cmax values. The original articles' full texts were not available in this
review; original abstracts and the OSP model observations were checked without
looking at model errors before deciding which reference values to retain.

| Arm | Source finding | Decision |
|---|---|---|
| [Alprazolam, Juhl 1984](https://pubmed.ncbi.nlm.nih.gov/6143766/), 1 mg | [OSP record 741](https://github.com/Open-Systems-Pharmacology/Alprazolam-Model/blob/master/Alprazolam-Model.json): aggregate N=17 matched healthy controls, mean-profile peak 0.014003 mg/L. The abstract gives mean maximum 17.2 but prints a conflicting `micrograms/mL` unit and says serum rather than the OSP plasma metadata. | Retain 0.014003 at silver with an explicit profile-max and unit caveat. Do not silently convert the abstract value. |
| [Cimetidine, Grahnén 1979](https://pubmed.ncbi.nlm.nih.gov/520401/), 400 mg | [OSP record 1213](https://github.com/Open-Systems-Pharmacology/Cimetidine-Model/blob/master/Cimetidine-Model.json) is one person's two-tablet Tagamet curve (N=1), peak 2.348425 mg/L. | Quarantine: an individual curve cannot supply a cohort reference. |
| [Mefenamic acid, Mahadik 2012](https://pubmed.ncbi.nlm.nih.gov/22275128/), 250 mg | [OSP record 548](https://github.com/Open-Systems-Pharmacology/Mefenamic-acid-Model/blob/master/Mefenamic_acid-Model.json) is an N=12 aggregate capsule curve. Its isolated 2.173397 mg/L point is a digitized mean-profile maximum; the original abstract reports neither this Cmax nor the matching dose or meal state. | Quarantine until the original full-text arm and tabulated Cmax are verified. |
| [Probenecid, Selen 1982](https://pubmed.ncbi.nlm.nih.gov/7175716/), 500 mg | The original abstract directly reports **mean peak 35.3 µg/mL** after a single 0.5 g oral dose in fasted healthy men; the [OSP record 16901](https://github.com/Open-Systems-Pharmacology/Probenecid-Model/blob/main/Probenecid-Model.json) curve peaks at 32.9 mg/L. | Use 35.3 mg/L (µg/mL and mg/L are numerically equal), retain silver because N and tablet formulation are OSP-only metadata. |
| [Triazolam, Kroboth 1995](https://pubmed.ncbi.nlm.nih.gov/7593708/), 0.25 mg | [OSP record 711](https://github.com/Open-Systems-Pharmacology/Triazolam-Model/blob/master/Triazolam-Model.json): aggregate N=12, mean-profile peak 0.001483 mg/L. The original abstract confirms 12 men and a single marketed oral tablet but provides no Cmax or fasting state. | Retain the provisional silver profile maximum with exact provenance. |

The OSP extraction now preserves the source formulation and data type and does
not select an individual curve as a cohort observation. The integration step
also skips N=1/individual records and respects existing `unverified` decisions
across OSP, curated, manual, and FDA imports. Before this guard, a dry-run
re-integration restored seven previously quarantined Cmax rows (atovaquone,
leflunomide, lopinavir, pilocarpine, prasugrel, sirolimus, venlafaxine), raising
the source count to 84. The guarded dry run stays at 77. The archived OSP
curves remain in `osp_observed.json` as traceable inputs, not scored labels.

After these **reference-only** changes, the scored development set is N=77
(previously 79). Meta AAFE is **2.9299** (conditional compound-bootstrap 95% CI
**2.3833–3.6509**), from **2.8980**. The fitted models did not change. This
repeatedly used development benchmark and the second AI source check are not an
independent, outcome-blinded clinical validation.
