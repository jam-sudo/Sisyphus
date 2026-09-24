# Development reference source follow-up — 2026-09-24

Four further Cmax rows were compared with original human studies or a regulatory
label. Selection did not use prediction error; these are development-data
repairs, not external validation.

| Record | Original-source finding | Disposition |
| --- | --- | --- |
| Amphetamine | The [ADZENYS XR-ODT label, §12.3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=c1179269-00b5-48ea-972d-31e614e99b7e) gives **44.9 ng/mL for d-amphetamine only**, after **18.8 mg total d+l amphetamine base** in an extended-release ODT (3:1 d:l), in 40 fasted adults. The local input had unspecified stereochemistry and paired the total dose with the d-only concentration. | Quarantined the Cmax and synthetic curve. Removed the same mismatched 18.8 mg/44.9 ng/mL row from both exploratory MMPK CSVs and updated their hash inventory. |
| Nifedipine | [Reitberg et al. 1987](https://pubmed.ncbi.nlm.nih.gov/3595068/) reports **78.9 ng/mL** mean parent plasma Cmax in 15 healthy men after one **10 mg Procardia immediate-release capsule, fasted**. The former 69 ng/mL attributed to an FDA label was not traceable to that arm. | Replaced with **0.0789 mg/L**. Removed the unmatched half-life and synthetic curve. |
| Felodipine | The prior **8.301 ng/mL at 5 mg** was derived from nonspecific “FDA label + OSP” provenance, without a matching arm. [Aguilar-Carrasco et al. 2015](https://scialert.net/fulltext/index.php?doi=ijp.2015.382.386) reports a different **5 mg extended-release** fasting arm at **4.63 nmol/L** (about **1.78 ng/mL**, using 384.25 g/mol), outside the immediate-release primary estimand. The local synthetic curve began at a fictitious **1 mg/L at time zero**. | Quarantined the unsupported reference and curve; did not substitute an extended-release observation for an unidentified original formulation. |
| Pitavastatin | [Zhang et al. 2015, Table I](https://pmc.ncbi.nlm.nih.gov/articles/PMC4460194/) reports **106.09 ng/mL** parent plasma Cmax, **321.25 ng·h/mL** AUC0–∞, and **9.52 h** half-life for the **2 mg single-dose calcium tablet** in 12 healthy volunteers after a 12-hour fast. | Replaced model-derived **57.949 ng/mL** with **0.10609 mg/L**, aligned AUC and half-life, and removed unmatched bioavailability and the synthetic curve. The same observed arm already occurs in the exploratory MMPK training CSV, so it cannot serve as independent evidence. |

After these changes, the reference pool has **125** Cmax rows and the
partially in-sample calibration set has **36**. Recalculation leaves the nominal
90% Meta residual half-width at **10.24×**, with **79/86 = 91.86%** coverage on
the repeatedly used development cohort. These are consumed development
diagnostics, not independent coverage under the [V1 external holdout
protocol](external_holdout_v1_protocol.md).
