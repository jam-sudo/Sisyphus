# Development calibration reference audit — 2026-09-24

The counts and interval below are this audit's intermediate state. The later
[source-context screen](development_source_context_screen_2026-09-24.md)
supersedes them with 130 Cmax records and 41 training records.

This was a residual-led audit of ten large-error **training** labels. It does not
change the 86 scored development benchmark records or supply independent evidence.

| Record | Source finding | Action |
| --- | --- | --- |
| Lanthanum carbonate | The [label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=bb263fe8-2bdf-4dba-807f-194f54a73654) reports a 1.0 ng/mL **lanthanum** Cmax in patients, without a matching 500 mg single-dose arm; tablet strengths are elemental-lanthanum equivalents. | Quarantined the molecular-carbonate/elemental-analyte mismatch. |
| Cefpodoxime proxetil | The [label](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=4587a7c1-cd6c-4946-a0dd-fcfe0c8504b9) assigns 3.1 mg/L to **cefpodoxime** after a **200 mg fed** tablet, not proxetil parent after 100 mg. | Quarantined. |
| Serdexmethylphenidate | The [AZSTARYS label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=00b5e716-5564-4bbd-acaf-df2bc45a5663) assigns 14 ng/mL to **dexmethylphenidate** after a **52.3/10.4 mg combination**. | Quarantined. |
| Primaquine | [Cuong et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC1885124/) report fasting geometric-mean parent Cmax **127 ng/mL** after **30 mg base**, versus the prior unsupported 1 ng/mL. | Corrected Cmax to 0.127 mg/L. |
| Flutamide | The [original study](https://doi.org/10.1002/j.1552-4604.1989.tb03381.x) gave **250 mg** on day 1; the [label table](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=0a905e25-42b6-4937-a689-f01a8f22e644) reports single-dose parent Cmax **25.2 ± 34.2 ng/mL** in 12 geriatric volunteers. | Corrected dose from 200 to 250 mg; downgraded to silver because the SD exceeds the mean. |
| Carglumic acid | The [label's Table 3](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=591f95ca-3d99-4fb6-a402-e5a383a746d0) assigns 8613 ng/mL to the **IV 8 mg/kg** arm; the oral arm was **100 mg/kg**, not 100 mg. | Quarantined. |
| Belzutifan | The [label](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=13e15ee0-d679-4fa9-9430-e2e2170474da&version=11) gives 1.5 μg/mL as an estimated **steady-state** Cmax at the recommended **120 mg daily** dose, not 20 mg single dose. | Quarantined. |
| Pazopanib | The [label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=b05ea49a-a764-5ff2-6e4a-26add69da9e2) gives 58.1 μg/mL for **800 mg once daily**, without a matched single-dose arm. | Quarantined. |
| Carisoprodol | The [label's Table 2](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=7633bbb8-1b59-421d-995d-d1edaebb5cc0) gives **1.8 μg/mL parent Cmax** after 350 mg single dose; its separate ~8 μg/mL value concerns meprobamate after 400 mg meprobamate. | Corrected parent Cmax to 1.8 mg/L. |
| Atorvastatin | The [original bioequivalence study](https://pubmed.ncbi.nlm.nih.gov/15497662/) reports originator-arm parent Cmax **17.05 ng/mL** and AUC0–t **102.55 ng·h/mL** after four 10 mg tablets in 24 healthy men. The old 5.3 ng/mL row had no traceable arm. | Re-sourced the 40 mg arm to 0.01705 mg/L parent Cmax. |

Unsupported synthetic concentration-time curves and ancillary PK parameters were
removed from these records. The available Cmax reference count changed from 145
to 139; training calibration records changed from 56 to 50. The development
residual diagnostic's nominal 90% Meta half-width changed from 14.40× to 5.34×,
and its coverage on the already-consumed 86-compound development set changed
from 91.86% to **73.26%**. These shifts are descriptive: the audit selected records
after seeing residuals, and the current fitted model saw some calibration
outcomes. The resulting interval also misses even the precommitted 85% lower
coverage gate on consumed development data. The artifact remains explicitly
**not** valid split conformal. A
defensible replacement needs nested out-of-fold predictions or untouched
calibration compounds, followed by a separate blinded external test under the
precommitted [V1 protocol](external_holdout_v1_protocol.md).

The calibration generator now fails rather than silently dropping a training
prediction, and the runtime checks hashes of both the calibration reference
file and the holdout-membership file. A separate read-only AI review
confirmed the first five source decisions and reproduced the former 8.30×
intermediate result; that review is an additional error check, **not** a
blinded external cohort or independent human source verification.
