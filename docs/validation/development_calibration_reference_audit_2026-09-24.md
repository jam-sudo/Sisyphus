# Development calibration reference audit — 2026-09-24

This was a residual-led audit of five large-error **training** labels. It does not
change the 86 scored development benchmark records or supply independent evidence.

| Record | Source finding | Action |
| --- | --- | --- |
| Lanthanum carbonate | The [label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=bb263fe8-2bdf-4dba-807f-194f54a73654) reports a 1.0 ng/mL **lanthanum** Cmax in patients, without a matching 500 mg single-dose arm; tablet strengths are elemental-lanthanum equivalents. | Quarantined the molecular-carbonate/elemental-analyte mismatch. |
| Cefpodoxime proxetil | The [label](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=4587a7c1-cd6c-4946-a0dd-fcfe0c8504b9) assigns 3.1 mg/L to **cefpodoxime** after a **200 mg fed** tablet, not proxetil parent after 100 mg. | Quarantined. |
| Serdexmethylphenidate | The [AZSTARYS label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=00b5e716-5564-4bbd-acaf-df2bc45a5663) assigns 14 ng/mL to **dexmethylphenidate** after a **52.3/10.4 mg combination**. | Quarantined. |
| Primaquine | [Cuong et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC1885124/) report fasting geometric-mean parent Cmax **127 ng/mL** after **30 mg base**, versus the prior unsupported 1 ng/mL. | Corrected Cmax to 0.127 mg/L. |
| Flutamide | The [original study](https://doi.org/10.1002/j.1552-4604.1989.tb03381.x) gave **250 mg** on day 1; the [label table](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=0a905e25-42b6-4937-a689-f01a8f22e644) reports single-dose parent Cmax **25.2 ng/mL**. | Corrected dose from 200 to 250 mg. |

Unsupported synthetic concentration-time curves and ancillary PK parameters were
removed from these records. The available Cmax reference count changed from 145
to 142; training calibration records changed from 56 to 53. The development
residual diagnostic's nominal 90% Meta half-width changed from 14.40× to 8.30×,
and its coverage on the already-consumed 86-compound development set changed
from 91.86% to 86.05%. These shifts are descriptive: the audit selected records
after seeing residuals, and the current fitted model saw some calibration
outcomes. The artifact remains explicitly **not** valid split conformal. A
defensible replacement needs nested out-of-fold predictions or untouched
calibration compounds, followed by a separate blinded external test under the
precommitted [V1 protocol](external_holdout_v1_protocol.md).
