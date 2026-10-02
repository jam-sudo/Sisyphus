# Additional development PK-arm source check — 2026-09-24

Four scored references were checked against original regulatory labels, without
selecting them by prediction error. None supports the exact single-dose,
fixed-dose parent-Cmax benchmark arm recorded locally.

| Record | Original-source finding | Decision |
| --- | --- | --- |
| Lopinavir | The [KALETRA label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=c7fc2d1e-802e-4da1-9763-3355f8aafe3a) assigns **9.8 mcg/mL** to lopinavir after **400/100 mg lopinavir/ritonavir twice daily with food for 3 weeks** in 19 HIV-1 patients. | Quarantine the unboosted single-dose label; co-treatment, accumulation, food, and population differ. |
| Pilocarpine | The [SALAGEN label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=62eda83d-d043-4c7e-8fcf-75d05efbf35b) assigns **15 ng/mL** to the final **5 mg pilocarpine hydrochloride** dose after **2 days of three-times-daily** administration. | Quarantine the single-dose label. |
| Temozolomide | The [TEMODAR label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=046a9011-3911-4d3f-a15f-fbb56d5aad56) reports **7.5 mcg/mL** and **23.4 mcg·h/mL** after a **single 150 mg/m²** oral dose. It does not provide the study subjects' body surface areas or a fixed 260 mg arm. | Quarantine the locally imputed 260 mg label and remove its synthetic curve. The observed Cmax remains valid for the label's area-normalized dose, but cannot be scored against this fixed-dose input. |
| Venlafaxine | The prior 75 mg / **35.5 ng/mL** citation named only unspecified bioequivalence studies and an Effexor label. The located [US label](https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?setid=9f697c00-f5ec-454a-a66c-503aef5fc609&type=pdf) instead reports **225 ng/mL** for immediate-release **75 mg every 12 hours** at steady state. No original single-dose 75 mg / 35.5 ng/mL arm was identified. | Quarantine pending recovery of the exact primary study and formulation; do not substitute the steady-state observation. |

The public-profile development cache now scores **81** compounds: Meta AAFE
**2.8927** (conditional compound-bootstrap 95% CI **2.3617–3.5990**), Engine
**3.9166**, direct ML **3.2648**. The descriptive in-domain slice is N=67,
Meta **2.9015**. The nominal 90% development-residual half-width remains
**10.24×**, covering **74/81 (91.4%)** of this repeatedly used cohort. No
fitted model changed. These results are not an independent validation or a
model-gain comparison.
