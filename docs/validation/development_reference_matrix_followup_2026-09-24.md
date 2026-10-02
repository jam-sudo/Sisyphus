# Development reference matrix follow-up — 2026-09-24

Four more training Cmax labels were checked against regulatory labels or original
study tables before considering model residuals. These are source-data repairs,
not independent validation or changes to fitted model weights.

| Record | Source finding and correction |
| --- | --- |
| Lofexidine | The [label, §12.3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=2dcc8288-adbe-45c3-b7bd-4d274001332d) assigns **0.82 ng/mL** parent plasma Cmax and **14.9 ng·h/mL** AUC0–∞ to one **0.36 mg oral solution** dose, not a 0.18 mg tablet. Absolute bioavailability is **72%**; the prior 30% was approximate first-pass conversion. The label expresses **tablet** strength as lofexidine parent equivalent; applying that basis to the solution is an inference, matching the salt-free structure. |
| Indapamide | The [tablet label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=71feffd5-4056-2b53-e053-2995a90a5b73) reports **115 ng/mL** after **2.5 mg**, in **whole blood**, and 260 ng/mL after 5 mg; whole-blood/plasma ratio is approximately 6:1 at peak. The replacement [Li et al. 2013 reference arm, Table 2](https://www.thieme-connect.com/products/ejournals/pdf/10.1055/s-0032-1331181.pdf) measured parent **plasma** Cmax **47.79 ng/mL**, AUC0–∞ **919.52 ng·h/mL**, and t½ **23.23 h** after a single fasted **5 mg** dose as two immediate-release 2.5 mg tablets in 20 healthy men. |
| Fluoxetine | The old **20 ng/mL** was an estimate from a broad label range. The [FDA SARAFEM label, Table 1](https://www.accessdata.fda.gov/drugsatfda_docs/label/2007/021860s001lbl.pdf) gives a single **20 mg fluoxetine-base-equivalent tablet** parent plasma mean Cmax **13.2 ng/mL** and median t½ **26.5 h** in 23 healthy women. Its **722.4 ng·h/mL** AUC is AUC0–t, so it was not stored as unqualified AUC. |
| Ezetimibe | The old **3.4 ng/mL** was the lower end of the [label's 3.4–5.5 ng/mL range](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=11828685-dc27-42d8-a6ff-f110018207d1), not a single observed arm. [Sun et al. 2023, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC9896679/) reports unconjugated **parent** plasma Cmax **3.48 ng/mL**, AUC0–∞ **68.62 ng·h/mL**, and t½ **19.38 h** for a single fasted **10 mg Ezetrol reference tablet**. The structure now includes the marketed stereochemistry ([PubChem CID 150311](https://pubchem.ncbi.nlm.nih.gov/compound/Ezetimibe)). |

Four synthetic concentration curves were removed because they are not observed
time-series data. A spurious 10 mg/3.4 ng/mL ezetimibe duplicate was removed
from both exploratory MMPK CSVs, and their SHA manifest was updated. The
development pool remains 127 Cmax records, including 38 training references.
Recalibration still yields a nominal 90% Meta half-width of **10.24×** and
**79/86 = 91.86%** coverage on the repeatedly used development benchmark.
These partially in-sample figures do not satisfy the [V1 external validation
protocol](external_holdout_v1_protocol.md).
