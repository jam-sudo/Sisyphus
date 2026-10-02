# Development benchmark steady-state proxy follow-up — 2026-09-24

Two scored development references used steady-state Cmax values converted to
putative single-dose values by a one-compartment accumulation formula. That
calculation was not an observed study arm. The correction rule was applied to
both records without choosing by prediction residual.

| Compound | Source adjudication | Replacement |
| --- | --- | --- |
| Cetirizine | The [cetirizine label](https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/211415s008lbl.pdf) reports **311 ng/mL after 10 mg daily for 10 days**. The previous **272.125 ng/mL** was extrapolated from that steady-state value. [Derakhshandeh and Mohebbi 2009, Table 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC3093629/) instead reports the **single fasted 10 mg Zyrtecset reference tablet** in 12 healthy men: parent plasma Cmax **266 ng/mL**, AUC0–∞ **2526 ng·h/mL**, and half-life **7.15 h**. The [French product monograph](https://base-donnees-publique.medicaments.gouv.fr/medicament/64949486/extrait) specifies **10 mg cetirizine dihydrochloride** per Zyrtecset tablet. | The model's parent-SMILES input uses **8.420813 mg cetirizine active moiety** (`10 × 388.89/461.82`), with observed Cmax **0.266 mg/L**. A spurious exploratory training row pairing **5 mg** with the label's **311 ng/mL steady-state** value was removed from both MMPK CSVs; the membership hashes were updated. |
| Febuxostat | The [ULORIC label](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=54de10ef-fe5f-4930-b91d-6bbb04c664bd&version=17) reports **1.6 µg/mL after multiple 40 mg daily doses**. The previous **1.5 mg/L** was an accumulation-adjusted estimate. [Khosravan et al. 2008, Table 3](https://bpspubs.onlinelibrary.wiley.com/doi/10.1111/j.1365-2125.2007.03016.x) reports the fasting reference after a **single 40 mg dose** in 23 completers: parent plasma Cmax **1.82 µg/mL**, AUC0–∞ **4.61 µg·h/mL**, and half-life **5.5 h**. | Replaced with **1.82 mg/L** and removed the synthetic time curve. |

The scored development cohort remains **N=86**. Only these two per-compound
cache rows changed; all other predictions and observations were bit-identical.
Meta AAFE changed from **2.8701** to **2.8815** (conditional compound-bootstrap
95% CI **2.3723–3.5325**); the current direct ML AAFE is **3.2960**. The
partially in-sample calibration set remains **36**, with the nominal 90%
Meta residual half-width at **10.24×** and **79/86** coverage on this consumed
development set. Neither the benchmark nor its confidence interval is an
independent external estimate under the [V1 protocol](external_holdout_v1_protocol.md).
