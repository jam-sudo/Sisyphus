# Development label provenance follow-up — 2026-09-24

Four scored development rows had generic secondary/model-source descriptions.
Their Cmax values were checked against original regulatory labels. The
decision did not use prediction error; this set remains a consumed development
benchmark.

| Compound | Original label finding | Change |
| --- | --- | --- |
| Alosetron | The [tablet label, §12.3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=2846a244-7540-442b-81bc-638e641497ce) reports parent plasma Cmax **about 5 ng/mL** in young men after one **1 mg alosetron** oral dose. The 1 mg tablet contains **1.124 mg alosetron hydrochloride**, expressed as 1 mg parent. | Kept observed **0.005 mg/L** and 1 mg parent dose; documented the arm. Removed general-population half-life and bioavailability values and a synthetic curve. |
| Azacitidine | The [ONUREG label, §12.3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=0e95e33f-8aba-4f19-b332-2416580d358b) directly reports a **single 300 mg oral** dose with parent plasma mean Cmax **145 ng/mL** and AUC **242 ng·h/mL**. It does not identify this arm as healthy-adult, fasted. | Kept **0.145 mg/L** and **0.242 mg·h/L**; removed the general half-life, relative-to-subcutaneous bioavailability, and synthetic curve. The arm remains outside documented V1 primary eligibility. |
| Clomipramine | The [capsule label](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=7d11010e-cf0f-4315-aced-7927a486a1ef) reports parent Cmax mean **92 ng/mL** (range 56–154) after one **50 mg clomipramine hydrochloride** oral dose; population and meal state are not specified. [NIST parent mass](https://webbook.nist.gov/cgi/cbook.cgi?ID=303-49-1) is **314.852 g/mol** and [NCATS salt mass](https://gsrs.ncats.nih.gov/ginas/app/beta/substances/2LXW0L6GWJ) is **351.31 g/mol**. The former **19 h** half-life was a lower bound reported for a different **150 mg** dose. | Converted model input to **44.811135 mg parent** (`50 × 314.852/351.31`), retained **0.092 mg/L**, downgraded to silver for unknown study context, and removed unmatched half-life and synthetic curve. |
| Tamoxifen | The [citrate-tablet label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=9b8a4211-120f-4981-ad69-928accb97637) reports mean parent Cmax **40 ng/mL** after a single **20 mg tamoxifen-base-equivalent** tablet; it contains **30.4 mg citrate salt**. The source's terminal half-life is a **5–7-day range**, not exactly 120 h. | Kept **0.04 mg/L** and 20 mg parent dose; removed the point half-life and synthetic curve. |

Three separately identified exploratory MMPK rows were removed from both CSVs:
alosetron **4 mg / 0.005 mg/L** matches the label's **1 mg** Cmax despite the **4 mg** dose; azacitidine
**300 mg / 0.145 mg/L** had unspecified ribose stereochemistry despite an
isomeric duplicate; tamoxifen **20 mg / 0.04 mg/L** omitted the marketed alkene
geometry despite an isomeric duplicate. The training-membership SHA inventory
was updated. Other source-distinct rows were retained.

Only clomipramine's prediction changed in the N=86 scored cache when its dose
was corrected. Meta development AAFE moved **2.8815 → 2.8852** (conditional
95% compound-bootstrap CI **2.3761–3.5348**); direct ML AAFE is **3.3002**.
The nominal 90% development residual half-width remains **10.24×**, with
**79/86** coverage. These figures do not prove external generalization; see
the [V1 protocol](external_holdout_v1_protocol.md).
