# Four more legacy silver Cmax arms — original-source check, 2026-09-24

These four rows were selected for opaque source citations before inspecting
prediction errors. This is a repair of the repeatedly used development data,
not an external validation.

| Arm | Original-source finding | Action |
|---|---|---|
| Acamprosate, two 333 mg calcium-salt tablets (600 mg parent equivalent) | The [2010 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC2853976/) states **180 ng/mL** after one dose, but does not identify a matching original arm or meal state. The [FDA medical review](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2004/21-431_Campral_Medr_P1.pdf) instead says approximately **94 ng/mL** for two 333 mg tablets. [Luo et al. 2015](https://pubmed.ncbi.nlm.nih.gov/26360834/) directly reports **244.64 ng/mL** for a single fasted 666 mg calcium-salt reference-formulation arm. That same 244.64 value is already one of two values combined in a historical MMPK training row: `sqrt(244.64 × 297.5) = 269.7784276 ng/mL`. | Quarantined the untraceable 180 ng/mL benchmark value; kept the parent-equivalent dose. Did not substitute a previously used corpus arm as a nominal holdout. |
| Ponatinib, 45 mg | [Narasimhan et al., *J Clin Pharm Ther* 2013](https://pubmed.ncbi.nlm.nih.gov/23888935/) reports a fasted healthy-volunteer single-dose geometric mean of **54.7 ng/mL**. The former journal/year were wrong. The [ICLUSIG label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=16d804b6-4957-43ee-b18c-3b36ec37c5ac) attributes **73 ng/mL** to **presumed steady state in cancer patients**, not a single dose. The 54.7 arm also contributes to an exploratory MMPK training aggregate, a development-data overlap caveat. | Retained 0.0547 mg/L, corrected the study citation and the separate FDA extraction context. |
| Posaconazole, 400 mg oral suspension | [Krishna et al., *Antimicrob Agents Chemother* 2009, Table 2 Part 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2650585/) reports arithmetic mean **151 ng/mL** (CV 58%) after one dose following a 10 h fast, without acidic beverage or esomeprazole. The same paper's **Part 3** has a different fasted control mean of 181 ng/mL. The previous Courtney 2004 attribution was wrong. | Retained 0.151 mg/L and named the exact Part 1 arm. |
| Upadacitinib, 15 mg extended-release | [Mohamed et al., *Clin Pharmacol Drug Dev* 2019, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6585649/) reports arithmetic mean **26.0 ± 9.7 ng/mL** after one fasted dose in 11 healthy volunteers. The previous *Clinical Pharmacokinetics* 2020 attribution points to a review, not the original study. | Retained 0.026 mg/L and corrected the original citation and formulation. |

After the acamprosate quarantine, the scored development set is **N=75**.
Meta AAFE is **2.8626** (conditional compound-bootstrap 95% CI
**2.3284–3.5706**), Engine **3.8252**, and direct ML **3.2418**. The paired
Meta/ML AAFE ratio is **0.8830** (95% CI **0.7735–1.0075**). The nominal 90%
development residual interval covers **69/75 (92.0%)**; its half-width stays
at **10.24-fold** from 36 partly in-sample calibration references. The model
weights did not change. These metrics are cohort-sensitive diagnostics and
cannot establish independent generalization or Meta superiority.
