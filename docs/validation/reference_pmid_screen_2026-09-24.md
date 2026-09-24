# Development-reference PMID screen — 2026-09-24

This is a source-integrity check of the 176 `clinical_pk.json` rows with Cmax,
not an independent evaluation. Fourteen rows have a PMID in their `source`
field. Their identifiers were resolved against the [NCBI PubMed ESummary
API](https://www.ncbi.nlm.nih.gov/books/NBK25499/), then the morphine source
and replacement arm were checked in the original publications.

The `morphine` row claimed oral 30 mg, Cmax 18.65 ng/mL and cited
[Bell 1985, PMID 2857025](https://pubmed.ncbi.nlm.nih.gov/2857025/). That study
compared **buccal** with **intramuscular** morphine in postoperative patients;
it contains no swallowed, fasted oral IR arm. The row's synthetic curve also
peaked above its stated Cmax. We replaced this development label with the
[Atrux-Tallau 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9705466/) pivotal
reference arm: healthy adults, fasted, single-dose oral immediate-release
Sevredol tablets, 3 × 10 mg morphine sulfate, arithmetic mean parent-plasma
Cmax 28.5 ± 11.9 ng/mL. The model dose is 22.5 mg parent base, using the
[morphine sulfate label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=5ceb3205-2f8d-4ba3-9f27-dc328bab4fa1&type=display)
equivalence of 30 mg sulfate to 22.5 mg morphine. The old synthetic curve was
removed. This is a reference correction on a **consumed development compound**,
not new independent evidence.

The other 13 PMIDs resolve to real papers, but their citation alone does not
verify the row's dose, formulation, route, analyte or Cmax arm. The highest
priority for original-table adjudication is
[theophylline/Becker 1984](https://pubmed.ncbi.nlm.nih.gov/6700656/)
(paper titled as a caffeine study),
[metformin/Khomitskaya 2018](https://pubmed.ncbi.nlm.nih.gov/29548719/)
(extended-release combination comparison), and
[verapamil/Rebello 2011](https://pubmed.ncbi.nlm.nih.gov/20413453/)
(aliskiren interaction study). The [digoxin and warfarin
rows](https://pubmed.ncbi.nlm.nih.gov/30945118/) cite a semaglutide interaction
study; verify the unboosted comparator arm before treating their recorded Cmax
as primary. These are **triage flags**, not findings that the other rows are
wrong. Digoxin is also in the 107-compound development cohort; the remaining
flagged examples above are in its training/reference side.

After morphine correction, a public-profile 107-compound rerun had zero skips;
the other 106 predictions matched the previous cache to 1e-8 relative. The
development Meta AAFE changed from 2.7342 to 2.7161. The new [prediction
cache](../../data/training/4track_holdout_predictions.json) and [compound
bootstrap](../../data/validation/4track_ci_2026-09-24_morphine_reference.json)
are still repeatedly accessed development evidence. No independent external
Cmax claim follows from this correction.

Legacy morphine TDM/SBI reports and the five-drug synthetic-observation
aggregate used the superseded 30 mg / 18.65 ng/mL pair. Their morphine rows,
pooled coverage/error summaries, and real-observation SBI comparison are
historical only until dose-matched reruns. The TDM benchmark scripts now use
the corrected reference; `sbi_compare_ibis.py` rejects the old 30 mg posterior
against the current 22.5 mg reference. A Cmax summary is not itself a measured
concentration at t=1 h, so the old comparison never established clinical
posterior accuracy.
