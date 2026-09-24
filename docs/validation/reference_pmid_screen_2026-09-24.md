# Development-reference PMID screen — 2026-09-24

This is a source-integrity check of the 173 `clinical_pk.json` rows carrying
Cmax at the start of this screen, not an independent evaluation. Ten rows had a PMID in
their `source` field. The original 14 identifiers were resolved against the
[NCBI PubMed ESummary API](https://www.ncbi.nlm.nih.gov/books/NBK25499/), then
the suspect sources were checked against the original publications.

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

The cited [Bækdal 2019 study](https://link.springer.com/content/pdf/10.1007/s40262-019-00756-2.pdf)
gave digoxin alone as one 500 μg oral dose in healthy adults. Its
[supplementary table 2c](https://media.springernature.com/original/springer-static/esm/art:10.1007%2Fs40262-019-00756-2/MediaObjects/40262_2019_756_MOESM1_ESM.pdf)
reports geometric-mean Cmax **3.11 ng/mL** (N=31), not the stored 1.5 ng/mL;
the digoxin reference is now 0.00311 mg/L and its synthetic curve was removed.
The same paper gave **25 mg racemic warfarin** and measured S-/R-warfarin
separately. It does not support the repository's 10 mg / 1.2783 mg/L total-parent
pair, so that Cmax and its synthetic curve were quarantined pending a matching
primary source. This leaves 175 Cmax rows and changes the 107-compound
development Meta AAFE from 2.7161 to 2.6976 solely through label correction.
The new [bootstrap artifact](../../data/validation/4track_ci_2026-09-24_digoxin_reference.json)
remains consumed development evidence; it is not a fresh model-performance gain.

Three further training/reference citations were adjudicated. [Becker
1984](https://pubmed.ncbi.nlm.nih.gov/6700656/) administered oral theophylline
at **5 mg/kg to 8–18-year-old patients with asthma** and reported mean peak
**8.4 mg/L**; it does not document the repository's fixed 300 mg / 8.7247 mg/L
healthy-adult pair. [Khomitskaya
2018](https://pubmed.ncbi.nlm.nih.gov/29548719/) studied **fed 1000 mg metformin
XR** (two 500 mg tablets) with dapagliflozin, not a 500 mg single-agent
immediate-release arm. The exact 500 mg / 1.03 mg/L value was instead located
in the [DailyMed immediate-release tablet label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=32ebd0fc-1491-451f-9258-61b315c0f499&type=display),
Table 1: fasting healthy adults, N=24, mean Cmax 1.03 mg/L. We restored this
reference at **389.926 mg metformin base** (500 mg HCl × 129.167/165.63), with
an explicit oral route and no synthetic curve. [Rebello
2011](https://pubmed.ncbi.nlm.nih.gov/20413453/) gave **verapamil 240 mg/day for
eight days** with aliskiren and analyzed R/S isomers at steady state, not a
single 80 mg parent Cmax. The theophylline and verapamil Cmax values and
synthetic curves were quarantined, leaving 173 Cmax rows. The development-residual
artifact was recomputed on 61 of 63 remaining training references; its nominal
90% half-width remained about 11.1-fold at that checkpoint.

Five more development citations failed arm-level checks. [Prescott
1993](https://pubmed.ncbi.nlm.nih.gov/9114910/) gave **20 mg/kg** paracetamol
to vegetarians and non-vegetarians, whose mean Cmax values were **11.7** and
**15.6 mg/L**, not the recorded fixed 1000 mg / 15.18 mg/L pair.
[Andersson 2001](https://pubmed.ncbi.nlm.nih.gov/11510629/) is an esomeprazole
drug-interaction **review**, not an amoxicillin PK study. [Kosuge
2001](https://pubmed.ncbi.nlm.nih.gov/11560871/) used **2 mg diazepam** with
and without diltiazem, not 10 mg. [Donzelli
2014](https://pubmed.ncbi.nlm.nih.gov/24218006/) studied a low-dose Basel
cocktail; the [author's dissertation](https://edoc.unibas.ch/38839/1/PhD%20Thesis%20M.%20Donzelli.pdf)
identifies **12.5 mg extended-release metoprolol**, not 100 mg.
[Gorski 2003](https://pubmed.ncbi.nlm.nih.gov/12966371/) used **3–8 mg
isotope-labeled oral midazolam** alongside an intravenous dose before and after
rifampin, not a fixed 2 mg oral arm. Without exact replacement primary arms,
these five Cmax labels and their synthetic curves were quarantined. There are
now **168 Cmax rows**. The development-residual artifact uses **57 of 58**
remaining training references (atenolol has an unknown route); its nominal
90% half-width is now **14.4-fold**. The N=107 development benchmark is
unchanged because these five were training references. The remaining PMID
citations are addressed below.

The final three PMID-linked development rows were checked against source
tables and methods. [Blomqvist
1988](https://pk-db.com/media/data/Blomqvist1988.pdf), Table 1, reports the
atenolol 50 mg Cmax after **four once-daily doses** (day 4, 922 nmol/L), so
it cannot label a single-dose prediction. [Wahlländer
1990](https://pk-db.com/media/data/Wahllaender1990.pdf) administered **200 mg
oral caffeine** in the afternoon and another **200 mg** the next morning to
patients with liver disease and controls, not the recorded 100 mg arm. The
atenolol and caffeine Cmax values, other unsupported PK parameters, and
synthetic curves were quarantined. [Elkoshi
2002](https://pk-db.com/media/data/Elkoshi2002.pdf), Table II, does contain
a usable fasting **single oral 20 mg** omeprazole reference arm: Losec,
N=40 healthy male volunteers, arithmetic mean Cmax **311 μg/L** and
AUC0–∞ **567 μg·h/L**. The recorded 141.2 μg/L was replaced with
0.311 mg/L; its AUC was corrected to 0.567 mg·h/L and synthetic curve
removed. There are now **166 Cmax rows**. All **56 of 56** remaining training
references run in the development-residual calculation, whose nominal 90%
half-width remains **14.4-fold**. The unchanged N=107 benchmark is consumed
development evidence; these corrections do not establish independent
performance.

After morphine correction, a public-profile 107-compound rerun had zero skips;
the other 106 predictions matched the previous cache to 1e-8 relative. The
development Meta AAFE changed from 2.7342 to 2.7161. That intermediate
[compound bootstrap](../../data/validation/4track_ci_2026-09-24_morphine_reference.json)
is retained for lineage; the current [prediction
cache](../../data/training/4track_holdout_predictions.json) includes the later
digoxin correction. All these figures are repeatedly accessed development
evidence. No independent external Cmax claim follows from either correction.

Legacy morphine TDM/SBI reports and the five-drug synthetic-observation
aggregate used the superseded 30 mg / 18.65 ng/mL pair. Their morphine rows,
pooled coverage/error summaries, and real-observation SBI comparison are
historical only until dose-matched reruns. The TDM benchmark scripts now use
the corrected reference; `sbi_compare_ibis.py` rejects the old 30 mg posterior
against the current 22.5 mg reference. A Cmax summary is not itself a measured
concentration at t=1 h, so the old comparison never established clinical
posterior accuracy.
