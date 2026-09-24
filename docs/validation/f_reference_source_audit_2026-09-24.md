# Human oral absolute-F source audit (2026-09-24)

The ten values previously used by `scripts/run_f_decomposition.py` and
`scripts/run_measured_adme_benchmark.py` mixed absolute systemic bioavailability
with absorption, relative bioavailability, animal estimates, and unmatched
patient studies. They cannot support the reported 10/10 engine-F under-call or
the N=10 measured-F Cmax improvement. The diagnostic now uses production's
converged oral/IV exposure ratio; the scored Cmax model was not changed.

| Drug | Prior F | Source check | Diagnostic decision |
| --- | ---: | --- | --- |
| Diclofenac | 0.55 | [DailyMed delayed-release label, PK Table 1](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=a65aa738-8ce2-4276-8e54-5ecf4f461d3a): human absolute F mean 0.55 (N=7, CV 40%). | Retain, provisional context match. |
| Sildenafil | 0.40 | [VIAGRA label §12.3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ae2079a2-f3a9-4739-9611-0742b71e4761): human absolute F mean 0.41, range 0.25–0.63. | Correct to 0.41, provisional context match. |
| Quinine | 0.80 | [Quinine sulfate label §12.3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=f567d5c7-ea5d-49a7-a035-b47208135f73): healthy-adult F range 0.76–0.88. | Retain as a range for F diagnosis; no point-value Cmax arm. |
| Alprazolam | 0.90 | [Original oral/IV crossover](https://pubmed.ncbi.nlm.nih.gov/6152055/) involved six fasted healthy men, but the available abstract does not verify the exact numeric F. | Exclude pending original full-text extraction. |
| Carbamazepine | 0.80 | [Oral/IV patient study](https://pubmed.ncbi.nlm.nih.gov/22278332/) reports F 0.78 under epilepsy-treatment conditions, not a matched healthy single-dose arm. | Exclude pending context match. |
| Clozapine | 0.55 | [Original oral/IV patient study](https://pubmed.ncbi.nlm.nih.gov/3203703/) reports mean F 0.27 in ten chronically treated male patients; the previous 0.55 conflicts. | Exclude pending context match and reconciliation. |
| Etodolac | 1.00 | [DailyMed label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=816576ac-c49d-4762-9722-e30493d6af83): 100% is relative bioavailability versus oral solution; ≥80% systemic availability is inferred from mass balance. | Exclude; no exact absolute-F anchor. |
| Febuxostat | 0.85 | [EMA assessment, p. 19](https://www.ema.europa.eu/en/documents/assessment-report/adenuric-epar-public-assessment-report_en.pdf): absolute human F not determined; ≈84% refers to absorption inferred from mass balance. | Exclude. |
| Dasatinib | 0.25 | [EMA original assessment](https://www.ema.europa.eu/en/documents/scientific-discussion/sprycel-epar-scientific-discussion_en.pdf): cited 14–34% F is animal data, not a human absolute-F estimate. | Exclude. |
| Clopidogrel | 0.50 | [DailyMed label §12.3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=52adfb2c-2062-495c-9954-39eeecae2b41): at least 50% *absorbed*, inferred from urinary metabolites of a prodrug. | Exclude; absorption is not parent systemic F. |

With the three retained references, current engine F / reference F is 0.98
for diclofenac, 0.42 for sildenafil, and 0.20–0.23 for quinine. Inputs and
formulations are not fully paired. This small set diagnoses individual cases;
it does not establish a population-level F bias or measured-F accuracy gain.
