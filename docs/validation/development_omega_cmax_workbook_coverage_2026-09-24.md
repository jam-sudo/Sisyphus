# Omega Cmax upstream workbook coverage (2026-09-24)

The fitted Cmax snapshot is derived from the pinned [Omega aggregate CSV](https://github.com/jam-sudo/Omega/blob/08a45047a2b5dcdca8c9a8f36ff1fe3b50ed3d6d/data/ml/clinical/mmpk_clean.csv), but that aggregate has **two** upstream workbooks. The [original workbook](https://github.com/jam-sudo/Omega/blob/08a45047a2b5dcdca8c9a8f36ff1fe3b50ed3d6d/data/external/mmpk/approved.xlsx) has SHA256 `4521a2d89dde71c9e9381ab0a920e84dfc69c714860de377cd73b338572026d2`; the [2024 supplement](https://github.com/jam-sudo/Omega/blob/08a45047a2b5dcdca8c9a8f36ff1fe3b50ed3d6d/data/external/mmpk/approved_2024.xlsx) has SHA256 `bfdc1bac564dcf3a29f2f8021f69c4f6e087a687bea329433ccd47d233b4d32f`. The model metadata now names both immutable sources. The aggregate CSV remains the actual retraining input; the workbooks are provenance sources, not files required at inference time.

An exact-name, dose, arm-count, and geometric-Cmax reconciliation found **898** of the current 907 fitted aggregates numerically in the original workbook and **eight** in the supplement. The remaining **felbamate** row is deliberately corrected from the original workbook's unit error, as [adjudicated separately](development_felbamate_source_correction_2026-09-24.md). The original-workbook count fell by one when the [dolasetron active-metabolite aggregate](development_dolasetron_analyte_followup_2026-09-24.md) was quarantined. The eight supplemental fitted aggregates are:

| Drug | Dose (mg) | Supplement rows | Arm Cmax (ng/mL) | Fitted Cmax (mg/L) | Source PMID / URL |
|---|---:|---|---|---:|---|
| Acoramidis | 50 | 2 | 2110 | 2.11 | [31172685](https://pubmed.ncbi.nlm.nih.gov/31172685/) |
| Aprocitentan | 25 | 7–8 | 1580, 1266 | 1.4143126 | [30962677](https://pubmed.ncbi.nlm.nih.gov/30962677/), [36352054](https://pubmed.ncbi.nlm.nih.gov/36352054/) |
| Ensartinib | 225 | 14 | 264 | 0.264 | [34757657](https://pubmed.ncbi.nlm.nih.gov/34757657/) |
| Givinostat | 50 | 15 | 53 | 0.053 | [21365126](https://pubmed.ncbi.nlm.nih.gov/21365126/) |
| Lazertinib | 240 | 20–21 | 457.54, 325.79 | 0.3860854 | [36495784](https://pubmed.ncbi.nlm.nih.gov/36495784/) |
| Mavorixafor | 50 | 22 | 159 | 0.159 | [17452489](https://pubmed.ncbi.nlm.nih.gov/17452489/) |
| Seladelpar | 10 | 26 | 71.9 | 0.0719 | [2019 original poster](https://www.postersessiononline.eu/173580348_eu/congresos/ILC2019/aula/-FRI_41_ILC2019.pdf) |
| Vadadustat | 450 | 28–29 | 53000, 46600 | 49.6970824 | [33661566](https://pubmed.ncbi.nlm.nih.gov/33661566/), [35172045](https://pubmed.ncbi.nlm.nih.gov/35172045/) |

The 2024 supplement contains a ninth identity, elafibranor, which is not in the current fitted snapshot. The [seladelpar source poster](https://www.postersessiononline.eu/173580348_eu/congresos/ILC2019/aula/-FRI_41_ILC2019.pdf) independently displays the normal-function cohort's 71.9 ng/mL Cmax. This audit establishes upstream row coverage and fixes the missing provenance pointer; it does not independently re-adjudicate every study's population, formulation, dose basis, or plasma matrix. No fitted value or model prediction changed.
