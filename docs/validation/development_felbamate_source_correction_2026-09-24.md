# Felbamate training-label unit adjudication (2026-09-24)

The pinned [Omega MMPK snapshot](https://github.com/jam-sudo/Omega/blob/08a45047a2b5dcdca8c9a8f36ff1fe3b50ed3d6d/data/ml/clinical/mmpk_clean.csv)
contains one felbamate row: 600 mg oral dose, Cmax 0.0089 mg/L, one study.
That implies a reported peak of 8.9 ng/mL.

[Richens et al. 1997](https://bpspubs.onlinelibrary.wiley.com/doi/pdf/10.1046/j.1365-2125.1997.00642.x),
Table 1, reports Cmax **8.9** for young subjects after the 600 mg single dose,
alongside AUC **244** and pooled single-dose CL/F **31.2 mL/min**. Its PDF text
extraction mangles the microgram glyph, so the extracted unit string alone is
not reliable. The dose/AUC identity resolves the scale: 600 mg / (244 mg·h/L)
= 41 mL/min, of the same order as the reported CL/F; interpreting the AUC as
244 ng·h/mL instead gives 41 **L**/min. Thus the table's concentration unit is
µg/mL, and 8.9 µg/mL = **8.9 mg/L**, 1,000 times the archived Omega value.
The paper's Figure 1 concentration axis is likewise on a tens-of-µg/mL scale.

The original Omega snapshot stays byte-pinned for provenance. The Cmax
retraining recipe corrects this single row after holdout exclusion, asserts
the source value before conversion, and derives its log target from 8.9/600.
This is a source-label correction, not model tuning or external validation.
The independent external V1 cohort and release gate remain outstanding.
