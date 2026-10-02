# Codeine development reference follow-up — 2026-09-24

The scored codeine row previously used a 60 mg immediate-release codeine
phosphate arm from [Band et al. 1994](https://pubmed.ncbi.nlm.nih.gov/7983238/).
Its abstract reports 138.8 ng/mL parent Cmax but does not establish the
parent-equivalent dose or fasting state. The row was therefore silver.

The [FDA clinical pharmacology review for NDA 202245](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2011/202245Orig1s000ClinPharmR.pdf),
study S30-T30-PVFS, directly matches a single **30 mg codeine sulfate
immediate-release tablet** in 36 overnight-fasted healthy volunteers to a
parent-plasma Cmax of **71.4 ± 21.2 ng/mL** (Table 1, arithmetic mean ± SD).
The model uses codeine free-base SMILES. The FDA review identifies the sulfate
as a trihydrate; the [USP monograph](https://doi.usp.org/USPNF/USPNF_M19640_04_01.html)
gives 750.85 g/mol for two codeine molecules per sulfate trihydrate, and
[NIST](https://webbook.nist.gov/cgi/cbook.cgi?ID=C76573&Mask=3600&Units=CAL)
gives 299.3642 g/mol for codeine. The model dose is therefore
`30 × (2 × 299.3642) / 750.85 = 23.922024 mg` parent equivalent; Cmax is
0.0714 mg/L. This matched, directly tabulated arm replaces the ambiguous
1994 arm, and the row is gold. No concentration–time curve was restored.

The public-profile N=79 development Meta AAFE is now **2.9442** (conditional
bootstrap 95% CI **2.4084–3.6653**), versus **2.9348** immediately before this
reference correction. Engine is **3.8818**, direct ML **3.2988**, and
descriptive in-domain Meta **3.0312** (N=65). The paired Meta/ML ratio remains
**0.8925** (CI **0.7890–1.0048**); the development-residual 90% half-width
remains 10.24× and covers 72/79. Fitted models did not change. This
repeatedly accessed cohort cannot establish independent generalization.
