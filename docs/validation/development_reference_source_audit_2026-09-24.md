# Development Cmax reference source audit — 2026-09-24

This is an audit of the repeatedly used development benchmark, not an
independent model evaluation. Nine scored rows used an estimated label, cited
the wrong analyte/arm, or lacked support for the stated Cmax. The rule was
applied to all nine rows before re-scoring; no model weights or fitted
artifacts changed.

| Drug | Previous benchmark label | Source finding | Current disposition |
|---|---|---|---|
| Leflunomide | 100 mg, 7.2 mg/L | [Li et al. 2019](https://pubmed.ncbi.nlm.nih.gov/31016613/) gave 10 mg leflunomide and measured **teriflunomide**, the active metabolite, at 0.718 ± 0.169 mg/L. The old value was a tenfold dose extrapolation of the wrong analyte. | Parent Cmax quarantined. |
| Sirolimus | 5 mg, 0.024 mg/L | The previous label was dose-proportional extrapolation, not an observed 5 mg arm. [Brattström et al. 2000](https://pubmed.ncbi.nlm.nih.gov/11034258/) used body-surface-area doses and measured whole blood; an exact-dose, matrix-matched source remains to be selected. | Cmax quarantined. |
| Paroxetine | 20 mg, 0.005 mg/L | The [PAXIL CR label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=027c7db2-2a36-477f-90bf-608ff3a0e14e&type=display) directly reports single oral **25 mg controlled-release** paroxetine in 23 healthy adults: mean parent Cmax **5.5 ng/mL**, AUC0–∞ **261 ng·h/mL**. A 20 mg value cannot be inferred reliably because exposure is nonlinear. | Replaced with exact 25 mg / 0.0055 mg/L arm; release formulation is now explicit. |
| Nilotinib | 400 mg, 1.95 mg/L | The [nilotinib capsule label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=e27c3e09-bc4f-4f01-be05-1c43ac72285d&type=display), Table 11, directly reports a fasted healthy-subject single **200 mg** arm with geometric mean parent Cmax **615 ng/mL** and AUC0–∞ **10,620 ng·h/mL**. The previous value was a simulated steady-state peak. | Replaced with exact 200 mg / 0.615 mg/L arm. |
| Clopidogrel | 300 mg, 0.3 mg/L | The cited generic FDA label did not report this parent Cmax. [Ganesan et al. 2013](https://bpspubs.onlinelibrary.wiley.com/doi/10.1111/bcp.12017), Table 1, directly reports parent clopidogrel Cmax **14.5 ± 9.6 ng/mL** in 24 women with stable coronary disease after a single 300 mg oral dose. | Replaced with exact 300 mg / 0.0145 mg/L parent arm; population caveat retained. |
| Fesoterodine | 8 mg, 0.00189 mg/L | The [fesoterodine label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=1d3fdfcf-7406-4be0-90e9-b10c9257bf69), Table 8, identifies **1.89 ng/mL as active 5-HMT** after **4 mg** in extensive CYP2D6 metabolizers; parent fesoterodine is undetectable in plasma. | Parent Cmax quarantined. |
| Molnupiravir | 800 mg, 2.33 mg/L | The [LAGEVRIO label](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=1b0da643-ab23-4a0b-a9ec-a434522446d0), Table 2, identifies **2330 ng/mL as NHC**, measured after **multiple** 800 mg doses every 12 h in patients. | Parent single-dose Cmax quarantined. |
| Valacyclovir | 1000 mg, 4.03 mg/L | The [valacyclovir label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=4ab809f3-8898-463c-b55a-f452c87d0779), Table 4, identifies **4.03 mcg/mL and AUC 14.4 mcg·h/mL as acyclovir** after adult **1 g valacyclovir**, based on historical estimates. The dose matches, but the analyte does not. | Parent Cmax and AUC quarantined. |
| Valganciclovir | 900 mg, 5.8 mg/L | The [valganciclovir label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=85177859-e0f5-6d83-17f3-29cae1db96c3), Table 10, reports **ganciclovir** after 900 mg valganciclovir once daily with food; parent valganciclovir Cmax is approximately **3%** of ganciclovir Cmax. The 5.8 mg/L value is not a supported parent value. | Parent Cmax quarantined. |

The scored benchmark is now **N=101** from the unchanged 107-compound split.
Meta AAFE is **2.7830** (compound bootstrap 95% CI **2.3321–3.3626**), versus
**2.6950** (2.2968–3.1599) for the prior N=105 cache. The corrected
clopidogrel reference and four excluded metabolite labels account for the
change; this is a change in label integrity, not model quality. Both cohorts
have repeatedly informed system development.

The engine observes `venous_blood`, whose compartment concentration is amount
divided by whole-blood volume. Production blood:plasma ratio estimates are
usually clamped to 1, so ordinary plasma comparisons use that approximation.
The approximation is especially weak for drugs with strong red-cell partitioning;
the sirolimus reference cannot be used as a generic plasma label. The complete
cohort still mixes formulation, food state, matrix, and population contexts.
An independently curated, outcome-blinded cohort is required before claiming
external accuracy or a quality score of 85 or more.
