# Development Cmax reference source audit — 2026-09-24

This is an audit of the repeatedly used development benchmark, not an
independent model evaluation. Four scored rows explicitly used an estimated or
simulated label, or cited a different analyte/arm. The rule was applied to all
four rows before re-scoring; no model weights or fitted artifacts changed.

| Drug | Previous benchmark label | Source finding | Current disposition |
|---|---|---|---|
| Leflunomide | 100 mg, 7.2 mg/L | [Li et al. 2019](https://pubmed.ncbi.nlm.nih.gov/31016613/) gave 10 mg leflunomide and measured **teriflunomide**, the active metabolite, at 0.718 ± 0.169 mg/L. The old value was a tenfold dose extrapolation of the wrong analyte. | Parent Cmax quarantined. |
| Sirolimus | 5 mg, 0.024 mg/L | The previous label was dose-proportional extrapolation, not an observed 5 mg arm. [Brattström et al. 2000](https://pubmed.ncbi.nlm.nih.gov/11034258/) used body-surface-area doses and measured whole blood; an exact-dose, matrix-matched source remains to be selected. | Cmax quarantined. |
| Paroxetine | 20 mg, 0.005 mg/L | The [PAXIL CR label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=027c7db2-2a36-477f-90bf-608ff3a0e14e&type=display) directly reports single oral **25 mg controlled-release** paroxetine in 23 healthy adults: mean parent Cmax **5.5 ng/mL**, AUC0–∞ **261 ng·h/mL**. A 20 mg value cannot be inferred reliably because exposure is nonlinear. | Replaced with exact 25 mg / 0.0055 mg/L arm; release formulation is now explicit. |
| Nilotinib | 400 mg, 1.95 mg/L | The [nilotinib capsule label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=e27c3e09-bc4f-4f01-be05-1c43ac72285d&type=display), Table 11, directly reports a fasted healthy-subject single **200 mg** arm with geometric mean parent Cmax **615 ng/mL** and AUC0–∞ **10,620 ng·h/mL**. The previous value was a simulated steady-state peak. | Replaced with exact 200 mg / 0.615 mg/L arm. |

The scored benchmark is now **N=105** from the unchanged 107-compound split.
Meta AAFE is **2.6950** (compound bootstrap 95% CI **2.2968–3.1599**), versus
**2.6976** (2.3097–3.1656) for the prior N=107 cache. The similarity of the
aggregate hides large label changes: the corrected paroxetine prediction is
14.9-fold high and remains a controlled-release formulation outside the
model's observed inputs. This audit improves label integrity, not predictive
performance. Both cohorts have repeatedly informed system development.

The engine observes `venous_blood`, whose compartment concentration is amount
divided by whole-blood volume. Production blood:plasma ratio estimates are
usually clamped to 1, so ordinary plasma comparisons use that approximation.
The approximation is especially weak for drugs with strong red-cell partitioning;
the sirolimus reference cannot be used as a generic plasma label. The complete
cohort still mixes formulation, food state, matrix, and population contexts.
An independently curated, outcome-blinded cohort is required before claiming
external accuracy or a quality score of 85 or more.
