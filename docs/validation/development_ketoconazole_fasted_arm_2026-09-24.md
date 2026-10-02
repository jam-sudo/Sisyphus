# Ketoconazole development-arm source resolution — 2026-09-24

The prior scored 200 mg / approximately 3.5 mg/L value came from a
[ketoconazole tablet label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=369f31da-d3c3-49b8-956f-f70598760164)
describing administration with a meal. The same-dose [original study by Huang
et al.](https://doi.org/10.1128/AAC.30.2.206) reports a more explicit,
overnight-fasted tablet arm. Its [full-text Table
2](https://e-lactancia.org/media/papers/Ketoconazol-FKAntAgChem1986.pdf),
printed p. 208, gives arithmetic mean parent-drug plasma Cmax
**4.22 ± 2.47 μg/mL** after one 200 mg tablet in **23 healthy young men**.
The study recruited 24, but one subject withdrew and was excluded from
analysis. The separate suspension and solution Cmax values, 5.04 and
6.17 μg/mL, were not pooled with the tablet.

The scored development arm now uses **4.22 mg/L** at the unchanged 200 mg
oral dose. The paper says a standard lunch was provided about **3–4 hours
after dosing**. Thus the arm is closer to the intended fasted setting than
the prior fed-label arm, but it does **not** prove the ≥4-hour postdose
fast required for an External Holdout V1 primary arm.

No fitted model changed. On the repeatedly accessed N=73 development set,
Meta AAFE moves **2.8322 → 2.8395** (conditional bootstrap 95% CI
**2.3011–3.5561**), Engine AAFE is **3.8718**, and direct ML AAFE is
**3.2549**. These are source-adjudicated development diagnostics, not an
independent accuracy test or a clinical release gate.
