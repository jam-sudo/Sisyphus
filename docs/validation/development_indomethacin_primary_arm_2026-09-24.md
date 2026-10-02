# Indomethacin scored-arm source resolution — 2026-09-24

The former 25 mg / 1.54 mg/L development label came from DrugBank/PubChem,
which quote 1.54 ± 0.76 mcg/mL for fasting subjects. The original study was
not recovered. A US capsule label gives only an approximate 1 mcg/mL for a
single 25 mg capsule, so the exact old arm remained secondary-source silver.

The [Health Canada APO-INDOMETHACIN product
monograph](https://pdf.hres.ca/dpd_pm/00073611.PDF), section 14.2, pp. 32–34,
reports a randomized, single-dose comparative bioavailability study in
**12 fasted healthy adult men**. Each received **two 25 mg capsules
(50 mg total)**. The INDOCID reference-product column gives parent
indomethacin plasma Cmax **3107.0 ng/mL arithmetic mean (CV 35.3%)**;
the geometric mean is 2838.4 ng/mL. The scored reference now uses the
arithmetic mean, **3.107 mg/L**, and the matching **50 mg oral dose**.
The test product's 2979.2 ng/mL and NOVOMETHACIN comparator's
2544.3 ng/mL are separate formulation arms and were not pooled.

This changes reference provenance and input dose, not fitted model weights.
The repeatedly accessed N=73 development Meta AAFE changes from **2.8319**
to **2.8322**; the conditional compound-bootstrap 95% CI is
**2.2980–3.5446**. Engine AAFE is **3.8619**, direct ML **3.2466**.
The paired Meta/ML AAFE ratio is **0.8724** (conditional 95% CI
**0.7601–0.9956**). These are retrospective, adaptively selected
development diagnostics and do not establish external accuracy or clinical
readiness.
