# Three legacy silver PK arms: original-source check — 2026-09-24

The apixaban, famotidine, and sildenafil development references had brief
source strings that did not identify the exact arm. Decisions below use the
original study or product label, before examining model errors.

| Arm | Original-source finding | Reference action |
|---|---|---|
| Apixaban, 5 mg | [Wang et al. 2016, Table 2](https://doi.org/10.1002/jcph.628) reports parent-plasma Cmax geometric mean **126 ng/mL** in eight healthy controls after one 5 mg oral dose. It reports a 20 h mean half-life with wide variability, while the reference had a generic 12 h half-life and 50% bioavailability from another source. Meal state and formulation are not specified. The previous citation was a secondary review. | Keep 0.126 mg/L at silver, cite the original control arm, and remove mixed-arm parameters and the unsupported fasted claim. |
| Famotidine, 20 mg | [DailyMed Section 12.3](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=044a510f-33df-4484-8f07-d821f52b9376) reports **73 ng/mL** serum Cmax in eight healthy subjects after one oral 20 mg dose **without** probenecid. The matching AUC is **424 ng·h/mL over 0–10 h**. The reference's **580 ng·h/mL** AUC belongs to a different pediatric **0.5 mg/kg** arm. The label gives a general 2.5–3.5 h half-life range, not an arm-specific 2.5 h point estimate; meal state is not reported. | Keep 0.073 mg/L at silver and remove the mixed-arm AUC and half-life. The generic AUC field does not encode the 0–10 h horizon. |
| Sildenafil, 50 mg | [Nichols et al. 2002, Table 3](https://doi.org/10.1046/j.0306-5251.2001.00027.x) reports parent-plasma Cmax geometric mean **271 ng/mL** and harmonic-mean half-life **2.96 h** in the fasted 50 mg tablet arm of its N=32 dose-proportionality crossover. The prior 270 ng/mL and 2.9 h were rounded. The paper's **41%** bioavailability comes from a separate N=12 oral/IV arm. Table 1 also has a different 50 mg oral Cmax (**159 ng/mL**) in that separate arm; the reference represents Table 3. | Use 0.271 mg/L and 2.96 h, cite Table 3 and remove cross-study bioavailability. |

The [Viagra label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=d905dc8d-917f-4ea3-a4ee-a1ecf6967d4e)
confirms that a 50 mg sildenafil citrate tablet contains the equivalent of
50 mg sildenafil; the model's 50 mg parent dose is on the correct mass basis.

These are reference-integrity corrections, not changes to the fitted models or
an independent clinical validation. The scored cohort remains N=77. Development
Meta AAFE changes **2.9299 → 2.9301** (conditional bootstrap 95% CI
**2.3833–3.6510**).
