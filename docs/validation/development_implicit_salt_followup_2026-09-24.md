# Development implicit-salt reference follow-up — 2026-09-24

A follow-up screen of the 79 scored references surfaced four products whose source text did
not make their salt-mass dose obvious to the model's parent-SMILES input. The
source arm was checked before changing its benchmark value.

| Drug | Primary source arm | Corrected parent-equivalent input |
| --- | --- | ---: |
| Amantadine | [DailyMed label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=5ae327cf-8946-4682-9025-64653673e367): one 100 mg hydrochloride soft-gel capsule in 24 healthy men, parent plasma Cmax 0.22 ± 0.03 mcg/mL and half-life 17 ± 4 h. Meal state is unspecified in this arm. | `100 × 151.25 / 187.71 = 80.576421 mg` ([parent](https://pubchem.ncbi.nlm.nih.gov/compound/Amantadine), [HCl](https://pubchem.ncbi.nlm.nih.gov/compound/Amantadine-Hydrochloride)). |
| Fluvoxamine | [Van Harten et al. 1991, Table 1](https://pillbuys.com/research/Fluvoxamine/8.pdf) ([PubMed](https://pubmed.ncbi.nlm.nih.gov/1801963/)): one 50 mg maleate immediate-release tablet after overnight fast in 12 volunteers; arithmetic mean parent plasma Cmax **17 ng/mL**, replacing an untraceable 15 ng/mL. | `50 × 318.33 / 434.4 = 36.640193 mg` ([parent](https://pubchem.ncbi.nlm.nih.gov/compound/fluvoxamine), [maleate](https://pubchem.ncbi.nlm.nih.gov/compound/fluvoxamine-maleate)). |
| Hydroxyzine | [Swedish regulator product summary, sections 2 and 5.2](https://docetp.mpa.se/LMF/Hydroxyzine%20Orifarm%20film-coated%20tablet%20ENG%20SmPC_09001bee807a397d.pdf): 25 mg tablet contains 25 mg hydroxyzine dihydrochloride; a single adult 25 mg dose typically yields parent plasma Cmax 30 ng/mL. Exact study population and meal state are not identified, so this remains silver. | `25 × 374.9 / 447.83 = 20.928701 mg` ([parent](https://pubchem.ncbi.nlm.nih.gov/compound/Hydroxyzine), [product salt formula/MW](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=7534a3bb-b4cd-415f-8612-53b37dc6d55f)). |
| Trazodone | [Kale and Agrawal 2015, Table 1](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2015.00224/full): one 100 mg hydrochloride reference tablet under fed conditions; arithmetic mean parent plasma Cmax **1546.9 ± 315.4 ng/mL**. This replaces 1.62 mg/L attributed to [Nilsen and Dale 1992](https://pubmed.ncbi.nlm.nih.gov/1438031/), whose abstract instead reports **serum** peaks 1.88 mg/L fasted and 1.47 mg/L fed. | `100 × 371.864 / 408.33 = 91.069478 mg` ([parent](https://webbook.nist.gov/cgi/cbook.cgi?ID=19794-93-5), [HCl label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=66da652a-72b4-c17f-e053-2a91aa0a8d69)). |

The fluvoxamine row's previous half-life was not tied to the selected arm,
and its 84% absolute bioavailability was contradicted by the primary paper's
statement that absolute bioavailability was unknown; both were removed. The
amantadine half-life was aligned to the same 24-person label arm. No curve
values were added. The upstream hydroxyzine FDA-extraction record had also
misattributed [Simons et al. 1984](https://pubmed.ncbi.nlm.nih.gov/6141198)'s
72.5 ng/mL **serum** peak after 0.7 mg/kg (mean 39 mg) to a 25 mg dose; it
is marked superseded so integration cannot restore that mismatched pair.

The unchanged fitted system now gives public-profile N=79 development Meta
AAFE **2.9312** (conditional bootstrap 95% CI **2.3983–3.6464**), Engine
**3.8694**, direct ML **3.2683**, and descriptive in-domain Meta **3.0149**
(N=65). Paired Meta/ML ratio **0.8968** (CI **0.7922–1.0106**) includes 1.
The development-residual nominal 90% half-width remains 10.24×, covering
72/79. The movement from 2.9442 is reference correction, not a model gain
or independent validation.
