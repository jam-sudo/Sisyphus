# Clinical reference structure and dose screen — 2026-09-24

This is a reference-data audit, **not** an independent Cmax validation. The
machine-readable [screen](../../data/validation/reference_structure_screen_2026-09-24.json)
records the pre-correction local InChIKey, PubChem CID, InChIKey, and formula for
each `clinical_pk.json` name. The screened file SHA256 was
`366f605002b467a1f708534077a2b78cf3cb8e09f9785262f03b2ecf8b243ce1`;
the corrected file SHA256 is
`7a3ea5db58ce237631d894b228c58ffdd2b7f492a8830c5e1a653b2b1ddf026e`.
Queries used [PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
`compound/name/{name}/property/InChIKey,MolecularFormula,SMILES/JSON`, with
`d-amphetamine` as the documented alias for the local `d_amphetamine` key.

Of 331 entries, 10 lack local SMILES and 321 resolved to PubChem records. Three
connectivity-key differences—atovaquone, darunavir ethanolate, rifabutin—match
the previously adjudicated tautomer/formulation cases. The screen found 100
same-connectivity full-key differences before correction; these are **triage
leads**, not 100 errors. Names may denote mixtures, products, or ambiguous
stereochemistry. Two amphetamine records were corrected below, leaving 98
unadjudicated full-key differences. Do not replace them wholesale from a name
lookup; adjudicate the administered chemical and original study first.

| Record | Finding | Correction |
|---|---|---|
| `d_amphetamine` | Local SMILES encoded **(R)**, while [PubChem CID 5826](https://pubchem.ncbi.nlm.nih.gov/compound/5826) identifies d-amphetamine as **(S)**. The 0.058 mg/L Cmax was extrapolated from a claimed 10 mg value to 20 mg, not an observed 20 mg arm. The cited [PMID 12818571](https://pubmed.ncbi.nlm.nih.gov/12818571/) is an unrelated mitochondrial paper. The [Adderall IR label](https://www.dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=f22635fe-821d-4cde-aa12-419f8b53db81&type=display) describes a 3:1 d/l mixture and 10-to-30 mg dose proportionality, but does not verify that 20 mg Cmax. | Corrected the enantiomer; removed the extrapolated Cmax and its synthetic concentration curve from the observed reference pool. Retained the row as `unverified` for provenance. Its 20 mg product dose is **not** a d-amphetamine base-equivalent dose. |
| `dextroamphetamine` | Local SMILES omitted stereochemistry. The [dextroamphetamine sulfate label](https://www.dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=54b122b0-2ec8-43c9-86e8-a3538a83c776&type=display) reports 36.6 ng/mL after **three** 5 mg immediate-release tablets, while the reference recorded 5 mg total. | Corrected to (S)-amphetamine and 11.008 mg active-base equivalent for 15 mg sulfate (`15 × 270.42 / 368.49`). The observed 0.0366 mg/L remains unchanged. |

The screen compares chemical names with PubChem structures; it does not verify
every dose, Cmax value, formulation, analyte, or primary source in the other
records. The 98 remaining stereo differences and 10 records without SMILES
remain explicit reference-quality work. These corrections do not create an
unconsumed external cohort or change the 107-compound development result.
