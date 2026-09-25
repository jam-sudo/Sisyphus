# Clinical reference structure and dose screen — 2026-09-24

This is a reference-data audit, **not** an independent Cmax validation. The
machine-readable [screen](../../data/validation/reference_structure_screen_2026-09-24.json)
records the pre-correction local InChIKey, PubChem CID, InChIKey, and formula for
each `clinical_pk.json` name. The screened file SHA256 was
`366f605002b467a1f708534077a2b78cf3cb8e09f9785262f03b2ecf8b243ce1`;
the first corrected file SHA256 was
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
full-key differences at that checkpoint. Do not replace them wholesale from a name
lookup; adjudicate the administered chemical and original study first.

| Record | Finding | Correction |
|---|---|---|
| `d_amphetamine` | Local SMILES encoded **(R)**, while [PubChem CID 5826](https://pubchem.ncbi.nlm.nih.gov/compound/5826) identifies d-amphetamine as **(S)**. The 0.058 mg/L Cmax was extrapolated from a claimed 10 mg value to 20 mg, not an observed 20 mg arm. The cited [PMID 12818571](https://pubmed.ncbi.nlm.nih.gov/12818571/) is an unrelated mitochondrial paper. The [Adderall IR label](https://www.dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=f22635fe-821d-4cde-aa12-419f8b53db81&type=display) describes a 3:1 d/l mixture and 10-to-30 mg dose proportionality, but does not verify that 20 mg Cmax. | Corrected the enantiomer; removed the extrapolated Cmax and its synthetic concentration curve from the observed reference pool. Retained the row as `unverified` for provenance. Its 20 mg product dose is **not** a d-amphetamine base-equivalent dose. |
| `dextroamphetamine` | Local SMILES omitted stereochemistry. The [dextroamphetamine sulfate label](https://www.dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=54b122b0-2ec8-43c9-86e8-a3538a83c776&type=display) reports 36.6 ng/mL after **three** 5 mg immediate-release tablets, while the reference recorded 5 mg total. | Corrected to (S)-amphetamine and 11.008 mg active-base equivalent for 15 mg sulfate (`15 × 270.42 / 368.49`). The observed 0.0366 mg/L remains unchanged. |

## 2026-09-25 stereochemistry follow-up

The named administered drugs below had an exact connectivity-key match but an
unspecified full stereochemical key in `clinical_pk.json`. Their product or
chemical descriptions identify a specific isomer. We used the named PubChem
compound's isomeric SMILES, then independently confirmed with RDKit that its
full InChIKey matches the source record. Doses and observed PK values did not
change.

| Reference | Identity evidence | Corrected full InChIKey |
|---|---|---|
| `atorvastatin` | [PubChem CID 60823](https://pubchem.ncbi.nlm.nih.gov/compound/60823); [Lipitor description](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=a60cc18b-0631-4cf0-b021-9f52224ece65) specifies stereochemistry | `XUKUURHRXDUEBC-KAYWLYCHSA-N` |
| `clarithromycin` | [PubChem CID 84029](https://pubchem.ncbi.nlm.nih.gov/compound/84029); the reference's [Health Canada product monograph](https://pdf.hres.ca/dpd_pm/00050373.PDF) identifies clarithromycin | `AGOYDEPGAOXOCK-KCBOHYOISA-N` |
| `entacapone` | [PubChem CID 5281081](https://pubchem.ncbi.nlm.nih.gov/compound/5281081); [DailyMed description](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ee02a04a-4b42-4cfa-8d9d-b1459c0ee9fd) identifies the administered `(E)` isomer | `JRURYQJSLYLRLN-BJMVGYQFSA-N` |
| `isotretinoin` | [PubChem CID 5282379](https://pubchem.ncbi.nlm.nih.gov/compound/5282379); [DailyMed description](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=189c014a-1d66-4ba9-ae47-9ed6138ca82b) identifies 13-cis-retinoic acid | `SHGAZHPCJJPHSC-XFYACQKRSA-N` |
| `naproxen oral` | [PubChem CID 156391](https://pubchem.ncbi.nlm.nih.gov/compound/156391); [DailyMed description](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=c38ae25f-d73e-4f85-b450-87df87e989da) identifies `(S)`-naproxen | `CMWTZPSULFXXJA-VIFPVBQESA-N` |

The updated `clinical_pk.json` SHA256 is
`74adfc1da32cf23ef1414ebdfd0075825e12f702f8144b984e5526537f967839`.
Recomputing against the 2026-09-24 PubChem-name snapshot now gives 88
same-connectivity full-key differences and 10 rows without local SMILES. The
snapshot also shows four connectivity mismatches: the three previously
adjudicated cases plus `ulipristal`, intentionally changed later to the
administered acetate ester while its snapshot lookup still names the free
parent. The 88 are triage leads, including intentionally unspecified mixtures,
not 88 confirmed errors. Among current rows with observed Cmax, the three
remaining full-key differences are
[`ketoconazole`](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2022/214133Orig1s000MultidisciplineR.pdf)
(racemate), `ranitidine`
(E/Z identity not specified by its arm), and
[`tramadol`](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=2993f70d-2359-443e-8d7b-68de5332f708)
(racemic parent measurement); assigning PubChem's single stereoisomer to
their mixed or unspecified arms would create a false identity.

The screen compares chemical names with PubChem structures; it does not verify
every dose, Cmax value, formulation, analyte, or primary source in the other
records. The remaining stereo differences and records without SMILES remain
explicit reference-quality work. These corrections do not create an
unconsumed external cohort or change the 73-compound development result.
