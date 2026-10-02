# Development covalent-ester identity follow-up — 2026-09-24

The scored `ulipristal` record paired the ulipristal structure
(`C28H35NO3`, [PubChem CID 13559281](https://pubchem.ncbi.nlm.nih.gov/compound/13559281))
with **30 mg / 176 ng/mL** from the [FDA ella label, Section 12.3 Table 2](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=2bf93d23-cddd-4613-9066-5b5fa090404b).
That label actually reports a single fasted **30 mg ulipristal acetate** tablet
in 20 healthy women and measures **ulipristal acetate** parent in plasma
(arithmetic mean Cmax 176 ± 89 ng/mL). [PubChem CID 130904](https://pubchem.ncbi.nlm.nih.gov/compound/130904)
identifies the acetate as `C30H37NO4`, InChIKey
`OOLLAFOLCSJHRE-ZHAKMVSLSA-N`. It is a covalent acetate ester, not an ionic
counterion; dividing the dose by a molecular-weight ratio would still leave
the wrong input molecule and analyte.

The historical reference key and its 107-compound development-split slot
remain `ulipristal`; its explicit drug name and input SMILES now identify
the PubChem **ulipristal acetate ester**, and the exact 30 mg / 0.176 mg/L
arm remains scored. The former non-acetate SMILES is retained solely as a
training-exclusion key, because existing fitted public ADME datasets excluded
that structure. This is an identity repair in the
already used development set, not a newly acquired test compound. The
archived structure screen had correctly matched the old SMILES to the
*name* ulipristal but did not test whether the PK label measured that molecule.
The old curated and FDA-extraction records are superseded to prevent them
from restoring the mismatched parent structure.

The fitted models and score cohort size remain unchanged. Public-profile
development N=79 Meta AAFE is now **2.8940** (conditional compound-bootstrap
95% CI **2.3695–3.6057**), Engine **3.8023**, direct ML **3.3043**, and
descriptive in-domain Meta **3.0056** (N=64; the acetate ester is flagged
outside the applicability domain). The paired Meta/ML ratio is **0.8758**
(conditional CI **0.7703–0.9925**). That interval falls below 1, but the
cohort has repeatedly informed system selection and cannot establish
independent superiority. The movement from 2.9040 is a reference correction,
not a model gain.
