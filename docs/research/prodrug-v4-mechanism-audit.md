# Prodrug v4 mechanism audit (2026-10-29)

Scope: `prodrug-validation-gate-xfail`, the three strict expected failures in
`tests/regression/test_prodrug_v2_validation_gate.py`. The referenced v3
literature note is absent from this public worktree; the registry's v3 metadata,
the test gate, and the linked primary sources were reviewed instead.

## Existing routing

`reference_man.yaml` already defines CES2 and ALPI abundance at `gut_wall`,
and SPR at `gut_wall`, `liver`, and `kidney`. The registry already gives
tebipenem pivoxil a CES2 affinity, fostamatinib an ALPI affinity, and
sepiapterin an SPR affinity. `augment_for_active_species` creates one activation
edge per node/enzyme intersection. There is no missing enzyme tag to add for
these three drugs, and no engine change is needed to discover these sites.

The ALPI abundance is traceable to [Al-Majdoub et al. 2021, Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC8048492/):
3.89 ± 3.52 pmol/mg intestinal mucosal protein. That paper does **not** report
CES2, despite the existing CES2 YAML comment citing it. The CES2 abundance and
all three substrate affinities remain estimates, not measured human
drug/enzyme kinetics. No numeric value was changed in this audit.

## Evidence and missing data

| Drug | Verified mechanism | Missing measurement for a defensible change |
| --- | --- | --- |
| Sepiapterin | Human [SPR kinetics](https://pmc.ncbi.nlm.nih.gov/articles/PMC3696693/) are reported for recombinant enzyme; [Smith et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30922814/) and [Gao et al. 2024](https://pubmed.ncbi.nlm.nih.gov/38156759/) show extensive conversion to BH4. | Human tissue SPR abundance and the presystemic fraction/clearance of sepiapterin or BH4 are not quantified by those sources. The gate observes **parent sepiapterin**, so an unmeasured BH4 first-pass loss alone cannot justify correcting its parent Cmax. |
| Tebipenem pivoxil | The [UTEBZI label](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=f0511453-5f6f-4e3a-b7d0-a5cfde985656) says hydrolysis occurs in gastric/intestinal fluid and/or enterocytes through intestinal esterases. | The label does not identify CES2 as the exclusive isoform or give a human CES2/tebipenem-pivoxil Vmax and Km. It also gives no rate for fluid-phase hydrolysis. The registry CES2 affinity is class extrapolated. |
| Fostamatinib | [Matsukane et al. 2022](https://pubmed.ncbi.nlm.nih.gov/35781630/) and the [phase I study](https://pubmed.ncbi.nlm.nih.gov/27858108/) place cleavage by intestinal alkaline phosphatase at the enterocyte brush border. | Neither source provides ALPI/fostamatinib Vmax and Km. The registry ALPI affinity is class extrapolated; moving activity to lumen nodes without measured surface-accessible abundance and kinetics would add unsupported parameters. |

The current active-species graph sends converted mass from each site directly
to the active plasma pool. Representing BH4 transit and depletion through the
portal vein and liver would require active-species tissue compartments and
measured first-pass clearance. A fractional loss chosen from parent or active
Cmax would be gate fitting, so it was not introduced.

## Reproduction

Public worktree, `PYTHONPATH=$PWD/src`, `n_mc_samples=0`,
`pytest -q tests/regression/test_prodrug_v2_validation_gate.py -rx`:

| Drug | Observed Cmax (mg/L) | Before predicted (mg/L) | Before fold | After fold |
| --- | ---: | ---: | ---: | ---: |
| Sepiapterin (parent) | 0.0024 | 13.897 | 5790.25× | 5790.25× |
| Tebipenem (active) | 4.01 | 0.3693 | 10.86× | 10.86× |
| R406 (active) | 0.61 | 0.04830 | 12.63× | 12.63× |
| Remdesivir (parent control) | 4.38 | 5.894 | 1.35× | 1.35× |

The gate remains one pass and three expected failures. No physiology, registry,
graph, or engine values changed. Non-prodrug controls at 100 mg oral were
unchanged: caffeine 1.350027703 mg/L and warfarin 2.583685895 mg/L, a 0.0%
benchmark effect for these controls. The full benchmark was not rerun because
there is no numerical model change.
