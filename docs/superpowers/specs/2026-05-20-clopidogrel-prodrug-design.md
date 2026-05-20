# Clopidogrel Prodrug Registry Entry — B-03

**Date**: 2026-05-20
**Backlog**: B-03 — Clopidogrel (#11 residual)
**Prerequisite**: B-04 multi-enzyme prodrug yield schema shipped 2026-05-19

## Goal

Add clopidogrel to `data/sbi/prodrug_activation_registry.json` without introducing a drug-specific branch. The entry should represent the two competing hepatic fates:

- CES1 hydrolysis to inactive carboxylate: high-flux dead-end path, active yield 0.
- CYP oxidative bioactivation to the active thiol metabolite: low-flux active-producing path, active yield 1.

The 107-holdout reference for clopidogrel is parent clopidogrel Cmax. Therefore `observation_species` must remain `parent`; the active metabolite is simulated for mechanistic completeness only.

## Existing Ontology Constraint

Sisyphus currently does not carry an independent `CYP2C19` physiology tag in `data/physiology/reference_man.yaml`. The existing DrugBank normalization maps CYP2C19 and CYP2C8 to the Sisyphus `CYP2C9` 2C-subfamily surrogate, while CYP3A4 is represented directly.

For this B-03 implementation:

- `CES1` models the inactive esterase path.
- `CYP3A4` models the represented CYP3A path.
- `CYP2C9` models the existing Sisyphus 2C-subfamily surrogate, including CYP2C19 contribution under the current ontology.

Adding a true `CYP2C19` physiology abundance is out of scope because it would alter global hepatic clearance routing for every CYP2C19-annotated drug, not just clopidogrel.

## Registry Shape

The clopidogrel registry entry uses B-04 per-enzyme yields:

```json
"enzyme_affinity_for_conversion": {
  "CES1": {
    "mean": 0.030,
    "cv": 0.70,
    "yield": {"mean": 0.0, "cv": 0.0}
  },
  "CYP3A4": {
    "mean": 0.030,
    "cv": 0.70,
    "yield": {"mean": 1.0, "cv": 0.30}
  },
  "CYP2C9": {
    "mean": 0.030,
    "cv": 0.70,
    "yield": {"mean": 1.0, "cv": 0.30}
  }
}
```

At current physiology means this gives an approximate intrinsic path split of:

- CES1: `8.0e7 * 0.030 * 6e-5 = 144 L/h`
- CYP3A4: `9.2475e6 * 0.030 * 6e-5 = 16.6 L/h`
- CYP2C9 surrogate: `6.48e6 * 0.030 * 6e-5 = 11.7 L/h`

Active-producing intrinsic capacity is therefore about 16% of the represented conversion capacity, matching the literature statement that roughly 10-15% of absorbed clopidogrel enters the active thiol pathway while most is hydrolyzed by CES1.

## Active Species Disposition

The active species is the clopidogrel active thiol metabolite (DrugBank DBMET01163; average MW 355.836, formula C16H18ClNO4S). Disposition is `ceiling_accepted`:

- the active thiol is chemically labile,
- plasma assays often require stabilization or derivatization,
- covalent P2Y12 binding makes a simple linear 1C CL/V model a mechanistic approximation.

Because `observation_species="parent"`, active CL/V uncertainty does not define the holdout scoring target.

## Double-Count Control

Add clopidogrel to `data/transporters/cyp_clearance_overrides.json` with `metabolic_fraction=0.0`. This zeroes the default XGBoost-derived parent hepatic metabolic path so the explicit ProdrugActivationEdges carry clopidogrel parent hepatic consumption. Without this, parent clearance is double-counted.

## Acceptance Gates

1. Registry seed/schema tests pass, including multi-enzyme all-yield rule.
2. `lookup_active_metabolite()` returns per-enzyme yields for CES1, CYP3A4, and CYP2C9.
3. `predict(clopidogrel, 300 mg, oral)` succeeds with parent-observation semantics and nonzero parent Cmax.
4. `scripts/run_engine_benchmark.py --save-json data/training/4track_holdout_predictions.json` is re-run and the aggregate delta is documented.

## Documentation Updates

- `docs/claude/experiment-log.md`: append B-03 result and benchmark delta.
- `docs/claude/backlog.md`: remove B-03 after promotion/shipment.
- `README.md`: update validation metrics only if the public-clone benchmark moves materially; otherwise document the prodrug registry expansion and fix the stale irinotecan limitation sentence.
