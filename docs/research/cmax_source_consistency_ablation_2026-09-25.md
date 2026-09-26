# Direct Cmax source-consistency ablation (2026-09-25)

**Decision: NO-GO.** A metadata-level filter for reported non-modified-release
formulations and unresolved salt annotations does not improve out-of-fold accuracy.
On identical scaffold folds and identical scored rows, the filtered model has 3.5%
higher AAFE than the existing full-data model (paired 95% CI of the ratio
0.985–1.088). Its score falls within ten equal-size random-subset controls; those
controls cannot rule out a smaller source-quality effect.
The production model, metadata and dataset are unchanged.

## Reproduce

```
.venv/bin/python scripts/cmax_source_consistency_ablation.py
```

The script fetches (if not cached in `~/.cache/sisyphus/omega_mmpk/`) and SHA256-verifies the
two pinned Omega workbooks listed in `scripts/retrain_cmax_public.py`, verifies the fitted
dataset SHA and row count against `models/direct_pk/xgboost_cmax.meta.json`, and writes:

| Artifact | SHA256 |
|---|---|
| `data/validation/cmax_source_consistency_ledger_2026-09-25.csv` (906 rows) | `841f686f…030aac` |
| `data/validation/cmax_source_consistency_ablation_2026-09-25.json` | `7ef04934…bbef10d` |

Two consecutive runs after the inclusion-rule correction produced byte-identical
artifacts (~11 s per run).

The initial 577-row calculation accepted drug groups with formulation recorded in
only some source arms. That violated the stated all-arm inclusion rule; its results
are superseded by the 516-row calculation below.

## Predeclared design

- **Population:** the 906 fitted rows of `data/training/cmax_omega_public_clean.csv`
  (already holdout-excluded, whole-blood rows removed, 114 administered/analyte
  mismatches and two quarantined rows removed, and dose/unit adjudications applied by the production recipe).
- **Row→source mapping:** each fitted row is mapped to its source arm group by
  (analyte name, source dose). The mapping is accepted only if the arm count equals
  `n_studies` and the arms' geometric mean Cmax reproduces the original Omega aggregate
  (rtol 1e-6). All 906 rows passed; any failure raises.
- **Inclusion rule** (fixed before modeling; no N=73, P0, P1, N28 or N50 data consulted):
  a row is included only if all of the following hold.
  1. **Salt annotation:** no salt reported in any source arm, or the production recipe
     applied an adjudicated salt→parent conversion. A reported salt without conversion is
     excluded as `dose_mass_basis_unverified`. The workbook has no dose-basis field:
     neither a blank salt field nor an annotated salt proves the administered mass basis.
  2. **Formulation reported** in every source arm (`formulation` column non-empty).
  3. **No reported modified release:** no arm labelled `tablet ER`, `capsule ER`, `ER`,
     `enteric-coated tablet` or `OCAS tablet`. The model has no formulation input.
- **Frozen model:** 2057 `compute_features` inputs, the shipped hyperparameters and the
  target log10(Cmax/dose) are read from the shipped metadata and dataset.
- **Comparison:** 5 Murcko-scaffold folds (`scaffold_split_indices`, the same folds as the
  shipped CV). Every arm is scored on the same 516 included rows, each predicted by a
  model that never saw that row's scaffold fold.
  - *full_data*: train on all 906 rows minus the test fold.
  - *filtered_data*: train on the 516 included rows minus the test fold.
  - *random control*: train on a random 516 of 906 rows (seeds 0–9) minus the test fold.
- **Uncertainty:** paired row bootstrap of the filtered/full AAFE ratio (2000 resamples,
  seed 42). The random control gives the spread due to training-set size alone.

## Source inventory (906 fitted rows)

| Attribute | Row-level evidence in the source | Counts |
|---|---|---|
| Parent-plasma analyte/matrix | **None.** Neither workbook has a matrix or analyte column. The production recipe removed known blood-matrix and analyte-mismatch rows, but the remaining rows are not individually verified here. | `matrix_evidence = not_recorded_in_source` for 906/906 |
| Oral route | None at row level. Oral-only by workbook scope. | not row-verifiable |
| Dose mass basis | `salt` column; production-recipe conversions | 649 no salt reported; 10 salt converted to parent; 247 salt reported, basis unverified |
| Formulation | `formulation` column; every source arm must be populated | 726 reported non-modified release (tablet, capsule, solution, suspension, …); 11 modified release; 169 have ≥1 unreported arm |
| Fed/fasted | Free-text `comments` only | 10 rows have ≥1 arm commented fed/food (idelalisib, capecitabine, panobinostat, palovarotene, amifampridine, ritonavir, stiripentol, pibrentasvir, acemetacin, viloxazine); **fasted state is never recorded**, so it is unverifiable for 896 rows |

Exclusions: 210 basis unverified; 138 formulation unreported; 31 both; 6 basis
unverified and modified release; 5 modified release. **Included: 516 / 906 (57.0%).**

Because fasted status and matrix cannot be verified for any row, a subset with fully
source-verified parent-plasma, fasted oral Cmax does not exist. Food context was
inventoried but was not an inclusion criterion. Requiring it would leave zero rows.

## Result (N = 516 scored rows, identical across arms)

| Arm | Train N | AAFE | Geometric bias | % within 2-fold | % within 3-fold | R² (log) |
|---|---|---|---|---|---|---|
| full_data (existing) | 906 | **3.282** | 0.946 | 36.8 | 55.2 | 0.329 |
| filtered_data | 516 | 3.396 | 1.083 | 37.4 | 55.6 | 0.294 |
| random 516 control (10 seeds) | 516 | mean 3.479 (range 3.368–3.628) | — | — | — | — |

- **Paired AAFE ratio (filtered/full):** 1.0348, 95% CI [0.9849, 1.0882].
- Filtered has lower absolute error than full on 252 of 516 rows (49%). Mean increase
  in |log10 error| is 0.0149 (SE 0.0106).
- Filtered versus the size-matched random control: 3.396 lies inside the control range
  (3.368–3.628), 3rd-lowest of 11 when ranked with the 10 seeds. This small
  control does not establish whether the filter has an effect beyond sample size.
- Geometric bias moves from 0.946 (full) to 1.083 (filtered), but AAFE does not improve.
- Sanity check: the shipped metadata reports all-906-row scaffold-CV AAFE 3.382, and the
  full-data arm restricted to the 516 included rows gives 3.282; the scored
  populations differ, so these AAFE values should not be compared as model gains.

Largest per-drug changes, as log10 error (predicted − observed); positive means
overprediction:

| Filtered worse | full | filtered | Filtered better | full | filtered |
|---|---|---|---|---|---|
| haloperidol | +0.37 | +1.20 | rifaximin | +3.19 | +2.08 |
| dihydro-α-ergocryptine | +0.41 | +1.23 | lamotrigine | −0.91 | −0.24 |
| ebastine | +0.88 | +1.68 | dydrogesterone | +1.15 | +0.50 |
| fidaxomicin | +0.50 | +1.17 | etrasimod | −1.20 | −0.62 |
| etilefrine | +0.09 | +0.75 | artesunate | +1.12 | +0.55 |

All 516 per-drug OOF errors, including the random-control mean, are in the results JSON
under `per_drug_oof_log10_error`.

## Limitations

- The inclusion rule reflects what the source can verify, not verified label quality.
  The source records only whether a salt or formulation was reported. Rows with no salt
  reported may still use salt-basis doses, and the 247 excluded salt rows may already
  use parent-equivalent doses. For example, the production recipe notes that cysteamine
  is already reported in parent mass. No row was relabelled from drug names.
- Fed/fasted status and matrix are not row-verifiable, so the most clinically important
  consistency axes could not be tested.
- The 516 scored rows are a non-random subset. The metrics above compare arms only on
  that subset and are not comparable to the headline N=73 benchmark.
- Only 5 folds and one fixed hyperparameter set were used. Nothing was tuned for the
  filtered set, by design, so a re-tuned filtered model was not tested.

## Go/no-go

**No-go.** The metadata-level filter does not beat full-data training
(paired ratio 1.035, CI includes 1 and leans worse). Do not change the
production training set on this basis. This result does not test source-verified
correction of individual dose or Cmax labels.
