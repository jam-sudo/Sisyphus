# P0 source adjudication before prediction unsealing

Date: 2026-09-23. All 186 frozen candidates and 452 arms have had a first-pass
original-source review. Source Cmax values have been read, but the sealed
prediction file has not been opened or compared with them. This note clarifies
the source-eligibility rule in `self_run_pilot_p0.md` before final arm selection.
It does not change the frozen candidate order, doses, predictions, or model.

- Require an original study, regulatory review, or study-results record that
  identifies the dosed parent, plasma analyte, dose, formulation, fasting
  state, and a numeric Cmax central estimate. A primary-paper abstract counts
  only when it explicitly provides every required item. An approximate value
  read from a curve without a defined statistic does not count.
- A conventional tablet, capsule, orally disintegrating tablet, or ordinary
  oral solution/suspension counts as immediate release when the source gives
  no modified-release mechanism. Osmotic, enteric, delayed, colon-targeted,
  sustained, and prolonged-release products do not. Unknown dosage form stays
  unresolved. Solubilizers and solid dispersions are recorded as formulation
  caveats, not automatic exclusions.
- Require an original-source fasted condition. A meal within four hours after
  dosing makes the arm ineligible even if the pre-dose period was fasted.
  The first dose of a repeated-dose study can count only when its Cmax is from
  the pre-accumulation interval before the second dose. Healthy older and
  postmenopausal adults count; an impairment-study control must explicitly be
  healthy and have normal relevant organ function.
- Dosed-parent identity must match the manifest structure. Exclude measured
  metabolites, prodrug activation products, whole-blood-only assays, and
  fixed-combination products. A salt/cocrystal formulation counts only if the
  original states the dose in parent active-moiety units matching the frozen
  manifest dose; otherwise leave it unresolved. Do not change a frozen dose.
- Record whether Cmax is arithmetic mean, geometric mean, model-based LSmean,
  or median. Any exact labelled central estimate can enter the primary pilot;
  report statistic type and a same-statistic sensitivity where sample size
  permits. Record assay-limit and source sample-size caveats. A later revision
  of the same regulatory interview form may replace a dead link when arm
  identity and non-Cmax PK metadata match.
- Use every source-eligible arm of a compound. If any remaining arm might be
  eligible but lacks mandatory evidence, leave the compound unresolved rather
  than scoring a favorable subset. Identical source cohorts listed under two
  FRDB IDs count once, keeping the first ID in frozen order. If more than 50
  compounds qualify, use the first 50 in frozen order; otherwise report the
  actual smaller sample. Freeze an arm-level adjudication manifest and its
  hash before opening the prediction file. Never decide eligibility using
  prediction error.

The original AI `verified` labels are a conservative sensitivity set, and
`verified` plus `verified_with_caveat` is an inclusive ceiling, not the primary
cohort. Final inclusion follows the rules above and must record each override.
