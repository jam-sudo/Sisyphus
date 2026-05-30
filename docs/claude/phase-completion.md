---
last_updated: 2026-04-20
parent: ../../CLAUDE.md
charter: Archive of shipped Sisyphus phases and tracks with scope + test count at ship time
---

# Phase Completion Log

Milestones that are **done** and no longer represent active work. Kept for archaeology and onboarding context. For the currently-shipped numbers, see the metrics table in top-level [CLAUDE.md](../../CLAUDE.md).

## Shipped phases

- **Phase 0 (Skeleton)** — Graph + YAML builder + flow conservation. First CI green.
- **Phase 1 (Engine v0.1)** — ODE compiler, 6 flux types, LSODA solver, MC propagation. Validated against Omega ODE output for midazolam/warfarin/caffeine (±5%).
- **Phase 2 (Prediction v0.2)** — `predict/` chemistry + ADME + IVIVE, `ml/` XGBoost ensemble + meta-learner, pipeline, MC uncertainty, CLI. Meta AAFE 2.058 at ship, N=61. 12 TDC ADME models.
- **Phase 3 (Extensibility v0.3)** — SC injection + pediatric model + tumor compartment via YAML only, `engine/` diff = 0, 17/17 tests.
- **Phase 4 (Production v1.0)** — DDI (22 tests), PK/PD link effect compartment + sigmoid Emax (28 tests), deterministic predict 414 ms mean (target ≤500 ms), CLI `ddi` command.

## Shipped tracks (clinical / research)

- **Track B (Clinical)** — multi-dose v2.0 (DosingRegimen, event-driven solver, ConcentrationProfile) + TDM v2.1 Bayesian update (importance sampling).
- **MIPD** — TDM posterior → dose recommendation (`sisyphus dose-adjust`, 14 tests).
- **Track A (SBI amortizer)** — 50 drugs × 1000 θ NPE, cumulative IBIS speedup 36,097× on 5 anchors, 11/13 coverage-primary gate, Coverage-primary gate at 10pp nominal.
- **Track C1 (hierarchical SBI)** — per-(population, drug) EngineSimulator cache, adult + pediatric_5y populations, 2kθ NSF, 8/26 KS+coverage strict gate.
- **Track D1 + follow-up (neural surrogate)** — ensemble-std gate + hybrid routing. 5-anchor tournament 2.5× cumulative with correct accuracy on all drugs.
- **Track D2 (TDM CI calibration)** — empirical weighted quantile CI, conformal CI floor (`min_ci_half_width_fraction=0.5`), 12/15 ≈80% coverage on 5-drug × 3-scenario benchmark.
- **Phase 2.0.5 (SBI reparameterization)** — logit(fup) + acid/statin training expansion. SBI routing 10/13 → 12/13 SBC.
- **P4 Continuous Hierarchical Infrastructure (2026-04-16)** — continuous (BW, age) NPE on 4-pop grid × 13 drugs, 41/52 SBC pass. Validation spec complete.
- **P6 SBI likelihood reweighting (2026-04-19)** — `bayesian_update(method="sbi", sbi_reweight=True)` opt-in + per-drug routing. Morphine bias +52% → +2%.
- **P7 Ketorolac AD flag (2026-04-19)** — `HIGH_ACID_LOW_FUP` AD flag (pKa<5 AND measured fup<0.02). Ketorolac + ibuprofen flagged.
- **OATP Phase 1 (2026-04-15)** — ActiveTransportEdge scaffolding, pravastatin calibration.
- **OATP Phase 2A (data, 2026-04-20)** — 5 statins in `data/transporters/oatp1b1.json`.
- **OATP Phase 2B (SLCO1B1 phenotype, 2026-04-20)** — `predict/phenotype.py` transporter extension, CPIC activity scaling, 11 unit tests.
- **CYP phenotype layer (2026-04-14)** — `sisyphus tdm --phenotype CYP2D6:PM` with CPIC activity scaling. 17 tests.
- **Multi-obs SBI (2026-04-14)** — log-normal likelihood post-hoc importance reweighting.

## P4.5 — Correlated Physiology Prior Infrastructure (2026-04-22)

Shipped: `Distribution.correlation_group` field, `sisyphus.physiology` package with
`CorrelationSpec` + `sample_correlated` + `assert_sampled`, `generate_physiology(rng=)`
opt-in, `reference_man.yaml` liver node migrated with Achour 2021 CVs.

Gates A-E passed. SBC demonstration intentionally deferred to P4.5a (requires
amortizer retraining with sampled physiology).

Meta AAFE 2.6946 (±0.001 of 2.695 headline — Gate A invariant).
OATP1B1 sampled independently per Achour 2021 empirical r<0.3 threshold.

Files: spec `2026-04-22-achour-abundance-correlation-design.md`,
plan `2026-04-22-achour-abundance-correlation.md`,
`data/physiology/achour2021_liver_abundance.csv`,
`data/physiology/achour2021_correlation.json`.

## CLI surface (shipped)

`sisyphus {predict, simulate, tdm, ddi, dose-adjust, benchmark}`.
