# Product Scope and Architecture

## Supported production surface

Sisyphus production scope is one task: structure-only Cmax prediction from
canonical parent SMILES and an oral dose. The authoritative output is
`CmaxPrediction`: final Cmax, component tracks, effective weights, residual
interval, structural applicability flags, execution status, and resource
profile.

`EngineSimulation` is a separate mechanistic research output. Its curve, Cmax,
Tmax, AUC, and half-life come from one coherent PBPK solve. A Meta Cmax must never
be presented as the peak of the engine curve or combined with engine AUC/Tmax
without endpoint-source labels.

The web product exposes prediction and development evidence only. DDI, PGx,
PKPD, multi-dose TDM, and MIPD modules remain importable research APIs but are
not production-supported or clinically validated workflows.

## Shared execution context

Every graph-solving entry point uses `prepare_simulation_context`. It performs
chemistry/ADME calculation, non-CYP and transporter disposition detection,
drug-on-graph construction, reference-enzyme snapshotting, phenotype scaling,
active-species augmentation, axial expansion, hepatic-fu contract validation,
compilation, and deterministic mean realization. CLI simulation/TDM/DDI/dosing
must not maintain a second simplified builder.

## Error and applicability semantics

Structural AD membership is not an empirical confidence probability. Public
results therefore use `medium`/`low` only in the legacy confidence adapter and
expose `in_applicability_domain` plus flags directly. A future confidence level
requires an out-of-fold error-risk model and independent calibration.

`execution_status` distinguishes `ok` from explicit degraded states such as
`degraded_missing_engine` and `degraded_missing_ml`. `predict(strict=True)` is mandatory for benchmark
generation and audited deployments; interactive use may retain explicit ML
fallbacks.

PGx results distinguish requested, actually applied, and unsupported tags.
Isoforms absent from the physiology graph are not merged into family members and
are never reported as applied.

## Resource profiles

Runtime paths resolve through `ResourceConfig`, never cwd. The default `public`
profile ignores gitignored DrugBank and residual-logP artifacts even when present
locally. Licensed enrichment requires `SISYPHUS_PROFILE=licensed_research` and is
not eligible for a public benchmark. `SISYPHUS_ROOT` points installed code to a
versioned data/model bundle.

## Experimental module gates

- DDI requires dynamic perpetrator exposure, unbound inhibition, TDI/induction,
  and an external clinical-pair benchmark before production promotion.
- PGx requires distinct supported isoforms and genotype/phenotype clinical
  validation; abundance scaling alone is mechanistic exploration.
- TDM/MIPD requires temporally external real-patient validation, posterior
  coverage, target attainment, and a population-PK comparator.
- Linear dose scaling is rejected when saturable metabolism is present unless an
  experimental override is explicit; production nonlinear dosing requires
  forward simulation across candidate doses.
- PKPD presets remain illustrative until independently validated.

## Accuracy decision

N=107 is development data and N=28 is a consumed temporal challenge. No further
architecture or weight decision may claim external evidence from them. The next
production decision follows `docs/validation/external_holdout_v1_protocol.md`.
If Meta does not beat direct ML by the pre-registered margin, direct ML becomes
the default and the engine remains the mechanistic research layer.
