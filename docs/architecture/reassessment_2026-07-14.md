# Repository Reassessment — 2026-07-14

## Verdict

The supported product is now narrowly defined as oral SMILES + dose → population
Cmax for screening and ranking. The live API, CLI production command, web console,
model card, resource loader, and external-evaluation contract implement that same
scope. DDI, PGx, PKPD, TDM, MIPD, multi-dose, and IV simulation remain experimental
research APIs and are not presented as validated product workflows.

The implementation/governance gate is **10/10**: every identified repository-level
correctness, provenance, deployment, input-contract, interval-semantics, and
holdout-procedure issue has an executable control and a passing test. This score
does not mean predictive validity is 10/10. Independent external AAFE and interval
coverage remain unavailable until the custodian-controlled protocol is executed.

## Closed findings

1. Oral-only product input is enforced; invalid routes, non-finite/non-positive
   doses, oversized structures, and extra API fields fail closed.
2. Engine, ML, CL/F, and VDss tracks are reported only when valid for the route;
   IV cannot silently use oral-only hybrid tracks.
3. Development residual and Monte Carlo parameter intervals are separate fields.
   The residual band is never called split-conformal or independent coverage.
4. Runtime resources are root/profile resolved. Active model manifests are
   SHA-pinned and fail closed on model or feature-code mismatch.
5. Public-profile provenance excludes optional licensed artifacts and accompanies
   every prediction.
6. N=107 is classified as repeatedly accessed development data; N=28 is a consumed
   temporal challenge; the contaminated N50 is prohibited from reporting.
7. External-holdout manifest, source plan, labels, and predictions use executable
   JSON Schemas. Audit, prediction, and scoring stages bind hashes and refuse
   contamination, eligibility drift, source-quota failure, degraded execution,
   or post-freeze file changes.
8. The new acquisition design enumerates about 900 identities, verifies at least
   550, allocates 120–150 calibration records, freezes N=260 final test records,
   and seals a reserve. Final labels remain custodian-controlled.
9. The web product contains only prediction and development evidence; preset doses
   cannot be client-rescaled, and curves/endpoints come from one coherent solve.
10. Wheel installation, locked dependencies, API contract, web build/smoke, JSON/
    YAML parsing, tracked-file hygiene, and the full Python suite pass.

## Evidence boundary

- Development Meta AAFE: 2.743 on N=107, useful for diagnosis only.
- Consumed temporal challenge: N=28, diagnostic only.
- Independent external AAFE: not available.
- External residual-interval coverage: not available.
- Clinical dose-setting authorization: explicitly absent.

The next scientific decision is therefore binary and preregistered: retain the
Meta/PBPK production path only if the blinded external comparison beats direct ML
under the gates in `docs/validation/external_holdout_v1_protocol.md`; otherwise
direct ML becomes the production default and PBPK remains a research layer.

## Verification snapshot

- Full Python suite: 1329 passed, 25 skipped, 3 expected failures, 1 non-strict
  expected-failure pass; zero failures.
- External-holdout focused suite after final freeze-manifest addition: 17 passed.
- Web production build and 12 smoke assertions: passed.
- Installed-wheel strict oral prediction from outside the source tree: passed.
- Ruff on the CI-gated production/test/script surface: passed.
- Dependency, JSON/YAML, diff-whitespace, public-data parity, and internal-reference
  checks: passed.
- Local Docker execution: not run because Docker is unavailable in this host;
  container build is enforced in CI.
