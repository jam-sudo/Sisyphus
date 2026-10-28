# Engine Audit Findings — 2026-06-04

Scope: audit only. No code changes were made as part of this review.

## Executive Summary

The engine does not appear fundamentally broken because it descended from Omega. The larger issue is that Omega parity previously masked at least one shared physiological bug, and several engine-level contracts remain under-specified after the recent FLUX/RBP corrections.

The highest-risk area is not numerical integration or graph compilation. It is the semantic contract between:

- whole-blood vs plasma concentration basis,
- hepatic ECM clearance vs hepatic unbound-fraction correction,
- named clearance models vs the actual implemented equations,
- SciPy RHS vs pure-JAX RHS behavior.

For Beta-track readiness, these should be treated as correctness blockers or near-blockers before further performance tuning.

## Findings

### P1 — Blood-Pool Observation May Report Blood Concentration as Plasma Concentration

Current `solver` output stores blood-pool concentrations as `A / V`. After the RBP concentration-basis correction, this is best interpreted as whole-blood concentration for blood-pool nodes. However, `pk.endpoints.compute_endpoints()` treats the observation-node series as plasma concentration and derives Cmax/AUC directly from it.

This creates a likely factor-of-RBP error when `RBP != 1`.

Evidence:

- `src/sisyphus/engine/solver.py`
  - `solve()` stores `concentration = amount / volume` for each node.
  - `solve_mc()` follows the same pattern.
- `src/sisyphus/pk/endpoints.py`
  - `compute_endpoints()` documents and computes PK endpoints from "plasma concentrations" but receives the raw observation-node series.
- The RBP concentration-basis design discussed reporting-basis handling, but left its endpoint contract unresolved.
- `tests/unit/test_rbp_concentration_basis.py`
  - Tests cover flow and clearance basis, but not final reporting conversion.

Observed probe:

- Minimal graph with only `venous_blood`, volume `5 L`, IV amount `100 mg`, `RBP = 2`.
- Current reported Cmax: `20`.
- Expected plasma-basis Cmax if blood concentration is converted by RBP: `10`.

Expected impact:

- Headline holdout may not move if production ADME clamps or routes most RBP values to `1`.
- Measured ADME, curated compounds, and future datasets with real RBP values can be materially affected.
- This can make performance appear worse or better depending on the sign of existing error cancellation.

Recommended action:

- Define the public observation contract explicitly: is `PredictionResult.cmax` plasma, blood, or observation-node-native?
- If the intended output is clinical plasma Cmax, convert blood-pool observation nodes from blood basis to plasma basis before endpoint computation.
- Add a regression test with `RBP != 1` that pins endpoint reporting, not only flux behavior.

### P1/P2 — Hepatic `fu_correction_applicable` Is Not Applied by Extended ECM Clearance

The reference physiology marks liver with `fu_correction_applicable: 1.0`, and the YAML comment says hepatic intracellular fu correction is applied by `ClearanceFlux` and `ProdrugActivationFlux`. But the production liver clearance edge uses the `extended` model, and the extended ECM branch does not apply `fu_correction_liver`.

Evidence:

- `data/physiology/reference_man.yaml`
  - Liver node includes `fu_correction_applicable: 1.0`.
  - Liver clearance edge uses `model: extended`.
- `src/sisyphus/engine/flux.py`
  - `well_stirred` and `parallel_tube` branches apply `fu_correction_liver` when the node flag is enabled.
  - `extended` branch uses `fup` directly.
- `tests/unit/test_flux_fu_correction_integration.py`
  - Test comments indicate ECM was outside the original B-11 hepatic fu-correction scope.

Expected impact:

- If `fu_correction_liver` is non-identity, the standard hepatic ECM path can silently ignore a correction that the YAML claims should apply.
- This is a semantic correctness problem even if current default parameters make the numerical effect small or zero.

Recommended action:

- Decide whether ECM should include the hepatic intracellular fu correction.
- If yes, implement and test it specifically for `extended`.
- If no, update YAML/comments/docs so the model contract does not imply otherwise.

### P2 — `parallel_tube` Is Exposed as a Model but Does Not Implement True Parallel-Tube Extraction

`parallel_tube` is available as a clearance model type, but the current implementation behaves like a well-stirred/intrinsic sink approximation rather than a true parallel-tube extraction model with axial concentration decay.

Evidence:

- `src/sisyphus/graph/types.py`
  - `ClearanceModel` includes `"parallel_tube"` as a valid model.
- `src/sisyphus/engine/flux.py`
  - The `parallel_tube` branch uses the same intrinsic-clearance sink shape as `well_stirred`.
  - Comments note that true PT needs an axial gradient and that `parallel_tube` is not wired in the reference graph.

Expected impact:

- No headline impact if reference physiology does not use `parallel_tube`.
- Platform users can get misleading results if they select `model: parallel_tube` expecting canonical parallel-tube behavior.

Recommended action:

- Either fail loudly for `parallel_tube` until implemented, rename it to reflect the approximation, or implement a real parallel-tube formulation.
- Add a test that protects the chosen semantics.

### P2 — Pure-JAX RHS Can Diverge from SciPy RHS for Clearance Semantics

The public JAX solver path appears to wrap the NumPy RHS, but the pure-JAX RHS implementation is not semantically complete relative to the SciPy RHS.

Evidence:

- `src/sisyphus/engine/rhs_jax.py`
  - Fails loudly for `extended` ECM, which is good.
  - Implements `well_stirred` and `parallel_tube` without hepatic `fu_correction_liver`.
- `src/sisyphus/engine/params_jax.py`
  - `JaxParams` does not carry `fu_correction_liver` or `fu_correction_applicable`.
- `src/sisyphus/engine/flux.py`
  - SciPy RHS applies hepatic fu correction in supported branches.

Expected impact:

- Production SciPy path is probably unaffected.
- Pure-JAX experiments on non-ECM graphs with flagged hepatic correction can silently drift from SciPy results.

Recommended action:

- Treat pure-JAX RHS as experimental unless parity is explicitly tested.
- Add SciPy-vs-pure-JAX parity tests for each supported clearance branch and node flag combination.
- Either add missing parameters to `JaxParams` or fail loudly when unsupported semantics are requested.

### P3 — ActiveTransport Semantics Are Target-Local but Documentation Mentions Efflux/Secretion Use Cases

The ActiveTransport implementation reads transporter abundance and IVIVE scaling from the target node. The test suite intentionally protects target-local uptake semantics. However, the docstring mentions gut efflux, renal secretion, and BBB efflux, which imply source-to-target directional transport semantics.

Evidence:

- `src/sisyphus/engine/flux.py`
  - ActiveTransport docstring lists uptake and efflux-style examples.
  - Implementation reads `node_transporters(self.target_name)` and target IVIVE scaling.
- `tests/unit/test_active_transport.py`
  - Explicitly documents and tests target-node transporter semantics.
- Reference physiology currently has no `active_transport` edges, so this is a future-platform risk rather than current headline risk.

Expected impact:

- Low current numerical risk.
- High future usability risk if users add active transport edges based on the docstring.

Recommended action:

- Clarify the contract as target-local uptake only, or add explicit transport direction/mode fields.
- Add tests for any supported efflux/secretion semantics before exposing them as platform features.

### P2 Documentation Gap — README Engine Validation Still Reads Like Full Omega Parity

The current integration test comments state that midazolam/propranolol no longer use Omega parity targets because Omega shared the flow-limitation double-count bug. The README engine-validation table still reads like all four drugs remain Omega parity checks.

Evidence:

- `tests/integration/test_engine_validation.py`
  - Notes that high-extraction midazolam/propranolol now use FLUX-1-corrected Sisyphus snapshots.
- `README.md`
  - Engine validation table still presents the older Omega parity framing.

Expected impact:

- This can mislead reviewers into thinking the engine is still fully validated against Omega.
- More importantly, it obscures the lesson that Omega parity is not a sufficient correctness oracle.

Recommended action:

- Update validation documentation after reconciling against the current benchmark artifacts.
- State explicitly which checks are Omega parity and which are Sisyphus regression snapshots after known Omega/Sisyphus shared-bug correction.

## What Looks Healthy

- The compiler/solver architecture remains broadly sound:
  - identity-blind engine design,
  - graph-topology compilation,
  - mass-conservation orientation,
  - YAML-time flow validation.
- The recent FLUX-1 extraction-ceiling correction is covered by targeted regression tests.
- RBP dynamics tests now cover flow and clearance concentration basis.
- Targeted audit test run passed:

```text
49 passed, 1 skipped
```

Command used:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  tests/unit/test_flux.py \
  tests/unit/test_ecm_flux.py \
  tests/unit/test_active_transport.py \
  tests/unit/test_rbp_concentration_basis.py \
  tests/integration/test_engine_validation.py \
  -q -p no:cacheprovider
```

## Recommended Fix Order

1. Lock the observation concentration contract and fix/report-test RBP endpoint basis.
2. Resolve hepatic `fu_correction_liver` semantics for extended ECM.
3. Make `parallel_tube` either real, renamed, or fail-loud.
4. Add SciPy-vs-pure-JAX parity gates or restrict pure-JAX semantics.
5. Clarify ActiveTransport directionality before exposing it as platform YAML.
6. Reconcile README validation language with current FLUX-1-era test targets.

## Bottom Line

The engine should not be discarded simply because it inherited data or validation expectations from Omega. The current evidence points to a more specific problem: a few physiological contracts are still ambiguous, and those ambiguities can create error cancellation or hidden performance ceilings.

For commercial-tool-level performance, the next step should be engine contract hardening before additional ML or calibration layers are trusted.
