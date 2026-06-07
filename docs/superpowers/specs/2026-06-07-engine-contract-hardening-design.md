---
date: 2026-06-07
status: design — approved for spec review
parent: ../../../CLAUDE.md
supersedes_audit: docs/engine_audit_findings_2026-06-04.md (findings 2–6; finding 1/RBP-2 already shipped)
---

# Engine Contract Hardening — Design

## Theme

Every advertised engine contract must equal its implementation. Where physics and
architecture permit, implement the real behavior (true parallel-tube topology, JAX↔SciPy
parity, active-transport directionality). Where physics or architecture forbid faking it,
fail loud (the extended-ECM `fu_correction` drop). No silent divergence, no misleading
docstrings, no dead claims.

This closes findings 2–6 of `docs/engine_audit_findings_2026-06-04.md` (verified by the
meta-review `docs/engine_audit_review_2026-06-04.md`). Finding 1 (blood-vs-plasma reporting
basis) already shipped as RBP-2; only its documentation echo remains (WS-6).

## Invariants honored (load-bearing)

- **#1 identity-blind engine** — no name matching introduced anywhere in `engine/`. The
  builder may generate derived node names (`<organ>__ax{i}`); the engine still sees them as
  opaque.
- **#8 hard no-touch** — `engine/compiler.py` and `engine/solver.py` are NOT modified.
  Editable engine modules: `flux.py`, `rhs_jax.py`, `params_jax.py`, `solver_jax.py`,
  `uncertainty.py`, plus a new `engine/contracts.py`. Editable elsewhere: `graph/types.py`,
  `graph/body.py`, `graph/builder.py`, `pipeline/predict.py`, tests, docs.
- **#4 flow conservation** — preserved by construction in WS-3 (proof below).
- **Headline 2.784 bit-identical** — across all five workstreams (justification below). Guarded
  by the existing `test_cached_holdout_aafe_is_2p784`.

### Why the headline cannot move

- Reference physiology (`data/physiology/reference_man.yaml`) has **no** `parallel_tube`
  clearance edge and **no** `active_transport` edge → WS-3 and WS-5 touch zero production code
  paths. The liver uses `model: extended`; the gut uses `well_stirred`; the kidney uses
  `gfr_filtration`.
- All 19 `hepatic_fu_correction` registry entries are `mean = 1.0` → WS-2's guard never fires
  for any current drug.
- SciPy is the production backend; JAX is opt-in/experimental → WS-4 changes only the non-
  production backend.
- WS-6 is documentation only.

---

## WS-2 — Extended-ECM `fu_correction` engine-level fail-loud

### Problem
The liver node is flagged `fu_correction_applicable: 1.0` and its clearance edge uses
`model: extended`. The extended (ECM) branch **intentionally** does not apply
`fu_correction_liver` (the ECM models concentrative hepatic uptake explicitly via
`PS_active/PS_inf/PS_eff`; multiplying the empirical `fu_inc/fu_plasma` factor on top would
double-count). This is correct physics — but a curated non-identity value that lands *only* on
extended/gfr clearance is **silently dropped**. `pipeline.predict` currently warns; the user's
decision is to make the engine **fail loud**.

`fu_correction` is **not** dead, however: the same flagged liver node honors it via
`ProdrugActivationFlux` (e.g. clopidogrel). So a value that is honored *somewhere* on the node
is correct, not a violation.

### Predicate (false-positive-free)
Raise **iff** a flagged node (`fu_correction_applicable > 0`) carries a non-identity resolved
`fu_correction_liver` (`!= 1.0`) **and has no `fu_correction`-honoring flux** — i.e. no
`well_stirred`/`parallel_tube` clearance edge and no `prodrug_activation` edge originate from
that node. (Extended/gfr-only ⇒ the value is *entirely* dropped ⇒ violation. Prodrug or
well_stirred coexisting ⇒ honored ⇒ no raise.)

### Implementation
New `engine/contracts.py`, two functions — operating on the **graph** (not per-sample params),
which is both genuinely param-free and avoids a `cv>0` pitfall (see note):
- `flagged_nodes_without_honoring_flux(graph) -> list[str]` — **graph-static** analysis over
  `graph.nodes` (each carries the `fu_correction_applicable: float` field) and `graph.edges`:
  for each node flagged (`> 0`) that is the **source** of a `ClearanceEdge` with `model in
  {extended, gfr_filtration}`, return it iff no `ClearanceEdge(model in {well_stirred,
  parallel_tube})` and no `ProdrugActivationEdge` also originate from that node.
- `assert_fu_correction_honored(graph, fu_correction_liver_mean: float) -> None` — calls the
  above, then raises `ValueError` if the set is non-empty **and** `fu_correction_liver_mean
  != 1.0`. Message names the offending node(s) and the dropped value.

**Check the mean, validate once.** The intent lives in `drug.fu_correction_liver.mean` (the
registry's anti-fudge guard enforces `mean >= 1.0`). Do **not** test a per-sample realized
value: `ResolvedParams.drug_param` returns `.mean`, but under MC the *drug is sampled first*,
so a hypothetical `mean=1.0, cv>0` entry would realize `!= 1.0` per sample and spuriously trip
a per-sample guard. Passing the original `mean` makes the check sample-independent and run once.

Identity-blind: uses node flags + edge types/models only, never names.

### Wiring (no-touch respected)
- `engine/uncertainty.py` `propagate` / `propagate_fast`: call `assert_fu_correction_honored(
  graph, drug.fu_correction_liver.mean)` once before the sample loop (uses the *original*
  graph + drug, before realization).
- `pipeline/predict.py`: replace the `_fu_correction_drop_warning` *warning* with a call that
  raises (fail fast at the API boundary, friendly message). It already has `graph` and
  `drug.fu_correction_liver.mean`.
- **Honest limitation**: raw `engine/solver.py::solve` (no-touch low-level API) is not
  auto-guarded; document that the contract is enforced at the engine's orchestration layer
  (`uncertainty`) and the production pipeline, which are the legitimate solve entry points.

### Error handling
`ValueError` (graph/input authoring inconsistency — matches CLAUDE.md "graph validation
failure → ValueError"), not a structured low-confidence result.

### Tests
- extended-only flagged node + non-1.0 → raises.
- extended + coexisting `prodrug_activation` on the same flagged node + non-1.0 → passes.
- extended + non-1.0 but node **not** flagged → passes.
- `fu_correction_liver == 1.0` → passes.
- **Regression guard**: no current holdout/registry drug trips the guard (asserts headline
  path unaffected); `test_cached_holdout_aafe_is_2p784` still passes.

---

## WS-3 — Real parallel-tube via axial sub-compartment expansion

### Principle (math verified)
N well-stirred sub-tanks in series, each with `volume/N` and `CLint/N`, full blood flow `Q`
passing through serially, converge to the parallel-tube extraction. Per-tank pass-through
(blood basis):

```
C_i / C_{i-1} = 1 / (1 + fu_b·CLint / (N·Q)) ,   fu_b = fup / RBP
C_N / C_0     = [1 + fu_b·CLint/(N·Q)]^(-N)  ->  exp(-fu_b·CLint/Q)
E             = 1 - C_N/C_0                   ->  1 - exp(-fu_b·CLint/Q)   (parallel-tube)
```

At `N = 1` this reduces to the engine's emergent well-stirred extraction
`E = fu_b·CLint/(Q + fu_b·CLint)`. The engine itself is unchanged — extraction emerges from the
post-FLUX-1 split of intrinsic-clearance sink + convective `Q·c_out` edge, per tank.

**Transient (not just steady-state) correctness.** Tanks-in-series is the canonical plug-flow-
with-dispersion approximation: the chain's residence-time distribution is gamma (mean
`V·Kp/Q`, conserved under the split), sharpening toward a pure transport delay as N→∞. So the
expansion reproduces the parallel-tube *transient* — relevant because Cmax/AUC are transient —
not merely the steady-state extraction ratio.

### Builder expansion (engine diff = 0 lines)
- Add `Node.axial_subcompartments: int = 1` to `graph/types.py` (default 1 = current
  behavior).
- A new builder step (in `graph/builder.py`, or a pure transform `expand_axial(graph)` in
  `graph/body.py`) detects a `ClearanceEdge` with `model == "parallel_tube"`. It expands that
  edge's **source organ** into `N` serial sub-tanks (`N = node.axial_subcompartments`, default
  **10** when `parallel_tube` is requested without an explicit N):
  - Create `N` tank nodes. **Extensive** properties divided by N: `volume/N`, `enzymes/N`,
    `transporters/N`. **Intensive** properties copied unchanged: `ivive_scaling`,
    `fu_correction_applicable`, `node_type`. ⇒ per-tank `CLint = ivive·Σ(abundance/N·affinity)
    = CLint_total/N`, matching the convergence derivation.
  - **Kp/PS resolution (critical):** set each tank's `lookup_name` to the parent organ's
    `lookup_name` (or the parent name if unset). `ResolvedParams._build_kp_map`
    (`compiler.py:182`) resolves Kp via `node.lookup_name` — the existing facility (e.g.
    `"adipose"` for `"adipose_tissue"`) — so `drug_kp("liver__ax3")` returns the liver Kp.
    Without this the tanks default to `Kp=1.0` and corrupt the distribution volume / residence
    time. Engine stays identity-blind (it reads `lookup_name`, never matches a literal name).
  - Redirect every inbound `FlowEdge` (`→ organ`) to `tank_1`; redirect every outbound
    `FlowEdge` (`organ →`) from `tank_N`.
  - Add `N-1` internal `FlowEdge`s `tank_i → tank_{i+1}` with `flow_rate = Q_total` (sum of the
    organ's resolved inbound flow rates).
  - Replace the single `parallel_tube` clearance edge with `N` `well_stirred` clearance edges
    `tank_i → <sink>`.
  - `remove_node(organ)`.
- **N rationale**: N is a numerical discretization for PT convergence, NOT tuned to Cmax loss
  (invariant #8 is about fudging to loss). Default 10 chosen for convergence (<~2% deviation
  from analytic `E_PT` at typical CLint); documented as such.
- **Bit-identity guard**: `expand_axial` early-returns the *unchanged* graph object (same node /
  edge ordering) when no `parallel_tube` edge exists. Production graphs are untouched ⇒ no
  `sample()` RNG-order shift, headline bit-identical.

### Flow conservation proof (invariant #4)
The validator (`BodyGraph.validate`) checks per-node `inflow ≈ outflow` over `FlowEdge.flow_rate`
(absolute L/h), skipping `sink`/`lumen`. After expansion:
- External nodes (e.g. `portal_vein`, `arterial_blood`, `venous_blood`) only have edges
  *redirected* (same `flow_rate`, new endpoint) ⇒ their balance is unchanged.
- Each internal tank: `Q_total` in = `Q_total` out ⇒ balanced.
- The original organ is removed ⇒ not validated.
∴ conservation holds by construction.

### Scope guard (fail-loud)
Axial expansion supports **perfusion organs whose only edges are flow-in / flow-out /
clearance**. If an organ tagged for expansion also has `diffusion`, `prodrug_activation`,
`active_transport`, or `one_compartment_elimination` edges → `NotImplementedError` with
guidance. (Production liver is `extended`, untouched.)

### Flux-layer cleanup
- Remove the now-unreachable single-tank `parallel_tube` branch from `flux.py`
  `ClearanceFluxSpec.apply` and the `cl_pt` branch from `rhs_jax.py` (the builder guarantees no
  `parallel_tube` edge survives to compile).
- Defensive: if a `parallel_tube` clearance edge reaches `ClearanceFluxSpec.from_edge`
  unexpanded → `ValueError` ("parallel_tube must be expanded via axial_subcompartments before
  compile").
- Update the `graph/types.py` `ClearanceEdge` docstring (line ~137) and `flux.py` docstring.
- Consistency with WS-2: once the `parallel_tube` flux branch is gone, drop the (now-inert)
  `parallel_tube` reference from `contracts.py::flagged_nodes_without_honoring_flux`'s
  honoring set, leaving `{well_stirred clearance, prodrug_activation}`.

### Tests
- **Convergence**: minimal `blood_in → organ(N) → blood_out` + clearance-to-sink graph; assert
  realized extraction → `1 - exp(-fu_b·CLint/Q)` as N grows (e.g. N=20 within tolerance), and
  N=1 matches well_stirred `fu_b·CLint/(Q+fu_b·CLint)`.
- **Conservation**: expanded graph passes `BodyGraph.validate()` (zero violations).
- **Identity-blind**: randomize all node names, expansion + solve produce identical numerics.
- **Scope guard**: expanding an organ with a diffusion/prodrug edge raises.
- **Fallout migration**:
  - `tests/unit/test_flux_fu_correction_integration.py` — PT-branch direct-construction tests
    rewritten against `well_stirred` (identical gating) or the new expansion mechanism.
  - `scripts/run_chain_benchmark.py` — D/E/F (`parallel_tube`) now produce genuine PT extraction
    (numbers change, intended); add `axial_subcompartments` where needed.

> WS-3 is the largest workstream; it may be split into its own implementation plan / PR.

---

## WS-5 — ActiveTransport directionality

### Problem
Both SciPy (`flux.py`) and JAX (`rhs_jax.py`) read transporter abundance/IVIVE from the
**target** node (uptake-local), while the docstring advertises efflux/secretion use cases
(gut P-gp, renal OAT/OCT, BBB P-gp) whose transporters sit at the **source**.

### Physics (verified)
- **uptake** (default): transporter at **target**, driving (substrate) concentration from
  **source**, IVIVE from target. (OATP-style: pump on the target organ membrane, kinetics
  driven by the source-side blood conc.) = current behavior.
- **efflux**: transporter at **source**, driving concentration from **source**, IVIVE from
  source. (P-gp in gut_wall pumps gut_wall→lumen on the gut_wall intracellular conc.)
- In both cases the driving concentration is the **source** (the compartment drug leaves); only
  the transporter-bearing node (and its IVIVE) switches.

### Implementation
- Add `ActiveTransportEdge.direction: str = "uptake"` to `graph/types.py` (`"uptake"` |
  `"efflux"`; default preserves current behavior).
- `flux.py` `ActiveTransportFluxSpec`: select `transporter_node = target if direction ==
  "uptake" else source`; read abundance + IVIVE from `transporter_node`; keep driving conc from
  source.
- `rhs_jax.py` / `params_jax.py`: split active-transport edges into uptake/efflux index sets at
  build time (mirroring the clearance-model split); JAX mirrors the node selection.

### Honest caveat
No production graph instantiates `active_transport`; there is no validation oracle. The
deliverable is **contract + backend-parity correctness**, not numerical validation against PK
data. The existing "IVIVE handles unit conversion" convention is preserved as-is (out of
scope to re-derive).

### Tests
- uptake reads target transporters; efflux reads source transporters; both directions produce
  the physically expected sign/magnitude on a minimal graph.
- SciPy↔JAX agree for both directions (folded into WS-4's parity suite).
- default (`direction` unset) == current behavior (no regression).

---

## WS-4 — JAX ↔ SciPy parity

### Gaps (current)
1. JAX `well_stirred`/`parallel_tube` ignore `fu_correction_liver` (no field in `JaxParams`);
   SciPy applies it. (`parallel_tube` is removed by WS-3, so only `well_stirred` remains.)
2. JAX `active_transport` uses an **approximation** (aggregate `Vmax` + abundance-weighted
   `Km`, `params_jax.py:163–184`) that diverges from SciPy's exact per-transporter MM sum when
   transporters have differing `Km`. (Gap the audit did not call out.)
3. Direction (WS-5) must be mirrored.

### Implementation
- (1) Add `drug_fu_correction_liver: float` and `node_fu_correction_applicable: jnp.ndarray`
  to `JaxParams`; populate in `resolve_to_jax`; in the JAX `well_stirred` branch
  `fup_eff = jnp.where(node_fu_correction_applicable[src] > 0.5, fup*corr, fup)`. (Extended
  already fail-louds under JAX, so no ECM change.)
- (2) **Parity guarantee = no silent divergence.** Keep the aggregate path where it is *exact*
  (single transporter, or multiple with equal `Km`); **fail loud** (`NotImplementedError` in
  `resolve_to_jax`, params_jax.py) when a node has ≥2 active transporters with **distinct** `Km`.
  Placement note: `Km` is a per-drug value (`drug_transporter_km(tag)`) unavailable at
  `make_jax_rhs` build time (topology-only); the divergence is detectable only in
  `resolve_to_jax`, which already iterates per-node transporters with their `Km` (lines
  172–179). This achieves the parity goal with minimal code on a consumer-less experimental
  path. *(Optional alternative if full numerics are later wanted: exact padded per-transporter
  MM arrays.)*
- (3) Mirror WS-5 direction node-selection in JAX.

### Tests — comprehensive SciPy↔JAX parity suite
Compare at the **RHS level** — evaluate `make_jax_rhs(...)` vs `compiler.make_rhs(...)` on
identical `(t, y, params)` and assert `dydt` agreement to ~1e-9 (x64). Per branch on small
synthetic graphs: flow, well_stirred clearance (with and without `fu_correction`),
gfr_filtration, transit, absorption, diffusion, active_transport (uptake + efflux). RHS-level
(not integrated-trajectory) comparison isolates flux math from integrator differences (SciPy
`solve_ivp` vs the JAX solver), which would otherwise loosen the tolerance. A separate,
looser end-to-end Cmax check is optional. Plus: the distinct-`Km` multi-transporter case
fail-louds rather than silently diverging.

---

## WS-6 — README engine-validation reconciliation (docs only)

The README engine-validation table still frames all four drugs as Omega-parity checks. Per the
meta-review, **3 of 4 rows** are post-FLUX-1/RBP-2 Sisyphus regression snapshots (Omega shared
the flow-limitation double-count bug), and **RBP-2 is absent** from the README. Reconcile the
table against the current `tests/integration/test_engine_validation.py` targets: label each row
as Omega-parity vs Sisyphus-regression-snapshot, and add the RBP-2 basis note. No code.

---

## Sequencing

1. **WS-6** (docs, independent, low-risk).
2. **WS-2** (engine guard — `contracts.py` + wiring + tests).
3. **WS-3** (axial PT builder + single-tank PT removal in both backends + fallout migration).
   *Largest; candidate for its own plan/PR.*
4. **WS-5** (active-transport direction, SciPy + JAX).
5. **WS-4** (JAX `fu_correction` + MM divergence guard + comprehensive parity suite — validates
   everything after WS-3/WS-5 land).

## Planning prerequisites (verify before/at implementation)
- Read `graph/builder.py` for the exact `flow_fraction → flow_rate` conversion timing so WS-3's
  internal edges receive the correct `Q_total` (the conservation proof assumes absolute
  flow_rate, confirmed in `body.py::validate`).
- Confirm `run_chain_benchmark.py`'s liver config has only flow + clearance edges (else its
  `parallel_tube` configs hit WS-3's scope guard and need adjustment). Also: it builds graphs
  programmatically, so it must call `expand_axial` before compile (or hit the defensive guard) —
  i.e. expansion is a pre-compile transform, not solely a YAML-builder step.
- New dataclass fields (`Node.axial_subcompartments`, `ActiveTransportEdge.direction`) have
  defaults ⇒ keyword construction and `dataclasses.replace` (sample/realize_means) are
  back-compatible; scan tests for any *positional* Node/Edge construction that the new field
  ordering could break.

## Out of scope
- Re-deriving the active-transport unit-conversion model (WS-5 preserves it).
- Full exact padded multi-transporter MM in JAX (WS-4 uses fail-loud-on-divergence).
- Any change to production physiology, ML artifacts, or the meta-learner — the headline path is
  bit-identical.
