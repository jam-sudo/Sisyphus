# B1.x — multi-species engine-convection feasibility spike

> **Context.** The parked "transported reactive-metabolite" follow-up, **reframed by the 2026-06-18 ultrathink**. The original framing (deepen the reactive-metabolite/zonal-tox thread by convecting the metabolite downstream) is physiologically wrong for its own use case: a *reactive* metabolite is **consumed locally** (high reactivity → short diffusion length → centrilobular binding — exactly why APAP necrosis is zone-3 and why the B1/GSH-pool **post-processor is the correct model there**). Convection matters for **stable / active metabolites** (active-metabolite Cmax contribution, sequential metabolism, metabolite-mediated DDI, metabolite systemic PK), not for DILI. So this is a **general multi-species engine-capability spike**, justified by stable-metabolite PK. Deliverable: a yes/no feasibility verdict + a Damköhler map of when convection matters — not a product. Harness-isolated; headline **2.731 bit-identical**.

## 1. The question

**Can the engine carry a second chemical species (a metabolite) through a per-zone formation → convection → consumption chain, using graph construction with EXISTING flux types only (zero `src/sisyphus/engine/` change), conserving mass?** And, as the decision payoff: **across the Damköhler number (consumption/transport ratio), when does downstream convection materially move the metabolite's spatial profile versus the local-only post-processor (B1/B1.x)?**

**Hard stop-rule (harness-isolation boundary):** attempt the construction with existing flux types + graph construction only. If it cannot build / compile / solve without touching `src/sisyphus/engine/` (or the builder/validation), **STOP and report "needs a new convective-species flux / multi-species engine support"** — itself a valid verdict. Do **not** build the engine change in this spike.

## 2. What is actually under test (and what is not)

`transit` with rate `k = Q/V` is **mathematically identical** to flow-convection between well-stirred tanks (both are the `(Q/V)·A` first-order inter-tank transfer the parent axial chain already uses). So **transport is the trivial part** — transit edges move mass downstream by construction. The non-trivial things the spike genuinely tests:

- **(T1) Multi-species compilation.** Does `ODECompiler` create independent state variables for metabolite nodes and assemble a correct RHS when a second species' nodes coexist with the parent's?
- **(T2) Per-zone formation.** Do **N** distinct `prodrug_activation` edges `parent__axi → metabolite__axi` (one per zone) compile — sidestepping the single-target-node "trap" (one edge for the whole organ) by graph construction? (`ProdrugActivationFluxSpec` routes `parent(source) → active(target)` with `mw_active/mw_parent` scaling, drawing on the *source* zone's enzymes — confirmed `flux.py:775`.)
- **(T3) Multi-role nodes.** Does a node that is simultaneously a prodrug **target**, a transit **source**, and a clearance **source** compile and solve?
- **(T4) Builder/validation acceptance.** Does the builder accept metabolite nodes + a transit chain in this topology? (Flow-conservation validation checks only `FlowEdge` in≈out per node, so transit-based convection is not flagged — `body.py:98`; the spike confirms nothing else rejects it.)
- **(T5) Mass conservation.** Does the engine's own `SimResult.mass_balance_error` stay below tolerance for the multi-species solve?

**Explicitly out of scope (reported, not built):** the engine's state is one drug's concentration across nodes, so a metabolite node carrying **distinct molecular properties** (its own `fup`/`Kp`/`CLint` ≠ parent) is a *separate* capability. This spike uses **molecule-independent transport** — geometric `transit` (edge-level `k=Q/V`) and a first-order consumption — so it tests the *topology + mass balance*, and **reports** whether distinct per-species properties are expressible or need engine support (a likely "needs engine support" sub-finding). The convection physics does not depend on molecule-specific params, so this scoping does not weaken the feasibility answer.

## 3. The construction (existing flux types, synthetic skeleton)

A minimal synthetic two-species axial skeleton, built directly via the graph API (the `_axial_graph` pattern), `N` zones inlet→outlet:
- **parent chain:** a source → `parent__ax1..N` (the existing axial convection + a parent clearance), giving each zone a parent concentration `c_parent,i(t)`;
- **formation:** `N` `prodrug_activation` edges `parent__axi → metabolite__axi`, zonatable enzyme (the bioactivation gradient, via `zonation_weights`);
- **convection:** `N−1` `transit` edges `metabolite__axi → metabolite__ax(i+1)`, rate `k_conv = Q/V_zone`;
- **consumption:** `N` first-order consumption edges on `metabolite__axi`, rate `k_detox` (the swept quantity);
- **outlet efflux:** `metabolite__axN → sink`, captured for the balance.

The **Damköhler number** per zone is `Da = k_detox / k_conv`. The sweep varies `k_detox` (hence `Da`) across the reactive→stable range, holding `k_conv` fixed.

## 4. The verdict & pre-registered gates

**A YES verdict requires all of:**
- **G-build/solve (T1–T4):** the chain builds (passes validation), compiles, and solves without any `src/sisyphus/engine/` or builder change. (If not → STOP, "needs engine support" verdict.)
- **G-mass (T5):** `SimResult.mass_balance_error < tol` (the engine's own conservation invariant), across the sweep.
- **G-convection-correct:** (a) vs a **no-convection control** (`k_conv → 0`: metabolite formed + consumed locally, the local-only / post-processor analog), the convected metabolite profile shifts **downstream** (peak-zone or center-of-mass moves toward the outlet) in the low-Da regime; (b) a **uniform-formation** case in the low-Da regime accumulates monotonically inlet→outlet (plug-flow accumulation).

**The decision map (centerpiece, the payoff):** a **Damköhler sweep** reporting, per `Da`, the metabolite profile's center-of-mass shift (convected − local-only):
- **Da ≫ 1 (reactive regime):** shift ≈ 0 — engine and local-only **agree** → **directly validates that the B1/GSH-pool post-processor is correct for reactive metabolites**.
- **Da ≪ 1 (stable regime):** shift is material toward the outlet — the engine captures downstream transport the post-processor misses → **this is where multi-species engine modeling earns its complexity** (stable/active-metabolite PK).
- The crossover `Da ≈ 1` is the reported boundary.

**Honest-negative paths:** if the construction needs an engine change (STOP verdict), or `mass_balance_error` does not close, or the Damköhler map shows convection never matters even at low Da (would itself be surprising), report as-is. No parameter tuned to force a YES.

## 5. Components (all new, harness-isolated)
- **New** `scripts/spike_multispecies_convection.py`: build the synthetic 2-species axial graph, solve across the `Da` sweep + the no-convection control + the uniform-formation case; compute `mass_balance_error`, center-of-mass shift, the Da map, and the distinct-per-species-properties probe (attempt + report); write the verdict report. Reuses the `_axial_graph` construction pattern + `zonation_weights`.
- **New** `tests/integration/test_multispecies_convection_spike.py`: G-build/solve (compiles + solves), G-mass (`mass_balance_error < tol`), G-convection-correct (downstream shift vs control at low Da; high-Da agreement), Da-map monotonicity (shift decreases as Da grows), headline-isolation guard (`4track_holdout_predictions.json` untouched; `test_cached_holdout_aafe_is_2p731` + `test_mm_headline_bit_identity` pass). **If the STOP verdict triggers**, the test asserts the documented failure mode (build/compile error) instead, so the spike's negative outcome is itself pinned.
- **New** `data/validation/multispecies_convection_spike_2026-06-18.{json,md}`: the YES/NO verdict, the T1–T5 findings, the distinct-per-species-properties sub-finding, and the Damköhler decision map (when engine-convection matters vs when the post-processor suffices).
- experiment-log entry; **dead-ends entry** if the verdict is NO (needs engine change) — recording exactly which flux/builder limitation blocks pure-graph multi-species, so a future engine-side project is scoped.

## 6. Non-goals
No `src/sisyphus/engine/` or builder change (hard stop-rule). No distinct per-species molecular properties (reported, not built). No zonated/saturable detox (a first-order rate is enough for the transport question). No GSH-pool coupling, no quantitative tox. No reactive-metabolite DILI claim (the post-processor owns that; this spike's high-Da arm *validates* it). Not reusing the full IVIVE `_axial_graph` clearance complexity — a purpose-built minimal skeleton isolates the multi-species question.

## 7. Constraints (operational)
Harness-isolated; **zero** `src/sisyphus/engine/` / builder / `predict()` / `reference_man.yaml` change (the entire premise — a change means the STOP verdict, not a code edit); headline **2.731 bit-identical**. NO fitting; NO cherry-picking (gates pre-registered; honest-negative + STOP paths explicit). Commit as `jam-sudo <jam-sudo@users.noreply.github.com>`, NO Claude/AI trailer, `git commit --no-verify`. Tests with `/opt/miniconda3/bin/python -m pytest`. `ruff check src tests scripts` line-length 100. Reuses the axial machinery (PR #79), `zonation_weights` (DE-50), and the existing flux registry (`prodrug_activation`/`transit`/`clearance`).
