# Engine Contract Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make WS-2/WS-5/WS-4/WS-6 of the engine contracts match their implementations — fail loud on the extended-ECM `fu_correction` drop, add active-transport directionality, close JAX↔SciPy parity gaps, and reconcile the README. (WS-3 axial parallel-tube is a separate plan.)

**Architecture:** Cross-edge contract checks live in a new `engine/contracts.py` invoked by the solve orchestrators (no compiler.py/solver.py edits — invariant #8). Active-transport gains a `direction` field honored by both backends. JAX gains `fu_correction` and a Michaelis-Menten divergence guard. All changes are headline-bit-identical: production physiology has no `parallel_tube`/`active_transport` edge and every `fu_correction_liver` registry value is 1.0.

**Tech Stack:** Python 3.10+, frozen dataclasses, NumPy, SciPy `solve_ivp`, JAX (x64), pytest, ruff (line length 100).

**Spec:** `docs/superpowers/specs/2026-06-07-engine-contract-hardening-design.md`

---

## File Structure

- `src/sisyphus/engine/contracts.py` — **NEW.** Graph-static contract validation (WS-2). One responsibility: cross-edge/cross-node checks a per-edge FluxSpec cannot make.
- `src/sisyphus/engine/uncertainty.py` — **MODIFY.** Call the WS-2 guard before the MC sample loop.
- `src/sisyphus/pipeline/predict.py` — **MODIFY.** Replace the `_fu_correction_drop_warning` *warning* with a raising guard.
- `src/sisyphus/graph/types.py` — **MODIFY.** Add `ActiveTransportEdge.direction`.
- `src/sisyphus/engine/flux.py` — **MODIFY.** Honor `direction` in `ActiveTransportFluxSpec`.
- `src/sisyphus/engine/params_jax.py` — **MODIFY.** Add `fu_correction` fields + MM-divergence guard.
- `src/sisyphus/engine/rhs_jax.py` — **MODIFY.** Apply `fu_correction` in well_stirred; split transport edges by direction.
- `README.md` — **MODIFY.** Reconcile the engine-validation table (WS-6).
- Tests: `tests/unit/test_engine_contracts.py` (NEW), `tests/unit/test_active_transport_direction.py` (NEW), `tests/unit/test_jax_scipy_parity.py` (NEW), plus an addition to `tests/unit/test_active_transport.py`.

**Headline guard (run after every task):** `pytest tests/regression/ -k cached_holdout -q` must stay green (the `test_cached_holdout_aafe_is_2p784` pin). The full headline path is bit-identical.

---

## Task 1: WS-6 — Reconcile README engine-validation table (docs only)

**Files:**
- Read: `tests/integration/test_engine_validation.py`
- Modify: `README.md` (the engine-validation table + surrounding prose)

- [ ] **Step 1: Read the current validation targets**

Run: `grep -n "midazolam\|propranolol\|warfarin\|caffeine\|Omega\|parity\|FLUX-1\|RBP\|snapshot" tests/integration/test_engine_validation.py`
Note, per drug row, whether the test asserts an **Omega-parity** target or a **post-FLUX-1/RBP-2 Sisyphus regression snapshot** (the test comments state this explicitly; the meta-review found 3 of 4 rows are now Sisyphus snapshots, warfarin included).

- [ ] **Step 2: Locate the README table**

Run: `grep -n "Engine validation\|midazolam\|propranolol\|Omega" README.md`

- [ ] **Step 3: Edit the README table**

For each of the 4 drug rows, append a parenthetical label matching Step 1: `(Omega parity)` or `(Sisyphus snapshot, post-FLUX-1/RBP-2)`. Add one sentence directly under the table:

```markdown
> Engine-validation targets: high-extraction drugs (midazolam, propranolol, warfarin) use
> post-FLUX-1/RBP-2 Sisyphus regression snapshots, not Omega parity — Omega shared the
> flow-limitation double-count bug FLUX-1 fixed, so Omega parity is no longer a correctness
> oracle for them. The blood:plasma concentration basis is RBP-2 (whole-blood pools reported
> on a plasma basis); see `docs/superpowers/specs/2026-06-04-rbp-concentration-basis-design.md`.
```

- [ ] **Step 4: Verify consistency**

Run: `grep -n "Omega parity\|Sisyphus snapshot\|RBP-2" README.md`
Expected: each of the 4 rows labeled; the RBP-2 note present. Confirm no row still implies blanket Omega parity.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(readme): reconcile engine-validation table with FLUX-1/RBP-2 era targets"
```

---

## Task 2: WS-2 — `engine/contracts.py` fu_correction guard

**Files:**
- Create: `src/sisyphus/engine/contracts.py`
- Test: `tests/unit/test_engine_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for engine-level contract validation (WS-2)."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.engine.contracts import (
    assert_fu_correction_honored,
    flagged_nodes_without_honoring_flux,
)
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    ClearanceEdge,
    Node,
    ProdrugActivationEdge,
)


def _node(name: str, flagged: bool) -> Node:
    return Node(
        name=name,
        node_type="organ",
        volume=Distribution(1.0),
        ivive_scaling=1.0e-4,
        fu_correction_applicable=1.0 if flagged else 0.0,
    )


def _graph_extended_only_flagged() -> BodyGraph:
    g = BodyGraph()
    g.add_node(_node("liver", flagged=True))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model="extended"))
    return g


def _graph_extended_plus_prodrug_flagged() -> BodyGraph:
    g = _graph_extended_only_flagged()
    g.add_node(Node(name="active", node_type="organ", volume=Distribution(1.0)))
    g.add_edge(
        ProdrugActivationEdge(
            source="liver",
            target="active",
            enzyme_tags=frozenset({"CYP3A4"}),
            conversion_yield=Distribution(1.0),
            mw_parent=300.0,
            mw_active=280.0,
        )
    )
    return g


def test_flagged_extended_only_is_an_offender():
    g = _graph_extended_only_flagged()
    assert flagged_nodes_without_honoring_flux(g) == ["liver"]


def test_prodrug_coexistence_is_not_an_offender():
    g = _graph_extended_plus_prodrug_flagged()
    assert flagged_nodes_without_honoring_flux(g) == []


def test_unflagged_extended_is_not_an_offender():
    g = BodyGraph()
    g.add_node(_node("liver", flagged=False))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model="extended"))
    assert flagged_nodes_without_honoring_flux(g) == []


def test_assert_raises_on_nonidentity_total_drop():
    g = _graph_extended_only_flagged()
    with pytest.raises(ValueError, match="entirely dropped"):
        assert_fu_correction_honored(g, fu_correction_liver_mean=1.4)


def test_assert_noop_on_identity_value():
    g = _graph_extended_only_flagged()
    assert_fu_correction_honored(g, fu_correction_liver_mean=1.0)  # no raise


def test_assert_noop_when_prodrug_honors_it():
    g = _graph_extended_plus_prodrug_flagged()
    assert_fu_correction_honored(g, fu_correction_liver_mean=1.4)  # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_engine_contracts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sisyphus.engine.contracts'`.

- [ ] **Step 3: Implement `engine/contracts.py`**

```python
"""Engine-level contract validation.

Cross-edge / cross-node checks a single per-edge FluxSpec cannot make (FluxSpecs
are identity-blind and see only their own edge). Invoked by the engine's solve
orchestrators (``uncertainty``) and the production pipeline before integration.
Does NOT touch ``compiler.py`` / ``solver.py`` (invariant #8).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sisyphus.graph.types import ClearanceEdge, ProdrugActivationEdge

if TYPE_CHECKING:
    from sisyphus.graph.body import BodyGraph

# Clearance models that do NOT apply the B-11 fu_correction_liver factor
# (extended ECM models hepatic uptake explicitly; gfr is a plasma sink).
_FU_CORRECTION_DROP_MODELS = frozenset({"extended", "gfr_filtration"})
# Clearance models that DO apply it (well-stirred family).
_FU_CORRECTION_HONORING_MODELS = frozenset({"well_stirred", "parallel_tube"})


def flagged_nodes_without_honoring_flux(graph: "BodyGraph") -> list[str]:
    """Return flagged nodes whose fu_correction would be *entirely* dropped.

    A node is returned iff: it is flagged ``fu_correction_applicable > 0``; it is
    the source of a ClearanceEdge with a drop-model (extended / gfr_filtration);
    and NO ClearanceEdge with a honoring model (well_stirred / parallel_tube) and
    NO ProdrugActivationEdge also originate from it.

    Identity-blind: inspects node flags + edge types/models only, never names.
    """
    flagged = {
        name for name, node in graph.nodes.items()
        if node.fu_correction_applicable > 0
    }
    if not flagged:
        return []

    has_drop: set[str] = set()
    has_honoring: set[str] = set()
    for edge in graph.edges:
        if isinstance(edge, ClearanceEdge) and edge.source in flagged:
            if edge.model in _FU_CORRECTION_DROP_MODELS:
                has_drop.add(edge.source)
            elif edge.model in _FU_CORRECTION_HONORING_MODELS:
                has_honoring.add(edge.source)
        elif isinstance(edge, ProdrugActivationEdge) and edge.source in flagged:
            has_honoring.add(edge.source)

    return sorted(has_drop - has_honoring)


def assert_fu_correction_honored(graph: "BodyGraph", fu_correction_liver_mean: float) -> None:
    """Raise ``ValueError`` if a non-identity fu_correction_liver is entirely dropped.

    No-op when the value is the identity ``1.0`` (the production default), so the
    headline path stays bit-identical. Check the *mean* (sample-independent),
    never a per-MC-sample realized value.
    """
    if fu_correction_liver_mean == 1.0:
        return
    offenders = flagged_nodes_without_honoring_flux(graph)
    if offenders:
        raise ValueError(
            f"fu_correction_liver={fu_correction_liver_mean:.3g} is entirely "
            f"dropped at flagged node(s) {offenders}: their clearance uses a "
            f"model that does not apply it (extended ECM / gfr_filtration) and no "
            f"well_stirred/parallel_tube clearance or prodrug_activation edge at "
            f"the node applies it either. Remove the curated value, switch the "
            f"node's clearance model, or model uptake via transporter params."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_engine_contracts.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/engine/contracts.py tests/unit/test_engine_contracts.py
git commit -m "feat(engine): contract guard for entirely-dropped fu_correction"
```

---

## Task 3: WS-2 — Wire the guard into the solve orchestrators

**Files:**
- Modify: `src/sisyphus/engine/uncertainty.py` (`propagate`, `propagate_fast` — add the call before the sample loop)
- Modify: `src/sisyphus/pipeline/predict.py:499-501` (replace warning with raise)
- Test: append to `tests/unit/test_engine_contracts.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/unit/test_engine_contracts.py`:

```python
def test_pipeline_raises_when_curated_value_would_be_dropped(monkeypatch):
    """A non-1.0 fu_correction_liver on the production (extended) liver raises."""
    import sisyphus.predict.hepatic_fu_correction as hfc
    from sisyphus.core import Distribution
    from sisyphus.pipeline.predict import predict

    # Force a non-identity hepatic fu_correction for any SMILES.
    monkeypatch.setattr(
        hfc, "lookup_hepatic_fu_correction", lambda smiles: Distribution(1.4, cv=0.0)
    )
    with pytest.raises(ValueError, match="entirely dropped"):
        predict("CCO", dose_mg=100.0, route="oral")
```

> Note: `ivive.py` imports `lookup_hepatic_fu_correction` *inside* the function (`ivive.py:680`), so patching the module attribute takes effect. If the import is module-level in your tree, patch `sisyphus.predict.ivive.lookup_hepatic_fu_correction` instead.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_engine_contracts.py::test_pipeline_raises_when_curated_value_would_be_dropped -q`
Expected: FAIL — `predict` returns a result (with a warning) instead of raising.

- [ ] **Step 3: Replace the pipeline warning with a raise**

In `src/sisyphus/pipeline/predict.py`, replace lines 495-501 (the `_fu_warn` block) with:

```python
        # WS-2 contract guard: a curated (non-1.0) fu_correction_liver that would
        # be ENTIRELY dropped (flagged node, drop-model clearance, no honoring
        # flux) is a contract violation — fail loud rather than silently no-op.
        # No-op today (every registry value is 1.0) → headline bit-identical.
        from sisyphus.engine.contracts import assert_fu_correction_honored
        assert_fu_correction_honored(graph, drug.fu_correction_liver.mean)
```

Then delete the now-unused `_fu_correction_drop_warning` function (lines 180-223) and its `_FU_CORRECTION_DROP_MODELS` constant (line 177) — they are superseded by `engine/contracts.py`.

Run: `grep -n "_fu_correction_drop_warning\|_FU_CORRECTION_DROP_MODELS" src/sisyphus/pipeline/predict.py`
Expected: no matches remain.

- [ ] **Step 4: Wire the guard into `uncertainty.py`**

In `src/sisyphus/engine/uncertainty.py`, add the import at module top:

```python
from sisyphus.engine.contracts import assert_fu_correction_honored
```

In `propagate`, immediately after the docstring (before `results: list[SimResult] = []`):

```python
        assert_fu_correction_honored(graph, drug.fu_correction_liver.mean)
```

Add the identical line at the same position in `propagate_fast`.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/unit/test_engine_contracts.py -q`
Expected: PASS (7 passed).

Run: `pytest tests/regression/ -k cached_holdout -q`
Expected: PASS (headline bit-identical — registry values are all 1.0, guard is a no-op).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/engine/uncertainty.py src/sisyphus/pipeline/predict.py tests/unit/test_engine_contracts.py
git commit -m "feat(engine): fail loud when curated fu_correction is entirely dropped"
```

---

## Task 4: WS-5 — `ActiveTransportEdge.direction` + SciPy efflux

**Files:**
- Modify: `src/sisyphus/graph/types.py:160` (add field)
- Modify: `src/sisyphus/engine/flux.py` (`ActiveTransportFluxSpec`: store + honor direction)
- Test: `tests/unit/test_active_transport_direction.py` (NEW)

- [ ] **Step 1: Write the failing test**

```python
"""WS-5: ActiveTransportFluxSpec honors edge direction (uptake vs efflux)."""
from __future__ import annotations

import numpy as np

import sisyphus.engine.flux  # noqa: F401 — registers FluxSpec implementations
from sisyphus.core import Distribution, DrugOnGraph, TransporterKinetics
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ActiveTransportEdge, FlowEdge, Node


def _drug() -> DrugOnGraph:
    return DrugOnGraph(
        name="d", smiles="CCO", dose_mg=100.0, route="oral",
        administration_node="gut_wall", mw=424.0, pka=4.5, compound_type="acid",
        fup=Distribution(0.5), rbp=Distribution(1.0), kp_method="rodgers_rowland",
        kp_overrides={}, peff=Distribution(1.0), solubility=Distribution(1.0),
        enzyme_affinity={}, renal_clearance=Distribution(0.0),
        transporter_kinetics={"PGP": TransporterKinetics(jmax=Distribution(228.0), km=Distribution(50.0))},
    )


def _efflux_graph() -> BodyGraph:
    """gut_wall (PGP, source) --efflux--> lumen (no transporters)."""
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(1.0),
                    transporters={"PGP": Distribution(1.0e10)}, ivive_scaling=1.0e-4))
    g.add_node(Node(name="lumen", node_type="lumen", volume=Distribution(1.0)))
    g.add_edge(ActiveTransportEdge(source="gut_wall", target="lumen", direction="efflux"))
    return g


def test_efflux_reads_source_transporters():
    """Efflux: transporter sits at the SOURCE (gut_wall); flux moves mass out of it."""
    g = _efflux_graph()
    compiled = ODECompiler().compile(g)
    rhs = compiled.make_rhs(ResolvedParams(g, _drug()))
    y = np.zeros(compiled.n_states)
    y[compiled.state_index["gut_wall"]] = 10.0
    dydt = rhs(0.0, y)
    assert dydt[compiled.state_index["gut_wall"]] < 0, "efflux should remove mass from source"
    assert dydt[compiled.state_index["lumen"]] > 0, "efflux should add mass to target"


def test_uptake_default_unchanged():
    """direction defaults to 'uptake' → transporter read at TARGET (legacy behavior)."""
    g = BodyGraph()
    g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_node(Node(name="liver", node_type="organ", volume=Distribution(1.5),
                    transporters={"PGP": Distribution(1.0e10)}, ivive_scaling=1.0e-4))
    g.add_edge(FlowEdge(source="blood", target="liver", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="liver", target="blood", flow_rate=Distribution(10.0)))
    g.add_edge(ActiveTransportEdge(source="blood", target="liver"))  # no direction → uptake
    compiled = ODECompiler().compile(g)
    rhs = compiled.make_rhs(ResolvedParams(g, _drug()))
    y = np.zeros(compiled.n_states)
    y[compiled.state_index["blood"]] = 10.0
    dydt = rhs(0.0, y)
    # Uptake reads target (liver) transporters; mass moves blood → liver via transport.
    assert dydt[compiled.state_index["liver"]] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_active_transport_direction.py -q`
Expected: FAIL — `ActiveTransportEdge.__init__() got an unexpected keyword argument 'direction'`.

- [ ] **Step 3: Add the `direction` field**

In `src/sisyphus/graph/types.py`, change `ActiveTransportEdge` (line 160) to:

```python
    edge_type: str = field(default="active_transport", init=False)
    # WS-5: "uptake" → transporter at TARGET (e.g. hepatic OATP, blood→liver);
    # "efflux" → transporter at SOURCE (e.g. gut P-gp, gut_wall→lumen). Driving
    # (substrate) concentration is the SOURCE in both cases.
    direction: str = "uptake"
```

- [ ] **Step 4: Honor direction in the SciPy flux**

In `src/sisyphus/engine/flux.py`, give `ActiveTransportFluxSpec` a constructor that stores direction and a `from_edge` that reads it, then select the transporter node in `apply`. Replace the class's `from_edge` and the transporter/IVIVE reads:

```python
    def __init__(
        self,
        edge_id: int,
        source_idx: int,
        target_idx: int,
        source_name: str,
        target_name: str,
        direction: str = "uptake",
    ) -> None:
        super().__init__(edge_id, source_idx, target_idx, source_name, target_name)
        self.direction = direction

    @classmethod
    def from_edge(cls, edge_id: int, edge, state_index: dict[str, int]) -> "ActiveTransportFluxSpec":
        return cls(
            edge_id,
            state_index[edge.source],
            state_index[edge.target],
            edge.source,
            edge.target,
            getattr(edge, "direction", "uptake"),
        )
```

In `apply`, make exactly two targeted line changes (the source-concentration computation
`c_mg_l = y[self.source_idx] / v_source` is unchanged). First, immediately after the early
`if c_um <= 0: return` line, add the direction-selected node:

```python
        # WS-5: transporter sits at the target for uptake, the source for efflux.
        # The driving (substrate) concentration is always the source (computed above).
        transporter_node = self.target_name if self.direction == "uptake" else self.source_name
```

Then change the existing line `node_transporters = params.node_transporters(self.target_name)` to:

```python
        node_transporters = params.node_transporters(transporter_node)
```

and change the existing line `ivive = params.node_param(self.target_name, "ivive_scaling")` to:

```python
        ivive = params.node_param(transporter_node, "ivive_scaling")
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/unit/test_active_transport_direction.py tests/unit/test_active_transport.py -q`
Expected: PASS (existing uptake tests still pass — default direction preserves behavior).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/graph/types.py src/sisyphus/engine/flux.py tests/unit/test_active_transport_direction.py
git commit -m "feat(engine): active-transport direction (uptake/efflux), SciPy path"
```

---

## Task 5: WS-5 — JAX active-transport direction mirror

**Files:**
- Modify: `src/sisyphus/engine/rhs_jax.py` (split transport edges into uptake/efflux index sets)
- Test: append a JAX-parity case to `tests/unit/test_active_transport_direction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_active_transport_direction.py`:

```python
def test_efflux_scipy_jax_parity():
    import pytest
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from sisyphus.engine.params_jax import resolve_to_jax
    from sisyphus.engine.rhs_jax import make_jax_rhs

    g = _efflux_graph()
    compiled = ODECompiler().compile(g)
    params = ResolvedParams(g, _drug())
    rhs_scipy = compiled.make_rhs(params)
    y = np.zeros(compiled.n_states)
    y[compiled.state_index["gut_wall"]] = 10.0
    dydt_scipy = rhs_scipy(0.0, y)

    jax_params = resolve_to_jax(compiled, params)
    dydt_jax = np.asarray(make_jax_rhs(compiled)(0.0, jnp.asarray(y), jax_params))
    np.testing.assert_allclose(dydt_jax, dydt_scipy, rtol=1e-6, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_active_transport_direction.py::test_efflux_scipy_jax_parity -q`
Expected: FAIL — JAX reads the target node's transporters (lumen has none) → transport flux 0, mismatch with SciPy.

- [ ] **Step 3: Split transport edges by direction in the JAX build**

In `src/sisyphus/engine/rhs_jax.py`, replace the single transport index collection with
uptake/efflux triples (src, transporter-node, target). Where `transport_src/transport_tgt` are
declared (lines 148-149), use six lists:

```python
    # Active transport, split by direction. The transporter-bearing node differs:
    #   uptake → target node; efflux → source node. Substrate conc is always the source.
    transport_up_src, transport_up_node, transport_up_tgt = [], [], []
    transport_ef_src, transport_ef_node, transport_ef_tgt = [], [], []
```

In the build loop, replace the `ActiveTransportFluxSpec` branch (lines 197-199):

```python
        elif isinstance(spec, ActiveTransportFluxSpec):
            if getattr(spec, "direction", "uptake") == "uptake":
                transport_up_src.append(spec.source_idx)
                transport_up_node.append(spec.target_idx)   # transporter at target
                transport_up_tgt.append(spec.target_idx)
            else:
                transport_ef_src.append(spec.source_idx)
                transport_ef_node.append(spec.source_idx)   # transporter at source
                transport_ef_tgt.append(spec.target_idx)
```

Convert to JAX arrays and presence flags (alongside the other conversions); remove the old
`_transport_src`/`_transport_tgt` arrays and the `has_transport` flag:

```python
    _t_up_src = jnp.array(transport_up_src, dtype=jnp.int32)
    _t_up_node = jnp.array(transport_up_node, dtype=jnp.int32)
    _t_up_tgt = jnp.array(transport_up_tgt, dtype=jnp.int32)
    _t_ef_src = jnp.array(transport_ef_src, dtype=jnp.int32)
    _t_ef_node = jnp.array(transport_ef_node, dtype=jnp.int32)
    _t_ef_tgt = jnp.array(transport_ef_tgt, dtype=jnp.int32)
    has_t_up = len(transport_up_src) > 0
    has_t_ef = len(transport_ef_src) > 0
```

Replace the single transport block (lines 419-440) with a closure plus a scatter per direction:

```python
        # 6. Active transport (Michaelis-Menten), split by direction.
        #    Substrate conc from the SOURCE; Vmax/Km/ivive from the transporter node
        #    (target for uptake, source for efflux). mass moves source → edge target.
        def _transport_mass(src_idx, node_idx):
            mw = params.drug_mw
            v_src_t = params.node_volumes[src_idx]
            vmax = params.node_transport_vmax[node_idx]
            km = params.node_transport_km[node_idx]
            ivive_t = params.node_ivive_scaling[node_idx]
            c_mg_l = jnp.where(v_src_t > 0.0, y[src_idx] / v_src_t, 0.0)
            c_um = jnp.where(mw > 0.0, c_mg_l * 1000.0 / mw, 0.0)
            mm_rate = jnp.where(
                (vmax > 0.0) & (km > 0.0) & (c_um > 0.0),
                vmax * c_um / (km + c_um),
                0.0,
            )
            return mm_rate * ivive_t

        if has_t_up:
            mass = _transport_mass(_t_up_src, _t_up_node)
            dydt = dydt.at[_t_up_src].add(-mass)
            dydt = dydt.at[_t_up_tgt].add(mass)
        if has_t_ef:
            mass = _transport_mass(_t_ef_src, _t_ef_node)
            dydt = dydt.at[_t_ef_src].add(-mass)
            dydt = dydt.at[_t_ef_tgt].add(mass)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/test_active_transport_direction.py tests/unit/test_active_transport.py -q`
Expected: PASS — uptake parity (existing `test_active_transport_scipy_jax_parity`) and the new efflux parity both green.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/engine/rhs_jax.py tests/unit/test_active_transport_direction.py
git commit -m "feat(engine): mirror active-transport direction in the JAX RHS"
```

---

## Task 6: WS-4 — `fu_correction` in the JAX well_stirred branch

**Files:**
- Modify: `src/sisyphus/engine/params_jax.py` (add 2 fields + populate + pytree)
- Modify: `src/sisyphus/engine/rhs_jax.py` (apply in well_stirred)
- Test: `tests/unit/test_jax_scipy_parity.py` (NEW)

- [ ] **Step 1: Write the failing test**

```python
"""WS-4: JAX↔SciPy RHS-level parity per flux branch."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import sisyphus.engine.flux  # noqa: F401,E402
from sisyphus.core import Distribution, DrugOnGraph  # noqa: E402
from sisyphus.engine.compiler import ODECompiler, ResolvedParams  # noqa: E402
from sisyphus.engine.params_jax import resolve_to_jax  # noqa: E402
from sisyphus.engine.rhs_jax import make_jax_rhs  # noqa: E402
from sisyphus.graph.body import BodyGraph  # noqa: E402
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node  # noqa: E402


def _drug(fu_corr: float = 1.0) -> DrugOnGraph:
    return DrugOnGraph(
        name="d", smiles="CCO", dose_mg=100.0, route="oral",
        administration_node="blood", mw=300.0, pka=4.5, compound_type="acid",
        fup=Distribution(0.3), rbp=Distribution(1.0), kp_method="provided",
        kp_overrides={"organ": Distribution(1.0)}, peff=Distribution(1.0),
        solubility=Distribution(1.0), enzyme_affinity={"CYP3A4": Distribution(5.0)},
        renal_clearance=Distribution(0.0), fu_correction_liver=Distribution(fu_corr),
    )


def _ws_flagged_graph(flagged: bool = True) -> BodyGraph:
    g = BodyGraph()
    g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(5.0)))
    g.add_node(Node(name="organ", node_type="organ", volume=Distribution(1.0),
                    enzymes={"CYP3A4": Distribution(1.0e6)}, ivive_scaling=1.0e-6,
                    fu_correction_applicable=1.0 if flagged else 0.0, lookup_name="organ"))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source="blood", target="organ", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="organ", target="blood", flow_rate=Distribution(10.0)))
    g.add_edge(ClearanceEdge(source="organ", target="sink", model="well_stirred"))
    return g


def _rhs_pair(graph, drug):
    compiled = ODECompiler().compile(graph)
    params = ResolvedParams(graph, drug)
    y = np.zeros(compiled.n_states)
    y[compiled.state_index["organ"]] = 3.0
    y[compiled.state_index["blood"]] = 7.0
    dydt_scipy = compiled.make_rhs(params)(0.0, y)
    dydt_jax = np.asarray(make_jax_rhs(compiled)(0.0, jnp.asarray(y), resolve_to_jax(compiled, params)))
    return dydt_scipy, dydt_jax


def test_ws_fu_correction_parity():
    """Flagged well_stirred node with non-identity fu_correction: JAX == SciPy."""
    s, j = _rhs_pair(_ws_flagged_graph(), _drug(fu_corr=1.5))
    np.testing.assert_allclose(j, s, rtol=1e-9, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_jax_scipy_parity.py::test_ws_fu_correction_parity -q`
Expected: FAIL — JAX ignores `fu_correction_liver`; SciPy multiplies `fup×1.5` → mismatch on the clearance/blood rows.

- [ ] **Step 3: Add fields to `JaxParams`**

In `src/sisyphus/engine/params_jax.py`, add to the dataclass (after `node_is_blood`):

```python
    node_fu_correction_applicable: jnp.ndarray  # (n_nodes,) 1.0 if flagged
```

and (in the scalar drug block):

```python
    drug_fu_correction_liver: float
```

Add both to `_jaxparams_flatten`'s `children` tuple (preserve field order), and to the `JaxParams(...)` constructor call at the end of `resolve_to_jax`:

```python
        node_fu_correction_applicable=jnp.array(fu_correction_applicable, dtype=jnp.float64),
        drug_fu_correction_liver=params.drug_param("fu_correction_liver"),
```

In the per-node loop of `resolve_to_jax`, alongside `is_blood.append(...)`:

```python
        fu_correction_applicable.append(params.node_param(name, "fu_correction_applicable"))
```

and initialize `fu_correction_applicable = []` with the other per-node lists.

- [ ] **Step 4: Apply it in the JAX well_stirred branch**

In `src/sisyphus/engine/rhs_jax.py`, in the `has_cl_ws` block, replace `cl_intrinsic_ws = fup * clint` with:

```python
            # WS-4: hepatic intracellular fu correction at flagged nodes (parity
            # with the SciPy well_stirred branch).
            fu_corr = params.drug_fu_correction_liver
            applicable = params.node_fu_correction_applicable[_cl_ws_src]
            fup_eff = jnp.where(applicable > 0.5, fup * fu_corr, fup)
            cl_intrinsic_ws = fup_eff * clint
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/unit/test_jax_scipy_parity.py -q`
Expected: PASS.

Run: `pytest tests/unit/test_active_transport.py -q`
Expected: PASS (JaxParams pytree still round-trips).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/engine/params_jax.py src/sisyphus/engine/rhs_jax.py tests/unit/test_jax_scipy_parity.py
git commit -m "feat(engine): apply fu_correction in the JAX well_stirred branch (parity)"
```

---

## Task 7: WS-4 — JAX Michaelis-Menten divergence guard

**Files:**
- Modify: `src/sisyphus/engine/params_jax.py` (`resolve_to_jax` per-node transporter loop)
- Test: append to `tests/unit/test_jax_scipy_parity.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_jax_scipy_parity.py`:

```python
def test_jax_fails_loud_on_distinct_km_multitransporter():
    """≥2 active transporters with distinct Km at one node: the JAX aggregate
    Vmax/weighted-Km approximation diverges from SciPy → fail loud, not silent."""
    from sisyphus.core import TransporterKinetics
    from sisyphus.graph.types import ActiveTransportEdge

    g = BodyGraph()
    g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_node(Node(name="organ", node_type="organ", volume=Distribution(1.0),
                    transporters={"A": Distribution(1e10), "B": Distribution(1e10)},
                    ivive_scaling=1e-4))
    g.add_edge(FlowEdge(source="blood", target="organ", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="organ", target="blood", flow_rate=Distribution(10.0)))
    g.add_edge(ActiveTransportEdge(source="blood", target="organ"))
    drug = DrugOnGraph(
        name="d", smiles="CCO", dose_mg=100.0, route="oral", administration_node="blood",
        mw=300.0, pka=4.5, compound_type="acid", fup=Distribution(0.3), rbp=Distribution(1.0),
        kp_method="provided", kp_overrides={"organ": Distribution(1.0)}, peff=Distribution(1.0),
        solubility=Distribution(1.0), enzyme_affinity={}, renal_clearance=Distribution(0.0),
        transporter_kinetics={
            "A": TransporterKinetics(jmax=Distribution(100.0), km=Distribution(10.0)),
            "B": TransporterKinetics(jmax=Distribution(100.0), km=Distribution(80.0)),
        },
    )
    compiled = ODECompiler().compile(g)
    with pytest.raises(NotImplementedError, match="distinct Km"):
        resolve_to_jax(compiled, ResolvedParams(g, drug))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_jax_scipy_parity.py::test_jax_fails_loud_on_distinct_km_multitransporter -q`
Expected: FAIL — no exception raised (silent approximation).

- [ ] **Step 3: Add the guard in `resolve_to_jax`**

In `src/sisyphus/engine/params_jax.py`, inside the per-node transporter loop (after the existing `for tag, abundance in transporters.items():` accumulation), collect the distinct active Km values and raise:

```python
        active_kms = set()
        for tag, abundance in transporters.items():
            j = params.drug_transporter_jmax(tag)
            km = params.drug_transporter_km(tag)
            if j > 0 and km > 0 and abundance > 0:
                active_kms.add(round(km, 9))
        if len(active_kms) > 1:
            raise NotImplementedError(
                f"node {name!r} has {len(active_kms)} active transporters with "
                f"distinct Km {sorted(active_kms)}; the JAX aggregate-Vmax / "
                f"weighted-Km approximation diverges from SciPy's exact "
                f"per-transporter Michaelis-Menten sum. Use backend='scipy' "
                f"(default), or implement exact padded per-transporter MM in JAX."
            )
```

(Place this block right before or after the existing `vmax_sum/km_weighted_num` accumulation — it reuses the same `params.drug_transporter_*` lookups.)

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/test_jax_scipy_parity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/engine/params_jax.py tests/unit/test_jax_scipy_parity.py
git commit -m "feat(engine): fail loud on JAX multi-transporter distinct-Km divergence"
```

---

## Task 8: WS-4 — Comprehensive per-branch parity suite

**Files:**
- Modify: `tests/unit/test_jax_scipy_parity.py` (add the remaining branch cases)

- [ ] **Step 1: Write the parity cases**

Append one RHS-level parity test per remaining branch. Each builds a minimal graph exercising the branch, then asserts `make_jax_rhs` ≡ `compiler.make_rhs` at a fixed `(t, y, params)` to `rtol=1e-9, atol=1e-12`. Cover: flow-only, gfr_filtration, transit, absorption, diffusion, well_stirred without fu_correction (unflagged node).

```python
def _assert_parity(graph, drug, fill):
    compiled = ODECompiler().compile(graph)
    params = ResolvedParams(graph, drug)
    y = np.zeros(compiled.n_states)
    for name, amt in fill.items():
        y[compiled.state_index[name]] = amt
    s = compiled.make_rhs(params)(0.0, y)
    j = np.asarray(make_jax_rhs(compiled)(0.0, jnp.asarray(y), resolve_to_jax(compiled, params)))
    np.testing.assert_allclose(j, s, rtol=1e-9, atol=1e-12)


def test_ws_no_fu_correction_parity():
    g = _ws_flagged_graph(flagged=False)  # well_stirred WITHOUT the fu_correction flag
    _assert_parity(g, _drug(fu_corr=1.0), {"organ": 3.0, "blood": 7.0})


def test_gfr_parity():
    import dataclasses
    g = BodyGraph()
    g.add_node(Node(name="blood", node_type="blood_pool", volume=Distribution(5.0)))
    g.add_node(Node(name="kidney", node_type="organ", volume=Distribution(1.0),
                    lookup_name="kidney"))
    g.add_node(Node(name="urine", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source="blood", target="kidney", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="kidney", target="blood", flow_rate=Distribution(10.0)))
    g.add_edge(ClearanceEdge(source="kidney", target="urine", model="gfr_filtration"))
    drug = dataclasses.replace(_drug(), renal_clearance=Distribution(5.0))
    _assert_parity(g, drug, {"kidney": 2.0, "blood": 8.0})
```

> For transit/absorption/diffusion follow the same shape: add the relevant edge type
> (`TransitEdge(transit_rate=...)`, `AbsorptionEdge(ka_fraction=...)`, `DiffusionEdge(ps_product=...)`)
> on a 2-node graph and call `_assert_parity`. Both `DrugOnGraph` and the edge dataclasses are
> frozen — use `dataclasses.replace(...)` for variants and construct graphs with the node fields
> you want directly (never mutate a frozen instance). Each test is ~10 lines.

- [ ] **Step 2: Run the suite**

Run: `pytest tests/unit/test_jax_scipy_parity.py -q`
Expected: PASS (all branch parity cases green).

- [ ] **Step 3: Full engine + regression sweep**

Run: `pytest tests/unit tests/integration tests/regression -q`
Expected: PASS, including `test_cached_holdout_aafe_is_2p784` (headline bit-identical).

- [ ] **Step 4: Lint**

Run: `ruff check src/sisyphus/engine src/sisyphus/pipeline tests/unit/test_engine_contracts.py tests/unit/test_active_transport_direction.py tests/unit/test_jax_scipy_parity.py`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_jax_scipy_parity.py
git commit -m "test(engine): comprehensive per-branch SciPy/JAX RHS parity suite"
```

---

## Done criteria

- WS-2: a non-1.0 `fu_correction_liver` that would be entirely dropped raises `ValueError` from both pipeline and `uncertainty` paths; prodrug coexistence does not.
- WS-5: `ActiveTransportEdge.direction` honored in SciPy and JAX; default `uptake` preserves behavior.
- WS-4: JAX well_stirred applies `fu_correction`; distinct-Km multi-transporter nodes fail loud; per-branch RHS parity to 1e-9.
- WS-6: README engine-validation table labels each row Omega-parity vs Sisyphus-snapshot and notes RBP-2.
- `test_cached_holdout_aafe_is_2p784` green throughout (headline 2.784 bit-identical).
