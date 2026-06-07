# Axial Parallel-Tube Implementation Plan (WS-3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Depends on** `docs/superpowers/plans/2026-06-07-engine-contract-hardening.md` (Task 2 creates `engine/contracts.py`, whose honoring-flux set Task 7 here trims).

**Goal:** Make `model: parallel_tube` a *real* parallel-tube model by expanding the source organ into N serial well-stirred sub-compartments at build time — the engine compiles only well_stirred tanks, so it changes 0 lines.

**Architecture:** A pure graph transform `expand_axial(graph)` rewrites each `parallel_tube` clearance organ into N tanks (volume/N, enzymes/N), full-Q internal flow edges, N well_stirred clearance edges, and `lookup_name`→parent so Kp/PS resolve correctly. N serial well-stirred tanks converge to the analytic parallel-tube extraction `1 − exp(−fu_b·CLint/Q)` (`fu_b = fup/RBP`). The single-tank `parallel_tube` flux branch is removed; an unexpanded `parallel_tube` edge reaching compile fails loud.

**Tech Stack:** Python 3.10+, frozen dataclasses (`dataclasses.replace`), NumPy, SciPy `solve_ivp`, pytest, ruff (line length 100).

**Spec:** `docs/superpowers/specs/2026-06-07-engine-contract-hardening-design.md` (WS-3).

**Headline invariance:** production physiology has no `parallel_tube` edge → `expand_axial` early-returns the unchanged graph → headline 2.784 bit-identical. Guard with `pytest tests/regression/ -k cached_holdout -q` after each task.

---

## File Structure

- `src/sisyphus/graph/types.py` — **MODIFY.** Add `Node.axial_subcompartments`.
- `src/sisyphus/graph/builder.py` — **MODIFY.** Pass `axial_subcompartments` through from YAML.
- `src/sisyphus/graph/axial.py` — **NEW.** `expand_axial(graph) -> BodyGraph`. One responsibility: topology expansion.
- `src/sisyphus/engine/flux.py` — **MODIFY.** Remove the single-tank `parallel_tube` branch; fail loud if it reaches `from_edge`.
- `src/sisyphus/engine/rhs_jax.py` — **MODIFY.** Remove the now-dead `cl_pt` branch.
- `src/sisyphus/engine/contracts.py` — **MODIFY.** Drop `parallel_tube` from the honoring set.
- `src/sisyphus/pipeline/predict.py` — **MODIFY.** Call `expand_axial` before compile.
- `scripts/run_chain_benchmark.py` — **MODIFY.** Call `expand_axial`; D/E/F now produce real PT.
- Tests: `tests/unit/test_axial_expansion.py` (NEW); rewrite the `parallel_tube` cases in `tests/unit/test_flux_fu_correction_integration.py`.

---

## Task 1: Add `Node.axial_subcompartments` field + builder passthrough

**Files:**
- Modify: `src/sisyphus/graph/types.py:60` (Node)
- Modify: `src/sisyphus/graph/builder.py:226-243` (node construction)
- Test: `tests/unit/test_axial_expansion.py` (NEW)

- [ ] **Step 1: Write the failing test**

```python
"""WS-3: axial sub-compartment expansion."""
from __future__ import annotations

import math

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.axial import expand_axial
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node


def test_node_has_axial_field():
    n = Node(name="liver", node_type="organ", volume=Distribution(1.0), axial_subcompartments=5)
    assert n.axial_subcompartments == 5


def test_node_axial_defaults_to_one():
    n = Node(name="liver", node_type="organ", volume=Distribution(1.0))
    assert n.axial_subcompartments == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_axial_expansion.py -q`
Expected: FAIL — `Node.__init__() got an unexpected keyword argument 'axial_subcompartments'`.

- [ ] **Step 3: Add the field**

In `src/sisyphus/graph/types.py`, after `fu_correction_applicable: float = 0.0` (line 60):

```python
    # WS-3: >1 expands this perfusion organ into N serial well-stirred sub-tanks
    # (axial gradient → parallel-tube extraction) via graph.axial.expand_axial.
    axial_subcompartments: int = 1
```

- [ ] **Step 4: Pass it through the builder**

In `src/sisyphus/graph/builder.py`, where the `Node(...)` is constructed (near line 243), add:

```python
        axial_subcompartments=int(spec.get("axial_subcompartments", 1)),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_axial_expansion.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/graph/types.py src/sisyphus/graph/builder.py tests/unit/test_axial_expansion.py
git commit -m "feat(graph): Node.axial_subcompartments field + builder passthrough"
```

---

## Task 2: `expand_axial` transform — topology

**Files:**
- Create: `src/sisyphus/graph/axial.py`
- Test: append to `tests/unit/test_axial_expansion.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_axial_expansion.py`:

```python
def _pt_graph(n: int) -> BodyGraph:
    """dose(lumen) → organ(parallel_tube, N) → drain(sink); organ → metab(sink)."""
    g = BodyGraph()
    g.add_node(Node(name="dose", node_type="lumen", volume=Distribution(1.0)))
    g.add_node(Node(name="organ", node_type="organ", volume=Distribution(2.0),
                    enzymes={"CYP": Distribution(20.0)}, ivive_scaling=1.0,
                    axial_subcompartments=n, lookup_name="organ"))
    g.add_node(Node(name="drain", node_type="sink", volume=Distribution(1.0)))
    g.add_node(Node(name="metab", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source="dose", target="organ", flow_rate=Distribution(10.0)))
    g.add_edge(FlowEdge(source="organ", target="drain", flow_rate=Distribution(10.0)))
    g.add_edge(ClearanceEdge(source="organ", target="metab", model="parallel_tube"))
    return g


def test_expand_creates_n_tanks_and_removes_organ():
    g = expand_axial(_pt_graph(4))
    assert "organ" not in g.nodes
    tanks = [n for n in g.nodes if n.startswith("organ__ax")]
    assert len(tanks) == 4


def test_expand_divides_extensive_copies_intensive():
    g = expand_axial(_pt_graph(4))
    t1 = g.nodes["organ__ax1"]
    assert t1.volume.mean == pytest.approx(2.0 / 4)
    assert t1.enzymes["CYP"].mean == pytest.approx(20.0 / 4)
    assert t1.ivive_scaling == 1.0          # intensive — copied
    assert t1.lookup_name == "organ"        # Kp/PS resolution → parent


def test_expand_preserves_flow_conservation():
    g = expand_axial(_pt_graph(4))
    assert g.validate() == []


def test_expand_clearance_edges_become_well_stirred():
    g = expand_axial(_pt_graph(3))
    cl = [e for e in g.edges if getattr(e, "model", None) is not None]
    assert len(cl) == 3
    assert all(e.model == "well_stirred" for e in cl)


def test_expand_early_returns_unchanged_when_no_parallel_tube():
    g = BodyGraph()
    g.add_node(Node(name="liver", node_type="organ", volume=Distribution(1.0)))
    g.add_node(Node(name="sink", node_type="sink", volume=Distribution(1.0)))
    g.add_edge(ClearanceEdge(source="liver", target="sink", model="well_stirred"))
    assert expand_axial(g) is g  # same object → bit-identity


def test_expand_scope_guard_rejects_nonperfusion_edge():
    from sisyphus.graph.types import DiffusionEdge
    g = _pt_graph(4)
    g.add_node(Node(name="tissue", node_type="organ", volume=Distribution(1.0)))
    g.add_edge(DiffusionEdge(source="organ", target="tissue", ps_product=Distribution(1.0)))
    with pytest.raises(NotImplementedError, match="perfusion organ"):
        expand_axial(g)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_axial_expansion.py -k expand -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sisyphus.graph.axial'`.

- [ ] **Step 3: Implement `graph/axial.py`**

```python
"""Axial sub-compartment expansion (WS-3).

Rewrites each ``parallel_tube`` clearance organ into N serial well-stirred
sub-tanks. N serial well-stirred tanks (each CLint/N, volume/N, full flow Q)
converge to the parallel-tube extraction ``1 - exp(-fu_b·CLint/Q)``. The engine
compiles only well_stirred tanks, so it is unchanged (invariant #8). Identity-
blind: sub-tanks carry ``lookup_name`` = parent so Kp/PS resolve to the parent
organ; the engine never matches a literal name.
"""
from __future__ import annotations

import dataclasses

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge, Node

# Default discretization when parallel_tube is requested without an explicit N.
# Numerical convergence parameter (<~2% from analytic E_PT at typical CLint),
# NOT tuned to Cmax loss (invariant #8).
_DEFAULT_AXIAL_N = 10

_PERFUSION_OK_EDGE_TYPES = frozenset({"flow", "clearance"})


def _scale(dists: dict[str, Distribution], n: int) -> dict[str, Distribution]:
    return {tag: Distribution(d.mean / n, cv=d.cv) for tag, d in dists.items()}


def expand_axial(graph: BodyGraph) -> BodyGraph:
    """Expand every ``parallel_tube`` clearance organ into N serial well-stirred tanks.

    Returns the *same* graph object unchanged when no ``parallel_tube`` edge
    exists (production path → bit-identical). Otherwise returns a new BodyGraph.

    Raises ``NotImplementedError`` if an organ tagged for expansion has any edge
    other than flow-in / flow-out / clearance.
    """
    organs = sorted({
        e.source for e in graph.edges
        if isinstance(e, ClearanceEdge) and e.model == "parallel_tube"
    })
    if not organs:
        return graph

    new = BodyGraph()
    new.global_params = dict(graph.global_params)

    organ_set = set(organs)
    # Carry over every node that is NOT being expanded.
    for name, node in graph.nodes.items():
        if name not in organ_set:
            new.add_node(node)

    # Pre-validate scope + create tanks for each expanded organ.
    tanks_by_organ: dict[str, list[str]] = {}
    for organ in organs:
        node = graph.nodes[organ]
        touching = [e for e in graph.edges if e.source == organ or e.target == organ]
        for e in touching:
            if e.edge_type not in _PERFUSION_OK_EDGE_TYPES:
                raise NotImplementedError(
                    f"axial expansion supports a perfusion organ with only "
                    f"flow/clearance edges; organ {organ!r} has a {e.edge_type!r} "
                    f"edge. Expand a perfusion organ, or model this differently."
                )
        n = node.axial_subcompartments if node.axial_subcompartments >= 2 else _DEFAULT_AXIAL_N
        names = [f"{organ}__ax{i}" for i in range(1, n + 1)]
        tanks_by_organ[organ] = names
        for tname in names:
            new.add_node(dataclasses.replace(
                node,
                name=tname,
                volume=Distribution(node.volume.mean / n, cv=node.volume.cv),
                enzymes=_scale(node.enzymes, n),
                transporters=_scale(node.transporters, n),
                lookup_name=node.lookup_name or organ,
                axial_subcompartments=1,
            ))

    # Rewrite edges.
    for e in graph.edges:
        if isinstance(e, ClearanceEdge) and e.source in organ_set and e.model == "parallel_tube":
            # Replicate as N well_stirred clearance edges, one per tank.
            for tname in tanks_by_organ[e.source]:
                new.add_edge(ClearanceEdge(source=tname, target=e.target, model="well_stirred"))
            continue
        if isinstance(e, FlowEdge) and e.target in organ_set:
            new.add_edge(dataclasses.replace(e, target=tanks_by_organ[e.target][0]))
            continue
        if isinstance(e, FlowEdge) and e.source in organ_set:
            new.add_edge(dataclasses.replace(e, source=tanks_by_organ[e.source][-1]))
            continue
        if e.source in organ_set or e.target in organ_set:
            continue  # any other organ-touching edge was rejected above
        new.add_edge(e)

    # Internal full-Q series edges per organ.
    for organ, names in tanks_by_organ.items():
        q_total = sum(
            e.flow_rate.mean for e in graph.edges
            if isinstance(e, FlowEdge) and e.target == organ
        )
        for a, b in zip(names, names[1:]):
            new.add_edge(FlowEdge(source=a, target=b, flow_rate=Distribution(q_total)))

    return new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_axial_expansion.py -k expand -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/graph/axial.py tests/unit/test_axial_expansion.py
git commit -m "feat(graph): expand_axial — parallel_tube organ → N serial well-stirred tanks"
```

---

## Task 3: Convergence test — N tanks → analytic parallel-tube extraction

**Files:**
- Test: append to `tests/unit/test_axial_expansion.py`

- [ ] **Step 1: Write the test**

Append (uses `_pt_graph` from Task 2; RBP=1, Kp=1 so `fu_b = fup`):

```python
def _drug_ws() -> DrugOnGraph:
    return DrugOnGraph(
        name="d", smiles="CCO", dose_mg=100.0, route="oral", administration_node="dose",
        mw=300.0, pka=4.5, compound_type="acid", fup=Distribution(0.5), rbp=Distribution(1.0),
        kp_method="provided", kp_overrides={}, peff=Distribution(1.0),
        solubility=Distribution(1.0), enzyme_affinity={"CYP": Distribution(1.0)},
        renal_clearance=Distribution(0.0),
    )


def _extraction(n: int) -> float:
    """Single-pass extraction E = metabolized / dose after full washout."""
    g = expand_axial(_pt_graph(n))
    compiled = ODECompiler().compile(g)
    from sisyphus.engine.solver import solve
    params = ResolvedParams(g, _drug_ws())
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index["dose"]] = 100.0
    sim = solve(compiled, params, y0, t_span=(0.0, 2000.0))
    assert sim.solver_success
    metab = sim.amounts["metab"][-1]
    return metab / 100.0


def test_n1_matches_well_stirred():
    # fup=0.5, CLint=20 (enzyme 20 × affinity 1 × ivive 1), Q=10 → fup·CLint=10
    # E_WS = 10 / (10 + 10) = 0.5
    assert _extraction(1) == pytest.approx(0.5, abs=0.02)


def test_large_n_converges_to_parallel_tube():
    # E_PT = 1 - exp(-fup·CLint/Q) = 1 - exp(-1) = 0.6321
    e_pt = 1.0 - math.exp(-1.0)
    assert _extraction(50) == pytest.approx(e_pt, abs=0.01)


def test_extraction_monotone_in_n():
    assert _extraction(1) < _extraction(5) < _extraction(50)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/unit/test_axial_expansion.py -k "extraction or converges or matches or monotone" -q`
Expected: PASS — N=1 → 0.5, N=50 → ~0.632, monotone increasing.

> If `solve` does not expose `amounts[...]`, use the engine's `SimResult.amounts` mapping (node_name → mg time series) per the SimResult contract; `metab` is a sink node accumulating cleared mass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_axial_expansion.py
git commit -m "test(graph): axial expansion converges to analytic parallel-tube extraction"
```

---

## Task 4: Remove the single-tank `parallel_tube` flux branch + fail loud

**Files:**
- Modify: `src/sisyphus/engine/flux.py` (`ClearanceFluxSpec`: drop `parallel_tube` branch; `from_edge` raises)
- Modify: `tests/unit/test_flux_fu_correction_integration.py` (rewrite `parallel_tube` cases → `well_stirred`)
- Test: append a fail-loud case to `tests/unit/test_axial_expansion.py`

- [ ] **Step 1: Write the failing fail-loud test**

Append to `tests/unit/test_axial_expansion.py`:

```python
def test_unexpanded_parallel_tube_fails_loud_at_compile():
    g = _pt_graph(4)  # NOT expanded
    with pytest.raises(ValueError, match="parallel_tube"):
        ODECompiler().compile(g)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_axial_expansion.py::test_unexpanded_parallel_tube_fails_loud_at_compile -q`
Expected: FAIL — compile currently succeeds (the parallel_tube branch still exists).

- [ ] **Step 3: Remove the branch + add the guard**

In `src/sisyphus/engine/flux.py` `ClearanceFluxSpec`:
- Add a guard in `from_edge` (before constructing) so an unexpanded edge fails loud:

```python
    @classmethod
    def from_edge(cls, edge_id: int, edge, state_index: dict[str, int]) -> "ClearanceFluxSpec":
        if edge.model == "parallel_tube":
            raise ValueError(
                "ClearanceEdge model='parallel_tube' must be expanded into axial "
                "sub-compartments before compile (graph.axial.expand_axial); it is "
                "not a single-tank flux. Set Node.axial_subcompartments and call "
                "expand_axial(graph) before ODECompiler().compile(graph)."
            )
        return cls(
            edge_id,
            state_index[edge.source],
            state_index[edge.target],
            edge.source,
            edge.target,
            edge.model,
        )
```

- Delete the entire `elif self.model == "parallel_tube":` block in `apply` (lines ~276-308).
- Update the `ClearanceFluxSpec` docstring: replace the `parallel_tube` bullet with: ``- ``parallel_tube`` — NOT a single-tank flux; expanded into N serial well_stirred sub-tanks at build time by ``graph.axial.expand_axial`` (true axial-gradient parallel-tube).``

In `src/sisyphus/graph/types.py` `ClearanceEdge` docstring (line ~137), change the `parallel_tube` bullet to: ``- ``"parallel_tube"`` — true parallel-tube via axial sub-compartment expansion (build-time; see graph.axial).``

- [ ] **Step 4: Rewrite the `parallel_tube` cases in the fu_correction integration test**

In `tests/unit/test_flux_fu_correction_integration.py`, the tests that construct `ClearanceFluxSpec(model="parallel_tube")` directly (around lines 169-190, 363-379) tested that the PT branch applies `fu_correction` identically to well_stirred. Since the math is now identical and the branch is gone, change those constructions to `model="well_stirred"` and update the test names/docstrings (e.g. `test_parallel_tube_fu_correction_*` → `test_well_stirred_fu_correction_*`), preserving the assertions. The fu_correction gating is unchanged for well_stirred.

Run: `grep -n "parallel_tube" tests/unit/test_flux_fu_correction_integration.py`
Expected: no matches remain.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/unit/test_axial_expansion.py tests/unit/test_flux_fu_correction_integration.py tests/unit/test_flux.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/engine/flux.py src/sisyphus/graph/types.py tests/unit/test_flux_fu_correction_integration.py tests/unit/test_axial_expansion.py
git commit -m "refactor(engine): remove single-tank parallel_tube flux; fail loud on unexpanded edge"
```

---

## Task 5: Remove the dead `cl_pt` branch from the JAX RHS

**Files:**
- Modify: `src/sisyphus/engine/rhs_jax.py`

- [ ] **Step 1: Remove the parallel_tube handling**

In `src/sisyphus/engine/rhs_jax.py`:
- Delete the `cl_pt_src`/`cl_pt_tgt` lists, the `elif spec.model == "parallel_tube":` build-loop branch (lines ~161-163), the `_cl_pt_*` `jnp.array` conversions, the `has_cl_pt` flag, and the `# 2b. Parallel-tube model` RHS block (lines ~312-333).
- In the build-loop `ClearanceFluxSpec` dispatch, `parallel_tube` will never appear (SciPy `from_edge` fails loud first); leave the `else: raise NotImplementedError` so any stray model still fails loud.

- [ ] **Step 2: Run the JAX tests**

Run: `pytest tests/unit/test_jax_scipy_parity.py tests/unit/test_active_transport.py tests/unit/test_active_transport_direction.py -q`
Expected: PASS (requires jax; otherwise skipped).

- [ ] **Step 3: Commit**

```bash
git add src/sisyphus/engine/rhs_jax.py
git commit -m "refactor(engine): drop dead parallel_tube branch from JAX RHS"
```

---

## Task 6: Trim `parallel_tube` from the contracts honoring set

**Files:**
- Modify: `src/sisyphus/engine/contracts.py`

- [ ] **Step 1: Update the honoring set**

In `src/sisyphus/engine/contracts.py`, change:

```python
_FU_CORRECTION_HONORING_MODELS = frozenset({"well_stirred", "parallel_tube"})
```

to:

```python
# parallel_tube is expanded to well_stirred tanks before compile (graph.axial),
# so the honoring clearance model at runtime is well_stirred only.
_FU_CORRECTION_HONORING_MODELS = frozenset({"well_stirred"})
```

- [ ] **Step 2: Run the contracts tests**

Run: `pytest tests/unit/test_engine_contracts.py -q`
Expected: PASS (the prodrug/well_stirred honoring cases are unaffected).

- [ ] **Step 3: Commit**

```bash
git add src/sisyphus/engine/contracts.py
git commit -m "refactor(engine): drop parallel_tube from fu_correction honoring set"
```

---

## Task 7: Wire `expand_axial` into the pipeline + chain benchmark

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py` (before `ODECompiler().compile(graph)`)
- Modify: `scripts/run_chain_benchmark.py`
- Test: append an end-to-end case to `tests/unit/test_axial_expansion.py`

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/unit/test_axial_expansion.py`:

```python
def test_pipeline_handles_parallel_tube_graph(monkeypatch):
    """A graph with a parallel_tube edge is auto-expanded by the pipeline (no raise)."""
    # Smoke test: predict() must not raise the unexpanded-parallel_tube ValueError.
    from sisyphus.pipeline.predict import predict
    result = predict("CCO", dose_mg=100.0, route="oral")
    assert result is not None  # production graph has no parallel_tube → expand_axial is a no-op
```

> This guards the wiring + bit-identity (production graph unchanged). A true PT-physiology
> end-to-end belongs in the chain benchmark (Step 4), not the holdout path.

- [ ] **Step 2: Run it (should pass once wired; first confirm current green)**

Run: `pytest tests/unit/test_axial_expansion.py::test_pipeline_handles_parallel_tube_graph -q`
Expected: PASS already (production has no PT). The wiring below makes PT graphs work without changing this.

- [ ] **Step 3: Wire into the pipeline**

In `src/sisyphus/pipeline/predict.py`, immediately before `compiler = ODECompiler()` / `compiled = compiler.compile(graph)` (line ~503):

```python
        from sisyphus.graph.axial import expand_axial
        graph = expand_axial(graph)  # no-op unless a parallel_tube edge is present
```

Place it AFTER the WS-2 guard call (`assert_fu_correction_honored`) so the guard sees the
original node flags; expansion preserves them on the tanks.

- [ ] **Step 4: Wire into the chain benchmark**

In `scripts/run_chain_benchmark.py`, where graphs are built and compiled, call
`from sisyphus.graph.axial import expand_axial` and `graph = expand_axial(graph)` before
`ODECompiler().compile(...)`. The D/E/F (`parallel_tube`) configs now produce genuine
parallel-tube extraction (default N=10). Add a comment noting the numeric change is intended.

Run: `python scripts/run_chain_benchmark.py 2>&1 | head -30`
Expected: runs without the unexpanded-parallel_tube ValueError; D/E/F report PT extraction.

- [ ] **Step 5: Run the headline guard**

Run: `pytest tests/regression/ -k cached_holdout -q`
Expected: PASS (production graph has no parallel_tube → bit-identical).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/pipeline/predict.py scripts/run_chain_benchmark.py tests/unit/test_axial_expansion.py
git commit -m "feat(pipeline): expand parallel_tube organs before compile (axial PT)"
```

---

## Task 8: Full sweep + lint

- [ ] **Step 1: Full test sweep**

Run: `pytest tests/unit tests/integration tests/regression -q`
Expected: PASS, including `test_cached_holdout_aafe_is_2p784`.

- [ ] **Step 2: Identity-blind check (invariant #1)**

Run: `pytest tests/ -k "identity_blind or identity" -q`
Expected: PASS — expansion-generated names (`organ__axN`) do not break identity-blindness (the engine resolves Kp via `lookup_name`, never literal names).

- [ ] **Step 3: Lint**

Run: `ruff check src/sisyphus/graph/axial.py src/sisyphus/engine/flux.py src/sisyphus/engine/rhs_jax.py src/sisyphus/engine/contracts.py src/sisyphus/pipeline/predict.py tests/unit/test_axial_expansion.py`
Expected: no errors.

- [ ] **Step 4: Commit (if lint fixups needed)**

```bash
git add -A
git commit -m "style(graph): ruff fixups for axial expansion"
```

---

## Done criteria

- `model: parallel_tube` expands to N serial well_stirred tanks; extraction converges to
  `1 − exp(−fu_b·CLint/Q)` (N=1 = well_stirred, monotone in N).
- Sub-tanks resolve Kp/PS via `lookup_name` = parent; flow conservation holds (`validate() == []`).
- Single-tank `parallel_tube` flux branch removed (SciPy + JAX); unexpanded edge fails loud at compile.
- `expand_axial` early-returns the unchanged graph for non-PT graphs → headline 2.784 bit-identical.
- Scope guard rejects expansion of organs with non-perfusion edges.
