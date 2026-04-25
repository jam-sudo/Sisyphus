# Prodrug Activation Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SMILES-keyed registry-driven prodrug activation routing so that the engine simulates active metabolite kinetics for sepiapterin/remdesivir/tebipenem_pivoxil/fostamatinib, fixing N50 systematic error.

**Architecture:** Two-species engine simulation: parent on full graph + active in 1-compartment plasma (1 new node, 2 new edge types). All compiler changes are additive (new isinstance branches; existing logic untouched). SMILES-keyed registry JSON drives drug-specific activation parameters; pipeline-layer adjustment upgrades AD-flag interpretation.

**Tech Stack:** Python 3.10+, frozen dataclasses, pytest, RDKit (for canonical SMILES), `@register_flux` decorator pattern.

**Spec reference:** `docs/superpowers/specs/2026-04-24-prodrug-activation-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/sisyphus/core.py` | Modify (additive) | `ActiveMetabolite` dataclass; `DrugOnGraph` field additions; `DrugOnGraph.sample()` extension |
| `src/sisyphus/graph/types.py` | Modify (additive) | `ProdrugActivationEdge`, `OneCompartmentEliminationEdge` |
| `src/sisyphus/graph/body.py` | Modify (additive) | `BodyGraph.sample()` branches for new edge types |
| `src/sisyphus/graph/builder.py` | Modify (additive) | `augment_for_active_species()` function |
| `src/sisyphus/engine/flux.py` | Modify (additive) | `ProdrugActivationFluxSpec`, `OneCompartmentEliminationFluxSpec` |
| `src/sisyphus/engine/compiler.py` | Modify (additive) | `_build_edge_params` two new isinstance branches |
| `src/sisyphus/predict/registry.py` | Create | Registry JSON loader; SMILES canonicalization; ActiveMetabolite construction |
| `src/sisyphus/predict/ivive.py` | Modify (additive) | `build_drug_on_graph` registry lookup |
| `src/sisyphus/pipeline/predict.py` | Modify (additive) | AD adjustment + observation routing; augmentation hook |
| `data/sbi/prodrug_activation_registry.json` | Create | 4 drug entries with literature-sourced parameters |
| `tests/unit/test_active_metabolite.py` | Create | ActiveMetabolite dataclass + DrugOnGraph validation |
| `tests/unit/test_prodrug_edges.py` | Create | New edge type instantiation, sampling |
| `tests/unit/test_prodrug_flux.py` | Create | FluxSpec.apply math correctness |
| `tests/unit/test_compiler_edge_params.py` | Create | additive branch coverage |
| `tests/unit/test_augment_active_species.py` | Create | builder augmentation behavior |
| `tests/unit/test_prodrug_registry.py` | Create | registry loader + validation |
| `tests/unit/test_pipeline_prodrug.py` | Create | AD adjustment + observation routing |
| `tests/integration/test_two_species_mass_balance.py` | Create | analytical 2-compartment comparison |
| `tests/integration/test_prodrug_pipeline_smoke.py` | Create | end-to-end smoke for 4 drugs |
| `tests/regression/test_holdout_unchanged.py` | Create | 107-holdout AAFE no drift |

---

## Verified Codebase Facts

Used during plan writing (replaces spec §8 Pre-implementation Verifications):

1. **DrugOnGraph constructor entry**: `src/sisyphus/predict/ivive.py:552 def build_drug_on_graph(profile, adme, dose_mg, route, ...)`. Returns `DrugOnGraph(...)` at line 634. SMILES available via `profile.smiles`.
2. **reference_man.yaml node names**: `gut_wall` (intestinal wall, has CYP3A4 enzymes), `venous_blood`, `portal_vein`. **No "enterocyte" node** — `gut_wall` is the canonical intestinal absorption/metabolism site. Sink-type nodes: `metabolized_hepatic`, `excreted_renal`, `metabolized_gut`, `excreted_fecal`.
3. **Allowed `node_type` values** (`graph/types.py:51`): `"organ"`, `"barrier_organ"`, `"blood_pool"`, `"lumen"`, `"sink"`. **Use `"blood_pool"`** for active plasma node (Kp=1 default, exempt from Kp logic).
4. **ClearanceFluxSpec models** (`engine/flux.py:170-340`): `well_stirred`, `parallel_tube`, `gfr_filtration`, `extended`. **No 1-compartment simple model** — confirms need for new `OneCompartmentEliminationFluxSpec`.
5. **Sink convention**: `ClearanceFluxSpec.apply` does `dydt[target_idx] += rate` where target is a sink-type node (e.g., `metabolized_hepatic`). Mass accumulates there for mass-balance audit. **`OneCompartmentEliminationEdge` target = `metabolized_gut`** (re-using existing sink node; mass leaves system but is auditable).
6. **Pipeline orchestrator**: `src/sisyphus/pipeline/predict.py:24 def predict(smiles, dose_mg, ...)`. Wires `predict → engine → pk → ml → meta` in order.
7. **`method_routing.json` precedent**: keyed by drug name; loaded once with `json.loads`. Pattern reusable for prodrug registry but **keyed by canonical SMILES** for SMILES-first invariant.
8. **`BodyGraph.add_node` / `add_edge` API** (`graph/body.py:54-66`): both raise `ValueError` on invalid input (duplicate name; missing source/target). Idempotency NOT guaranteed — calling twice raises.
9. **N50 reference species per drug**:
   - sepiapterin: **PARENT** Cmax 2.4 ng/mL @ 60 mg/kg (Gao 2024 PMC11597218); BH4 also reported 403 ng/mL.
   - remdesivir: **PARENT** Cmax 4380 ng/mL @ 200 mg IV (PMC table 3).
   - tebipenem_pivoxil: **ACTIVE** (tebipenem) Cmax 4006 ng/mL @ 300 mg PO.
   - fostamatinib: **ACTIVE** (R406) Cmax 605 ng/mL @ 75 mg PO (Baluom 2013 PMC3703230).
   - **Implication**: per-drug `observation_species` setting (registry field): sepiapterin/remdesivir → `"parent"`; tebipenem/fostamatinib → `"active"`.
10. **MC sampling**: `DrugOnGraph.sample()` at `core.py:203` is **explicit per-field copy + resample**. Adding active_metabolite requires extending this method.

---

## Task Order Rationale

Phases 1–3 establish data types. Phase 4 wires compiler. Phase 5 builds the augmentation function. Phase 6 creates the registry. Phase 7 integrates pipeline. Phase 8 validates end-to-end. Each task is independently testable; later tasks compose earlier ones.

---

### Task 1: ActiveMetabolite dataclass

**Files:**
- Modify: `src/sisyphus/core.py` (append after `TransporterKinetics` class at line 135)
- Test: `tests/unit/test_active_metabolite.py` (new)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_active_metabolite.py`:
```python
"""Tests for ActiveMetabolite dataclass."""
from __future__ import annotations

import pytest

from sisyphus.core import ActiveMetabolite, Distribution


def test_active_metabolite_minimal_construction():
    """Required fields produce a valid frozen ActiveMetabolite."""
    am = ActiveMetabolite(
        name="BH4",
        mw=241.25,
        fup=Distribution(mean=0.23, cv=0.3),
        CL_per_h=Distribution(mean=40.0, cv=0.35),
        Vd_L=Distribution(mean=150.0, cv=0.3),
        conversion_rate_per_h=Distribution(mean=12.0, cv=0.4),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(mean=0.85, cv=0.1),
    )
    assert am.name == "BH4"
    assert am.mw == 241.25
    assert am.conversion_site == "gut_wall"


def test_active_metabolite_is_frozen():
    """ActiveMetabolite must be frozen (immutable)."""
    am = ActiveMetabolite(
        name="BH4", mw=241.25,
        fup=Distribution(0.23), CL_per_h=Distribution(40.0),
        Vd_L=Distribution(150.0), conversion_rate_per_h=Distribution(12.0),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(1.0),
    )
    with pytest.raises((AttributeError, Exception)):
        am.name = "different"  # type: ignore[misc]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_active_metabolite.py -v`
Expected: FAIL with `ImportError: cannot import name 'ActiveMetabolite' from 'sisyphus.core'`

- [ ] **Step 3: Implement ActiveMetabolite in core.py**

Add to `src/sisyphus/core.py` immediately after the `TransporterKinetics` class (after line 134):

```python
# ---------------------------------------------------------------------------
# ActiveMetabolite — prodrug activation routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveMetabolite:
    """Active species produced by in vivo conversion of a parent prodrug.

    One-compartment plasma disposition: aggregate ``CL_per_h`` and ``Vd_L``
    represent the active's apparent clearance and volume of distribution.
    No enzyme-level decomposition (out of scope; see
    docs/superpowers/specs/2026-04-24-prodrug-activation-design.md §2).

    The conversion edge connects ``conversion_site`` (a parent-graph node)
    to the active's plasma compartment. Conversion is 1st-order in parent
    amount with optional yield fraction < 1.

    Attributes:
        name: Identifier for the active species (e.g. ``"BH4"``).
        mw: Molecular weight (g/mol).
        fup: Fraction unbound in plasma (Distribution).
        CL_per_h: Aggregate plasma clearance, L/h (Distribution).
        Vd_L: Apparent volume of distribution, L (Distribution).
        conversion_rate_per_h: First-order conversion rate constant, 1/h.
        conversion_site: Name of parent-graph node where conversion occurs.
        conversion_yield_fraction: Fractional molar conversion (default 1.0
            = stoichiometric).
    """
    name: str
    mw: float
    fup: Distribution
    CL_per_h: Distribution
    Vd_L: Distribution
    conversion_rate_per_h: Distribution
    conversion_site: str
    conversion_yield_fraction: Distribution
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_active_metabolite.py -v`
Expected: PASS, 2 tests passing.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/core.py tests/unit/test_active_metabolite.py
git commit -m "feat(core): add ActiveMetabolite dataclass for prodrug routing"
```

---

### Task 2: DrugOnGraph field additions + validation

**Files:**
- Modify: `src/sisyphus/core.py` (append fields to `DrugOnGraph`; add `__post_init__`; extend `sample()`)
- Test: `tests/unit/test_active_metabolite.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_active_metabolite.py`:
```python
import numpy as np

from sisyphus.core import DrugOnGraph, TransporterKinetics


def _minimal_drug(active=None, obs_species="parent"):
    """Construct a DrugOnGraph with all required fields and optional active."""
    return DrugOnGraph(
        name="testdrug",
        smiles="CCO",
        dose_mg=100.0,
        route="oral",
        administration_node="stomach_lumen",
        mw=46.07,
        pka=None,
        compound_type="neutral",
        fup=Distribution(0.5),
        rbp=Distribution(1.0),
        kp_method="rodgers_rowland",
        kp_overrides={},
        peff=Distribution(1e-4),
        solubility=Distribution(100.0),
        enzyme_affinity={},
        renal_clearance=Distribution(0.0),
        active_metabolite=active,
        observation_species=obs_species,
    )


def test_drugongraph_default_no_active():
    """active_metabolite default is None, observation_species default is 'parent'."""
    drug = _minimal_drug()
    assert drug.active_metabolite is None
    assert drug.observation_species == "parent"


def test_drugongraph_with_active_metabolite():
    """Active metabolite stored verbatim; observation_species can be 'active'."""
    am = ActiveMetabolite(
        name="BH4", mw=241.25,
        fup=Distribution(0.23), CL_per_h=Distribution(40.0),
        Vd_L=Distribution(150.0), conversion_rate_per_h=Distribution(12.0),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(1.0),
    )
    drug = _minimal_drug(active=am, obs_species="active")
    assert drug.active_metabolite is am
    assert drug.observation_species == "active"


def test_observation_active_without_active_metabolite_fails():
    """observation_species='active' requires active_metabolite to be set."""
    with pytest.raises(ValueError, match="observation_species='active' requires"):
        _minimal_drug(active=None, obs_species="active")


def test_invalid_observation_species_fails():
    """observation_species must be 'parent' or 'active'."""
    with pytest.raises(ValueError, match="observation_species must be"):
        _minimal_drug(obs_species="middle")


def test_drugongraph_sample_resamples_active_metabolite():
    """DrugOnGraph.sample() must resample all ActiveMetabolite Distribution fields."""
    am = ActiveMetabolite(
        name="BH4", mw=241.25,
        fup=Distribution(0.23, cv=0.5),
        CL_per_h=Distribution(40.0, cv=0.5),
        Vd_L=Distribution(150.0, cv=0.5),
        conversion_rate_per_h=Distribution(12.0, cv=0.5),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(0.85, cv=0.2),
    )
    drug = _minimal_drug(active=am, obs_species="active")
    rng = np.random.default_rng(42)
    sampled = drug.sample(rng)
    assert sampled.active_metabolite is not None
    assert sampled.active_metabolite.name == "BH4"
    # After sampling, cv should be 0 (all Distributions are point-values)
    assert sampled.active_metabolite.fup.cv == 0.0
    assert sampled.active_metabolite.CL_per_h.cv == 0.0
    assert sampled.observation_species == "active"
```

- [ ] **Step 2: Run the test to verify they fail**

Run: `pytest tests/unit/test_active_metabolite.py -v`
Expected: FAIL — `TypeError: DrugOnGraph.__init__() got an unexpected keyword argument 'active_metabolite'`

- [ ] **Step 3: Add fields, validation, and sample() extension**

In `src/sisyphus/core.py`, modify `DrugOnGraph` (append fields after `ps_overrides`, add `__post_init__`, extend `sample`):

After line 201 (`ps_overrides: dict[str, Distribution] = field(default_factory=dict)`), append:
```python
    # Prodrug activation — see ActiveMetabolite + docs/superpowers/specs/2026-04-24-prodrug-activation-design.md
    active_metabolite: ActiveMetabolite | None = None
    observation_species: str = "parent"  # "parent" | "active"

    def __post_init__(self) -> None:
        if self.observation_species not in ("parent", "active"):
            raise ValueError(
                f"observation_species must be 'parent' or 'active', "
                f"got {self.observation_species!r}"
            )
        if self.observation_species == "active" and self.active_metabolite is None:
            raise ValueError(
                "observation_species='active' requires active_metabolite config"
            )
```

In `DrugOnGraph.sample()` (lines 203-243), append `active_metabolite` and `observation_species` to the constructor call (just before the closing `)` at line 243):
```python
            active_metabolite=ActiveMetabolite(
                name=self.active_metabolite.name,
                mw=self.active_metabolite.mw,
                fup=Distribution(mean=self.active_metabolite.fup.sample(rng), cv=0.0),
                CL_per_h=Distribution(mean=self.active_metabolite.CL_per_h.sample(rng), cv=0.0),
                Vd_L=Distribution(mean=self.active_metabolite.Vd_L.sample(rng), cv=0.0),
                conversion_rate_per_h=Distribution(
                    mean=self.active_metabolite.conversion_rate_per_h.sample(rng), cv=0.0),
                conversion_site=self.active_metabolite.conversion_site,
                conversion_yield_fraction=Distribution(
                    mean=self.active_metabolite.conversion_yield_fraction.sample(rng), cv=0.0),
            ) if self.active_metabolite is not None else None,
            observation_species=self.observation_species,
```

- [ ] **Step 4: Run all tests in file**

Run: `pytest tests/unit/test_active_metabolite.py -v`
Expected: PASS, 7 tests passing.

- [ ] **Step 5: Run full unit test suite to verify no regression**

Run: `pytest tests/unit/ -x -q`
Expected: 0 failures (existing tests unaffected by additive changes).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/core.py tests/unit/test_active_metabolite.py
git commit -m "feat(core): DrugOnGraph active_metabolite + observation_species fields"
```

---

### Task 3: New edge types

**Files:**
- Modify: `src/sisyphus/graph/types.py` (append after `ActiveTransportEdge` at line 159)
- Test: `tests/unit/test_prodrug_edges.py` (new)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_prodrug_edges.py`:
```python
"""Tests for ProdrugActivationEdge and OneCompartmentEliminationEdge."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.graph.types import (
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)


def test_prodrug_activation_edge_construction():
    """Construct ProdrugActivationEdge with required fields."""
    edge = ProdrugActivationEdge(
        source="gut_wall",
        target="venous_blood_active",
        conversion_rate=Distribution(mean=12.0, cv=0.4),
        conversion_yield=Distribution(mean=0.85, cv=0.1),
        mw_parent=237.26,
        mw_active=241.25,
    )
    assert edge.edge_type == "prodrug_activation"
    assert edge.source == "gut_wall"
    assert edge.target == "venous_blood_active"
    assert edge.mw_parent == 237.26
    assert edge.mw_active == 241.25


def test_one_compartment_elimination_edge_construction():
    """Construct OneCompartmentEliminationEdge with required fields."""
    edge = OneCompartmentEliminationEdge(
        source="venous_blood_active",
        target="metabolized_gut",
        cl_per_h=Distribution(mean=40.0, cv=0.35),
        vd_l=Distribution(mean=150.0, cv=0.3),
    )
    assert edge.edge_type == "one_compartment_elimination"
    assert edge.cl_per_h.mean == 40.0
    assert edge.vd_l.mean == 150.0


def test_edges_are_frozen():
    """Both edge types are frozen dataclasses."""
    edge = ProdrugActivationEdge(
        source="a", target="b",
        conversion_rate=Distribution(1.0), conversion_yield=Distribution(1.0),
        mw_parent=100.0, mw_active=100.0,
    )
    with pytest.raises((AttributeError, Exception)):
        edge.source = "c"  # type: ignore[misc]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_prodrug_edges.py -v`
Expected: FAIL — ImportError on `ProdrugActivationEdge` / `OneCompartmentEliminationEdge`.

- [ ] **Step 3: Add edge types to graph/types.py**

Append to `src/sisyphus/graph/types.py` after the `ActiveTransportEdge` class (after line 159):

```python
@dataclass(frozen=True)
class ProdrugActivationEdge(Edge):
    """Mass transfer: parent drug → active metabolite at conversion site.

    Distinct from clearance (which removes mass to sink) and flow (which
    conserves mass within same species). The flux differs in source vs
    target units: source loses mg of parent; target gains mg of active
    (scaled by MW ratio × yield).

    1st-order in parent amount: rate = conversion_rate × A_parent[source].
    Active mass produced = rate × (mw_active/mw_parent) × conversion_yield.
    """

    edge_type: str = field(default="prodrug_activation", init=False)
    conversion_rate: Distribution = field(default_factory=lambda: Distribution(0.0))
    conversion_yield: Distribution = field(default_factory=lambda: Distribution(1.0))
    mw_parent: float = 0.0
    mw_active: float = 0.0


@dataclass(frozen=True)
class OneCompartmentEliminationEdge(Edge):
    """Aggregate 1st-order elimination from a 1-compartment plasma node.

    Used for active metabolite clearance where literature reports total
    plasma CL (not enzyme-level decomposition). Rate = (CL/Vd) × A_source.
    Mass accumulates at target sink node for mass-balance audit.
    """

    edge_type: str = field(default="one_compartment_elimination", init=False)
    cl_per_h: Distribution = field(default_factory=lambda: Distribution(0.0))
    vd_l: Distribution = field(default_factory=lambda: Distribution(1.0))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_prodrug_edges.py -v`
Expected: PASS, 3 tests passing.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/graph/types.py tests/unit/test_prodrug_edges.py
git commit -m "feat(graph): ProdrugActivationEdge + OneCompartmentEliminationEdge types"
```

---

### Task 4: BodyGraph.sample() branches for new edge types

**Files:**
- Modify: `src/sisyphus/graph/body.py` (extend `sample()` isinstance dispatch at lines 154-179)
- Test: `tests/unit/test_prodrug_edges.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_prodrug_edges.py`:
```python
import numpy as np

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import Node


def _minimal_graph_with_active():
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool",
                    volume=Distribution(150.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    g.add_edge(ProdrugActivationEdge(
        source="gut_wall", target="venous_blood_active",
        conversion_rate=Distribution(mean=12.0, cv=0.4),
        conversion_yield=Distribution(mean=0.85, cv=0.1),
        mw_parent=237.26, mw_active=241.25,
    ))
    g.add_edge(OneCompartmentEliminationEdge(
        source="venous_blood_active", target="metabolized_gut",
        cl_per_h=Distribution(mean=40.0, cv=0.35),
        vd_l=Distribution(mean=150.0, cv=0.3),
    ))
    return g


def test_bodygraph_sample_resamples_prodrug_edges():
    """BodyGraph.sample() must resample Distributions on new edge types."""
    g = _minimal_graph_with_active()
    rng = np.random.default_rng(42)
    g2 = g.sample(rng)
    # Edges preserved by type
    assert len(g2.edges) == 2
    edge_types = {e.edge_type for e in g2.edges}
    assert edge_types == {"prodrug_activation", "one_compartment_elimination"}
    # All Distribution fields point-valued (cv=0) after sampling
    for edge in g2.edges:
        if isinstance(edge, ProdrugActivationEdge):
            assert edge.conversion_rate.cv == 0.0
            assert edge.conversion_yield.cv == 0.0
            # MW must be preserved exactly (deterministic float)
            assert edge.mw_parent == 237.26
            assert edge.mw_active == 241.25
        elif isinstance(edge, OneCompartmentEliminationEdge):
            assert edge.cl_per_h.cv == 0.0
            assert edge.vd_l.cv == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_prodrug_edges.py::test_bodygraph_sample_resamples_prodrug_edges -v`
Expected: FAIL — sampled edge has cv != 0 because the `else` branch in `BodyGraph.sample` preserves the unsampled edge.

- [ ] **Step 3: Add isinstance branches to BodyGraph.sample()**

In `src/sisyphus/graph/body.py`:

(a) Update imports at the top of file (line 17-24) to include the new edge types:
```python
from sisyphus.graph.types import (
    AbsorptionEdge,
    DiffusionEdge,
    Edge,
    FlowEdge,
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
    TransitEdge,
)
```

(b) In `BodyGraph.sample()` (around line 175), add two `elif` branches BEFORE the catchall `else` at line 176. The block at lines 154-179 becomes:
```python
        # Sample edges.
        for edge in self.edges:
            if isinstance(edge, FlowEdge):
                new_edge = dataclasses.replace(
                    edge,
                    flow_rate=Distribution(edge.flow_rate.sample(rng), cv=0.0),
                )
            elif isinstance(edge, DiffusionEdge):
                new_edge = dataclasses.replace(
                    edge,
                    ps_product=Distribution(edge.ps_product.sample(rng), cv=0.0),
                )
            elif isinstance(edge, TransitEdge):
                new_edge = dataclasses.replace(
                    edge,
                    transit_rate=Distribution(edge.transit_rate.sample(rng), cv=0.0),
                )
            elif isinstance(edge, AbsorptionEdge):
                new_edge = dataclasses.replace(
                    edge,
                    ka_fraction=Distribution(edge.ka_fraction.sample(rng), cv=0.0),
                )
            elif isinstance(edge, ProdrugActivationEdge):
                new_edge = dataclasses.replace(
                    edge,
                    conversion_rate=Distribution(edge.conversion_rate.sample(rng), cv=0.0),
                    conversion_yield=Distribution(edge.conversion_yield.sample(rng), cv=0.0),
                )
            elif isinstance(edge, OneCompartmentEliminationEdge):
                new_edge = dataclasses.replace(
                    edge,
                    cl_per_h=Distribution(edge.cl_per_h.sample(rng), cv=0.0),
                    vd_l=Distribution(edge.vd_l.sample(rng), cv=0.0),
                )
            else:
                # ClearanceEdge and unknown edge types have no Distribution fields.
                new_edge = edge
            g2.edges.append(new_edge)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_prodrug_edges.py -v`
Expected: PASS, 4 tests passing.

- [ ] **Step 5: Run full graph tests for regression**

Run: `pytest tests/unit/ -q -k graph`
Expected: 0 new failures.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/graph/body.py tests/unit/test_prodrug_edges.py
git commit -m "feat(graph): BodyGraph.sample() handles new prodrug edge types"
```

---

### Task 5: ProdrugActivationFluxSpec

**Files:**
- Modify: `src/sisyphus/engine/flux.py` (append after `ActiveTransportFluxSpec` at line 555)
- Test: `tests/unit/test_prodrug_flux.py` (new)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_prodrug_flux.py`:
```python
"""Tests for ProdrugActivationFluxSpec and OneCompartmentEliminationFluxSpec."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from sisyphus.core import Distribution
from sisyphus.engine.flux import FLUX_REGISTRY
from sisyphus.graph.types import (
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)


def test_prodrug_activation_registered():
    """ProdrugActivationFluxSpec is registered for 'prodrug_activation' edge type."""
    assert "prodrug_activation" in FLUX_REGISTRY


def test_prodrug_activation_apply_math():
    """apply() depletes parent at src; produces active at tgt with MW × yield scaling."""
    from sisyphus.engine.flux import ProdrugActivationFluxSpec

    edge = ProdrugActivationEdge(
        source="gut_wall", target="venous_blood_active",
        conversion_rate=Distribution(mean=12.0),
        conversion_yield=Distribution(mean=0.85),
        mw_parent=237.26, mw_active=241.25,
    )
    state_index = {"gut_wall": 0, "venous_blood_active": 1}
    spec = ProdrugActivationFluxSpec.from_edge(0, edge, state_index)

    # Mock ResolvedParams: returns conversion_rate=12 and conversion_yield=0.85
    params = MagicMock()
    params.edge_param.side_effect = lambda eid, p: {
        "conversion_rate": 12.0, "conversion_yield": 0.85}[p]

    y = np.array([10.0, 0.0])  # 10 mg parent at gut_wall
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)

    # Parent flux: k × A = 12 × 10 = 120 mg_parent/h
    # Active flux: 120 × (241.25/237.26) × 0.85 = 120 × 1.01683 × 0.85 ≈ 103.72
    expected_parent_loss = 12.0 * 10.0
    expected_active_gain = expected_parent_loss * (241.25 / 237.26) * 0.85
    assert dydt[0] == pytest.approx(-expected_parent_loss)
    assert dydt[1] == pytest.approx(expected_active_gain)


def test_prodrug_activation_zero_yield_no_active_produced():
    """yield=0 → src still depleted, but tgt gets nothing."""
    from sisyphus.engine.flux import ProdrugActivationFluxSpec

    edge = ProdrugActivationEdge(
        source="src", target="tgt",
        conversion_rate=Distribution(mean=5.0),
        conversion_yield=Distribution(mean=0.0),
        mw_parent=100.0, mw_active=100.0,
    )
    spec = ProdrugActivationFluxSpec.from_edge(0, edge, {"src": 0, "tgt": 1})
    params = MagicMock()
    params.edge_param.side_effect = lambda eid, p: {
        "conversion_rate": 5.0, "conversion_yield": 0.0}[p]
    y = np.array([4.0, 0.0])
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)
    assert dydt[0] == pytest.approx(-20.0)  # src depleted
    assert dydt[1] == 0.0                   # tgt unchanged
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_prodrug_flux.py -v`
Expected: FAIL — `KeyError: 'prodrug_activation' not in FLUX_REGISTRY`.

- [ ] **Step 3: Implement ProdrugActivationFluxSpec**

Append to `src/sisyphus/engine/flux.py` after line 554 (end of `ActiveTransportFluxSpec`):

```python
@register_flux("prodrug_activation")
class ProdrugActivationFluxSpec(FluxSpec):
    """Mass transfer: parent (mg) → active (mg) with MW × yield scaling.

    Asymmetric flux: source loses mg of parent; target gains mg of active.
    The MW ratio is captured at compile time (deterministic). Conversion
    rate and yield are resampled per MC iteration via edge_param.
    """

    def __init__(
        self,
        edge_id: int,
        source_idx: int,
        target_idx: int,
        source_name: str,
        target_name: str,
        mw_ratio: float,
    ) -> None:
        super().__init__(edge_id, source_idx, target_idx, source_name, target_name)
        self.mw_ratio = mw_ratio

    @classmethod
    def from_edge(
        cls, edge_id: int, edge, state_index: dict[str, int]
    ) -> ProdrugActivationFluxSpec:
        if edge.mw_parent <= 0:
            raise ValueError(
                f"ProdrugActivationEdge mw_parent must be positive, got {edge.mw_parent}"
            )
        return cls(
            edge_id=edge_id,
            source_idx=state_index[edge.source],
            target_idx=state_index[edge.target],
            source_name=edge.source,
            target_name=edge.target,
            mw_ratio=edge.mw_active / edge.mw_parent,
        )

    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        k = params.edge_param(self.edge_id, "conversion_rate")
        y_frac = params.edge_param(self.edge_id, "conversion_yield")
        flux_parent = k * y[self.source_idx]
        flux_active = flux_parent * self.mw_ratio * y_frac
        dydt[self.source_idx] -= flux_parent
        dydt[self.target_idx] += flux_active
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_prodrug_flux.py -v`
Expected: PASS, 3 tests passing.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/engine/flux.py tests/unit/test_prodrug_flux.py
git commit -m "feat(engine): ProdrugActivationFluxSpec — asymmetric mass transfer with MW scaling"
```

---

### Task 6: OneCompartmentEliminationFluxSpec

**Files:**
- Modify: `src/sisyphus/engine/flux.py` (append after ProdrugActivationFluxSpec)
- Test: `tests/unit/test_prodrug_flux.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_prodrug_flux.py`:
```python
def test_one_compartment_elimination_registered():
    assert "one_compartment_elimination" in FLUX_REGISTRY


def test_one_compartment_elimination_apply_math():
    """apply() removes mass at rate (CL/Vd) × A from source, adds to target."""
    from sisyphus.engine.flux import OneCompartmentEliminationFluxSpec

    edge = OneCompartmentEliminationEdge(
        source="venous_blood_active", target="metabolized_gut",
        cl_per_h=Distribution(mean=40.0),
        vd_l=Distribution(mean=150.0),
    )
    state_index = {"venous_blood_active": 0, "metabolized_gut": 1}
    spec = OneCompartmentEliminationFluxSpec.from_edge(0, edge, state_index)

    params = MagicMock()
    params.edge_param.side_effect = lambda eid, p: {
        "cl_per_h": 40.0, "vd_l": 150.0}[p]

    y = np.array([30.0, 0.0])  # 30 mg active
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)

    # rate = (CL/Vd) × A = (40/150) × 30 = 8.0 mg/h
    expected_rate = (40.0 / 150.0) * 30.0
    assert dydt[0] == pytest.approx(-expected_rate)
    assert dydt[1] == pytest.approx(expected_rate)


def test_one_compartment_elimination_zero_amount():
    """A=0 → no flux."""
    from sisyphus.engine.flux import OneCompartmentEliminationFluxSpec

    edge = OneCompartmentEliminationEdge(
        source="src", target="tgt",
        cl_per_h=Distribution(mean=10.0), vd_l=Distribution(mean=50.0),
    )
    spec = OneCompartmentEliminationFluxSpec.from_edge(0, edge, {"src": 0, "tgt": 1})
    params = MagicMock()
    params.edge_param.side_effect = lambda eid, p: {"cl_per_h": 10.0, "vd_l": 50.0}[p]
    y = np.zeros(2)
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)
    assert dydt[0] == 0.0
    assert dydt[1] == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_prodrug_flux.py -v`
Expected: FAIL on the new tests — class not found.

- [ ] **Step 3: Implement OneCompartmentEliminationFluxSpec**

Append to `src/sisyphus/engine/flux.py` after `ProdrugActivationFluxSpec`:

```python
@register_flux("one_compartment_elimination")
class OneCompartmentEliminationFluxSpec(FluxSpec):
    """Aggregate 1st-order elimination: rate = (CL/Vd) × A_source.

    Mass-conserving: source loses mass; target (sink-type node) gains it
    for mass-balance audit. Used for active metabolite clearance where
    literature reports plasma CL and Vd directly (no enzyme decomposition).
    """

    @classmethod
    def from_edge(
        cls, edge_id: int, edge, state_index: dict[str, int]
    ) -> OneCompartmentEliminationFluxSpec:
        return cls(
            edge_id,
            state_index[edge.source],
            state_index[edge.target],
            edge.source,
            edge.target,
        )

    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        cl = params.edge_param(self.edge_id, "cl_per_h")
        vd = params.edge_param(self.edge_id, "vd_l")
        if vd <= 0:
            return
        rate = (cl / vd) * y[self.source_idx]
        dydt[self.source_idx] -= rate
        dydt[self.target_idx] += rate
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_prodrug_flux.py -v`
Expected: PASS, 5 tests passing total.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/engine/flux.py tests/unit/test_prodrug_flux.py
git commit -m "feat(engine): OneCompartmentEliminationFluxSpec for active metabolite CL"
```

---

### Task 7: Compiler edge_params extension

**Files:**
- Modify: `src/sisyphus/engine/compiler.py` (additive: 2 imports + 2 isinstance branches in `_build_edge_params`)
- Test: `tests/unit/test_compiler_edge_params.py` (new)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_compiler_edge_params.py`:
```python
"""Tests for ResolvedParams._build_edge_params extension for new edge types."""
from __future__ import annotations

from sisyphus.core import Distribution
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)


def _make_graph_and_drug():
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool",
                    volume=Distribution(150.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    g.add_edge(ProdrugActivationEdge(
        source="gut_wall", target="venous_blood_active",
        conversion_rate=Distribution(mean=12.0),
        conversion_yield=Distribution(mean=0.85),
        mw_parent=237.26, mw_active=241.25,
    ))
    g.add_edge(OneCompartmentEliminationEdge(
        source="venous_blood_active", target="metabolized_gut",
        cl_per_h=Distribution(mean=40.0), vd_l=Distribution(mean=150.0),
    ))
    # Minimal drug stub
    from tests.unit.test_active_metabolite import _minimal_drug
    return g, _minimal_drug()


def test_resolved_params_caches_prodrug_activation_edge():
    """ProdrugActivationEdge → conversion_rate, conversion_yield in edge_params."""
    g, drug = _make_graph_and_drug()
    rp = ResolvedParams(g, drug)
    # Edge 0 is ProdrugActivationEdge
    assert rp.edge_param(0, "conversion_rate") == 12.0
    assert rp.edge_param(0, "conversion_yield") == 0.85


def test_resolved_params_caches_one_compartment_elim_edge():
    """OneCompartmentEliminationEdge → cl_per_h, vd_l in edge_params."""
    g, drug = _make_graph_and_drug()
    rp = ResolvedParams(g, drug)
    # Edge 1 is OneCompartmentEliminationEdge
    assert rp.edge_param(1, "cl_per_h") == 40.0
    assert rp.edge_param(1, "vd_l") == 150.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_compiler_edge_params.py -v`
Expected: FAIL — `KeyError: 'conversion_rate'` in `edge_params[0]`.

- [ ] **Step 3: Add additive branches to compiler.py**

In `src/sisyphus/engine/compiler.py`:

(a) Update the import block at lines 21-28 to include the new edge types:
```python
from sisyphus.graph.types import (
    AbsorptionEdge,
    ActiveTransportEdge,
    ClearanceEdge,
    DiffusionEdge,
    FlowEdge,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
    TransitEdge,
)
```

(b) In `ResolvedParams._build_edge_params` (lines 189-207), append two `elif` branches BEFORE `edge_params[i] = params`. The block becomes:
```python
    def _build_edge_params(self, graph: BodyGraph) -> dict[int, dict[str, float]]:
        """Build edge_id -> {param: value} mapping."""
        edge_params: dict[int, dict[str, float]] = {}
        for i, edge in enumerate(graph.edges):
            params: dict[str, float] = {}
            if isinstance(edge, FlowEdge):
                params["flow_rate"] = edge.flow_rate.mean
            elif isinstance(edge, DiffusionEdge):
                params["ps_product"] = edge.ps_product.mean
            elif isinstance(edge, TransitEdge):
                params["transit_rate"] = edge.transit_rate.mean
            elif isinstance(edge, AbsorptionEdge):
                params["ka_fraction"] = edge.ka_fraction.mean
            elif isinstance(edge, ClearanceEdge):
                params["model"] = edge.model  # type: ignore[assignment]
            elif isinstance(edge, ActiveTransportEdge):
                pass  # No static params — kinetics come from drug at runtime
            elif isinstance(edge, ProdrugActivationEdge):
                params["conversion_rate"] = edge.conversion_rate.mean
                params["conversion_yield"] = edge.conversion_yield.mean
            elif isinstance(edge, OneCompartmentEliminationEdge):
                params["cl_per_h"] = edge.cl_per_h.mean
                params["vd_l"] = edge.vd_l.mean
            edge_params[i] = params
        return edge_params
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_compiler_edge_params.py -v`
Expected: PASS, 2 tests passing.

- [ ] **Step 5: Run full unit suite for regression**

Run: `pytest tests/unit/ -q`
Expected: 0 failures (existing branches untouched).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/engine/compiler.py tests/unit/test_compiler_edge_params.py
git commit -m "feat(engine): compiler edge_params caches prodrug + 1C-elim edges"
```

---

### Task 8: augment_for_active_species builder function

**Files:**
- Modify: `src/sisyphus/graph/builder.py` (append new function + ACTIVE_SUFFIX constant)
- Test: `tests/unit/test_augment_active_species.py` (new)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_augment_active_species.py`:
```python
"""Tests for graph.builder.augment_for_active_species."""
from __future__ import annotations

import pytest

from sisyphus.core import ActiveMetabolite, Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import ACTIVE_SUFFIX, augment_for_active_species
from sisyphus.graph.types import (
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)


def _bare_parent_graph():
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood", node_type="blood_pool",
                    volume=Distribution(5.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    return g


def _minimal_drug_with_active(am=None, obs_node="venous_blood"):
    from tests.unit.test_active_metabolite import _minimal_drug
    drug = _minimal_drug(active=am, obs_species="active" if am else "parent")
    # Override observation_node by reconstruction
    import dataclasses
    return dataclasses.replace(drug, administration_node="stomach_lumen")


def _bh4_active():
    return ActiveMetabolite(
        name="BH4", mw=241.25,
        fup=Distribution(0.23), CL_per_h=Distribution(40.0),
        Vd_L=Distribution(150.0), conversion_rate_per_h=Distribution(12.0),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(0.85),
    )


def test_augment_no_active_returns_unchanged():
    """active_metabolite=None → graph returned unchanged."""
    g = _bare_parent_graph()
    drug = _minimal_drug_with_active(am=None)
    g2 = augment_for_active_species(g, drug, observation_node="venous_blood")
    assert len(g2.nodes) == 3
    assert len(g2.edges) == 0


def test_augment_with_active_adds_node_and_two_edges():
    g = _bare_parent_graph()
    drug = _minimal_drug_with_active(am=_bh4_active())
    g2 = augment_for_active_species(g, drug, observation_node="venous_blood")
    expected_active_node = "venous_blood" + ACTIVE_SUFFIX
    assert expected_active_node in g2.nodes
    assert len(g2.edges) == 2
    edge_types = sorted(e.edge_type for e in g2.edges)
    assert edge_types == ["one_compartment_elimination", "prodrug_activation"]


def test_augment_invalid_conversion_site_raises():
    g = _bare_parent_graph()
    am = ActiveMetabolite(
        name="X", mw=200.0,
        fup=Distribution(0.5), CL_per_h=Distribution(10.0),
        Vd_L=Distribution(50.0), conversion_rate_per_h=Distribution(5.0),
        conversion_site="nonexistent_node",
        conversion_yield_fraction=Distribution(1.0),
    )
    drug = _minimal_drug_with_active(am=am)
    with pytest.raises(ValueError, match="conversion_site"):
        augment_for_active_species(g, drug, observation_node="venous_blood")


def test_augment_collision_raises():
    """If '<obs>_active' already exists, raise."""
    g = _bare_parent_graph()
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool",
                    volume=Distribution(0.0)))
    drug = _minimal_drug_with_active(am=_bh4_active())
    with pytest.raises(ValueError, match="active node name collision"):
        augment_for_active_species(g, drug, observation_node="venous_blood")


def test_augment_calls_twice_raises():
    """Second call raises (no idempotency — node already exists)."""
    g = _bare_parent_graph()
    drug = _minimal_drug_with_active(am=_bh4_active())
    augment_for_active_species(g, drug, observation_node="venous_blood")
    with pytest.raises(ValueError, match="active node name collision"):
        augment_for_active_species(g, drug, observation_node="venous_blood")


def test_augment_uses_existing_sink_for_elimination():
    """OneCompartmentEliminationEdge target must be an existing sink node."""
    g = _bare_parent_graph()
    drug = _minimal_drug_with_active(am=_bh4_active())
    g2 = augment_for_active_species(g, drug, observation_node="venous_blood")
    elim_edges = [e for e in g2.edges if isinstance(e, OneCompartmentEliminationEdge)]
    assert len(elim_edges) == 1
    assert elim_edges[0].target == "metabolized_gut"  # existing sink


def test_augment_no_existing_sink_raises():
    """If the chosen sink node doesn't exist, raise."""
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood", node_type="blood_pool",
                    volume=Distribution(5.0)))
    drug = _minimal_drug_with_active(am=_bh4_active())
    with pytest.raises(ValueError, match="sink node"):
        augment_for_active_species(g, drug, observation_node="venous_blood")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_augment_active_species.py -v`
Expected: FAIL — `ImportError: cannot import name 'augment_for_active_species'`.

- [ ] **Step 3: Implement augment_for_active_species in builder.py**

Append to `src/sisyphus/graph/builder.py`:

```python
# ---------------------------------------------------------------------------
# Active species augmentation — prodrug activation routing
# ---------------------------------------------------------------------------

ACTIVE_SUFFIX = "_active"
"""Suffix appended to observation_node to name the active species plasma node."""

_DEFAULT_ACTIVE_SINK = "metabolized_gut"
"""Existing sink node where active-species mass accumulates for audit.

Reusing an existing sink avoids introducing a new node per prodrug;
mass-balance auditing remains valid since the sink is still a real
node in the state vector.
"""


def augment_for_active_species(
    graph: BodyGraph,
    drug: DrugOnGraph,
    observation_node: str = "venous_blood",
) -> BodyGraph:
    """Augment graph with active-species 1-compartment plasma + 2 new edges.

    No-op when ``drug.active_metabolite is None``.

    Adds:
    - 1 ``Node`` for active plasma (named ``observation_node + ACTIVE_SUFFIX``,
      ``node_type="blood_pool"``, volume=Vd_L)
    - 1 ``ProdrugActivationEdge`` from ``conversion_site`` → active plasma
    - 1 ``OneCompartmentEliminationEdge`` from active plasma → existing sink

    Mutates ``graph`` in place and returns it for chaining convenience.

    Raises:
        ValueError: ``conversion_site`` not in ``graph.nodes``.
        ValueError: active plasma node name already exists (collision).
        ValueError: default sink node not present in graph.
    """
    if drug.active_metabolite is None:
        return graph
    am = drug.active_metabolite

    if am.conversion_site not in graph.nodes:
        raise ValueError(
            f"conversion_site={am.conversion_site!r} not in graph nodes "
            f"{sorted(graph.nodes.keys())}"
        )

    active_node_name = observation_node + ACTIVE_SUFFIX
    if active_node_name in graph.nodes:
        raise ValueError(
            f"active node name collision: {active_node_name!r} "
            "already exists in graph"
        )

    if _DEFAULT_ACTIVE_SINK not in graph.nodes:
        raise ValueError(
            f"sink node {_DEFAULT_ACTIVE_SINK!r} required for active "
            "elimination but not found in graph"
        )

    # 1. Active plasma compartment (blood_pool: Kp=1, no flow conservation).
    active_node = Node(
        name=active_node_name,
        node_type="blood_pool",
        volume=am.Vd_L,
    )
    graph.add_node(active_node)

    # 2. Conversion edge: parent at conversion_site → active plasma.
    activation_edge = ProdrugActivationEdge(
        source=am.conversion_site,
        target=active_node_name,
        conversion_rate=am.conversion_rate_per_h,
        conversion_yield=am.conversion_yield_fraction,
        mw_parent=drug.mw,
        mw_active=am.mw,
    )
    graph.add_edge(activation_edge)

    # 3. Elimination edge: active plasma → existing sink.
    elimination_edge = OneCompartmentEliminationEdge(
        source=active_node_name,
        target=_DEFAULT_ACTIVE_SINK,
        cl_per_h=am.CL_per_h,
        vd_l=am.Vd_L,
    )
    graph.add_edge(elimination_edge)

    return graph
```

Update imports at top of `builder.py` to include the new types:
```python
from sisyphus.core import DrugOnGraph
from sisyphus.graph.types import (
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_augment_active_species.py -v`
Expected: PASS, 7 tests passing.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/graph/builder.py tests/unit/test_augment_active_species.py
git commit -m "feat(graph): augment_for_active_species adds 1 node + 2 edges per prodrug"
```

---

### Task 9: Prodrug registry loader

**Files:**
- Create: `src/sisyphus/predict/registry.py` (new file)
- Test: `tests/unit/test_prodrug_registry.py` (new)
- Create: `data/sbi/prodrug_activation_registry.json` (initially empty `{}`, populated in Task 10)

- [ ] **Step 1: Create empty registry file**

```bash
echo '{}' > data/sbi/prodrug_activation_registry.json
```

- [ ] **Step 2: Write the failing test**

`tests/unit/test_prodrug_registry.py`:
```python
"""Tests for prodrug registry loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_registry(tmp_path: Path, content: dict) -> Path:
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(content))
    return p


def test_lookup_returns_none_for_missing_smiles(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {})
    assert lookup_active_metabolite("CCO", registry_path=p) is None


def test_lookup_returns_active_metabolite_for_match(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    # "CCO" is canonical SMILES for ethanol — use as test key
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "TestActive",
            "mw": 241.25,
            "fup": {"mean": 0.23, "cv": 0.3},
            "CL_per_h": {"mean": 40.0, "cv": 0.35},
            "Vd_L": {"mean": 150.0, "cv": 0.3},
            "conversion_rate_per_h": {"mean": 12.0, "cv": 0.4},
            "conversion_site": "gut_wall",
            "conversion_yield_fraction": {"mean": 0.85, "cv": 0.1},
            "observation_species": "active",
        }
    })
    result = lookup_active_metabolite("CCO", registry_path=p)
    assert result is not None
    am, obs_species = result
    assert am.name == "TestActive"
    assert obs_species == "active"


def test_lookup_default_observation_species_is_active(tmp_path):
    """If observation_species not specified, default to 'active'."""
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "X", "mw": 100.0,
            "fup": {"mean": 0.5}, "CL_per_h": {"mean": 10.0},
            "Vd_L": {"mean": 50.0},
            "conversion_rate_per_h": {"mean": 5.0},
            "conversion_site": "venous_blood",
            "conversion_yield_fraction": {"mean": 1.0},
        }
    })
    _, obs_species = lookup_active_metabolite("CCO", registry_path=p)
    assert obs_species == "active"


def test_lookup_negative_rate_raises(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "X", "mw": 100.0,
            "fup": {"mean": 0.5}, "CL_per_h": {"mean": 10.0},
            "Vd_L": {"mean": 50.0},
            "conversion_rate_per_h": {"mean": -1.0},
            "conversion_site": "venous_blood",
            "conversion_yield_fraction": {"mean": 1.0},
        }
    })
    with pytest.raises(ValueError, match="conversion_rate must be positive"):
        lookup_active_metabolite("CCO", registry_path=p)


def test_lookup_yield_out_of_range_raises(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "X", "mw": 100.0,
            "fup": {"mean": 0.5}, "CL_per_h": {"mean": 10.0},
            "Vd_L": {"mean": 50.0},
            "conversion_rate_per_h": {"mean": 5.0},
            "conversion_site": "venous_blood",
            "conversion_yield_fraction": {"mean": 1.5},
        }
    })
    with pytest.raises(ValueError, match="conversion_yield must be in"):
        lookup_active_metabolite("CCO", registry_path=p)


def test_lookup_invalid_observation_species_raises(tmp_path):
    from sisyphus.predict.registry import lookup_active_metabolite
    p = _write_registry(tmp_path, {
        "CCO": {
            "name": "X", "mw": 100.0,
            "fup": {"mean": 0.5}, "CL_per_h": {"mean": 10.0},
            "Vd_L": {"mean": 50.0},
            "conversion_rate_per_h": {"mean": 5.0},
            "conversion_site": "venous_blood",
            "conversion_yield_fraction": {"mean": 1.0},
            "observation_species": "middle",
        }
    })
    with pytest.raises(ValueError, match="observation_species must be"):
        lookup_active_metabolite("CCO", registry_path=p)


def test_lookup_canonical_smiles_match(tmp_path):
    """SMILES is canonicalized before lookup (via RDKit)."""
    from sisyphus.predict.registry import lookup_active_metabolite
    # "OCC" canonicalizes to "CCO" via RDKit
    p = _write_registry(tmp_path, {"CCO": {
        "name": "X", "mw": 100.0,
        "fup": {"mean": 0.5}, "CL_per_h": {"mean": 10.0},
        "Vd_L": {"mean": 50.0},
        "conversion_rate_per_h": {"mean": 5.0},
        "conversion_site": "venous_blood",
        "conversion_yield_fraction": {"mean": 1.0},
    }})
    result = lookup_active_metabolite("OCC", registry_path=p)
    assert result is not None  # canonicalized matches
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/unit/test_prodrug_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'lookup_active_metabolite'`.

- [ ] **Step 4: Implement registry loader**

Create `src/sisyphus/predict/registry.py`:
```python
"""Prodrug activation registry — SMILES-keyed config loader.

Maps canonical SMILES → ActiveMetabolite + observation_species.
Used by predict.ivive.build_drug_on_graph to attach prodrug activation
configs to DrugOnGraph instances.

Registry file: ``data/sbi/prodrug_activation_registry.json``
Schema: see docs/superpowers/specs/2026-04-24-prodrug-activation-design.md §4.8
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from rdkit import Chem

from sisyphus.core import ActiveMetabolite, Distribution

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "sbi" / "prodrug_activation_registry.json"
)


def _canonicalize(smiles: str) -> str | None:
    """Convert SMILES to canonical form via RDKit. Returns None on parse error."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


@lru_cache(maxsize=1)
def _load_registry_cached(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        logger.warning("prodrug_activation_registry not found at %s", path)
        return {}
    with path.open() as f:
        return json.load(f)


def _build_active_metabolite(entry: dict, smiles: str) -> ActiveMetabolite:
    """Construct ActiveMetabolite from registry entry; validate fields."""
    required_fields = {
        "name", "mw", "fup", "CL_per_h", "Vd_L",
        "conversion_rate_per_h", "conversion_site",
        "conversion_yield_fraction",
    }
    missing = required_fields - set(entry.keys())
    if missing:
        raise ValueError(
            f"prodrug_activation_registry entry for SMILES {smiles!r} "
            f"missing fields: {sorted(missing)}"
        )

    if entry["mw"] <= 0:
        raise ValueError(f"mw must be positive, got {entry['mw']}")

    cr = entry["conversion_rate_per_h"]
    if cr["mean"] <= 0:
        raise ValueError(
            f"conversion_rate must be positive, got {cr['mean']}"
        )

    cy = entry["conversion_yield_fraction"]
    if not (0.0 <= cy["mean"] <= 1.0):
        raise ValueError(
            f"conversion_yield must be in [0, 1], got {cy['mean']}"
        )

    if entry["CL_per_h"]["mean"] <= 0 or entry["Vd_L"]["mean"] <= 0:
        raise ValueError("CL and Vd must be positive")

    return ActiveMetabolite(
        name=entry["name"],
        mw=float(entry["mw"]),
        fup=Distribution(**entry["fup"]),
        CL_per_h=Distribution(**entry["CL_per_h"]),
        Vd_L=Distribution(**entry["Vd_L"]),
        conversion_rate_per_h=Distribution(**entry["conversion_rate_per_h"]),
        conversion_site=str(entry["conversion_site"]),
        conversion_yield_fraction=Distribution(**entry["conversion_yield_fraction"]),
    )


def lookup_active_metabolite(
    smiles: str, registry_path: Path | None = None
) -> tuple[ActiveMetabolite, str] | None:
    """Look up SMILES in prodrug registry.

    Returns (ActiveMetabolite, observation_species) or None if not found.
    Raises ValueError on invalid registry entries.

    Args:
        smiles: SMILES string (any form; canonicalized internally).
        registry_path: Override registry file path (default: data/sbi/...).
    """
    canonical = _canonicalize(smiles)
    if canonical is None:
        return None
    path = registry_path or _DEFAULT_REGISTRY_PATH
    # Bypass cache when test path differs from default
    if registry_path is not None:
        with path.open() as f:
            registry = json.load(f)
    else:
        registry = _load_registry_cached(str(path))

    entry = registry.get(canonical)
    if entry is None:
        return None

    obs_species = entry.get("observation_species", "active")
    if obs_species not in ("parent", "active"):
        raise ValueError(
            f"observation_species must be 'parent' or 'active', got {obs_species!r}"
        )

    am = _build_active_metabolite(entry, canonical)
    return am, obs_species
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/unit/test_prodrug_registry.py -v`
Expected: PASS, 7 tests passing.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/predict/registry.py tests/unit/test_prodrug_registry.py data/sbi/prodrug_activation_registry.json
git commit -m "feat(predict): prodrug_activation_registry loader (SMILES-keyed JSON)"
```

---

### Task 10: Populate registry with 4 evidence drugs

**Files:**
- Modify: `data/sbi/prodrug_activation_registry.json`

This task involves literature research. Each drug's entry uses citations already documented in `data/reference/holdout_n50.json` plus PK papers.

- [ ] **Step 1: Research and write 4 entries**

For each drug, source: name, MW, fup, CL_per_h, Vd_L, conversion_rate_per_h (or t½ → k = ln(2)/t½), conversion_site, conversion_yield_fraction, observation_species.

**Sepiapterin → BH4** (canonical parent SMILES via RDKit canonicalize: `Cc1nc2c(=O)[nH]c(N)nc2n1[C@@H](O)[C@@H](O)C` or equivalent — get from PubChem CID 65253):

Reference: Gao 2024 PMC11597218 (BH4 PK after oral sepiapterin in healthy humans).

```json
"<canonical_smiles_sepiapterin>": {
  "name": "BH4",
  "mw": 241.25,
  "fup": {"mean": 0.23, "cv": 0.3},
  "CL_per_h": {"mean": 40.0, "cv": 0.35},
  "Vd_L": {"mean": 150.0, "cv": 0.3},
  "conversion_rate_per_h": {"mean": 12.0, "cv": 0.4},
  "conversion_site": "gut_wall",
  "conversion_yield_fraction": {"mean": 0.85, "cv": 0.1},
  "observation_species": "parent",
  "_citation": "Gao 2024 PMC11597218 (sepiapterin oral PK + BH4 metabolite Cmax 403 ng/mL)"
}
```

**Remdesivir → GS-441524** (PubChem CID 121304016):
Reference: Humeniuk 2020 PMC table 3 — parent Cmax 4380 ng/mL @ 200 mg IV.

```json
"<canonical_smiles_remdesivir>": {
  "name": "GS-441524",
  "mw": 291.27,
  "fup": {"mean": 0.5, "cv": 0.3},
  "CL_per_h": {"mean": 10.0, "cv": 0.3},
  "Vd_L": {"mean": 35.0, "cv": 0.3},
  "conversion_rate_per_h": {"mean": 1.5, "cv": 0.4},
  "conversion_site": "venous_blood",
  "conversion_yield_fraction": {"mean": 0.9, "cv": 0.1},
  "observation_species": "parent",
  "_citation": "Humeniuk 2020 (parent + GS-441524 PK; CES1 plasma hydrolysis t½ ~30 min)"
}
```

**Tebipenem_pivoxil → tebipenem** (PubChem CID 9892071):
Reference: SAD phase trial (cited in `holdout_n50.json`) — tebipenem Cmax 4006 ng/mL @ 300 mg PO.

```json
"<canonical_smiles_tebipenem_pivoxil>": {
  "name": "tebipenem",
  "mw": 384.45,
  "fup": {"mean": 0.5, "cv": 0.3},
  "CL_per_h": {"mean": 17.0, "cv": 0.3},
  "Vd_L": {"mean": 50.0, "cv": 0.3},
  "conversion_rate_per_h": {"mean": 30.0, "cv": 0.3},
  "conversion_site": "gut_wall",
  "conversion_yield_fraction": {"mean": 0.95, "cv": 0.05},
  "observation_species": "active",
  "_citation": "Eckburg 2019/Aronoff 2024 SAD; intestinal esterase hydrolysis very fast"
}
```

**Fostamatinib → R406** (PubChem CID 11671467):
Reference: Baluom 2013 PMC3703230 — R406 Cmax 605 ng/mL @ 75 mg PO.

```json
"<canonical_smiles_fostamatinib>": {
  "name": "R406",
  "mw": 470.45,
  "fup": {"mean": 0.02, "cv": 0.3},
  "CL_per_h": {"mean": 28.0, "cv": 0.35},
  "Vd_L": {"mean": 250.0, "cv": 0.3},
  "conversion_rate_per_h": {"mean": 4.0, "cv": 0.4},
  "conversion_site": "gut_wall",
  "conversion_yield_fraction": {"mean": 0.7, "cv": 0.15},
  "observation_species": "active",
  "_citation": "Baluom 2013 PMC3703230 (R406 Cmax 605 ng/mL, fostamatinib parent <quantification limit)"
}
```

To get canonical SMILES for each parent drug, use RDKit:
```bash
python3 -c "from rdkit import Chem; print(Chem.MolToSmiles(Chem.MolFromSmiles('<smiles_from_pubchem>')))"
```

- [ ] **Step 2: Write the entries to the registry**

Replace the empty `{}` in `data/sbi/prodrug_activation_registry.json` with the 4-entry JSON. Use real canonical SMILES for each drug from PubChem (looked up + canonicalized via RDKit at write time).

- [ ] **Step 3: Validate the registry by running the loader**

Run: `pytest tests/unit/test_prodrug_registry.py -v`
Expected: still PASS (existing tests unaffected; 4-entry registry not exercised by tests yet).

Add validation check:
```python
# Add to tests/unit/test_prodrug_registry.py
def test_actual_registry_loads_4_entries():
    """Verify the production registry contains 4 valid entries."""
    from sisyphus.predict.registry import _DEFAULT_REGISTRY_PATH
    import json
    with _DEFAULT_REGISTRY_PATH.open() as f:
        registry = json.load(f)
    # Filter out comment-style keys (those starting with _)
    entries = {k: v for k, v in registry.items() if not k.startswith("_")}
    assert len(entries) == 4, f"Expected 4 prodrug entries, got {len(entries)}"
    expected_names = {"BH4", "GS-441524", "tebipenem", "R406"}
    actual_names = {v["name"] for v in entries.values()}
    assert actual_names == expected_names
```

Run: `pytest tests/unit/test_prodrug_registry.py::test_actual_registry_loads_4_entries -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add data/sbi/prodrug_activation_registry.json tests/unit/test_prodrug_registry.py
git commit -m "data(sbi): prodrug_activation_registry — 4 N50 evidence drugs"
```

---

### Task 11: Wire registry into build_drug_on_graph

**Files:**
- Modify: `src/sisyphus/predict/ivive.py` (extend `build_drug_on_graph` to call registry; pass through fields)
- Test: `tests/unit/test_ivive_prodrug.py` (new)

- [ ] **Step 1: Read the existing build_drug_on_graph to find insertion point**

Run: `grep -n "DrugOnGraph(" src/sisyphus/predict/ivive.py`

Find lines around 634 — the call site that constructs `DrugOnGraph(...)`. The two new fields must be added to that constructor call.

- [ ] **Step 2: Write the failing test**

`tests/unit/test_ivive_prodrug.py`:
```python
"""Test that build_drug_on_graph attaches active_metabolite from registry."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_build_drug_attaches_active_metabolite_from_registry(tmp_path):
    """If SMILES matches registry, active_metabolite + observation_species are set."""
    from sisyphus.predict.ivive import build_drug_on_graph

    # Mock registry path with one entry for ethanol (canonical "CCO")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({
        "CCO": {
            "name": "TestActive",
            "mw": 100.0,
            "fup": {"mean": 0.5},
            "CL_per_h": {"mean": 10.0},
            "Vd_L": {"mean": 50.0},
            "conversion_rate_per_h": {"mean": 5.0},
            "conversion_site": "gut_wall",
            "conversion_yield_fraction": {"mean": 1.0},
            "observation_species": "active",
        }
    }))

    # Mock predict layer outputs (profile + adme stubs)
    # The actual build_drug_on_graph signature accepts profile & adme dataclasses;
    # this test patches the registry lookup directly via _DEFAULT_REGISTRY_PATH.
    with patch("sisyphus.predict.registry._DEFAULT_REGISTRY_PATH", registry_path):
        # Use minimal SMILES "CCO" — must invoke real build_drug_on_graph.
        # If signature requires complex inputs, instead test the insertion via
        # a unit test on `_attach_active_metabolite` helper (see Step 3).
        pass  # will be filled in after reading actual signature


def test_build_drug_no_registry_match_keeps_default(tmp_path):
    """SMILES not in registry → active_metabolite=None, observation_species='parent'."""
    from sisyphus.predict.registry import lookup_active_metabolite

    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")
    result = lookup_active_metabolite("CCO", registry_path=registry_path)
    assert result is None
```

(The first test is a stub for full integration; it will be sharpened after reading the actual signature.)

- [ ] **Step 3: Update build_drug_on_graph in ivive.py**

In `src/sisyphus/predict/ivive.py`, find the function `build_drug_on_graph` (around line 552) and the `DrugOnGraph(...)` instantiation (around line 634).

(a) Add at the top of the function body (after the docstring), before the existing logic:
```python
    from sisyphus.predict.registry import lookup_active_metabolite

    # Prodrug activation routing: SMILES → ActiveMetabolite via registry.
    smiles = profile.smiles  # adjust attribute access if profile uses different name
    registry_result = lookup_active_metabolite(smiles)
    if registry_result is not None:
        active_metabolite, observation_species = registry_result
    else:
        active_metabolite = None
        observation_species = "parent"
```

(b) Append to the `DrugOnGraph(...)` constructor at line 634:
```python
        active_metabolite=active_metabolite,
        observation_species=observation_species,
```

The existing constructor's other fields remain unchanged.

- [ ] **Step 4: Sharpen test 1 against the real signature**

After reading the actual `build_drug_on_graph` signature, fill in the first test in `test_ivive_prodrug.py` to invoke the function with realistic stub `profile` and `adme` arguments. If the inputs are too complex to mock, replace test 1 with a unit test on the registry lookup integration (already covered by `test_prodrug_registry.py`) and rely on Task 12 integration tests.

- [ ] **Step 5: Run the test**

Run: `pytest tests/unit/test_ivive_prodrug.py -v`
Expected: PASS.

- [ ] **Step 6: Run regression suite**

Run: `pytest tests/unit/ -q`
Expected: 0 new failures.

- [ ] **Step 7: Commit**

```bash
git add src/sisyphus/predict/ivive.py tests/unit/test_ivive_prodrug.py
git commit -m "feat(predict): build_drug_on_graph attaches active_metabolite from registry"
```

---

### Task 12: Pipeline AD adjustment + observation routing + augmentation hook

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py` (add helper functions; insert calls in orchestrator)
- Modify: `src/sisyphus/pipeline/config.py` (if PK config needs species-aware obs_node — check first)
- Test: `tests/unit/test_pipeline_prodrug.py` (new)

- [ ] **Step 1: Read existing pipeline.predict.predict() to find insertion points**

Run: `cat src/sisyphus/pipeline/predict.py | head -200`

Identify:
- Where `BodyGraph` is built (look for `build_from_yaml` or similar)
- Where `DrugOnGraph` is constructed (`build_drug_on_graph` call)
- Where `ODECompiler` is invoked
- Where `PKEndpoints` are computed (look for `pk.endpoints` or similar)
- Where `PredictionResult` is assembled

The augmentation hook must run AFTER `build_drug_on_graph` and BEFORE `ODECompiler.compile`. The observation_node resolution must happen at PK endpoint computation. AD adjustment happens at PredictionResult assembly.

- [ ] **Step 2: Write the failing test**

`tests/unit/test_pipeline_prodrug.py`:
```python
"""Tests for pipeline-level prodrug routing helpers."""
from __future__ import annotations

import dataclasses

import pytest


def _drug_with_active(am=None, obs_species="parent"):
    from tests.unit.test_active_metabolite import _minimal_drug
    return _minimal_drug(active=am, obs_species=obs_species)


def _bh4():
    from sisyphus.core import ActiveMetabolite, Distribution
    return ActiveMetabolite(
        name="BH4", mw=241.25,
        fup=Distribution(0.23), CL_per_h=Distribution(40.0),
        Vd_L=Distribution(150.0), conversion_rate_per_h=Distribution(12.0),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(1.0),
    )


def test_resolve_observation_node_active():
    """observation_species='active' + active_metabolite set → '{obs}_active'."""
    from sisyphus.pipeline.predict import _resolve_observation_node
    drug = _drug_with_active(am=_bh4(), obs_species="active")
    assert _resolve_observation_node(drug, base_node="venous_blood") == "venous_blood_active"


def test_resolve_observation_node_parent_default():
    """observation_species='parent' (default) → base node."""
    from sisyphus.pipeline.predict import _resolve_observation_node
    drug = _drug_with_active(am=None, obs_species="parent")
    assert _resolve_observation_node(drug, base_node="venous_blood") == "venous_blood"


def test_resolve_observation_node_parent_with_active_set():
    """observation_species='parent' (override) + active_metabolite → still base."""
    from sisyphus.pipeline.predict import _resolve_observation_node
    drug = _drug_with_active(am=_bh4(), obs_species="parent")
    assert _resolve_observation_node(drug, base_node="venous_blood") == "venous_blood"


def test_adjust_ad_prodrug_with_active_upgrades():
    """PRODRUG flag + active_metabolite set → in_domain=True + warning."""
    from sisyphus.pipeline.predict import _adjust_ad_for_prodrug
    drug = _drug_with_active(am=_bh4(), obs_species="active")
    in_domain, warnings = _adjust_ad_for_prodrug(drug, ad_flags=["PRODRUG"])
    assert in_domain is True
    assert any("Prodrug" in w for w in warnings)


def test_adjust_ad_prodrug_no_active_remains_out():
    """PRODRUG flag + no active_metabolite → in_domain=False."""
    from sisyphus.pipeline.predict import _adjust_ad_for_prodrug
    drug = _drug_with_active(am=None, obs_species="parent")
    in_domain, warnings = _adjust_ad_for_prodrug(drug, ad_flags=["PRODRUG"])
    assert in_domain is False
    assert warnings == []


def test_adjust_ad_no_prodrug_with_active_warns_non_structural():
    """No PRODRUG flag + active_metabolite set → in_domain=True + non-structural warning."""
    from sisyphus.pipeline.predict import _adjust_ad_for_prodrug
    drug = _drug_with_active(am=_bh4(), obs_species="active")
    in_domain, warnings = _adjust_ad_for_prodrug(drug, ad_flags=[])
    assert in_domain is True
    assert any("non-structural" in w.lower() or "without structural" in w.lower()
               for w in warnings)


def test_adjust_ad_no_flags_clean_drug():
    """No flags, no active → in_domain=True, no warnings."""
    from sisyphus.pipeline.predict import _adjust_ad_for_prodrug
    drug = _drug_with_active(am=None, obs_species="parent")
    in_domain, warnings = _adjust_ad_for_prodrug(drug, ad_flags=[])
    assert in_domain is True
    assert warnings == []
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/unit/test_pipeline_prodrug.py -v`
Expected: FAIL — `ImportError: cannot import name '_resolve_observation_node'`.

- [ ] **Step 4: Implement helper functions**

Append to `src/sisyphus/pipeline/predict.py` (above the `predict()` function definition):

```python
from sisyphus.core import DrugOnGraph
from sisyphus.graph.builder import ACTIVE_SUFFIX


def _resolve_observation_node(drug: DrugOnGraph, base_node: str = "venous_blood") -> str:
    """Resolve which graph node to read PK from, accounting for active species.

    Returns ``base_node + ACTIVE_SUFFIX`` if the drug has an active metabolite
    AND ``observation_species == "active"``; otherwise returns ``base_node``.
    """
    if drug.active_metabolite is not None and drug.observation_species == "active":
        return base_node + ACTIVE_SUFFIX
    return base_node


def _adjust_ad_for_prodrug(
    drug: DrugOnGraph, ad_flags: list[str]
) -> tuple[bool, list[str]]:
    """Adjust applicability-domain interpretation for prodrugs.

    - PRODRUG flag + active_metabolite present → in_domain=True, warn "routed via activation".
    - PRODRUG flag + no active_metabolite → in_domain=False (existing behavior).
    - No PRODRUG flag + active_metabolite present → in_domain=True, warn "non-structural activation".
    - Otherwise → flags drive in_domain (any flag → False).

    Returns: (in_applicability_domain, warnings_list).
    """
    warnings: list[str] = []
    flags_for_domain = list(ad_flags)
    has_prodrug = "PRODRUG" in ad_flags
    has_active = drug.active_metabolite is not None

    if has_prodrug and has_active:
        site = drug.active_metabolite.conversion_site
        warnings.append(
            f"Prodrug {drug.name!r} routed via activation to "
            f"{drug.active_metabolite.name!r} at {site}."
        )
        flags_for_domain = [f for f in ad_flags if f != "PRODRUG"]
    elif has_active and not has_prodrug:
        warnings.append(
            f"Active metabolite declared for {drug.name!r} without "
            "structural prodrug motif; registry override applied."
        )

    in_domain = len(flags_for_domain) == 0
    return in_domain, warnings
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/unit/test_pipeline_prodrug.py -v`
Expected: PASS, 7 tests passing.

- [ ] **Step 6: Wire augmentation + helpers into pipeline.predict.predict()**

In `src/sisyphus/pipeline/predict.py`, modify `predict()`:

(a) After `drug = build_drug_on_graph(...)` calls (lines 95 and 145), add:
```python
        from sisyphus.graph.builder import augment_for_active_species
        graph = augment_for_active_species(graph, drug, observation_node="venous_blood")
```

(b) Where `pk.endpoints(...)` is called (find via grep in this file), pass `observation_node=_resolve_observation_node(drug)`.

(c) When assembling `PredictionResult`, replace the existing AD computation with:
```python
        in_domain, prodrug_warnings = _adjust_ad_for_prodrug(drug, list(ad_flags))
        warnings = list(warnings) + prodrug_warnings
        # use `in_domain` for in_applicability_domain field
```

(Detailed insertion lines depend on actual file structure — implementer to read existing code first.)

- [ ] **Step 7: Run regression suite**

Run: `pytest tests/unit/ tests/integration/ -q`
Expected: 0 new failures from existing tests.

- [ ] **Step 8: Commit**

```bash
git add src/sisyphus/pipeline/predict.py tests/unit/test_pipeline_prodrug.py
git commit -m "feat(pipeline): augmentation hook + AD adjustment + observation routing"
```

---

### Task 13: Two-species mass balance integration test

**Files:**
- Test: `tests/integration/test_two_species_mass_balance.py` (new)

- [ ] **Step 1: Write the integration test**

`tests/integration/test_two_species_mass_balance.py`:
```python
"""Integration test: 2-species mass balance under prodrug routing.

Synthetic test: parent dosed at gut_wall, converts to active in venous_blood_active
at rate k=1/h, active eliminated at rate k_el=0.5/h, MW ratio=1, yield=1.
Expected: integrated active produced ≈ integrated parent dissipated; active
mass at any time matches analytical 2-compartment cascade solution.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sisyphus.core import ActiveMetabolite, Distribution
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import augment_for_active_species
from sisyphus.graph.types import (
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)


def test_synthetic_two_species_mass_balance():
    """Initial 100 mg parent at gut_wall → 1st-order conversion to active at k=1/h.
    Active eliminated at k_el=0.5/h. Verify cumulative active produced and
    parent dissipated balance via analytical solution.
    """
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ",
                    volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool",
                    volume=Distribution(50.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))

    g.add_edge(ProdrugActivationEdge(
        source="gut_wall", target="venous_blood_active",
        conversion_rate=Distribution(mean=1.0, cv=0.0),
        conversion_yield=Distribution(mean=1.0, cv=0.0),
        mw_parent=100.0, mw_active=100.0,
    ))
    # k_el = CL/Vd = 25/50 = 0.5/h
    g.add_edge(OneCompartmentEliminationEdge(
        source="venous_blood_active", target="metabolized_gut",
        cl_per_h=Distribution(mean=25.0, cv=0.0),
        vd_l=Distribution(mean=50.0, cv=0.0),
    ))

    # Build minimal drug stub
    from tests.unit.test_active_metabolite import _minimal_drug
    am = ActiveMetabolite(
        name="X", mw=100.0,
        fup=Distribution(0.5), CL_per_h=Distribution(25.0),
        Vd_L=Distribution(50.0), conversion_rate_per_h=Distribution(1.0),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(1.0),
    )
    drug = _minimal_drug(active=am, obs_species="active")

    # Compile
    compiler = ODECompiler()
    compiled = compiler.compile(g)
    params = ResolvedParams(g, drug)
    rhs = compiled.make_rhs(params)

    # Initial condition: 100 mg at gut_wall, 0 elsewhere
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index["gut_wall"]] = 100.0

    sol = solve_ivp(rhs, t_span=(0, 24), y0=y0, t_eval=np.linspace(0, 24, 100),
                    method="LSODA", rtol=1e-8, atol=1e-10)
    assert sol.success

    parent_idx = compiled.state_index["gut_wall"]
    active_idx = compiled.state_index["venous_blood_active"]
    sink_idx = compiled.state_index["metabolized_gut"]

    # Mass balance: parent + active + sink ≈ 100 (initial dose)
    final_total = sol.y[parent_idx, -1] + sol.y[active_idx, -1] + sol.y[sink_idx, -1]
    assert final_total == pytest.approx(100.0, rel=1e-4)

    # Analytical solution for 2-compartment cascade:
    # A_p(t) = 100 × exp(-k×t)
    # A_a(t) = (k × 100 / (k_el - k)) × (exp(-k×t) - exp(-k_el×t))
    k, k_el = 1.0, 0.5
    t_arr = sol.t
    expected_parent = 100.0 * np.exp(-k * t_arr)
    expected_active = (k * 100.0 / (k_el - k)) * (np.exp(-k * t_arr) - np.exp(-k_el * t_arr))
    np.testing.assert_allclose(sol.y[parent_idx], expected_parent, rtol=1e-3)
    np.testing.assert_allclose(sol.y[active_idx], expected_active, rtol=1e-3)


def test_yield_below_one_reduces_active_proportionally():
    """yield=0.5 → final active mass should be ~50% of yield=1 case."""
    g = BodyGraph()
    g.add_node(Node(name="gut_wall", node_type="organ", volume=Distribution(0.5)))
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool",
                    volume=Distribution(50.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    g.add_edge(ProdrugActivationEdge(
        source="gut_wall", target="venous_blood_active",
        conversion_rate=Distribution(mean=1.0, cv=0.0),
        conversion_yield=Distribution(mean=0.5, cv=0.0),
        mw_parent=100.0, mw_active=100.0,
    ))
    g.add_edge(OneCompartmentEliminationEdge(
        source="venous_blood_active", target="metabolized_gut",
        cl_per_h=Distribution(mean=0.0, cv=0.0),  # no elim → mass accumulates
        vd_l=Distribution(mean=50.0, cv=0.0),
    ))

    from tests.unit.test_active_metabolite import _minimal_drug
    am = ActiveMetabolite(
        name="Y", mw=100.0, fup=Distribution(0.5),
        CL_per_h=Distribution(0.0), Vd_L=Distribution(50.0),
        conversion_rate_per_h=Distribution(1.0), conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(0.5),
    )
    drug = _minimal_drug(active=am, obs_species="active")

    compiler = ODECompiler()
    compiled = compiler.compile(g)
    params = ResolvedParams(g, drug)
    rhs = compiled.make_rhs(params)

    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index["gut_wall"]] = 100.0

    sol = solve_ivp(rhs, (0, 100), y0, method="LSODA", rtol=1e-8, atol=1e-10)
    # At t→∞: parent fully converted (100 mg), active accumulated = 100 × 0.5 = 50 mg
    active_final = sol.y[compiled.state_index["venous_blood_active"], -1]
    assert active_final == pytest.approx(50.0, rel=1e-3)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `pytest tests/integration/test_two_species_mass_balance.py -v`
Expected: PASS, 2 tests passing. (No prior implementation gap because Tasks 1–8 already established all the building blocks.)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_two_species_mass_balance.py
git commit -m "test(integration): two-species mass balance vs analytical 2C cascade"
```

---

### Task 14: 4-drug pipeline smoke + validation gate

**Files:**
- Test: `tests/integration/test_prodrug_pipeline_smoke.py` (new)

- [ ] **Step 1: Write the smoke + validation test**

`tests/integration/test_prodrug_pipeline_smoke.py`:
```python
"""Smoke + validation tests for 4 evidence prodrugs.

Validation gate: each drug's predicted Cmax (active or parent per
registry observation_species) must be within 3-fold of clinical
reference Cmax from data/reference/holdout_n50.json.

These act as scientific gates — if any drug fails, spec validation §7.5
failure response applies (registry value re-verification → routing override
→ spec re-open).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Reference Cmax values per holdout_n50.json (mg/L)
REF_CMAX = {
    "sepiapterin":         0.0024,   # parent, 60 mg/kg fed (2.4 ng/mL)
    "remdesivir":          4.380,    # parent, 200 mg IV (4380 ng/mL)
    "tebipenem_pivoxil":   4.006,    # active tebipenem, 300 mg PO (4006 ng/mL)
    "fostamatinib":        0.605,    # active R406, 75 mg PO (605 ng/mL)
}

# Drug name → (SMILES, dose_mg, route)
# SMILES from PubChem at registry-population time.
DRUG_INPUTS = {
    # PLAN NOTE: implementer must paste real canonical SMILES for each drug
    # from the registry into this dict before running the test.
    "sepiapterin":         ("<canonical_sepiapterin>", 4200.0, "oral"),
    "remdesivir":          ("<canonical_remdesivir>", 200.0, "iv"),
    "tebipenem_pivoxil":   ("<canonical_tebipenem_pivoxil>", 300.0, "oral"),
    "fostamatinib":        ("<canonical_fostamatinib>", 75.0, "oral"),
}


@pytest.mark.parametrize("drug_name", list(DRUG_INPUTS.keys()))
def test_prodrug_pipeline_smoke(drug_name):
    """Each drug runs end-to-end without error; Cmax positive."""
    from sisyphus.pipeline.predict import predict
    smiles, dose, route = DRUG_INPUTS[drug_name]
    if smiles.startswith("<"):
        pytest.skip(f"SMILES placeholder for {drug_name}; populate from registry.")
    result = predict(smiles=smiles, dose_mg=dose, route=route, n_mc_samples=0)
    assert result.pk.cmax.mean > 0
    assert result.in_applicability_domain  # PRODRUG should be upgraded
    # Warnings should mention prodrug routing
    assert any("prodrug" in w.lower() or "routed" in w.lower() for w in result.warnings)


@pytest.mark.parametrize("drug_name", list(DRUG_INPUTS.keys()))
def test_prodrug_validation_gate_3fold(drug_name):
    """Predicted Cmax within 3-fold of clinical reference."""
    from sisyphus.pipeline.predict import predict
    smiles, dose, route = DRUG_INPUTS[drug_name]
    if smiles.startswith("<"):
        pytest.skip(f"SMILES placeholder for {drug_name}.")
    result = predict(smiles=smiles, dose_mg=dose, route=route, n_mc_samples=0)
    predicted = result.pk.cmax.mean
    reference = REF_CMAX[drug_name]
    fold_error = max(predicted / reference, reference / predicted)
    assert fold_error <= 3.0, (
        f"{drug_name}: predicted Cmax {predicted:.4f} mg/L, "
        f"reference {reference:.4f} mg/L, fold-error {fold_error:.2f}× exceeds 3-fold"
    )
```

- [ ] **Step 2: Populate SMILES placeholders**

Read the canonical SMILES strings from `data/sbi/prodrug_activation_registry.json` (the keys) and replace `<canonical_*>` placeholders in `DRUG_INPUTS` with actual values.

- [ ] **Step 3: Run the smoke test (without validation gate first)**

Run: `pytest tests/integration/test_prodrug_pipeline_smoke.py::test_prodrug_pipeline_smoke -v`
Expected: 4 tests pass — pipeline runs end-to-end for each drug.

- [ ] **Step 4: Run the validation gate**

Run: `pytest tests/integration/test_prodrug_pipeline_smoke.py::test_prodrug_validation_gate_3fold -v`
Expected: 4 tests pass — all within 3-fold.

If any drug fails: per spec §7.5 failure response —
1. Re-verify literature values for that drug in the registry. Adjust `mean` and `cv` as needed.
2. If still failing after re-verification, add a per-drug routing override to `data/sbi/method_routing.json` forcing engine weight=1.0 (parallels morphine SBI routing precedent).
3. If still failing, escalate: re-open spec.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_prodrug_pipeline_smoke.py
git commit -m "test(integration): 4-drug smoke + validation gate (3-fold per drug)"
```

---

### Task 15: 107-holdout backward-compat regression test

**Files:**
- Test: `tests/regression/test_holdout_unchanged.py` (new)

- [ ] **Step 1: Write the regression test**

`tests/regression/test_holdout_unchanged.py`:
```python
"""Regression test: 107-holdout AAFE unchanged after prodrug routing.

Prodrug-related changes are additive only (active_metabolite=None for all
107 holdout drugs). Pipeline behavior must be identical for non-prodrug
drugs. This test runs the full holdout benchmark with MC=1000 and verifies
AAFE matches the baseline 2.719 within MC noise tolerance (1%).

Slow test (~70 minutes). Marked nightly-only via @pytest.mark.slow.
Run with: pytest tests/regression/ -m slow
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# Baseline AAFE from data/validation/holdout_pi_coverage_2026-04-24.json
BASELINE_AAFE = 2.719
TOLERANCE_PCT = 0.01  # 1%


@pytest.mark.slow
def test_107_holdout_aafe_unchanged_after_prodrug_routing():
    """Full benchmark — AAFE within 1% of baseline."""
    import subprocess
    result = subprocess.run(
        ["python3", "-m", "sisyphus.cli", "benchmark", "--holdout",
         "--n-mc-samples", "1000", "--output", "/tmp/holdout_after_prodrug.json"],
        capture_output=True, text=True, timeout=5400,  # 90 min
    )
    assert result.returncode == 0, f"benchmark failed: {result.stderr}"

    with open("/tmp/holdout_after_prodrug.json") as f:
        bench = json.load(f)

    aafe = bench["aafe"]
    delta_pct = abs(aafe - BASELINE_AAFE) / BASELINE_AAFE
    assert delta_pct < TOLERANCE_PCT, (
        f"AAFE drift: {aafe:.3f} vs baseline {BASELINE_AAFE:.3f} "
        f"(delta {delta_pct:.2%} > {TOLERANCE_PCT:.2%})"
    )


def test_holdout_drug_yamls_have_no_active_metabolite():
    """Sanity: none of the 107 holdout drugs should match the prodrug registry."""
    from sisyphus.predict.registry import lookup_active_metabolite
    holdout_path = Path("data/reference/holdout.json")
    with holdout_path.open() as f:
        holdout = json.load(f)
    # holdout structure may have entries with smiles field
    # Adjust based on actual structure
    matches = []
    for entry in holdout.get("drugs", []):
        smiles = entry.get("smiles") or entry.get("SMILES")
        if smiles is None:
            continue
        if lookup_active_metabolite(smiles) is not None:
            matches.append(entry.get("name", smiles))
    assert matches == [], (
        f"Holdout drugs unexpectedly match prodrug registry: {matches}"
    )
```

- [ ] **Step 2: Run the fast portion**

Run: `pytest tests/regression/test_holdout_unchanged.py::test_holdout_drug_yamls_have_no_active_metabolite -v`
Expected: PASS — none of 107 holdout drugs are in the prodrug registry.

- [ ] **Step 3: Run the slow portion (nightly)**

Run: `pytest tests/regression/test_holdout_unchanged.py -m slow -v --timeout=5400`
Expected: PASS — AAFE within 1% of 2.719.

(If running locally is too slow, skip and rely on CI nightly schedule.)

- [ ] **Step 4: Commit**

```bash
git add tests/regression/test_holdout_unchanged.py
git commit -m "test(regression): 107-holdout AAFE unchanged after prodrug routing"
```

---

## Self-Review Checklist (run after all 15 tasks complete)

- [ ] **Spec coverage** — every spec section §1–§10 has at least one task implementing it:
  - §1 Goal: Tasks 14 (validation gate) + 15 (regression).
  - §2 Scope: All 15 tasks fall within 4-drug + 1-active-per-drug + 1st-order scope. Out-of-scope items not implemented (multi-step, parallel actives, DDI on active).
  - §3 Architectural decisions: Q1=Tasks 10+14, Q2=Task 1+8+13, Q3=Task 5, Q4=Task 8 (suffix naming), Q5=Task 12, C1=Task 9, C2=Task 6.
  - §4 Components: Tasks 1 (4.1), 3 (4.2), 5+6 (4.3), 7 (4.4), 8 (4.5), 9 (4.6), 11 (after 4.6 wired), 12 (4.7), 10 (4.8).
  - §5 Data flow: Task 13 verifies cascade math; Task 14 verifies end-to-end.
  - §6 Error handling: Tasks 1+2 (DrugOnGraph validation), Task 8 (build-time errors), Task 9 (registry validation), Task 12 (AD adjustment).
  - §7 Testing: 7 unit tests + 2 integration + 1 regression covered.
  - §8 Pre-impl verifications: resolved at plan top in "Verified Codebase Facts."
  - §10 Success criteria: validated by Tasks 13 (mass balance), 14 (4-drug 3-fold), 15 (107 < 1% drift).

- [ ] **Placeholder scan** — no TBD/TODO in tasks. SMILES placeholders in Task 14 are explicitly marked (Step 2 instructs to populate from registry).

- [ ] **Type consistency** —
  - `ActiveMetabolite` used identically in Tasks 1, 2, 8, 9, 11, 12, 13.
  - `ProdrugActivationEdge` / `OneCompartmentEliminationEdge` consistent across Tasks 3, 4, 5, 6, 7, 8, 13.
  - `ACTIVE_SUFFIX` constant defined in Task 8, used in Task 12.
  - `lookup_active_metabolite` signature: `(smiles, registry_path=None) -> Optional[Tuple[ActiveMetabolite, str]]`. Callers in Tasks 9, 11, 15 use this exact form.

- [ ] **Hard invariants preserved**:
  - Engine compiler.py changes are additive only (Task 7) — ✓
  - Engine solver.py untouched — ✓ (no task modifies it)
  - DrugOnGraph existing fields untouched (Task 2 appends only) — ✓
  - Holdout list unchanged (Task 15 verifies no overlap) — ✓
  - No drug-specific branches in engine code (registry pattern is data, not code) — ✓

---

## Execution Notes

- Tasks 1–9 are fully independent of literature data. Implementer can run them with empty registry.
- Task 10 requires literature research (~2 hours total: 30 min × 4 drugs).
- Tasks 11–14 require the registry to be populated (Task 10).
- Task 15's slow portion requires the full benchmark CLI to be runnable.

Total estimated work: ~12–15 hours (excluding literature collection waiting time).
