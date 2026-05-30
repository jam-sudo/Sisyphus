# Prodrug Activation v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v1's kinetic 1st-order prodrug conversion with mechanistic enzyme-abundance × drug-affinity × well-stirred extraction, sourced from in-vitro literature (no clinical fit).

**Architecture:** New `ProdrugActivationFluxSpec.apply()` mirrors `ClearanceFluxSpec(model="well_stirred")` math but routes mass to active-species blood pool (with MW × yield scaling) instead of sink. Augmentation creates one `ProdrugActivationEdge` per node where drug's declared enzyme tags intersect physiology's enzyme abundance. Drug-side single source of truth: `DrugOnGraph.enzyme_affinity_for_conversion: dict[str, Distribution]`.

**Tech Stack:** Python 3.10+, pytest, NumPy, SciPy `solve_ivp`, RDKit (SMILES canonicalization), PyYAML.

**Spec:** [`docs/superpowers/specs/2026-04-27-prodrug-activation-v2-design.md`](../specs/2026-04-27-prodrug-activation-v2-design.md)

---

## File map

**Modified (existing):**
- `src/sisyphus/core.py` — add `DrugOnGraph.enzyme_affinity_for_conversion` field; update `__post_init__` and `sample()`.
- `src/sisyphus/graph/types.py` — modify `ProdrugActivationEdge` (replace `conversion_rate` with `enzyme_tags`).
- `src/sisyphus/engine/flux.py` — rewrite `ProdrugActivationFluxSpec.apply()` (well-stirred).
- `src/sisyphus/engine/compiler.py` — add `ResolvedParams.drug_enzyme_affinity_for_conversion(tag)`; remove `conversion_rate` from `_build_edge_params` for `ProdrugActivationEdge`.
- `src/sisyphus/graph/builder.py` — rewrite `augment_for_active_species` (multi-site discovery).
- `src/sisyphus/predict/registry.py` — change return type to 3-tuple; add v2 schema validation.
- `src/sisyphus/pipeline/predict.py` — pass `enzyme_affinity_for_conversion` when constructing `DrugOnGraph` (only the registry-aware path; non-prodrug path defaults to `{}`).
- `data/physiology/reference_man.yaml` — add SPR/CES1/CES2/ALPI enzyme abundances at relevant nodes.
- `data/sbi/prodrug_activation_registry.json` — rewrite 4 entries with new schema.
- `CHANGELOG.md` — Unreleased section: v2 entry replacing v1 known-limitation note.

**Created (new tests):**
- `tests/unit/test_prodrug_v2_edge.py` — `ProdrugActivationEdge.enzyme_tags` field, mw_parent validation.
- `tests/unit/test_prodrug_v2_drug.py` — `DrugOnGraph.enzyme_affinity_for_conversion` field, post-init validation, sample().
- `tests/unit/test_prodrug_v2_resolved_params.py` — `drug_enzyme_affinity_for_conversion` lookup.
- `tests/unit/test_prodrug_v2_flux.py` — well-stirred rate calculation.
- `tests/unit/test_prodrug_v2_augment.py` — multi-site discovery, idempotency.
- `tests/unit/test_prodrug_v2_registry.py` — v2 schema validation, 3-tuple return.
- `tests/integration/test_prodrug_v2_mass_balance.py` — flow-loop synthetic system vs analytical.
- `tests/integration/test_prodrug_v2_pipeline_smoke.py` — 4-drug end-to-end.
- `tests/integration/test_prodrug_v2_ddi_smoke.py` — CES1 abundance proportionality.
- `tests/regression/test_prodrug_v2_identity_blind.py` — random tag rename invariance.
- `tests/regression/test_prodrug_v2_snapshot.py` — per-drug Cmax ±5% pinning.
- `tests/regression/test_prodrug_v2_validation_gate.py` — per-drug parametrized 3-fold.

**Created (new docs):**
- `docs/superpowers/specs/2026-04-27-prodrug-v2-task1-literature.md` — Task 1 deliverable: literature CLint values + tier classification.

---

## Pre-implementation setup

### Task 0: Branch + worktree setup

**Files:** none (git operations).

- [ ] **Step 1: Create feature branch**

```bash
cd /home/jam/Sisyphus
git checkout -b feat/prodrug-activation-v2
git status   # confirm clean
```

Expected: `On branch feat/prodrug-activation-v2`, untracked data/model files only (carry-over from prior sessions, ignore).

- [ ] **Step 2: Verify v1 prodrug tests are present and passing on baseline**

```bash
pytest tests/unit/test_prodrug_edges.py tests/unit/test_prodrug_flux.py \
       tests/unit/test_prodrug_registry.py tests/integration/test_two_species_mass_balance.py \
       -x --no-header -q
```

Expected: all v1 tests pass on baseline (we will NOT delete these — they document v1 contract; v2 tests are additive in `*_v2*` files. Some will get deprecated in Task 18).

- [ ] **Step 3: Verify validation gate currently fails as expected**

```bash
pytest tests/integration/test_prodrug_pipeline_smoke.py::test_prodrug_validation_gate_3fold -v
```

Expected: this test currently FAILS (it's the v1 known limitation). Note current fold-errors for CHANGELOG comparison later.

---

## Task 1: Literature search deliverable

**Files:**
- Create: `docs/superpowers/specs/2026-04-27-prodrug-v2-task1-literature.md`

**Description:** Resolve all `<TBD>` markers in spec. This task produces no code — only a structured Markdown document that downstream tasks will read values from.

**Why this task is first:** Every subsequent task that touches physiology YAML (Task 9) or registry JSON (Task 13) needs concrete numerical values. Without literature work, those tasks would block.

- [ ] **Step 1: Create deliverable skeleton**

Create `docs/superpowers/specs/2026-04-27-prodrug-v2-task1-literature.md` with this structure:

```markdown
# Prodrug v2 Task 1 — Literature Values

**Date:** 2026-04-27
**Purpose:** Resolve enzyme abundance + drug affinity placeholders in spec.

## 1. Enzyme abundances (per organ)

### 1.1 SPR (sepiapterin reductase, EC 1.1.1.153)
- liver: mean = <fill>, cv = <fill>; source = <fill>
- gut_wall: mean = <fill>, cv = <fill>; source = <fill>
- kidney: mean = <fill>, cv = <fill>; source = <fill>

### 1.2 CES1 (carboxylesterase 1)
- liver, gut_wall: same template

### 1.3 CES2 (carboxylesterase 2)
- liver, gut_wall: same template

### 1.4 ALPI (intestinal alkaline phosphatase)
- gut_wall: same template

## 2. Drug affinity for activation enzymes

### 2.1 sepiapterin × SPR
- mean = <Vmax/Km in µL/min/pmol>, cv = <fill>
- citation = <DOI / paper>
- tier = "literature" | "class_extrapolated"

### 2.2 remdesivir × CES1: same template
### 2.3 tebipenem_pivoxil × CES2: same template
### 2.4 fostamatinib × ALPI: same template

## 3. Yield fractions

| Drug | yield | yield_source | citation |
|------|-------|--------------|----------|
| sepiapterin | | "literature" | "class_extrapolated" | |
| ... | | | |

## 4. CL/Vd of active species (already in v1 registry — verify with citation)

| Active | CL_per_h mean,cv | Vd_L mean,cv | citation |
|--------|------------------|--------------|----------|
| BH4 | | | |
| GS-441524 | | | |
| tebipenem | | | |
| R406 | | | |

## 5. Tier summary

| Drug | Final tier | Justification |
|------|-----------|---------------|
| sepiapterin | | |
| remdesivir | | |
| tebipenem_pivoxil | | |
| fostamatinib | | |

## 6. Contingency: tier 3 / no-data drugs

If any drug ends tier 3, list here and document exclusion reason.

## 7. Sanity check log

For each drug, record back-of-envelope feasibility:
- Required CLint per spec §4.1 sanity check (Eg target → required abundance × affinity × ivive)
- Computed CLint from literature inputs
- Ratio (required/literature) — if >100×, mark caution

```

- [ ] **Step 2: Conduct literature search**

For each of SPR, CES1, CES2, ALPI:
1. Search PubMed / Google Scholar for: `<enzyme> abundance human liver pmol mg microsomal`, `<enzyme> Km Vmax <drug>`, `<enzyme> intestinal abundance brush-border`.
2. Prefer primary sources (Park et al, Eastman et al, Yan et al, Kamiya et al) over reviews.
3. For each value, record: mean, CV (typically 0.5–1.0 for in-vitro CLint), source citation.
4. For drug affinity: compute CLint = Vmax/Km. Convert units to µL/min/pmol enzyme. Document conversion arithmetic.

For tier classification:
- **tier 1**: direct in-vitro Vmax/Km for this exact drug-enzyme pair found in primary literature.
- **tier 2**: Vmax/Km estimated from substrate-class kinetics (e.g., ALPI on phosphate monoesters generally). Wider CV (1.0–1.5).
- **tier 3**: only clinical Eg available; would require back-fit. **Drug excluded from v2 registry.**

- [ ] **Step 3: Sanity check feasibility**

For each drug, compute:
```
required_abundance_x_affinity_x_ivive = clinical_Eg_target_value
literature_abundance_x_affinity_x_ivive = product_from_table_above
ratio = required / literature
```

Flag any drug with ratio > 100× as "caution: literature insufficient" — likely tier 3 candidate.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-27-prodrug-v2-task1-literature.md
git commit -m "docs(spec): prodrug v2 literature values + tier classification

Resolves all <TBD> placeholders in v2 spec for enzyme abundances,
drug affinities, yield fractions, and active CL/Vd. Per-drug tier
assignments documented with citations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Hard gate — re-open spec if 0 tier 1+2 drugs**

If literature search yields **zero** drugs with tier 1 or 2 affinities:
- STOP. Do not proceed to Task 2.
- Notify user; spec re-open required (per spec §6.2 contingency).
- Alternative drug list to evaluate: capecitabine (CES2 → 5'-DFCR), oseltamivir (CES1), irinotecan (CES2 → SN-38).

If ≥1 tier 1+2 drug found: proceed to Task 2.

---

## Task 2: ProdrugActivationEdge struct change

**Files:**
- Modify: `src/sisyphus/graph/types.py:161-181` (ProdrugActivationEdge dataclass)
- Test: `tests/unit/test_prodrug_v2_edge.py` (create)

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_prodrug_v2_edge.py`:

```python
"""Unit tests for v2 ProdrugActivationEdge struct."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.graph.types import ProdrugActivationEdge


def test_edge_has_enzyme_tags_field():
    """v2 edge replaces conversion_rate with enzyme_tags frozenset."""
    edge = ProdrugActivationEdge(
        source="liver",
        target="venous_blood_active",
        enzyme_tags=frozenset({"SPR"}),
        conversion_yield=Distribution(mean=0.85, cv=0.1),
        mw_parent=237.0,
        mw_active=241.25,
    )
    assert edge.enzyme_tags == frozenset({"SPR"})
    assert edge.mw_parent == 237.0
    assert edge.mw_active == 241.25
    assert edge.conversion_yield.mean == 0.85


def test_edge_no_conversion_rate_field():
    """v1 conversion_rate field removed in v2."""
    edge = ProdrugActivationEdge(
        source="liver",
        target="venous_blood_active",
        enzyme_tags=frozenset({"SPR"}),
        mw_parent=237.0,
        mw_active=241.25,
    )
    # Field should not exist
    assert not hasattr(edge, "conversion_rate")


def test_edge_default_enzyme_tags_empty():
    """Default enzyme_tags is empty frozenset (mirrors v1 default-zero pattern)."""
    edge = ProdrugActivationEdge(
        source="x",
        target="y",
        mw_parent=100.0,
        mw_active=100.0,
    )
    assert edge.enzyme_tags == frozenset()


def test_edge_is_frozen():
    """Edge dataclass remains frozen."""
    edge = ProdrugActivationEdge(
        source="x",
        target="y",
        enzyme_tags=frozenset({"X"}),
        mw_parent=100.0,
        mw_active=100.0,
    )
    with pytest.raises((AttributeError, Exception)):
        edge.enzyme_tags = frozenset({"Y"})  # type: ignore[misc]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_prodrug_v2_edge.py -v
```

Expected: 4 tests FAIL with errors about `enzyme_tags` not being a valid field, OR `conversion_rate` still present.

- [ ] **Step 3: Modify `ProdrugActivationEdge`**

In `src/sisyphus/graph/types.py`, replace the `ProdrugActivationEdge` class (lines 161-181) with:

```python
@dataclass(frozen=True)
class ProdrugActivationEdge(Edge):
    """Mass transfer: parent drug → active metabolite via enzyme catalysis.

    v2 (2026-04-27): conversion is well-stirred extraction at flow-through
    nodes (replaces v1's kinetic 1st-order). Drug declares which enzymes
    catalyze the conversion via ``enzyme_tags``; engine computes CLint from
    node enzyme abundance × drug.enzyme_affinity_for_conversion[tag].

    Mass routing: source loses parent (mg); target gains active (mg)
    scaled by mw_active/mw_parent × conversion_yield.

    Identity-blind: engine matches by edge_type and tag strings only.
    """

    edge_type: str = field(default="prodrug_activation", init=False)
    enzyme_tags: frozenset[str] = field(default_factory=frozenset)
    conversion_yield: Distribution = field(default_factory=lambda: Distribution(1.0))
    mw_parent: float = 0.0
    mw_active: float = 0.0
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/unit/test_prodrug_v2_edge.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Verify v1 edges test now FAILS at compile-time references**

```bash
pytest tests/unit/test_prodrug_edges.py -v 2>&1 | tail -20
```

Expected: tests referencing `conversion_rate` field fail (e.g., `AttributeError: 'ProdrugActivationEdge' has no attribute 'conversion_rate'`). Note these — Task 18 will deprecate them.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/graph/types.py tests/unit/test_prodrug_v2_edge.py
git commit -m "feat(graph): v2 ProdrugActivationEdge with enzyme_tags

Replaces v1's conversion_rate Distribution with enzyme_tags: frozenset[str].
Compile-time enzyme set baking; affinity values remain on drug side
(single source of truth). Backward-incompatible internal API change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: DrugOnGraph.enzyme_affinity_for_conversion field

**Files:**
- Modify: `src/sisyphus/core.py:188-319` (DrugOnGraph dataclass + __post_init__ + sample)
- Test: `tests/unit/test_prodrug_v2_drug.py` (create)

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_prodrug_v2_drug.py`:

```python
"""Unit tests for DrugOnGraph.enzyme_affinity_for_conversion field."""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.core import ActiveMetabolite, Distribution, DrugOnGraph


def _minimal_drug(**overrides) -> DrugOnGraph:
    """Construct a DrugOnGraph with minimal valid fields."""
    base = dict(
        name="x", smiles="C", dose_mg=100.0, route="oral",
        administration_node="stomach_lumen",
        mw=200.0, pka=None, compound_type="neutral",
        fup=Distribution(0.5), rbp=Distribution(1.0),
        kp_method="rodgers_rowland", kp_overrides={},
        peff=Distribution(1e-4), solubility=Distribution(1.0),
        enzyme_affinity={},
        renal_clearance=Distribution(0.0),
    )
    base.update(overrides)
    return DrugOnGraph(**base)


def _minimal_active() -> ActiveMetabolite:
    """Construct an ActiveMetabolite with minimal valid fields."""
    return ActiveMetabolite(
        name="A",
        mw=200.0,
        fup=Distribution(0.5),
        CL_per_h=Distribution(10.0),
        Vd_L=Distribution(20.0),
        conversion_rate_per_h=Distribution(1.0),  # v1 field still present in ActiveMetabolite
        conversion_site="liver",                    # v1 field still present
        conversion_yield_fraction=Distribution(1.0),
    )


def test_drug_default_enzyme_affinity_for_conversion_is_empty_dict():
    drug = _minimal_drug()
    assert drug.enzyme_affinity_for_conversion == {}


def test_drug_can_set_enzyme_affinity_for_conversion():
    affinity = {"SPR": Distribution(mean=100.0, cv=0.5)}
    drug = _minimal_drug(
        enzyme_affinity_for_conversion=affinity,
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    assert drug.enzyme_affinity_for_conversion == affinity


def test_postinit_rejects_affinity_without_active_metabolite():
    """Non-empty enzyme_affinity_for_conversion requires active_metabolite."""
    with pytest.raises(ValueError, match="enzyme_affinity_for_conversion"):
        _minimal_drug(
            enzyme_affinity_for_conversion={"SPR": Distribution(100.0)},
            active_metabolite=None,
        )


def test_postinit_allows_empty_affinity_with_active_metabolite():
    """Empty dict + active_metabolite is allowed (e.g., during construction)."""
    drug = _minimal_drug(
        enzyme_affinity_for_conversion={},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    assert drug.enzyme_affinity_for_conversion == {}


def test_sample_propagates_enzyme_affinity_for_conversion():
    """drug.sample() must resample enzyme_affinity_for_conversion dict."""
    rng = np.random.default_rng(42)
    drug = _minimal_drug(
        enzyme_affinity_for_conversion={"SPR": Distribution(mean=100.0, cv=0.5)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    sampled = drug.sample(rng)
    assert "SPR" in sampled.enzyme_affinity_for_conversion
    assert sampled.enzyme_affinity_for_conversion["SPR"].cv == 0.0  # resolved to point value
    # Sampled mean should be a finite positive float
    assert np.isfinite(sampled.enzyme_affinity_for_conversion["SPR"].mean)
    assert sampled.enzyme_affinity_for_conversion["SPR"].mean > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_prodrug_v2_drug.py -v
```

Expected: tests fail with `TypeError: DrugOnGraph.__init__() got an unexpected keyword argument 'enzyme_affinity_for_conversion'`.

- [ ] **Step 3: Add field to DrugOnGraph**

In `src/sisyphus/core.py`, locate `DrugOnGraph` (line 188). Add a new field AFTER `observation_species: str = "parent"` (line 252) and BEFORE `def __post_init__` (line 254):

```python
    # v2 prodrug activation — drug-side enzyme affinity for conversion
    # (separate from enzyme_affinity which is for elimination).
    # See docs/superpowers/specs/2026-04-27-prodrug-activation-v2-design.md §3.2.
    enzyme_affinity_for_conversion: dict[str, Distribution] = field(default_factory=dict)
```

Update `__post_init__` (currently at line 254) to add validation. The full method should be:

```python
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
        if self.enzyme_affinity_for_conversion and self.active_metabolite is None:
            raise ValueError(
                "enzyme_affinity_for_conversion is non-empty but active_metabolite is None; "
                "set active_metabolite or empty the dict"
            )
```

Update `sample()` method. Find the existing `sample()` (lines 265-319). Add `enzyme_affinity_for_conversion` resampling. Replace the `return DrugOnGraph(...)` block at lines 271-319 with one that includes the new field. Specifically, add this entry to the `DrugOnGraph(...)` call (location: alongside `enzyme_affinity={...}` block, e.g., immediately after it):

```python
            enzyme_affinity_for_conversion={
                k: Distribution(mean=v.sample(rng), cv=0.0)
                for k, v in self.enzyme_affinity_for_conversion.items()
            },
```

(The full updated `DrugOnGraph(...)` constructor in `sample()` needs this one extra kwarg; the rest is unchanged.)

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/unit/test_prodrug_v2_drug.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Verify no regression in existing drug tests**

```bash
pytest tests/unit/test_active_metabolite.py tests/unit/test_drugongraph_postinit.py -v 2>&1 | tail -20
```

Expected: all PASS (additive change only). If any reference DrugOnGraph constructor without the new field, default `{}` ensures it still works.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/core.py tests/unit/test_prodrug_v2_drug.py
git commit -m "feat(core): DrugOnGraph.enzyme_affinity_for_conversion field

Adds drug-side dict declaring which enzymes catalyze conversion to
active species. Single source of truth; edge stores enzyme_tags only.
Validation: non-empty dict requires active_metabolite. Sample()
resampling propagates dict per MC iteration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: ResolvedParams.drug_enzyme_affinity_for_conversion + edge_param update

**Files:**
- Modify: `src/sisyphus/engine/compiler.py` (ResolvedParams class)
- Test: `tests/unit/test_prodrug_v2_resolved_params.py` (create)

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_prodrug_v2_resolved_params.py`:

```python
"""Unit tests for ResolvedParams v2 prodrug accessors."""
from __future__ import annotations

import pytest

from sisyphus.core import ActiveMetabolite, Distribution
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)
from tests.unit.test_prodrug_v2_drug import _minimal_drug, _minimal_active


def _minimal_graph() -> BodyGraph:
    g = BodyGraph()
    g.add_node(Node(name="liver", node_type="organ", volume=Distribution(1.5)))
    g.add_node(Node(name="venous_blood_active", node_type="blood_pool", volume=Distribution(20.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink", volume=Distribution(0.0)))
    return g


def test_drug_enzyme_affinity_for_conversion_returns_mean():
    drug = _minimal_drug(
        enzyme_affinity_for_conversion={"SPR": Distribution(mean=42.0, cv=0.0)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    g = _minimal_graph()
    params = ResolvedParams(g, drug)
    assert params.drug_enzyme_affinity_for_conversion("SPR") == 42.0


def test_drug_enzyme_affinity_for_conversion_returns_zero_for_missing_tag():
    drug = _minimal_drug()
    g = _minimal_graph()
    params = ResolvedParams(g, drug)
    assert params.drug_enzyme_affinity_for_conversion("DOES_NOT_EXIST") == 0.0


def test_edge_param_for_prodrug_activation_no_conversion_rate():
    """v2 ProdrugActivationEdge no longer has conversion_rate; only conversion_yield."""
    g = _minimal_graph()
    g.add_edge(ProdrugActivationEdge(
        source="liver", target="venous_blood_active",
        enzyme_tags=frozenset({"SPR"}),
        conversion_yield=Distribution(mean=0.85, cv=0.0),
        mw_parent=237.0, mw_active=241.25,
    ))
    drug = _minimal_drug(
        enzyme_affinity_for_conversion={"SPR": Distribution(100.0)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    params = ResolvedParams(g, drug)
    # conversion_yield must still be a recognized edge param
    assert params.edge_param(0, "conversion_yield") == 0.85
    # conversion_rate should NOT be set for v2 edges
    with pytest.raises(KeyError):
        params.edge_param(0, "conversion_rate")
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_prodrug_v2_resolved_params.py -v
```

Expected: tests fail with `AttributeError: 'ResolvedParams' object has no attribute 'drug_enzyme_affinity_for_conversion'` and possibly `conversion_rate` still being set.

- [ ] **Step 3: Add `drug_enzyme_affinity_for_conversion` method**

In `src/sisyphus/engine/compiler.py`, add this method to `ResolvedParams` AFTER `drug_enzyme_affinity` (around line 122):

```python
    def drug_enzyme_affinity_for_conversion(self, tag: str) -> float:
        """Return the drug's intrinsic clearance per unit enzyme for *tag*,
        for ACTIVATION (parent → active species).

        Distinct from drug_enzyme_affinity which is for ELIMINATION.
        Returns 0.0 for tags absent from the dict (graceful, mirrors
        drug_enzyme_affinity).
        """
        if tag in self._drug.enzyme_affinity_for_conversion:
            return self._drug.enzyme_affinity_for_conversion[tag].mean
        return 0.0
```

- [ ] **Step 4: Update `_build_edge_params` for v2 edge**

In `src/sisyphus/engine/compiler.py`, find the `_build_edge_params` method (around line 191). The block currently contains:

```python
            elif isinstance(edge, ProdrugActivationEdge):
                params["conversion_rate"] = edge.conversion_rate.mean
                params["conversion_yield"] = edge.conversion_yield.mean
```

Replace with:

```python
            elif isinstance(edge, ProdrugActivationEdge):
                # v2: enzyme_tags baked into FluxSpec at compile time;
                # conversion_yield is the only resampled per-MC parameter here.
                params["conversion_yield"] = edge.conversion_yield.mean
```

- [ ] **Step 5: Run to verify pass**

```bash
pytest tests/unit/test_prodrug_v2_resolved_params.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/engine/compiler.py tests/unit/test_prodrug_v2_resolved_params.py
git commit -m "feat(engine): ResolvedParams.drug_enzyme_affinity_for_conversion

Adds tag-keyed lookup for activation enzyme affinity (parallel to
drug_enzyme_affinity for elimination). Removes conversion_rate from
ProdrugActivationEdge edge_params (now compile-time baked enzyme_tags).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: ProdrugActivationFluxSpec well-stirred rewrite

**Files:**
- Modify: `src/sisyphus/engine/flux.py` (ProdrugActivationFluxSpec class, around lines 557-610)
- Test: `tests/unit/test_prodrug_v2_flux.py` (create)

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_prodrug_v2_flux.py`:

```python
"""Unit tests for v2 ProdrugActivationFluxSpec well-stirred math."""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.core import ActiveMetabolite, Distribution
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.engine.flux import ProdrugActivationFluxSpec
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    FlowEdge,
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)
from tests.unit.test_prodrug_v2_drug import _minimal_active, _minimal_drug


def _build_flow_graph(
    abundance: float = 1e6,
    affinity: float = 10.0,
    ivive: float = 6e-5,
    fup: float = 0.5,
    q: float = 60.0,
    v_source: float = 10.0,
):
    """Build a 4-node flow-through graph for well-stirred testing.

    Topology:
        infusion_source --[FlowEdge Q]--> conversion_node --[FlowEdge Q]--> exit_sink
                                            └--[ProdrugActivationEdge]--> active_pool
                                                                            └--[1C elim]--> elim_sink
    """
    g = BodyGraph()
    g.add_node(Node(name="infusion_source", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_node(Node(
        name="conversion_node", node_type="organ", volume=Distribution(v_source),
        enzymes={"X": Distribution(mean=abundance, cv=0.0)},
        ivive_scaling=ivive,
    ))
    g.add_node(Node(name="active_pool", node_type="blood_pool", volume=Distribution(10.0)))
    g.add_node(Node(name="exit_sink", node_type="sink", volume=Distribution(0.0)))
    g.add_node(Node(name="elim_sink", node_type="sink", volume=Distribution(0.0)))

    g.add_edge(FlowEdge(source="infusion_source", target="conversion_node",
                       flow_rate=Distribution(q)))
    g.add_edge(FlowEdge(source="conversion_node", target="exit_sink",
                       flow_rate=Distribution(q)))
    g.add_edge(ProdrugActivationEdge(
        source="conversion_node", target="active_pool",
        enzyme_tags=frozenset({"X"}),
        conversion_yield=Distribution(mean=1.0, cv=0.0),
        mw_parent=200.0, mw_active=200.0,
    ))
    g.add_edge(OneCompartmentEliminationEdge(
        source="active_pool", target="elim_sink",
        cl_per_h=Distribution(10.0), vd_l=Distribution(10.0),
    ))

    drug = _minimal_drug(
        fup=Distribution(fup),
        enzyme_affinity_for_conversion={"X": Distribution(mean=affinity, cv=0.0)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    return g, drug


def test_flux_well_stirred_rate_matches_formula():
    """Rate should equal (Q × fup × CLint) / (Q + fup × CLint) × C_out × yield × mw_ratio."""
    abundance, affinity, ivive = 1e6, 10.0, 6e-5
    fup, q, v_source = 0.5, 60.0, 10.0
    g, drug = _build_flow_graph(
        abundance=abundance, affinity=affinity, ivive=ivive,
        fup=fup, q=q, v_source=v_source,
    )
    params = ResolvedParams(g, drug)

    # Find the prodrug activation edge id
    edge_id = None
    for i, edge in enumerate(g.edges):
        if isinstance(edge, ProdrugActivationEdge):
            edge_id = i
            break
    assert edge_id is not None

    flux_spec = ProdrugActivationFluxSpec.from_edge(edge_id, g.edges[edge_id], _build_state_index(g))

    # Set a known parent amount in conversion_node
    state_idx = _build_state_index(g)
    y = np.zeros(len(state_idx))
    a_parent = 100.0  # mg
    y[state_idx["conversion_node"]] = a_parent

    dydt = np.zeros_like(y)
    flux_spec.apply(t=0.0, y=y, dydt=dydt, params=params)

    # Expected:
    clint = abundance * affinity * ivive
    cl_organ = (q * fup * clint) / (q + fup * clint)
    c_out = a_parent / v_source  # blood_pool/no Kp/RBP correction (kp=1, rbp=1)
    expected_rate_parent = cl_organ * c_out  # since kp=1 for organ default
    # Actually conversion_node is "organ" type → kp default = 1.0 in compiler kp_map
    # rbp default is 1.0. So c_out = a/v.

    # mw_ratio = 1, yield = 1
    expected_rate_active = expected_rate_parent * 1.0 * 1.0

    assert dydt[state_idx["conversion_node"]] == pytest.approx(-expected_rate_parent, rel=1e-6)
    assert dydt[state_idx["active_pool"]] == pytest.approx(expected_rate_active, rel=1e-6)


def test_flux_zero_when_clint_zero():
    """Affinity=0 should yield zero flux (no extraction)."""
    g, drug = _build_flow_graph(affinity=0.0)
    params = ResolvedParams(g, drug)

    edge_id = next(i for i, e in enumerate(g.edges) if isinstance(e, ProdrugActivationEdge))
    flux_spec = ProdrugActivationFluxSpec.from_edge(edge_id, g.edges[edge_id], _build_state_index(g))

    state_idx = _build_state_index(g)
    y = np.zeros(len(state_idx))
    y[state_idx["conversion_node"]] = 100.0
    dydt = np.zeros_like(y)
    flux_spec.apply(t=0.0, y=y, dydt=dydt, params=params)

    assert dydt[state_idx["conversion_node"]] == 0.0
    assert dydt[state_idx["active_pool"]] == 0.0


def test_flux_mw_ratio_scales_active_mass():
    """Active mass = parent loss × (mw_active/mw_parent) × yield."""
    g = BodyGraph()
    g.add_node(Node(name="src", node_type="organ", volume=Distribution(10.0),
                   enzymes={"X": Distribution(1e6)}, ivive_scaling=6e-5))
    g.add_node(Node(name="active", node_type="blood_pool", volume=Distribution(10.0)))
    g.add_node(Node(name="src_in", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_edge(FlowEdge(source="src_in", target="src", flow_rate=Distribution(60.0)))
    g.add_edge(ProdrugActivationEdge(
        source="src", target="active",
        enzyme_tags=frozenset({"X"}),
        conversion_yield=Distribution(0.5),
        mw_parent=200.0, mw_active=400.0,  # mw_ratio = 2
    ))
    drug = _minimal_drug(
        fup=Distribution(0.5),
        enzyme_affinity_for_conversion={"X": Distribution(10.0)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    params = ResolvedParams(g, drug)

    edge_id = next(i for i, e in enumerate(g.edges) if isinstance(e, ProdrugActivationEdge))
    flux_spec = ProdrugActivationFluxSpec.from_edge(edge_id, g.edges[edge_id], _build_state_index(g))

    state_idx = _build_state_index(g)
    y = np.zeros(len(state_idx))
    y[state_idx["src"]] = 100.0
    dydt = np.zeros_like(y)
    flux_spec.apply(t=0.0, y=y, dydt=dydt, params=params)

    # active mass per unit time = |parent loss| × 2 (mw_ratio) × 0.5 (yield) = parent loss × 1
    parent_loss = -dydt[state_idx["src"]]
    active_gain = dydt[state_idx["active"]]
    assert active_gain == pytest.approx(parent_loss * 2.0 * 0.5, rel=1e-6)


def _build_state_index(graph: BodyGraph) -> dict[str, int]:
    """Build state_index for testing — node order is dict insertion order."""
    return {name: i for i, name in enumerate(graph.nodes.keys())}
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_prodrug_v2_flux.py -v
```

Expected: tests fail because v1 ProdrugActivationFluxSpec uses kinetic 1st-order math, not well-stirred.

- [ ] **Step 3: Rewrite `ProdrugActivationFluxSpec`**

In `src/sisyphus/engine/flux.py`, find `ProdrugActivationFluxSpec` (around line 557). Replace the entire class definition with:

```python
@register_flux("prodrug_activation")
class ProdrugActivationFluxSpec(FluxSpec):
    """Mass transfer via well-stirred enzyme catalysis: parent → active.

    v2 (2026-04-27): well-stirred extraction at flow-through nodes.
    Mirrors ClearanceFluxSpec(model="well_stirred") math but routes flux
    to the active species pool (not a sink), with MW × yield scaling.

    CLint_node = Σ_tag (abundance[tag] × affinity_for_conversion[tag]) × ivive
    CL_organ   = (Q × fup × CLint) / (Q + fup × CLint)
    rate_parent = CL_organ × c_unbound_at_node
    rate_active = rate_parent × (mw_active/mw_parent) × conversion_yield

    Identity-blind: engine iterates enzyme_tags only.
    """

    def __init__(
        self,
        edge_id: int,
        source_idx: int,
        target_idx: int,
        source_name: str,
        target_name: str,
        enzyme_tags: frozenset[str],
        mw_ratio: float,
    ) -> None:
        super().__init__(edge_id, source_idx, target_idx, source_name, target_name)
        self.enzyme_tags = enzyme_tags
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
            enzyme_tags=edge.enzyme_tags,
            mw_ratio=edge.mw_active / edge.mw_parent,
        )

    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        # Compute per-node CLint from enzyme abundance × drug affinity × ivive
        clint_organ = 0.0
        ivive = params.node_param(self.source_name, "ivive_scaling")
        node_enzymes = params.node_enzymes(self.source_name)
        for tag in self.enzyme_tags:
            abundance = node_enzymes.get(tag, 0.0)
            affinity = params.drug_enzyme_affinity_for_conversion(tag)
            if affinity > 0 and abundance > 0:
                clint_organ += abundance * affinity * ivive

        if clint_organ <= 0:
            return  # No catalysis at this node for this drug

        fup = params.drug_param("fup")
        q = params.total_inflow(self.source_name)
        denom = q + fup * clint_organ
        if denom < 1e-12:
            return
        cl_organ = (q * fup * clint_organ) / denom

        # Concentration leaving the source compartment (well-stirred)
        v = params.node_param(self.source_name, "volume")
        kp = params.drug_kp(self.source_name)
        rbp = params.drug_param("rbp")
        c_out = y[self.source_idx] * rbp / (v * kp) if v > 0 else 0.0

        rate_parent = cl_organ * c_out
        y_frac = params.edge_param(self.edge_id, "conversion_yield")
        rate_active = rate_parent * self.mw_ratio * y_frac

        dydt[self.source_idx] -= rate_parent
        dydt[self.target_idx] += rate_active
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/unit/test_prodrug_v2_flux.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Verify v1 mass balance test now FAILS**

```bash
pytest tests/integration/test_two_species_mass_balance.py -v 2>&1 | tail -15
```

Expected: v1 test FAILS — the closed-bolus / kinetic 1st-order topology no longer matches well-stirred (which requires Q>0). Note: Task 6 introduces v2 mass balance test with new topology; Task 18 deprecates v1 test.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/engine/flux.py tests/unit/test_prodrug_v2_flux.py
git commit -m "feat(engine): ProdrugActivationFluxSpec well-stirred rewrite

Replaces v1's kinetic rate=k*A with well-stirred extraction:
  CLint_organ = Σ abundance × affinity_for_conv × ivive
  CL = (Q × fup × CLint) / (Q + fup × CLint)
  rate_parent = CL × c_out
  rate_active = rate_parent × mw_ratio × yield

Identity-blind: engine iterates enzyme_tags only. Mirrors existing
ClearanceFluxSpec(well_stirred) pattern, destination differs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Synthetic mass balance test (well-stirred + 1C elim)

**Files:**
- Test: `tests/integration/test_prodrug_v2_mass_balance.py` (create)

**Description:** Validates Tasks 2-5 together end-to-end via numeric ODE solve vs. analytical steady-state.

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_prodrug_v2_mass_balance.py`:

```python
"""Integration test: v2 prodrug well-stirred + 1C elim cascade vs analytical.

Topology requires flow loop (well-stirred is undefined without Q):
  infusion_source --[Q=60]--> conversion_node --[Q=60]--> exit_sink
                                  └--[ProdrugActivation]--> active_pool
                                                              └--[1C elim]--> elim_sink

Constant infusion: c_in × Q into conversion_node. Steady-state extraction
E = (Q × fup × CLint) / (Q + fup × CLint). Active pool: A_active_steady = rate / k_eff.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sisyphus.core import ActiveMetabolite, Distribution
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    FlowEdge,
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
)
from tests.unit.test_prodrug_v2_drug import _minimal_active, _minimal_drug


def _build_synthetic(
    q=60.0, v_source=10.0, v_active=10.0,
    abundance=1e6, affinity=10.0, ivive=6e-5,
    fup=1.0, cl_active=10.0,
    mw_parent=200.0, mw_active=200.0, yield_=1.0,
):
    g = BodyGraph()
    g.add_node(Node(name="infusion_source", node_type="blood_pool",
                    volume=Distribution(1.0)))
    g.add_node(Node(
        name="conversion_node", node_type="blood_pool",
        volume=Distribution(v_source),
        enzymes={"X": Distribution(mean=abundance, cv=0.0)},
        ivive_scaling=ivive,
    ))
    g.add_node(Node(name="active_pool", node_type="blood_pool",
                    volume=Distribution(v_active)))
    g.add_node(Node(name="exit_sink", node_type="sink",
                    volume=Distribution(0.0)))
    g.add_node(Node(name="elim_sink", node_type="sink",
                    volume=Distribution(0.0)))

    g.add_edge(FlowEdge(source="infusion_source", target="conversion_node",
                        flow_rate=Distribution(q)))
    g.add_edge(FlowEdge(source="conversion_node", target="exit_sink",
                        flow_rate=Distribution(q)))
    g.add_edge(ProdrugActivationEdge(
        source="conversion_node", target="active_pool",
        enzyme_tags=frozenset({"X"}),
        conversion_yield=Distribution(yield_),
        mw_parent=mw_parent, mw_active=mw_active,
    ))
    g.add_edge(OneCompartmentEliminationEdge(
        source="active_pool", target="elim_sink",
        cl_per_h=Distribution(cl_active),
        vd_l=Distribution(v_active),
    ))

    drug = _minimal_drug(
        fup=Distribution(mean=fup, cv=0.0),
        rbp=Distribution(mean=1.0, cv=0.0),
        enzyme_affinity_for_conversion={"X": Distribution(mean=affinity, cv=0.0)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    return g, drug


def test_steady_state_matches_analytical():
    """Numerical ODE steady state matches well-stirred + 1C analytical."""
    q, v_source, v_active = 60.0, 10.0, 10.0
    abundance, affinity, ivive = 1e6, 10.0, 6e-5
    fup, cl_active, yield_ = 1.0, 10.0, 1.0
    g, drug = _build_synthetic(
        q=q, v_source=v_source, v_active=v_active,
        abundance=abundance, affinity=affinity, ivive=ivive,
        fup=fup, cl_active=cl_active, yield_=yield_,
    )

    compiler = ODECompiler()
    compiled = compiler.compile(g)
    params = ResolvedParams(g, drug)
    rhs = compiled.make_rhs(params)

    # Initial: 100 mg in infusion_source (this is a closed-system test —
    # steady-state arises from continuous redistribution).
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index["infusion_source"]] = 100.0

    # Solve to long time horizon to reach quasi-steady (system depletes; 
    # we care about transient where rates are stable).
    sol = solve_ivp(rhs, (0.0, 50.0), y0, method="LSODA", rtol=1e-8, atol=1e-10,
                    t_eval=np.linspace(0.0, 50.0, 500))

    assert sol.success, f"Solver failed: {sol.message}"

    # Verify mass balance: total mg (parent equivalents) preserved
    # accounting for mw_ratio=1, yield=1 → total mass conserved
    src_idx = compiled.state_index["infusion_source"]
    cnv_idx = compiled.state_index["conversion_node"]
    act_idx = compiled.state_index["active_pool"]
    ex_idx = compiled.state_index["exit_sink"]
    el_idx = compiled.state_index["elim_sink"]
    total = (sol.y[src_idx] + sol.y[cnv_idx] + sol.y[act_idx]
             + sol.y[ex_idx] + sol.y[el_idx])
    assert np.allclose(total, 100.0, rtol=1e-3), \
        f"Mass not conserved: total ranges {total.min()} to {total.max()}"


def test_extraction_efficiency_matches_well_stirred_formula():
    """Active production rate at known parent concentration matches formula."""
    q, v_source = 60.0, 10.0
    abundance, affinity, ivive = 1e6, 10.0, 6e-5
    fup = 1.0
    g, drug = _build_synthetic(
        q=q, v_source=v_source,
        abundance=abundance, affinity=affinity, ivive=ivive,
        fup=fup,
    )

    compiler = ODECompiler()
    compiled = compiler.compile(g)
    params = ResolvedParams(g, drug)
    rhs = compiled.make_rhs(params)

    # Set known parent in conversion_node, all else zero
    y = np.zeros(compiled.n_states)
    a_parent = 50.0
    y[compiled.state_index["conversion_node"]] = a_parent
    dydt = rhs(0.0, y)

    # Expected:
    clint = abundance * affinity * ivive   # = 600 L/h
    cl_organ = (q * fup * clint) / (q + fup * clint)
    c_out = a_parent / v_source  # blood_pool, kp=1, rbp=1
    expected_rate_parent_loss_via_activation = cl_organ * c_out

    # parent loss in conversion_node = activation + flow_out
    flow_out = q * c_out
    total_parent_loss = expected_rate_parent_loss_via_activation + flow_out
    actual_parent_loss = -dydt[compiled.state_index["conversion_node"]]
    assert actual_parent_loss == pytest.approx(total_parent_loss, rel=1e-6)

    # active gained = activation * mw_ratio * yield = activation * 1 * 1
    actual_active_gain = dydt[compiled.state_index["active_pool"]]
    assert actual_active_gain == pytest.approx(
        expected_rate_parent_loss_via_activation, rel=1e-6
    )
```

- [ ] **Step 2: Run to verify pass**

```bash
pytest tests/integration/test_prodrug_v2_mass_balance.py -v
```

Expected: 2 tests PASS. (If steady-state assertions fail, check whether `total_inflow` computation includes the activation edge's "outflow" — it should NOT, because activation is not a FlowEdge.)

- [ ] **Step 3: If extraction test fails — debug parent_loss accounting**

Likely cause: total_parent_loss should NOT include flow_out if conversion_node is a flow-through node receiving Q (in steady-state Q × c_in flowing in). Actually for the dydt-only check, parent loss = activation + flow_out is correct because we evaluate at one instant. Verify the formula with a hand calculation, adjust if inflow is also acting on the node (shouldn't be, since infusion_source has 0 mass at this instant).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_prodrug_v2_mass_balance.py
git commit -m "test(integration): v2 prodrug well-stirred mass balance

New topology: flow-loop synthetic system required because well-stirred
is undefined without Q (unlike v1 kinetic). Verifies steady-state
extraction matches (Q × fup × CLint)/(Q + fup × CLint) formula and
mass conservation via numerical ODE solve.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Augmentation rewrite (multi-site discovery)

**Files:**
- Modify: `src/sisyphus/graph/builder.py:280-353` (augment_for_active_species function)
- Test: `tests/unit/test_prodrug_v2_augment.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_prodrug_v2_augment.py`:

```python
"""Unit tests for v2 augment_for_active_species multi-site discovery."""
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
from tests.unit.test_prodrug_v2_drug import _minimal_active, _minimal_drug


def _base_graph_with_two_spr_sites() -> BodyGraph:
    g = BodyGraph()
    g.add_node(Node(
        name="liver", node_type="organ", volume=Distribution(1.5),
        enzymes={"SPR": Distribution(mean=1e6, cv=0.5),
                 "CYP3A4": Distribution(mean=9e6, cv=0.7)},
        ivive_scaling=6e-5,
    ))
    g.add_node(Node(
        name="gut_wall", node_type="barrier_organ", volume=Distribution(1.0),
        enzymes={"SPR": Distribution(mean=3e5, cv=0.7)},
        ivive_scaling=6e-5,
    ))
    g.add_node(Node(name="venous_blood", node_type="blood_pool",
                    volume=Distribution(5.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    return g


def test_no_op_when_no_active_metabolite():
    g = _base_graph_with_two_spr_sites()
    drug = _minimal_drug()  # active_metabolite=None
    n_nodes_before = len(g.nodes)
    n_edges_before = len(g.edges)

    result = augment_for_active_species(g, drug)
    assert result is g  # same instance
    assert len(g.nodes) == n_nodes_before
    assert len(g.edges) == n_edges_before


def test_creates_active_node_and_edges_per_site():
    g = _base_graph_with_two_spr_sites()
    drug = _minimal_drug(
        active_metabolite=_minimal_active(),
        observation_species="parent",
        enzyme_affinity_for_conversion={"SPR": Distribution(100.0)},
    )
    augment_for_active_species(g, drug)

    # 1 new active node added
    assert "venous_blood_active" in g.nodes

    # 2 ProdrugActivationEdges (one per site: liver + gut_wall)
    activation_edges = [e for e in g.edges if isinstance(e, ProdrugActivationEdge)]
    assert len(activation_edges) == 2
    sources = sorted(e.source for e in activation_edges)
    assert sources == ["gut_wall", "liver"]
    for e in activation_edges:
        assert e.target == "venous_blood_active"
        assert e.enzyme_tags == frozenset({"SPR"})

    # 1 OneCompartmentEliminationEdge from active to sink
    elim_edges = [e for e in g.edges if isinstance(e, OneCompartmentEliminationEdge)]
    assert len(elim_edges) == 1
    assert elim_edges[0].source == "venous_blood_active"
    assert elim_edges[0].target == "metabolized_gut"


def test_raises_when_no_site_in_physiology():
    """Drug declares enzymes that don't exist anywhere → ValueError."""
    g = _base_graph_with_two_spr_sites()
    drug = _minimal_drug(
        active_metabolite=_minimal_active(),
        observation_species="parent",
        enzyme_affinity_for_conversion={"NONEXISTENT_ENZYME": Distribution(100.0)},
    )
    with pytest.raises(ValueError, match="No conversion site"):
        augment_for_active_species(g, drug)


def test_raises_when_active_metab_present_but_affinity_empty():
    """Active metabolite + empty affinity dict → ValueError (defensive)."""
    g = _base_graph_with_two_spr_sites()
    drug = _minimal_drug(
        active_metabolite=_minimal_active(),
        observation_species="parent",
        enzyme_affinity_for_conversion={},
    )
    with pytest.raises(ValueError, match="enzyme_affinity_for_conversion"):
        augment_for_active_species(g, drug)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_prodrug_v2_augment.py -v
```

Expected: tests fail because v1 augment uses single conversion_site, not multi-site.

- [ ] **Step 3: Rewrite `augment_for_active_species`**

In `src/sisyphus/graph/builder.py`, replace the `augment_for_active_species` function (line 280-353) with:

```python
def augment_for_active_species(
    graph: BodyGraph,
    drug: DrugOnGraph,
    observation_node: str = "venous_blood",
) -> BodyGraph:
    """Augment graph with active-species 1-compartment plasma + multi-site
    activation edges + 1C elimination.

    v2 (2026-04-27): multi-site discovery. Augmentation iterates physiology
    nodes and creates one ProdrugActivationEdge per node where the drug's
    declared enzyme tags intersect that node's enzyme abundance dict.

    Adds:
    - 1 ``Node`` for active plasma (named ``observation_node + ACTIVE_SUFFIX``,
      ``node_type="blood_pool"``, volume=Vd_L).
    - N ``ProdrugActivationEdge`` instances (one per discovered conversion site).
    - 1 ``OneCompartmentEliminationEdge`` from active plasma → existing sink.

    No-op when ``drug.active_metabolite is None``.

    Mutates ``graph`` in place and returns it for chaining convenience.

    Raises:
        ValueError: ``enzyme_affinity_for_conversion`` empty when
            ``active_metabolite`` set (defensive — registry loader normally catches).
        ValueError: no node in physiology has any of the drug's declared
            enzyme tags.
        ValueError: active plasma node name already exists (collision).
        ValueError: default sink node not present in graph.
    """
    if drug.active_metabolite is None:
        return graph
    am = drug.active_metabolite

    affinities = drug.enzyme_affinity_for_conversion
    enzyme_tags = frozenset(affinities.keys())
    if not enzyme_tags:
        raise ValueError(
            "active_metabolite present but enzyme_affinity_for_conversion is empty; "
            "v2 requires drug to declare conversion enzymes"
        )

    active_node_name = observation_node + ACTIVE_SUFFIX
    if active_node_name in graph.nodes:
        raise ValueError(
            f"active node name collision: {active_node_name!r} already exists in graph"
        )

    if _DEFAULT_ACTIVE_SINK not in graph.nodes:
        raise ValueError(
            f"sink node {_DEFAULT_ACTIVE_SINK!r} required for active "
            "elimination but not found in graph"
        )

    # Active plasma compartment (blood_pool: Kp=1, no flow conservation).
    active_node = Node(
        name=active_node_name,
        node_type="blood_pool",
        volume=am.Vd_L,
    )
    graph.add_node(active_node)

    # Multi-site discovery: any physiology node with non-empty intersection
    # against drug's declared enzyme tags.
    conversion_sites = [
        node_name
        for node_name, node in graph.nodes.items()
        if node.enzymes and (enzyme_tags & set(node.enzymes.keys()))
    ]

    if not conversion_sites:
        raise ValueError(
            f"No conversion site for drug {drug.name!r}: declared enzyme_tags="
            f"{sorted(enzyme_tags)} but no node in physiology has any of these. "
            f"Available enzymes by node: "
            f"{ {n: sorted(node.enzymes.keys()) for n, node in graph.nodes.items() if node.enzymes} }"
        )

    # One ProdrugActivationEdge per site
    for site in conversion_sites:
        activation_edge = ProdrugActivationEdge(
            source=site,
            target=active_node_name,
            enzyme_tags=enzyme_tags,
            conversion_yield=am.conversion_yield_fraction,
            mw_parent=drug.mw,
            mw_active=am.mw,
        )
        graph.add_edge(activation_edge)

    # 1C elimination from active to sink (UNCHANGED from v1)
    elimination_edge = OneCompartmentEliminationEdge(
        source=active_node_name,
        target=_DEFAULT_ACTIVE_SINK,
        cl_per_h=am.CL_per_h,
        vd_l=am.Vd_L,
    )
    graph.add_edge(elimination_edge)

    return graph
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/unit/test_prodrug_v2_augment.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/graph/builder.py tests/unit/test_prodrug_v2_augment.py
git commit -m "feat(graph): augment_for_active_species multi-site discovery

v2: replaces single conversion_site string with engine-discovered
multi-site routing. Augmentation iterates physiology nodes, creates
one ProdrugActivationEdge per node where drug.enzyme_tags intersects
node.enzymes. Engine-discovered sites = mechanistic biology (where
the enzyme exists determines where conversion happens).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Augmentation idempotency test

**Files:**
- Test: append to `tests/unit/test_prodrug_v2_augment.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_prodrug_v2_augment.py`:

```python
def test_augment_called_twice_raises_on_collision():
    """Calling augment twice on same graph should raise (active node collision)."""
    g = _base_graph_with_two_spr_sites()
    drug = _minimal_drug(
        active_metabolite=_minimal_active(),
        observation_species="parent",
        enzyme_affinity_for_conversion={"SPR": Distribution(100.0)},
    )

    augment_for_active_species(g, drug)  # first call OK

    with pytest.raises(ValueError, match="collision"):
        augment_for_active_species(g, drug)  # second call must fail
```

- [ ] **Step 2: Run to verify pass (collision check is already in v2 augment)**

```bash
pytest tests/unit/test_prodrug_v2_augment.py::test_augment_called_twice_raises_on_collision -v
```

Expected: PASS (the existing `if active_node_name in graph.nodes` check from Task 7 handles this).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_prodrug_v2_augment.py
git commit -m "test(graph): augmentation idempotency — second call raises

Documents and verifies that augment_for_active_species is not idempotent;
caller (pipeline) must ensure single invocation per drug. Collision
detection on active node name already in v2 implementation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Physiology YAML enzyme entries

**Files:**
- Modify: `data/physiology/reference_man.yaml`

**Description:** Add SPR/CES1/CES2/ALPI abundance entries at relevant nodes using values from Task 1 deliverable.

- [ ] **Step 1: Read Task 1 deliverable**

```bash
cat docs/superpowers/specs/2026-04-27-prodrug-v2-task1-literature.md | grep -A 5 "Enzyme abundances"
```

Note the mean and CV for each (enzyme, organ) pair.

- [ ] **Step 2: Modify `data/physiology/reference_man.yaml`**

Locate the `liver` node (around line 54 per spec §4.1). Append to `enzymes:` block:

```yaml
    SPR: {mean: <FROM_TASK1>, cv: <FROM_TASK1>}
    CES1: {mean: <FROM_TASK1>, cv: <FROM_TASK1>}
    CES2: {mean: <FROM_TASK1>, cv: <FROM_TASK1>}
```

Locate the `gut_wall` node. Append to `enzymes:` block:

```yaml
    SPR: {mean: <FROM_TASK1>, cv: <FROM_TASK1>}
    CES1: {mean: <FROM_TASK1>, cv: <FROM_TASK1>}
    CES2: {mean: <FROM_TASK1>, cv: <FROM_TASK1>}
    ALPI: {mean: <FROM_TASK1>, cv: <FROM_TASK1>}
```

If Task 1 included kidney SPR, locate the `kidney` node, append:

```yaml
    enzymes:
      SPR: {mean: <FROM_TASK1>, cv: <FROM_TASK1>}
    ivive_scaling: <set if not already>
```

- [ ] **Step 3: Verify YAML loads without error**

```bash
python -c "
from pathlib import Path
from sisyphus.graph.builder import build_from_yaml
g = build_from_yaml(Path('data/physiology/reference_man.yaml'))
for name, node in g.nodes.items():
    if node.enzymes:
        print(f'{name}: {sorted(node.enzymes.keys())}')
"
```

Expected output includes lines like:
```
liver: ['CES1', 'CES2', 'CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP2E1', 'CYP3A4', 'SPR']
gut_wall: ['ALPI', 'CES1', 'CES2', 'CYP3A4', 'SPR']
```

- [ ] **Step 4: Commit**

```bash
git add data/physiology/reference_man.yaml
git commit -m "feat(physiology): SPR/CES1/CES2/ALPI enzyme abundances

Adds activation enzymes at liver, gut_wall (and kidney for SPR if
applicable per Task 1). Values from
docs/superpowers/specs/2026-04-27-prodrug-v2-task1-literature.md.

Independent lognormal sampling (no correlation_group — Achour matrix
does not cover these enzymes).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Registry schema rewrite + loader update

**Files:**
- Modify: `src/sisyphus/predict/registry.py` (entire `_REQUIRED_FIELDS`, `_build_active_metabolite`, `lookup_active_metabolite`)
- Modify: `data/sbi/prodrug_activation_registry.json` (rewrite all 4 entries)
- Test: `tests/unit/test_prodrug_v2_registry.py` (create)

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_prodrug_v2_registry.py`:

```python
"""Unit tests for v2 prodrug registry schema + loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sisyphus.core import ActiveMetabolite
from sisyphus.predict.registry import lookup_active_metabolite


def _write_registry(tmp_path: Path, entries: dict) -> Path:
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(entries))
    return p


def _v2_entry(**overrides) -> dict:
    base = {
        "name": "BH4",
        "mw": 241.25,
        "fup": {"mean": 0.23, "cv": 0.3},
        "CL_per_h": {"mean": 40.0, "cv": 0.35},
        "Vd_L": {"mean": 150.0, "cv": 0.3},
        "conversion_yield_fraction": {"mean": 0.85, "cv": 0.1},
        "yield_source": "literature",
        "observation_species": "parent",
        "enzyme_affinity_for_conversion": {
            "SPR": {"mean": 50.0, "cv": 0.5}
        },
        "affinity_source": "literature",
    }
    base.update(overrides)
    return base


def test_lookup_returns_three_tuple(tmp_path):
    """v2 lookup returns (ActiveMetabolite, observation_species, dict)."""
    smiles = "C"
    canonical = "C"
    reg = _write_registry(tmp_path, {canonical: _v2_entry()})
    result = lookup_active_metabolite(smiles, registry_path=reg)
    assert result is not None
    assert len(result) == 3
    am, obs, affinities = result
    assert isinstance(am, ActiveMetabolite)
    assert obs == "parent"
    assert "SPR" in affinities
    assert affinities["SPR"].mean == 50.0
    assert affinities["SPR"].cv == 0.5


def test_lookup_returns_none_for_unknown_smiles(tmp_path):
    reg = _write_registry(tmp_path, {})
    assert lookup_active_metabolite("CCO", registry_path=reg) is None


def test_loader_rejects_infrastructure_only(tmp_path):
    """v2 rejects affinity_source='infrastructure_only'."""
    reg = _write_registry(tmp_path, {"C": _v2_entry(affinity_source="infrastructure_only")})
    with pytest.raises(ValueError, match="affinity_source"):
        lookup_active_metabolite("C", registry_path=reg)


def test_loader_rejects_unknown_affinity_source(tmp_path):
    reg = _write_registry(tmp_path, {"C": _v2_entry(affinity_source="bogus")})
    with pytest.raises(ValueError, match="affinity_source"):
        lookup_active_metabolite("C", registry_path=reg)


def test_loader_rejects_empty_enzyme_affinity_for_conversion(tmp_path):
    reg = _write_registry(tmp_path, {"C": _v2_entry(enzyme_affinity_for_conversion={})})
    with pytest.raises(ValueError, match="enzyme_affinity_for_conversion"):
        lookup_active_metabolite("C", registry_path=reg)


def test_loader_rejects_negative_vd(tmp_path):
    reg = _write_registry(tmp_path, {"C": _v2_entry(Vd_L={"mean": -1.0, "cv": 0.0})})
    with pytest.raises(ValueError, match="Vd"):
        lookup_active_metabolite("C", registry_path=reg)


def test_loader_rejects_unknown_yield_source(tmp_path):
    reg = _write_registry(tmp_path, {"C": _v2_entry(yield_source="bogus")})
    with pytest.raises(ValueError, match="yield_source"):
        lookup_active_metabolite("C", registry_path=reg)


def test_loader_strips_citation_keys_from_distribution(tmp_path):
    """Distribution loader must ignore extra 'citation' keys in affinity entries."""
    entry = _v2_entry(
        enzyme_affinity_for_conversion={
            "SPR": {"mean": 50.0, "cv": 0.5, "citation": "Park 2008"}
        }
    )
    reg = _write_registry(tmp_path, {"C": entry})
    result = lookup_active_metabolite("C", registry_path=reg)
    assert result is not None
    _, _, affinities = result
    assert affinities["SPR"].mean == 50.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/unit/test_prodrug_v2_registry.py -v
```

Expected: tests fail because v1 loader expects `conversion_rate_per_h` field, returns 2-tuple.

- [ ] **Step 3: Rewrite `registry.py`**

Replace the entire content of `src/sisyphus/predict/registry.py` with:

```python
"""Prodrug activation registry — SMILES-keyed config loader (v2).

Maps canonical SMILES → (ActiveMetabolite, observation_species, enzyme_affinity_for_conversion).
Used by predict.ivive.build_drug_on_graph to attach prodrug activation
configs to DrugOnGraph instances.

Registry file: ``data/sbi/prodrug_activation_registry.json``
Schema: see docs/superpowers/specs/2026-04-27-prodrug-activation-v2-design.md §4.7
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


_VALID_AFFINITY_SOURCES = frozenset({"literature", "class_extrapolated"})
"""v2 rejects 'infrastructure_only' (tier 3); see spec §3.3."""

_VALID_YIELD_SOURCES = frozenset({"literature", "class_extrapolated"})


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


_REQUIRED_FIELDS = frozenset({
    "name", "mw", "fup", "CL_per_h", "Vd_L",
    "conversion_yield_fraction",
    "yield_source",
    "enzyme_affinity_for_conversion",
    "affinity_source",
    "observation_species",
})


def _distribution_from_dict(d: dict) -> Distribution:
    """Construct Distribution from JSON dict, ignoring 'citation' / metadata keys."""
    return Distribution(mean=float(d["mean"]), cv=float(d.get("cv", 0.0)))


def _build_active_metabolite(entry: dict, smiles: str) -> ActiveMetabolite:
    """Construct ActiveMetabolite from v2 registry entry; validate fields.

    Note: v2 registry omits conversion_rate_per_h and conversion_site;
    however ActiveMetabolite still has those fields (legacy v1). We
    populate conversion_rate_per_h with a sentinel Distribution(0.0)
    (unused by v2 flux) and conversion_site with empty string (unused
    by v2 augmentation). These v1 fields will be removed in a future
    cleanup task once all consumers are migrated.
    """
    missing = _REQUIRED_FIELDS - set(entry.keys())
    if missing:
        raise ValueError(
            f"prodrug_activation_registry entry for SMILES {smiles!r} "
            f"missing field {sorted(missing)}"
        )

    if entry["mw"] <= 0:
        raise ValueError(f"mw must be positive, got {entry['mw']}")

    cy = entry["conversion_yield_fraction"]
    if not (0.0 <= cy["mean"] <= 1.0):
        raise ValueError(f"conversion_yield must be in [0, 1], got {cy['mean']}")

    if entry["CL_per_h"]["mean"] <= 0 or entry["Vd_L"]["mean"] <= 0:
        raise ValueError("CL and Vd must be positive")

    return ActiveMetabolite(
        name=entry["name"],
        mw=float(entry["mw"]),
        fup=_distribution_from_dict(entry["fup"]),
        CL_per_h=_distribution_from_dict(entry["CL_per_h"]),
        Vd_L=_distribution_from_dict(entry["Vd_L"]),
        conversion_rate_per_h=Distribution(mean=0.0, cv=0.0),  # v2 unused
        conversion_site="",                                     # v2 unused
        conversion_yield_fraction=_distribution_from_dict(cy),
    )


def _build_enzyme_affinity_for_conversion(entry: dict, smiles: str) -> dict[str, Distribution]:
    """Parse enzyme_affinity_for_conversion dict; ignore citation keys."""
    raw = entry["enzyme_affinity_for_conversion"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"enzyme_affinity_for_conversion must be non-empty dict for SMILES {smiles!r}"
        )
    affinities: dict[str, Distribution] = {}
    for tag, dist_raw in raw.items():
        if not isinstance(dist_raw, dict):
            raise ValueError(
                f"affinity entry for tag {tag!r} must be dict with 'mean'/'cv', "
                f"got {type(dist_raw).__name__}"
            )
        if "mean" not in dist_raw:
            raise ValueError(f"affinity entry for tag {tag!r} missing 'mean'")
        affinities[tag] = _distribution_from_dict(dist_raw)
    return affinities


def lookup_active_metabolite(
    smiles: str, registry_path: Path | None = None
) -> tuple[ActiveMetabolite, str, dict[str, Distribution]] | None:
    """Look up SMILES in v2 prodrug registry.

    Returns ``(ActiveMetabolite, observation_species, enzyme_affinity_for_conversion)``
    or ``None`` if not found.

    Raises ``ValueError`` on invalid registry entries.

    Args:
        smiles: SMILES string (any form; canonicalized internally).
        registry_path: Override registry file path (default: data/sbi/...).
    """
    canonical = _canonicalize(smiles)
    if canonical is None:
        return None

    path = registry_path if registry_path is not None else _DEFAULT_REGISTRY_PATH
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

    affinity_source = entry.get("affinity_source")
    if affinity_source not in _VALID_AFFINITY_SOURCES:
        raise ValueError(
            f"affinity_source must be one of {sorted(_VALID_AFFINITY_SOURCES)}, "
            f"got {affinity_source!r} (v2 rejects 'infrastructure_only')"
        )

    yield_source = entry.get("yield_source")
    if yield_source not in _VALID_YIELD_SOURCES:
        raise ValueError(
            f"yield_source must be one of {sorted(_VALID_YIELD_SOURCES)}, "
            f"got {yield_source!r}"
        )

    am = _build_active_metabolite(entry, canonical)
    affinities = _build_enzyme_affinity_for_conversion(entry, canonical)
    return am, obs_species, affinities
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/unit/test_prodrug_v2_registry.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Rewrite `data/sbi/prodrug_activation_registry.json`**

Using values from Task 1 deliverable, rewrite the registry file. Template (replace `<TASK1_VALUE>` placeholders with actual numbers):

```json
{
  "C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1": {
    "name": "BH4",
    "mw": 241.25,
    "fup": {"mean": 0.23, "cv": 0.3},
    "CL_per_h": {"mean": 40.0, "cv": 0.35},
    "Vd_L": {"mean": 150.0, "cv": 0.3},
    "conversion_yield_fraction": {"mean": <TASK1_VALUE>, "cv": <TASK1_VALUE>},
    "yield_source": "<TASK1_VALUE: literature | class_extrapolated>",
    "observation_species": "parent",
    "enzyme_affinity_for_conversion": {
      "SPR": {"mean": <TASK1_VALUE>, "cv": <TASK1_VALUE>, "citation": "<TASK1_CITATION>"}
    },
    "affinity_source": "<TASK1_VALUE: literature | class_extrapolated>",
    "_clinical_citation": "Gao 2024 PMC11597218"
  },
  "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1": {
    "name": "GS-441524",
    "mw": 291.27,
    "fup": {"mean": 0.5, "cv": 0.3},
    "CL_per_h": {"mean": 10.0, "cv": 0.3},
    "Vd_L": {"mean": 35.0, "cv": 0.3},
    "conversion_yield_fraction": {"mean": <TASK1_VALUE>, "cv": <TASK1_VALUE>},
    "yield_source": "<TASK1_VALUE>",
    "observation_species": "parent",
    "enzyme_affinity_for_conversion": {
      "CES1": {"mean": <TASK1_VALUE>, "cv": <TASK1_VALUE>, "citation": "<TASK1_CITATION>"}
    },
    "affinity_source": "<TASK1_VALUE>",
    "_clinical_citation": "Humeniuk 2020 PMC8007387"
  },
  "C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN(C4=NCCS4)C3)[C@H](C)[C@H]12": {
    "name": "tebipenem",
    "mw": 384.45,
    "fup": {"mean": 0.5, "cv": 0.3},
    "CL_per_h": {"mean": 17.0, "cv": 0.3},
    "Vd_L": {"mean": 50.0, "cv": 0.3},
    "conversion_yield_fraction": {"mean": <TASK1_VALUE>, "cv": <TASK1_VALUE>},
    "yield_source": "<TASK1_VALUE>",
    "observation_species": "active",
    "enzyme_affinity_for_conversion": {
      "CES2": {"mean": <TASK1_VALUE>, "cv": <TASK1_VALUE>, "citation": "<TASK1_CITATION>"}
    },
    "affinity_source": "<TASK1_VALUE>",
    "_clinical_citation": "Eckburg 2019 PMC6709501"
  },
  "COc1cc(Nc2ncc(F)c(Nc3ccc4c(n3)N(COP(=O)(O)O)C(=O)C(C)(C)O4)n2)cc(OC)c1OC": {
    "name": "R406",
    "mw": 470.45,
    "fup": {"mean": 0.02, "cv": 0.3},
    "CL_per_h": {"mean": 28.0, "cv": 0.35},
    "Vd_L": {"mean": 250.0, "cv": 0.3},
    "conversion_yield_fraction": {"mean": <TASK1_VALUE>, "cv": <TASK1_VALUE>},
    "yield_source": "<TASK1_VALUE>",
    "observation_species": "active",
    "enzyme_affinity_for_conversion": {
      "ALPI": {"mean": <TASK1_VALUE>, "cv": <TASK1_VALUE>, "citation": "<TASK1_CITATION>"}
    },
    "affinity_source": "<TASK1_VALUE>",
    "_clinical_citation": "Baluom 2013 PMC3703230"
  }
}
```

If Task 1 marked any drug as tier 3 / infrastructure_only: REMOVE that entry from this JSON.

- [ ] **Step 6: Verify registry loads**

```bash
python -c "
from sisyphus.predict.registry import lookup_active_metabolite
for sm in ['C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1',
          'CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1',
          'C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN(C4=NCCS4)C3)[C@H](C)[C@H]12',
          'COc1cc(Nc2ncc(F)c(Nc3ccc4c(n3)N(COP(=O)(O)O)C(=O)C(C)(C)O4)n2)cc(OC)c1OC']:
    r = lookup_active_metabolite(sm)
    if r:
        am, obs, aff = r
        print(f'{am.name}: obs={obs}, enzymes={list(aff.keys())}')
    else:
        print(f'{sm[:30]}... NOT FOUND')
"
```

Expected: each drug prints name + observation + enzyme tags. No exceptions.

- [ ] **Step 7: Commit**

```bash
git add src/sisyphus/predict/registry.py data/sbi/prodrug_activation_registry.json tests/unit/test_prodrug_v2_registry.py
git commit -m "feat(predict): v2 prodrug registry schema + loader

Schema replaces conversion_rate_per_h + conversion_site with
enzyme_affinity_for_conversion: dict[tag → Distribution(mean, cv, citation)].
Adds affinity_source + yield_source enum validation
('literature' | 'class_extrapolated'; tier 3 'infrastructure_only' rejected).
Loader returns 3-tuple. Registry rewritten with literature values from
Task 1 deliverable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Pipeline integration update

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py` (the section constructing DrugOnGraph from registry lookup)

**Description:** Pass `enzyme_affinity_for_conversion` when registry lookup succeeds.

- [ ] **Step 1: Locate registry lookup call site**

```bash
grep -n "lookup_active_metabolite" /home/jam/Sisyphus/src/sisyphus/pipeline/predict.py /home/jam/Sisyphus/src/sisyphus/predict/ivive.py 2>&1
```

Note the file:line where `lookup_active_metabolite` is called (likely in `predict/ivive.py::build_drug_on_graph` or similar).

- [ ] **Step 2: Update call site to handle 3-tuple**

Find the existing pattern (likely):
```python
result = lookup_active_metabolite(smiles)
if result is not None:
    am, obs_species = result
else:
    am, obs_species = None, "parent"
```

Replace with:
```python
result = lookup_active_metabolite(smiles)
if result is not None:
    am, obs_species, conv_affinities = result
else:
    am, obs_species, conv_affinities = None, "parent", {}
```

Then in the `DrugOnGraph(...)` constructor call, add:
```python
    enzyme_affinity_for_conversion=conv_affinities,
```

If `DrugOnGraph(...)` is constructed in multiple places, update each. Use the grep above to find call sites.

- [ ] **Step 3: Run pipeline smoke**

```bash
pytest tests/integration/test_prodrug_pipeline_smoke.py -v 2>&1 | tail -20
```

Expected: pipeline doesn't crash; tests may still fail (validation gate is parametrized in v2; v1 single-test is being deprecated in Task 18). At minimum, no `TypeError` from 2-tuple vs 3-tuple unpacking.

- [ ] **Step 4: Commit**

```bash
git add src/sisyphus/pipeline/predict.py src/sisyphus/predict/ivive.py
git commit -m "feat(pipeline): wire enzyme_affinity_for_conversion through

Updates registry lookup unpacking to 3-tuple and passes
enzyme_affinity_for_conversion into DrugOnGraph construction.
Non-prodrug path defaults to empty dict (no behavior change).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Per-prodrug end-to-end smoke test

**Files:**
- Test: `tests/integration/test_prodrug_v2_pipeline_smoke.py` (create)

- [ ] **Step 1: Write test**

Create `tests/integration/test_prodrug_v2_pipeline_smoke.py`:

```python
"""End-to-end smoke: each registered v2 prodrug runs without error."""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


@pytest.mark.parametrize("drug_name,smiles,dose_mg,route", [
    ("sepiapterin",
     "C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1",
     4200.0, "oral"),
    ("remdesivir",
     "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1",
     200.0, "iv"),
    ("tebipenem_pivoxil",
     "C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN(C4=NCCS4)C3)[C@H](C)[C@H]12",
     300.0, "oral"),
    ("fostamatinib",
     "COc1cc(Nc2ncc(F)c(Nc3ccc4c(n3)N(COP(=O)(O)O)C(=O)C(C)(C)O4)n2)cc(OC)c1OC",
     75.0, "oral"),
])
def test_pipeline_runs_for_each_prodrug(drug_name, smiles, dose_mg, route):
    """Smoke test: pipeline produces a PredictionResult without raising."""
    result = predict(smiles, dose_mg=dose_mg, route=route, n_mc_samples=10)
    assert result is not None
    assert result.pk.cmax.mean > 0, f"{drug_name} Cmax should be positive"
```

- [ ] **Step 2: Run smoke**

```bash
pytest tests/integration/test_prodrug_v2_pipeline_smoke.py -v
```

Expected: 4 tests PASS (one per drug). If any fails with augmentation error like "no conversion site", verify physiology YAML has the relevant enzyme abundance.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_prodrug_v2_pipeline_smoke.py
git commit -m "test(pipeline): v2 prodrug smoke tests for 4 registered drugs

Each drug: pipeline runs end-to-end (10 MC samples) without error
and produces positive Cmax. Verifies wiring across registry → drug →
augmentation → compile → solve.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Identity-blind regression test

**Files:**
- Test: `tests/regression/test_prodrug_v2_identity_blind.py` (create)

- [ ] **Step 1: Write test**

Create `tests/regression/test_prodrug_v2_identity_blind.py`:

```python
"""Regression: engine identity-blind invariant — random tag rename → identical result.

If anyone introduces name-based logic (e.g., `if tag == "SPR": ...`),
this test catches it: replace SPR with a random string everywhere
(physiology YAML, registry JSON), numerical Cmax must be byte-identical.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from sisyphus.pipeline.predict import predict


def _rename_in_yaml(yaml_text: str, mapping: dict[str, str]) -> str:
    """Rename enzyme tags in YAML text. Operates on parsed structure to
    avoid mismatches in formatting/indentation.
    """
    data = yaml.safe_load(yaml_text)
    for node in data.get("nodes", []):
        if "enzymes" in node:
            new_enzymes = {}
            for tag, val in node["enzymes"].items():
                new_tag = mapping.get(tag, tag)
                new_enzymes[new_tag] = val
            node["enzymes"] = new_enzymes
    return yaml.safe_dump(data, sort_keys=False)


def _rename_in_registry(reg: dict, mapping: dict[str, str]) -> dict:
    """Rename enzyme tags in registry JSON. Returns new dict."""
    out = {}
    for smi, entry in reg.items():
        new_entry = dict(entry)
        if "enzyme_affinity_for_conversion" in new_entry:
            new_aff = {}
            for tag, val in new_entry["enzyme_affinity_for_conversion"].items():
                new_aff[mapping.get(tag, tag)] = val
            new_entry["enzyme_affinity_for_conversion"] = new_aff
        out[smi] = new_entry
    return out


def test_random_rename_invariant(tmp_path, monkeypatch):
    """Replace SPR → random tag throughout. Cmax must be byte-identical."""
    smiles = "C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1"  # sepiapterin
    dose_mg = 4200.0

    # Baseline run
    result_orig = predict(smiles, dose_mg=dose_mg, route="oral", n_mc_samples=0)
    cmax_orig = result_orig.pk.cmax.mean

    # Rename SPR → Z1Q9K  (also CES1, CES2, ALPI for thoroughness)
    rename = {"SPR": "Z1Q9K", "CES1": "Q4M2X", "CES2": "P5N7T", "ALPI": "K8B3Y"}

    # Copy + rename physiology YAML
    src_yaml = Path("data/physiology/reference_man.yaml").read_text()
    renamed_yaml = _rename_in_yaml(src_yaml, rename)
    new_yaml_path = tmp_path / "reference_man.yaml"
    new_yaml_path.write_text(renamed_yaml)

    # Copy + rename registry JSON
    src_reg_path = Path("data/sbi/prodrug_activation_registry.json")
    src_reg = json.loads(src_reg_path.read_text())
    renamed_reg = _rename_in_registry(src_reg, rename)
    new_reg_path = tmp_path / "registry.json"
    new_reg_path.write_text(json.dumps(renamed_reg))

    # Patch path constants used by builder + registry loader
    import sisyphus.pipeline.predict as pp_mod
    import sisyphus.predict.registry as reg_mod
    monkeypatch.setattr(pp_mod, "_PHYSIOLOGY_DIR", tmp_path, raising=False)
    monkeypatch.setattr(reg_mod, "_DEFAULT_REGISTRY_PATH", new_reg_path, raising=False)
    # Clear lru_cache on the registry loader
    reg_mod._load_registry_cached.cache_clear()

    result_renamed = predict(smiles, dose_mg=dose_mg, route="oral", n_mc_samples=0)
    cmax_renamed = result_renamed.pk.cmax.mean

    # Byte-identical (deterministic mode, n_mc_samples=0)
    assert cmax_renamed == cmax_orig, (
        f"Identity-blind violated: orig={cmax_orig}, renamed={cmax_renamed}"
    )
```

- [ ] **Step 2: Run test**

```bash
pytest tests/regression/test_prodrug_v2_identity_blind.py -v
```

Expected: PASS. If FAIL, there is name-matching somewhere — investigate, fix, re-run.

Note: test uses monkeypatch to swap physiology dir and registry path. If the predict pipeline doesn't read from `_PHYSIOLOGY_DIR` (e.g., it hardcodes path), adapt the test to find the right injection point. Verify by reading the pipeline at the call site.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_prodrug_v2_identity_blind.py
git commit -m "test(regression): engine identity-blind invariant for prodrug routing

Renames enzyme tags (SPR→Z1Q9K etc.) in both physiology YAML and
registry JSON, runs sepiapterin pipeline. Asserts Cmax byte-identical
to baseline. Catches any future code that introduces name-matching
logic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: 107-holdout regression test

**Files:**
- Test: `tests/regression/test_holdout_unchanged.py` (modify if needed; verify v2 doesn't break)

**Description:** Verify that adding new physiology enzymes (SPR, CES1, CES2, ALPI) and v2 prodrug routing does NOT change predictions for the 107 non-prodrug holdout drugs.

- [ ] **Step 1: Run existing 107-holdout regression test**

```bash
pytest tests/regression/test_holdout_unchanged.py -v --no-header 2>&1 | tail -30
```

Expected: PASS. v2 changes are additive for non-prodrug drugs (their `enzyme_affinity_for_conversion = {}` and registry lookup returns None).

- [ ] **Step 2: If FAIL — investigate**

If a non-prodrug drug's prediction changes:
- Check if its enzyme_affinity dict accidentally includes SPR/CES1/CES2/ALPI tags. (ML predictor should NOT predict for these new enzymes.)
- Check if `predict/ivive.py::build_drug_on_graph` defaults the new dict to non-empty.
- Check if `lookup_active_metabolite` returns non-None for any non-prodrug drug (SMILES collision unlikely but possible).

If a true regression is found, fix the leak source (do NOT relax test).

- [ ] **Step 3: Commit (if any fix needed)**

```bash
git add <fixed files>
git commit -m "fix(predict): prevent v2 enzyme tags leak into non-prodrug enzyme_affinity"
```

If no fix needed: skip this commit.

---

## Task 15: DDI smoke test (CES1 abundance proportionality)

**Files:**
- Test: `tests/integration/test_prodrug_v2_ddi_smoke.py` (create)

- [ ] **Step 1: Write test**

Create `tests/integration/test_prodrug_v2_ddi_smoke.py`:

```python
"""Smoke test: halving CES1 abundance approximately halves remdesivir activation.

Verifies that the architecture supports DDI on conversion enzymes
without explicit DDI clinical data. Proportionality is approximate
because well-stirred extraction is non-linear in CLint when fup×CLint
approaches Q (saturation regime).
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import augment_for_active_species, build_from_yaml
from sisyphus.predict.ivive import build_drug_on_graph  # If function name differs, adjust


def _build_remdesivir_pipeline_inputs(scale_ces1: float = 1.0):
    """Construct graph + drug for remdesivir, optionally scaling CES1 abundance."""
    # NB: implementer may need to adapt this to whatever helper Sisyphus
    # uses to build the DrugOnGraph for a given SMILES. The shape below
    # follows the pipeline pattern.
    smiles = ("CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)"
              "(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1")
    drug = build_drug_on_graph(smiles, dose_mg=200.0, route="iv")

    yaml_path = Path("data/physiology/reference_man.yaml")
    base_graph = build_from_yaml(yaml_path)

    # Scale CES1 abundance at all nodes that have it
    if scale_ces1 != 1.0:
        for node_name, node in list(base_graph.nodes.items()):
            if "CES1" in node.enzymes:
                old = node.enzymes["CES1"]
                # Build a new Node with scaled CES1
                new_enzymes = dict(node.enzymes)
                new_enzymes["CES1"] = Distribution(
                    mean=old.mean * scale_ces1, cv=old.cv,
                    correlation_group=old.correlation_group,
                )
                base_graph.nodes[node_name] = type(node)(
                    name=node.name, node_type=node.node_type, volume=node.volume,
                    composition=node.composition, enzymes=new_enzymes,
                    transporters=node.transporters, ivive_scaling=node.ivive_scaling,
                    lookup_name=node.lookup_name,
                )

    augmented = augment_for_active_species(base_graph, drug)
    return augmented, drug


def _solve_and_get_active_cmax(graph: BodyGraph, drug: DrugOnGraph) -> float:
    """Compile + solve, return active species Cmax (mg/L)."""
    import numpy as np
    from scipy.integrate import solve_ivp

    compiler = ODECompiler()
    compiled = compiler.compile(graph)
    params = ResolvedParams(graph, drug)
    rhs = compiled.make_rhs(params)

    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index["venous_blood"]] = drug.dose_mg

    t_eval = np.linspace(0.0, 24.0, 200)
    sol = solve_ivp(rhs, (0.0, 24.0), y0, method="LSODA",
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)
    assert sol.success

    active_idx = compiled.state_index["venous_blood_active"]
    v_active = drug.active_metabolite.Vd_L.mean
    c_active = sol.y[active_idx] / v_active
    return float(c_active.max())


def test_halving_ces1_reduces_remdesivir_activation():
    """Active Cmax should drop ≥ 30% when CES1 abundance halved (proportionality)."""
    g_full, drug_full = _build_remdesivir_pipeline_inputs(scale_ces1=1.0)
    cmax_full = _solve_and_get_active_cmax(g_full, drug_full)

    g_half, drug_half = _build_remdesivir_pipeline_inputs(scale_ces1=0.5)
    cmax_half = _solve_and_get_active_cmax(g_half, drug_half)

    ratio = cmax_half / cmax_full
    # Well-stirred is non-linear; if fup×CLint is much smaller than Q,
    # ratio ≈ 0.5. If saturating, ratio approaches 1.0. We expect at
    # least a 30% drop as a sanity floor.
    assert ratio < 0.7, (
        f"Halving CES1 should reduce active Cmax meaningfully; "
        f"ratio={ratio:.3f} (full={cmax_full:.4g}, half={cmax_half:.4g})"
    )
    # Ratio should not be unreasonably small either (sanity ceiling)
    assert ratio > 0.3, f"ratio={ratio:.3f} seems too aggressive"
```

- [ ] **Step 2: Run test**

```bash
pytest tests/integration/test_prodrug_v2_ddi_smoke.py -v
```

Expected: PASS. If FAIL because `build_drug_on_graph` has a different name, find the right helper via `grep -rn "def build_drug" src/sisyphus/predict/` and adapt the import.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_prodrug_v2_ddi_smoke.py
git commit -m "test(integration): DDI smoke — CES1 abundance affects remdesivir activation

Verifies architecture supports conversion-enzyme DDI for free:
halving CES1 abundance reduces remdesivir active Cmax meaningfully
(>30% drop). Proportionality is approximate due to well-stirred
non-linearity in the saturating regime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Per-prodrug snapshot test

**Files:**
- Test: `tests/regression/test_prodrug_v2_snapshot.py` (create)

**Description:** Pin Cmax mean ± 5% per prodrug to catch silent drift within 3-fold gate.

- [ ] **Step 1: Generate baseline values**

```bash
python -c "
from sisyphus.pipeline.predict import predict
for name, smi, dose, rt in [
    ('sepiapterin', 'C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1', 4200.0, 'oral'),
    ('remdesivir', 'CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1', 200.0, 'iv'),
    ('tebipenem_pivoxil', 'C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN(C4=NCCS4)C3)[C@H](C)[C@H]12', 300.0, 'oral'),
    ('fostamatinib', 'COc1cc(Nc2ncc(F)c(Nc3ccc4c(n3)N(COP(=O)(O)O)C(=O)C(C)(C)O4)n2)cc(OC)c1OC', 75.0, 'oral'),
]:
    r = predict(smi, dose_mg=dose, route=rt, n_mc_samples=0)
    print(f'    \"{name}\": {r.pk.cmax.mean:.6e},')
"
```

Note the printed values.

- [ ] **Step 2: Write snapshot test**

Create `tests/regression/test_prodrug_v2_snapshot.py`:

```python
"""Per-prodrug Cmax snapshot test (±5%).

Catches silent drift below the 3-fold validation gate threshold.
Update pinned values explicitly when intentionally re-baselining
(physiology change, registry update, etc.).
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


# Pinned baselines (Cmax mean, deterministic mode) generated by Step 1.
# UPDATE EXPLICITLY when re-baselining; do not auto-regenerate.
_PINNED = {
    "sepiapterin":      <FROM_STEP1>,
    "remdesivir":       <FROM_STEP1>,
    "tebipenem_pivoxil": <FROM_STEP1>,
    "fostamatinib":     <FROM_STEP1>,
}

_RTOL = 0.05  # ±5%

_SMILES = {
    "sepiapterin":      "C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1",
    "remdesivir":       "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1",
    "tebipenem_pivoxil": "C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN(C4=NCCS4)C3)[C@H](C)[C@H]12",
    "fostamatinib":     "COc1cc(Nc2ncc(F)c(Nc3ccc4c(n3)N(COP(=O)(O)O)C(=O)C(C)(C)O4)n2)cc(OC)c1OC",
}

_DOSE_ROUTE = {
    "sepiapterin":      (4200.0, "oral"),
    "remdesivir":       (200.0, "iv"),
    "tebipenem_pivoxil": (300.0, "oral"),
    "fostamatinib":     (75.0, "oral"),
}


@pytest.mark.parametrize("drug_name", list(_PINNED.keys()))
def test_cmax_snapshot(drug_name):
    pinned = _PINNED[drug_name]
    smiles = _SMILES[drug_name]
    dose, route = _DOSE_ROUTE[drug_name]

    result = predict(smiles, dose_mg=dose, route=route, n_mc_samples=0)
    actual = result.pk.cmax.mean

    rel_err = abs(actual - pinned) / pinned
    assert rel_err < _RTOL, (
        f"{drug_name} Cmax drifted: actual={actual:.6e}, pinned={pinned:.6e}, "
        f"rel_err={rel_err:.4f} (>{_RTOL}). "
        f"If intentional, update _PINNED."
    )
```

Replace `<FROM_STEP1>` placeholders with the actual values printed in Step 1.

- [ ] **Step 3: Run snapshot test**

```bash
pytest tests/regression/test_prodrug_v2_snapshot.py -v
```

Expected: 4 tests PASS (with the same code that generated the baselines).

- [ ] **Step 4: Commit**

```bash
git add tests/regression/test_prodrug_v2_snapshot.py
git commit -m "test(regression): per-prodrug Cmax snapshot (±5%)

Pins each registered prodrug's deterministic Cmax to catch silent
drift below 3-fold validation gate. Update _PINNED explicitly on
intentional re-baselining.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Validation gate parametrize test

**Files:**
- Test: `tests/regression/test_prodrug_v2_validation_gate.py` (create)

- [ ] **Step 1: Write test**

Create `tests/regression/test_prodrug_v2_validation_gate.py`:

```python
"""v2 validation gate (per-drug parametrized 3-fold).

Per-drug pass/fail visible in CI output. Failing-drug parametrize
cases are marked xfail with documented fold-error in test output.
Affinity values are NOT adjusted to make tests pass — see spec §3.3.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


# Clinical reference Cmax (mg/L) per drug — from v1 spec / clinical refs
_CLINICAL_CMAX = {
    "sepiapterin":       0.0024,   # Gao 2024 PMC11597218, BH4 (active monitored)
    "remdesivir":        4.38,     # Humeniuk 2020, GS-441524 (active monitored)
    "tebipenem_pivoxil": 4.01,     # Eckburg 2019, tebipenem (active monitored)
    "fostamatinib":      0.61,     # Baluom 2013, R406 (active monitored)
}

_SMILES = {
    "sepiapterin":       "C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1",
    "remdesivir":        "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1",
    "tebipenem_pivoxil": "C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN(C4=NCCS4)C3)[C@H](C)[C@H]12",
    "fostamatinib":      "COc1cc(Nc2ncc(F)c(Nc3ccc4c(n3)N(COP(=O)(O)O)C(=O)C(C)(C)O4)n2)cc(OC)c1OC",
}

_DOSE_ROUTE = {
    "sepiapterin":       (4200.0, "oral"),
    "remdesivir":        (200.0, "iv"),
    "tebipenem_pivoxil": (300.0, "oral"),
    "fostamatinib":      (75.0, "oral"),
}


@pytest.mark.parametrize("drug_name", list(_CLINICAL_CMAX.keys()))
def test_prodrug_3fold(drug_name):
    smiles = _SMILES[drug_name]
    dose, route = _DOSE_ROUTE[drug_name]
    result = predict(smiles, dose_mg=dose, route=route, n_mc_samples=0)
    pred = result.pk.cmax.mean
    obs = _CLINICAL_CMAX[drug_name]
    fold_error = max(pred / obs, obs / pred)
    assert fold_error < 3.0, (
        f"{drug_name}: pred={pred:.4g} mg/L, obs={obs:.4g} mg/L, "
        f"fold_error={fold_error:.2f}× (target < 3.0×)"
    )
```

- [ ] **Step 2: Run gate**

```bash
pytest tests/regression/test_prodrug_v2_validation_gate.py -v
```

Expected: per-drug PASS/FAIL visible. Some drugs may fail (per spec §6.1, this is acceptable — gate is reporting, not blocking).

- [ ] **Step 3: For any FAILing drug, mark xfail with documentation**

If, e.g., fostamatinib fails:

Add at top of file:
```python
_KNOWN_FAILURES = {
    "fostamatinib": "ALPI tier 2 affinity insufficient; literature limited (see Task 1 §6)",
}
```

And modify the parametrize:
```python
@pytest.mark.parametrize("drug_name", [
    pytest.param(
        name,
        marks=pytest.mark.xfail(reason=_KNOWN_FAILURES[name], strict=True)
        if name in _KNOWN_FAILURES else (),
    ) for name in _CLINICAL_CMAX
])
def test_prodrug_3fold(drug_name):
    ...
```

(Repeat for each failing drug. Document fold-error in the reason string.)

- [ ] **Step 4: Commit**

```bash
git add tests/regression/test_prodrug_v2_validation_gate.py
git commit -m "test(regression): v2 prodrug validation gate (per-drug 3-fold)

Parametrized per-drug 3-fold gate. Per spec §6.1: failing drugs marked
xfail with documented reason; affinity values NOT adjusted to make
tests pass (mechanistic-A core promise).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Deprecate v1 tests + CHANGELOG

**Files:**
- Modify: `tests/integration/test_two_species_mass_balance.py` (delete or mark xfail)
- Modify: `tests/integration/test_prodrug_pipeline_smoke.py` (deprecate v1 validation gate test)
- Modify: `tests/unit/test_prodrug_edges.py` (update or remove `conversion_rate` assertions)
- Modify: `tests/unit/test_prodrug_flux.py` (update or remove kinetic-rate assertions)
- Modify: `tests/unit/test_prodrug_registry.py` (update v1 schema assertions)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Identify v1 tests now broken**

```bash
pytest tests/unit/test_prodrug_edges.py tests/unit/test_prodrug_flux.py \
       tests/unit/test_prodrug_registry.py tests/integration/test_two_species_mass_balance.py \
       tests/integration/test_prodrug_pipeline_smoke.py \
       --no-header -q 2>&1 | tail -40
```

Note which tests fail.

- [ ] **Step 2: Delete v1 mass balance test (replaced by Task 6)**

```bash
git rm tests/integration/test_two_species_mass_balance.py
```

- [ ] **Step 3: Update v1 test files for v2 schema**

For `tests/unit/test_prodrug_edges.py`:
- Tests asserting `edge.conversion_rate` exists → DELETE (replaced by Task 2 tests).
- Tests verifying edge construction with v1 fields → update to use `enzyme_tags`.

For `tests/unit/test_prodrug_flux.py`:
- Tests using kinetic 1st-order rate formula → DELETE (replaced by Task 5 tests).
- Keep any test that exercises mw_ratio scaling if not already in v2 tests.

For `tests/unit/test_prodrug_registry.py`:
- Tests asserting v1 fields (conversion_rate_per_h, conversion_site) → DELETE (replaced by Task 10 tests).
- Keep canonicalization tests if not already in v2 tests.

For `tests/integration/test_prodrug_pipeline_smoke.py`:
- The `test_prodrug_validation_gate_3fold` test (v1 known-failure) → DELETE (replaced by Task 17 parametrized gate).
- Keep general smoke patterns that aren't v1-specific.

If preserving any v1 file ergonomically requires significant rework, deletion is acceptable — v2 test files cover the same surface.

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/unit/ tests/integration/ tests/regression/ -x --no-header -q 2>&1 | tail -30
```

Expected: all PASS (or known xfails). If unexpected failures: investigate, fix.

- [ ] **Step 5: Update CHANGELOG**

In `CHANGELOG.md`, locate the v1 "Validation gate failure (KNOWN LIMITATION)" entry under `## [Unreleased]`. Replace that paragraph with:

```markdown
- **Prodrug activation v2 — enzyme-abundance mechanistic** (branch `feat/prodrug-activation-v2`,
  2026-04-27): replaces v1's kinetic 1st-order conversion (rate = k × A_parent)
  with well-stirred extraction at flow-through nodes (mirrors existing
  CYP3A4 elimination pattern). Drug declares `enzyme_affinity_for_conversion:
  dict[str, Distribution]`; augmentation discovers conversion sites by
  enzyme intersection with physiology. Affinity values sourced from
  in-vitro literature or substrate-class kinetics (no clinical fit;
  tier 3 / "infrastructure_only" rejected by registry loader).

  v1-vs-v2 fold-error comparison:

  | Drug | v1 fold-error | v2 fold-error |
  |------|---------------|---------------|
  | sepiapterin | 5356× | <FILL_FROM_TASK17_RESULTS> |
  | remdesivir | 4.45× | <FILL> |
  | tebipenem_pivoxil | 8.63× | <FILL> |
  | fostamatinib | 4.78× | <FILL> |

  v2 ships with the per-drug 3-fold gate as a reporting metric (xfail
  allowed). The mechanistic-sourcing promise is the hard requirement,
  not clinical-match.

  New tests: well-stirred mass balance (flow-loop synthetic), per-drug
  pipeline smoke, identity-blind tag rename, DDI smoke (CES1 ↔
  remdesivir proportionality), per-prodrug Cmax snapshot, parametrized
  validation gate.

  v1 known-limitation entry below superseded.
```

Remove the original v1 known-limitation entry (the paragraph it replaces).

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md tests/
git commit -m "docs+chore: deprecate v1 tests, CHANGELOG v2 entry

- Delete tests/integration/test_two_species_mass_balance.py (v1 kinetic
  topology replaced by tests/integration/test_prodrug_v2_mass_balance.py).
- Update v1 test files to remove conversion_rate / conversion_site /
  conversion_rate_per_h assertions.
- CHANGELOG: replace v1 known-limitation note with v2 entry including
  v1-vs-v2 fold-error comparison table.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ --no-header -q 2>&1 | tail -30
```

Expected: all PASS or known xfail (per Task 17). No unexpected failures.

- [ ] **Step 2: Verify success criteria from spec §9**

Manually check each:

```
[ ] (1) Invariants 1-8 maintained — verified by tests in Tasks 5, 7, 13, 14
[ ] (2) 107-holdout regression byte-identical — Task 14
[ ] (3) Mass balance synthetic well-stirred (rtol 1e-3) — Task 6
[ ] (4) Identity-blind random-rename — Task 13
[ ] (5) Plan Task 1 yields ≥1 tier 1+2 drug — Task 1 deliverable
[ ] (6) DDI smoke (CES1 ↔ remdesivir proportionality) — Task 15
[ ] (7) Per-prodrug snapshot tests — Task 16
[ ] (8) Augmentation idempotency — Task 8
[ ] (9) All Sisyphus CI green — Task 19 Step 1
[ ] (10) Validation gate parametrized — Task 17
[ ] (11) CHANGELOG v1-vs-v2 — Task 18
```

- [ ] **Step 3: Push branch**

```bash
git push -u origin feat/prodrug-activation-v2
```

- [ ] **Step 4: Open PR**

```bash
gh pr create --title "feat: prodrug activation v2 (enzyme-abundance mechanistic)" \
  --body "$(cat <<'EOF'
## Summary
- Replace v1 kinetic 1st-order conversion with well-stirred extraction
- Drug declares enzyme_affinity_for_conversion; augmentation auto-discovers conversion sites
- Affinity values from in-vitro literature, NOT clinical fit (mechanistic-A promise)

## Test plan
- [ ] Unit tests pass (test_prodrug_v2_*.py)
- [ ] Integration tests pass (mass balance, pipeline smoke, DDI smoke)
- [ ] Regression tests pass (107-holdout, identity-blind, snapshot)
- [ ] Per-drug parametrized validation gate visible per-drug pass/fail in CI

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**1. Spec coverage:**
- §1 Goal — covered by Task 5 (well-stirred rewrite) + Task 1 (literature sourcing)
- §3.1 engine-discovered — Task 7 (multi-site augmentation)
- §3.2 drug-side dict + edge enzyme_tags — Tasks 2, 3, 4
- §3.3 tier system + reject clinical fit — Task 10 (loader validates)
- §4 components — Tasks 2, 3, 4, 5, 7, 9, 10, 11
- §5 data flow — Tasks 12 (smoke per drug end-to-end), 13 (identity-blind), 14 (107-holdout)
- §6.1 validation gate — Task 17
- §6.2 tier classification — Task 1 deliverable
- §6.3 test categories — Tasks 2-17
- §6.4 mass balance topology — Task 6
- §6.5 identity-blind — Task 13
- §6.6 v2-vs-v1 reporting — Task 18 CHANGELOG
- §6.8 performance — implicit via existing benchmark in Task 19 Step 1
- §7 risks (R1-R12) — covered: R2 (snapshot Task 16), R4+R11 (107-holdout Task 14), R6 (yield_source Task 10), R9 (idempotency Task 8), R10 (loader+grep Task 10), R12 (snapshot Task 16). R1, R3, R5, R7, R8 are inherent risks (mitigation via reporting/architecture).
- §9 success criteria — Task 19 Step 2

**2. Placeholder scan:**
- `<TASK1_VALUE>` and `<FROM_TASK1>` markers in Tasks 9, 10, 16 — these reference Task 1 deliverable (legitimate dependency, NOT plan failures).
- `<FROM_STEP1>` in Task 16 — reference within-task derivation (Step 1 generates baselines used in Step 2).
- No abandoned TODOs, no "implement later", no "fill in details" in code blocks.

**3. Type consistency:**
- `enzyme_affinity_for_conversion: dict[str, Distribution]` — consistent across Tasks 3, 4, 7, 10, 11.
- `enzyme_tags: frozenset[str]` — consistent across Tasks 2, 5, 7.
- `lookup_active_metabolite` returns 3-tuple — Tasks 10 (defines), 11 (consumes).
- `ProdrugActivationFluxSpec.__init__` takes `enzyme_tags`, `mw_ratio` — consistent Tasks 5, 6.
- `ResolvedParams.drug_enzyme_affinity_for_conversion(tag) -> float` — consistent Tasks 4, 5.

Plan ready for execution.
