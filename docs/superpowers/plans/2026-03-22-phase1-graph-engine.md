# Phase 1: Graph Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the graph-based ODE engine: physiology YAML + compound YAML → BodyGraph → compile → solve → Cmax, matching Omega ODE output ±5% for midazolam, caffeine, warfarin, propranolol.

**Architecture:** BodyGraph (from YAML) is compiled into an ODE skeleton by ODECompiler. The skeleton maps each edge to a FluxSpec that computes mass transfer rates. ResolvedParams provides point-value parameter lookups from sampled graph + drug. scipy's `solve_ivp` (LSODA) integrates the RHS. PK endpoints (Cmax, AUC) are extracted from SimResult.

**Tech Stack:** Python 3.10+, numpy, scipy, pyyaml, pytest, dataclasses

**Omega target values to match (±5%):**

| Drug | Dose | Omega Cmax (mg/L) | Omega Tmax (h) |
|------|------|-------------------|----------------|
| Midazolam | 2 mg oral | 0.006943 | 1.5 |
| Caffeine | 100 mg oral | 1.7139 | 1.0 |
| Warfarin | 10 mg oral | 0.4922 | 3.0 |
| Propranolol | 80 mg oral | 0.1355 | 1.5 |

---

## File Structure

### Files to modify (Phase 0 skeleton → Phase 1 implementation):
- `src/sisyphus/core.py` — Implement `Distribution.sample()`, add `particle_radius_um` and `ps_overrides` to `DrugOnGraph`
- `src/sisyphus/graph/types.py` — Add `ka_fraction` to `AbsorptionEdge`, add `ivive_scaling` to `Node`
- `src/sisyphus/graph/body.py` — Implement `add_node`, `add_edge`, `remove_node`, `validate`, `sample`
- `src/sisyphus/graph/builder.py` — Implement `build_from_yaml`, `merge_overlay`, `_parse_distribution`
- `src/sisyphus/engine/compiler.py` — Implement `ODECompiler.compile()`, `CompiledODE.make_rhs()`, `ResolvedParams`
- `src/sisyphus/engine/flux.py` — Add 5 concrete FluxSpec implementations
- `src/sisyphus/engine/solver.py` — Implement `solve()` wrapping scipy
- `src/sisyphus/pk/endpoints.py` — Implement `compute_endpoints()`
- `src/sisyphus/pk/nca.py` — Implement `auc_trapezoidal()`, `terminal_half_life()`

### Files to create:
- `src/sisyphus/compounds.py` — Compound YAML → DrugOnGraph loader
- `data/physiology/reference_man.yaml` — ICRP Reference Man (70 kg, 34-node graph)
- `data/compounds/midazolam.yaml`
- `data/compounds/caffeine.yaml`
- `data/compounds/warfarin.yaml`
- `data/compounds/propranolol.yaml`
- `tests/unit/test_distribution.py`
- `tests/unit/test_body_graph.py`
- `tests/unit/test_builder.py`
- `tests/unit/test_compiler.py`
- `tests/unit/test_flux.py`
- `tests/unit/test_solver.py`
- `tests/unit/test_endpoints.py`
- `tests/integration/test_engine_validation.py`

---

## Reference Data: Omega's 35-State Model (for Sisyphus graph)

### Nodes (34 — excludes SC depot from Omega's 35)

**Blood pools:** venous_blood (3.7 L), arterial_blood (1.5 L), portal_vein (0.05 L)

**Perfusion-limited organs (11):**

| Node | Volume (L) | Flow fraction | Portal? | Enzymes |
|------|-----------|---------------|---------|---------|
| lung | 0.50 | 1.0 (full CO) | no | — |
| brain | 1.45 | 0.12 | no | — |
| heart | 0.33 | 0.04 | no | — |
| kidney | 0.31 | 0.19 | no | — |
| liver | 1.80 | 0.065 (HA only) | receives portal | CYP3A4, CYP2D6, CYP1A2, CYP2C9, CYP2E1 |
| spleen | 0.15 | 0.03 | yes | — |
| gut_wall | 1.03 | 0.15 | yes | CYP3A4 |
| pancreas | 0.10 | 0.01 | yes | — |
| thymus | 0.02 | 0.002 | no | — |
| reproductive | 0.04 | 0.002 | no | — |
| rest | 2.50 | 0.069 | no | — |

**Permeability-limited organs (4 × 2 nodes = 8):**

| Organ | Total Vol (L) | Flow fraction | Default PS (L/h) |
|-------|--------------|---------------|-------------------|
| adipose | 14.5 | 0.052 | 10.0 |
| muscle | 28.0 | 0.17 | 10.0 |
| bone | 4.86 | 0.05 | 10.0 |
| skin | 3.30 | 0.05 | 10.0 |

Each becomes `{organ}_vasc` (blood_pool) + `{organ}_tissue` (organ) nodes.

**GI lumen (8):** stomach, duodenum, jejunum1, jejunum2, ileum1, ileum2, ileum3, colon

| Segment | Transit time (h) | Ka fraction |
|---------|-----------------|-------------|
| stomach | 0.25 | 0.0 |
| duodenum | 0.26 | 1.0 |
| jejunum1 | 0.475 | 1.0 |
| jejunum2 | 0.475 | 1.0 |
| ileum1 | 0.68 | 0.8 |
| ileum2 | 0.68 | 0.6 |
| ileum3 | 0.68 | 0.3 |
| colon | 13.5 | 0.05 |

**Sinks (4):** metabolized_hepatic, excreted_renal, metabolized_gut, excreted_fecal

### Flow fractions verification

Non-lung arterial fractions: 0.12 + 0.04 + 0.19 + 0.065 + 0.03 + 0.15 + 0.01 + 0.002 + 0.002 + 0.069 + 0.052 + 0.17 + 0.05 + 0.05 = **1.0** ✓

Portal organs (gut_wall + spleen + pancreas) flow = 0.15 + 0.03 + 0.01 = 0.19 CO → portal_vein → liver

Total liver inflow = HA (0.065) + portal (0.19) = 0.255 CO = 99.45 L/h

### Tissue composition (Rodgers & Rowland 2006)

| Tissue | fn | fp | fw | pH |
|--------|------|------|------|------|
| lung | 0.0030 | 0.0128 | 0.811 | 7.00 |
| brain | 0.0391 | 0.0550 | 0.620 | 7.00 |
| heart | 0.0117 | 0.0166 | 0.758 | 7.00 |
| kidney | 0.0121 | 0.0242 | 0.783 | 7.00 |
| liver | 0.0348 | 0.0252 | 0.751 | 7.00 |
| spleen | 0.0077 | 0.0113 | 0.788 | 7.00 |
| gut_wall | 0.0163 | 0.0185 | 0.718 | 7.00 |
| pancreas | 0.0348 | 0.0252 | 0.751 | 7.00 |
| thymus | 0.0132 | 0.0100 | 0.700 | 7.00 |
| reproductive | 0.0132 | 0.0100 | 0.700 | 7.00 |
| rest | 0.0132 | 0.0100 | 0.700 | 7.00 |
| adipose | 0.7021 | 0.0022 | 0.150 | 7.00 |
| muscle | 0.0238 | 0.0072 | 0.760 | 7.00 |
| bone | 0.0174 | 0.0010 | 0.439 | 7.00 |
| skin | 0.0284 | 0.0111 | 0.718 | 7.00 |
| plasma | 0.0023 | 0.0199 | 0.945 | 7.40 |

### IVIVE constants

- MPPGL (liver): 45 mg/g
- MPPI (gut): ~20 mg/g
- Liver weight: 1500 g → total microsomal protein = 67,500 mg
- Gut weight: 1000 g → total microsomal protein = 20,000 mg
- ivive_scaling (global): 60 / 1e6 = 6×10⁻⁵ (converts µL/min to L/h)

### Omega ODE equations (key patterns)

**Perfusion-limited organ:**
```
c_out = y[idx] × rbp / (volume × kp)
dydt[idx] = Q × c_art − Q × c_out
```

**Permeability-limited organ:**
```
cu_vasc = fup × c_vasc / rbp
cu_extra = fup × c_extra / kp
dydt[idx_v] = Q × c_art − Q × c_vasc_out − PS × (cu_vasc − cu_extra)
dydt[idx_e] = PS × (cu_vasc − cu_extra)
```

**Liver (well-stirred):**
```
c_liver_in = (q_ha × c_art + q_portal × c_pv) / q_total
clh = (q_total × fup × clint) / (q_total + fup × clint)
met_rate = clh × c_liver_in
dydt[liver] = q_ha×c_art + q_portal×c_pv − q_total×c_out − met_rate
```

**GI absorption:**
```
ka = 2.88 × peff × ka_fraction / particle_radius_um  (h⁻¹)
absorption = ka × y[segment]
transit_out = (1/transit_time) × y[segment]
dydt[segment] = −absorption − transit_out + transit_in
dydt[gut_wall] += absorption
```

---

## Task 1: Distribution — Implement Sampling

**Files:**
- Modify: `src/sisyphus/core.py`
- Create: `tests/unit/test_distribution.py`

- [ ] **Step 1: Write failing tests for Distribution.sample()**

```python
# tests/unit/test_distribution.py
import numpy as np
import pytest
from sisyphus.core import Distribution


class TestDistribution:
    def test_deterministic_returns_mean(self):
        d = Distribution(mean=5.0, cv=0.0)
        rng = np.random.default_rng(42)
        assert d.sample(rng) == 5.0

    def test_lognormal_positive(self):
        d = Distribution(mean=5.0, cv=0.3, dist_type="lognormal")
        rng = np.random.default_rng(42)
        samples = [d.sample(rng) for _ in range(1000)]
        assert all(s > 0 for s in samples)
        assert abs(np.mean(samples) - 5.0) / 5.0 < 0.1  # within 10% of mean

    def test_normal_sampling(self):
        d = Distribution(mean=100.0, cv=0.1, dist_type="normal")
        rng = np.random.default_rng(42)
        samples = [d.sample(rng) for _ in range(1000)]
        assert abs(np.mean(samples) - 100.0) < 5.0
        assert abs(np.std(samples) - 10.0) < 3.0

    def test_std_property(self):
        d = Distribution(mean=100.0, cv=0.1)
        assert d.std == pytest.approx(10.0)

    def test_frozen(self):
        d = Distribution(mean=1.0)
        with pytest.raises(AttributeError):
            d.mean = 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_distribution.py -v`
Expected: FAIL on `NotImplementedError`

- [ ] **Step 3: Implement Distribution.sample()**

In `src/sisyphus/core.py`, replace the `raise NotImplementedError` in `sample()`:

```python
def sample(self, rng: np.random.Generator) -> float:
    if self.cv == 0.0:
        return self.mean
    sigma = self.cv * abs(self.mean)
    if self.dist_type == "lognormal":
        # Parameterize so E[X] = mean, CV = cv
        mu_ln = np.log(self.mean**2 / np.sqrt(sigma**2 + self.mean**2))
        sigma_ln = np.sqrt(np.log(1 + (sigma / self.mean) ** 2))
        return float(rng.lognormal(mu_ln, sigma_ln))
    elif self.dist_type == "normal":
        return float(rng.normal(self.mean, sigma))
    elif self.dist_type == "uniform":
        half = sigma * np.sqrt(3)  # uniform with same std
        return float(rng.uniform(self.mean - half, self.mean + half))
    raise ValueError(f"Unknown dist_type: {self.dist_type}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_distribution.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/core.py tests/unit/test_distribution.py
git commit -m "feat(core): implement Distribution.sample() with lognormal/normal/uniform"
```

---

## Task 2: Skeleton Modifications — Add Phase 1 Fields

**Files:**
- Modify: `src/sisyphus/core.py` — Add `particle_radius_um`, `ps_overrides` to DrugOnGraph
- Modify: `src/sisyphus/graph/types.py` — Add `ka_fraction` to AbsorptionEdge, `ivive_scaling` to Node

- [ ] **Step 1: Add fields to DrugOnGraph in core.py**

After `renal_clearance` field, add:

```python
    # Formulation (absorption model)
    particle_radius_um: float = 25.0

    # Permeability-surface area overrides for perm-limited organs
    ps_overrides: dict[str, Distribution] = field(default_factory=dict)
```

Note: `particle_radius_um` has a default so it goes after required fields. Move `renal_clearance` (no default) above the default fields block, or give `ps_overrides` and `particle_radius_um` sensible defaults.

- [ ] **Step 2: Add ka_fraction to AbsorptionEdge in graph/types.py**

```python
@dataclass
class AbsorptionEdge(Edge):
    ka_fraction: Distribution = field(default_factory=lambda: Distribution(1.0))
    def __post_init__(self) -> None:
        self.edge_type = "absorption"
```

- [ ] **Step 3: Add ivive_scaling to Node**

```python
@dataclass
class Node:
    name: str
    node_type: str
    volume: Distribution
    composition: TissueComposition | None = None
    enzymes: dict[str, Distribution] = field(default_factory=dict)
    transporters: dict[str, Distribution] = field(default_factory=dict)
    ivive_scaling: float = 0.0  # MPPGL × organ_weight × 60/1e6 for enzyme-bearing nodes
```

- [ ] **Step 4: Run existing tests + ruff**

Run: `ruff check src/ && pytest tests/ -v`
Expected: All pass (no existing tests broken)

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/core.py src/sisyphus/graph/types.py
git commit -m "feat(core,graph): add Phase 1 fields — ka_fraction, ivive_scaling, ps_overrides"
```

---

## Task 3: BodyGraph — Core Operations

**Files:**
- Modify: `src/sisyphus/graph/body.py`
- Create: `tests/unit/test_body_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_body_graph.py
import numpy as np
import pytest
from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import Edge, FlowEdge, Node


class TestBodyGraph:
    def test_add_node(self):
        g = BodyGraph()
        n = Node(name="a", node_type="organ", volume=Distribution(1.0))
        g.add_node(n)
        assert "a" in g.nodes
        assert g.nodes["a"] is n

    def test_add_duplicate_node_raises(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
        with pytest.raises(ValueError, match="duplicate"):
            g.add_node(Node(name="a", node_type="organ", volume=Distribution(2.0)))

    def test_add_edge(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
        g.add_node(Node(name="b", node_type="organ", volume=Distribution(1.0)))
        e = FlowEdge(source="a", target="b", edge_type="flow",
                     flow_rate=Distribution(10.0))
        g.add_edge(e)
        assert len(g.edges) == 1

    def test_add_edge_missing_node_raises(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
        with pytest.raises(ValueError, match="not found"):
            g.add_edge(FlowEdge(source="a", target="z", edge_type="flow",
                                flow_rate=Distribution(10.0)))

    def test_remove_node(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ", volume=Distribution(1.0)))
        g.add_node(Node(name="b", node_type="organ", volume=Distribution(1.0)))
        g.add_edge(FlowEdge(source="a", target="b", edge_type="flow",
                            flow_rate=Distribution(10.0)))
        g.remove_node("a")
        assert "a" not in g.nodes
        assert len(g.edges) == 0  # edge removed too

    def test_sample(self):
        g = BodyGraph()
        g.add_node(Node(name="a", node_type="organ",
                        volume=Distribution(1.0, cv=0.1)))
        g.global_params["co"] = Distribution(390.0, cv=0.1)
        rng = np.random.default_rng(42)
        g2 = g.sample(rng)
        assert g2.nodes["a"].volume.cv == 0.0  # sampled = deterministic
        assert g2.global_params["co"].cv == 0.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_body_graph.py -v`

- [ ] **Step 3: Implement BodyGraph methods**

Implement `add_node`, `add_edge`, `remove_node`, `validate` (flow conservation check), and `sample` in `body.py`. Key logic for `validate`:

```python
def validate(self) -> list[str]:
    errors = []
    # Check all edge endpoints exist
    for e in self.edges:
        if e.source not in self.nodes:
            errors.append(f"Edge source '{e.source}' not found")
        if e.target not in self.nodes:
            errors.append(f"Edge target '{e.target}' not found")
    # Flow conservation: for each non-lung node, sum(inflow) ≈ sum(outflow)
    # (Implementation: group FlowEdges by node, compare in vs out)
    return errors
```

For `sample`: recursively sample all Distribution fields in nodes, edges, and global_params. Return a new BodyGraph with cv=0 everywhere.

- [ ] **Step 4: Run tests — all pass**

Run: `pytest tests/unit/test_body_graph.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/graph/body.py tests/unit/test_body_graph.py
git commit -m "feat(graph): implement BodyGraph add/remove/validate/sample"
```

---

## Task 4: YAML Data Files

**Files:**
- Create: `data/physiology/reference_man.yaml`
- Create: `data/compounds/midazolam.yaml`
- Create: `data/compounds/caffeine.yaml`
- Create: `data/compounds/warfarin.yaml`
- Create: `data/compounds/propranolol.yaml`

- [ ] **Step 1: Create reference_man.yaml**

This is the full physiology specification. Must include all 34 nodes, all edges (flow, transit, absorption, clearance, diffusion), tissue compositions, enzyme abundances, and GI parameters. Use the reference data tables above. All flow fractions from Omega's body.py.

Key structure:
```yaml
# data/physiology/reference_man.yaml
description: "ICRP Reference Man, 70 kg, 30 y, healthy"
source: "ICRP 2002, Davies & Morris 1993, Rodgers & Rowland 2006"

global_params:
  cardiac_output: {mean: 390.0, cv: 0.10, unit: L/h}
  body_weight: {mean: 70.0, cv: 0.0, unit: kg}
  hematocrit: {mean: 0.45, cv: 0.0}
  ivive_scaling: 0.00006  # 60/1e6: converts µL/min → L/h

nodes:
  venous_blood:
    type: blood_pool
    volume: {mean: 3.7}
  arterial_blood:
    type: blood_pool
    volume: {mean: 1.5}
  lung:
    type: organ
    volume: {mean: 0.50}
    composition: {fn: 0.0030, fp: 0.0128, fw: 0.811, pH: 7.00}
  # ... all 34 nodes with volumes, compositions, enzymes ...
  liver:
    type: organ
    volume: {mean: 1.80}
    composition: {fn: 0.0348, fp: 0.0252, fw: 0.751, pH: 7.00}
    enzymes:
      CYP3A4: {mean: 9247500}   # 137 pmol/mg × 45 mg/g × 1500 g = total pmol
      CYP2D6: {mean: 675000}    # 10 × 45 × 1500
      CYP1A2: {mean: 3037500}   # 45 × 45 × 1500
      CYP2C9: {mean: 6480000}   # 96 × 45 × 1500
      CYP2E1: {mean: 3307500}   # 49 × 45 × 1500
  gut_wall:
    type: barrier_organ
    volume: {mean: 1.03}
    composition: {fn: 0.0163, fp: 0.0185, fw: 0.718, pH: 7.00}
    enzymes:
      CYP3A4: {mean: 600000}    # ~30 pmol/mg × 20 mg/g × 1000 g

edges:
  # Pulmonary circulation
  - {source: venous_blood, target: lung, type: flow, flow_fraction: 1.0}
  - {source: lung, target: arterial_blood, type: flow, flow_fraction: 1.0}
  # Systemic distribution — one pair per organ
  - {source: arterial_blood, target: brain, type: flow, flow_fraction: 0.12}
  - {source: brain, target: venous_blood, type: flow}
  # ... etc for all organs ...
  # Portal drainage
  - {source: gut_wall, target: portal_vein, type: flow}
  - {source: spleen, target: portal_vein, type: flow}
  - {source: pancreas, target: portal_vein, type: flow}
  - {source: portal_vein, target: liver, type: flow}
  - {source: arterial_blood, target: liver, type: flow, flow_fraction: 0.065}
  - {source: liver, target: venous_blood, type: flow}
  # Permeability-limited (diffusion)
  - {source: adipose_vasc, target: adipose_tissue, type: diffusion, ps_product: {mean: 10.0}}
  # GI transit
  - {source: stomach_lumen, target: duodenum_lumen, type: transit, transit_rate: {mean: 4.0}}
  # Absorption
  - {source: duodenum_lumen, target: gut_wall, type: absorption, ka_fraction: {mean: 1.0}}
  # Clearance
  - {source: liver, target: metabolized_hepatic, type: clearance, model: well_stirred}
  - {source: gut_wall, target: metabolized_gut, type: clearance, model: well_stirred}
  - {source: kidney, target: excreted_renal, type: clearance, model: gfr_filtration}
  # Fecal
  - {source: colon_lumen, target: excreted_fecal, type: transit, transit_rate: {mean: 0.074}}
```

Transit rates = 1/transit_time: stomach 1/0.25=4.0, duodenum 1/0.26=3.846, jejunum1 1/0.475=2.105, etc.

- [ ] **Step 2: Create compound YAML files**

Each compound YAML contains the DrugOnGraph parameters. Per-enzyme affinities are back-calculated from Omega's CLint values:

```
affinity = (clint_organ × fm_enzyme) / (enzyme_abundance_total × ivive_scaling)
```

For midazolam CYP3A4 in liver:
```
affinity = (750 × 0.93) / (9247500 × 6e-5) = 697.5 / 554.85 = 1.257 µL/min/pmol
```

Format:
```yaml
# data/compounds/midazolam.yaml
name: Midazolam
smiles: "Clc1ccc2c(c1)C(=NCc3nccn3C)c1cc(F)ccc1N2"
mw: 325.77
pka: 6.15
compound_type: base
dose_mg: 2.0
route: oral

fup: 0.032
rbp: 0.66
kp_method: provided

enzyme_affinity:  # µL/min/pmol — back-calculated to match Omega CLint
  CYP3A4: 1.257
  CYP3A5: 0.741

kp_overrides:
  lung: 0.6
  brain: 6.61
  # ... all organs ...

ps_overrides:
  adipose: 8.0
  muscle: 25.0
  bone: 5.0
  skin: 8.0

peff: 5.37
solubility: 0.024
particle_radius_um: 25.0
renal_clearance: 0.0
```

Repeat for caffeine, warfarin, propranolol using Omega values from the reference tables.

- [ ] **Step 3: Commit**

```bash
git add data/physiology/reference_man.yaml data/compounds/
git commit -m "data: add reference_man.yaml and 4 compound configs"
```

---

## Task 5: YAML Builder + Compound Loader

**Files:**
- Modify: `src/sisyphus/graph/builder.py`
- Create: `src/sisyphus/compounds.py`
- Create: `tests/unit/test_builder.py`

- [ ] **Step 1: Write failing tests for builder**

```python
# tests/unit/test_builder.py
from pathlib import Path
import pytest
from sisyphus.graph.builder import build_from_yaml
from sisyphus.graph.body import BodyGraph


class TestBuilder:
    def test_build_reference_man(self):
        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        assert isinstance(g, BodyGraph)
        assert len(g.nodes) == 34
        assert "liver" in g.nodes
        assert "venous_blood" in g.nodes
        assert g.nodes["liver"].enzymes  # has enzymes

    def test_flow_conservation(self):
        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        errors = g.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_cardiac_output_in_global_params(self):
        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        assert "cardiac_output" in g.global_params
        co = g.global_params["cardiac_output"]
        assert co.mean == pytest.approx(390.0)
```

- [ ] **Step 2: Write failing tests for compound loader**

```python
# tests/unit/test_builder.py (continued)
from sisyphus.compounds import load_compound
from sisyphus.core import DrugOnGraph


class TestCompoundLoader:
    def test_load_midazolam(self):
        drug = load_compound(Path("data/compounds/midazolam.yaml"))
        assert isinstance(drug, DrugOnGraph)
        assert drug.name == "Midazolam"
        assert drug.dose_mg == 2.0
        assert drug.fup.mean == pytest.approx(0.032)
        assert "CYP3A4" in drug.enzyme_affinity

    def test_load_caffeine(self):
        drug = load_compound(Path("data/compounds/caffeine.yaml"))
        assert drug.fup.mean == pytest.approx(0.65)
        assert drug.compound_type == "neutral"
```

- [ ] **Step 3: Implement builder.py**

Parse YAML using `pyyaml`. For each node entry, construct a `Node`. For each edge entry, construct the appropriate Edge subclass. Convert `flow_fraction` to absolute `flow_rate` by multiplying with `cardiac_output`. Call `validate()` at the end.

Key flow fraction → absolute flow conversion:
```python
co = global_params["cardiac_output"].mean
for edge in edges:
    if hasattr(edge, "flow_rate") and "flow_fraction" in raw_edge:
        edge.flow_rate = Distribution(raw_edge["flow_fraction"] * co)
```

For return-flow edges (organ → venous_blood or portal_vein): infer flow_rate = same as the incoming flow.

- [ ] **Step 4: Implement compounds.py**

```python
# src/sisyphus/compounds.py
def load_compound(path: Path) -> DrugOnGraph:
    """Load a compound YAML and return a DrugOnGraph."""
    # Parse YAML, construct Distribution for each parameter
    # Map enzyme_affinity dict entries to Distribution
    # Map kp_overrides, ps_overrides
    ...
```

- [ ] **Step 5: Run tests — all pass**

Run: `pytest tests/unit/test_builder.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/graph/builder.py src/sisyphus/compounds.py tests/unit/test_builder.py
git commit -m "feat(graph): implement YAML builder and compound loader"
```

---

## Task 6: ODE Compiler + ResolvedParams

**Files:**
- Modify: `src/sisyphus/engine/compiler.py`
- Create: `tests/unit/test_compiler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_compiler.py
import numpy as np
import pytest
from sisyphus.core import Distribution
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import FlowEdge, Node


def make_two_node_graph() -> BodyGraph:
    """Minimal graph: A → B with flow."""
    g = BodyGraph()
    g.add_node(Node(name="a", node_type="blood_pool", volume=Distribution(1.0)))
    g.add_node(Node(name="b", node_type="organ", volume=Distribution(2.0)))
    g.add_edge(FlowEdge(source="a", target="b", edge_type="flow",
                        flow_rate=Distribution(10.0)))
    g.global_params["cardiac_output"] = Distribution(390.0)
    return g


class TestODECompiler:
    def test_compile_assigns_states(self):
        g = make_two_node_graph()
        compiler = ODECompiler()
        compiled = compiler.compile(g)
        assert compiled.n_states == 2
        assert "a" in compiled.state_index
        assert "b" in compiled.state_index

    def test_compile_produces_callable_rhs(self):
        g = make_two_node_graph()
        compiler = ODECompiler()
        compiled = compiler.compile(g)
        # Need a minimal DrugOnGraph and ResolvedParams to test make_rhs
        # (Details in Step 3)
```

- [ ] **Step 2: Run tests to verify failure**

- [ ] **Step 3: Implement ODECompiler.compile()**

```python
def compile(self, graph: BodyGraph) -> CompiledODE:
    compiled = CompiledODE()
    # Assign one state per node
    for i, name in enumerate(sorted(graph.nodes.keys())):
        compiled.state_index[name] = i
    compiled.n_states = len(graph.nodes)
    # Map each edge to its FluxSpec via FLUX_REGISTRY
    for edge_id, edge in enumerate(graph.edges):
        flux_cls = FLUX_REGISTRY[edge.edge_type]
        spec = flux_cls.from_edge(edge_id, edge, compiled.state_index)
        compiled.flux_specs.append(spec)
    return compiled
```

Implement `CompiledODE.make_rhs()`:
```python
def make_rhs(self, params: ResolvedParams):
    def rhs(t, y):
        dydt = np.zeros(self.n_states)
        for spec in self.flux_specs:
            spec.apply(t, y, dydt, params)
        return dydt
    return rhs
```

- [ ] **Step 4: Implement ResolvedParams**

Construct from realized (sampled) BodyGraph + DrugOnGraph. Store pre-computed lookups for fast access:

```python
class ResolvedParams:
    def __init__(self, graph, drug):
        self._volumes = {n: node.volume.mean for n, node in graph.nodes.items()}
        self._enzymes = {n: {t: d.mean for t, d in node.enzymes.items()}
                         for n, node in graph.nodes.items()}
        self._drug = drug
        self._kp = self._compute_kp(graph, drug)
        # Pre-compute total inflow per node from flow edges
        self._inflows = self._compute_inflows(graph)
```

- [ ] **Step 5: Run tests — pass**

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/engine/compiler.py tests/unit/test_compiler.py
git commit -m "feat(engine): implement ODECompiler, CompiledODE, ResolvedParams"
```

---

## Task 7: Flux Implementations

**Files:**
- Modify: `src/sisyphus/engine/flux.py`
- Create: `tests/unit/test_flux.py`

This is the core physics. 5 FluxSpec subclasses, each registered in FLUX_REGISTRY.

- [ ] **Step 1: Write failing tests for FlowFluxSpec**

```python
# tests/unit/test_flux.py
import numpy as np
from sisyphus.engine.flux import FLUX_REGISTRY

class TestFlowFlux:
    def test_registered(self):
        assert "flow" in FLUX_REGISTRY

    def test_mass_conservation(self):
        """Flow from A to B: A loses what B gains."""
        # Set up 2-node system, apply flux, check dydt sums to 0
        ...
```

- [ ] **Step 2: Implement FlowFluxSpec**

```python
@register_flux("flow")
class FlowFluxSpec(FluxSpec):
    """Convective transport: dA/dt = Q × (C_in − C_out)"""
    def __init__(self, edge_id, source_idx, target_idx, source_name, target_name):
        super().__init__(edge_id, source_idx, target_idx)
        self.source_name = source_name
        self.target_name = target_name

    @classmethod
    def from_edge(cls, edge_id, edge, state_index):
        return cls(edge_id, state_index[edge.source], state_index[edge.target],
                   edge.source, edge.target)

    def apply(self, t, y, dydt, params):
        q = params.edge_param(self.edge_id, "flow_rate")
        v_source = params.node_param(self.source_name, "volume")
        c_source = y[self.source_idx] / v_source

        # For tissue nodes, outflow conc corrected by Kp and RBP
        kp = params.drug_kp(self.source_name)
        rbp = params.drug_param("rbp")
        if kp > 0 and kp != 1.0:
            c_out = y[self.source_idx] * rbp / (v_source * kp)
        else:
            c_out = c_source

        dydt[self.source_idx] -= q * c_out
        dydt[self.target_idx] += q * c_out
```

- [ ] **Step 3: Implement ClearanceFluxSpec**

```python
@register_flux("clearance")
class ClearanceFluxSpec(FluxSpec):
    """Enzyme-mediated elimination (well-stirred or GFR)."""
    def __init__(self, edge_id, source_idx, target_idx, source_name, model):
        super().__init__(edge_id, source_idx, target_idx)
        self.source_name = source_name
        self.model = model

    @classmethod
    def from_edge(cls, edge_id, edge, state_index):
        return cls(edge_id, state_index[edge.source], state_index[edge.target],
                   edge.source, edge.model)

    def apply(self, t, y, dydt, params):
        if self.model == "well_stirred":
            # CLint = Σ(abundance × affinity) × ivive_scaling
            clint_organ = 0.0
            ivive = params.global_param("ivive_scaling")
            for tag, abundance in params.node_enzymes(self.source_name).items():
                affinity = params.drug_enzyme_affinity(tag)
                if affinity > 0 and abundance > 0:
                    clint_organ += abundance * affinity * ivive
            fup = params.drug_param("fup")
            q = params.total_inflow(self.source_name)
            clh = (q * fup * clint_organ) / max(q + fup * clint_organ, 1e-12)
            v = params.node_param(self.source_name, "volume")
            kp = params.drug_kp(self.source_name)
            rbp = params.drug_param("rbp")
            c_in = y[self.source_idx] * rbp / (v * kp)
            rate = clh * c_in
        elif self.model == "gfr_filtration":
            renal_cl = params.drug_param("renal_clearance")
            v = params.node_param(self.source_name, "volume")
            kp = params.drug_kp(self.source_name)
            rbp = params.drug_param("rbp")
            c_plasma = y[self.source_idx] * rbp / (v * kp)
            rate = renal_cl * c_plasma
        else:
            raise ValueError(f"Unknown clearance model: {self.model}")
        dydt[self.source_idx] -= rate
        dydt[self.target_idx] += rate
```

- [ ] **Step 4: Implement TransitFluxSpec**

```python
@register_flux("transit")
class TransitFluxSpec(FluxSpec):
    """First-order transit: rate = k × A_source"""
    # transit_rate from edge, applied as first-order transfer
```

- [ ] **Step 5: Implement AbsorptionFluxSpec**

```python
@register_flux("absorption")
class AbsorptionFluxSpec(FluxSpec):
    """Drug absorption from lumen to tissue."""
    # ka = 2.88 × peff × ka_fraction / particle_radius_um
    # absorption = ka × y[source]
```

- [ ] **Step 6: Implement DiffusionFluxSpec**

```python
@register_flux("diffusion")
class DiffusionFluxSpec(FluxSpec):
    """PS-limited exchange between vascular and tissue compartments."""
    # PS from edge or drug ps_overrides
    # cu_vasc = fup * c_vasc / rbp
    # cu_extra = fup * c_extra / kp
    # flux = PS * (cu_vasc - cu_extra)
```

- [ ] **Step 7: Run all flux tests**

Run: `pytest tests/unit/test_flux.py -v`

- [ ] **Step 8: Commit**

```bash
git add src/sisyphus/engine/flux.py tests/unit/test_flux.py
git commit -m "feat(engine): implement 5 FluxSpec types — flow, clearance, transit, absorption, diffusion"
```

---

## Task 8: ODE Solver Wrapper

**Files:**
- Modify: `src/sisyphus/engine/solver.py`
- Create: `tests/unit/test_solver.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_solver.py
import numpy as np
from sisyphus.core import Distribution, SimResult

class TestSolver:
    def test_simple_decay(self):
        """Single compartment with first-order elimination."""
        # Build a 2-node graph (organ + sink), compile, solve
        # Verify exponential decay: C(t) = C0 * exp(-k*t)
        ...

    def test_mass_balance(self):
        """Total mass (all compartments) = dose at all times."""
        ...
```

- [ ] **Step 2: Implement solver**

```python
def solve(compiled, params, y0, t_span, t_eval=None) -> SimResult:
    rhs = compiled.make_rhs(params)
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = scipy.integrate.solve_ivp(
        rhs, t_span, y0, method="LSODA",
        t_eval=t_eval, rtol=1e-8, atol=1e-10
    )
    # Convert solution to SimResult
    concentrations = {}
    amounts = {}
    for name, idx in compiled.state_index.items():
        amounts[name] = sol.y[idx]
        v = params.node_param(name, "volume")
        kp = params.drug_kp(name)
        rbp = params.drug_param("rbp")
        concentrations[name] = sol.y[idx] * rbp / (v * kp) if v > 0 else sol.y[idx]
    # Mass balance check
    total = sum(sol.y[i] for i in range(compiled.n_states))
    dose = params.drug_param("dose_mg")
    mbe = float(np.max(np.abs(total - dose) / max(dose, 1e-12)))
    return SimResult(
        time_h=sol.t, concentrations=concentrations,
        amounts=amounts, mass_balance_error=mbe,
        solver_success=sol.success
    )
```

- [ ] **Step 3: Run tests — pass**

- [ ] **Step 4: Commit**

```bash
git add src/sisyphus/engine/solver.py tests/unit/test_solver.py
git commit -m "feat(engine): implement ODE solver wrapper with LSODA and mass balance check"
```

---

## Task 9: PK Endpoints

**Files:**
- Modify: `src/sisyphus/pk/endpoints.py`
- Modify: `src/sisyphus/pk/nca.py`
- Create: `tests/unit/test_endpoints.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_endpoints.py
import numpy as np
import pytest
from sisyphus.core import SimResult
from sisyphus.pk.endpoints import compute_endpoints
from sisyphus.pk.nca import auc_trapezoidal

class TestEndpoints:
    def test_cmax_from_known_profile(self):
        """Known peak at t=2, C=10."""
        time = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        conc = np.array([0, 5, 10, 7, 4, 2], dtype=float)
        result = SimResult(
            time_h=time,
            concentrations={"venous_blood": conc},
            amounts={"venous_blood": conc * 3.7},
            mass_balance_error=0.0, solver_success=True
        )
        pk = compute_endpoints(result)
        assert pk.cmax.mean == pytest.approx(10.0)
        assert pk.tmax.mean == pytest.approx(2.0)

    def test_auc_trapezoidal(self):
        time = np.array([0, 1, 2], dtype=float)
        conc = np.array([0, 10, 0], dtype=float)
        auc = auc_trapezoidal(time, conc)
        assert auc == pytest.approx(10.0)  # triangle
```

- [ ] **Step 2: Implement nca.py (auc_trapezoidal, terminal_half_life)**

```python
def auc_trapezoidal(time, conc):
    return float(np.trapz(conc, time))
```

- [ ] **Step 3: Implement endpoints.py**

```python
def compute_endpoints(result, observation_node="venous_blood"):
    conc = result.concentrations[observation_node]
    time = result.time_h
    cmax = float(np.max(conc))
    tmax = float(time[np.argmax(conc)])
    auc = auc_trapezoidal(time, conc)
    t_half = terminal_half_life(time, conc)
    return PKEndpoints(
        cmax=Distribution(cmax), tmax=Distribution(tmax),
        auc_0t=Distribution(auc),
        t_half=Distribution(t_half) if t_half else None,
    )
```

- [ ] **Step 4: Run tests — pass**

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/pk/endpoints.py src/sisyphus/pk/nca.py tests/unit/test_endpoints.py
git commit -m "feat(pk): implement Cmax, AUC, t½ extraction from SimResult"
```

---

## Task 10: Integration Validation — 4 Drugs vs Omega ±5%

**Files:**
- Create: `tests/integration/test_engine_validation.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_engine_validation.py
from pathlib import Path
import pytest
from sisyphus.graph.builder import build_from_yaml
from sisyphus.compounds import load_compound
from sisyphus.engine.compiler import ODECompiler
from sisyphus.engine.solver import solve
from sisyphus.pk.endpoints import compute_endpoints
import numpy as np


OMEGA_TARGETS = {
    "midazolam": {"cmax": 0.006943, "tmax": 1.5, "dose": 2.0},
    "caffeine":  {"cmax": 1.7139,   "tmax": 1.0, "dose": 100.0},
    "warfarin":  {"cmax": 0.4922,   "tmax": 3.0, "dose": 10.0},
    "propranolol": {"cmax": 0.1355, "tmax": 1.5, "dose": 80.0},
}


def run_drug(drug_name: str):
    """Full pipeline: YAML → BodyGraph → compile → solve → PKEndpoints."""
    graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
    drug = load_compound(Path(f"data/compounds/{drug_name}.yaml"))

    compiler = ODECompiler()
    compiled = compiler.compile(graph)

    # Sample to point values (deterministic, cv=0)
    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = drug.sample(rng)

    from sisyphus.engine.compiler import ResolvedParams
    params = ResolvedParams(realized_graph, realized_drug)

    # Initial conditions: dose in administration node
    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg

    result = solve(compiled, params, y0, t_span=(0, 24), t_eval=np.linspace(0, 24, 1000))
    pk = compute_endpoints(result, observation_node="venous_blood")
    return pk


class TestEngineValidation:
    @pytest.mark.parametrize("drug", OMEGA_TARGETS.keys())
    def test_cmax_within_5pct(self, drug):
        pk = run_drug(drug)
        target = OMEGA_TARGETS[drug]["cmax"]
        actual = pk.cmax.mean
        rel_error = abs(actual - target) / target
        assert rel_error < 0.05, (
            f"{drug}: Cmax={actual:.6f}, target={target:.6f}, error={rel_error:.1%}"
        )

    @pytest.mark.parametrize("drug", OMEGA_TARGETS.keys())
    def test_mass_balance(self, drug):
        """Mass balance error < 1e-6."""
        graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        drug_obj = load_compound(Path(f"data/compounds/{drug}.yaml"))
        # ... run solver and check mass_balance_error < 1e-6
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/integration/test_engine_validation.py -v`

If any drug exceeds ±5%, debug by:
1. Comparing individual flux contributions
2. Checking CLint values match Omega
3. Verifying flow fractions and volumes
4. Adjusting enzyme abundances/affinities

- [ ] **Step 3: Iterate on calibration until all 4 drugs pass ±5%**

This may require adjusting:
- Enzyme abundances in `reference_man.yaml`
- Per-drug affinities in compound YAMLs
- Gut CYP3A4 abundance (to match gut first-pass)
- Absorption model constants

- [ ] **Step 4: Final commit**

```bash
git add tests/integration/test_engine_validation.py
git commit -m "test(integration): validate 4 drugs ±5% vs Omega ODE"
```

- [ ] **Step 5: Tag release**

```bash
git tag v0.1.0
```

---

## Dependency Graph

```
Task 1 (Distribution) ──► Task 3 (BodyGraph) ──► Task 5 (Builder+Loader)
                           │                       │
Task 2 (Skeleton mods) ────┘                       │
                                                   ▼
Task 4 (YAML data) ──────────────────────► Task 6 (Compiler)
                                                   │
                                                   ▼
                                           Task 7 (Flux impls)
                                                   │
                                                   ▼
                                           Task 8 (Solver)
                                                   │
                                                   ▼
                                           Task 9 (PK endpoints)
                                                   │
                                                   ▼
                                           Task 10 (Integration)
```

Tasks 1+2 can run in parallel. Task 4 (data files) is independent of code tasks.
Tasks 7's 5 flux implementations are independent of each other.
