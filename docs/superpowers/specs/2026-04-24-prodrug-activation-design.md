# Prodrug Activation Routing — Design Spec

**Status**: Draft (pending user review)
**Date**: 2026-04-24
**Evidence**: N50 secondary holdout cycle 2026Q2 (FROZEN, commit `b366035`, AAFE 5.249), where sepiapterin 4936× single-handedly dominates. Remdesivir (0.21×), tebipenem_pivoxil (0.10×), fostamatinib (0.27×) also fail because the engine treats parents as the observed species when the clinical reference is the active metabolite.
**Source memory**: `project_engine_improvements_from_n50.md` direction #1.

---

## 1. Goal

Add infrastructure to route Cmax observation to an **active metabolite** species for prodrug drugs, so that engine predictions match clinical reports for the four N50 evidence drugs.

**Out of goal**: Headline AAFE improvement on the 107-holdout (these 4 drugs are not in 107 — N50 only). The validation gate is per-drug fold-error on the 4 evidence drugs (target ≤ 3-fold each). Headline change deferred to 2026Q3 next N50 cycle.

---

## 2. Scope

**In scope**:
- 4 evidence drugs: sepiapterin → BH4, remdesivir → GS-441524, tebipenem_pivoxil → tebipenem, fostamatinib → R406.
- Two-species engine simulation: parent on full graph + active in 1-compartment plasma.
- 1st-order conversion kinetics at a single named site.
- User-provided active metabolite parameters (no SMILES-based ADME prediction for the active).
- SMILES-keyed registry JSON for prodrug activation configs (`data/sbi/prodrug_activation_registry.json`).
- Pipeline-layer applicability-domain adjustment so detected prodrugs with registry entries are no longer marked out-of-domain.

**Out of scope** (deferred / explicit non-goals):
- Multiple sequential active metabolites (e.g., remdesivir → GS-441524 → GS-443902 triphosphate). First step only.
- Multiple parallel active metabolites per parent.
- DDI on the active species (DDI requires enzyme-level CL decomposition; active uses aggregate CL).
- Enterohepatic recirculation of active.
- Class-generic auto-routing (SMARTS → automatic activation). Per Omega dead-end #11 pattern, automated structural overrides are rejected.
- ML/meta-learner retraining for prodrugs.
- N50 re-benchmarking (frozen until 2026Q3).

---

## 3. Architectural Decisions

Five design axes resolved through brainstorming (Q1–Q5) plus two critical resolutions found during architectural review (C1, C2).

| Axis | Decision | Rejected | Rationale |
|---|---|---|---|
| **Q1 Scope** | 4 evidence drugs | Single-drug PoC; SMARTS auto-class | Evidence-driven, extensible YAML, no automation risk |
| **Q2 Representation** | 2-species + user-provided active params | Single-species reparameterization; predicted active SMILES | invariant alignment + endogenous-metabolite ADME unreliable |
| **Q3 Kinetics** | 1st-order, single named site | Michaelis-Menten; enzyme-abundance | Literature data exists for 1st-order, MM data absent |
| **Q4 State space** | 1-compartment active, `_active` suffix | Full 15-organ duplicate; (species, node) tuple | YAGNI sufficient for plasma Cmax target; SimResult contract preserved |
| **Q5 AD flag** | Pipeline-layer interpretation override | SMARTS-stage suppression; new HANDLED flag | Layer boundary preserved; minimal flag surface |
| **C1 Config injection** | SMILES-keyed registry JSON (`data/sbi/prodrug_activation_registry.json`) | Drug YAML upgrade; CLI flag | Sisyphus is SMILES-first; `method_routing.json` precedent |
| **C2 Active elimination edge** | New edge type `OneCompartmentEliminationEdge` | Extend `ClearanceEdge`; merge with prodrug_activation flux | One edge type per kinetic mechanism (existing 6-type pattern) |

### 3.1 Invariant compliance

**Compiler invariant** (CLAUDE.md #8 "Do not modify `engine/compiler.py`, `engine/solver.py`"):
Interpretation: additive changes only, applied symmetrically with the existing "DrugOnGraph existing fields" qualifier in the same invariant. Specifically allowed under this reading:
- New methods on `ResolvedParams` (no modification of existing methods)
- New `isinstance` branches in `_build_edge_params` (existing branches untouched)

Existing logic modification remains forbidden.

**Other invariants verified**:
- #1 Engine identity-blind: ✓ (all logic dispatches on edge type, never on node/drug names)
- #2 All parameters Distribution: ✓ (active fields all Distribution)
- #3 Compile once, parameterize many: ✓ (compile-per-drug; MC samples reuse compiled ODE)
- #4 Flow conservation: ✓ (no new flow edges; conversion is non-flow mass transfer)
- #5 Holdout inviolable: ✓ (no holdout drug touched; 4 evidence drugs are N50)
- #6 No drug-specific branches in engine: ✓ (registry JSON, not in-code branches)
- #7 ≤20 files per directory: ✓ (changes touch 7 files total)

---

## 4. Components

### 4.1 `src/sisyphus/core.py` — additive

```python
@dataclass(frozen=True)
class ActiveMetabolite:
    name: str
    mw: float
    fup: Distribution
    CL_per_h: Distribution
    Vd_L: Distribution
    conversion_rate_per_h: Distribution
    conversion_site: str
    conversion_yield_fraction: Distribution  # default Distribution(mean=1.0, cv=0.0)


@dataclass(frozen=True)
class DrugOnGraph:
    # ... existing 18 fields unchanged ...
    active_metabolite: Optional[ActiveMetabolite] = None
    observation_species: str = "parent"  # "parent" | "active"
```

`__post_init__` validation: see §6.

### 4.2 `src/sisyphus/graph/types.py` — two new edge types

```python
@dataclass(frozen=True)
class ProdrugActivationEdge(Edge):
    """Mass transfer: parent (mg) → active (mg) with MW ratio × yield."""
    conversion_rate: Distribution       # 1/h
    conversion_yield: Distribution      # dimensionless [0, 1]
    mw_parent: float
    mw_active: float
    edge_type: str = field(default="prodrug_activation", init=False)


@dataclass(frozen=True)
class OneCompartmentEliminationEdge(Edge):
    """Aggregate 1st-order elimination: dA/dt = -(CL/Vd) × A."""
    cl_per_h: Distribution
    vd_l: Distribution
    edge_type: str = field(default="one_compartment_elimination", init=False)
```

Both extend `Edge` base; `target` field set to a sentinel for elimination (no real target node; mass leaves the system).

### 4.3 `src/sisyphus/engine/flux.py` — two new FluxSpec classes

```python
@register_flux("prodrug_activation")
class ProdrugActivationFluxSpec(FluxSpec):
    def apply(self, t, y, dydt, params):
        k = params.edge_param(self.edge_id, "conversion_rate")
        y_frac = params.edge_param(self.edge_id, "conversion_yield")
        flux_p = k * y[self.src_idx]               # mg_parent/h
        flux_a = flux_p * self.mw_ratio * y_frac   # mg_active/h
        dydt[self.src_idx] -= flux_p
        dydt[self.tgt_idx] += flux_a


@register_flux("one_compartment_elimination")
class OneCompartmentEliminationFluxSpec(FluxSpec):
    def apply(self, t, y, dydt, params):
        cl = params.edge_param(self.edge_id, "cl_per_h")
        vd = params.edge_param(self.edge_id, "vd_l")
        dydt[self.src_idx] -= (cl / vd) * y[self.src_idx]
```

### 4.4 `src/sisyphus/engine/compiler.py` — additive only

Two new branches in `ResolvedParams._build_edge_params`:

```python
elif isinstance(edge, ProdrugActivationEdge):
    params["conversion_rate"] = edge.conversion_rate.mean
    params["conversion_yield"] = edge.conversion_yield.mean
elif isinstance(edge, OneCompartmentEliminationEdge):
    params["cl_per_h"] = edge.cl_per_h.mean
    params["vd_l"] = edge.vd_l.mean
```

No modification of existing branches or `make_rhs`/`compile`.

### 4.5 `src/sisyphus/graph/builder.py` — augmentation routine

```python
ACTIVE_SUFFIX = "_active"

def augment_for_active_species(graph: BodyGraph, drug: DrugOnGraph) -> BodyGraph:
    if drug.active_metabolite is None:
        return graph
    am = drug.active_metabolite
    # Validations: see §6
    active_node_name = drug.observation_node + ACTIVE_SUFFIX
    graph.add_node(Node(name=active_node_name, node_type="central_compartment",
                        volume=am.Vd_L, ...))
    graph.add_edge(ProdrugActivationEdge(
        source=am.conversion_site, target=active_node_name,
        conversion_rate=am.conversion_rate_per_h,
        conversion_yield=am.conversion_yield_fraction,
        mw_parent=drug.mw, mw_active=am.mw))
    graph.add_edge(OneCompartmentEliminationEdge(
        source=active_node_name, target="__sink__",
        cl_per_h=am.CL_per_h, vd_l=am.Vd_L))
    return graph
```

### 4.6 `src/sisyphus/predict/registry.py` (new) — registry loader

```python
PRODRUG_REGISTRY_PATH = Path("data/sbi/prodrug_activation_registry.json")

def load_active_metabolite_for_smiles(smiles: str) -> Optional[ActiveMetabolite]:
    """Look up canonical SMILES in registry; return ActiveMetabolite or None."""
    canonical = canonicalize_smiles(smiles)
    registry = _load_registry()
    entry = registry.get(canonical)
    if entry is None:
        return None
    return ActiveMetabolite(
        name=entry["name"],
        mw=entry["mw"],
        fup=Distribution(**entry["fup"]),
        CL_per_h=Distribution(**entry["CL_per_h"]),
        Vd_L=Distribution(**entry["Vd_L"]),
        conversion_rate_per_h=Distribution(**entry["conversion_rate_per_h"]),
        conversion_site=entry["conversion_site"],
        conversion_yield_fraction=Distribution(**entry["conversion_yield_fraction"]),
    )
```

`predict.build_drug_on_graph` (or whatever the actual entry function is named — see §8) calls this and assigns to `DrugOnGraph.active_metabolite`. If the registry returns non-None, `observation_species` defaults to `"active"`.

### 4.7 `src/sisyphus/pipeline/*.py` — AD adjustment + observation routing

```python
def _resolve_observation_node(drug: DrugOnGraph) -> str:
    if drug.active_metabolite is not None and drug.observation_species == "active":
        return drug.observation_node + ACTIVE_SUFFIX
    return drug.observation_node

def _adjust_ad_for_prodrug(drug, ad_flags):
    warnings = []
    flags_for_domain = list(ad_flags)
    if "PRODRUG" in ad_flags and drug.active_metabolite is not None:
        warnings.append(
            f"Prodrug {drug.name!r} routed via activation to "
            f"{drug.active_metabolite.name!r} at {drug.active_metabolite.conversion_site}.")
        flags_for_domain = [f for f in ad_flags if f != "PRODRUG"]
    in_domain = len(flags_for_domain) == 0
    return in_domain, warnings
```

Location pin: `src/sisyphus/pipeline/config.py` (PK calls `_resolve_observation_node`; pipeline orchestrator calls `_adjust_ad_for_prodrug`).

### 4.8 `data/sbi/prodrug_activation_registry.json` (new) — 4 entries

Schema:
```json
{
  "<canonical_smiles>": {
    "name": "BH4",
    "mw": 241.25,
    "fup": {"mean": 0.23, "cv": 0.3},
    "CL_per_h": {"mean": 40, "cv": 0.35},
    "Vd_L": {"mean": 150, "cv": 0.3},
    "conversion_rate_per_h": {"mean": 12, "cv": 0.4},
    "conversion_site": "enterocyte",
    "conversion_yield_fraction": {"mean": 0.85, "cv": 0.1}
  },
  ...
}
```

Literature collection per drug ~30 min (4 × 30 = 2 hours), to be done during plan execution.

### 4.9 File change summary

| File | Type | LoC estimate |
|---|---|---|
| `src/sisyphus/core.py` | additive | +40 |
| `src/sisyphus/graph/types.py` | additive | +25 |
| `src/sisyphus/engine/flux.py` | additive | +50 |
| `src/sisyphus/engine/compiler.py` | additive (2 branches) | +6 |
| `src/sisyphus/graph/builder.py` | new function | +80 |
| `src/sisyphus/predict/registry.py` | new file | +60 |
| `src/sisyphus/pipeline/config.py` | additive (2 functions) | +30 |
| `data/sbi/prodrug_activation_registry.json` | new file (4 entries) | +60 |

**Existing-line modifications: 0.**

---

## 5. Data Flow

End-to-end with sepiapterin as worked example.

```
SMILES (sepiapterin)
    │
    ▼
predict.build_drug_on_graph:
    chemistry.py → SMARTS detection → "PRODRUG" flag
    adme.py → parent fup/CL/Vd/Kp predicted
    registry.load_active_metabolite_for_smiles(canonical_smiles)
        → ActiveMetabolite(name="BH4", mw=241.25, ...)
    DrugOnGraph(..., active_metabolite=<above>, observation_species="active")
    │
    ▼
graph pipeline:
    build_from_yaml(reference_man.yaml) → 15-node BodyGraph
    augment_for_active_species(graph, drug):
        validate "enterocyte" ∈ graph.nodes
        validate "venous_blood_active" ∉ graph.nodes
        add Node("venous_blood_active", volume=Vd_L)
        add ProdrugActivationEdge(enterocyte → venous_blood_active, k, yield, MWs)
        add OneCompartmentEliminationEdge(venous_blood_active → sink, CL, Vd)
    → 16-node augmented graph
    │
    ▼
ODECompiler.compile (per-drug):
    state_index sorted alphabetically: ..., venous_blood: 14, venous_blood_active: 15
    flux_specs built via FLUX_REGISTRY (existing 6 types + prodrug_activation + one_compartment_elimination)
    → CompiledODE (16 states)
    │
    ▼
MC loop (n_samples=1000):
    sampled_drug.active_metabolite — all Distributions resampled
    ResolvedParams._build_edge_params caches:
        new ProdrugActivationEdge → {conversion_rate, conversion_yield} (mean)
        new OneCompartmentEliminationEdge → {cl_per_h, vd_l} (mean)
    rhs(t, y) evaluated:
        existing 6 flux types operate on parent states 0–14 (unchanged)
        ProdrugActivationFluxSpec: dydt[enterocyte] -= flux_p; dydt[v_b_active] += flux_a
        OneCompartmentEliminationFluxSpec: dydt[v_b_active] -= (CL/Vd) × y
    SimResult.concentrations:
        "venous_blood": [...]            (parent, mg/L)
        "venous_blood_active": [...]      (active = BH4, mg/L)
        ... (parent organ trajectories)
    │
    ▼
PK endpoints:
    obs_node = _resolve_observation_node(drug)  → "venous_blood_active"
    PKEndpoints(Cmax, Tmax, AUC, ...) computed from concentrations[obs_node]
    │
    ▼
Pipeline assembly:
    engine_pk = PKEndpoints (BH4 plasma)
    ml_pk = ml.predict(parent_smiles)    (unchanged ML track)
    meta_pk = meta_learner.combine(...)  (standard weights, no override)
    in_domain, warnings = _adjust_ad_for_prodrug(drug, ad_flags)
        → in_domain=True (PRODRUG removed from AD judgment)
        → warnings += ["Prodrug sepiapterin routed via activation to BH4 at enterocyte."]

PredictionResult(
    pk=meta_pk,
    method="hybrid" (existing logic),
    in_applicability_domain=True,
    ad_flags=[..., "PRODRUG", ...],   (preserved for audit trail)
    warnings=[..., "Prodrug routed..."],
    cmax_90ci=(lo, hi))
```

### 5.1 Mass balance invariant

Active mass produced should equal parent mass dissipated × (MW_active/MW_parent) × yield:

```
∫₀^T flux_active(t) dt / MW_active  ≈  yield × ∫₀^T flux_parent(t) dt / MW_parent
```

Tolerance < 1e-3 (verified in integration test).

### 5.2 Backward compatibility (active_metabolite=None)

For all 107-holdout drugs and existing training data:
- registry lookup returns None → `active_metabolite=None`, `observation_species="parent"` (default)
- `augment_for_active_species` early returns → graph unchanged (15 nodes)
- compile produces 15 states (unchanged)
- RHS uses only existing 6 flux types
- pk uses observation_node="venous_blood" (existing default)
- AD adjustment skipped (no PRODRUG flag, or PRODRUG without active_metabolite leaves in_domain=False as today)
- PredictionResult: identical to current behavior

107-holdout regression tolerance: AAFE delta < 1% of MC=1000 baseline (2.719).

### 5.3 ML track (explicit non-modification)

ML track receives parent SMILES, predicts Cmax via existing XGBoost ensemble. For prodrugs in MMPK training data, label Cmax is the clinical (active) value, so ML has implicitly learned a parent-SMILES → active-Cmax mapping. Generalization to new prodrugs from this mapping is unreliable but consistent with prior model.

Meta-learner combines engine_pk and ml_pk with standard weights (`_W_VDSS=0.20`, base 0.60/0.40/0.00). No method override for prodrugs in this spec. If validation fails (§7.5), `data/sbi/method_routing.json` per-drug routing override is the contingency mechanism (consistent with morphine SBI routing precedent).

---

## 6. Error Handling

### 6.1 Build-time hard failures (`ValueError`)

| Condition | Location | Message template |
|---|---|---|
| Registry parse error (missing required field) | registry loader | `"prodrug_activation_registry entry for SMILES {s} missing field {f}"` |
| `conversion_rate_per_h.mean ≤ 0` | registry loader | `"conversion_rate must be positive, got {x}"` |
| `conversion_yield_fraction.mean ∉ [0, 1]` | registry loader | `"conversion_yield must be in [0, 1], got {x}"` |
| `mw_active ≤ 0` or `mw_parent ≤ 0` | registry loader | `"MW must be positive"` |
| `CL_per_h.mean ≤ 0` or `Vd_L.mean ≤ 0` | registry loader | `"CL and Vd must be positive"` |
| `observation_species="active"` with `active_metabolite=None` | `DrugOnGraph.__post_init__` | `"observation_species='active' requires active_metabolite config"` |
| `observation_species ∉ {"parent","active"}` | `__post_init__` | `"observation_species must be 'parent' or 'active', got {x}"` |
| `conversion_site ∉ graph.nodes` | `augment_for_active_species` | `"conversion_site={x!r} not in graph nodes {sorted(nodes)}"` |
| Active node name collision (`{obs}_active` already exists) | augment | `"active node name collision: {x!r}"` |
| Non-canonical SMILES key in registry | registry loader | `"registry key must be canonical SMILES"` |

### 6.2 AD-flag soft warnings (in_applicability_domain=True with warnings)

| Condition | Warning |
|---|---|
| PRODRUG motif + registry match | `"Prodrug {drug.name!r} routed via activation to {active.name!r} at {site}."` |
| Registry match but no PRODRUG motif (non-structural activation) | `"Active metabolite declared for {drug.name!r} without structural prodrug motif; registry override applied."` |
| `observation_species="parent"` with `active_metabolite is not None` | `"Prodrug simulated but observation routed to parent; active trajectory available in SimResult.concentrations[{active_node}]."` |
| `conversion_yield_fraction.mean < 0.5` | `"Low activation yield ({y:.2f}); partial conversion dominates."` |
| `conversion_rate_per_h.mean < 0.1` | `"Slow conversion (k={k} 1/h); single-compartment approximation may lose fidelity."` |

PRODRUG motif **without** registry match: existing behavior (out-of-domain, low confidence). Unchanged.

### 6.3 Run-time solver failures (existing mechanism)

`solve_ivp` failure → `SimResult.solver_success=False` → `PredictionResult.confidence="low"` (no change).

Stiffness risk: sepiapterin k=12/h (conversion) vs k_el=CL/Vd≈0.27/h (elimination), ratio 44. BDF/LSODA solver should handle. CI smoke test enforces wall-time < 10s per prodrug.

### 6.4 Mass balance drift

Existing `SimResult.mass_balance_error` is parent-only. To verify 2-species mass balance without modifying `engine/solver.py`, post-hoc verification at pipeline layer:

```python
def _verify_two_species_mass_balance(sim_result, drug):
    if drug.active_metabolite is None:
        return  # parent-only, existing path
    parent_dissipated = ...  # from concentrations[conversion_site]
    active_produced = ...    # from concentrations[active_node]
    expected = parent_dissipated * (mw_a/mw_p) * yield_mean
    if abs(active_produced - expected) / expected > 1e-3:
        # log warning, do not fail
```

### 6.5 Orthogonality matrix (SMARTS detection × registry match)

| SMARTS PRODRUG | Registry match | Result |
|---|---|---|
| Yes | Yes | Active routing + PRODRUG flag retained (audit) + AD upgraded + warning |
| Yes | No | Existing: AD=False, confidence=low |
| No | Yes | Active routing + non-structural-activation warning |
| No | No | Existing parent-only path |

---

## 7. Testing Strategy

41 new tests across 5 categories.

### 7.1 Unit tests (~25, `tests/unit/`)

- `test_active_metabolite.py` — dataclass + `__post_init__` validation
- `test_prodrug_activation_flux.py` — `ProdrugActivationFluxSpec.apply` math
- `test_one_compartment_elim_flux.py` — `OneCompartmentEliminationFluxSpec.apply`
- `test_augment_active_species.py` — builder augmentation idempotency, validations, node/edge counts
- `test_prodrug_registry_loader.py` — JSON parse, validation, canonicalization
- `test_pipeline_ad_adjustment.py` — `_adjust_ad_for_prodrug`, `_resolve_observation_node`
- `test_compiler_extension.py` — additive `_build_edge_params` branches

PASS criterion: 100% pytest pass, ruff clean.

### 7.2 Integration tests (~6, `tests/integration/`)

- `test_two_species_mass_balance.py` — synthetic prodrug, analytical 2-compartment solution comparison, tolerance < 1e-3
- `test_compile_per_drug.py` — augment + compile produces 16-state CompiledODE
- `test_resample_propagates_active.py` — MC=100 resample preserves Distribution semantics on all active fields
- `test_observation_routing.py` — `observation_species` correctly routes PK extraction
- `test_clearance_separation.py` — parent enzyme_affinity changes don't affect active Cmax (aggregate CL isolation)
- `test_backward_compat_no_active.py` — 30 non-prodrug drugs, ε-identical (1e-9) to current SimResult

### 7.3 Smoke tests (~3, `tests/smoke/`)

- `test_sepiapterin_pipeline.py` — full SMILES → PredictionResult, wall time < 10s (MC=100)
- `test_prodrug_4drugs_smoke.py` — all 4 evidence drugs run without error
- `test_solver_stiffness.py` — k=12/k_el=0.27 stiffness handled

### 7.4 Regression / Backward compat (~3, `tests/regression/`)

- `test_107_holdout_aafe_unchanged.py` — MC=1000 full benchmark, AAFE 2.719 ± 1% MC tolerance
- `test_existing_drug_yaml_unchanged.py` — concentration key sets identical for non-prodrug drugs
- `test_meta_weights_unchanged.py` — meta-learner weights `_W_VDSS=0.20`, base 0.60/0.40/0.00 unchanged

### 7.5 Validation — 4 evidence drug scientific gates (~4, `tests/validation/`)

| Drug | Active | Pre-spec FE | Post-spec target | Tolerance |
|---|---|---|---|---|
| Sepiapterin | BH4 | 4936× | within 3-fold | per-drug |
| Remdesivir | GS-441524 | 4.76× | within 3-fold | per-drug |
| Tebipenem_pivoxil | Tebipenem | 10× | within 3-fold | per-drug |
| Fostamatinib | R406 | 3.7× | within 3-fold | per-drug |

PASS criterion: all 4 within 3-fold of clinical reference active Cmax.

**Failure response**:
1. Re-verify registry literature values (k, yield, CL, Vd, fup).
2. If registry values are best-available, add per-drug routing override in `data/sbi/method_routing.json` to force engine weight=1.0.
3. If still failing, re-open spec.

### 7.6 CI integration

`.github/workflows/ci.yml` extension:
- 7.1–7.3, 7.5 on every push (~5 minutes)
- 7.4 `test_107_holdout_aafe_unchanged.py` nightly only (~70 minutes, MC=1000)
- 7.5 acts as merge gate (block if any of 4 drugs fails 3-fold).

---

## 8. Pre-implementation Verifications

Before plan writing, the following must be resolved (open at spec time):

1. **Predict entry point**: actual function name and signature for `build_drug_on_graph` or equivalent (`smiles_to_drug_on_graph`?). Adjust §4.6 accordingly.
2. **`reference_man.yaml` actual node names**: confirm "enterocyte" exists, or substitute the correct name (`gut_wall`, `intestinal_epithelium`, etc.). All 4 drugs' `conversion_site` values resolved against actual graph.
3. **`ClearanceFluxSpec` model variants**: confirm existing supported model strings to ensure `OneCompartmentEliminationEdge`/Spec is a non-overlapping addition.
4. **Pipeline compile hookpoint**: locate where `ODECompiler.compile` is called in pipeline orchestrator; insert `augment_for_active_species` call there.
5. **`clinical_pk.json` species alignment**: confirm 4-drug Cmax reference values are for active species. If parent values, source new active reference data into `data/reference/prodrug_active_cmax.json`.
6. **MC resample mechanism**: confirm uncertainty MC loop auto-discovers Distribution fields on `DrugOnGraph` (otherwise active_metabolite Distributions need explicit registration).
7. **`method_routing.json` loader pattern**: reuse for prodrug registry loader.
8. **`add_node`/`add_edge` API on BodyGraph**: confirm builder signatures for `augment_for_active_species`.
9. **Node type taxonomy**: confirm valid `node_type` value for the active 1-compartment plasma node (`central_compartment`, `tissue`, `blood_pool`, or a new type to introduce). The §4.5 placeholder `node_type="central_compartment"` is illustrative.
10. **Clearance sink convention**: confirm how existing `ClearanceEdge` represents elimination-to-environment — real `sink` node, sentinel target string, or no target at all (FluxSpec only decrements source). The §4.3 `OneCompartmentEliminationFluxSpec.apply` writes only `dydt[src_idx]`; whether `from_edge` must resolve `tgt_idx` depends on the existing pattern.

These are read-only verifications. Each takes ~5–10 minutes during plan writing.

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Registry literature values wrong → validation fail | Med | High | 4-drug validation gate (§7.5) catches it; literature re-verification + method_routing override; spec reopen as last resort |
| Stiff ODE on fast-conversion + slow-elimination → solver timeout | Low | Med | Smoke test wall-time gate; BDF/LSODA solver; CV reduction if needed |
| Meta-learner combination yields worse than engine-alone for prodrugs | Med | Low | method_routing.json override path documented; not a blocker for spec ship |
| Compiler additive changes interpretation rejected by reviewer | Low | High | C1=(a) reasoning documented in §3.1; alternative C1=(b)/(c) noted in critique log |
| 107-holdout regression > 1% (silent breakage of non-prodrug path) | Low | High | Nightly regression CI gate (§7.4); active_metabolite=None path verified ε-identical (§7.2 backward compat test) |
| N50 cycle 2026Q3 still doesn't show net AAFE improvement (non-prodrug N50 errors dominate) | High | Low | Headline AAFE not the success metric; per-drug fold-error is. §1 goal explicitly excludes headline. |
| Drug YAML/registry conflict if both paths exist later | Low | Low | C1=(ii) chosen specifically to avoid this; registry is single source for activation config |

---

## 10. Success Criteria

Spec ships successfully if and only if:

1. **All 4 evidence drugs**: clinical active Cmax predicted within 3-fold (§7.5).
2. **107-holdout AAFE**: 2.719 ± 1% (no regression on non-prodrug, §7.4).
3. **All 41 new tests pass** (§7.1–7.5).
4. **No existing test regressions**.
5. **All 8 pre-implementation verifications resolved** during plan writing (§8).
6. **Mass balance invariant** holds (§5.1, integration test in §7.2).
7. **No modification of existing logic** in `engine/compiler.py` or `engine/solver.py` (additive branches only, §3.1).

Failure on any criterion blocks merge. Per-drug failure on §7.5 may be remediated via `method_routing.json` override (existing precedent) without re-opening spec.

---

## 11. Out-of-Scope Future Work (post-spec roadmap notes)

Not part of this spec, listed for future planning context:

- **Multiple sequential active metabolites** (e.g., remdesivir → GS-441524 → GS-443902): would require species pipeline of arbitrary length.
- **DDI on active species**: requires enzyme-level CL decomposition for active. Reuses parent's `enzyme_affinity` machinery on a new ActiveMetabolite enzyme dict.
- **Active tissue distribution**: full multi-compartment active for drugs where active has its own significant tissue binding. Likely requires species-aware compiler or full graph duplicate.
- **Prodrug class generalization**: SMARTS → registry auto-population from a curated metabolite database. Currently rejected per Omega dead-end #11.
- **N50 retire and 2026Q3 cycle**: per spec §7 of N50, current cycle is FROZEN. New N50 in 2026Q3 will validate spec impact.

---

## Revision History

| Date | Change |
|---|---|
| 2026-04-24 | Initial draft. Brainstorming Q1–Q5 + critical review C1, C2 + 5 design sections. |
