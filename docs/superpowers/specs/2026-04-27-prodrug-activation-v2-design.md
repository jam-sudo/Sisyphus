# Prodrug Activation v2 — Enzyme-Abundance Mechanistic Design

**Date:** 2026-04-27
**Author:** Hypatia (via brainstorming skill)
**Predecessor:** v1 spec `docs/superpowers/specs/2026-04-24-prodrug-activation-design.md` (commit `40e15f2`); v1 merged `af9d2be` (2026-04-26)
**Status:** Design complete, awaiting user review before plan creation

---

## 0. Context

v1 (commits `ee3704e`..`a2ad03e`, merged 2026-04-26) shipped infrastructure with kinetic 1st-order conversion (`rate = k × A_parent`). Mass balance verified, but 4-drug 3-fold validation gate failed:

| Drug | v1 Predicted | Reference | Fold-error |
|---|---|---|---|
| sepiapterin | 12.85 mg/L | 0.0024 mg/L | 5356× |
| tebipenem_pivoxil | 0.46 | 4.01 | 8.63× |
| fostamatinib | 0.13 | 0.61 | 4.78× |
| remdesivir | 0.98 | 4.38 | 4.45× |

**v1 root cause** (diagnosed): gut_wall flow-through residence τ ≈ V/Q = 1.03 L / 58 L·h⁻¹ ≈ 64 s. First-order `conversion_rate_per_h = 12` converts only 1 − exp(−k·τ) ≈ 19% per pass. Sepiapterin's ~99.99% pre-systemic loss requires k ≈ 500/h — order-of-magnitude beyond conservative literature values. **The 1st-order rate-constant abstraction at flow-through compartments cannot capture fast first-pass extraction efficiently**.

v2 reformulates conversion as well-stirred extraction (Q-bounded), parallel to Sisyphus's existing CYP3A4 elimination pattern. This addresses the kinetic limitation **and** replaces v1's per-drug `conversion_rate_per_h` knob with mechanistic enzyme-abundance × drug-affinity values (literature-derived).

---

## 1. Goal

Replace v1's phenomenological per-drug rate constant with a mechanistic enzyme-abundance × drug-affinity × well-stirred extraction model. Enable predictive (not fitted) Cmax for new prodrugs whose activation enzyme affinity is independently measured.

**Mechanistic promise**: affinity values are NOT back-fit to clinical data — they come from in vitro literature or substrate-class kinetics. Validation failure is informative (where literature/architecture is insufficient), not project-failing.

Passing the 4-drug 3-fold validation gate is preferred but **NOT required** for v2 to ship. The hard requirement is the mechanistic-sourcing promise.

---

## 2. Scope

**In v2 scope:**
- Engine reformulation: `ProdrugActivationFluxSpec` rewritten as well-stirred extraction (replaces kinetic 1st-order math).
- Drug API: new `DrugOnGraph.enzyme_affinity_for_conversion: dict[str, Distribution]` field (additive).
- Physiology YAML: enzyme abundance entries for SPR, CES1, CES2, ALPI at relevant nodes.
- Registry schema: replace `conversion_rate_per_h` + `conversion_site` with `enzyme_affinity_for_conversion` dict + `affinity_source` enum + `yield_source` enum.
- Augmentation: multi-site discovery (any node with enzyme_tags ∩ drug.affinity_keys creates conversion edge).
- Tests: mass balance well-stirred analytical, identity-blind rename, multi-site discovery, DDI smoke, per-prodrug snapshot.

**Out of v2 scope** (see §8 for v3+ deferrals):
- First-pass metabolism of active species.
- ML predictor for activation enzyme affinity.
- Conversion enzyme genetic polymorphism.
- Special populations.
- Multi-step activation cascade.
- Reversible conversion.
- Tier 3 (infrastructure-only / fitted) registry entries.
- DDI clinical scenarios beyond a single smoke test.

---

## 3. Architectural decisions

### 3.1 Conversion site discovery: engine-discovered (Q1=C)

Drugs declare `enzyme_affinity_for_conversion: dict[enzyme_tag → Distribution]`. Augmentation iterates physiology nodes, creates one `ProdrugActivationEdge` per node where `enzyme_tags ∩ node.enzymes` is non-empty. No `conversion_site` string per drug.

**Rationale:**
- Mechanistic — where conversion happens is determined by where the enzyme exists (biology).
- Architecturally consistent with Sisyphus's elimination pattern (CYP3A4 abundance distributed via physiology, drug declares per-CYP affinity, engine sums abundance × affinity at each node).
- DDI / polymorphism / induction inherit naturally (abundance × inhibition factor applies uniformly across all sites).
- Generalizes — new prodrug = register affinity, sites discovered automatically. CLAUDE.md "pick the choice that generalizes".

**Reject A (single-site v1-style)**: forces ad-hoc lumping decision for biologically multi-site activation (sepiapterin SPR is liver+gut+kidney; remdesivir CES1 is liver+intestine).
**Reject B (multi-site explicit list)**: strict subset of C; exposes anatomy-knowledge to user unnecessarily.

### 3.2 Drug-side affinity field: separate dict (Q2=b refined)

`DrugOnGraph.enzyme_affinity_for_conversion: dict[str, Distribution]` (new, additive). Keeps v1's `enzyme_affinity` (elimination CYPs) untouched. Edge stores `enzyme_tags: frozenset[str]` (structure only); affinity values remain on the drug (single source of truth).

**Rationale:**
- Same enzyme catalyzes activation for one substrate, elimination for another (CES1: remdesivir = activation, methylphenidate = elimination). Routing decision belongs at the (drug, enzyme) pair, not at the enzyme alone.
- **Reject (a) single-dict + enzyme-level routing flag**: biologically unsound — enzymes cannot be globally classified as activation vs. elimination.
- **Reject (c) dict + activation_enzymes set**: coordination-prone (set out of sync with dict).

### 3.3 Affinity values sourcing: literature only, no clinical fit (Q3=A refined)

Three tiers:
- **tier 1 (literature)** — directly measured in vitro Vmax/Km for this exact prodrug-enzyme pair.
- **tier 2 (class_extrapolated)** — estimated from the enzyme's substrate-class kinetics (e.g., ALPI activity on phosphate monoesters). Wider CV (≈1.0–1.5).
- **tier 3 (infrastructure_only / clinical fit)** — **excluded from v2 registry** (loader rejects). Future v3 may enable.

**Reject pure clinical back-fit**: would reproduce v1's per-drug-knob criticism in different clothing. v2's mechanistic claim requires affinity values be independent of the clinical data we then predict. Aligns with Sisyphus invariant #8 ("no fudge to Cmax loss") and CLAUDE.md "pick the choice that generalizes".

**Failure mode is informative, not catastrophic.** If literature mean × abundance does not predict clinical Eg within 3-fold, the gap reveals which input is wrong (literature value, abundance estimate, IVIVE scaling, missing enzyme contribution). Refit is forbidden.

---

## 4. Components

### 4.1 Physiology YAML (`data/physiology/reference_man.yaml`)

Add enzyme abundance entries (all values **placeholder, TBD by Plan Task 0 literature search**):

```yaml
- name: liver
  enzymes:
    CYP3A4, CYP2D6, ...                         # existing
    SPR:  {mean: <TBD>, cv: <TBD>}              # NEW
    CES1: {mean: <TBD>, cv: <TBD>}              # NEW
    CES2: {mean: <TBD>, cv: <TBD>}              # NEW

- name: gut_wall
  enzymes:
    CYP3A4: 21224338                            # existing
    SPR:  {mean: <TBD>, cv: <TBD>}              # NEW
    CES1: {mean: <TBD>, cv: <TBD>}              # NEW
    CES2: {mean: <TBD>, cv: <TBD>}              # NEW
    ALPI: {mean: <TBD>, cv: <TBD>}              # NEW

- name: kidney
  enzymes:
    SPR: {mean: <TBD>, cv: <TBD>}               # NEW (minor; defer if validation passes without)
```

**Format:** existing `{mean, cv}` dict pattern. No `correlation_group` for new enzymes (independent lognormal — Achour matrix does not cover SPR/CES/ALPI).
**Sanity check (math feasibility, kept for transparency):** sepiapterin Eg ≈ 99.99% requires `abundance × affinity × ivive ≈ 6.5e9` (for liver). With reasonable SPR kcat/Km ≈ 50000 µL·min⁻¹·pmol⁻¹ and abundance ≈ 1e5 pmol/liver, the architecture is mathematically capable of this Eg. No architecture-level barrier — only literature-input correctness determines validation outcome.

### 4.2 Drug data structure (`src/sisyphus/core.py`)

Add ONE field to `DrugOnGraph` (additive — invariant #8 preserved):

```python
@dataclass(frozen=True)
class DrugOnGraph:
    # ... 19 existing fields including v1's active_metabolite, observation_species ...
    enzyme_affinity_for_conversion: dict[str, Distribution] = field(default_factory=dict)  # NEW

    def __post_init__(self) -> None:
        # ... existing validation ...
        if self.enzyme_affinity_for_conversion and self.active_metabolite is None:
            raise ValueError(
                "enzyme_affinity_for_conversion non-empty requires active_metabolite to be set"
            )
```

`ActiveMetabolite` (v1 dataclass): UNCHANGED.

### 4.3 Edge type (`src/sisyphus/graph/types.py`)

Modify v1's `ProdrugActivationEdge` (replace `conversion_rate` with `enzyme_tags`):

```python
@dataclass(frozen=True)
class ProdrugActivationEdge(Edge):
    edge_type: str = field(default="prodrug_activation", init=False)
    enzyme_tags: frozenset[str] = field(default_factory=frozenset)         # NEW (replaces conversion_rate)
    conversion_yield: Distribution = field(default_factory=lambda: Distribution(1.0))  # KEEP
    mw_parent: float = 0.0                                                 # KEEP (validates >0 in from_edge)
    mw_active: float = 0.0                                                 # KEEP
    # extraction_model field NOT added (YAGNI; well_stirred hardcoded in flux)
```

`OneCompartmentEliminationEdge`: UNCHANGED.

**Internal API break (R10)**: `conversion_rate` field removed. Grep for direct construction outside augmentation/tests; migrate as part of v2.

### 4.4 Flux (`src/sisyphus/engine/flux.py`)

Rewrite `ProdrugActivationFluxSpec.apply()` body (~50 LoC). Mirror `ClearanceFluxSpec(model="well_stirred")` exactly, but route flux to active node (not sink) with MW × yield scaling:

```python
@register_flux("prodrug_activation")
class ProdrugActivationFluxSpec(FluxSpec):
    def __init__(self, edge_id, source_idx, target_idx, source_name, target_name,
                 enzyme_tags, mw_ratio):
        super().__init__(edge_id, source_idx, target_idx, source_name, target_name)
        self.enzyme_tags = enzyme_tags    # frozenset[str], compile-time
        self.mw_ratio = mw_ratio          # float, compile-time

    @classmethod
    def from_edge(cls, edge_id, edge, state_index):
        if edge.mw_parent <= 0:
            raise ValueError(f"mw_parent must be positive, got {edge.mw_parent}")
        return cls(
            edge_id, state_index[edge.source], state_index[edge.target],
            edge.source, edge.target,
            enzyme_tags=edge.enzyme_tags,
            mw_ratio=edge.mw_active / edge.mw_parent,
        )

    def apply(self, t, y, dydt, params):
        # well-stirred CLint computation — identity-blind
        clint_organ = 0.0
        ivive = params.node_param(self.source_name, "ivive_scaling")
        node_enzymes = params.node_enzymes(self.source_name)
        for tag in self.enzyme_tags:
            abundance = node_enzymes.get(tag, 0.0)
            affinity = params.drug_enzyme_affinity_for_conversion(tag)   # NEW lookup
            if affinity > 0 and abundance > 0:
                clint_organ += abundance * affinity * ivive

        if clint_organ <= 0:
            return

        fup = params.drug_param("fup")
        q = params.total_inflow(self.source_name)
        denom = q + fup * clint_organ
        if denom < 1e-12:
            return
        cl_organ = (q * fup * clint_organ) / denom

        v = params.node_param(self.source_name, "volume")
        kp = params.drug_kp(self.source_name)
        rbp = params.drug_param("rbp")
        c_out = y[self.source_idx] * rbp / (v * kp) if v > 0 else 0.0

        rate_parent = cl_organ * c_out
        y_frac = params.edge_param(self.edge_id, "conversion_yield")
        rate_active = rate_parent * self.mw_ratio * y_frac

        dydt[self.source_idx] -= rate_parent
        dydt[self.target_idx] += rate_active

@register_flux("one_compartment_elimination")
class OneCompartmentEliminationFluxSpec(FluxSpec):
    pass  # UNCHANGED
```

### 4.5 ResolvedParams (`src/sisyphus/engine/compiler.py`)

Add ONE method (additive):

```python
def drug_enzyme_affinity_for_conversion(self, tag: str) -> float:
    """Lookup drug.enzyme_affinity_for_conversion[tag] sample value.
    Returns 0.0 if tag absent (graceful, mirrors drug_enzyme_affinity behavior)."""
    ...
```

Existing `ResolvedParams` methods: UNCHANGED. Sample resolution per MC iteration uses standard pattern (resample `drug.enzyme_affinity_for_conversion` dict like other Distribution dicts).

### 4.6 Augmentation (`src/sisyphus/graph/builder.py`)

Rewrite `augment_for_active_species`:

```python
def augment_for_active_species(graph, drug, observation_node="venous_blood"):
    if drug.active_metabolite is None:
        return graph

    am = drug.active_metabolite
    affinities = drug.enzyme_affinity_for_conversion
    enzyme_tags = frozenset(affinities.keys())

    # Tier-1+2 enforcement: registry loader rejects empty affinities for active_metabolite drugs.
    # Defensive check here in case of direct construction.
    if not enzyme_tags:
        raise ValueError("active_metabolite present but enzyme_affinity_for_conversion empty")

    # Add active species 1C node
    active_node_name = observation_node + ACTIVE_SUFFIX
    active_node = Node(name=active_node_name, node_type="blood_pool", volume=am.Vd_L, ...)
    graph_aug = graph.add_node(active_node)

    # Multi-site discovery: any node with enzyme intersection
    conversion_sites = [
        node_name for node_name, node in graph_aug.nodes.items()
        if node.enzymes and (enzyme_tags & set(node.enzymes.keys()))
    ]

    if not conversion_sites:
        raise ValueError(
            f"No conversion site: drug declares enzyme_tags={enzyme_tags} "
            f"but no node in physiology has these enzymes. Check physiology YAML."
        )

    # ProdrugActivationEdge per site
    for site in conversion_sites:
        edge = ProdrugActivationEdge(
            source=site, target=active_node_name,
            enzyme_tags=enzyme_tags,
            conversion_yield=am.conversion_yield_fraction,
            mw_parent=drug.mw, mw_active=am.mw,
        )
        graph_aug = graph_aug.add_edge(edge)

    # 1C elimination from active to sink (UNCHANGED from v1)
    elim_edge = OneCompartmentEliminationEdge(
        source=active_node_name, target=_DEFAULT_ACTIVE_SINK,
        cl_per_h=am.CL_per_h, vd_l=am.Vd_L,
    )
    return graph_aug.add_edge(elim_edge)
```

**Idempotency**: Function called once per pipeline invocation. Two-call behavior detail (raise vs detect-and-return-unchanged) deferred to Plan; idempotency unit test is mandatory (R9).

### 4.7 Registry schema (`data/sbi/prodrug_activation_registry.json`)

Replace v1 schema:

```json
{
  "<canonical_smiles>": {
    "name": "BH4",
    "mw": 241.25,
    "fup":  {"mean": 0.23, "cv": 0.3},
    "CL_per_h": {"mean": 40.0, "cv": 0.35},
    "Vd_L": {"mean": 150.0, "cv": 0.3},
    "conversion_yield_fraction": {"mean": 0.85, "cv": 0.1},
    "yield_source": "literature",          // NEW: "literature" | "class_extrapolated"
    "observation_species": "parent",
    "enzyme_affinity_for_conversion": {     // NEW (replaces conversion_rate_per_h, conversion_site)
      "SPR": {
        "mean": "<TBD by Task 0>",
        "cv": "<TBD>",
        "citation": "Park 2008 J Biol Chem ..."   // per-enzyme citation
      }
    },
    "affinity_source": "literature",        // NEW: "literature" | "class_extrapolated"
                                            // (tier 3 "infrastructure_only" rejected by loader in v2)
    "_clinical_citation": "Gao 2024 PMC11597218..."
  }
}
```

**Removed fields:** `conversion_rate_per_h`, `conversion_site`, top-level `_citation` (per-enzyme citation supersedes).
**Migration:** v1's 4 entries fully rewritten by Plan Task 0 + Task 1 (literature collection + value entry). No v1 schema compat shim — internal migration only.

### 4.8 Registry loader (`src/sisyphus/predict/registry.py`)

`lookup_active_metabolite(smiles, registry_path=None) -> tuple[ActiveMetabolite, str, dict] | None`:
- Returns 3-tuple: `(ActiveMetabolite, observation_species, enzyme_affinity_for_conversion_dict)`.
- New validation: `affinity_source ∈ {"literature", "class_extrapolated"}` (rejects "infrastructure_only" in v2).
- New validation: `yield_source` enum same.
- New validation: `enzyme_affinity_for_conversion` non-empty.
- Existing: SMILES canonicalization via RDKit, mandatory field checks.

### 4.9 Pipeline (`src/sisyphus/pipeline/predict.py`)

Flow UNCHANGED. ONE expanded field-passing call:

```python
result = lookup_active_metabolite(canonical_smiles)
if result is not None:
    active_metab, obs_species, conv_affinities = result
else:
    active_metab, obs_species, conv_affinities = None, "parent", {}

drug = DrugOnGraph(
    ...,
    active_metabolite=active_metab,
    observation_species=obs_species,
    enzyme_affinity_for_conversion=conv_affinities,  # NEW
)
graph_augmented = augment_for_active_species(base_graph, drug, observation_node="venous_blood")
# ... compile + solve ...
```

`_resolve_observation_node`, `_adjust_ad_for_prodrug`: UNCHANGED.

### 4.10 Reuse / change accounting

| Item | Status |
|---|---|
| `ActiveMetabolite` | UNCHANGED |
| `OneCompartmentEliminationEdge` + Spec | UNCHANGED |
| `_resolve_observation_node`, `_adjust_ad_for_prodrug` | UNCHANGED |
| `ACTIVE_SUFFIX`, `_DEFAULT_ACTIVE_SINK` | UNCHANGED |
| `DrugOnGraph` 19 existing fields | UNCHANGED (1 new field added) |
| Pipeline integration hook | UNCHANGED (signature pass-through expanded) |
| 107-holdout regression test | UNCHANGED (in spirit; verifies invariance) |
| `ProdrugActivationEdge` struct | CHANGED (`conversion_rate` → `enzyme_tags`) |
| `ProdrugActivationFluxSpec.apply` body | REWRITTEN (~50 LoC, well_stirred) |
| Registry schema | CHANGED (3 fields removed, 4 added) |
| Registry loader return type | CHANGED (2-tuple → 3-tuple) |
| `augment_for_active_species` body | REWRITTEN (multi-site discovery) |
| Mass balance synthetic test | NEW topology (flow loop required) |
| Physiology YAML enzyme entries | NEW (additive) |
| `ResolvedParams.drug_enzyme_affinity_for_conversion` | NEW method |

**Reuse: 9/15 (60%). Changed/rewritten: 6. New: 3.** (Honest accounting; v1 retrospective claim of 12/15 was over-stated.)

---

## 5. Data flow

### 5.1 End-to-end (sepiapterin example)

```
1. SMILES canonicalize via RDKit
2. Registry lookup → (ActiveMetabolite(BH4), "parent", {"SPR": Dist(...)})
3. ADME prediction (parent only, unchanged path)
4. DrugOnGraph(... enzyme_affinity_for_conversion={"SPR": Dist(...)} ...)
5. base_graph = BodyGraph from physiology YAML (now includes SPR/CES1/CES2/ALPI)
6. augment_for_active_species:
   - add venous_blood_active node (V = BH4.Vd_L)
   - discover sites: liver, gut_wall, kidney (any node with SPR)
   - add ProdrugActivationEdge per site (enzyme_tags={"SPR"} baked at compile time)
   - add OneCompartmentEliminationEdge(venous_blood_active → metabolized_gut)
7. Compile → ODE skeleton + state_index
8. for i in 1..1000 MC:
   - sample params (resolve all Distributions including SPR abundance × affinity)
   - solve ODE; flux evaluation per RHS step:
     - existing fluxes (flow, clearance, transit, absorption, diffusion) UNCHANGED
     - ProdrugActivationFluxSpec(liver→active): well_stirred extraction, mass redirect to active
     - ProdrugActivationFluxSpec(gut_wall→active): same math, different source
     - OneCompartmentEliminationFluxSpec(active→sink): keff = CL/Vd
   - SimResult with concentrations["venous_blood"], ["venous_blood_active"]
9. PK extraction: observation_species="parent" → C[venous_blood]
10. ad_flags: PRODRUG_REGISTERED + warning
11. PredictionResult(pk, method="hybrid", in_AD=True, ad_flags=[...], cmax_90ci)
```

### 5.2 Cascade arithmetic (multi-site)

ODE solver handles cascade automatically via local fluxes:
- t = 0.5 h: parent in lumen → gut_wall (absorption).
- gut_wall flux: extract `E_gut × Q × c_unbound` → some to active, residual flows to portal_vein.
- portal_vein → liver (FlowFluxSpec).
- liver flux: extract `E_liver × Q × c_unbound` → some to active (SPR), some to sink (CYP3A4 elimination).
- Combined pre-systemic activation = `Eg + (1−Eg) × Eh × yield_fraction` (cascade arithmetic, automatic — engine does not need explicit cascade math).

### 5.3 Backward compat path (non-prodrug 107-holdout)

```
2. Registry lookup: SMILES no match → None
4. DrugOnGraph(... active_metabolite=None, enzyme_affinity_for_conversion={} ...)
6. augment_for_active_species: active_metab is None → graph unchanged
7. Compile: identical to v1 baseline
8. Flux: zero new ProdrugActivationFluxSpec instances
→ Numerical results byte-identical to pre-v2.
```

**Critical**: ADME prediction path must NOT populate `enzyme_affinity` dict with new tags (SPR, CES1, CES2, ALPI) for non-prodrug drugs. ML predictor remains restricted to existing CYP/UGT tags. Verified by 107-holdout regression test.

### 5.4 Identity-blind invariant verification

Engine sees only string tags. Test: replace `"SPR"` with random `"Z1Q9K"` everywhere (drug + physiology + registry). Numerical output must be byte-identical (rtol < 1e-9). No name-matching anywhere in engine code.

### 5.5 MC consistency

Per MC iteration:
- `params.drug_enzyme_affinity_for_conversion("SPR")` returns ONE sample value, used at ALL conversion sites for that drug in that iteration.
- `params.node_enzymes("liver")["SPR"]` returns ONE sample value per site, independently.
- Multi-site fluxes share drug-side affinity, differ in site-side abundance/Q/kp.
- Pattern identical to existing `ClearanceFluxSpec(well_stirred)` — no new MC machinery.

---

## 6. Validation strategy

### 6.1 Validation gate (per-drug parametrized)

```python
@pytest.mark.parametrize("drug,clinical_cmax,enzyme", [
    ("sepiapterin", 0.0024, "SPR"),
    ("remdesivir", 4.38, "CES1"),
    ("tebipenem_pivoxil", 4.01, "CES2"),
    ("fostamatinib", 0.61, "ALPI"),
])
def test_prodrug_3fold(drug, clinical_cmax, enzyme):
    pred = predict_cmax_mean(drug)
    assert max(pred / clinical_cmax, clinical_cmax / pred) < 3.0
```

**Per-drug pass/fail visible** in CI output (parametrize natural granularity). Aggregate is implicit.

**Failure protocol**: failing-drug parametrize cases marked `xfail`, not relaxed. Per-drug fold-error documented in CHANGELOG. **Affinity values NOT adjusted to make test pass** (mechanistic-A core promise).

### 6.2 Tier classification (Plan Task 0 deliverable)

| Drug | Enzyme | Candidate sources | Final tier |
|---|---|---|---|
| sepiapterin | SPR | Park, Kim et al SPR characterization | TBD |
| remdesivir | CES1 | Eastman 2020 ACS Cent Sci, Yan 2017 | TBD |
| tebipenem_pivoxil | CES2 / intestinal esterase | Kamiya, Imai review | TBD |
| fostamatinib | ALPI | Class kinetics for phosphate monoesters | TBD |

Plan Task 0 outputs: tier (1/2/3) per drug with citations. Drugs assigned tier 3 are excluded from v2 registry.

**Contingency — 0 tier 1+2 drugs found**: spec re-open. Architecture is still valid (resolves v1 kinetic limitation) but mechanistic claim is empty without literature. Options: (a) ship as "infrastructure-only" v2 release noting limitation, (b) substitute prodrugs with better literature support (capecitabine, oseltamivir, irinotecan).

### 6.3 Test categories

**Unit tests:**
- `test_prodrug_activation_flux_well_stirred` — rate matches well_stirred formula given known inputs.
- `test_augment_multi_site_discovery` — drug enzyme_tags ⊕ multi-node physiology → all sites edged.
- `test_augment_no_site_raises` — empty intersection → ValueError.
- `test_augment_idempotency` — two-call behavior (detail TBD by Plan).
- `test_resolved_params_drug_enzyme_affinity_for_conversion` — sampling consistency (single sample per MC, used at all sites).
- `test_registry_loader_v2_schema` — enum validation, RDKit canonicalization, mandatory fields.
- `test_registry_validates_negative_vd`, `test_registry_validates_unknown_affinity_source` — edge cases.
- `test_drugongraph_postinit_validation` — `enzyme_affinity_for_conversion` + `active_metabolite` consistency.
- `test_mw_ratio_one` — `mw_parent = mw_active` boundary (yield != 1).

**Integration tests:**
- `test_prodrug_synthetic_well_stirred_mass_balance` (NEW topology, flow loop required — see §6.4).
- `test_prodrug_pipeline_smoke_4drugs` — end-to-end execution for all 4 prodrugs.
- `test_prodrug_observation_routing` — parent vs active observation modes.
- `test_prodrug_ddi_smoke_ces1` — half CES1 abundance → remdesivir activation halves (proportionality).

**Regression tests:**
- `test_prodrug_3fold` — validation gate (§6.1).
- `test_holdout_unchanged` — 107-holdout numerical invariance (byte-identical to pre-v2).
- `test_engine_identity_blind_rename` — random tag rename → byte-identical results.
- `test_per_prodrug_snapshot` — each prodrug Cmax mean ± 5% pinned (catches silent drift within 3-fold).

### 6.4 Mass balance synthetic test (new topology)

Well-stirred requires flow context. v1's static-compartment kinetic test does NOT translate.

**Topology:**
```
infusion_source (constant rate r mg/h)
   │ FlowEdge (Q = 60 L/h)
   ▼
conversion_node (V = 10 L, fup = 1.0, abundance × affinity × ivive = 10 L/h)
   │ FlowEdge (Q = 60 L/h)            │ ProdrugActivationEdge(enzyme_tags={X}, mw_ratio=1, yield=1)
   ▼                                   ▼
exit_sink                           active_pool (V_active = 10 L)
                                       │ OneCompartmentEliminationEdge (CL = 10, Vd = 10)
                                       ▼
                                   elim_sink
```

**Steady-state analytical:**
- E = (Q × fup × CLint) / (Q + fup × CLint) = (60 × 1 × 10) / (60 + 10) = 8.571 L/h
- c_node_steady = c_in × Q / (Q + fup × CLint) = c_in × 60/70
- rate_to_active = E × c_node × yield × mw_ratio = 8.571 × c_node
- A_active_steady = rate_to_active / k_eff_active = 8.571 × c_node / 1.0

Verify v2 reaches this steady state within `rtol = 1e-3`.

### 6.5 Identity-blind regression test

Helper `rename_enzyme_tags(obj, mapping)`: deep dict-key rename in (a) `drug.enzyme_affinity_for_conversion`, (b) physiology YAML `node.enzymes`, (c) registry JSON. Numerical output identical (rtol 1e-9). Helper implementation deferred to Plan; pattern reusable for retroactive verification of CYP path.

### 6.6 v2-vs-v1 reporting (CHANGELOG, not test)

CHANGELOG entry includes per-drug fold-error comparison:

| Drug | v1 fold-error | v2 fold-error | Improvement |
|---|---|---|---|
| sepiapterin | 5356× | TBD | TBD× |
| ... | | | |

**Reporting requirement, not automated test** — avoids v1 baseline re-run cost in CI.

### 6.7 MC interval reporting (diagnostic)

For each prodrug, report:
- Predicted Cmax mean
- 90% MC PI (p5, p95)
- Clinical reference
- "In PI" indicator

Diagnostic only (not gate). Wide PI on tier 2 drugs is informative not failure. Note Sisyphus's documented PI under-coverage (29.9% at nominal 90% per CLAUDE.md) — interpret per-prodrug PI accordingly.

### 6.8 Performance (low priority)

v2 adds ~10 edges total (4 prodrugs × 2–3 sites). Non-prodrug 107 unchanged. State vector grows ~6%. Expected impact: negligible. Verify via existing benchmark smoke (`scripts/run_engine_benchmark.py`); no new perf test added.

---

## 7. Risks

| ID | Risk | Probability | Mitigation |
|---|---|---|---|
| R1 | Literature affinity may not match clinical Eg within 3-fold | Medium-high | Tier 2 wide CV; affinity NOT refit; failure informative |
| R2 | Multi-site discovery picks unintended sites if physiology YAML changes externally | Low | Per-prodrug snapshot test catches drift |
| R3 | Active species 1C model loses first-pass active metabolism | Low impact for v2 4 drugs | Deferred to v3 (D1) |
| R4 | Future ML retraining adds CES1/SPR predictions for non-prodrugs → 107-holdout numerical change | Medium (future) | Regression test as guard; CHANGELOG when intentionally re-baselined |
| R5 | Plan Task 0 finds zero tier 1+2 drugs | Low-medium | Spec re-open contingency; alternative drug list (capecitabine, oseltamivir, irinotecan) |
| R6 | Yield fraction back-fit risk (v1 yield values may be clinically inferred) | Medium | `yield_source` enum with same tier system as affinity |
| R7 | New enzyme correlation gap (no Achour matrix entries for SPR/CES/ALPI) | Low impact | Deferred (correlation extension future work) |
| R8 | Tier 2 wide CV lognormal heavy tail → extreme MC samples | Low | Accept and report wide PI; no clamping (would bias) |
| R9 | Augmentation idempotency | Very low | One-line idempotency unit test |
| R10 | `ProdrugActivationEdge.conversion_rate` removal breaks external code | Very low (internal API) | Grep verification across codebase |
| R11 | Physiology enzyme additions invalidate v1 metric AAFE 2.695 | Low | 107-holdout regression test |
| R12 | Silent prodrug Cmax drift within 3-fold | Medium | Per-prodrug snapshot test (±5% threshold) |

---

## 8. Out-of-scope (v3+ deferred)

| ID | Item |
|---|---|
| D1 | First-pass active species PBPK (gut→liver→systemic for active) |
| D2 | DDI clinical scenarios (only smoke test in v2) |
| D3 | ML predictor for activation enzyme affinity |
| D4 | Conversion enzyme genetic polymorphism (CES1 G143E etc.) |
| D5 | Special populations (pediatric, hepatic impairment) |
| D6 | Registry-free auto-routing (SMILES → automatic prodrug detection) |
| D7 | Multi-step activation cascade (capecitabine → 5'-DFCR → 5'-DFUR → 5-FU) |
| D8 | Reversible conversion |
| D9 | Tier 3 (infrastructure-only / fitted) registry entries |
| D10 | Static compartment kinetic conversion mode |

---

## 9. Success criteria

| # | Criterion | Type |
|---|---|---|
| 1 | Invariants 1–8 maintained (engine identity-blind, all Distributions, compile-once, flow conservation, holdout inviolable, no drug-specific branches, 20 files/dir, hard no-touch) | Hard gate |
| 2 | 107-holdout regression byte-identical | Hard gate |
| 3 | Mass balance synthetic well_stirred test (rtol 1e-3) | Hard gate |
| 4 | Identity-blind random-rename test | Hard gate |
| 5 | Plan Task 0 yields ≥1 tier 1+2 drug | Hard gate (else spec re-open) |
| 6 | DDI smoke test (CES1 abundance ↔ remdesivir activation proportionality) | Hard gate |
| 7 | Per-prodrug snapshot tests | Hard gate |
| 8 | Augmentation idempotency unit test | Hard gate |
| 9 | All Sisyphus CI green (existing tests not regressed) | Hard gate |
| 10 | Validation gate 3-fold per-drug parametrized | Reporting (xfail allowed) |
| 11 | CHANGELOG v1-vs-v2 fold-error table | Reporting |

**8 hard gates, 2 reporting items.** Validation gate (10) is reporting because v2's mechanistic promise excludes back-fit; clinical match is sought but not enforced.

---

## 10. Open questions (Plan Task 0 must resolve)

- SPR / CES1 / CES2 / ALPI abundance per organ + CV (physiology YAML values).
- Per-drug enzyme affinity (mean + cv) values from literature or class extrapolation, with citations.
- Per-drug yield fraction provenance (literature or class).
- Tier (1/2/3) classification per drug with citations.
- Verification of literature claims referenced in §3.3 (Park, Eastman 2020, Yan 2017, Kamiya).

Plan Task 0 produces a deliverable file (e.g., `docs/superpowers/specs/2026-04-27-prodrug-v2-task0-literature.md`) with all five resolved before any implementation task starts.

---

## 11. References

- v1 spec: `docs/superpowers/specs/2026-04-24-prodrug-activation-design.md`
- v1 plan: `docs/superpowers/plans/2026-04-25-prodrug-activation.md`
- v1 merge: `af9d2be` (2026-04-26)
- v1 validation failure: CHANGELOG.md Unreleased section
- Sisyphus invariants: CLAUDE.md §Invariants
- Sisyphus elimination pattern (template for v2): `src/sisyphus/engine/flux.py::ClearanceFluxSpec`
