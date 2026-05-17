# Multi-Enzyme Prodrug Conversion Schema — Per-Enzyme Yield (B-04)

**Date**: 2026-05-17
**Issue**: [#11](https://github.com/jam-sudo/Sisyphus/issues/11) (prerequisite to closing the remaining 1/3 — clopidogrel)
**Backlog ref**: B-04 (currently blocked-by B-03; this spec inverts that to **B-04 must ship before B-03 can be implemented**)
**Predecessor**: v0.3.4 prodrug registry expansion (PR #34, `docs/superpowers/specs/2026-05-08-prodrug-registry-expansion-design.md`)

---

## 1. Goal

Allow a single prodrug registry entry to declare **per-enzyme conversion yields**, so drugs whose parent metabolism splits into a dead-end branch and an active-producing branch can be represented mechanistically.

Concrete trigger: **clopidogrel** (B-03). Its parent is hydrolyzed by hepatic CES1 to inactive SR26334 (~85% of dose) and *separately* bioactivated by CYP2C19 / CYP3A4 to R-130964 (~15% of dose). One enzyme tag with one yield cannot represent this dual fate without violating either mass balance (yield-per-edge applied uniformly) or the mechanistic-A doctrine (back-calibration of a single "effective" yield that conflates two pathways).

This is the structural blocker discovered on 2026-05-17 during the B-03 design pass; see §3 below.

---

## 2. Background

### 2.1 Existing registry shape (post-PR #34, 6 entries)

`data/sbi/prodrug_activation_registry.json` is SMILES-keyed. Every entry carries one entry-level `conversion_yield_fraction: {mean, cv}` plus an `enzyme_affinity_for_conversion: {enzyme: {mean, cv, citation}}` dict. All six current entries declare **exactly one enzyme**:

| Drug | Active species | Enzyme | Entry yield |
|---|---|---|---|
| BH4 | sapropterin | SPR | 0.85 |
| GS-441524 | remdesivir-active | CES1 | 0.90 |
| tebipenem | tebipenem-active | CES2 | 0.95 |
| R406 | fostamatinib-active | ALPI | 0.90 |
| simvastatin | acid | CES1 | 0.30 |
| irinotecan | SN-38 | CES2 | 0.05 |

### 2.2 Existing builder behaviour (post-PR `5da805b`, "multi-site discovery")

`augment_for_active_species` (in `src/sisyphus/graph/builder.py`) iterates physiology nodes, finds each that holds at least one of the drug's declared enzyme tags, and emits one `ProdrugActivationEdge` per discovered site:

```python
for site in conversion_sites:
    activation_edge = ProdrugActivationEdge(
        source=site,
        target=active_node_name,
        enzyme_tags=enzyme_tags,
        conversion_yield=am.conversion_yield_fraction,   # ← entry-level, shared
        mw_parent=drug.mw,
        mw_active=am.mw,
    )
```

Every emitted edge carries the **same** entry-level yield Distribution. The engine resolves and reads it per-edge via `params.edge_param(edge_id, "conversion_yield")`, so the engine is already per-edge capable — only the builder erases the per-enzyme distinction.

### 2.3 Existing engine flux (no change needed)

`ProdrugActivationFluxSpec.apply` (in `src/sisyphus/engine/flux.py:605-643`) consumes parent at the well-stirred extraction rate `cl_organ × c_out` and adds `rate_parent × mw_ratio × yield` to the active node. The yield is read per-edge — yield=0 is valid and produces a parent-consuming, active-non-producing edge (a "dead-end" edge mechanistically).

---

## 3. The Structural Blocker for Clopidogrel (B-03)

Clopidogrel's in vivo parent disposition has two co-occurring hepatic fates:

| Fraction | Path | Enzyme | Yield to active |
|---|---|---|---|
| ~85% | parent → SR26334 (inactive carboxylic acid; dead-end) | CES1 | 0 |
| ~15% | parent → 2-oxo → R-130964 (active thiol) | CYP2C19 / CYP3A4 | ~1.0 in single-step approximation |

With the current entry-level-only schema, three options exist and all break:

1. **Register only CYP2C19 with yield=0.15.** Loses the CES1 dead-end branch entirely. With `metabolic_fraction=0` zeroing all XGBoost-derived enzyme paths to avoid double-counting the lone CYP2C19 ProdrugActivationEdge, the parent's only remaining clearance is that single edge — ~15% of in vivo total hepatic CL — leaving parent **~6× under-cleared** (1/0.15 = 6.67×). Parent AUC ~6× over → active production proportionally over.
2. **Register both CES1 and CYP2C19, single entry-level yield (e.g., 0.15).** Engine applies yield=0.15 to *both* edges' fluxes. Total mass routed to active_node is approximately correct (0.85 × 0.15 + 0.15 × 0.15 = 0.15 of dose), but the model is wrong in two ways: (a) **species identity** — the active_node is parameterised for R-130964 (MW 504), yet 85% of the routed mass mechanistically corresponds to SR26334 (different molecule, different in-vivo fate); (b) **rate profile** — CES1 (~8.0e7 pmol hepatic, fast hydrolysis) and CYP2C19 (~1.4e6 pmol, slower oxidation) have very different per-edge flux time-courses, so the realised active species accumulates on the CES1 timescale rather than the CYP2C19 timescale, distorting Cmax magnitude and Tmax.
3. **Register both with yield=0.85 (CES1's share).** Yield 0.85 applied to both edges produces ~85% of dose at active_node (since both edges share the yield). That is ~5–6× too much active species vs the in vivo ~15%, on top of the same species-identity and rate-profile problems as Option 2.

The schema needs **per-enzyme yield** so CES1 can declare yield=0 (dead-end) and CYP2C19 can declare yield≈1 (full conversion of CYP fraction).

---

## 4. Approaches Considered

### 4.1 (Recommended) Optional per-enzyme `yield` field with entry-level fallback

`enzyme_affinity_for_conversion[enzyme]` gains an optional `yield: {mean, cv}` field. When present, the builder reads it for that enzyme's edges. When absent, the builder falls back to the entry-level `conversion_yield_fraction` (current behaviour). The six existing entries — each with a single enzyme and a single entry-level yield — remain valid and bit-identical post-migration.

**Pros**: minimal schema delta; backward-compatible; existing tests untouched; engine unchanged.

**Cons**: two yield sources (entry-level + per-enzyme) in the same data file. Mitigated by an explicit precedence rule and a regression test that any drug declaring multiple enzymes MUST declare per-enzyme yields (forbid ambiguity).

### 4.2 (Rejected) Replace entry-level yield with required per-enzyme yields

Cleaner schema (single source of truth) but forces migration of all 6 existing entries. Higher churn for no current benefit — existing entries are unambiguous.

### 4.3 (Out of scope) Attribute yield to existing `ClearanceFluxSpec` instead of adding `ProdrugActivationEdge`

Deeper architectural rework: instead of separate prodrug edges, attach an "active-yield attribution" to the engine's existing well-stirred hepatic clearance edges, so the same parent consumption serves both XGBoost CLint accounting and active species accounting. Avoids the double-count problem natively. Deferred — would require revisiting how XGBoost CLint is decomposed and would intersect the `metabolic_fraction` override registry. Tracked as a follow-up architectural option only if §4.1 proves insufficient for non-clopidogrel multi-fate prodrugs.

---

## 5. Proposed Schema (§4.1 detail)

### 5.1 Registry diff (sketch)

```json
"<clopidogrel-SMILES>": {
  "name": "clopidogrel",
  ...
  "conversion_yield_fraction": {"mean": 0.15, "cv": 0.40},
  "yield_source": "literature",
  "enzyme_affinity_for_conversion": {
    "CES1": {
      "mean": <lit>, "cv": <lit>,
      "yield": {"mean": 0.0, "cv": 0.0},
      "citation": "Tang 2006 DMD 34:603-7 (clopidogrel CES1 hydrolysis → inactive SR26334, ~85% of dose dead-end)"
    },
    "CYP2C19": {
      "mean": <lit>, "cv": <lit>,
      "yield": {"mean": 1.0, "cv": 0.30},
      "citation": "Kazui 2010 DMD 38:92-9 (CYP2C19 + CYP3A4 → 2-oxo → R-130964 active thiol; single-step approximation collapses both steps)"
    }
  },
  ...
}
```

For backward compat, **existing 6 entries are unchanged**. They have one enzyme, no per-enzyme `yield` key, and the builder reads `conversion_yield_fraction` as today.

### 5.2 Builder diff (sketch)

```python
# OLD
for site in conversion_sites:
    activation_edge = ProdrugActivationEdge(
        ...,
        conversion_yield=am.conversion_yield_fraction,
        ...,
    )

# NEW: per-enzyme yield with entry-level fallback
for site in conversion_sites:
    node_enzymes = graph.nodes[site].enzymes
    for tag in enzyme_tags & set(node_enzymes.keys()):
        per_enz_yield = am.enzyme_yields.get(tag)        # None if not declared
        yld = per_enz_yield if per_enz_yield is not None else am.conversion_yield_fraction
        activation_edge = ProdrugActivationEdge(
            source=site,
            target=active_node_name,
            enzyme_tags=frozenset({tag}),                 # per-enzyme edge
            conversion_yield=yld,
            mw_parent=drug.mw,
            mw_active=am.mw,
        )
        graph.add_edge(activation_edge)
```

Per-enzyme edges (one edge per tag per site) replace the current "all tags collapsed into one edge per site". This means existing entries with one enzyme produce the same edge count and structure; entries with N enzymes produce N edges per site.

### 5.3 `ActiveMetabolite` dataclass diff

`active_metabolite` (in `src/sisyphus/core.py`) gains a parallel `enzyme_yields: dict[str, Distribution]` populated by the registry loader (`src/sisyphus/predict/prodrug_activation.py`) from per-enzyme `yield` fields. Empty dict for entries that omit them.

### 5.4 Validation rule

A new regression test asserts: any registry entry with `len(enzyme_affinity_for_conversion) >= 2` MUST declare per-enzyme `yield` for every enzyme listed. Mixed (some declared, some omitted) is rejected at load time. This forbids the ambiguity §4.1 introduces.

---

## 6. Backward Compatibility

The 6 existing entries (BH4, GS-441524, tebipenem, R406, simvastatin, irinotecan) are all single-enzyme. With the per-enzyme yield optional + entry-level fallback:

- Loader: no behavioural change for single-enzyme entries.
- Builder: each existing entry now emits one edge per site (same as today, since `enzyme_tags` was always one-element for these entries; the previous builder used `enzyme_tags=enzyme_tags` over a single-element set, producing one edge — bit-identical structure).
- Engine: no change.
- Tests: `tests/regression/test_prodrug_registry_seed.py` frozenset seed list unchanged; per-drug snapshot tests in `tests/regression/test_prodrug_v2_snapshot.py` should remain bit-identical.

Mass-balance and 107-holdout AAFE should be **bit-identical pre/post this PR** in isolation. The headline shift comes only when B-03 (clopidogrel) lands on top of this schema.

---

## 7. Tests

### 7.1 New unit tests (`tests/unit/test_prodrug_per_enzyme_yield.py`)

- `test_single_enzyme_entry_falls_back_to_entry_level_yield` — existing entries' equivalent.
- `test_multi_enzyme_entry_reads_per_enzyme_yields` — synthetic 2-enzyme entry produces 2 edges per site with the right yields.
- `test_multi_enzyme_entry_missing_per_enzyme_yield_raises` — partial declaration rejected.
- `test_dead_end_yield_zero_is_valid` — yield=0 produces edge with zero active contribution but non-zero parent consumption.

### 7.2 Schema regression update (`tests/regression/test_prodrug_v3_registry_schema.py`)

Add the §5.4 validation rule check.

### 7.3 Snapshot regression (`tests/regression/test_prodrug_v2_snapshot.py`)

No changes required if the §6 backward-compat claim holds. Run pre/post and confirm bit-identical Cmax for the 6 existing entries.

---

## 8. B-03 Application (downstream, separate PR)

Once B-04 ships, clopidogrel becomes implementable. The B-03 spec — to be written when this lands — will cover:

1. Literature curation for CES1 affinity (Tang 2006), CYP2C19 affinity (Kazui 2010), R-130964 PK (Karazniewicz-Lada 2014). Disposition expected **ceiling_accepted** per the v3 mechanistic-A doctrine (covalent P2Y12 binding sink prevents standard CL/Vd measurement of R-130964 plasma).
2. Registry entry with `observation_species="parent"` (107-holdout reference is parent clopidogrel Cmax). Switching to "active" would not collapse magnitudes wildly — clopidogrel parent and R-130964 plasma Cmax are similar molar concentrations after 75 mg PO (Caplain 1999 / Karazniewicz-Lada 2014 both ~3-5 ng/mL; ~2-3× molar ratio) — but the time-course, decay profile, and AUC differ enough that "active" vs the holdout's parent reference would still inject a non-trivial species-mismatch error. The conservative choice for an apples-to-apples holdout comparison is `parent`.
3. `data/transporters/cyp_clearance_overrides.json` entry: `metabolic_fraction=0.0` for clopidogrel to zero out engine-derived hepatic CL on parent, since the two ProdrugActivationEdges (CES1 dead-end + CYP2C19 active) now carry the full hepatic CL of clopidogrel parent.
4. Integration test gating `cmax > 0` with generous upper bound (ceiling_accepted disposition).
5. **107-holdout regen** with documented AAFE delta (clopidogrel is in the holdout; this is a real headline shift). Bootstrap CIs refreshed.

---

## 9. Risks

- **§4.3 architectural concern.** Multiple ProdrugActivationEdges + `metabolic_fraction=0` still routes 100% of parent through ProdrugActivationEdges' well-stirred extraction at hepatic sites. The literature-anchored CES1 + CYP2C19 affinities must SUM to approximately the in vivo total hepatic CL of clopidogrel parent (~150-250 L/h). If they don't, parent CL is mis-calibrated; the mechanistic-A doctrine accepts this as a ceiling rather than back-calibrating. To be addressed during B-03 implementation.
- **R-130964 plasma kinetics violate the 1C linear model.** Covalent P2Y12 binding is non-saturable in vivo at therapeutic doses but the engine's `OneCompartmentEliminationEdge` is linear. Acceptable as ceiling_accepted; B-03 spec will document. If this proves to be a project-blocking error pattern, the §4.3 "attribute yield to existing ClearanceEdge" architectural option re-opens.
- **Validation rule rejection of mixed declaration** (§5.4) could surprise future maintainers. Mitigated by a clear error message naming the offending entry + enzyme.
- **`metabolic_fraction` remains a scalar.** `data/transporters/cyp_clearance_overrides.json` scales all enzyme affinities uniformly per drug. For clopidogrel (where `=0` zeros every engine-derived enzyme path and the two ProdrugActivationEdges carry all hepatic CL) this is sufficient. It will become insufficient for a future drug whose XGBoost-derived CYP path should be only partially zeroed (e.g., a drug where CYP3A4 contributes to both a dead-end fate and an active-producing fate, requiring per-enzyme down-scaling). A per-enzyme `metabolic_fraction` extension is out of scope for B-04; flag for B-03-prasugrel and B-03-ticagrelor implementation cycles.

---

## 10. Out of Scope

- §4.2 (replace entry-level yield with required per-enzyme).
- §4.3 (architectural rework of yield attribution).
- B-03 implementation (separate spec post-merge).
- Prasugrel / ticagrelor application (separate B-03-analog cycles).

---

## 11. Mechanistic-A Doctrine Path (this PR, schema-only)

N/A — this PR ships no registry data changes. All literature anchoring lives in B-03.

---

## 12. Acceptance

- 6 existing entries: snapshot Cmax bit-identical pre/post.
- 107-holdout AAFE bit-identical pre/post (no registry data changes in this PR).
- New unit tests pass.
- Schema regression test catches partial per-enzyme declaration.
- B-03 (clopidogrel) becomes implementable; its spec is the next step after this lands.
