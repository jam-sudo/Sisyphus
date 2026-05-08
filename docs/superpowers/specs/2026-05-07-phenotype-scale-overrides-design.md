# Phenotype Scale Overrides (v0.3.3) — Design

**Date**: 2026-05-07
**Issue**: [#31](https://github.com/jam-sudo/Sisyphus/issues/31) (capability request from GenoADME)
**Target version**: v0.3.3
**Branch**: `feat/phenotype-scale-overrides`
**Predecessor**: PR #32 (v0.3.2 NAT2/UGT1A1 + back-solve cancellation fix, merged `1e06ded`)

---

## 1. Goal

Add a `phenotype_scale_overrides: dict[str, float] | None = None` keyword to both `apply_phenotype_to_graph()` and `predict()`. When provided, the override replaces the default `PHENOTYPE_SCALES[phenotype]` for the matching gene at scaling time. When omitted or `{}`, behavior is identical to current.

The dimension shape is **flat `{gene: scale}`**. Substrate dimension is implicit in the SMILES being passed to `predict()` per call. Phenotype dimension is implicit in the `phenotypes` argument of the same call.

This unblocks GenoADME v0.3 milestone (Niemi 2006 men-stratum AUC PM/EM 3.32 [1.74, 4.91]; current Sisyphus pin yields 4.482 — at the upper edge of CI). GenoADME wants to compress toward the central 3.32 via per-substrate calibration on their side; Sisyphus provides only the API hook with no built-in calibration tables.

## 2. Background

### 2.1 What's in main today (post-v0.3.2)

`src/sisyphus/predict/phenotype.py:97-172` `apply_phenotype_to_graph(graph, phenotypes, node="liver")`:
- Iterates `phenotypes` dict
- Looks up `scale = PHENOTYPE_SCALES[phenotype]` (PM=0.10, IM=0.50, EM=1.00, RM=1.50, UM=2.00)
- For transporter genes (`SLCO1B1` → `OATP1B1`), scales `liver.transporters[OATP1B1].mean`
- For enzyme genes (`CYP1A2`, `NAT2`, `UGT1A1`, etc.), scales `liver.enzymes[tag].mean`
- Returns new BodyGraph with scaled Distribution

`src/sisyphus/pipeline/predict.py` calls `apply_phenotype_to_graph` with `phenotypes` argument when provided. v0.3.2 added pre-phenotype enzyme abundance snapshot to make CYP/UGT/NAT propagation work correctly.

### 2.2 Why issue #31

GenoADME's v0.3 meta-analysis extracted SLCO1B1/pravastatin PM/EM AUC ratio data: Niemi 2006 men-stratum 3.32 [1.74, 4.91]. Sisyphus current pin (post-#32) produces 4.482 AUC ratio — inside the CI but at the upper edge. CPIC's 0.10× PM scale is community-standard but compounds non-linearly through the graph layer for some substrates.

GenoADME wants to inject a per-substrate effective scale (e.g., 0.30 for pravastatin) without committing Sisyphus to substrate-specific calibration tables. This issue is downstream-side empirical, not a Sisyphus calibration question — Sisyphus's CPIC-default 0.10 is correct for the community-standard scale; GenoADME has richer empirical data for specific substrates.

## 3. Architecture

**Engine: zero changes.** Override only changes the abundance scale at `apply_phenotype_to_graph` time. The scaled abundance flows through the rest of the pipeline (v0.3.2 back-solve fix included) identically to a default-scale call.

```
predict(SMILES, phenotypes={"SLCO1B1": "PM"}, phenotype_scale_overrides={"SLCO1B1": 0.30})
  │
  ▼
graph = build_from_yaml(reference_man.yaml)
liver_enzymes_pre = snapshot (pre-phenotype, per v0.3.2)
  │
  ▼
graph = apply_phenotype_to_graph(graph, phenotypes={"SLCO1B1": "PM"},
                                  phenotype_scale_overrides={"SLCO1B1": 0.30})
  → for tag="SLCO1B1": scale = PHENOTYPE_SCALES["PM"] = 0.10
  → override hit: scale = 0.30
  → liver.transporters["OATP1B1"].mean *= 0.30   (vs default ×0.10)
  │
  ▼
build_drug_on_graph(..., liver_enzymes=liver_enzymes_pre)
engine: rate_OATP1B1 = scaled_abundance × Vmax_kinetics  ← MM path, propagates
```

**Substrate dimension is caller-resolved.** GenoADME's batch pipeline:
1. Looks up SMILES → drug name (or InChIKey → drug name) in their meta-analysis table
2. Selects the appropriate override scale per (gene, drug) tuple
3. Passes resolved `{gene: scale}` to `predict()`

Sisyphus does not maintain a substrate-keyed override registry — that responsibility lives in the caller's curation layer (GenoADME's case: their PGx meta-analysis tables).

## 4. API

### 4.1 `apply_phenotype_to_graph` extension

```python
def apply_phenotype_to_graph(
    graph: BodyGraph,
    phenotypes: dict[str, str],
    node: str = "liver",
    phenotype_scale_overrides: dict[str, float] | None = None,
) -> BodyGraph:
    """Return a new BodyGraph with enzyme/transporter abundances scaled.

    Args:
        graph: Input body graph (unchanged).
        phenotypes: {tag: phenotype_code} from parse_phenotype_spec.
        node: Which node to scale. Default "liver".
        phenotype_scale_overrides: Optional {gene: effective_scale}.
            When provided AND a gene matches a key in phenotypes, the
            override value replaces PHENOTYPE_SCALES[phenotype] for
            that gene's effect on the matched node's enzyme/transporter
            abundance. Values are caller-supplied and caller-justified —
            Sisyphus does not endorse specific values. Negative values
            raise ValueError. Default None preserves current behavior.

    Returns: New BodyGraph with scaled Distribution for matched
        enzymes/transporters.
    """
```

Validation:
- Negative override values → `ValueError` (mathematically meaningless: a negative abundance multiplier).
- Override key not in `phenotypes` dict → silent no-op + `logger.info("phenotype: override key %s not in phenotypes dict, ignored", tag)`.
- Override key in `phenotypes` but tag not in graph → existing "tag not found" warning surfaces (override has no effect because there's nothing to scale).
- No upper bound on positive scale values — caller responsibility.

### 4.2 `predict()` extension

```python
def predict(
    smiles: str,
    dose_mg: float = 100.0,
    route: str = "oral",
    n_mc_samples: int = 0,
    kp_method: str | None = None,
    phenotypes: dict[str, str] | None = None,
    phenotype_scale_overrides: dict[str, float] | None = None,
    *,
    infusion_duration_min: float | None = None,
) -> PredictionResult:
```

`predict()` forwards `phenotype_scale_overrides` verbatim to its internal `apply_phenotype_to_graph` call. No other change.

## 5. Implementation

### 5.1 `src/sisyphus/predict/phenotype.py`

In the `for tag, phenotype in phenotypes.items()` loop (around line 131), after `scale = PHENOTYPE_SCALES[phenotype]`:

```python
        scale = PHENOTYPE_SCALES[phenotype]
        if phenotype_scale_overrides is not None and tag in phenotype_scale_overrides:
            override_scale = phenotype_scale_overrides[tag]
            if override_scale < 0:
                raise ValueError(
                    f"phenotype_scale_overrides[{tag!r}]={override_scale} is negative"
                )
            logger.info(
                "phenotype: override %s default scale %.3f -> %.3f",
                tag, scale, override_scale,
            )
            scale = override_scale
```

After the loop, log any override keys that were not in `phenotypes`:
```python
if phenotype_scale_overrides:
    unused_overrides = set(phenotype_scale_overrides) - set(phenotypes)
    if unused_overrides:
        logger.info("phenotype: overrides for %s not in phenotypes dict, ignored", sorted(unused_overrides))
```

### 5.2 `src/sisyphus/pipeline/predict.py`

Add `phenotype_scale_overrides: dict[str, float] | None = None` to the `predict()` signature. In the existing `if phenotypes: graph = apply_phenotype_to_graph(graph, phenotypes)` block (post-v0.3.2 around line 269-271), forward the kwarg:

```python
        if phenotypes:
            from sisyphus.predict.phenotype import apply_phenotype_to_graph
            graph = apply_phenotype_to_graph(
                graph, phenotypes,
                phenotype_scale_overrides=phenotype_scale_overrides,
            )
```

## 6. Tests

### 6.1 Unit (`tests/unit/test_phenotype_scale_overrides.py`, NEW)

- `apply_phenotype_to_graph` with `phenotype_scale_overrides=None` → identical to no kwarg
- `apply_phenotype_to_graph` with `phenotype_scale_overrides={}` → identical to no kwarg
- `apply_phenotype_to_graph` with `{"SLCO1B1": 0.30}` → liver.transporters["OATP1B1"].mean reduced to 30% (vs 10% default for PM)
- `apply_phenotype_to_graph` with `{"CYP1A2": 0.50}` + phenotypes `{"CYP1A2": "PM"}` → liver.enzymes["CYP1A2"].mean × 0.50 (vs default 0.10)
- Negative override raises ValueError
- Override key for gene not in phenotypes → no-op + logger.info captured
- Multiple genes overridden simultaneously work independently

### 6.2 Integration (`tests/integration/test_phenotype_scale_overrides_pravastatin.py`, NEW)

End-to-end via `predict()`:
- `predict(pravastatin_smiles, phenotypes={"SLCO1B1": "PM"}).cmax_pm` (default 0.10)
- `predict(pravastatin_smiles, phenotypes={"SLCO1B1": "PM"}, phenotype_scale_overrides={"SLCO1B1": 0.30}).cmax_pm_override` (override 0.30)
- Default PM Cmax > Override PM Cmax (override is closer to EM, less compression of OATP1B1, lower Cmax shift)
- The ratio of (PM/EM Cmax) is smaller under override than under default (compresses toward EM)
- `predict(non_pravastatin)` with the SAME override but different SMILES is identical to the no-override call (override only applies when the gene is in `phenotypes`, which is a per-call argument — but if `phenotypes={"SLCO1B1": "PM"}` is also passed for non-pravastatin, the override applies to that gene's abundance scaling regardless of substrate; the substrate dimension is the SMILES being predicted, which determines whether the drug actually USES the OATP1B1 transporter path)

(The third bullet documents an important subtlety: the override applies at graph-level, so it scales abundance regardless of which drug is being predicted. The substrate-scoping is implicit — if the SMILES doesn't use OATP1B1, scaling OATP1B1 abundance has no effect on its Cmax. Caller should not pass overrides for genes their SMILES doesn't actually depend on.)

### 6.3 Backward compatibility

- `tests/unit/test_phenotype.py` (existing) — must pass unchanged
- `tests/unit/test_pipeline_phenotypes.py` (existing) — must pass unchanged
- `tests/integration/test_phenotype_nat2.py`, `test_phenotype_ugt1a1.py`, `test_phenotype_cyp_propagation.py` (v0.3.2) — must pass unchanged
- `tests/integration/test_holdout_regression.py` — Meta 2.679 pin holds (production `predict()` does not pass `phenotype_scale_overrides`)

## 7. Failure modes

### 7.1 Override applied to gene not in graph
Existing `apply_phenotype_to_graph` already warns "tag not found" when the gene is not in `liver.enzymes` or `liver.transporters`. Override is a no-op in this case (nothing to scale). Same warning surfaces. No new failure mode.

### 7.2 Override conflicts with `phenotypes` mapping
If user passes `phenotypes={"SLCO1B1": "PM"}` and `phenotype_scale_overrides={"SLCO1B1": 0.30}`, the override replaces the PM scale. This is the intended semantic. Documented in the docstring.

### 7.3 Caller passes substrate-irrelevant override
e.g., `predict(metoprolol_smiles, phenotypes={"SLCO1B1": "PM"}, phenotype_scale_overrides={"SLCO1B1": 0.30})`. Metoprolol doesn't use OATP1B1. The override scales liver.transporters["OATP1B1"].mean × 0.30, but metoprolol's flux through OATP1B1 is zero (no transporter_kinetics for metoprolol), so Cmax is unaffected. Silent zero, identical to default-PM behavior for metoprolol. **No bug** — caller-supplied overrides for irrelevant genes are silently no-op via the engine's identity-blind invariant.

### 7.4 Caller-provided values are wrong
Sisyphus doesn't validate against literature. Caller's responsibility. Documented in docstring: "values are caller-supplied and caller-justified — Sisyphus does not endorse specific values."

## 8. Scope

### In scope (v0.3.3)

- Add `phenotype_scale_overrides` kwarg to `apply_phenotype_to_graph` and `predict()`
- Override branch in scale lookup loop with logger.info
- Negative-value ValueError
- Unit tests (7+ cases) + integration test (pravastatin empirical)
- v0.3.3 / `feat/phenotype-scale-overrides` branch + PR

### Out of scope

- Sisyphus-shipped substrate-specific calibration tables — caller-side only
- 3-level dict `{gene: {phenotype: {substrate: scale}}}` — substrate dimension implicit in per-call SMILES, phenotype implicit in per-call `phenotypes` argument; flat `{gene: scale}` is mechanically equivalent
- Override of `apply_phenotype_to_graph`'s `node` argument — only `liver` is supported in current architecture
- Multi-node overrides (gut_wall enzyme phenotype scaling) — separate concern, not requested

## 9. Estimated breakdown (4 tasks)

1. Failing test (`tests/unit/test_phenotype_scale_overrides.py`): Override changes the scaled abundance for a matching gene; test currently fails because kwarg not accepted.
2. `phenotype.py` extension: Add kwarg + override branch + logger.info. Test passes.
3. `pipeline/predict.py` extension: Forward kwarg through. Add explicit `phenotype_scale_overrides` parameter to `predict()` signature.
4. Integration test (`tests/integration/test_phenotype_scale_overrides_pravastatin.py`): pravastatin SLCO1B1:PM with override 0.30 produces compressed Cmax shift vs default 0.10.

## 10. Acceptance criteria

- [ ] `apply_phenotype_to_graph(graph, phenotypes={"SLCO1B1": "PM"}, phenotype_scale_overrides={"SLCO1B1": 0.30})` scales liver.transporters["OATP1B1"].mean to 30% of original (not 10%)
- [ ] `predict(SMILES, phenotypes={"SLCO1B1": "PM"}, phenotype_scale_overrides={"SLCO1B1": 0.30})` produces a Cmax that's between EM and default-PM (override compresses toward EM)
- [ ] Negative override raises ValueError
- [ ] All existing phenotype tests pass unchanged
- [ ] 107-holdout AAFE 2.679 pin holds
- [ ] CI green

## 11. References

- Issue [#31](https://github.com/jam-sudo/Sisyphus/issues/31)
- PR #32 (`1e06ded`) — v0.3.2 NAT2 + UGT1A1 + back-solve cancellation fix (predecessor; phenotype propagation infrastructure)
- GenoADME v0.3 meta-analysis: Niemi 2006 men-stratum SLCO1B1/pravastatin PM/EM AUC ratio 3.32 [1.74, 4.91]
- CLAUDE.md Invariants 1, 6 (engine identity-blind, no drug-specific branches in code — registry of overrides lives in caller, not Sisyphus)

## 12. Self-review

- ✅ Placeholder scan: 0
- ✅ Internal consistency: data + code + tests align
- ✅ Scope check: single API surface change (~50-100 lines), 4 tasks
- ✅ Ambiguity: override semantic (gene-keyed flat dict; substrate implicit in SMILES; phenotype implicit in phenotypes dict) explicitly documented in §3 and §4.1
- ✅ Backward compatibility: documented in §6.3 with named test files
