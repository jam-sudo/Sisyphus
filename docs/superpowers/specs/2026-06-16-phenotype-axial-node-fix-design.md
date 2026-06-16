# `apply_phenotype_to_graph` under axial expansion — correctness fix

> **Context.** Prerequisite for the first-pass-saturation arm of the PGx nonlinear-genotype
> validation (axial liver). Discovered during the 2026-06-16 axial-hepatic-saturation probe:
> PGx phenotype scaling silently no-ops on an axially-expanded organ, so every axial genotype
> fold came back exactly 1.0. Small, self-contained engine-correctness fix.

## 1. Purpose & non-goals

**Purpose.** Make `apply_phenotype_to_graph` correctly scale enzyme/transporter abundances when
the target organ has been **axially expanded** (`graph.axial.expand_axial`), by resolving target
nodes via the organ identity (`name` **or** `lookup_name`) instead of a literal node name.

**Non-goals.**
- **Non-axial behaviour is bit-identical.** A graph with a literal `liver` node scales exactly
  as today. Headline **2.731 untouched** (production never sets phenotypes by default, and never
  uses axial by default; the production PGx path on a non-axial graph is unchanged).
- No change to the scaling math, the override mechanism, or the public signature (the `node`
  parameter keeps its meaning — now interpreted as the organ *identity*, matching either form).
- Not the genotype validation itself (separate spec); not new PGx genes.

## 2. Background: the bug

`apply_phenotype_to_graph(graph, phenotypes, node="liver", ...)` does `if node not in
graph.nodes: warn + return graph` (silent no-op), else scales `graph.nodes["liver"]`.

`expand_axial` (`graph/axial.py`) replaces an organ with N serial sub-tanks
`liver__ax1..N`, each carrying `lookup_name = organ` (= `"liver"`) and `1/N` of the parent's
enzyme abundance (the engine is identity-blind: it resolves Kp/PS via `lookup_name`, never the
literal name). After expansion `"liver"` is **not** a key in `graph.nodes`, so the phenotype
call hits the `warn + return graph` branch and applies **no scaling** — the PM/EM genotype fold
collapses to exactly 1.0. The probe confirmed this is why axial genotype folds were inert until
worked around by scaling pre-expansion.

## 3. The fix

Resolve the target node(s) by organ identity:

```
targets = [n for n in graph.nodes.values()
           if n.name == node or getattr(n, "lookup_name", None) == node]
if not targets:
    warn + return graph        # preserve current no-op-on-truly-absent behaviour
```

Apply the existing per-node enzyme/transporter scaling to **each** target node (refactor the
current single-node scaling block into a helper applied per target), and return a new BodyGraph
with all matched nodes replaced. Because `expand_axial` splits abundance `1/N` across sub-tanks,
scaling each sub-tank by the same factor scales the organ total by that factor — correct.

**Backward compatibility.** On a non-axial graph exactly one node matches (`name == "liver"`;
no other node has `lookup_name == "liver"`), so the result is identical to today — same single
node, same scaling, same returned graph. Only genuinely-new behaviour is the axial case
(previously a silent no-op).

## 4. Testing

- **Non-axial unchanged (headline-safe):** on the reference graph, `apply_phenotype_to_graph(g,
  {"CYP2D6":"PM"})` scales `liver.enzymes["CYP2D6"]` by 0.10 exactly as before; a sampled
  prediction is bit-identical to pre-fix.
- **Axial scaling:** build the reference graph, set `liver.axial_subcompartments=5`, the liver
  clearance edge `model="parallel_tube"`, `expand_axial`; then `apply_phenotype_to_graph(...,
  {"CYP2D6":"IM"})` scales **every** `liver__ax{i}` `CYP2D6` abundance by 0.50, and the summed
  organ abundance is `0.50 ×` the pre-scale total.
- **Symptom regression:** the PM genotype AUC-fold on an axial liver is **≠ 1.0** (was exactly
  1.0 pre-fix) — the bug's observable.
- **Truly-absent node:** `apply_phenotype_to_graph(g, {...}, node="nonexistent")` still
  warns + returns the graph unchanged.
- **Override path:** `phenotype_scale_overrides` still works (used by the genotype harness),
  applied to each axial sub-tank.

## 5. Components

- Modify `src/sisyphus/predict/phenotype.py`: target-resolution + per-target scaling in
  `apply_phenotype_to_graph` (one function; ~refactor the single-node block into a loop).
- Test `tests/unit/test_phenotype_axial.py` (new).

## 6. Out of scope

The two-arm genotype validation (systemic multi-dose + first-pass axial) — its own spec.
Teaching other graph consumers about `lookup_name` (only `apply_phenotype_to_graph` is fixed
here). Any production-path axial activation.
