# `apply_phenotype_to_graph` under axial expansion — fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `apply_phenotype_to_graph` correctly scale enzyme/transporter abundances when the target organ has been axially expanded (`expand_axial`), by resolving target nodes via organ identity (`name` **or** `lookup_name`) instead of a literal node key — while keeping the non-axial path bit-identical.

**Architecture:** One function in `src/sisyphus/predict/phenotype.py` is refactored: the single-node scaling block becomes a pure helper `_scale_node(...)`, and `apply_phenotype_to_graph` resolves a **list** of target nodes (any node whose `name == node` or `lookup_name == node`) and applies the helper to each. Non-axial graphs match exactly one node → identical result → headline 2.731 untouched. Axial graphs match every `liver__ax{i}` sub-tank → all get scaled. Pure graph transform; no engine change.

**Tech Stack:** Python 3.10+, frozen dataclasses (`dataclasses.replace`), `pytest`. Engine+scipy at `/opt/miniconda3/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-16-phenotype-axial-node-fix-design.md`

**Constraints (load-bearing):**
- Commit as `jam-sudo <jam-sudo@users.noreply.github.com>`. **NEVER** add `Co-Authored-By: Claude` / any AI footer / "Generated with" line. Use `git commit --no-verify`.
- Stage ONLY the two files this plan touches. **NEVER** `git add README.md` or any untracked workspace file (`ChemRxiv_submission_metadata.md`, `docs/numeric_drift_followups_2026-06-12.md`, `docs/preprint_v3_revised_evaluation.md`). After each commit run `git show --name-only HEAD` to verify scope.
- Headline Meta AAFE 2.731 is inviolable: no `predict()` / `reference_man.yaml` / holdout change. Production never sets phenotypes by default and never uses axial by default, so this is harness/PGx-path-only.
- Run tests with `/opt/miniconda3/bin/python -m pytest`. `ruff check src tests` must pass (line-length 100).

---

## File Structure

- **Modify** `src/sisyphus/predict/phenotype.py` — extract `_scale_node` helper; rewrite `apply_phenotype_to_graph` target resolution + per-target loop. No signature change.
- **Create** `tests/unit/test_phenotype_axial.py` — non-axial bit-identity regression, axial per-sub-tank scaling, symptom regression (abundance changed where it was previously a silent no-op), truly-absent node warns, override path on axial.

---

### Task 1: Refactor scaling into `_scale_node` (behaviour-preserving)

**Files:**
- Modify: `src/sisyphus/predict/phenotype.py`
- Test: `tests/unit/test_phenotype_axial.py` (new)

The first move is a pure refactor: pull the per-node enzyme/transporter scaling out of `apply_phenotype_to_graph` into a helper, with `apply_phenotype_to_graph` still operating on the single `graph.nodes[node]` target. This must leave every existing `test_phenotype.py` / `test_phenotype_scale_overrides.py` test green (behaviour-preserving) before we change target resolution in Task 2.

- [ ] **Step 1: Write the failing test** (new file, non-axial bit-identity — this is the headline-safety guard and will keep passing through Task 2)

```python
"""Unit tests for apply_phenotype_to_graph under axial expansion (and its
non-axial bit-identity guard). Spec: 2026-06-16-phenotype-axial-node-fix-design.md
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from sisyphus.graph.axial import expand_axial
from sisyphus.graph.builder import build_from_yaml
from sisyphus.graph.types import ClearanceEdge
from sisyphus.predict.phenotype import apply_phenotype_to_graph

ROOT = Path(__file__).resolve().parent.parent.parent
_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"


def _ref_graph():
    return build_from_yaml(_PHYS)


def test_nonaxial_scaling_is_bit_identical():
    """On a normal (non-axial) graph, the liver CYP2D6 abundance is scaled by
    exactly the PM factor (0.10) — unchanged from pre-fix behaviour. This is the
    headline-safety guard: production's non-axial PGx path must not move."""
    g = _ref_graph()
    base = g.nodes["liver"].enzymes["CYP2D6"].mean
    out = apply_phenotype_to_graph(g, {"CYP2D6": "PM"})
    assert out.nodes["liver"].enzymes["CYP2D6"].mean == pytest.approx(base * 0.10, rel=0, abs=0)
    # Untouched enzymes preserved exactly.
    assert out.nodes["liver"].enzymes["CYP3A4"].mean == g.nodes["liver"].enzymes["CYP3A4"].mean
    # CV preserved (population variability rides on top of the phenotype).
    assert out.nodes["liver"].enzymes["CYP2D6"].cv == g.nodes["liver"].enzymes["CYP2D6"].cv
```

- [ ] **Step 2: Run test to verify it passes already** (the refactor must preserve this exact behaviour)

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_phenotype_axial.py::test_nonaxial_scaling_is_bit_identical -v`
Expected: PASS (current code already scales the liver node correctly; this test pins it so the refactor can't break it).

- [ ] **Step 3: Extract the `_scale_node` helper**

In `src/sisyphus/predict/phenotype.py`, add a module-level helper above `apply_phenotype_to_graph` that contains the exact scaling logic currently inlined (enzyme vs transporter, override handling, applied/unknown tracking). It returns the new node plus the applied/unknown lists; it does **no** logging (the caller aggregates and logs once):

```python
def _scale_node(
    target: "Node",  # noqa: F821  (BodyGraph node; typed via graph.types.Node)
    phenotypes: dict[str, str],
    phenotype_scale_overrides: dict[str, float] | None,
) -> tuple["Node", list[str], list[str]]:  # noqa: F821
    """Scale one node's enzyme/transporter abundances by the phenotype factors.

    Pure: returns ``(new_node, applied, unknown)`` and does not log. ``applied``
    and ``unknown`` are human-readable tags for the caller to aggregate.
    """
    target_enzymes = target.enzymes
    target_transporters = getattr(target, "transporters", {}) or {}
    new_enzymes: dict[str, Distribution] = dict(target_enzymes)
    new_transporters: dict[str, Distribution] = dict(target_transporters)
    applied: list[str] = []
    unknown: list[str] = []

    for tag, phenotype in phenotypes.items():
        scale = PHENOTYPE_SCALES[phenotype]
        if phenotype_scale_overrides is not None and tag in phenotype_scale_overrides:
            override_scale = phenotype_scale_overrides[tag]
            if override_scale < 0:
                raise ValueError(
                    f"phenotype_scale_overrides[{tag!r}]={override_scale} is negative"
                )
            scale = override_scale
        transporter_tag = TRANSPORTER_ALIASES.get(tag)
        if transporter_tag is not None:
            if transporter_tag not in target_transporters:
                unknown.append(f"{tag}→{transporter_tag}")
                continue
            old = target_transporters[transporter_tag]
            new_transporters[transporter_tag] = Distribution(
                mean=old.mean * scale, cv=old.cv, dist_type=old.dist_type,
            )
            applied.append(f"{tag}:{phenotype}({scale}×)→transporter")
        else:
            if tag not in target_enzymes:
                unknown.append(tag)
                continue
            old = target_enzymes[tag]
            new_enzymes[tag] = Distribution(
                mean=old.mean * scale, cv=old.cv, dist_type=old.dist_type,
            )
            applied.append(f"{tag}:{phenotype}({scale}×)")

    new_node = replace(target, enzymes=new_enzymes, transporters=new_transporters)
    return new_node, applied, unknown
```

Add the `Node` import for the annotation at the top of the file:

```python
from sisyphus.graph.types import Node
```

(and drop the `# noqa: F821` forward-ref hints once `Node` is imported — annotate `target: Node -> tuple[Node, list[str], list[str]]`).

- [ ] **Step 4: Rewrite `apply_phenotype_to_graph` to call the helper on the single node** (still single-node — Task 2 generalises the target set)

Replace the body after the `if not phenotypes` / `if node not in graph.nodes` guards with:

```python
    target = graph.nodes[node]
    new_node, applied, unknown = _scale_node(target, phenotypes, phenotype_scale_overrides)

    if phenotype_scale_overrides:
        unused_overrides = sorted(set(phenotype_scale_overrides) - set(phenotypes))
        if unused_overrides:
            logger.info(
                "phenotype: overrides for %s not in phenotypes dict, ignored",
                unused_overrides,
            )
    if unknown:
        available = sorted(
            list(target.enzymes)
            + [f"(transporter){t}" for t in (getattr(target, "transporters", {}) or {})]
        )
        logger.warning(
            "phenotype: tags %s not found in %s (available: %s)", unknown, node, available,
        )
    if applied:
        logger.info("phenotype: applied %s at %s", ", ".join(applied), node)

    new_graph = BodyGraph()
    new_graph.nodes = dict(graph.nodes)
    new_graph.nodes[node] = new_node
    new_graph.edges = list(graph.edges)
    new_graph.global_params = dict(graph.global_params)
    return new_graph
```

- [ ] **Step 5: Run the full phenotype suite to verify the refactor preserves behaviour**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_phenotype.py tests/unit/test_phenotype_scale_overrides.py tests/unit/test_pipeline_phenotypes.py tests/unit/test_phenotype_axial.py -v`
Expected: ALL PASS (refactor is behaviour-preserving; the new bit-identity test passes).

- [ ] **Step 6: Lint**

Run: `ruff check src/sisyphus/predict/phenotype.py tests/unit/test_phenotype_axial.py`
Expected: no errors (line-length 100).

- [ ] **Step 7: Commit**

```bash
git add src/sisyphus/predict/phenotype.py tests/unit/test_phenotype_axial.py
git commit --no-verify -m "refactor(phenotype): extract _scale_node helper (behaviour-preserving)"
git show --name-only HEAD
```
Verify the `git show --name-only` output lists ONLY those two files.

---

### Task 2: Resolve targets by `name` OR `lookup_name` (the fix) + axial tests

**Files:**
- Modify: `src/sisyphus/predict/phenotype.py`
- Test: `tests/unit/test_phenotype_axial.py`

Now generalise target resolution so an axially-expanded organ (whose sub-tanks carry `lookup_name == "liver"` but no node is literally named `"liver"`) is scaled on every sub-tank.

- [ ] **Step 1: Write the failing axial tests**

Append to `tests/unit/test_phenotype_axial.py`:

```python
def _axial_liver_graph(n_sub: int = 5):
    """Reference graph with the liver turned into a parallel_tube organ and
    axially expanded into ``n_sub`` serial well_stirred sub-tanks. After
    expand_axial there is NO node literally named 'liver'; the sub-tanks
    'liver__ax{i}' carry lookup_name='liver' and 1/n of the enzyme abundance."""
    g = build_from_yaml(_PHYS)
    g.nodes["liver"] = dataclasses.replace(g.nodes["liver"], axial_subcompartments=n_sub)
    g.edges = [
        dataclasses.replace(e, model="parallel_tube")
        if isinstance(e, ClearanceEdge) and e.source == "liver"
        else e
        for e in g.edges
    ]
    return expand_axial(g)


def _liver_subtanks(graph):
    return [n for n in graph.nodes.values() if (n.lookup_name or n.name) == "liver"]


def test_axial_expansion_produces_subtanks_no_literal_liver():
    """Precondition: the axial graph has sub-tanks (lookup_name='liver') and no
    node literally named 'liver'. Guards the test setup itself."""
    g = _axial_liver_graph(5)
    assert "liver" not in g.nodes
    subs = _liver_subtanks(g)
    assert len(subs) == 5
    assert all("CYP2D6" in s.enzymes for s in subs)


def test_axial_scaling_applies_to_every_subtank():
    """The fix: IM (0.50×) scales CYP2D6 on EVERY liver sub-tank, and the summed
    organ abundance is 0.50× the pre-scale organ total."""
    g = _axial_liver_graph(5)
    pre_total = sum(s.enzymes["CYP2D6"].mean for s in _liver_subtanks(g))
    out = apply_phenotype_to_graph(g, {"CYP2D6": "IM"})
    out_subs = _liver_subtanks(out)
    assert len(out_subs) == 5
    for s in out_subs:
        # each sub-tank held 1/5 of the abundance; each is scaled by 0.50
        assert s.enzymes["CYP2D6"].mean == pytest.approx(pre_total / 5 * 0.50)
    assert sum(s.enzymes["CYP2D6"].mean for s in out_subs) == pytest.approx(pre_total * 0.50)


def test_axial_symptom_regression_abundance_changes():
    """The bug's observable: pre-fix, apply_phenotype was a silent no-op on the
    axial graph (every sub-tank abundance unchanged → genotype fold collapsed to
    exactly 1.0). Post-fix the PM (0.10×) abundance MUST differ from the input —
    the mechanistic precursor of a fold != 1.0 that the v2.2a flux consumes."""
    g = _axial_liver_graph(5)
    pre = {s.name: s.enzymes["CYP2D6"].mean for s in _liver_subtanks(g)}
    out = apply_phenotype_to_graph(g, {"CYP2D6": "PM"})
    post = {s.name: s.enzymes["CYP2D6"].mean for s in _liver_subtanks(out)}
    assert post.keys() == pre.keys()
    assert all(post[k] == pytest.approx(pre[k] * 0.10) for k in pre)
    assert all(post[k] != pytest.approx(pre[k]) for k in pre)  # NOT a no-op


def test_axial_override_path_applies_per_subtank():
    """phenotype_scale_overrides (used by the genotype harness) reaches every
    sub-tank."""
    g = _axial_liver_graph(4)
    pre_total = sum(s.enzymes["CYP2D6"].mean for s in _liver_subtanks(g))
    out = apply_phenotype_to_graph(
        g, {"CYP2D6": "PM"}, phenotype_scale_overrides={"CYP2D6": 0.33}
    )
    assert sum(s.enzymes["CYP2D6"].mean for s in _liver_subtanks(out)) == pytest.approx(
        pre_total * 0.33
    )


def test_truly_absent_node_warns_and_returns_unchanged(caplog):
    """A node that matches neither name nor lookup_name still warns + returns the
    graph object unchanged (preserved no-op behaviour)."""
    import logging

    g = _ref_graph()
    with caplog.at_level(logging.WARNING):
        out = apply_phenotype_to_graph(g, {"CYP2D6": "PM"}, node="nonexistent_organ")
    assert out is g
    assert any("nonexistent_organ" in r.getMessage() for r in caplog.records)
```

- [ ] **Step 2: Run the axial tests to verify they fail on the current single-node code**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_phenotype_axial.py -v -k "axial or truly_absent"`
Expected: FAIL — `test_axial_scaling_applies_to_every_subtank`, `test_axial_symptom_regression_abundance_changes`, `test_axial_override_path_applies_per_subtank` fail (current `if node not in graph.nodes` → `"liver"` absent → silent no-op returns the graph unchanged; abundances equal input). `test_truly_absent_node_warns_and_returns_unchanged` and `test_axial_expansion_produces_subtanks_no_literal_liver` should already PASS.

- [ ] **Step 3: Implement target resolution by identity**

In `apply_phenotype_to_graph`, replace the single-node resolution:

```python
    if node not in graph.nodes:
        logger.warning("phenotype: node %r not in graph, skipping", node)
        return graph

    target = graph.nodes[node]
    new_node, applied, unknown = _scale_node(target, phenotypes, phenotype_scale_overrides)
    ...
    new_graph.nodes[node] = new_node
```

with multi-target resolution + a per-target loop:

```python
    targets = [
        n for n in graph.nodes.values()
        if n.name == node or (getattr(n, "lookup_name", "") or "") == node
    ]
    if not targets:
        logger.warning("phenotype: node %r not in graph, skipping", node)
        return graph

    new_nodes = dict(graph.nodes)
    all_applied: list[str] = []
    all_unknown: list[str] = []
    last_target = targets[-1]
    for target in targets:
        new_node, applied, unknown = _scale_node(
            target, phenotypes, phenotype_scale_overrides
        )
        new_nodes[target.name] = new_node
        all_applied.extend(applied)
        all_unknown.extend(unknown)

    if phenotype_scale_overrides:
        unused_overrides = sorted(set(phenotype_scale_overrides) - set(phenotypes))
        if unused_overrides:
            logger.info(
                "phenotype: overrides for %s not in phenotypes dict, ignored",
                unused_overrides,
            )
    if all_unknown:
        available = sorted(
            list(last_target.enzymes)
            + [f"(transporter){t}" for t in (getattr(last_target, "transporters", {}) or {})]
        )
        logger.warning(
            "phenotype: tags %s not found in %s (available: %s)",
            sorted(set(all_unknown)), node, available,
        )
    if all_applied:
        logger.info(
            "phenotype: applied %s at %s (%d node(s))",
            ", ".join(sorted(set(all_applied))), node, len(targets),
        )

    new_graph = BodyGraph()
    new_graph.nodes = new_nodes
    new_graph.edges = list(graph.edges)
    new_graph.global_params = dict(graph.global_params)
    return new_graph
```

Note: on a non-axial graph `targets == [graph.nodes["liver"]]` (the literal liver has `lookup_name == ""`, no other node has `lookup_name == "liver"`), so `new_nodes["liver"]` is replaced exactly as before → `test_nonaxial_scaling_is_bit_identical` still passes.

- [ ] **Step 4: Run the full axial file + the non-axial guard**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_phenotype_axial.py -v`
Expected: ALL PASS (axial scaling + symptom regression + override + truly-absent + bit-identity).

- [ ] **Step 5: Run the whole phenotype + pipeline suite to confirm no regression**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_phenotype.py tests/unit/test_phenotype_scale_overrides.py tests/unit/test_pipeline_phenotypes.py tests/unit/test_phenotype_axial.py -v`
Expected: ALL PASS.

- [ ] **Step 6: Lint**

Run: `ruff check src/sisyphus/predict/phenotype.py tests/unit/test_phenotype_axial.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/sisyphus/predict/phenotype.py tests/unit/test_phenotype_axial.py
git commit --no-verify -m "fix(phenotype): scale every axial sub-tank by resolving node identity (name or lookup_name)"
git show --name-only HEAD
```
Verify the `git show --name-only` output lists ONLY those two files.

---

### Task 3: Headline-isolation sanity + docstring update

**Files:**
- Modify: `src/sisyphus/predict/phenotype.py` (docstring only)

- [ ] **Step 1: Update the `node` parameter docstring to reflect identity resolution**

In `apply_phenotype_to_graph`'s docstring, change the `node` arg description to:

```python
        node: Which organ to scale, by identity. A node matches if its
            ``name`` equals ``node`` OR its ``lookup_name`` equals ``node``.
            On a normal graph this is the single literal node (e.g. "liver").
            On an axially-expanded graph (``graph.axial.expand_axial``) it is
            every sub-tank ``liver__ax{i}`` (which carry ``lookup_name="liver"``),
            so the organ total is scaled correctly. Default "liver".
```

- [ ] **Step 2: Confirm headline path is untouched (no predict()/YAML/holdout change)**

Run: `git diff --name-only main...HEAD`
Expected: exactly `src/sisyphus/predict/phenotype.py` and `tests/unit/test_phenotype_axial.py` (plus this plan + the spec doc if committed on the branch). NO `predict.py`/`pipeline`/`reference_man.yaml`/`holdout.json`/`4track_holdout_predictions.json`.

- [ ] **Step 3: Run the cached-holdout headline pin to prove 2.731 untouched**

Run: `/opt/miniconda3/bin/python -m pytest tests/regression/test_cached_holdout_aafe_is_2p731.py -v` (use the actual pin path if named differently — `grep -rl "2p731\|2.731" tests/regression`).
Expected: PASS (this code path is never on the production predict path; the pin is unaffected).

- [ ] **Step 4: Lint + commit**

```bash
ruff check src/sisyphus/predict/phenotype.py
git add src/sisyphus/predict/phenotype.py
git commit --no-verify -m "docs(phenotype): document node identity resolution under axial expansion"
git show --name-only HEAD
```
Verify scope.

---

## Self-Review

**Spec coverage (against `2026-06-16-phenotype-axial-node-fix-design.md`):**
- §3 fix (resolve by `name` OR `lookup_name`, scale each target, truly-absent → no-op): Task 2 Step 3. ✓
- §3 backward compat (non-axial exactly one match, identical result): `test_nonaxial_scaling_is_bit_identical` (Task 1) + the note in Task 2 Step 3. ✓
- §4 non-axial unchanged / headline-safe: Task 1 Step 1, Task 3 Step 3. ✓
- §4 axial scaling (every sub-tank, summed total = scale×): `test_axial_scaling_applies_to_every_subtank`. ✓
- §4 symptom regression (axial fold ≠ 1.0 → abundance changes, was a no-op): `test_axial_symptom_regression_abundance_changes`. ✓ (mechanistic abundance-level proxy for the AUC-fold observable — deterministic and fast; the v2.2a saturable flux consumes exactly this scaled abundance, so a changed abundance is the necessary-and-sufficient precursor of fold ≠ 1.0.)
- §4 truly-absent node warns + returns unchanged: `test_truly_absent_node_warns_and_returns_unchanged`. ✓
- §4 override path on each sub-tank: `test_axial_override_path_applies_per_subtank`. ✓
- §5 components (modify `phenotype.py`, new `tests/unit/test_phenotype_axial.py`): exactly these two files. ✓

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `_scale_node(target, phenotypes, overrides) -> (Node, list[str], list[str])` defined Task 1, called identically Task 2. `_liver_subtanks` / `_axial_liver_graph` helpers defined once in the test file. `Distribution`, `replace`, `BodyGraph`, `Node`, `PHENOTYPE_SCALES`, `TRANSPORTER_ALIASES` are existing imports in `phenotype.py` (add only `Node`).

**One open verification for the implementer:** confirm the cached-holdout pin's exact test path with `grep -rl "2p731" tests/regression` (Task 3 Step 3) — the file is named `test_cached_holdout_aafe_is_2p731` per CLAUDE.md but verify before running.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-16-phenotype-axial-node-fix.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task + two-stage review, fast iteration.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
