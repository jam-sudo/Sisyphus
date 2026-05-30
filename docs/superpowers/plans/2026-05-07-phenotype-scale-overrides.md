# Phenotype Scale Overrides (v0.3.3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `phenotype_scale_overrides: dict[str, float] | None = None` keyword to both `apply_phenotype_to_graph()` and `predict()` so downstream callers (GenoADME) can inject per-substrate-effective phenotype scales without Sisyphus committing to substrate-specific calibration tables.

**Architecture:** Flat `{gene: scale}` dict — substrate dimension implicit in per-call SMILES, phenotype dimension implicit in per-call `phenotypes` argument. Override replaces `PHENOTYPE_SCALES[phenotype]` for matching gene at scaling time. No engine changes. No calibration registry shipped.

**Tech Stack:** Python 3.10+, pytest, existing phenotype/pipeline modules.

**Spec:** `docs/superpowers/specs/2026-05-07-phenotype-scale-overrides-design.md` (commit `8dd6cf7`).

**Branch:** `feat/phenotype-scale-overrides` from `main` (HEAD `8dd6cf7`, post-PR #32 v0.3.2 merge).

---

## Pre-flight: branch setup

- [ ] **Step 0a: Confirm clean main and create feature branch**

```bash
git status
git checkout -b feat/phenotype-scale-overrides
git log --oneline -3
```

Expected: clean working tree on main; new branch created; HEAD shows `8dd6cf7` (spec) + `1e06ded` (v0.3.2 merge) + earlier.

---

## File Structure

**Create:**
- `tests/unit/test_phenotype_scale_overrides.py` — 7 unit tests for `apply_phenotype_to_graph` extension
- `tests/integration/test_phenotype_scale_overrides_pravastatin.py` — end-to-end via `predict()` with empirical pravastatin SLCO1B1:PM compression

**Modify:**
- `src/sisyphus/predict/phenotype.py` — `apply_phenotype_to_graph` signature + override branch in scale lookup loop
- `src/sisyphus/pipeline/predict.py` — `predict()` signature + forward kwarg to internal `apply_phenotype_to_graph` call
- `docs/claude/experiment-log.md` — v0.3.3 entry (closing operation)

**Untouched (verify diff = 0 lines):**
- `src/sisyphus/engine/*` — engine is identity-blind
- `data/training/4track_holdout_predictions.json` — holdout invariance
- All other tests not listed above (must pass unchanged)

---

## Task 1: Failing unit test for phenotype_scale_overrides

**Why first:** TDD. Locks in the expected API + behavior before implementation. Tests should fail because `apply_phenotype_to_graph` doesn't yet accept the new kwarg.

**Files:**
- Create: `tests/unit/test_phenotype_scale_overrides.py`

- [ ] **Step 1: Write the failing test file**

```python
"""Unit tests for phenotype_scale_overrides kwarg on apply_phenotype_to_graph (issue #31)."""
from __future__ import annotations

import logging
import pathlib

import pytest

from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.phenotype import apply_phenotype_to_graph


def _fresh_graph():
    return build_from_yaml(pathlib.Path("data/physiology/reference_man.yaml"))


def test_overrides_none_preserves_existing():
    """phenotype_scale_overrides=None must produce identical output to no kwarg."""
    g = _fresh_graph()
    a = apply_phenotype_to_graph(g, {"CYP1A2": "PM"})
    b = apply_phenotype_to_graph(g, {"CYP1A2": "PM"}, phenotype_scale_overrides=None)
    assert a.nodes["liver"].enzymes["CYP1A2"].mean == pytest.approx(
        b.nodes["liver"].enzymes["CYP1A2"].mean
    )


def test_overrides_empty_preserves_existing():
    g = _fresh_graph()
    a = apply_phenotype_to_graph(g, {"CYP1A2": "PM"})
    b = apply_phenotype_to_graph(g, {"CYP1A2": "PM"}, phenotype_scale_overrides={})
    assert a.nodes["liver"].enzymes["CYP1A2"].mean == pytest.approx(
        b.nodes["liver"].enzymes["CYP1A2"].mean
    )


def test_override_replaces_default_scale_enzyme():
    """SLCO1B1:PM with override 0.30 scales OATP1B1 abundance to 30% (vs default 10%)."""
    g = _fresh_graph()
    original = g.nodes["liver"].transporters["OATP1B1"].mean

    default_pm = apply_phenotype_to_graph(g, {"SLCO1B1": "PM"})
    overridden = apply_phenotype_to_graph(
        g, {"SLCO1B1": "PM"},
        phenotype_scale_overrides={"SLCO1B1": 0.30},
    )

    assert default_pm.nodes["liver"].transporters["OATP1B1"].mean == pytest.approx(
        original * 0.10
    )
    assert overridden.nodes["liver"].transporters["OATP1B1"].mean == pytest.approx(
        original * 0.30
    )


def test_override_replaces_default_scale_cyp():
    """CYP1A2:PM with override 0.50 scales abundance to 50% (vs default 10%)."""
    g = _fresh_graph()
    original = g.nodes["liver"].enzymes["CYP1A2"].mean

    overridden = apply_phenotype_to_graph(
        g, {"CYP1A2": "PM"},
        phenotype_scale_overrides={"CYP1A2": 0.50},
    )
    assert overridden.nodes["liver"].enzymes["CYP1A2"].mean == pytest.approx(
        original * 0.50
    )


def test_negative_override_raises():
    g = _fresh_graph()
    with pytest.raises(ValueError, match="negative"):
        apply_phenotype_to_graph(
            g, {"CYP1A2": "PM"},
            phenotype_scale_overrides={"CYP1A2": -0.1},
        )


def test_override_for_gene_not_in_phenotypes_logs_info(caplog):
    """Override key for a gene not in the phenotypes dict is silently ignored."""
    g = _fresh_graph()
    caplog.set_level(logging.INFO)
    out = apply_phenotype_to_graph(
        g, {"CYP1A2": "PM"},
        phenotype_scale_overrides={"CYP2C9": 0.20},
    )
    # CYP2C9 abundance unchanged (override not applied because CYP2C9 not in phenotypes)
    assert out.nodes["liver"].enzymes["CYP2C9"].mean == pytest.approx(
        g.nodes["liver"].enzymes["CYP2C9"].mean
    )
    # logger.info note about ignored override
    info_records = [r for r in caplog.records if r.levelno == logging.INFO and "ignored" in r.getMessage()]
    assert info_records, "expected logger.info about ignored override key"


def test_multiple_genes_overridden_independently():
    g = _fresh_graph()
    cyp_orig = g.nodes["liver"].enzymes["CYP1A2"].mean
    nat2_orig = g.nodes["liver"].enzymes["NAT2"].mean

    out = apply_phenotype_to_graph(
        g,
        {"CYP1A2": "PM", "NAT2": "IM"},
        phenotype_scale_overrides={"CYP1A2": 0.40, "NAT2": 0.65},
    )
    assert out.nodes["liver"].enzymes["CYP1A2"].mean == pytest.approx(cyp_orig * 0.40)
    assert out.nodes["liver"].enzymes["NAT2"].mean == pytest.approx(nat2_orig * 0.65)
```

- [ ] **Step 2: Run test to verify it fails (kwarg not accepted)**

Run: `pytest tests/unit/test_phenotype_scale_overrides.py -v`

Expected: TypeError on `phenotype_scale_overrides` keyword (function doesn't accept it). All 7 tests FAIL on this signature error.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_phenotype_scale_overrides.py
git commit -m "$(cat <<'EOF'
test(phenotype): failing tests for phenotype_scale_overrides (TDD target)

7 unit tests locking in the API surface from spec section 4.1:
- None / {} preserve existing behavior
- Override replaces PHENOTYPE_SCALES default for matching (gene, phenotype) tuple
- Works for both transporter (SLCO1B1 -> OATP1B1) and enzyme (CYP1A2) paths
- Negative value raises ValueError
- Override key for gene not in phenotypes dict logs info, no scale applied
- Multiple gene overrides apply independently

Tests fail pre-implementation (signature does not accept new kwarg).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement phenotype_scale_overrides in phenotype.py

**Why:** Closes the 7 failing tests from Task 1.

**Files:**
- Modify: `src/sisyphus/predict/phenotype.py:97-172` (`apply_phenotype_to_graph`)

- [ ] **Step 1: Locate the current function body**

Run: `sed -n '97,172p' src/sisyphus/predict/phenotype.py`

Expected: existing function `apply_phenotype_to_graph(graph, phenotypes, node="liver")` with the for-loop iterating phenotypes and applying `PHENOTYPE_SCALES[phenotype]` scaling.

- [ ] **Step 2: Add `phenotype_scale_overrides` kwarg + override branch**

Open `src/sisyphus/predict/phenotype.py`. Update the function signature to:

```python
def apply_phenotype_to_graph(
    graph: BodyGraph,
    phenotypes: dict[str, str],
    node: str = "liver",
    phenotype_scale_overrides: dict[str, float] | None = None,
) -> BodyGraph:
```

Update the docstring to add this Args entry (between `node:` and the `Returns:`):

```python
        phenotype_scale_overrides: Optional ``{gene: effective_scale}`` dict.
            When provided AND a gene matches a key in ``phenotypes``, the
            override value replaces ``PHENOTYPE_SCALES[phenotype]`` for that
            gene's effect on the matched node's enzyme/transporter abundance.
            Values are caller-supplied and caller-justified — Sisyphus does
            not endorse specific values. Negative values raise ValueError.
            Default ``None`` preserves current behavior.
```

In the body, locate the line `scale = PHENOTYPE_SCALES[phenotype]` (currently around line 132). Replace that line and the immediately-following code with:

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

After the `for tag, phenotype in phenotypes.items():` loop closes (and before the `if unknown:` block, currently around line 157), insert:

```python
    if phenotype_scale_overrides:
        unused_overrides = sorted(set(phenotype_scale_overrides) - set(phenotypes))
        if unused_overrides:
            logger.info(
                "phenotype: overrides for %s not in phenotypes dict, ignored",
                unused_overrides,
            )
```

- [ ] **Step 3: Run unit tests to verify they pass**

Run: `pytest tests/unit/test_phenotype_scale_overrides.py -v`

Expected: 7 PASS.

- [ ] **Step 4: Run existing phenotype tests for backward compatibility**

Run: `pytest tests/unit/test_phenotype.py tests/unit/test_pipeline_phenotypes.py -v`

Expected: All PASS unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/predict/phenotype.py
git commit -m "$(cat <<'EOF'
feat(phenotype): phenotype_scale_overrides kwarg on apply_phenotype_to_graph

Override branch in scale-lookup loop: when a gene tag appears in both
phenotypes and phenotype_scale_overrides, the override value replaces
PHENOTYPE_SCALES[phenotype] for that gene's abundance scaling.
logger.info notes both successful overrides and unused override keys.

Validation: negative values raise ValueError. No upper bound on positive
values (caller responsibility). Backward-compatible: None / {} preserve
existing behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire phenotype_scale_overrides through predict()

**Why:** Exposes the override at the pipeline-level entry point so callers (e.g., GenoADME) can use `predict()` directly without dropping to graph-layer APIs.

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py` (signature + forward to apply_phenotype_to_graph)

- [ ] **Step 1: Locate the predict() signature**

Run: `grep -n "^def predict" src/sisyphus/pipeline/predict.py`

Expected: line ~70-80 starting `def predict(...)` with existing kwargs (smiles, dose_mg, route, n_mc_samples, kp_method, phenotypes, infusion_duration_min).

- [ ] **Step 2: Add `phenotype_scale_overrides` kwarg to predict() signature**

Add the new kwarg in the same group as `phenotypes` (visually adjacent — they're a related pair):

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

(If the existing signature uses different formatting/order, preserve that — only insert `phenotype_scale_overrides` immediately after `phenotypes`.)

- [ ] **Step 3: Forward the kwarg to apply_phenotype_to_graph**

Locate the existing `if phenotypes: ... apply_phenotype_to_graph(graph, phenotypes)` block (post-v0.3.2, around line 269-271). Update to:

```python
        if phenotypes:
            from sisyphus.predict.phenotype import apply_phenotype_to_graph
            graph = apply_phenotype_to_graph(
                graph, phenotypes,
                phenotype_scale_overrides=phenotype_scale_overrides,
            )
```

- [ ] **Step 4: Update predict() docstring**

Find the existing `phenotypes:` docstring entry (around line 105-115). Add an analogous entry immediately after, e.g.:

```python
        phenotype_scale_overrides: Optional ``{gene: effective_scale}`` dict
            forwarded to ``apply_phenotype_to_graph``. When provided and a
            gene matches a key in ``phenotypes``, the override replaces the
            CPIC default scale for that gene. Values are caller-supplied
            and caller-justified — Sisyphus does not endorse specific values.
```

- [ ] **Step 5: Spot-check pipeline integration**

Run:

```bash
python3 << 'PYEOF'
from sisyphus.pipeline.predict import predict
prava = ("CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
         "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O")

em = predict(prava, dose_mg=40.0, phenotypes={"SLCO1B1": "EM"})
default_pm = predict(prava, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"})
override_pm = predict(
    prava, dose_mg=40.0,
    phenotypes={"SLCO1B1": "PM"},
    phenotype_scale_overrides={"SLCO1B1": 0.30},
)
print("EM Cmax:", em.engine_pk.cmax.mean)
print("Default PM Cmax:", default_pm.engine_pk.cmax.mean)
print("Override PM (0.30) Cmax:", override_pm.engine_pk.cmax.mean)
print("Default PM/EM ratio:", default_pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean)
print("Override PM/EM ratio:", override_pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean)
print("Override compresses:", em.engine_pk.cmax.mean < override_pm.engine_pk.cmax.mean < default_pm.engine_pk.cmax.mean)
PYEOF
```

Expected: Override PM Cmax falls between EM and Default PM (compression toward EM); ordering `EM < Override PM < Default PM`. Override PM/EM ratio is smaller than Default PM/EM ratio.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/pipeline/predict.py
git commit -m "$(cat <<'EOF'
feat(pipeline): forward phenotype_scale_overrides through predict()

predict() now accepts phenotype_scale_overrides kwarg adjacent to
phenotypes. Forwarded verbatim to apply_phenotype_to_graph. Default
None preserves existing behavior. 107-holdout invariant (production
benchmark uses default None).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Integration test — pravastatin SLCO1B1 PM/EM compression

**Why:** Empirical end-to-end gate. Confirms the override actually compresses the PM/EM Cmax shift toward EM (the GenoADME use case).

**Files:**
- Create: `tests/integration/test_phenotype_scale_overrides_pravastatin.py`

- [ ] **Step 1: Write the integration test**

```python
"""Integration test for phenotype_scale_overrides via predict() — pravastatin SLCO1B1 (#31).

End-to-end gate: override 0.30 (vs CPIC default 0.10) for SLCO1B1:PM
compresses pravastatin Cmax toward EM. The GenoADME use case is
calibrating Sisyphus's PM/EM AUC ratio toward Niemi 2006 men-stratum
central 3.32 (vs current default ~4.5); the equivalent Cmax-side
ordering test is what we verify here.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


_PRAVASTATIN = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)


@pytest.mark.slow
def test_pravastatin_slco1b1_pm_override_compresses_toward_em():
    """SLCO1B1:PM override 0.30 must produce a Cmax between EM and default-PM.

    Compression ordering: EM < Override-PM < Default-PM.
    Default PM scales OATP1B1 abundance × 0.10 → maximum uptake reduction
    → highest Cmax. Override 0.30 scales × 0.30 → less uptake reduction
    → Cmax closer to EM. EM is unchanged baseline.
    """
    em = predict(_PRAVASTATIN, dose_mg=40.0, phenotypes={"SLCO1B1": "EM"})
    default_pm = predict(_PRAVASTATIN, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"})
    override_pm = predict(
        _PRAVASTATIN, dose_mg=40.0,
        phenotypes={"SLCO1B1": "PM"},
        phenotype_scale_overrides={"SLCO1B1": 0.30},
    )
    assert em.engine_pk is not None and default_pm.engine_pk is not None and override_pm.engine_pk is not None

    em_cmax = em.engine_pk.cmax.mean
    default_pm_cmax = default_pm.engine_pk.cmax.mean
    override_pm_cmax = override_pm.engine_pk.cmax.mean

    assert em_cmax < override_pm_cmax < default_pm_cmax, (
        f"compression ordering violated: EM={em_cmax:.4f}, "
        f"Override-PM={override_pm_cmax:.4f}, Default-PM={default_pm_cmax:.4f}"
    )

    # Override compresses the PM/EM ratio toward unity
    default_ratio = default_pm_cmax / em_cmax
    override_ratio = override_pm_cmax / em_cmax
    assert 1.0 < override_ratio < default_ratio, (
        f"override ratio {override_ratio:.3f} not between 1.0 and "
        f"default {default_ratio:.3f}"
    )


@pytest.mark.slow
def test_pravastatin_no_override_unchanged():
    """Calling predict() without phenotype_scale_overrides must produce
    identical Cmax to omitting the kwarg entirely (backward compat)."""
    a = predict(_PRAVASTATIN, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"})
    b = predict(
        _PRAVASTATIN, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"},
        phenotype_scale_overrides=None,
    )
    c = predict(
        _PRAVASTATIN, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"},
        phenotype_scale_overrides={},
    )
    assert a.engine_pk.cmax.mean == pytest.approx(b.engine_pk.cmax.mean)
    assert a.engine_pk.cmax.mean == pytest.approx(c.engine_pk.cmax.mean)
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/integration/test_phenotype_scale_overrides_pravastatin.py -v`

Expected: 2 PASS.

- [ ] **Step 3: Run full test suite for regression check**

Run: `pytest tests/unit tests/regression tests/integration -q --no-header 2>&1 | tail -10`

Expected: all PASS or pre-existing xfails only (rosuvastatin / atorvastatin / fluvastatin Peff). No new regressions.

- [ ] **Step 4: Run holdout invariance check**

Run: `pytest tests/integration/test_holdout_regression.py -v`

Expected: PASS (Meta 2.679 pin holds).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_phenotype_scale_overrides_pravastatin.py
git commit -m "$(cat <<'EOF'
test(integration): pravastatin SLCO1B1:PM override compression (issue #31)

End-to-end gate via predict(): override 0.30 (vs CPIC default 0.10) for
SLCO1B1:PM compresses pravastatin Cmax toward EM. Ordering:
EM < Override-PM < Default-PM, override PM/EM ratio < default PM/EM ratio.

Plus backward-compat gate: None / {} produce identical Cmax to no kwarg.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Closing operations

- [ ] **Add experiment-log entry**

Open `docs/claude/experiment-log.md` and prepend (above the most-recent v0.3.2 entry):

```markdown
## 2026-05-07 — v0.3.3 phenotype_scale_overrides API hook

**Branch**: `feat/phenotype-scale-overrides` (PR pending)
**Spec**: `docs/superpowers/specs/2026-05-07-phenotype-scale-overrides-design.md` (commit `8dd6cf7`)
**Closes**: issue #31 (capability request from GenoADME — per-substrate effective phenotype scale injection)

### What shipped

`apply_phenotype_to_graph()` and `predict()` now accept a `phenotype_scale_overrides: dict[str, float] | None = None` keyword. When provided AND a gene matches a key in `phenotypes`, the override value replaces `PHENOTYPE_SCALES[phenotype]` for that gene's effect on the matched node's enzyme/transporter abundance. Negative values raise ValueError; no upper bound on positive values (caller responsibility).

Signature shape: flat `{gene: scale}` dict — substrate dimension implicit in per-call SMILES, phenotype dimension implicit in per-call `phenotypes` argument. Mechanically equivalent to GenoADME's originally-proposed 3-level `{gene: {phenotype: {substrate: scale}}}`, simpler.

Sisyphus ships **no calibration tables**. Caller (GenoADME's case) is responsible for resolving `(SMILES, gene, phenotype) → override scale` from their own meta-analysis tables, and passing the resolved scale via `phenotype_scale_overrides` per call.

### Empirical example (pravastatin)

| call | OATP1B1 abundance scaling | Cmax shift vs EM |
|---|---|---|
| `phenotypes={"SLCO1B1": "EM"}` | 1.00× | 1.0× (baseline) |
| `phenotypes={"SLCO1B1": "PM"}`, no override | 0.10× (CPIC) | ~3.0× |
| `phenotypes={"SLCO1B1": "PM"}, phenotype_scale_overrides={"SLCO1B1": 0.30}` | 0.30× | < 3.0× (compressed toward EM) |

### 107-holdout impact

Bit-identical (Meta 2.679 pin holds). Production benchmark uses default `phenotype_scale_overrides=None`.

### Open follow-ups

- GenoADME applies their meta-analysis-derived overrides and re-computes 1000G PM/EM AUC ratio against Niemi 2006 men-stratum central 3.32. Out of scope for Sisyphus.
- Multi-node overrides (gut_wall enzyme phenotype scaling) — not requested, separate concern.
```

- [ ] **Commit experiment-log update + push branch**

```bash
git add docs/claude/experiment-log.md
git commit -m "$(cat <<'EOF'
docs(experiment-log): v0.3.3 phenotype_scale_overrides entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

git push -u origin feat/phenotype-scale-overrides
```

- [ ] **Create PR**

```bash
gh pr create --title "feat(phenotype): phenotype_scale_overrides API hook (v0.3.3, #31)" --body "$(cat <<'EOF'
## Summary
- Closes #31 — capability request from GenoADME (per-substrate effective phenotype scale injection)
- Adds `phenotype_scale_overrides: dict[str, float] | None` kwarg to `apply_phenotype_to_graph()` and `predict()`
- Flat `{gene: scale}` signature (simplified from GenoADME's 3-level proposal — substrate dimension implicit in per-call SMILES; phenotype dimension implicit in per-call phenotypes argument)
- No calibration tables shipped — caller (GenoADME) resolves substrate→override mapping in their own meta-analysis layer

## Empirical (pravastatin SLCO1B1:PM)

| call | OATP1B1 abundance scaling | Cmax compression vs EM |
|---|---|---|
| no phenotype | 1.00× | 1.0× |
| PM, no override | 0.10× (CPIC default) | ~3.0× |
| PM, override 0.30 | 0.30× | between EM and default PM |

Override compresses PM/EM Cmax shift toward EM, as required by the GenoADME use case (Niemi 2006 men-stratum target ~3.32 central).

## Test plan
- [x] `pytest tests/unit/test_phenotype_scale_overrides.py -v` — 7 PASS
- [x] `pytest tests/integration/test_phenotype_scale_overrides_pravastatin.py -v` — 2 PASS
- [x] `pytest tests/unit/test_phenotype.py tests/unit/test_pipeline_phenotypes.py -v` — all PASS (backward compat)
- [x] `pytest tests/integration/test_holdout_regression.py -v` — Meta 2.679 invariant
- [x] CI green

## Architecture
- Engine: 0 line changes
- 107-holdout AAFE bit-identical
- v0.3.2 phenotype back-solve fix (PR #32) preserved — override applies BEFORE the snapshot semantics

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**1. Spec coverage**
- §3 architecture (override location, substrate caller-resolved): Tasks 2, 3 implement
- §4.1 apply_phenotype_to_graph signature: Task 2
- §4.2 predict signature: Task 3
- §5.1 phenotype.py loop: Task 2 (override branch + unused-keys logger.info)
- §5.2 pipeline forwarding: Task 3
- §6.1 unit tests: Task 1 (7 cases match §6.1 list)
- §6.2 integration test: Task 4
- §6.3 backward compat: closing operations + Tasks 2/3 verify existing tests stay green
- §10 acceptance criteria: covered by Tasks 1-4 + closing holdout check

No gaps.

**2. Placeholder scan**
All steps include explicit code blocks. No "TBD/TODO" leftovers.

**3. Type consistency**
- `dict[str, float] | None` consistent across `apply_phenotype_to_graph`, `predict`, and tests
- `phenotype_scale_overrides` parameter name identical in both functions and signature
- Tests use the canonical name everywhere

No issues. Ready for execution.
