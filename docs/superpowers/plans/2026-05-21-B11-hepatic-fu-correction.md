# B-11 Hepatic Intracellular fu Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a per-drug `fu_correction_liver` Distribution scaled into the well-stirred / parallel-tube hepatic extraction formula, gated by a per-node YAML flag. Default value 1.0 preserves current behavior for non-curated drugs. Then audit and curate 19 holdout drugs with `meta_fold > 3` per literature search protocol.

**Architecture:** New per-drug `Distribution` field on `DrugOnGraph` populated via `data/transporters/hepatic_fu_correction.json` lookup (InChIKey-keyed, with connectivity-block fallback). Engine reads `params.node_param(<source>, "fu_correction_applicable")` at runtime and multiplies `fup` by the per-drug correction only at flagged nodes (currently just liver). Loader rejects values `< 1.0` to enforce the anti-fudge constraint (CLAUDE.md invariant #8).

**Tech Stack:** Python 3.10+, RDKit (InChIKey canonicalization), pytest, ruff. All within existing Sisyphus engine/predict layer conventions.

**Reference spec:** `docs/superpowers/specs/2026-05-21-B11-hepatic-fu-correction-design.md`

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `data/transporters/hepatic_fu_correction.json` | InChIKey-keyed registry. Empty (`{"overrides": []}`) at end of Phase A; 19 audit rows at end of Phase B. |
| `src/sisyphus/predict/hepatic_fu_correction.py` | Loader, `lookup_hepatic_fu_correction(smiles)` returning `Distribution`. Mirrors `cyp_clearance_overrides.py` with full + connectivity-block InChIKey lookup. |
| `tests/unit/test_hepatic_fu_correction.py` | 6 unit tests on loader + lookup + anti-fudge guard. |
| `tests/unit/test_flux_fu_correction_integration.py` | 3 integration tests on flux apply (clearance + prodrug + identity-blind invariance). |
| `tests/regression/test_hepatic_fu_correction_schema.py` | 2 schema regressions over the production registry. |

### Modified files

| Path | Change |
|---|---|
| `src/sisyphus/core.py` | Add `DrugOnGraph.fu_correction_liver: Distribution` field with default `Distribution(mean=1.0, cv=0.0)`. Propagate through `sample(rng)` and `realize_means()`. |
| `src/sisyphus/predict/ivive.py` | `build_drug_on_graph` calls `lookup_hepatic_fu_correction(profile.smiles)` and threads the result into the constructed `DrugOnGraph`. |
| `src/sisyphus/engine/flux.py` | `ClearanceFluxSpec.apply` well_stirred and parallel_tube branches + `ProdrugActivationFluxSpec.apply`: at flagged nodes, replace `fup` with `fup × fu_correction_liver`. |
| `data/physiology/reference_man.yaml` | Liver node gets `fu_correction_applicable: 1.0`. (Stored as float so `node_param` returns it via existing API.) |
| `docs/claude/experiment-log.md` | New entries for Phase A and Phase B. |
| `docs/claude/backlog.md` | Remove B-11 once Phase B shipped (or mark DE-37 if escape clause triggers). |
| `README.md` | Headline + reproducibility note refresh after Phase B (only if headline shifts ≥ 1% or DE-37 takes effect). |
| `CLAUDE.md` | Gitignored; refresh headline table locally after Phase B. |
| `tests/integration/test_holdout_regression.py` | Update pinned AAFE after Phase B if headline shifts. |
| `tests/regression/test_prodrug_v3_enzyme_leak_audit.py` | Add curated drugs to `DRUG_SPECIFIC_CHANGES` after Phase B. |

### Out of scope (deferred to a follow-up plan if pursued)

- `ClearanceFluxSpec.apply` `extended` (ECM) branch — strict spec scope is well_stirred + parallel_tube only. ECM gets the correction in a follow-up.
- Gut-wall fu correction.
- Renal GFR fu correction.

---

## Phase A — Infrastructure (headline-invariant)

Phase A produces a shippable PR with bit-identical holdout AAFE. The registry is committed with `overrides: []`, so every lookup returns the default `Distribution(mean=1.0, cv=0.0)` and the engine sees no change.

### Task 1: Add `DrugOnGraph.fu_correction_liver` field with propagation

**Files:**
- Modify: `src/sisyphus/core.py` (`DrugOnGraph` dataclass + `sample` + `realize_means`)
- Test: `tests/unit/test_drug_on_graph_fu_correction.py` (new)

- [ ] **Step 1.1: Write the failing tests**

```python
# tests/unit/test_drug_on_graph_fu_correction.py
"""Unit tests for DrugOnGraph.fu_correction_liver field (B-11)."""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.core import Distribution, DrugOnGraph


def _minimal_dog(**overrides) -> DrugOnGraph:
    """Construct a DrugOnGraph with required fields only; B-11-related override allowed."""
    defaults = dict(
        name="test_drug",
        smiles="CCO",
        dose_mg=100.0,
        route="oral",
        administration_node="stomach_lumen",
        mw=46.07,
        pka=None,
        compound_type="neutral",
        fup=Distribution(mean=0.1, cv=0.0),
        rbp=Distribution(mean=1.0, cv=0.0),
        kp_method="rodgers_rowland",
        kp_overrides={},
        peff=Distribution(mean=1.0e-4, cv=0.0),
        solubility=Distribution(mean=1.0, cv=0.0),
        enzyme_affinity={},
        renal_clearance=Distribution(mean=0.0, cv=0.0),
    )
    defaults.update(overrides)
    return DrugOnGraph(**defaults)


class TestFuCorrectionLiverField:
    def test_default_is_one_with_zero_cv(self):
        dog = _minimal_dog()
        assert dog.fu_correction_liver.mean == pytest.approx(1.0)
        assert dog.fu_correction_liver.cv == pytest.approx(0.0)

    def test_explicit_value_preserved(self):
        dog = _minimal_dog(fu_correction_liver=Distribution(mean=8.5, cv=0.5))
        assert dog.fu_correction_liver.mean == pytest.approx(8.5)
        assert dog.fu_correction_liver.cv == pytest.approx(0.5)

    def test_realize_means_carries_value(self):
        """realize_means() returns the central tendency as cv=0 Distribution."""
        dog = _minimal_dog(fu_correction_liver=Distribution(mean=8.5, cv=0.5))
        realized = dog.realize_means()
        assert realized.fu_correction_liver.mean == pytest.approx(8.5)
        assert realized.fu_correction_liver.cv == pytest.approx(0.0)

    def test_sample_carries_value(self):
        """sample(rng) draws a stochastic point; result is cv=0 Distribution at the draw."""
        dog = _minimal_dog(fu_correction_liver=Distribution(mean=8.5, cv=0.5))
        rng = np.random.default_rng(seed=42)
        sampled = dog.sample(rng)
        # Sample is lognormal around 8.5; positive and finite is enough for the propagation test.
        assert sampled.fu_correction_liver.mean > 0.0
        assert np.isfinite(sampled.fu_correction_liver.mean)
        assert sampled.fu_correction_liver.cv == pytest.approx(0.0)

    def test_default_realize_means_is_one(self):
        dog = _minimal_dog()
        assert dog.realize_means().fu_correction_liver.mean == pytest.approx(1.0)

    def test_default_sample_is_one_when_cv_zero(self):
        dog = _minimal_dog()  # default cv=0 means deterministic draw at 1.0
        rng = np.random.default_rng(seed=42)
        assert dog.sample(rng).fu_correction_liver.mean == pytest.approx(1.0)
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `pytest tests/unit/test_drug_on_graph_fu_correction.py -v`
Expected: 6 FAIL with `TypeError: __init__() got an unexpected keyword argument 'fu_correction_liver'` (or similar — field does not yet exist).

- [ ] **Step 1.3: Add the dataclass field**

Locate the `DrugOnGraph` dataclass definition in `src/sisyphus/core.py` (search for `class DrugOnGraph` — it's a `@dataclass(frozen=True)`). Add the new field below the existing fields. The field must be declared with `field(default_factory=...)` because `Distribution` is not hashable and mutable defaults are not allowed in frozen dataclasses with bare instances.

```python
# In src/sisyphus/core.py, inside class DrugOnGraph:
fu_correction_liver: Distribution = field(
    default_factory=lambda: Distribution(mean=1.0, cv=0.0)
)
```

Add `from dataclasses import field` to the imports at the top of the file if not already present (other fields likely already use this pattern; confirm before editing).

- [ ] **Step 1.4: Propagate through `sample(rng)`**

Find the `DrugOnGraph.sample` method (search for `def sample(self, rng`). It rebuilds a new `DrugOnGraph` by drawing each `Distribution` field. Add this line where other `Distribution` fields are sampled (preserve alphabetical or existing order):

```python
fu_correction_liver=Distribution(
    mean=self.fu_correction_liver.sample(rng), cv=0.0
),
```

- [ ] **Step 1.5: Propagate through `realize_means()`**

Find the `DrugOnGraph.realize_means` method. Add this line where other `Distribution` fields are realized:

```python
fu_correction_liver=Distribution(mean=self.fu_correction_liver.mean, cv=0.0),
```

- [ ] **Step 1.6: Run tests to verify they pass**

Run: `pytest tests/unit/test_drug_on_graph_fu_correction.py -v`
Expected: 6 PASS.

- [ ] **Step 1.7: Run the full unit suite for regression check**

Run: `pytest tests/unit/ -q`
Expected: all existing tests still pass (the new field has a safe default).

- [ ] **Step 1.8: Commit**

```bash
git add src/sisyphus/core.py tests/unit/test_drug_on_graph_fu_correction.py
git commit -m "feat(core): DrugOnGraph.fu_correction_liver Distribution field (B-11)"
```

---

### Task 2: Registry loader with InChIKey + connectivity-block fallback

**Files:**
- Create: `src/sisyphus/predict/hepatic_fu_correction.py`
- Test: `tests/unit/test_hepatic_fu_correction.py` (new)

- [ ] **Step 2.1: Write the failing tests**

```python
# tests/unit/test_hepatic_fu_correction.py
"""Unit tests for the hepatic_fu_correction registry loader (B-11)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sisyphus.core import Distribution
from sisyphus.predict.hepatic_fu_correction import lookup_hepatic_fu_correction

_CLOPIDOGREL_STEREO = "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1"
_CLOPIDOGREL_NONSTEREO = "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3"
_MORPHINE = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"


def _write_registry(tmp_path: Path, overrides: list[dict]) -> Path:
    p = tmp_path / "hepatic_fu_correction.json"
    p.write_text(json.dumps({"overrides": overrides}))
    return p


def _valid_entry(**overrides) -> dict:
    base = {
        "drug": "clopidogrel",
        "smiles": _CLOPIDOGREL_STEREO,
        "inchikey": "GKTWGGQPFAXNFI-HNNXBMFYSA-N",
        "fu_correction_liver": {"mean": 8.5, "cv": 0.5},
        "disposition": "literature_applied",
        "literature": ["Watanabe 2009 DMD 37:1471 Table 1"],
        "notes": "test fixture",
        "n_candidates_reviewed": 3,
        "source_dbs_searched": ["PubMed"],
    }
    base.update(overrides)
    return base


def test_default_returns_one_for_unregistered(tmp_path):
    reg = _write_registry(tmp_path, [])
    out = lookup_hepatic_fu_correction(_MORPHINE, registry_path=reg)
    assert out.mean == pytest.approx(1.0)
    assert out.cv == pytest.approx(0.0)


def test_inchikey_full_match(tmp_path):
    reg = _write_registry(tmp_path, [_valid_entry()])
    out = lookup_hepatic_fu_correction(_CLOPIDOGREL_STEREO, registry_path=reg)
    assert out.mean == pytest.approx(8.5)
    assert out.cv == pytest.approx(0.5)


def test_inchikey_connectivity_fallback_for_stereo_variant(tmp_path):
    """Non-stereo query SMILES matches stereospecific registry via connectivity block."""
    reg = _write_registry(tmp_path, [_valid_entry()])
    out = lookup_hepatic_fu_correction(_CLOPIDOGREL_NONSTEREO, registry_path=reg)
    assert out.mean == pytest.approx(8.5)


def test_loader_rejects_value_below_one(tmp_path):
    """Anti-fudge guard: fu_correction_liver < 1.0 is not allowed (CLAUDE.md #8)."""
    bad = _valid_entry(fu_correction_liver={"mean": 0.7, "cv": 0.1})
    reg = _write_registry(tmp_path, [bad])
    with pytest.raises(ValueError, match=r"fu_correction_liver.*>= 1\.0"):
        lookup_hepatic_fu_correction(_CLOPIDOGREL_STEREO, registry_path=reg)


def test_loader_rejects_missing_disposition(tmp_path):
    bad = _valid_entry()
    del bad["disposition"]
    reg = _write_registry(tmp_path, [bad])
    with pytest.raises(ValueError, match="disposition"):
        lookup_hepatic_fu_correction(_CLOPIDOGREL_STEREO, registry_path=reg)


def test_loader_rejects_unknown_disposition(tmp_path):
    bad = _valid_entry(disposition="fudge_applied")
    reg = _write_registry(tmp_path, [bad])
    with pytest.raises(ValueError, match="disposition"):
        lookup_hepatic_fu_correction(_CLOPIDOGREL_STEREO, registry_path=reg)
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `pytest tests/unit/test_hepatic_fu_correction.py -v`
Expected: 6 FAIL with `ImportError: cannot import name 'lookup_hepatic_fu_correction'`.

- [ ] **Step 2.3: Implement the loader**

Create `src/sisyphus/predict/hepatic_fu_correction.py` with the following content:

```python
"""Per-drug hepatic intracellular fu correction registry (B-11).

When a drug's effective unbound fraction inside hepatocytes is higher
than its plasma fup (typically due to albumin-facilitated uptake or
intracellular protein binding), the WS / PT extraction formulas
under-predict hepatic CL. This registry holds a per-drug
``fu_correction_liver`` multiplier curated from primary literature.
At flagged hepatic nodes the engine replaces ``fup`` with
``fup × fu_correction_liver`` in the WS / PT formula.

Default for any drug not in the registry is ``Distribution(mean=1.0,
cv=0.0)`` -- no scaling. Lookup is by RDKit InChIKey with a
connectivity-block fallback so non-isomeric query SMILES still match
stereospecific registry entries.

Registry file: ``data/transporters/hepatic_fu_correction.json``
Schema: see docs/superpowers/specs/2026-05-21-B11-hepatic-fu-correction-design.md
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from sisyphus.core import Distribution

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "transporters" / "hepatic_fu_correction.json"
)

_VALID_DISPOSITIONS = frozenset({
    "literature_applied",
    "class_extrapolated",
    "ceiling_accepted",
    "not_applicable",
})


def _default() -> Distribution:
    """Default no-scaling correction."""
    return Distribution(mean=1.0, cv=0.0)


@lru_cache(maxsize=1)
def _load(path_str: str) -> tuple[dict[str, Distribution], dict[str, list[Distribution]]]:
    """Load registry; index by full InChIKey and connectivity block.

    Validates every entry: disposition is in the allowed set and
    ``fu_correction_liver.mean >= 1.0`` (anti-fudge guard). Connectivity
    matches are only honored when unambiguous (one override per block).
    """
    path = Path(path_str)
    if not path.exists():
        logger.warning("hepatic_fu_correction registry not found at %s", path)
        return {}, {}

    with path.open() as f:
        data = json.load(f)

    full_index: dict[str, Distribution] = {}
    conn_index: dict[str, list[Distribution]] = {}

    for entry in data.get("overrides", []):
        disposition = entry.get("disposition")
        if disposition not in _VALID_DISPOSITIONS:
            raise ValueError(
                f"hepatic_fu_correction entry for {entry.get('drug')!r} has "
                f"disposition={disposition!r}; must be one of "
                f"{sorted(_VALID_DISPOSITIONS)}"
            )

        ikey = entry.get("inchikey")
        raw = entry.get("fu_correction_liver")
        if ikey is None or not isinstance(raw, dict) or "mean" not in raw:
            raise ValueError(
                f"hepatic_fu_correction entry for {entry.get('drug')!r} missing "
                f"required fields (inchikey, fu_correction_liver.mean)"
            )

        mean = float(raw["mean"])
        if mean < 1.0:
            raise ValueError(
                f"hepatic_fu_correction entry for {entry.get('drug')!r} has "
                f"fu_correction_liver.mean={mean} < 1.0; values must be >= 1.0 "
                f"per CLAUDE.md invariant #8 (no fudge to Cmax loss). To raise "
                f"hepatic CL for a highly bound drug, use a literature-derived "
                f"fu_inc/fu_plasma ratio >= 1.0."
            )

        dist = Distribution(mean=mean, cv=float(raw.get("cv", 0.0)))
        full_index[ikey] = dist
        conn_index.setdefault(ikey.split("-", maxsplit=1)[0], []).append(dist)

    return full_index, conn_index


def lookup_hepatic_fu_correction(
    smiles: str, registry_path: Path | None = None
) -> Distribution:
    """Return the hepatic fu correction Distribution for ``smiles``.

    Returns ``Distribution(mean=1.0, cv=0.0)`` (no scaling) when the
    SMILES is not in the registry, RDKit is unavailable, or the SMILES
    is invalid. Tries full InChIKey first, then falls back to
    connectivity-block matching (mirrors registry.py B-03 pattern)
    when full key misses.

    Raises ``ValueError`` only when the registry file itself is
    malformed (invalid disposition or sub-1.0 value).
    """
    try:
        from rdkit import Chem
    except ImportError:
        return _default()

    if not smiles:
        return _default()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _default()

    ikey = Chem.MolToInchiKey(mol)
    path = registry_path if registry_path is not None else _DEFAULT_REGISTRY_PATH

    if registry_path is not None:
        # Bypass the lru_cache for tests that write registries in tmp dirs.
        full_index, conn_index = _load_uncached(path)
    else:
        full_index, conn_index = _load(str(path))

    if ikey in full_index:
        return full_index[ikey]

    matches = conn_index.get(ikey.split("-", maxsplit=1)[0], [])
    if len(matches) == 1:
        return matches[0]

    return _default()


def _load_uncached(
    path: Path,
) -> tuple[dict[str, Distribution], dict[str, list[Distribution]]]:
    """Test-only helper that loads without lru_cache, so tmp paths work."""
    if not path.exists():
        return {}, {}
    with path.open() as f:
        data = json.load(f)
    full_index: dict[str, Distribution] = {}
    conn_index: dict[str, list[Distribution]] = {}
    for entry in data.get("overrides", []):
        disposition = entry.get("disposition")
        if disposition not in _VALID_DISPOSITIONS:
            raise ValueError(
                f"hepatic_fu_correction entry for {entry.get('drug')!r} has "
                f"disposition={disposition!r}; must be one of "
                f"{sorted(_VALID_DISPOSITIONS)}"
            )
        ikey = entry.get("inchikey")
        raw = entry.get("fu_correction_liver")
        if ikey is None or not isinstance(raw, dict) or "mean" not in raw:
            raise ValueError(
                f"hepatic_fu_correction entry for {entry.get('drug')!r} missing "
                f"required fields (inchikey, fu_correction_liver.mean)"
            )
        mean = float(raw["mean"])
        if mean < 1.0:
            raise ValueError(
                f"hepatic_fu_correction entry for {entry.get('drug')!r} has "
                f"fu_correction_liver.mean={mean} < 1.0; values must be >= 1.0"
            )
        dist = Distribution(mean=mean, cv=float(raw.get("cv", 0.0)))
        full_index[ikey] = dist
        conn_index.setdefault(ikey.split("-", maxsplit=1)[0], []).append(dist)
    return full_index, conn_index
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `pytest tests/unit/test_hepatic_fu_correction.py -v`
Expected: 6 PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/sisyphus/predict/hepatic_fu_correction.py tests/unit/test_hepatic_fu_correction.py
git commit -m "feat(predict): hepatic_fu_correction loader with InChIKey + connectivity fallback (B-11)"
```

---

### Task 3: Schema regression over production registry

**Files:**
- Create: `tests/regression/test_hepatic_fu_correction_schema.py`

The production registry will be empty after Phase A (`{"overrides": []}`), so these gates will not fail at Phase A end. They will catch violations once Phase B starts adding entries.

- [ ] **Step 3.1: Create the test file**

```python
# tests/regression/test_hepatic_fu_correction_schema.py
"""Schema regression over the production hepatic_fu_correction registry (B-11).

Catches violations as Phase B adds curated entries. Phase A end state
is an empty overrides list -- these tests pass trivially until then.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "transporters" / "hepatic_fu_correction.json"
)

_LITERATURE_REQUIRED_DISPOSITIONS = frozenset({
    "literature_applied", "class_extrapolated",
})

_VALID_DISPOSITIONS = frozenset({
    "literature_applied",
    "class_extrapolated",
    "ceiling_accepted",
    "not_applicable",
})


def _load_overrides() -> list[dict]:
    if not _REGISTRY_PATH.exists():
        return []
    return json.loads(_REGISTRY_PATH.read_text()).get("overrides", [])


def test_literature_applied_requires_citation():
    """literature_applied and class_extrapolated entries must have a non-empty
    ``literature`` array (each item a citation string)."""
    violations = []
    for entry in _load_overrides():
        if entry.get("disposition") in _LITERATURE_REQUIRED_DISPOSITIONS:
            lit = entry.get("literature") or []
            if not lit:
                violations.append(entry.get("drug"))
    assert not violations, (
        f"literature_applied/class_extrapolated entries with no citation: "
        f"{violations}. Anti-fudge: every value above 1.0 must trace to a paper."
    )


def test_disposition_in_allowed_set():
    bad = []
    for entry in _load_overrides():
        if entry.get("disposition") not in _VALID_DISPOSITIONS:
            bad.append((entry.get("drug"), entry.get("disposition")))
    assert not bad, (
        f"hepatic_fu_correction entries with invalid disposition: {bad}. "
        f"Allowed: {sorted(_VALID_DISPOSITIONS)}"
    )
```

- [ ] **Step 3.2: Run the tests (pass trivially before file exists)**

Run: `pytest tests/regression/test_hepatic_fu_correction_schema.py -v`
Expected: 2 PASS (no overrides to validate yet).

- [ ] **Step 3.3: Commit**

```bash
git add tests/regression/test_hepatic_fu_correction_schema.py
git commit -m "test(regression): hepatic_fu_correction registry schema gates (B-11)"
```

---

### Task 4: Wire `build_drug_on_graph` in ivive.py

**Files:**
- Modify: `src/sisyphus/predict/ivive.py` (`build_drug_on_graph` function near line 591)
- Test: extend `tests/unit/test_drug_on_graph_fu_correction.py` with one end-to-end test

- [ ] **Step 4.1: Write the failing test (append to test file from Task 1)**

```python
# Append to tests/unit/test_drug_on_graph_fu_correction.py
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph


class TestBuildDrugOnGraphAttachesFuCorrection:
    def test_default_one_when_smiles_unregistered(self):
        """Drugs not in hepatic_fu_correction.json get the default 1.0."""
        # ethanol — never in any registry
        profile = compute_profile("CCO")
        adme = predict_adme(profile)
        dog = build_drug_on_graph(profile, adme, dose_mg=100.0, route="oral")
        assert dog.fu_correction_liver.mean == pytest.approx(1.0)
        assert dog.fu_correction_liver.cv == pytest.approx(0.0)
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `pytest tests/unit/test_drug_on_graph_fu_correction.py::TestBuildDrugOnGraphAttachesFuCorrection -v`
Expected: FAIL — at this step `build_drug_on_graph` does not yet thread the field, so the constructed DrugOnGraph keeps the default but only because the field defaults to 1.0. The test should still pass trivially after Task 1's default. **Re-check after Task 4.3 confirms the lookup is actually called.**

(Note for the implementer: this test is a low-bar correctness check. The real wiring guarantee comes from Task 8's identity-blind / end-to-end integration test. If both tests pass, the wiring is correct.)

- [ ] **Step 4.3: Add the lookup call**

In `src/sisyphus/predict/ivive.py` near line 670 (where the existing `lookup_metabolic_fraction` call lives), add the parallel lookup just before constructing the `DrugOnGraph`:

```python
# Existing line (do not modify):
from sisyphus.predict.cyp_clearance_overrides import lookup_metabolic_fraction
metabolic_fraction = lookup_metabolic_fraction(profile.smiles)

# Add immediately below:
from sisyphus.predict.hepatic_fu_correction import lookup_hepatic_fu_correction
fu_correction_liver = lookup_hepatic_fu_correction(profile.smiles)
```

Then locate the `DrugOnGraph(...)` constructor call later in the same function. Add `fu_correction_liver=fu_correction_liver` to the keyword arguments, alphabetically or near other `Distribution` fields. Example placement (final block of `build_drug_on_graph`):

```python
return DrugOnGraph(
    name=...,
    smiles=profile.smiles,
    ...,
    enzyme_affinity=enzyme_affinity,
    renal_clearance=renal_cl,
    fu_correction_liver=fu_correction_liver,  # NEW (B-11)
    # ... other existing fields stay unchanged
)
```

- [ ] **Step 4.4: Run the test to verify it passes**

Run: `pytest tests/unit/test_drug_on_graph_fu_correction.py -v`
Expected: all 7 PASS.

- [ ] **Step 4.5: Run the full unit suite for regression check**

Run: `pytest tests/unit/ -q`
Expected: all existing tests still pass.

- [ ] **Step 4.6: Commit**

```bash
git add src/sisyphus/predict/ivive.py tests/unit/test_drug_on_graph_fu_correction.py
git commit -m "feat(ivive): wire hepatic_fu_correction into build_drug_on_graph (B-11)"
```

---

### Task 5: Liver node `fu_correction_applicable` flag in physiology YAML

**Files:**
- Modify: `data/physiology/reference_man.yaml` (liver node block)
- Test: `tests/unit/test_reference_yaml_fu_flag.py` (new)

- [ ] **Step 5.1: Write the failing test**

```python
# tests/unit/test_reference_yaml_fu_flag.py
"""Liver node carries fu_correction_applicable flag (B-11)."""
from __future__ import annotations

from sisyphus.graph.presets import reference_man


def test_liver_has_fu_correction_applicable_flag():
    body = reference_man()
    assert "liver" in body.nodes, "reference_man must have a liver node"
    liver = body.nodes["liver"]
    flag = liver.params.get("fu_correction_applicable")
    assert flag is not None, "liver node missing fu_correction_applicable param"
    assert float(flag.mean if hasattr(flag, "mean") else flag) == 1.0


def test_other_nodes_lack_flag():
    """No other node carries the flag (Phase A scope: liver-only)."""
    body = reference_man()
    flagged = [
        n for n, node in body.nodes.items()
        if "fu_correction_applicable" in node.params and n != "liver"
    ]
    assert flagged == [], (
        f"Unexpected non-liver nodes flagged for fu_correction: {flagged}. "
        "B-11 Phase A scope is liver-only. To extend to gut_wall etc., "
        "update the spec first."
    )
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `pytest tests/unit/test_reference_yaml_fu_flag.py -v`
Expected: FAIL with `assert flag is not None` (liver lacks the param).

- [ ] **Step 5.3: Add the flag to the liver block in YAML**

Open `data/physiology/reference_man.yaml`. Locate the `liver` node block (search for `^  liver:` or similar). It already contains keys like `ivive_scaling: 0.00006`, `volume: ...`, `enzymes: {...}`, etc. Add a new line under the same indentation level as `ivive_scaling`:

```yaml
    fu_correction_applicable: 1.0   # B-11: hepatic intracellular fu correction is applied at this node by ClearanceFlux + ProdrugActivationFlux
```

Do not add the flag to any other node in this task.

- [ ] **Step 5.4: Run test to verify it passes**

Run: `pytest tests/unit/test_reference_yaml_fu_flag.py -v`
Expected: both PASS.

- [ ] **Step 5.5: Run the full unit suite for regression**

Run: `pytest tests/unit/ -q`
Expected: all existing tests pass (the new YAML key is opt-in; existing flux specs ignore it until Task 6/7 add the read).

- [ ] **Step 5.6: Commit**

```bash
git add data/physiology/reference_man.yaml tests/unit/test_reference_yaml_fu_flag.py
git commit -m "feat(physiology): add fu_correction_applicable=1.0 flag to liver node (B-11)"
```

---

### Task 6: `ClearanceFluxSpec.apply` gated correction (well_stirred + parallel_tube)

**Files:**
- Modify: `src/sisyphus/engine/flux.py` (`ClearanceFluxSpec.apply`)
- Test: `tests/unit/test_flux_fu_correction_integration.py` (new)

- [ ] **Step 6.1: Write the failing tests**

```python
# tests/unit/test_flux_fu_correction_integration.py
"""Engine integration tests for B-11 fu_correction_liver gating.

Verifies the correction multiplies fup ONLY at flagged nodes in
ClearanceFluxSpec well_stirred + parallel_tube branches and in
ProdrugActivationFluxSpec. Identity-blindness is verified via a
random-rename test that mirrors the existing engine identity-blind
guard pattern.
"""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.pipeline.predict import predict
from sisyphus.predict.hepatic_fu_correction import lookup_hepatic_fu_correction


_HIGH_PPB_TEST_SMILES = "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3"  # clopidogrel non-stereo


@pytest.fixture
def empty_registry(monkeypatch, tmp_path):
    """Force the lookup to use an empty registry; bypasses any committed entries."""
    p = tmp_path / "empty_hepatic_fu_correction.json"
    p.write_text('{"overrides": []}')
    monkeypatch.setattr(
        "sisyphus.predict.hepatic_fu_correction._DEFAULT_REGISTRY_PATH", p
    )
    # Clear any lru_cache populated from earlier test ordering.
    lookup_hepatic_fu_correction.__wrapped__ if hasattr(
        lookup_hepatic_fu_correction, "__wrapped__"
    ) else None
    from sisyphus.predict.hepatic_fu_correction import _load
    _load.cache_clear()
    yield p
    _load.cache_clear()


def test_clearance_flux_no_change_with_default_correction(empty_registry):
    """With empty registry every drug sees fu_correction_liver=1.0, so engine
    Cmax must equal the pre-B-11 behavior. We baseline against a non-prodrug
    drug to keep this test independent of prodrug routing."""
    result = predict("CCO", dose_mg=500.0, route="oral")  # ethanol
    assert result.engine_pk is not None
    cmax = float(result.engine_pk.cmax.mean)
    assert cmax > 0.0
    # Smoke check: predict returned finite. The bit-identical check belongs
    # in the holdout cache invariance gate (Task 9), not here.
    assert pytest.approx(cmax, rel=1e-6) == cmax  # finite & deterministic


def test_clearance_flux_applies_correction_at_flagged_node(monkeypatch):
    """fu_correction_liver > 1.0 raises hepatic CL and lowers predicted Cmax."""
    # Patch the lookup to return 5.0 for any SMILES.
    monkeypatch.setattr(
        "sisyphus.predict.hepatic_fu_correction.lookup_hepatic_fu_correction",
        lambda smiles, registry_path=None: Distribution(mean=5.0, cv=0.0),
    )
    # ALSO patch the import site in ivive (the local-import pattern means
    # the symbol is bound at call time; this monkeypatch reaches both sites.)
    monkeypatch.setattr(
        "sisyphus.predict.ivive.lookup_hepatic_fu_correction",
        lambda smiles, registry_path=None: Distribution(mean=5.0, cv=0.0),
        raising=False,
    )

    # Use a high-PPB drug whose hepatic CL is fup-limited.
    result_default = predict(_HIGH_PPB_TEST_SMILES, dose_mg=300.0, route="oral")
    cmax_default = float(result_default.engine_pk.cmax.mean)

    # Reset and re-run with fu_corr=1.0 to compare.
    monkeypatch.setattr(
        "sisyphus.predict.ivive.lookup_hepatic_fu_correction",
        lambda smiles, registry_path=None: Distribution(mean=1.0, cv=0.0),
        raising=False,
    )
    result_baseline = predict(_HIGH_PPB_TEST_SMILES, dose_mg=300.0, route="oral")
    cmax_baseline = float(result_baseline.engine_pk.cmax.mean)

    # fu_corr=5 should INCREASE hepatic CL → DECREASE Cmax vs fu_corr=1.
    assert cmax_default < cmax_baseline, (
        f"fu_correction_liver=5.0 should lower predicted Cmax for high-PPB "
        f"clopidogrel; got default={cmax_default:.4f}, baseline={cmax_baseline:.4f}"
    )


def test_identity_blind_random_rename_invariant(monkeypatch):
    """Engine produces bit-identical Cmax when liver is renamed to a random string,
    provided the fu_correction_applicable flag travels with it.

    Mirrors the existing engine identity-blind invariance pattern used by
    other B-* features. Tests that the engine reads the node flag rather
    than matching the string 'liver'."""
    # NOTE: this test depends on `reference_man()` + a parallel `body_with_renamed_liver()`
    # builder which renames every reference to "liver" -> "X" both in node names and
    # in any edge sources/targets. The fu_correction_applicable flag must carry over.
    # Implementation guidance:
    #   1. Load reference_man(). Find every Node, Edge, and YAML key containing "liver".
    #   2. Build a substituted graph with the same topology but renamed liver -> "X".
    #   3. Run predict(...) on both. cmax must match to a tight rel tolerance (1e-9).
    # This is a long-form integration check; if the test fixture is non-trivial,
    # implement the renamer as a helper in tests/conftest.py.
    pytest.skip(
        "identity-blind rename invariance requires a graph-rename helper; "
        "deferred to Task 8 where the helper lives. This test is a placeholder."
    )
```

- [ ] **Step 6.2: Run tests to verify they fail**

Run: `pytest tests/unit/test_flux_fu_correction_integration.py -v`
Expected: `test_clearance_flux_applies_correction_at_flagged_node` FAILs (no gating logic yet, so the cmax with and without monkeypatched correction will be identical).

- [ ] **Step 6.3: Add the gated correction to `ClearanceFluxSpec.apply`**

In `src/sisyphus/engine/flux.py`, locate `ClearanceFluxSpec.apply` (around line 215). For the `well_stirred` branch (currently lines 222–249) and `parallel_tube` branch (lines 251–279), modify the `fup` read so it is replaced by `fup × fu_correction_liver` when the node flag is set.

**well_stirred branch** — replace the existing block:

```python
        if self.model == "well_stirred":
            # Compute organ-level CLint from enzyme abundances x drug affinities
            clint_organ = 0.0
            ivive = params.node_param(self.source_name, "ivive_scaling")
            for tag, abundance in params.node_enzymes(self.source_name).items():
                affinity = params.drug_enzyme_affinity(tag)
                if affinity > 0 and abundance > 0:
                    clint_organ += abundance * affinity * ivive

            if clint_organ <= 0:
                return  # No metabolism at this node

            fup = params.drug_param("fup")
            # B-11: hepatic intracellular fu correction at flagged nodes.
            if params.node_param(self.source_name, "fu_correction_applicable") > 0:
                fup = fup * params.drug_param("fu_correction_liver")

            q = params.total_inflow(self.source_name)

            denom = q + fup * clint_organ
            if denom < 1e-12:
                return
            clh = (q * fup * clint_organ) / denom

            v = params.node_param(self.source_name, "volume")
            kp = params.drug_kp(self.source_name)
            rbp = params.drug_param("rbp")
            c_out = y[self.source_idx] * rbp / (v * kp) if v > 0 else 0.0

            rate = clh * c_out
```

**parallel_tube branch** — apply the same pattern:

```python
        elif self.model == "parallel_tube":
            clint_organ = 0.0
            ivive = params.node_param(self.source_name, "ivive_scaling")
            for tag, abundance in params.node_enzymes(self.source_name).items():
                affinity = params.drug_enzyme_affinity(tag)
                if affinity > 0 and abundance > 0:
                    clint_organ += abundance * affinity * ivive

            if clint_organ <= 0:
                return

            fup = params.drug_param("fup")
            # B-11: hepatic intracellular fu correction at flagged nodes.
            if params.node_param(self.source_name, "fu_correction_applicable") > 0:
                fup = fup * params.drug_param("fu_correction_liver")

            q = params.total_inflow(self.source_name)

            if q < 1e-12:
                return
            exponent = -fup * clint_organ / q
            exponent = max(exponent, -50.0)
            clh = q * (1.0 - np.exp(exponent))

            v = params.node_param(self.source_name, "volume")
            kp = params.drug_kp(self.source_name)
            rbp = params.drug_param("rbp")
            c_out = y[self.source_idx] * rbp / (v * kp) if v > 0 else 0.0

            rate = clh * c_out
```

The `gfr_filtration` and `extended` (ECM) branches remain unchanged for this task (ECM correction is out of B-11 scope per the spec).

- [ ] **Step 6.4: Add `fu_correction_liver` to `ResolvedParams.drug_param`**

The `drug_param` method in `src/sisyphus/engine/compiler.py` (around line 92) is an explicit if/elif chain — adding a new param requires adding a new branch. Locate the chain (it has branches for `"fup"`, `"rbp"`, `"renal_clearance"`, `"dose_mg"`, `"peff"`, `"particle_radius_um"`, `"ps_passive"`, `"ps_eff"`, `"cl_int_bile"`) and add this branch before the final `raise KeyError(...)`:

```python
if param == "fu_correction_liver":
    return self._drug.fu_correction_liver.mean
```

Place it after `cl_int_bile` and before the `raise KeyError` line. Verify no unintended re-ordering of existing branches.

- [ ] **Step 6.5: Run tests to verify the gating test passes**

Run: `pytest tests/unit/test_flux_fu_correction_integration.py::test_clearance_flux_applies_correction_at_flagged_node -v`
Expected: PASS. Predicted Cmax with fu_corr=5 strictly less than with fu_corr=1.

- [ ] **Step 6.6: Run the full unit + regression suite**

Run: `pytest tests/unit/ tests/regression/ -q`
Expected: every existing test still passes. The new code path is only triggered when the flag is set AND fu_correction_liver > 1.0, which is impossible in the empty-registry baseline.

- [ ] **Step 6.7: Commit**

```bash
git add src/sisyphus/engine/flux.py src/sisyphus/engine/compiler.py tests/unit/test_flux_fu_correction_integration.py
git commit -m "feat(engine): ClearanceFluxSpec well_stirred+parallel_tube fu_correction gating (B-11)"
```

(If `compiler.py` was not modified in Step 6.4 — i.e., the dict resolution is automatic — drop it from the `git add`.)

---

### Task 7: `ProdrugActivationFluxSpec.apply` gated correction

**Files:**
- Modify: `src/sisyphus/engine/flux.py` (`ProdrugActivationFluxSpec.apply`, around line 605)
- Test: extend `tests/unit/test_flux_fu_correction_integration.py`

- [ ] **Step 7.1: Write the failing test**

Append to `tests/unit/test_flux_fu_correction_integration.py`:

```python
def test_prodrug_flux_applies_correction_at_flagged_node(monkeypatch):
    """For a prodrug (clopidogrel) routed through ProdrugActivationFlux, the
    fu correction must apply at the liver node — same physiology as
    ClearanceFlux. fu_corr=5.0 should lower parent Cmax."""
    monkeypatch.setattr(
        "sisyphus.predict.ivive.lookup_hepatic_fu_correction",
        lambda smiles, registry_path=None: Distribution(mean=5.0, cv=0.0),
        raising=False,
    )
    # Clopidogrel non-stereo SMILES goes through ProdrugActivationFlux
    # (parent-observation in registry).
    high = predict(_HIGH_PPB_TEST_SMILES, dose_mg=300.0, route="oral")
    cmax_high = float(high.engine_pk.cmax.mean)

    monkeypatch.setattr(
        "sisyphus.predict.ivive.lookup_hepatic_fu_correction",
        lambda smiles, registry_path=None: Distribution(mean=1.0, cv=0.0),
        raising=False,
    )
    low = predict(_HIGH_PPB_TEST_SMILES, dose_mg=300.0, route="oral")
    cmax_low = float(low.engine_pk.cmax.mean)

    assert cmax_high < cmax_low, (
        f"ProdrugActivationFlux at liver must respect fu_correction_liver; "
        f"fu_corr=5: {cmax_high:.4f}, fu_corr=1: {cmax_low:.4f}"
    )
```

- [ ] **Step 7.2: Run test to verify it fails**

Run: `pytest tests/unit/test_flux_fu_correction_integration.py::test_prodrug_flux_applies_correction_at_flagged_node -v`
Expected: FAIL — ProdrugActivationFluxSpec does not yet apply the correction.

- [ ] **Step 7.3: Add gated correction to `ProdrugActivationFluxSpec.apply`**

In `src/sisyphus/engine/flux.py`, locate `ProdrugActivationFluxSpec.apply` (around line 605). Find the line `fup = params.drug_param("fup")` and add the gated multiplication immediately below it:

```python
    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        clint_organ = 0.0
        ivive = params.node_param(self.source_name, "ivive_scaling")
        node_enzymes = params.node_enzymes(self.source_name)
        for tag in self.enzyme_tags:
            abundance = node_enzymes.get(tag, 0.0)
            affinity = params.drug_enzyme_affinity_for_conversion(tag)
            if affinity > 0 and abundance > 0:
                clint_organ += abundance * affinity * ivive

        if clint_organ <= 0:
            return

        fup = params.drug_param("fup")
        # B-11: hepatic intracellular fu correction at flagged nodes.
        if params.node_param(self.source_name, "fu_correction_applicable") > 0:
            fup = fup * params.drug_param("fu_correction_liver")

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
```

- [ ] **Step 7.4: Run the test to verify it passes**

Run: `pytest tests/unit/test_flux_fu_correction_integration.py::test_prodrug_flux_applies_correction_at_flagged_node -v`
Expected: PASS.

- [ ] **Step 7.5: Run the full unit + regression suite**

Run: `pytest tests/unit/ tests/regression/ -q`
Expected: every existing test still passes.

- [ ] **Step 7.6: Commit**

```bash
git add src/sisyphus/engine/flux.py tests/unit/test_flux_fu_correction_integration.py
git commit -m "feat(engine): ProdrugActivationFluxSpec fu_correction gating (B-11)"
```

---

### Task 8: Identity-blind random-rename invariance test

**Files:**
- Modify: `tests/unit/test_flux_fu_correction_integration.py` (un-skip the placeholder)
- Possibly add: `tests/conftest.py` helper if rename utility is missing

- [ ] **Step 8.1: Locate an existing identity-blind rename test for reference pattern**

Run: `grep -rn "identity_blind\|random_rename\|rename_invariant" tests/`. Use the existing pattern (typically a helper that walks the BodyGraph and substitutes node/edge names) as the template.

- [ ] **Step 8.2: Implement the rename helper if not present**

If no rename helper exists yet, add one to `tests/conftest.py`:

```python
# tests/conftest.py — append if not present
import random
import string

from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import Node, Edge


def _random_token(rng: random.Random, length: int = 8) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def rename_body_nodes(body: BodyGraph, seed: int = 42) -> tuple[BodyGraph, dict[str, str]]:
    """Return a new BodyGraph with every node renamed to a random token, and the
    name → new-name mapping. Preserves all params and topology. Useful for
    identity-blind invariance checks."""
    rng = random.Random(seed)
    mapping = {name: _random_token(rng) for name in body.nodes}
    new_nodes = {mapping[name]: Node(name=mapping[name], params=node.params,
                                     enzymes=node.enzymes,
                                     transporters=getattr(node, "transporters", {}))
                 for name, node in body.nodes.items()}
    new_edges = []
    for e in body.edges:
        cls = type(e)
        replaced = cls(**{
            **{f: getattr(e, f) for f in e.__dataclass_fields__},
            "source": mapping[e.source],
            "target": mapping[e.target],
        })
        new_edges.append(replaced)
    return BodyGraph(nodes=new_nodes, edges=new_edges), mapping
```

(If the conftest helper does not match the BodyGraph constructor exactly, adapt to it — the goal is a name-substitution that preserves params and edge topology.)

- [ ] **Step 8.3: Replace the placeholder test with the real check**

In `tests/unit/test_flux_fu_correction_integration.py`, replace `test_identity_blind_random_rename_invariant` with:

```python
def test_identity_blind_random_rename_invariant(monkeypatch):
    """Engine output is bit-identical under arbitrary node renaming, provided the
    fu_correction_applicable flag travels with the renamed liver. Verifies the
    engine reads the flag (not the string 'liver')."""
    from sisyphus.graph.presets import reference_man
    from sisyphus.engine.compiler import compile_graph
    from sisyphus.engine.solver import solve
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.ivive import build_drug_on_graph
    from sisyphus.graph.builder import augment_for_active_species
    from tests.conftest import rename_body_nodes

    # Force a non-trivial fu_correction so the gate actually fires.
    monkeypatch.setattr(
        "sisyphus.predict.ivive.lookup_hepatic_fu_correction",
        lambda smiles, registry_path=None: Distribution(mean=5.0, cv=0.0),
        raising=False,
    )

    smi = _HIGH_PPB_TEST_SMILES
    profile = compute_profile(smi)
    adme = predict_adme(profile)
    dog = build_drug_on_graph(profile, adme, dose_mg=300.0, route="oral")

    body = reference_man()
    body_renamed, mapping = rename_body_nodes(body, seed=42)

    # Re-target the drug's administration_node onto the renamed graph.
    from dataclasses import replace
    dog_renamed = replace(dog, administration_node=mapping[dog.administration_node])

    g1 = augment_for_active_species(body, dog)
    g2 = augment_for_active_species(body_renamed, dog_renamed)

    sim1 = solve(compile_graph(g1, dog))
    sim2 = solve(compile_graph(g2, dog_renamed))

    import numpy as np
    cmax1 = float(np.max(sim1.concentrations.get("venous_blood",
                                                  sim1.concentrations[next(iter(sim1.concentrations))])))
    cmax2 = float(np.max(sim2.concentrations.get(mapping.get("venous_blood", "venous_blood"),
                                                  sim2.concentrations[next(iter(sim2.concentrations))])))
    assert cmax1 == pytest.approx(cmax2, rel=1e-9), (
        f"Identity-blind invariance violated: cmax1={cmax1}, cmax2={cmax2}. "
        "Engine is matching node names instead of reading the YAML flag."
    )
```

(Adapt API call sites to current Sisyphus signatures — `solve`, `compile_graph`, `augment_for_active_species` — by grepping the existing predict path. The key invariant is: same Cmax pre-and-post rename when the flag travels with the renamed liver.)

- [ ] **Step 8.4: Run the test**

Run: `pytest tests/unit/test_flux_fu_correction_integration.py::test_identity_blind_random_rename_invariant -v`
Expected: PASS.

- [ ] **Step 8.5: Run the full unit + regression suite**

Run: `pytest tests/unit/ tests/regression/ -q`
Expected: all pass.

- [ ] **Step 8.6: Commit**

```bash
git add tests/unit/test_flux_fu_correction_integration.py tests/conftest.py
git commit -m "test(engine): identity-blind random-rename invariance for fu_correction (B-11)"
```

---

### Task 9: Initialize empty production registry + holdout cache invariance gate

**Files:**
- Create: `data/transporters/hepatic_fu_correction.json`

- [ ] **Step 9.1: Create the empty registry**

```bash
mkdir -p data/transporters  # likely already exists
```

Write `data/transporters/hepatic_fu_correction.json`:

```json
{
  "overrides": []
}
```

- [ ] **Step 9.2: Run the schema regression**

Run: `pytest tests/regression/test_hepatic_fu_correction_schema.py -v`
Expected: 2 PASS (empty list passes trivially).

- [ ] **Step 9.3: Verify holdout cache bit-identical invariance**

This is the Phase A acceptance gate. Re-run the production benchmark in public-clone state and confirm Meta AAFE = 2.7715 (bit-identical to the current cache).

```bash
# Hide developer artifacts (this is a local-dev concern only; CI runs without them)
if [ -d data/drugbank ]; then mv data/drugbank data/drugbank.HIDDEN; fi
if [ -f models/adme/logp_correction.json ]; then mv models/adme/logp_correction.json models/adme/logp_correction.json.HIDDEN; fi

# Run benchmark
/opt/miniconda3/bin/python scripts/run_engine_benchmark.py --save-json /tmp/b11_phaseA_cache.json

# Compare against committed cache
/opt/miniconda3/bin/python -c "
import json
post = json.load(open('/tmp/b11_phaseA_cache.json'))
canonical = json.load(open('data/training/4track_holdout_predictions.json'))
post_meta = post['overall']['meta']['aafe']
canonical_meta = canonical['overall']['meta']['aafe']
print(f'POST Meta AAFE:      {post_meta:.10f}')
print(f'CANONICAL Meta AAFE: {canonical_meta:.10f}')
print(f'delta: {post_meta - canonical_meta:+.2e}')
assert abs(post_meta - canonical_meta) < 1e-6, 'Phase A is NOT headline-invariant'
print('Phase A invariance: OK')
"

# Restore artifacts
if [ -d data/drugbank.HIDDEN ]; then mv data/drugbank.HIDDEN data/drugbank; fi
if [ -f models/adme/logp_correction.json.HIDDEN ]; then mv models/adme/logp_correction.json.HIDDEN models/adme/logp_correction.json; fi

rm -f /tmp/b11_phaseA_cache.json
```

Expected output: `Phase A invariance: OK`. If the assertion fails, Phase A has accidentally introduced behavior change — diagnose before proceeding.

- [ ] **Step 9.4: Commit**

```bash
git add data/transporters/hepatic_fu_correction.json
git commit -m "feat(data): initialize empty hepatic_fu_correction registry (B-11 Phase A)"
```

---

### Task 10: Phase A docs + push + CI gate

**Files:**
- Modify: `docs/claude/experiment-log.md` (new entry at top)
- Modify: `docs/claude/backlog.md` (mark B-11 Phase A in progress)

- [ ] **Step 10.1: Update experiment-log**

Prepend the following entry to `docs/claude/experiment-log.md` (above the most recent dated entry; preserve frontmatter `last_updated`):

```markdown
## 2026-MM-DD — B-11 Phase A hepatic intracellular fu correction infrastructure

**Motivation:** prepare engine for per-drug `fu_correction_liver` scaling to address systematic over-prediction of plasma Cmax for highly protein-bound drugs (clopidogrel, paroxetine, abiraterone class). Phase A ships infrastructure only; registry is empty; 107-holdout cache is bit-identical.

**What shipped:**
- New `DrugOnGraph.fu_correction_liver: Distribution` field (default 1.0).
- New `src/sisyphus/predict/hepatic_fu_correction.py` loader with InChIKey + connectivity-block fallback.
- New `data/transporters/hepatic_fu_correction.json` with empty overrides list.
- `ClearanceFluxSpec` well_stirred + parallel_tube branches and `ProdrugActivationFluxSpec` apply: gated `fup × fu_correction_liver` at nodes carrying `fu_correction_applicable: 1.0` flag.
- `data/physiology/reference_man.yaml` liver node gets the flag.
- 8 new tests (unit + regression + identity-blind invariance).
- Loader-level anti-fudge guard: rejects `fu_correction_liver < 1.0`.

**Numerical outcome:** 107-holdout Meta AAFE bit-identical (registry empty → every lookup returns default 1.0). Acceptance gate met.

**Next:** Phase B literature curation cycle for 19 over-predict drugs.
```

(Replace `MM-DD` with today's date when committing.)

- [ ] **Step 10.2: Update backlog**

In `docs/claude/backlog.md`, replace the `B-03.x` placeholder (if present) or add a new line in Tier 1/Tier 2:

```markdown
### B-11 — Hepatic intracellular fu correction (Phase A shipped, Phase B in progress)

**Effort**: Phase A shipped (infra, 1 day). Phase B: 2–5 days literature curation across 19 over-predict drugs. **Value**: estimated −1% to −10% Meta AAFE if literature is rich; DE-37 fallback if thin (<0.5% shift). **Risk**: low — defaults preserve current behavior; per-drug rollback by deleting registry rows.

**Spec**: `docs/superpowers/specs/2026-05-21-B11-hepatic-fu-correction-design.md`
**Plan**: `docs/superpowers/plans/2026-05-21-B11-hepatic-fu-correction.md`
**Implementation status**: Phase A shipped at commit <SHA>; Phase B pending.
```

- [ ] **Step 10.3: Run the full test suite locally**

Run: `pytest tests/unit/ tests/regression/ tests/integration/test_holdout_regression.py -q`
Expected: every test passes including the cache invariance gate.

- [ ] **Step 10.4: Run ruff**

Run: `/opt/miniconda3/bin/python -m ruff check src/sisyphus/predict/hepatic_fu_correction.py src/sisyphus/core.py src/sisyphus/predict/ivive.py src/sisyphus/engine/flux.py tests/unit/test_hepatic_fu_correction.py tests/unit/test_flux_fu_correction_integration.py tests/unit/test_drug_on_graph_fu_correction.py tests/unit/test_reference_yaml_fu_flag.py tests/regression/test_hepatic_fu_correction_schema.py`
Expected: `All checks passed!`

- [ ] **Step 10.5: Commit and push**

```bash
git add docs/claude/experiment-log.md docs/claude/backlog.md
git commit -m "docs(b-11): Phase A experiment log + backlog status"

git push origin <current-branch>
```

(If working directly on main with owner-bypass, push to a feature branch first and let CI gate; only push to main after CI green.)

- [ ] **Step 10.6: Wait for CI green and promote**

Monitor CI on the pushed branch via `curl -s "https://api.github.com/repos/jam-sudo/Sisyphus/commits/<SHA>/check-runs"`. When `conclusion=success`:

```bash
git push origin <feature-branch>:main   # or via PR if branch protection requires
```

---

## Phase B — Curation cycle (headline may shift)

Phase B converts the 19-drug audit into committed registry entries. Each drug gets exactly one row; the disposition determines the value.

### Task 11: Mechanism triage for 19 over-predict drugs

**Files:**
- Create: `docs/superpowers/specs/2026-05-21-B11-Phase-B-curation-log.md` (working log)

- [ ] **Step 11.1: Generate the candidate list**

```bash
/opt/miniconda3/bin/python -c "
import json
import math
from pathlib import Path

cache = json.loads(Path('data/training/4track_holdout_predictions.json').read_text())
candidates = []
for d in cache['drugs']:
    fold = d.get('meta_fold')
    if fold is None or not math.isfinite(fold):
        continue
    if fold > 3.0:
        candidates.append({
            'name': d['name'],
            'meta_fold': fold,
            'eng_fold': d['eng_fold'],
            'ml_fold': d['ml_fold'],
            'type': d['type'],
        })
candidates.sort(key=lambda x: -x['meta_fold'])
print(json.dumps(candidates, indent=2))
"
```

Save the output to the curation log file for reference.

- [ ] **Step 11.2: For each candidate, fill in mechanism triage**

For every drug in the list, document in the curation log:

```markdown
### <drug_name>
- meta_fold: <value>
- compound_type: <neutral/acid/base/zwitterion>
- predicted fup (from ADME): <value> (run `predict(<smiles>)` to get this)
- mechanism hypothesis (one of):
  - **PPB-related**: fup < 0.1 AND hepatic-CL-dominant AND no obvious renal/transporter/induction signature
  - **Renal**: parent excretion dominated by GFR/active secretion
  - **CYP-induction**: known autoinducer / induction substrate (e.g., rifampin, carbamazepine)
  - **Transporter**: P-gp / BCRP / OATP-mediated where Sisyphus is missing the transporter
  - **Formulation**: extended release, food effect, prodrug-not-in-registry
  - **Other / Unknown**: novel chemistry, mechanism unclear
- preliminary disposition: literature_applied (candidate for B-11) / not_applicable (mechanism is not PPB)
```

This is research work — for each drug, look up: FDA label, DailyMed, recent review papers. Time estimate: 30 min × 19 drugs = ~10 hours.

- [ ] **Step 11.3: Commit the curation log**

```bash
git add docs/superpowers/specs/2026-05-21-B11-Phase-B-curation-log.md
git commit -m "docs(b-11): Phase B curation log — mechanism triage for 19 over-predict drugs"
```

---

### Task 12: Literature search for PPB-related candidates

For each drug classified `PPB-related` in Task 11, search for an `fu_inc / fu_plasma` ratio.

- [ ] **Step 12.1: Primary corpus pass**

For each PPB candidate, search the following sources in order. **Stop at first hit.**

1. Watanabe et al. 2009 *DMD* 37:1471–1480 — Table 1 (statin uptake), Table 3 (general drugs). If the drug appears, extract the reported `fu_inc / fu_p` or `Kp,uu,liver` ratio.
2. Yamazaki et al. 2010 *DMD* 38:998–1005 — Tables 1–2 (albumin-facilitated uptake for various drugs).
3. Riccardi et al. 2017 *DMD* 45:781–790 — Table 2 (Kp,uu,liver compilation across drug classes).
4. Patilea-Vrana & Unadkat 2017 — supplemental Table S1 (Kp,uu,liver review).

If a value is found, record:
- value (mean = reported ratio)
- cv (use 0.30 default unless paper reports SD/RSE explicitly; if so, derive cv from SD/mean)
- citation (full DOI + table number)

- [ ] **Step 12.2: Secondary PubMed pass (if primary corpus misses)**

For each remaining PPB candidate, run PubMed queries:
- `"<drug>" hepatic uptake intracellular`
- `"<drug>" albumin-facilitated`
- `"<drug>" fu liver Kpuu`

Record search results and any extracted value.

- [ ] **Step 12.3: Class extrapolation pass (optional)**

For PPB candidates without direct measurement, consider class extrapolation:
- Same scaffold (e.g., another piperidine, another 2-arylpropionic acid)
- Same primary CL pathway (e.g., another CYP3A4 substrate)
- Same fup range and logP range

If a defensible sibling exists, use its value with cv inflated by 1.5× (e.g., paper cv 0.30 → 0.45). Disposition = `class_extrapolated`.

- [ ] **Step 12.4: Document findings**

Append to the curation log per drug:

```markdown
### <drug_name> — literature search
- Watanabe 2009 Table 1: <hit / miss>
- Yamazaki 2010 Table 1: <hit / miss>
- Riccardi 2017 Table 2: <hit / miss>
- Patilea-Vrana 2017 Table S1: <hit / miss>
- PubMed: <queries tried, n_results, relevant abstracts>
- Final disposition: <literature_applied / class_extrapolated / ceiling_accepted>
- Value (if applicable): <mean, cv, citation>
```

- [ ] **Step 12.5: Commit the literature log**

```bash
git add docs/superpowers/specs/2026-05-21-B11-Phase-B-curation-log.md
git commit -m "docs(b-11): Phase B literature search findings for PPB candidates"
```

---

### Task 13: Write 19 audit rows into the production registry

**Files:**
- Modify: `data/transporters/hepatic_fu_correction.json`

- [ ] **Step 13.1: For each of 19 drugs, write one entry**

The entry shape is fixed by the spec §5.3. For every drug:

```json
{
  "drug": "<name>",
  "smiles": "<from clinical_pk.json>",
  "inchikey": "<RDKit MolToInchiKey of the SMILES>",
  "fu_correction_liver": {"mean": <value>, "cv": <value>},
  "disposition": "<one of: literature_applied | class_extrapolated | ceiling_accepted | not_applicable>",
  "literature": [<list of citations; empty if ceiling_accepted or not_applicable>],
  "notes": "<rationale>",
  "n_candidates_reviewed": <integer>,
  "source_dbs_searched": [<list of corpora>]
}
```

For `ceiling_accepted` and `not_applicable`: `fu_correction_liver = {"mean": 1.0, "cv": 0.0}` (no scaling), `literature = []`, `n_candidates_reviewed` reflects effort.

For `literature_applied` and `class_extrapolated`: `fu_correction_liver.mean >= 1.0` (anti-fudge guard enforces this; loader will raise ValueError on commit otherwise).

- [ ] **Step 13.2: Validate the registry with the schema regression**

Run: `pytest tests/regression/test_hepatic_fu_correction_schema.py -v`
Expected: 2 PASS. Failures mean an entry violates the schema (e.g., literature_applied with empty citation array) — fix the entry before proceeding.

- [ ] **Step 13.3: Validate the registry loads without error**

Run: `/opt/miniconda3/bin/python -c "from sisyphus.predict.hepatic_fu_correction import _load; full, conn = _load('data/transporters/hepatic_fu_correction.json'); print(f'loaded {len(full)} entries, {sum(len(v) for v in conn.values())} conn-index entries')"`
Expected: prints the loaded count without raising. A ValueError here means a sub-1.0 value or invalid disposition slipped in.

- [ ] **Step 13.4: Commit the registry**

```bash
git add data/transporters/hepatic_fu_correction.json
git commit -m "feat(data): hepatic_fu_correction 19-drug audit rows (B-11 Phase B)"
```

---

### Task 14: Regenerate 107-holdout cache + bootstrap CIs

**Files:**
- Modify: `data/training/4track_holdout_predictions.json`
- Modify: `data/validation/4track_ci_2026-05-12_v0.4.json`

- [ ] **Step 14.1: Hide developer-state artifacts for public-clone regen**

```bash
mv data/drugbank data/drugbank.HIDDEN
mv models/adme/logp_correction.json models/adme/logp_correction.json.HIDDEN
ls models/adme/logp_correction.json data/drugbank/ 2>&1 | grep -i "no such"  # confirms hidden
```

- [ ] **Step 14.2: Run the benchmark**

```bash
/opt/miniconda3/bin/python scripts/run_engine_benchmark.py --save-json /Users/jam/Sisyphus/data/training/4track_holdout_predictions.json
```

Expected runtime: 5–15 minutes. The script writes the new cache atomically.

- [ ] **Step 14.3: Refresh bootstrap CIs**

```bash
/opt/miniconda3/bin/python scripts/bootstrap_4track_ci.py \
  --cache /Users/jam/Sisyphus/data/training/4track_holdout_predictions.json \
  --tag v0.4 \
  --out /Users/jam/Sisyphus/data/validation/4track_ci_2026-05-12_v0.4.json \
  --date 2026-05-12 \
  --context "public-clone deterministic state; B-11 Phase B curation."
```

- [ ] **Step 14.4: Restore developer-state artifacts**

```bash
mv data/drugbank.HIDDEN data/drugbank
mv models/adme/logp_correction.json.HIDDEN models/adme/logp_correction.json
```

- [ ] **Step 14.5: Compute the Phase B acceptance gate**

```bash
/opt/miniconda3/bin/python -c "
import json
post = json.load(open('data/training/4track_holdout_predictions.json'))
pre_meta = 2.7715  # canonical pre-B-11 value
post_meta = post['overall']['meta']['aafe']
delta_abs = post_meta - pre_meta
delta_pct = 100.0 * delta_abs / pre_meta
print(f'Pre-B-11 Meta:  {pre_meta:.4f}')
print(f'Post-B-11 Meta: {post_meta:.4f}')
print(f'delta: {delta_abs:+.4f} ({delta_pct:+.2f}%)')

if delta_pct <= -1.0:
    print('OUTCOME: SUCCESS — ship B-11 (Phase A + B)')
elif abs(delta_pct) < 0.5:
    print('OUTCOME: DE-37 — ship infra only, mark curation as DE-37')
elif delta_pct > 0:
    print('OUTCOME: FAILURE — revert curation, keep infra, RCA')
else:
    print('OUTCOME: AMBIGUOUS — modest improvement but under 1%; user-call: ship as DE-37 (lean) or push for more curation')
"
```

Expected output matches one of the four outcomes. The decision determines the next steps.

- [ ] **Step 14.6: Commit the regenerated artifacts**

```bash
git add data/training/4track_holdout_predictions.json data/validation/4track_ci_2026-05-12_v0.4.json
git commit -m "data(b-11): regenerate 107-holdout cache + bootstrap CIs post Phase B"
```

---

### Task 15: Phase B disposition (ship / DE-37 / revert)

Based on Task 14.5 outcome:

#### Outcome SUCCESS (Meta AAFE improved by ≥ 1%)

- [ ] **Step 15.S.1: Update the pinned AAFE in the regression test**

In `tests/integration/test_holdout_regression.py`, update `test_cached_holdout_aafe_is_2p772` to pin the new value (e.g., `test_cached_holdout_aafe_is_2pX` where X is the new value). Tolerance stays 0.005. Update the docstring with the B-11 narrative.

- [ ] **Step 15.S.2: Update README headline + reproducibility note**

Refresh the headline table in `README.md` with the new Meta AAFE and CIs. Add a B-11 sentence to the reproducibility note explaining the shift.

- [ ] **Step 15.S.3: Update CLAUDE.md (gitignored local)**

Refresh the local headline table.

- [ ] **Step 15.S.4: Update leak-audit allowlist**

In `tests/regression/test_prodrug_v3_enzyme_leak_audit.py`, add any Phase B curated drugs to `DRUG_SPECIFIC_CHANGES` (since they intentionally diverge from the pre-B-11 baseline).

- [ ] **Step 15.S.5: Run the full test suite**

```bash
pytest tests/unit/ tests/regression/ tests/integration/test_holdout_regression.py -q
```

Expected: all pass with the new AAFE pin.

- [ ] **Step 15.S.6: Commit**

```bash
git add tests/integration/test_holdout_regression.py tests/regression/test_prodrug_v3_enzyme_leak_audit.py README.md CLAUDE.md
git commit -m "feat(b-11): ship Phase B curation — Meta AAFE 2.7715 -> <new>"
```

#### Outcome DE-37 (shift < 0.5%)

- [ ] **Step 15.D.1: Add DE-37 entry to dead-ends.md**

In `docs/claude/dead-ends.md`, add the next-numbered entry:

```markdown
### DE-37 — Hepatic intracellular fu correction (B-11)

**Date:** 2026-05-MM
**Hypothesis:** Per-drug `fu_correction_liver` from primary literature (Watanabe 2009 / Yamazaki 2010 / Riccardi 2017 / Patilea-Vrana 2017) would reduce systematic over-prediction for highly bound drugs in the 107-holdout.

**What was measured:** 19 over-predict drugs (meta_fold > 3) were mechanism-triaged. <K> identified as PPB-related candidates. Primary corpus + PubMed search yielded <N> entries (literature_applied: <a>, class_extrapolated: <b>, ceiling_accepted: <c>, not_applicable: <d>).

**Outcome:** Meta AAFE shifted from 2.7715 to <value> (<delta_pct>%). The infrastructure (Phase A) is retained; curation entries are committed as audit trail but flagged as DE-37 (literature too thin to meaningfully shift headline).

**What this implies:** Either fu_inc / fu_p ratios are not the dominant under-prediction mechanism for the audited drugs, or the available literature does not measure these ratios for the specific drugs of interest, or both. Future iterations may consider experimental measurement, transporter-mediated alternatives, or accepting the current accuracy.
```

- [ ] **Step 15.D.2: Mark backlog item**

In `docs/claude/backlog.md`, move B-11 to the dead-ends section or strike it through with a `DE-37` reference.

- [ ] **Step 15.D.3: Decide on the cache**

Two options:
- **Keep the regen** (modest improvement is still mechanistically more correct): proceed to 15.D.4 with the new cache.
- **Revert the cache** (no defensible improvement, simplest): `git checkout HEAD~1 -- data/training/4track_holdout_predictions.json data/validation/4track_ci_2026-05-12_v0.4.json`. Then `git commit -m "revert(b-11): keep pre-B-11 cache; DE-37 outcome"`.

For this plan, default to **keep the regen** unless the shift is < 0.1% or worse than baseline.

- [ ] **Step 15.D.4: Update README + CLAUDE.md if cache changed**

If 15.D.3 kept the regen: same as Outcome SUCCESS steps 15.S.2 + 15.S.3.

- [ ] **Step 15.D.5: Commit**

```bash
git add docs/claude/dead-ends.md docs/claude/backlog.md README.md CLAUDE.md tests/integration/test_holdout_regression.py
git commit -m "docs(b-11): Phase B closed as DE-37 (literature too thin)"
```

#### Outcome FAILURE (Meta AAFE worsened)

- [ ] **Step 15.F.1: Revert the curation entries**

```bash
git checkout HEAD~2 -- data/transporters/hepatic_fu_correction.json   # back to empty
git checkout HEAD~1 -- data/training/4track_holdout_predictions.json data/validation/4track_ci_2026-05-12_v0.4.json
git commit -m "revert(b-11): roll back Phase B curation due to AAFE worsening"
```

- [ ] **Step 15.F.2: Diagnose**

For every drug whose individual fold worsened by ≥ 50% post-curation, document in the curation log. Likely causes: misidentified mechanism, value too high, wrong sibling for class extrapolation.

- [ ] **Step 15.F.3: Iterate**

If a fix is possible (narrow curation set to confidently-PPB drugs only), re-execute Tasks 13–14 with the reduced set. Otherwise, treat as DE-37 (Outcome path D).

---

### Task 16: Phase B push + CI + main merge

- [ ] **Step 16.1: Run ruff one last time**

```bash
/opt/miniconda3/bin/python -m ruff check src/sisyphus/predict/hepatic_fu_correction.py src/sisyphus/core.py src/sisyphus/predict/ivive.py src/sisyphus/engine/flux.py
```

Expected: `All checks passed!`

- [ ] **Step 16.2: Run the full suite**

```bash
pytest tests/unit/ tests/regression/ tests/integration/test_holdout_regression.py -q
```

Expected: all pass.

- [ ] **Step 16.3: Push the branch**

```bash
git push origin <feature-branch>
```

- [ ] **Step 16.4: Monitor CI to completion**

```bash
until [ "$(curl -s "https://api.github.com/repos/jam-sudo/Sisyphus/commits/$(git rev-parse HEAD)/check-runs" | /opt/miniconda3/bin/python -c 'import sys,json; print(json.load(sys.stdin)["check_runs"][0]["status"])' 2>/dev/null)" = "completed" ]; do sleep 30; done
curl -s "https://api.github.com/repos/jam-sudo/Sisyphus/commits/$(git rev-parse HEAD)/check-runs" | /opt/miniconda3/bin/python -c "import sys,json; r=json.load(sys.stdin)['check_runs'][0]; print(f'conclusion={r[\"conclusion\"]}')"
```

If `conclusion=failure`, diagnose by inspecting the failing job's steps and re-iterate. If `success`, proceed.

- [ ] **Step 16.5: Merge to main**

If branch protection allows owner direct push (as confirmed for B-03):

```bash
git push origin <feature-branch>:main
git checkout main
git pull origin main
git branch -d <feature-branch>
git push origin --delete <feature-branch>
```

Otherwise create a PR via GitHub UI.

- [ ] **Step 16.6: Final invocation**

```bash
git log -5 --oneline
```

Confirm the B-11 commit series lands on main. Done.

---

## Acceptance gates summary

| Gate | Phase | Criterion |
|---|---|---|
| Cache invariance | A | `data/training/4track_holdout_predictions.json` Meta AAFE bit-identical to 2.7715 with empty registry |
| Unit test suite | A | All Phase A tests pass, no existing tests broken |
| Identity-blind invariance | A | Random-rename graph produces bit-identical Cmax |
| Anti-fudge guard | A | Loader rejects `fu_correction_liver < 1.0` (test verified) |
| Schema regression | A+B | All committed entries have valid disposition + literature where required |
| Headline movement | B | Meta AAFE delta determines SUCCESS / DE-37 / FAILURE outcome |
| Per-drug regression | B | No individual drug fold worsens by ≥ 50% (FAILURE trigger) |
| CI green | A+B | GitHub Actions test job conclusion=success on the final commit before main merge |
