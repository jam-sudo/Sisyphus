# Multi-Enzyme Prodrug Yield (B-04) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow a single prodrug registry entry to declare per-enzyme conversion yields, so drugs whose parent metabolism splits into a dead-end branch and an active-producing branch can be expressed mechanistically. Unblocks B-03 (clopidogrel: CES1 dead-end + CYP2C19 active).

**Architecture:** Optional `yield` field on each entry in `enzyme_affinity_for_conversion[<tag>]`. When present, the builder reads it for edges keyed on that tag. When absent, the builder falls back to the entry-level `conversion_yield_fraction` (current behaviour). The 6 existing single-enzyme entries remain unchanged. Builder emits one `ProdrugActivationEdge` per (site × tag) instead of one per site with collapsed tags. Engine already supports per-edge yield via `params.edge_param(edge_id, "conversion_yield")` — no engine changes.

**Tech Stack:** Python 3.10+ frozen dataclasses, pytest, JSON registries. No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md`.

---

## File Structure

**Modify:**
- `src/sisyphus/core.py` — add `enzyme_yields: dict[str, Distribution]` to `ActiveMetabolite`; propagate in `DrugOnGraph.sample()` and `DrugOnGraph.realize_means()`.
- `src/sisyphus/predict/registry.py` — parse optional per-enzyme `yield` field; validate "all-or-nothing" for multi-enzyme entries; return new 4-tuple `(am, obs_species, affinities, enzyme_yields)`.
- `src/sisyphus/predict/ivive.py` — unpack new 4-tuple from `lookup_active_metabolite`; thread `enzyme_yields` into `ActiveMetabolite` construction. (Single callsite at lines ~621-630.)
- `src/sisyphus/graph/builder.py` — replace single-edge-per-site loop with `(site × tag)` double loop; read `am.enzyme_yields.get(tag)` with entry-level fallback.

**Create:**
- `tests/unit/test_prodrug_per_enzyme_yield.py` — unit coverage for §7.1 of spec.
- `tests/regression/test_prodrug_v3_registry_schema.py` — schema validation gate (§7.2, §5.4 of spec).

**Untouched (verified bit-identical by Task 9 snapshot run):**
- `src/sisyphus/engine/flux.py` — engine already reads `conversion_yield` per edge.
- `src/sisyphus/graph/types.py` — `ProdrugActivationEdge` already carries per-edge yield Distribution.
- `data/sbi/prodrug_activation_registry.json` — no data changes in this PR.
- `tests/regression/test_prodrug_v2_snapshot.py` — re-run as verification only; pin values unchanged.

---

## Task 1: `ActiveMetabolite.enzyme_yields` field

**Goal:** Add the dataclass field. Defaults to empty dict (backward compat). No behavior change — wiring is in later tasks.

**Files:**
- Modify: `src/sisyphus/core.py:171-180` (the `ActiveMetabolite` field block)
- Test: `tests/unit/test_prodrug_per_enzyme_yield.py` (new)

- [ ] **Step 1.1: Write the failing dataclass test**

Create `tests/unit/test_prodrug_per_enzyme_yield.py`:

```python
"""Unit tests for per-enzyme prodrug yield (B-04).

See docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md
"""
from __future__ import annotations

import pytest

from sisyphus.core import ActiveMetabolite, Distribution


def _minimal_active(**overrides) -> ActiveMetabolite:
    base = dict(
        name="A",
        mw=200.0,
        fup=Distribution(0.5),
        CL_per_h=Distribution(10.0),
        Vd_L=Distribution(20.0),
        conversion_rate_per_h=Distribution(0.0),
        conversion_site="",
        conversion_yield_fraction=Distribution(1.0),
    )
    base.update(overrides)
    return ActiveMetabolite(**base)


class TestActiveMetaboliteEnzymeYields:
    def test_default_is_empty_dict(self):
        am = _minimal_active()
        assert am.enzyme_yields == {}

    def test_can_set_per_enzyme_yields(self):
        yields = {
            "CES1": Distribution(mean=0.0, cv=0.0),
            "CYP2C19": Distribution(mean=1.0, cv=0.30),
        }
        am = _minimal_active(enzyme_yields=yields)
        assert am.enzyme_yields == yields
```

- [ ] **Step 1.2: Run test to confirm failure**

Run: `pytest tests/unit/test_prodrug_per_enzyme_yield.py::TestActiveMetaboliteEnzymeYields -v`
Expected: FAIL — `TypeError: ActiveMetabolite.__init__() got an unexpected keyword argument 'enzyme_yields'`

- [ ] **Step 1.3: Add the field to `ActiveMetabolite`**

In `src/sisyphus/core.py`, locate the `ActiveMetabolite` dataclass (around line 171). The current trailing fields are:

```python
    conversion_yield_fraction: Distribution = field(
        default_factory=lambda: Distribution(1.0, cv=0.0)
    )
```

Replace with:

```python
    conversion_yield_fraction: Distribution = field(
        default_factory=lambda: Distribution(1.0, cv=0.0)
    )
    enzyme_yields: dict[str, Distribution] = field(default_factory=dict)
    """Per-enzyme conversion yield (B-04).

    Optional override of ``conversion_yield_fraction`` keyed by enzyme tag.
    When ``enzyme_yields[tag]`` is set, the builder uses it for edges
    catalyzed by ``tag``; when unset, edges fall back to the entry-level
    ``conversion_yield_fraction``. Empty dict (default) preserves the
    pre-B-04 single-yield behaviour.

    See docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md
    """
```

Also update the docstring near the top of the class (line ~157-169) — append to the `Attributes:` block:

```
        enzyme_yields: Optional per-enzyme yield overrides (B-04). Defaults
            to empty dict.
```

- [ ] **Step 1.4: Run tests to verify pass**

Run: `pytest tests/unit/test_prodrug_per_enzyme_yield.py::TestActiveMetaboliteEnzymeYields -v`
Expected: PASS (2 tests).

Also run the existing prodrug suite to confirm no regression:
Run: `pytest tests/unit/test_prodrug_v2_drug.py tests/unit/test_prodrug_v2_augment.py tests/unit/test_prodrug_v2_registry.py -v`
Expected: all PASS, no behavior change.

- [ ] **Step 1.5: Commit**

```bash
git add src/sisyphus/core.py tests/unit/test_prodrug_per_enzyme_yield.py
git commit -m "feat(core): ActiveMetabolite.enzyme_yields field (B-04)

Per-enzyme yield override for prodrug activation. Empty dict default
preserves pre-B-04 behaviour. Wiring lands in subsequent tasks.

Spec: docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md"
```

---

## Task 2: Propagate `enzyme_yields` through `DrugOnGraph.sample()` and `.realize_means()`

**Goal:** When `DrugOnGraph.sample(rng)` and `.realize_means()` rebuild their `ActiveMetabolite`, they currently drop any fields not explicitly listed. They must propagate the new `enzyme_yields` dict.

**Files:**
- Modify: `src/sisyphus/core.py:319-331` (`sample()` ActiveMetabolite reconstruction)
- Modify: `src/sisyphus/core.py:382-394` (`realize_means()` ActiveMetabolite reconstruction)
- Test: `tests/unit/test_prodrug_per_enzyme_yield.py`

- [ ] **Step 2.1: Write the failing test**

Append to `tests/unit/test_prodrug_per_enzyme_yield.py`:

```python
import numpy as np

from sisyphus.core import DrugOnGraph
from tests.unit.test_prodrug_v2_drug import _minimal_drug


class TestDrugOnGraphPropagatesEnzymeYields:
    def _drug_with_yields(self) -> DrugOnGraph:
        am = _minimal_active(
            enzyme_yields={
                "CES1": Distribution(mean=0.0, cv=0.0),
                "CYP2C19": Distribution(mean=1.0, cv=0.30),
            },
        )
        return _minimal_drug(
            active_metabolite=am,
            observation_species="parent",
            enzyme_affinity_for_conversion={
                "CES1": Distribution(100.0),
                "CYP2C19": Distribution(50.0),
            },
        )

    def test_sample_propagates_enzyme_yields(self):
        drug = self._drug_with_yields()
        rng = np.random.default_rng(42)
        sampled = drug.sample(rng)
        assert set(sampled.active_metabolite.enzyme_yields.keys()) == {"CES1", "CYP2C19"}
        # cv=0 entries must round-trip exactly
        assert sampled.active_metabolite.enzyme_yields["CES1"].mean == 0.0

    def test_realize_means_propagates_enzyme_yields(self):
        drug = self._drug_with_yields()
        realized = drug.realize_means()
        assert realized.active_metabolite.enzyme_yields["CES1"].mean == 0.0
        assert realized.active_metabolite.enzyme_yields["CYP2C19"].mean == 1.0
        # realize_means must produce cv=0 deterministic Distributions
        assert realized.active_metabolite.enzyme_yields["CYP2C19"].cv == 0.0

    def test_sample_propagates_empty_enzyme_yields(self):
        """Backward compat: existing entries (no per-enzyme yields) round-trip empty dict."""
        drug = _minimal_drug(
            active_metabolite=_minimal_active(),  # no enzyme_yields override
            observation_species="parent",
            enzyme_affinity_for_conversion={"SPR": Distribution(100.0)},
        )
        rng = np.random.default_rng(0)
        sampled = drug.sample(rng)
        assert sampled.active_metabolite.enzyme_yields == {}
```

- [ ] **Step 2.2: Run test to confirm failure**

Run: `pytest tests/unit/test_prodrug_per_enzyme_yield.py::TestDrugOnGraphPropagatesEnzymeYields -v`
Expected: FAIL — propagated `enzyme_yields` will be `{}` even when set (the reconstruction blocks drop it).

- [ ] **Step 2.3: Patch `DrugOnGraph.sample()`**

In `src/sisyphus/core.py`, locate the `ActiveMetabolite(...)` constructor inside `sample()` (around line 319). The current trailing field is:

```python
                conversion_yield_fraction=Distribution(
                    mean=self.active_metabolite.conversion_yield_fraction.sample(rng),
                    cv=0.0),
            ) if self.active_metabolite is not None else None,
```

Replace with:

```python
                conversion_yield_fraction=Distribution(
                    mean=self.active_metabolite.conversion_yield_fraction.sample(rng),
                    cv=0.0),
                enzyme_yields={
                    k: Distribution(mean=v.sample(rng), cv=0.0)
                    for k, v in self.active_metabolite.enzyme_yields.items()
                },
            ) if self.active_metabolite is not None else None,
```

- [ ] **Step 2.4: Patch `DrugOnGraph.realize_means()`**

In `src/sisyphus/core.py`, locate the analogous block inside `realize_means()` (around line 382). The current trailing field is:

```python
                conversion_yield_fraction=Distribution(
                    mean=self.active_metabolite.conversion_yield_fraction.mean,
                    cv=0.0),
            ) if self.active_metabolite is not None else None,
```

Replace with:

```python
                conversion_yield_fraction=Distribution(
                    mean=self.active_metabolite.conversion_yield_fraction.mean,
                    cv=0.0),
                enzyme_yields={
                    k: Distribution(mean=v.mean, cv=0.0)
                    for k, v in self.active_metabolite.enzyme_yields.items()
                },
            ) if self.active_metabolite is not None else None,
```

- [ ] **Step 2.5: Run tests to verify pass**

Run: `pytest tests/unit/test_prodrug_per_enzyme_yield.py -v`
Expected: PASS (all tests so far).

Run: `pytest tests/unit/test_prodrug_v2_drug.py -v`
Expected: PASS (backward compat preserved).

- [ ] **Step 2.6: Commit**

```bash
git add src/sisyphus/core.py tests/unit/test_prodrug_per_enzyme_yield.py
git commit -m "feat(core): propagate enzyme_yields through sample/realize_means (B-04)

DrugOnGraph.sample(rng) and .realize_means() now carry the
ActiveMetabolite.enzyme_yields dict through reconstruction. Empty-dict
default keeps existing single-enzyme entries bit-identical.

Spec: docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md"
```

---

## Task 3: Registry loader parses optional per-enzyme `yield` + validates all-or-nothing

**Goal:** `_build_enzyme_affinity_for_conversion` extracts an optional `yield: {mean, cv}` block per enzyme entry; a new sibling helper returns the `enzyme_yields` dict. `lookup_active_metabolite` returns a 4-tuple. Validation rule (§5.4): for entries with ≥2 enzymes, every enzyme MUST declare a `yield` (mixed declarations raise `ValueError`).

**Files:**
- Modify: `src/sisyphus/predict/registry.py`
- Test: `tests/unit/test_prodrug_per_enzyme_yield.py` (append)

- [ ] **Step 3.1: Write the failing loader test**

Append to `tests/unit/test_prodrug_per_enzyme_yield.py`:

```python
import json
from pathlib import Path

from sisyphus.predict.registry import lookup_active_metabolite
from tests.unit.test_prodrug_v2_registry import _v2_entry


class TestRegistryParsesPerEnzymeYield:
    def _write(self, tmp_path: Path, entries: dict) -> Path:
        p = tmp_path / "registry.json"
        p.write_text(json.dumps(entries))
        return p

    def test_lookup_returns_four_tuple(self, tmp_path):
        """Single-enzyme entry: 4-tuple, empty enzyme_yields dict."""
        reg = self._write(tmp_path, {"C": _v2_entry()})
        result = lookup_active_metabolite("C", registry_path=reg)
        assert result is not None
        assert len(result) == 4
        am, obs, affinities, enzyme_yields = result
        assert affinities["SPR"].mean == 50.0
        assert enzyme_yields == {}

    def test_single_enzyme_with_yield_is_parsed(self, tmp_path):
        """Optional per-enzyme yield on a single-enzyme entry round-trips."""
        entry = _v2_entry(
            enzyme_affinity_for_conversion={
                "SPR": {
                    "mean": 50.0,
                    "cv": 0.5,
                    "yield": {"mean": 0.5, "cv": 0.1},
                }
            }
        )
        reg = self._write(tmp_path, {"C": entry})
        _, _, _, enzyme_yields = lookup_active_metabolite("C", registry_path=reg)
        assert "SPR" in enzyme_yields
        assert enzyme_yields["SPR"].mean == 0.5
        assert enzyme_yields["SPR"].cv == 0.1

    def test_multi_enzyme_with_all_yields_is_parsed(self, tmp_path):
        """Multi-enzyme entry with per-enzyme yield on every enzyme."""
        entry = _v2_entry(
            enzyme_affinity_for_conversion={
                "CES1": {
                    "mean": 100.0, "cv": 0.5,
                    "yield": {"mean": 0.0, "cv": 0.0},
                },
                "CYP2C19": {
                    "mean": 30.0, "cv": 0.4,
                    "yield": {"mean": 1.0, "cv": 0.30},
                },
            }
        )
        reg = self._write(tmp_path, {"C": entry})
        _, _, affinities, enzyme_yields = lookup_active_metabolite("C", registry_path=reg)
        assert set(affinities.keys()) == {"CES1", "CYP2C19"}
        assert set(enzyme_yields.keys()) == {"CES1", "CYP2C19"}
        assert enzyme_yields["CES1"].mean == 0.0
        assert enzyme_yields["CYP2C19"].mean == 1.0

    def test_multi_enzyme_missing_yield_raises(self, tmp_path):
        """Multi-enzyme with partial declaration is rejected at load time."""
        entry = _v2_entry(
            enzyme_affinity_for_conversion={
                "CES1": {
                    "mean": 100.0, "cv": 0.5,
                    "yield": {"mean": 0.0, "cv": 0.0},
                },
                "CYP2C19": {"mean": 30.0, "cv": 0.4},  # no yield
            }
        )
        reg = self._write(tmp_path, {"C": entry})
        with pytest.raises(ValueError, match="yield"):
            lookup_active_metabolite("C", registry_path=reg)

    def test_yield_out_of_range_raises(self, tmp_path):
        """Per-enzyme yield must satisfy 0 <= mean <= 1."""
        entry = _v2_entry(
            enzyme_affinity_for_conversion={
                "SPR": {
                    "mean": 50.0, "cv": 0.5,
                    "yield": {"mean": 1.5, "cv": 0.0},
                }
            }
        )
        reg = self._write(tmp_path, {"C": entry})
        with pytest.raises(ValueError, match="yield"):
            lookup_active_metabolite("C", registry_path=reg)
```

- [ ] **Step 3.2: Run tests to confirm failure**

Run: `pytest tests/unit/test_prodrug_per_enzyme_yield.py::TestRegistryParsesPerEnzymeYield -v`
Expected: FAIL — `assert len(result) == 4` fails (current return is 3-tuple).

- [ ] **Step 3.3: Extend `_build_enzyme_affinity_for_conversion` to parse `yield`**

In `src/sisyphus/predict/registry.py`, replace the existing `_build_enzyme_affinity_for_conversion` function (lines 108-125) with:

```python
def _build_enzyme_affinity_for_conversion(
    entry: dict, smiles: str
) -> tuple[dict[str, Distribution], dict[str, Distribution]]:
    """Parse enzyme_affinity_for_conversion dict; ignore citation keys.

    Returns ``(affinities, enzyme_yields)``. ``enzyme_yields`` is empty
    when no entry declares a per-enzyme ``yield`` field. When ≥2 enzymes
    are declared, every enzyme MUST declare ``yield`` (all-or-nothing) —
    mixed declarations raise ValueError. See B-04 spec §5.4.
    """
    raw = entry["enzyme_affinity_for_conversion"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"enzyme_affinity_for_conversion must be non-empty dict for SMILES {smiles!r}"
        )
    affinities: dict[str, Distribution] = {}
    yields: dict[str, Distribution] = {}
    for tag, dist_raw in raw.items():
        if not isinstance(dist_raw, dict):
            raise ValueError(
                f"affinity entry for tag {tag!r} must be dict with 'mean'/'cv', "
                f"got {type(dist_raw).__name__}"
            )
        if "mean" not in dist_raw:
            raise ValueError(f"affinity entry for tag {tag!r} missing 'mean'")
        affinities[tag] = _distribution_from_dict(dist_raw)
        if "yield" in dist_raw:
            y = dist_raw["yield"]
            if not isinstance(y, dict) or "mean" not in y:
                raise ValueError(
                    f"per-enzyme yield for tag {tag!r} must be dict with 'mean'/'cv', "
                    f"got {y!r}"
                )
            y_mean = float(y["mean"])
            if not (0.0 <= y_mean <= 1.0):
                raise ValueError(
                    f"per-enzyme yield for tag {tag!r} must be in [0, 1], got {y_mean}"
                )
            yields[tag] = _distribution_from_dict(y)

    # All-or-nothing rule for multi-enzyme entries (spec §5.4).
    if len(affinities) >= 2 and yields and len(yields) != len(affinities):
        missing = sorted(set(affinities) - set(yields))
        raise ValueError(
            f"prodrug registry entry for SMILES {smiles!r}: multi-enzyme entries "
            f"must declare per-enzyme 'yield' for every enzyme or none. "
            f"Missing yield for: {missing}"
        )

    return affinities, yields
```

- [ ] **Step 3.4: Update `lookup_active_metabolite` to return 4-tuple**

In the same file, update `lookup_active_metabolite` signature, docstring, and the two trailing lines. Current trailing lines (around 173-175):

```python
    am = _build_active_metabolite(entry, canonical)
    affinities = _build_enzyme_affinity_for_conversion(entry, canonical)
    return am, obs_species, affinities
```

Replace with:

```python
    am = _build_active_metabolite(entry, canonical)
    affinities, enzyme_yields = _build_enzyme_affinity_for_conversion(entry, canonical)
    return am, obs_species, affinities, enzyme_yields
```

And update the function signature and docstring (around 128-137):

```python
def lookup_active_metabolite(
    smiles: str, registry_path: Path | None = None
) -> tuple[ActiveMetabolite, str, dict[str, Distribution], dict[str, Distribution]] | None:
    """Look up SMILES in v2 prodrug registry.

    Returns ``(ActiveMetabolite, observation_species,
    enzyme_affinity_for_conversion, enzyme_yields)`` or ``None`` if not
    found. ``enzyme_yields`` is empty when the entry does not declare
    per-enzyme yields (single-enzyme entries; backward-compat path).

    Raises ``ValueError`` on invalid registry entries.
    """
```

- [ ] **Step 3.5: Run new tests + existing registry tests**

Run: `pytest tests/unit/test_prodrug_per_enzyme_yield.py::TestRegistryParsesPerEnzymeYield -v`
Expected: PASS (5 tests).

Run: `pytest tests/unit/test_prodrug_v2_registry.py -v`
Expected: FAIL — `test_lookup_returns_three_tuple` and similar pre-existing tests unpack a 3-tuple. They must be updated.

- [ ] **Step 3.6: Update existing 3-tuple unpacking in `test_prodrug_v2_registry.py`**

In `tests/unit/test_prodrug_v2_registry.py`, every test that does
`am, obs, affinities = lookup_active_metabolite(...)` or asserts `len(result) == 3`
needs to be updated to a 4-tuple. Grep for the pattern and fix:

Run: `grep -n "len(result)\|am, obs, affinities" tests/unit/test_prodrug_v2_registry.py`

For `test_lookup_returns_three_tuple`, rename to `test_lookup_returns_four_tuple` and update:

```python
def test_lookup_returns_four_tuple(tmp_path):
    smiles = "C"
    canonical = "C"
    reg = _write_registry(tmp_path, {canonical: _v2_entry()})
    result = lookup_active_metabolite(smiles, registry_path=reg)
    assert result is not None
    assert len(result) == 4
    am, obs, affinities, enzyme_yields = result
    assert isinstance(am, ActiveMetabolite)
    assert obs == "parent"
    assert "SPR" in affinities
    assert affinities["SPR"].mean == 50.0
    assert affinities["SPR"].cv == 0.5
    assert enzyme_yields == {}
```

For any other test that unpacks the result (`test_loader_strips_citation_keys_from_distribution` and similar), change `_, _, affinities = result` to `_, _, affinities, _ = result`.

- [ ] **Step 3.7: Verify pass**

Run: `pytest tests/unit/test_prodrug_v2_registry.py tests/unit/test_prodrug_per_enzyme_yield.py -v`
Expected: all PASS.

- [ ] **Step 3.8: Commit**

```bash
git add src/sisyphus/predict/registry.py tests/unit/test_prodrug_v2_registry.py \
    tests/unit/test_prodrug_per_enzyme_yield.py
git commit -m "feat(registry): per-enzyme yield + all-or-nothing rule (B-04)

lookup_active_metabolite now returns a 4-tuple including enzyme_yields
parsed from optional 'yield' field on each enzyme_affinity_for_conversion
entry. Multi-enzyme entries must declare yield for every enzyme.

Spec §5.4: docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md"
```

---

## Task 4: Update `ivive.build_drug_on_graph` callsite to consume 4-tuple

**Goal:** The single internal caller of `lookup_active_metabolite` (`src/sisyphus/predict/ivive.py:621-630`) currently unpacks a 3-tuple. Switch to 4-tuple and thread the new `enzyme_yields` dict into the `ActiveMetabolite` it constructs.

**Files:**
- Modify: `src/sisyphus/predict/ivive.py:621-630` and the `ActiveMetabolite(...)` construction
- Test: existing prodrug snapshot tests (verification only)

- [ ] **Step 4.1: Locate the callsite**

Run: `grep -n "lookup_active_metabolite\|conv_affinities" src/sisyphus/predict/ivive.py`
Expect output: ~5 lines including the unpacking at line 626 and the use at line 728.

- [ ] **Step 4.2: Update the unpacking**

In `src/sisyphus/predict/ivive.py`, locate lines 621-630:

```python
    registry_result = lookup_active_metabolite(profile.smiles)
    if registry_result is not None:
        active_metabolite, observation_species, conv_affinities = registry_result
    else:
        active_metabolite = None
        observation_species = "parent"
        conv_affinities = {}
```

Replace with:

```python
    registry_result = lookup_active_metabolite(profile.smiles)
    if registry_result is not None:
        active_metabolite, observation_species, conv_affinities, conv_enzyme_yields = registry_result
        # Attach the per-enzyme yields onto the ActiveMetabolite instance.
        # registry builds AM with enzyme_yields={} by default; we replace
        # that field here since AM is frozen.
        if conv_enzyme_yields:
            from dataclasses import replace
            active_metabolite = replace(active_metabolite, enzyme_yields=conv_enzyme_yields)
    else:
        active_metabolite = None
        observation_species = "parent"
        conv_affinities = {}
```

Add the `from dataclasses import replace` import at the top of the file if it's not already there. Run `grep -n "from dataclasses" src/sisyphus/predict/ivive.py` first — if absent, add it to the existing imports near the top.

- [ ] **Step 4.3: Verify no other callers exist**

Run: `grep -rn "lookup_active_metabolite" src/ tests/`
Expected: only `src/sisyphus/predict/registry.py` (definition), `src/sisyphus/predict/ivive.py:621` (just patched), and test files in `tests/unit/test_prodrug_v2_registry.py` + `tests/unit/test_prodrug_per_enzyme_yield.py` (both already updated in Task 3).

If any production callsite outside these is found, update it to consume the 4-tuple.

- [ ] **Step 4.4: Run the prodrug snapshot regression**

Run: `pytest tests/regression/test_prodrug_v2_snapshot.py -v`
Expected: PASS for the 4 snapshot drugs (sepiapterin/remdesivir/tebipenem_pivoxil/fostamatinib) at the pinned Cmax values. None of these 4 declare per-enzyme yields, so behavior must be bit-identical.

If a snapshot drifts, stop and investigate — the wiring path through ivive is suspect.

- [ ] **Step 4.5: Run full prodrug suite**

Run: `pytest tests/unit/test_prodrug_v2_drug.py tests/unit/test_prodrug_v2_augment.py tests/unit/test_prodrug_v2_registry.py tests/unit/test_prodrug_per_enzyme_yield.py tests/regression/test_prodrug_v2_snapshot.py tests/regression/test_prodrug_v2_identity_blind.py tests/regression/test_prodrug_v2_validation_gate.py tests/regression/test_prodrug_registry_seed.py -v`
Expected: all PASS (snapshot bit-identical, no regression).

- [ ] **Step 4.6: Commit**

```bash
git add src/sisyphus/predict/ivive.py
git commit -m "feat(ivive): thread enzyme_yields into ActiveMetabolite (B-04)

build_drug_on_graph unpacks the new 4-tuple from
lookup_active_metabolite and replaces the ActiveMetabolite's
enzyme_yields field when the registry entry declares them. No-op
for single-enzyme entries (snapshot bit-identical).

Spec: docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md"
```

---

## Task 5: Builder emits one edge per (site × tag) with per-enzyme yield lookup

**Goal:** Replace the single-edge-per-site loop in `augment_for_active_species` with a double loop over `(site, tag)` pairs, where `tag` is intersected against `node.enzymes`. For each emitted edge, look up `am.enzyme_yields.get(tag)` and use it; fall back to `am.conversion_yield_fraction` when absent. `enzyme_tags` on the emitted edge becomes a single-element frozenset (one tag per edge).

**Files:**
- Modify: `src/sisyphus/graph/builder.py:358-368` (the `for site in conversion_sites:` loop)
- Test: `tests/unit/test_prodrug_per_enzyme_yield.py` (append)

- [ ] **Step 5.1: Write the failing builder test**

Append to `tests/unit/test_prodrug_per_enzyme_yield.py`:

```python
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import augment_for_active_species
from sisyphus.graph.types import Node, ProdrugActivationEdge


def _graph_with_ces1_and_cyp2c19_in_liver() -> BodyGraph:
    g = BodyGraph()
    g.add_node(Node(
        name="liver",
        node_type="organ",
        volume=Distribution(1.5),
        enzymes={
            "CES1": Distribution(mean=8e7, cv=0.5),
            "CYP2C19": Distribution(mean=1.4e6, cv=0.6),
        },
        ivive_scaling=6e-5,
    ))
    g.add_node(Node(name="venous_blood", node_type="blood_pool",
                    volume=Distribution(5.0)))
    g.add_node(Node(name="metabolized_gut", node_type="sink",
                    volume=Distribution(0.0)))
    return g


class TestBuilderPerEnzymeYield:
    def test_single_enzyme_entry_falls_back_to_entry_level_yield(self):
        """Backward compat: single-enzyme entry produces edge with entry-level yield."""
        g = _graph_with_ces1_and_cyp2c19_in_liver()
        am = _minimal_active(
            conversion_yield_fraction=Distribution(0.85, cv=0.1),
            # enzyme_yields={} default
        )
        drug = _minimal_drug(
            active_metabolite=am,
            observation_species="parent",
            enzyme_affinity_for_conversion={"CES1": Distribution(100.0)},
        )
        augment_for_active_species(g, drug)
        edges = [e for e in g.edges if isinstance(e, ProdrugActivationEdge)]
        assert len(edges) == 1
        assert edges[0].enzyme_tags == frozenset({"CES1"})
        assert edges[0].conversion_yield.mean == 0.85

    def test_multi_enzyme_entry_emits_per_tag_edges(self):
        """Multi-enzyme entry: one edge per tag per site, each with its own yield."""
        g = _graph_with_ces1_and_cyp2c19_in_liver()
        am = _minimal_active(
            conversion_yield_fraction=Distribution(0.15, cv=0.4),
            enzyme_yields={
                "CES1": Distribution(0.0, cv=0.0),
                "CYP2C19": Distribution(1.0, cv=0.30),
            },
        )
        drug = _minimal_drug(
            active_metabolite=am,
            observation_species="parent",
            enzyme_affinity_for_conversion={
                "CES1": Distribution(100.0),
                "CYP2C19": Distribution(50.0),
            },
        )
        augment_for_active_species(g, drug)
        edges = [e for e in g.edges if isinstance(e, ProdrugActivationEdge)]
        # 1 site × 2 tags = 2 edges
        assert len(edges) == 2
        by_tag = {next(iter(e.enzyme_tags)): e for e in edges}
        assert set(by_tag.keys()) == {"CES1", "CYP2C19"}
        assert by_tag["CES1"].conversion_yield.mean == 0.0
        assert by_tag["CYP2C19"].conversion_yield.mean == 1.0
        # Each edge carries exactly one tag (per-enzyme edge).
        assert by_tag["CES1"].enzyme_tags == frozenset({"CES1"})
        assert by_tag["CYP2C19"].enzyme_tags == frozenset({"CYP2C19"})

    def test_dead_end_yield_zero_is_valid(self):
        """yield=0 produces an edge that consumes parent but contributes no active."""
        g = _graph_with_ces1_and_cyp2c19_in_liver()
        am = _minimal_active(
            enzyme_yields={
                "CES1": Distribution(0.0, cv=0.0),
                "CYP2C19": Distribution(1.0, cv=0.0),
            },
        )
        drug = _minimal_drug(
            active_metabolite=am,
            observation_species="parent",
            enzyme_affinity_for_conversion={
                "CES1": Distribution(100.0),
                "CYP2C19": Distribution(50.0),
            },
        )
        augment_for_active_species(g, drug)
        ces1_edges = [
            e for e in g.edges
            if isinstance(e, ProdrugActivationEdge) and e.enzyme_tags == frozenset({"CES1"})
        ]
        assert len(ces1_edges) == 1
        # Yield is exactly 0 — engine flux multiplies active production by this.
        assert ces1_edges[0].conversion_yield.mean == 0.0
        # But parent-side flux (cl_organ × c_out) still consumes parent. The
        # edge exists; engine flux logic handles the rest. We verify the
        # edge is present, not its dynamics here.

    def test_multi_enzyme_entry_with_partial_node_coverage(self):
        """If a node holds only some of the declared tags, only those edges are emitted."""
        # liver has both CES1 and CYP2C19; add a gut_wall with only CES1.
        g = _graph_with_ces1_and_cyp2c19_in_liver()
        g.add_node(Node(
            name="gut_wall",
            node_type="barrier_organ",
            volume=Distribution(1.0),
            enzymes={"CES1": Distribution(mean=1e6, cv=0.5)},
            ivive_scaling=6e-5,
        ))
        am = _minimal_active(
            enzyme_yields={
                "CES1": Distribution(0.0, cv=0.0),
                "CYP2C19": Distribution(1.0, cv=0.0),
            },
        )
        drug = _minimal_drug(
            active_metabolite=am,
            observation_species="parent",
            enzyme_affinity_for_conversion={
                "CES1": Distribution(100.0),
                "CYP2C19": Distribution(50.0),
            },
        )
        augment_for_active_species(g, drug)
        edges = [e for e in g.edges if isinstance(e, ProdrugActivationEdge)]
        # liver: CES1 + CYP2C19 = 2 edges. gut_wall: CES1 only = 1 edge. total 3.
        assert len(edges) == 3
        by_source_tag = {(e.source, next(iter(e.enzyme_tags))) for e in edges}
        assert by_source_tag == {
            ("liver", "CES1"),
            ("liver", "CYP2C19"),
            ("gut_wall", "CES1"),
        }
```

- [ ] **Step 5.2: Run tests to confirm failure**

Run: `pytest tests/unit/test_prodrug_per_enzyme_yield.py::TestBuilderPerEnzymeYield -v`
Expected: FAIL — current builder emits 1 edge per site with collapsed tags; `len(edges) == 2` (multi) and `len(edges) == 3` (mixed coverage) will fail.

- [ ] **Step 5.3: Patch the builder loop**

In `src/sisyphus/graph/builder.py`, locate the loop near line 358:

```python
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
```

Replace with:

```python
    # One ProdrugActivationEdge per (site × tag) intersection.
    # Per-enzyme yield with entry-level fallback (B-04 §5.2).
    for site in conversion_sites:
        node_tags = set(graph.nodes[site].enzymes.keys())
        for tag in sorted(enzyme_tags & node_tags):  # sorted for deterministic edge order
            yld = am.enzyme_yields.get(tag, am.conversion_yield_fraction)
            activation_edge = ProdrugActivationEdge(
                source=site,
                target=active_node_name,
                enzyme_tags=frozenset({tag}),
                conversion_yield=yld,
                mw_parent=drug.mw,
                mw_active=am.mw,
            )
            graph.add_edge(activation_edge)
```

- [ ] **Step 5.4: Run new builder tests + existing builder tests**

Run: `pytest tests/unit/test_prodrug_per_enzyme_yield.py::TestBuilderPerEnzymeYield -v`
Expected: PASS (4 tests).

Run: `pytest tests/unit/test_prodrug_v2_augment.py -v`
Expected: PASS — existing test expects `len(activation_edges) == 2` for an SPR-only multi-site entry (liver + gut_wall). After Task 5, this becomes (2 sites × 1 tag) = 2 edges, unchanged. The `enzyme_tags == frozenset({"SPR"})` assertion also still holds (single-element frozenset).

- [ ] **Step 5.5: Run prodrug snapshot regression**

Run: `pytest tests/regression/test_prodrug_v2_snapshot.py -v`
Expected: PASS, bit-identical to pre-B-04. The 6 production entries are all single-enzyme — the builder now emits one edge per (site × single-tag) which is exactly the same edge count as before. Engine sums flux from those edges → identical result.

If a snapshot drifts: stop. Most likely cause is `enzyme_tags` was previously a multi-element frozenset for some entry that we missed in the audit. Investigate.

- [ ] **Step 5.6: Commit**

```bash
git add src/sisyphus/graph/builder.py tests/unit/test_prodrug_per_enzyme_yield.py
git commit -m "feat(builder): per-(site,tag) prodrug activation edges (B-04)

augment_for_active_species now emits one ProdrugActivationEdge per
(conversion_site × enzyme_tag) pair instead of one per site with
collapsed tags. Each edge uses am.enzyme_yields.get(tag, am.conversion_yield_fraction).

Single-enzyme entries (all 6 production entries) emit identical edge
counts and yields as before; snapshot bit-identical.

Spec §5.2: docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md"
```

---

## Task 6: Schema regression test for the all-or-nothing rule on production registry

**Goal:** Add a regression test that asserts every entry in the production registry either declares no per-enzyme yields or declares them on every enzyme. Companion to the loader-level test (Task 3) — this one guards the production JSON file.

**Files:**
- Create: `tests/regression/test_prodrug_v3_registry_schema.py`

- [ ] **Step 6.1: Write the regression test**

Create `tests/regression/test_prodrug_v3_registry_schema.py`:

```python
"""Schema regression for per-enzyme yield rule (B-04).

Asserts:
  1. Every entry in the production registry satisfies the all-or-nothing
     per-enzyme yield rule (spec §5.4): if any enzyme declares 'yield',
     all enzymes must.
  2. Multi-enzyme entries either declare per-enzyme yields or have a
     valid entry-level fallback.

Mirrors the test_oatp_registry_schema.py pattern (paired-registry gate).
"""
from __future__ import annotations

import json
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "data" / "sbi" / "prodrug_activation_registry.json"


def _load() -> dict:
    return json.loads(_REGISTRY_PATH.read_text())


def test_all_or_nothing_per_enzyme_yield():
    """For each registry entry, either zero or all enzymes declare 'yield'."""
    data = _load()
    for smiles, entry in data.items():
        if not isinstance(entry, dict):
            continue
        affinities = entry.get("enzyme_affinity_for_conversion", {})
        if not affinities:
            continue
        n_with_yield = sum(
            1 for tag, dist in affinities.items()
            if isinstance(dist, dict) and "yield" in dist
        )
        if n_with_yield == 0:
            # All fall back to entry-level conversion_yield_fraction. Valid.
            continue
        assert n_with_yield == len(affinities), (
            f"prodrug entry {entry.get('name', smiles)!r}: mixed per-enzyme "
            f"yield declaration. {n_with_yield}/{len(affinities)} enzymes "
            f"declare 'yield'. Must be all or none (spec §5.4)."
        )


def test_per_enzyme_yield_in_unit_interval():
    """Each per-enzyme yield 'mean' is in [0, 1]."""
    data = _load()
    for smiles, entry in data.items():
        if not isinstance(entry, dict):
            continue
        for tag, dist in entry.get("enzyme_affinity_for_conversion", {}).items():
            if not isinstance(dist, dict):
                continue
            y = dist.get("yield")
            if y is None:
                continue
            assert isinstance(y, dict) and "mean" in y, (
                f"{entry.get('name', smiles)!r} enzyme {tag!r}: 'yield' "
                f"must be a dict with 'mean', got {y!r}"
            )
            assert 0.0 <= float(y["mean"]) <= 1.0, (
                f"{entry.get('name', smiles)!r} enzyme {tag!r}: yield mean "
                f"{y['mean']} out of [0, 1]"
            )
```

- [ ] **Step 6.2: Run to confirm pass on current registry (all entries are single-enzyme, no per-enzyme yield)**

Run: `pytest tests/regression/test_prodrug_v3_registry_schema.py -v`
Expected: PASS — the 6 current entries are all single-enzyme without per-enzyme `yield`, so both tests' invariants are trivially satisfied.

- [ ] **Step 6.3: Commit**

```bash
git add tests/regression/test_prodrug_v3_registry_schema.py
git commit -m "test(prodrug): all-or-nothing per-enzyme yield schema gate (B-04)

Schema regression for spec §5.4 on the production registry. Trivially
passes today (6 single-enzyme entries); guards against future
multi-enzyme entries with mixed yield declarations.

Spec: docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md"
```

---

## Task 7: Full test sweep + backward-compat verification

**Goal:** Run the entire prodrug-relevant test surface to confirm no regressions outside the touched paths.

**Files:** (verification only — no changes)

- [ ] **Step 7.1: Run the full prodrug-related test suite**

Run:
```bash
pytest tests/unit/test_prodrug_v2_drug.py \
       tests/unit/test_prodrug_v2_edge.py \
       tests/unit/test_prodrug_v2_flux.py \
       tests/unit/test_prodrug_v2_augment.py \
       tests/unit/test_prodrug_v2_registry.py \
       tests/unit/test_prodrug_v2_resolved_params.py \
       tests/unit/test_prodrug_flux.py \
       tests/unit/test_prodrug_per_enzyme_yield.py \
       tests/regression/test_prodrug_v2_snapshot.py \
       tests/regression/test_prodrug_v2_identity_blind.py \
       tests/regression/test_prodrug_v2_validation_gate.py \
       tests/regression/test_prodrug_v3_enzyme_leak_audit.py \
       tests/regression/test_prodrug_v3_registry_schema.py \
       tests/regression/test_prodrug_registry_seed.py \
       -v
```

Expected: all PASS. The snapshot test in particular MUST be bit-identical to pre-B-04 for the 4 pinned drugs.

- [ ] **Step 7.2: Engine ID-blind invariant sanity (spec §6)**

Run: `pytest tests/regression/test_prodrug_v2_identity_blind.py -v`
Expected: PASS — engine is identity-blind to enzyme tag names, so per-tag splitting cannot have broken this.

- [ ] **Step 7.3: Run the broader unit suite to catch unrelated regressions**

Run: `pytest tests/unit/ -x -q`
Expected: PASS or only pre-existing xfails (no new failures introduced by B-04).

If a non-prodrug test fails, stop and investigate. Most likely a frozen-dataclass / `replace` interaction in `ivive.py:625` or a missed call-site outside `lookup_active_metabolite`.

- [ ] **Step 7.4: Lint**

Run: `ruff check src/sisyphus/core.py src/sisyphus/predict/registry.py src/sisyphus/predict/ivive.py src/sisyphus/graph/builder.py tests/unit/test_prodrug_per_enzyme_yield.py tests/regression/test_prodrug_v3_registry_schema.py`
Expected: no warnings. Fix any style issues inline.

- [ ] **Step 7.5: Holdout AAFE sanity check (optional but recommended)**

Run: `python scripts/run_engine_benchmark.py --quick 2>&1 | tail -20`
Expected: AAFE numbers (Meta 2.751, Engine 4.008, ML 3.012, In-domain 2.837 per CLAUDE.md headline) bit-identical to pre-B-04. None of the 107 holdout drugs declare per-enzyme yields, so this MUST be a no-op.

If the script doesn't have a `--quick` flag, skip this step and rely on the snapshot test (Step 7.1) as proxy.

If headline drifts: stop. Something in the wiring path (ivive `replace` call, builder edge counts) is leaking into the snapshot.

No commit for this task — it's pure verification.

---

## Task 8: Update backlog + experiment log

**Goal:** Mark B-04 as shipped in `docs/claude/backlog.md` and append a 2026-05-18 entry to `docs/claude/experiment-log.md`.

**Files:**
- Modify: `docs/claude/backlog.md` (remove the B-04 entry, since it ships in this PR per the "delete after promotion" workflow at the top of that file)
- Modify: `docs/claude/experiment-log.md` (prepend entry at top)
- Modify: `docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md` (status header)

- [ ] **Step 8.1: Remove B-04 from backlog**

In `docs/claude/backlog.md`, delete the `### B-04 — Multi-enzyme prodrug conversion schema (per-enzyme yield)` block (the entire section between the heading and the next `### B-03` heading).

Also update B-03's "Blocked by: B-04" line:

```markdown
**Blocked by**: B-04.
```
becomes:
```markdown
**Blocked by**: ~~B-04~~ (shipped 2026-05-18). Ready to implement.
```

- [ ] **Step 8.2: Prepend experiment-log entry**

In `docs/claude/experiment-log.md`, at the very top of the chronological log (under whatever header is there), insert:

```markdown
## 2026-05-18 — B-04 multi-enzyme prodrug yield schema (no headline impact)

**Commits**: (this PR's range)
**Outcome**: schema-only change; 107-holdout AAFE bit-identical pre/post.
**What shipped**:
- `ActiveMetabolite.enzyme_yields: dict[str, Distribution]` (default empty).
- Registry loader (`predict/registry.py`) parses optional per-enzyme
  `yield` on each `enzyme_affinity_for_conversion[<tag>]` block;
  multi-enzyme entries must declare `yield` for every enzyme or none
  (all-or-nothing rule, spec §5.4).
- Builder (`graph/builder.py`) emits one `ProdrugActivationEdge` per
  (site × tag) intersection instead of one per site with collapsed tags;
  each edge reads `am.enzyme_yields.get(tag, am.conversion_yield_fraction)`.
- New unit test file `tests/unit/test_prodrug_per_enzyme_yield.py`.
- New schema regression `tests/regression/test_prodrug_v3_registry_schema.py`.

**Why this matters**: unblocks B-03 (clopidogrel). Clopidogrel's hepatic
fate splits into CES1 → SR26334 (~85% inactive dead-end) and CYP2C19 →
R-130964 (~15% active). A single entry-level yield cannot represent
this without violating mass balance, species identity, or the
mechanistic-A doctrine (see §3 of the spec). Per-enzyme yield resolves
the structural blocker.

**Backward compat**: 6 existing single-enzyme entries unchanged.
Snapshot test (sepiapterin/remdesivir/tebipenem/fostamatinib) and the
production 107-holdout headline are bit-identical pre/post this PR.

**Next**: B-03 implementation (clopidogrel registry entry + 107-holdout
regen with documented AAFE delta).
```

- [ ] **Step 8.3: Mark spec as shipped**

In `docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md`, add a status line just under the title row (after `**Predecessor**:`):

```markdown
**Status**: Implemented 2026-05-18 (commits in this PR; see experiment-log).
```

- [ ] **Step 8.4: Commit**

```bash
git add docs/claude/backlog.md docs/claude/experiment-log.md \
    docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md
git commit -m "docs(b-04): mark shipped + unblock B-03 in backlog

- backlog.md: remove B-04 entry, update B-03 blocker line.
- experiment-log.md: 2026-05-18 entry.
- spec: status line."
```

---

## Task 9: Push and verify CI

**Goal:** Push the branch, wait for the "test" GitHub Actions check, and report the green status.

- [ ] **Step 9.1: Push**

Run: `git push origin main` (if working directly on main per session pattern; otherwise push the feature branch).

If main is protected, create a PR per `gh pr create` flow.

- [ ] **Step 9.2: Wait for CI**

Run: `gh run watch` or `gh pr checks` to wait for the "test" check.

- [ ] **Step 9.3: Verify green**

Run: `gh run list --limit 1`
Expected: latest run status `completed` / conclusion `success`.

If red: investigate, fix forward (NEW commit, not `--amend`), repeat.

---

## Self-Review (run before declaring plan complete)

**1. Spec coverage** (cross-reference spec sections against tasks):
- §3 structural blocker rationale → covered in motivation; no task needed (analysis only)
- §4.1 recommended approach (optional per-enzyme yield, entry-level fallback) → Tasks 1, 2, 3, 4, 5
- §5.1 registry diff sketch → Tasks 3, 6 (loader + schema gate)
- §5.2 builder diff sketch → Task 5
- §5.3 ActiveMetabolite dataclass diff → Tasks 1, 2
- §5.4 all-or-nothing validation rule → Tasks 3 (loader-level), 6 (production registry-level)
- §6 backward compatibility → Tasks 4, 5, 7 (snapshot bit-identical verification)
- §7.1 unit tests → Tasks 1, 2, 3, 5 (all four §7.1 named tests covered, plus additional defensive ones)
- §7.2 schema regression update → Task 6
- §7.3 snapshot regression (no changes if backward-compat holds) → Task 7 (verification)
- §8 B-03 application → not in scope (future spec)
- §9 risks → noted in plan motivation; no task action needed
- §10 out of scope → respected
- §12 acceptance criteria → Task 7 covers all four (snapshot bit-identical, 107-holdout bit-identical, new unit tests pass, schema gate catches partial declaration via Task 6 + Task 3)

No spec gaps.

**2. Placeholder scan:** searched for "TBD", "TODO", "implement later", "fill in details", "add appropriate", "similar to Task". None present.

**3. Type / signature consistency:**
- `ActiveMetabolite.enzyme_yields: dict[str, Distribution]` — used identically in Tasks 1, 2, 4, 5.
- `lookup_active_metabolite` return type `tuple[ActiveMetabolite, str, dict[str, Distribution], dict[str, Distribution]] | None` — declared in Task 3, unpacked in Task 4. Matches.
- `_build_enzyme_affinity_for_conversion` return type changes from `dict[str, Distribution]` to `tuple[dict, dict]` — Task 3 updates both signature and caller in the same task. Consistent.
- Builder loop variable: `tag` (singular) used in Task 5 lookup `am.enzyme_yields.get(tag, ...)`. Matches `enzyme_tags` (plural) being a frozenset on the edge with one element. Consistent.

No inconsistencies. Plan ready.
