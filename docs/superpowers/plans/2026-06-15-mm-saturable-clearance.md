# MM Saturable Metabolic Clearance (v2.2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the engine saturable (Michaelis–Menten) metabolic clearance that reduces exactly
to the current linear flux when no `Km` is supplied.

**Architecture:** A defaulted `enzyme_km` dict on `DrugOnGraph` (carried through `sample()`/
`realize_means()`), two `ResolvedParams` accessors, and a guarded per-enzyme saturation factor
`1/(1+C_u/Km)` in the `well_stirred` branch of `ClearanceFluxSpec.apply`. Production never sets
`enzyme_km`, so it takes the verbatim linear path → headline 2.731 bit-identical.

**Tech Stack:** Python 3.10, numpy, scipy, the Sisyphus engine. **Spec:**
`docs/superpowers/specs/2026-06-15-mm-saturable-clearance-design.md`.

**Conventions (every task):** commit as `jam-sudo`, **no** Claude trailer, `git commit
--no-verify`; tests via `python -m pytest`; ruff line-length 100 (`ruff check src tests` must
pass — CI runs it repo-wide). Engine+scipy at `/opt/miniconda3/bin/python`. Stage only the
files each task names; never `git add README.md` or untracked workspace files.

---

## File Structure

- `src/sisyphus/core.py` — **modify** `DrugOnGraph`: add `enzyme_km` field, carry it in
  `sample()` + `realize_means()`, validate in `__post_init__`.
- `src/sisyphus/engine/compiler.py` — **modify** `ResolvedParams`: add `drug_enzyme_km`,
  `drug_has_enzyme_km`.
- `src/sisyphus/engine/flux.py` — **modify** `ClearanceFluxSpec.apply`, `well_stirred` branch:
  guarded linear/saturable fork.
- `tests/unit/test_enzyme_km_contract.py` — **new** (Task 1).
- `tests/unit/test_mm_saturable_flux.py` — **new** (Task 2).
- `tests/regression/test_mm_headline_bit_identity.py` — **new** (Task 3).
- `docs/claude/experiment-log.md` — **append** (Task 3).

---

## Task 1: contract — `enzyme_km` field + realization + accessors

**Files:**
- Modify: `src/sisyphus/core.py` (DrugOnGraph: field, `__post_init__`, `sample`, `realize_means`)
- Modify: `src/sisyphus/engine/compiler.py` (ResolvedParams accessors)
- Test: `tests/unit/test_enzyme_km_contract.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_enzyme_km_contract.py
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.graph.builder import build_from_yaml


def _drug(enzyme_km=None):
    return DrugOnGraph(
        name="t", smiles="C", dose_mg=100.0, route="oral",
        administration_node="stomach_lumen", mw=300.0, pka=None, compound_type="neutral",
        fup=Distribution(0.1, 0.0), rbp=Distribution(1.0, 0.0),
        kp_method="rodgers_rowland", kp_overrides={},
        peff=Distribution(5.0, 0.0), solubility=Distribution(1000.0, 0.0),
        enzyme_affinity={"CYP2D6": Distribution(0.5, 0.0)},
        renal_clearance=Distribution(0.0, 0.0),
        enzyme_km=enzyme_km or {},
    )


def test_enzyme_km_defaults_empty():
    assert _drug().enzyme_km == {}


def test_enzyme_km_rejects_nonpositive():
    with pytest.raises(ValueError):
        _drug({"CYP2D6": Distribution(0.0, 0.0)})


def test_realize_means_preserves_enzyme_km():
    d = _drug({"CYP2D6": Distribution(2.5, 0.0)}).realize_means()
    assert d.enzyme_km["CYP2D6"].mean == pytest.approx(2.5)


def test_sample_preserves_enzyme_km():
    d = _drug({"CYP2D6": Distribution(2.5, 0.0)}).sample(np.random.default_rng(0))
    assert d.enzyme_km["CYP2D6"].mean == pytest.approx(2.5)


def test_accessors_present_and_absent():
    g = build_from_yaml(Path("data/physiology/reference_man.yaml")).realize_means()
    rp = ResolvedParams(g, _drug({"CYP2D6": Distribution(2.5, 0.0)}).realize_means())
    assert rp.drug_enzyme_km("CYP2D6") == pytest.approx(2.5)
    assert math.isinf(rp.drug_enzyme_km("ABSENT"))
    assert rp.drug_has_enzyme_km() is True
    rp_empty = ResolvedParams(g, _drug().realize_means())
    assert rp_empty.drug_has_enzyme_km() is False
    assert math.isinf(rp_empty.drug_enzyme_km("CYP2D6"))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_enzyme_km_contract.py -v`
Expected: FAIL (`DrugOnGraph.__init__` got an unexpected keyword 'enzyme_km').

- [ ] **Step 3: Add the field + validation** in `src/sisyphus/core.py`, immediately after the
  `enzyme_affinity_for_conversion` field (currently the last field, ~line 275) and before
  `__post_init__`:

```python
    # v2.2a saturable metabolism — per-enzyme Michaelis Km on the UNBOUND-conc basis (mg/L).
    # Empty = linear (current behaviour). Vmax_i = abundance_i * affinity_i * Km_i emerges
    # implicitly; the well_stirred flux multiplies enzyme i's CLint by 1/(1 + C_u/Km_i).
    enzyme_km: dict[str, Distribution] = field(default_factory=dict)
```

  Then add to `__post_init__` (after the existing checks):

```python
        for _tag, _dist in self.enzyme_km.items():
            if _dist.mean <= 0:
                raise ValueError(f"enzyme_km[{_tag!r}] mean must be > 0, got {_dist.mean}")
```

- [ ] **Step 4: Carry `enzyme_km` through realization.** In `DrugOnGraph.sample()`, add inside
  the `return DrugOnGraph(...)` call (e.g. right after the `enzyme_affinity_for_conversion=`
  block):

```python
            enzyme_km={
                k: Distribution(mean=v.sample(rng), cv=0.0) for k, v in self.enzyme_km.items()
            },
```

  And the analogous block in `realize_means()`:

```python
            enzyme_km={
                k: Distribution(mean=v.mean, cv=0.0) for k, v in self.enzyme_km.items()
            },
```

- [ ] **Step 5: Add the accessors** in `src/sisyphus/engine/compiler.py`, after
  `drug_enzyme_affinity_for_conversion` (~line 141):

```python
    def drug_enzyme_km(self, tag: str) -> float:
        """Return the drug's Michaelis Km (unbound mg/L) for *tag*, or +inf when absent.

        +inf collapses the saturation factor 1/(1 + C_u/Km) to 1 (linear limit).
        """
        if tag in self._drug.enzyme_km:
            return self._drug.enzyme_km[tag].mean
        return float("inf")

    def drug_has_enzyme_km(self) -> bool:
        """True if the drug carries any saturable-metabolism Km (selects the MM flux path)."""
        return bool(self._drug.enzyme_km)
```

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/unit/test_enzyme_km_contract.py -v`
Expected: PASS (5 tests). Then `ruff check src tests` → clean.

- [ ] **Step 7: Commit**

```bash
git add src/sisyphus/core.py src/sisyphus/engine/compiler.py tests/unit/test_enzyme_km_contract.py
git commit --no-verify -m "feat(engine): enzyme_km contract for saturable metabolism (v2.2a)"
```

---

## Task 2: saturable `well_stirred` flux (guarded fork) + oracle/behaviour tests

**Files:**
- Modify: `src/sisyphus/engine/flux.py` (`ClearanceFluxSpec.apply`, `well_stirred` branch)
- Test: `tests/unit/test_mm_saturable_flux.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mm_saturable_flux.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401  -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml

_YAML = Path("data/physiology/reference_man.yaml")
_RESID = "RESIDUAL_HEPATIC"


def _ws_graph():
    """Reference graph with the liver clearance edge forced to well_stirred (edges frozen)."""
    g = build_from_yaml(_YAML)
    g.edges[:] = [
        replace(e, model="well_stirred")
        if getattr(e, "source", None) == "liver" and getattr(e, "model", None) == "extended"
        else e
        for e in g.edges
    ]
    g.nodes["liver"].enzymes["CYP2D6"] = Distribution(1.0e6, 0.0)
    return g


def _drug(dose, enzyme_km=None):
    return DrugOnGraph(
        name="syn", smiles="C", dose_mg=dose, route="intravenous",
        administration_node="venous_blood", mw=300.0, pka=None, compound_type="neutral",
        fup=Distribution(0.3, 0.0), rbp=Distribution(1.0, 0.0),
        kp_method="provided",
        kp_overrides={t: Distribution(3.0, 0.0) for t in
                      ["adipose", "bone", "brain", "gut", "heart", "kidney", "liver",
                       "lung", "muscle", "skin", "spleen"]},
        peff=Distribution(5.0, 0.0), solubility=Distribution(1000.0, 0.0),
        enzyme_affinity={"CYP2D6": Distribution(2.0e-3, 0.0)},
        renal_clearance=Distribution(0.0, 0.0),
        enzyme_km=enzyme_km or {},
    )


def _auc(graph, drug, t_end=400.0):
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, t_end))
    conc, t = res.concentrations["venous_blood"], res.time_h
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(conc, t))


def test_mm_rate_oracle_matches_formula():
    # The flux's elimination rate equals fup * sum(ab*af*ivive/(1+C_u/Km)) * c_plasma.
    g = _ws_graph().realize_means()
    drug = _drug(100.0, {"CYP2D6": Distribution(0.05, 0.0)}).realize_means()
    params = ResolvedParams(g, drug)
    compiled = ODECompiler().compile(g)
    # locate the liver well_stirred clearance flux
    spec = next(s for s in compiled.flux_specs
                if getattr(s, "model", None) == "well_stirred"
                and getattr(s, "source_name", None) == "liver")
    y = np.zeros(compiled.n_states)
    y[spec.source_idx] = 50.0  # mg in liver
    dydt = np.zeros(compiled.n_states)
    spec.apply(0.0, y, dydt, params)
    # expected, computed from the same params accessors (checks the formula, not magnitudes)
    fup = params.drug_param("fup")
    if params.node_param("liver", "fu_correction_applicable") > 0:
        fup = fup * params.drug_param("fu_correction_liver")
    v = params.node_param("liver", "volume")
    kp = params.drug_kp("liver")
    c_plasma = y[spec.source_idx] / (v * kp)
    c_u = fup * c_plasma
    ivive = params.node_param("liver", "ivive_scaling")
    clint = 0.0
    for tag, ab in params.node_enzymes("liver").items():
        af = params.drug_enzyme_affinity(tag)
        if af > 0 and ab > 0:
            clint += ab * af * ivive / (1.0 + c_u / params.drug_enzyme_km(tag))
    expected = -(fup * clint) * c_plasma  # loss from the source compartment
    assert dydt[spec.source_idx] == pytest.approx(expected, rel=1e-9)


def test_linear_limit_matches_no_km():
    g = _ws_graph()
    auc_linear = _auc(g, _drug(100.0))
    auc_huge_km = _auc(g, _drug(100.0, {"CYP2D6": Distribution(1.0e12, 0.0)}))
    assert auc_huge_km == pytest.approx(auc_linear, rel=1e-6)


def test_dose_proportionality_breaks_under_saturation():
    g = _ws_graph()
    km = {"CYP2D6": Distribution(0.05, 0.0)}  # low Km ⇒ saturates at therapeutic conc
    auc_1x = _auc(g, _drug(100.0, km))
    auc_2x = _auc(g, _drug(200.0, km))
    # saturable elimination ⇒ supra-proportional exposure
    assert auc_2x / auc_1x > 2.05
    # linear control is exactly proportional
    lin_1x = _auc(g, _drug(100.0))
    lin_2x = _auc(g, _drug(200.0))
    assert lin_2x / lin_1x == pytest.approx(2.0, rel=1e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_mm_saturable_flux.py -v`
Expected: FAIL — `test_dose_proportionality_breaks` fails (currently linear: ratio ≈ 2.0), and
`test_mm_rate_oracle` fails (no saturation factor yet). (`test_linear_limit` may already pass.)

> If `compiled.flux_specs` is not the attribute name, find the compiled flux list by reading
> `ODECompiler.compile`'s return (`CompiledODE`) — adjust the `next(...)` accessor accordingly.

- [ ] **Step 3: Implement the guarded fork.** Replace the entire `if self.model ==
  "well_stirred":` block in `ClearanceFluxSpec.apply` (`src/sisyphus/engine/flux.py`) with:

```python
        if self.model == "well_stirred":
            if not params.drug_has_enzyme_km():
                # ===== LINEAR PATH — verbatim; production never sets enzyme_km, so this is
                # the only path the headline takes (byte-for-byte identical, FLUX-1/RBP-2). =====
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

                # FLUX-1: apply the *intrinsic* (flow-unlimited) clearance to c_out.
                cl_intrinsic = fup * clint_organ

                # RBP-2: the metabolic sink acts on unbound PLASMA (fup·CLint·C_plasma).
                v = params.node_param(self.source_name, "volume")
                kp = params.drug_kp(self.source_name)
                c_plasma = y[self.source_idx] / (v * kp) if v > 0 else 0.0

                rate = cl_intrinsic * c_plasma
            else:
                # ===== SATURABLE (MICHAELIS–MENTEN) PATH (v2.2a) =====
                # Per-enzyme CLint_i *= 1/(1 + C_u/Km_i); C_u = fup·c_plasma (well-stirred
                # unbound-plasma basis). Km_i = +inf (absent) ⇒ factor 1 (that enzyme stays
                # linear). Reduces to the linear path as every C_u/Km_i → 0.
                fup = params.drug_param("fup")
                if params.node_param(self.source_name, "fu_correction_applicable") > 0:
                    fup = fup * params.drug_param("fu_correction_liver")
                v = params.node_param(self.source_name, "volume")
                kp = params.drug_kp(self.source_name)
                c_plasma = y[self.source_idx] / (v * kp) if v > 0 else 0.0
                c_u = fup * c_plasma

                clint_organ = 0.0
                ivive = params.node_param(self.source_name, "ivive_scaling")
                for tag, abundance in params.node_enzymes(self.source_name).items():
                    affinity = params.drug_enzyme_affinity(tag)
                    if affinity > 0 and abundance > 0:
                        km = params.drug_enzyme_km(tag)
                        clint_organ += abundance * affinity * ivive / (1.0 + c_u / km)

                if clint_organ <= 0:
                    return

                rate = (fup * clint_organ) * c_plasma
```

  Leave the `elif self.model == "gfr_filtration":` / `extended` branches unchanged.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_mm_saturable_flux.py -v`
Expected: PASS (3 tests). Then `ruff check src tests` → clean.

- [ ] **Step 5: Confirm no regression in the existing engine suite**

Run: `python -m pytest tests/unit tests/integration -k "flux or clearance or engine or holdout" -q`
Expected: PASS (existing well_stirred/clearance/identity-blindness tests unaffected — the linear
path is verbatim).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/engine/flux.py tests/unit/test_mm_saturable_flux.py
git commit --no-verify -m "feat(engine): saturable Michaelis-Menten well_stirred clearance (v2.2a)"
```

---

## Task 3: headline bit-identity regression + docs wire-back

**Files:**
- Create: `tests/regression/test_mm_headline_bit_identity.py`
- Modify: `docs/claude/experiment-log.md`

- [ ] **Step 1: Record the baseline (linear-path) value.** Run this on the current code and note
  the printed Cmax — it is the pre-change engine output for an empty-`enzyme_km` drug:

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
import sisyphus.engine.flux  # noqa
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
g = build_from_yaml(Path("data/physiology/reference_man.yaml")).realize_means()
d = DrugOnGraph(name="caf", smiles="C", dose_mg=100.0, route="oral",
    administration_node="stomach_lumen", mw=194.0, pka=None, compound_type="base",
    fup=Distribution(0.65,0), rbp=Distribution(1.0,0), kp_method="rodgers_rowland",
    kp_overrides={}, peff=Distribution(5.0,0), solubility=Distribution(1000.0,0),
    enzyme_affinity={"CYP1A2": Distribution(1.0e-3,0)},
    renal_clearance=Distribution(0.0,0)).realize_means()
c = ODECompiler().compile(g); p = ResolvedParams(g, d)
y0 = np.zeros(c.n_states); y0[c.state_index["stomach_lumen"]] = d.dose_mg
r = solve(c, p, y0, t_span=(0.0, 200.0))
print(repr(float(r.concentrations["venous_blood"].max())))
PY
```

- [ ] **Step 2: Write the bit-identity regression test**, pasting the printed value into
  `_PINNED_CMAX`:

```python
# tests/regression/test_mm_headline_bit_identity.py
"""v2.2a is headline-isolated: a drug with empty enzyme_km takes the verbatim linear flux
path, so its engine output is bit-identical to pre-v2.2a. Pins one drug's Cmax to the value
recorded on the linear-path code (Task 3 Step 1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sisyphus.engine.flux  # noqa: F401
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml

_PINNED_CMAX = 0.0  # <-- paste the Step-1 value (full repr precision)


def test_empty_enzyme_km_is_bit_identical():
    g = build_from_yaml(Path("data/physiology/reference_man.yaml")).realize_means()
    d = DrugOnGraph(
        name="caf", smiles="C", dose_mg=100.0, route="oral",
        administration_node="stomach_lumen", mw=194.0, pka=None, compound_type="base",
        fup=Distribution(0.65, 0), rbp=Distribution(1.0, 0), kp_method="rodgers_rowland",
        kp_overrides={}, peff=Distribution(5.0, 0), solubility=Distribution(1000.0, 0),
        enzyme_affinity={"CYP1A2": Distribution(1.0e-3, 0)},
        renal_clearance=Distribution(0.0, 0),
    ).realize_means()
    assert d.enzyme_km == {}  # empty ⇒ linear path
    c = ODECompiler().compile(g)
    p = ResolvedParams(g, d)
    y0 = np.zeros(c.n_states)
    y0[c.state_index["stomach_lumen"]] = d.dose_mg
    r = solve(c, p, y0, t_span=(0.0, 200.0))
    assert float(r.concentrations["venous_blood"].max()) == _PINNED_CMAX
```

- [ ] **Step 3: Run it + the existing regression suite**

Run: `python -m pytest tests/regression/test_mm_headline_bit_identity.py tests/regression -q`
Expected: PASS — the new pin holds AND the existing holdout/cache pins
(`test_cached_holdout_aafe_is_2p731`, tebipenem, etc.) are unchanged ⇒ headline 2.731 untouched.

- [ ] **Step 4: Append the experiment-log entry** (top, under the header `---`):

```markdown
## 2026-06-15 — Engine capability: Michaelis–Menten saturable metabolic clearance (v2.2a)

Added saturable hepatic metabolism to the `well_stirred` clearance flux: per-enzyme
`CLint_i *= 1/(1 + C_u/Km_i)`, `C_u = fup·c_plasma`, gated on a new defaulted
`DrugOnGraph.enzyme_km` dict. **Headline 2.731 bit-identical** — production never sets
`enzyme_km`, so the flux takes the verbatim linear branch (regression-pinned
`tests/regression/test_mm_headline_bit_identity.py`; existing holdout/cache pins unchanged).
Tests: MM rate oracle (flux RHS = analytic), dose-proportionality-breaks (supra-proportional
AUC under saturation), linear-limit (Km→∞ ⇒ linear). `well_stirred` only; ECM intracellular
saturation, mechanism-based inhibition (omeprazole → v2.3), and any genotype/clinical
validation (→ v2.2b, with its own data-feasibility gate) are out of scope. Spec/plan
2026-06-15. Foundation for v2.2b (nonlinear genotype Cmax/AUC folds).
```

- [ ] **Step 5: Commit**

```bash
git add tests/regression/test_mm_headline_bit_identity.py docs/claude/experiment-log.md
git commit --no-verify -m "test(engine): pin v2.2a headline bit-identity + log MM capability"
```

---

## Final review

Run the full relevant suite: `python -m pytest tests/unit/test_enzyme_km_contract.py
tests/unit/test_mm_saturable_flux.py tests/regression -q` and `ruff check src tests` — all green.
Dispatch a final reviewer over the diff; confirm the two load-bearing invariants: (1) the linear
`well_stirred` block is byte-for-byte unchanged inside the guard, and (2) `enzyme_km` is carried
through BOTH `sample()` and `realize_means()`. Then use
superpowers:finishing-a-development-branch.

---

## Self-review notes

- **Spec coverage:** §2 identities → Task 2 oracle. §3 flux math → Task 2 Step 3. §4 contract +
  realization propagation → Task 1. §5 guard → Task 2 Step 3 (`if not drug_has_enzyme_km`). §6
  tests → Tasks 1–3 (bit-identity, rate oracle, dose-proportionality, linear-limit,
  identity-blindness via the unchanged existing suite, Task 2 Step 5).
- **Type consistency:** `drug_enzyme_km(tag)->float` (+inf absent), `drug_has_enzyme_km()->bool`,
  `enzyme_km: dict[str, Distribution]` — used identically across Tasks 1–2.
- **Headline safety is structural** (verbatim linear branch) AND pinned (Task 3). The one subtle
  failure mode — forgetting to carry `enzyme_km` through `sample()`/`realize_means()` — is caught
  by Task 1's `test_*_preserves_enzyme_km` and would otherwise make Task 2's saturation silently
  inert.
