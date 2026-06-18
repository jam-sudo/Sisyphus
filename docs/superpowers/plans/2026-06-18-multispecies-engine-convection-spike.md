# Multi-species Engine-Convection Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer yes/no whether the axial machinery (PR #79) composes with the engine's existing second-species (active-metabolite) capability — i.e. whether a metabolite can be spatially resolved across an axial chain (per-zone formation + inter-zone convection + per-zone consumption) using graph construction with EXISTING flux types only, conserving mass — and map (via a Damköhler sweep) when downstream convection materially shifts the metabolite profile vs the local-only post-processor.

**Architecture:** A harness-isolated probe script builds a synthetic two-species axial graph (parent chain + metabolite chain + shared sink) directly via the graph API and solves it, mirroring the production prodrug/active-metabolite construction. It is **run-then-pin** (a spike): build + run to determine the YES path (solves) vs the STOP path (needs an engine change), *then* write tests pinning the actual outcome. Zero `src/sisyphus/engine/` change — a needed engine change is itself the (valid) verdict, never a code edit.

**Tech Stack:** Python 3.10+, numpy, the Sisyphus engine reached only through the public graph API (`build_from_yaml`, `BodyGraph`, edge/node dataclasses, `ODECompiler`/`solve`), pytest, ruff (line-length 100). Run under `/opt/miniconda3/bin/python`.

**Operating constraints (every task):**
- Work on branch `feat/multispecies-convection-spike` (created; spec + this plan already committed there).
- Commit as `jam-sudo <jam-sudo@users.noreply.github.com>`, NO `Co-Authored-By: Claude` / AI trailer, always `git commit --no-verify`.
- Stage ONLY the files named per task via explicit `git add`. NEVER `git add -A`/`.`/`README.md`/the untracked workspace docs.
- Run scripts/tests with `/opt/miniconda3/bin/python`. `ruff check src tests scripts` (line-length 100) before each commit touching them.
- **Hard isolation:** NO edit under `src/sisyphus/engine/`, `src/sisyphus/predict/`, `src/sisyphus/pipeline/`, the builder (`src/sisyphus/graph/builder.py`/`body.py` validation), `data/physiology/`, or the holdout list. If the construction *requires* such a change to build/compile/solve → that is the **STOP verdict**; document it, do not make the change.

**Verified engine facts (do not re-derive; cite in code comments):**
- `ProdrugActivationEdge(source, target, enzyme_tags: frozenset, conversion_yield: Distribution=1.0, mw_parent: float, mw_active: float)` — flux removes parent at the source node's well-stirred intrinsic-clearance rate (using the drug's `enzyme_affinity_for_conversion[tag]` × source `enzymes[tag]`) and adds `× mw_active/mw_parent × yield` to the target (`flux.py:775`, `types.py:193`).
- `TransitEdge(source, target, transit_rate: Distribution)` — `rate = transit_rate × A_source`, **adds to target** (genuine convection; `flux.py:410`, `types.py:108`).
- `OneCompartmentEliminationEdge(source, target, cl_per_h: Distribution, vd_l: Distribution)` — `rate = (cl/vd) × A_source`, mass-conserving to the sink target (`flux.py:896`, `types.py:214`).
- `DrugOnGraph.enzyme_affinity_for_conversion: dict[str, Distribution]` is **required non-empty with a non-None `active_metabolite`** (validation `core.py:292`); `ActiveMetabolite(name: str, mw: float, conversion_yield_fraction: Distribution, …)` (`core.py:145`).
- `Node(node_type: str, volume: Distribution, enzymes: dict, lookup_name: str="")` (`types.py:26`).
- Solve pattern (from `scripts/validate_pgx_cmax_v2b.py`): `rg, rd = graph.realize_means(), drug.realize_means(); compiled = ODECompiler().compile(rg); params = ResolvedParams(rg, rd); y0 = np.zeros(compiled.n_states); y0[compiled.state_index[drug.administration_node]] = drug.dose_mg; res = solve(compiled, params, y0, t_span=(0, T), t_eval=...)`. `res.concentrations[name]`, `res.amounts[name]`, `res.mass_balance_error`.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/spike_multispecies_convection.py` (new) | Build the synthetic 2-species axial graph; solve across the Da sweep + no-convection control + uniform-formation case; compute `mass_balance_error`, species-level balance, center-of-mass shift, the Da map; write the verdict report. |
| `tests/integration/test_multispecies_convection_spike.py` (new) | Pin the ACTUAL outcome — YES-path gates OR the STOP-path failure assertion — + headline-isolation guard. |
| `data/validation/multispecies_convection_spike_2026-06-18.{json,md}` (new, generated) | YES/NO verdict, T1–T5 findings, distinct-per-species sub-finding, Damköhler decision map. |
| `docs/claude/experiment-log.md` (modify) | Dated entry. |
| `docs/claude/dead-ends.md` (modify, only if STOP / honest-negative) | DE entry recording the exact flux/builder limitation. |

---

## Task 0: Recipe discovery + 1-zone smoke (the feasibility crux)

**Goal:** establish the exact construction recipe by mirroring the production prodrug/active-metabolite path, and probe the core unknown — *can metabolite nodes beyond the single named `active_metabolite` coexist and solve?* — with the SMALLEST possible graph, before investing in the full sweep.

**Files:** scratch only this task (no commit yet) — the output is documented findings that Task 1 consumes.

- [ ] **Step 1: Study the production prodrug construction.**

Read `tests/unit/test_prodrug_v2_edge.py` and `tests/unit/test_engine_contracts.py` (and if needed `src/sisyphus/graph/body.py:178` `ProdrugActivationEdge` resolution). Document, in a scratch note, the exact recipe: how a graph gets a metabolite/active node, how `ProdrugActivationEdge` is added, how the `DrugOnGraph` declares `active_metabolite` + `enzyme_affinity_for_conversion`, and how the active species is read from the solve.

- [ ] **Step 2: Build the 1-zone smoke graph and run it.**

In a scratch script, build the smallest construction: take `build_from_yaml(_YAML)` (reuse the harness `_YAML`/`_drug` pattern from `scripts/validate_pgx_cmax_v2b.py` via importlib), add ONE `metabolite_m1` node (`Node(node_type="organ", volume=Distribution(1.0,0.0), enzymes={})`) and ONE `metabolite_sink` node, add a `ProdrugActivationEdge(source="liver", target="metabolite_m1", enzyme_tags=frozenset({gene_tag}), mw_parent=300.0, mw_active=300.0, conversion_yield=Distribution(1.0,0.0))`, a `TransitEdge(source="metabolite_m1", target="metabolite_sink", transit_rate=Distribution(1.0,0.0))`, and a `OneCompartmentEliminationEdge(source="metabolite_m1", target="metabolite_sink", cl_per_h=Distribution(1.0,0.0), vd_l=Distribution(1.0,0.0))`. Build a `_drug(...)` and set `enzyme_affinity_for_conversion={gene_tag: Distribution(<small CLint>,0.0)}` and `active_metabolite=ActiveMetabolite(name="metabolite_m1", mw=300.0, conversion_yield_fraction=Distribution(1.0,0.0))`. Solve with the standard pattern.

Run it: `/opt/miniconda3/bin/python <scratch>.py`.

- [ ] **Step 3: Classify the outcome.**

- **If it compiles + solves** and `res.amounts["metabolite_m1"]` / `res.amounts["metabolite_sink"]` are populated and `res.mass_balance_error` is small → **recipe confirmed; proceed to Task 1.** Note whether the metabolite node name must equal `active_metabolite.name` (it likely must — this constrains the axial case: each of the N metabolite nodes vs a single `active_metabolite`).
- **If it errors** (validation rejects the extra node; the compiler ties the metabolite to a single `active_metabolite.name` and cannot host N nodes; the prodrug target must be the named active node only; etc.) → **this is (or foreshadows) the STOP verdict.** Capture the exact error + the offending engine assumption. Try the minimal viable variant (e.g. if only the single named active node is allowed, that directly answers the axial-composition question = STOP "needs engine support for multiple metabolite nodes"). Proceed to Task 1 to formalize and pin the STOP finding.

**No commit in Task 0** — the recipe/finding feeds Task 1, which produces the committed script.

---

## Task 1: The spike script (build, run, classify YES vs STOP)

**Files:**
- Create: `scripts/spike_multispecies_convection.py`

- [ ] **Step 1: Write the script** using the Task-0 recipe. Structure (fill node/active-metabolite construction from the confirmed recipe; the edge constructors, Da math, and metrics below are exact):

```python
"""Multi-species engine-convection feasibility spike (Bridge B / B1.x).

Run-then-pin SPIKE. Builds a synthetic TWO-species axial graph (parent__ax1..N +
metabolite__ax1..N + shared metabolite_sink) via the graph API and asks whether the axial
machinery (PR #79) composes with the engine's existing active-metabolite species: can a
metabolite be SPATIALLY RESOLVED (per-zone formation) and CONVECTED inter-zone, conserving
mass, with ZERO src/engine change? Maps (Damkohler sweep) when convection shifts the profile
vs the local-only post-processor. If the construction needs an engine/builder change, that
is the STOP verdict (reported, not coded). Harness-isolated; headline 2.731 untouched.

Chain-level Damkohler Da = N*k_detox/k_conv (crossover ~1; Da>>1 local/reactive, Da<<1
convected/stable). Verified flux facts: TransitFluxSpec adds-to-target with edge transit_rate
(flux.py:410); OneCompartmentEliminationFluxSpec rate=(cl/vd)*A to sink (flux.py:896);
ProdrugActivationFluxSpec parent->active mw_active/mw_parent*yield at source enzymes
(flux.py:775).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from sisyphus.core import ActiveMetabolite, Distribution
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.types import (
    Node, OneCompartmentEliminationEdge, ProdrugActivationEdge, TransitEdge,
)
from sisyphus.validation.pgx_metrics import zonation_weights

_ROOT = Path(__file__).resolve().parent.parent
_HARNESS = _ROOT / "scripts" / "validate_pgx_cmax_v2b.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("v2b_harness", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load_harness()  # _axial_graph, _drug, _SYNTHETIC_GENE_ABUND, _T_EVAL, _YAML, ...

_GENE = "CYP3A4"
_N = 10
_MW = 300.0
_Q_OVER_V_ZONE = None  # set from the axial graph's per-zone volume + flow (Task-0 recipe)


def _build_two_species_axial(n, k_conv, k_detox, bio_ratio, bio_direction, uniform_formation):
    """Construct the parent axial chain + a metabolite axial chain + shared sink, per the
    Task-0 recipe. Returns (graph, drug). REUSE the confirmed node/active_metabolite
    construction. Edge wiring (exact):
      - formation:  ProdrugActivationEdge(parent__ax{i} -> metabolite__ax{i},
                    enzyme_tags={_GENE}, mw_parent=_MW, mw_active=_MW, conversion_yield=1)
                    with the gene zonated by zonation_weights(n, bio_ratio, bio_direction)
                    (or uniform if uniform_formation).
      - convection: TransitEdge(metabolite__ax{i} -> metabolite__ax{i+1}, transit_rate=k_conv)
      - consumption:OneCompartmentEliminationEdge(metabolite__ax{i} -> metabolite_sink,
                    cl_per_h=k_detox, vd_l=1.0)   # k_detox = cl/vd with vd=1
      - efflux:     TransitEdge(metabolite__axN -> metabolite_sink, transit_rate=k_conv)
    The drug sets enzyme_affinity_for_conversion={_GENE: <CLint>} and
    active_metabolite=ActiveMetabolite(name=<per Task-0 naming>, mw=_MW, ...).
    """
    raise NotImplementedError("fill from the Task-0 confirmed recipe")


def _solve(graph, drug):
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, float(h._T_EVAL[-1])), t_eval=h._T_EVAL)
    return res, compiled


def _metabolite_zone_amounts(res, n):
    """Time-integrated amount per metabolite zone, inlet->outlet."""
    trapz = getattr(np, "trapezoid", np.trapz)
    t = np.asarray(res.time_h)
    return [float(trapz(np.asarray(res.amounts[f"metabolite__ax{i+1}"]), t)) for i in range(n)]


def _center_of_mass(zone_amounts):
    w = np.asarray(zone_amounts, dtype=float)
    if w.sum() <= 0:
        return float("nan")
    z = np.arange(1, len(w) + 1)
    return float((z * w).sum() / w.sum())


def _species_balance(res, compiled, n):
    """Formed (= sink + chain residual at t_end, since formation is mass-preserving):
    returns (chain_end + sink_end) and the engine mass_balance_error."""
    chain_end = sum(float(np.asarray(res.amounts[f"metabolite__ax{i+1}"])[-1]) for i in range(n))
    sink_end = float(np.asarray(res.amounts["metabolite_sink"])[-1])
    return chain_end, sink_end, float(res.mass_balance_error)
```

Then a `run()` that: solves the convected case and the no-convection control (`k_conv→0` on inter-zone AND efflux) across a `Da` grid (chain `Da = n·k_detox/k_conv`; e.g. `Da ∈ {0.03, 0.1, 0.3, 1, 3, 10, 30}` by varying `k_detox` at fixed `k_conv`), plus a uniform-formation case; records per-Da center-of-mass shift `⟨z⟩_convected − ⟨z⟩_control`, `mass_balance_error`, and the species balance; and a `main()` that writes the report. Pin `k_conv` from the axial per-zone volume/flow (Task-0). **If `_build_two_species_axial` raised at Task 0 (STOP path), `run()` instead captures the build/compile error string as the verdict.**

- [ ] **Step 2: Run the script.**

Run: `/opt/miniconda3/bin/python scripts/spike_multispecies_convection.py`
- **YES path:** prints the Da map + balances; writes the report; `mass_balance_error` small; species balance closes; center-of-mass shift decreases monotonically as `Da` grows (large at low Da, ~0 at high Da).
- **STOP path:** prints the captured engine limitation; writes a report whose verdict is "needs engine support: <exact reason>". Do NOT edit `src/engine/` to force it through.

- [ ] **Step 3: Lint + commit (script only).**

```bash
/opt/miniconda3/bin/python -m ruff check scripts/spike_multispecies_convection.py
git add scripts/spike_multispecies_convection.py
git commit --no-verify -m "feat(validation): multi-species engine-convection spike (B1.x, run-then-pin)"
```

---

## Task 2: Pin the actual outcome with tests

**Files:**
- Create: `tests/integration/test_multispecies_convection_spike.py`

Write the test to match the **observed** Task-1 outcome (stack-independent assertions — signs/inequalities/tolerances, never pinned floats).

- [ ] **Step 1 (YES path): write the gate tests.**

```python
"""Gate tests for the multi-species engine-convection spike (B1.x). Pins the OBSERVED
outcome. Stack-independent assertions only."""
import importlib.util
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "ms_spike", _ROOT / "scripts" / "spike_multispecies_convection.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


spike = _load()
_R = spike.run()  # compute once (engine solves are slow); all tests read from this


def test_builds_and_solves():
    assert _R["verdict"] == "YES"            # composition succeeded (else see STOP test)


def test_mass_conserved():
    assert _R["mass_balance_error"] < 1e-3   # engine global invariant
    formed, in_system = _R["formed"], _R["chain_end"] + _R["sink_end"]
    assert abs(formed - in_system) / max(formed, 1e-30) < 1e-2   # species-level balance


def test_low_da_shows_downstream_shift():
    assert _R["shift_low_da"] > 0.05         # convected center-of-mass moves toward outlet


def test_high_da_agrees_with_local():
    assert abs(_R["shift_high_da"]) < _R["shift_low_da"]   # reactive regime ~ local-only


def test_da_map_monotone():
    shifts = _R["da_shift_curve"]            # ordered by increasing Da
    assert all(shifts[i] >= shifts[i + 1] - 1e-6 for i in range(len(shifts) - 1))


def test_headline_isolation_unchanged():
    p = _ROOT / "data" / "training" / "4track_holdout_predictions.json"
    d = json.loads(p.read_text())
    assert abs(d["overall"]["meta"]["aafe"] - 2.731) < 5e-3
```

- [ ] **Step 1 (STOP path, INSTEAD): pin the negative.** If Task 1 hit STOP, replace the gate tests with a single test asserting the documented failure mode, e.g.:

```python
def test_stop_verdict_documented():
    r = spike.run()
    assert r["verdict"] == "STOP"
    assert r["reason"]                       # non-empty engine-limitation string
```
plus `test_headline_isolation_unchanged` (verbatim above).

- [ ] **Step 2: Run + lint.**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_multispecies_convection_spike.py -q` → all pass.
Run: `/opt/miniconda3/bin/python -m ruff check tests/integration/test_multispecies_convection_spike.py` → clean.

- [ ] **Step 3: Commit.**

```bash
git add tests/integration/test_multispecies_convection_spike.py
git commit --no-verify -m "test(validation): pin multi-species convection spike outcome (B1.x)"
```

---

## Task 3: Report + docs

**Files:**
- Create (generated): `data/validation/multispecies_convection_spike_2026-06-18.{json,md}`
- Modify: `docs/claude/experiment-log.md`
- Modify (only if STOP / honest-negative): `docs/claude/dead-ends.md`

- [ ] **Step 1: Regenerate the report** from the committed script: `/opt/miniconda3/bin/python scripts/spike_multispecies_convection.py`. Confirm the `.md` states the verdict (YES with the Da map + balances, or STOP with the exact engine limitation), the T1–T5 findings, and the distinct-per-species-properties sub-finding.

- [ ] **Step 2: Prepend the experiment-log entry** (use the actual measured numbers/verdict) under the `---` at the top of `docs/claude/experiment-log.md`, dated `2026-06-18`, headed `Bridge B / B1.x: multi-species engine-convection spike (<YES|STOP>)`, noting: production already ships well-mixed parent→active-metabolite (so the test was axial *composition*); the verdict; if YES the Da crossover + that high-Da validates the post-processor; harness-isolated, headline 2.731 bit-identical. Bump `last_updated` to `2026-06-18` if not already.

- [ ] **Step 3 (only if STOP): add a dead-ends entry** `DE-52` (confirm the next id with `grep -oE "DE-[0-9]+" docs/claude/dead-ends.md | sort -t- -k2 -n | tail -1`) recording the exact flux/builder/compiler limitation that blocks a spatially-resolved metabolite, so a future engine-side project is scoped. If YES, SKIP (not a dead end).

- [ ] **Step 4: Verify headline bit-identity + commit.**

```bash
/opt/miniconda3/bin/python -m pytest -k "cached_holdout_aafe_is_2p731 or mm_headline_bit_identity" -q   # must pass
git add data/validation/multispecies_convection_spike_2026-06-18.json data/validation/multispecies_convection_spike_2026-06-18.md docs/claude/experiment-log.md
# add docs/claude/dead-ends.md ONLY if Step 3 created an entry
git commit --no-verify -m "validation(bridge-b): multi-species convection spike verdict + experiment-log (B1.x)"
```

---

## Self-Review (completed)

**Spec coverage:** §1 question + Da map → Task 1 `run()` + Task 3 report. §2 T1–T5 → Task 0 smoke (T1–T4) + Task 1 solve + Task 2 `test_mass_conserved` (T5). §2 edge-defined-pools + distinct-properties sub-finding → Task 0 naming note + Task 3 report. §3 construction (exact edges, mass-preserving formation, shared sink, k_conv=Q/V) → Task 1 `_build_two_species_axial` + verified-facts block. §3 chain-Da → Task 1 Da grid. §4 gates (G-build/solve, G-mass dual check, G-convection-correct center-of-mass + control, Da-map monotonicity, crossover reported) → Task 2 tests. §4 STOP path + honest-negative → Task 0/1 classify + Task 2 STOP test + Task 3 DE. §5 run-then-pin → Task ordering (0 smoke → 1 run → 2 pin). §6/§7 isolation + headline → operating-constraints + `test_headline_isolation_unchanged`.

**Placeholder scan:** the one `NotImplementedError` in `_build_two_species_axial` is deliberate and explicitly delegated to the Task-0 confirmed recipe (the node/active-metabolite construction cannot be pre-written without the live recipe — the essence of a spike). Every other code block is complete. No `TBD`/`add error handling`/uncoded steps.

**Type consistency:** edge constructors (`ProdrugActivationEdge`/`TransitEdge`/`OneCompartmentEliminationEdge`) and `ActiveMetabolite`/`Node` signatures match the verified-facts block. `run()` return keys (`verdict`, `mass_balance_error`, `formed`, `chain_end`, `sink_end`, `shift_low_da`, `shift_high_da`, `da_shift_curve`, `reason`) are consistent between Task 1 and the Task 2 tests.

**Known live-verification point:** the metabolite node ↔ single `active_metabolite.name` constraint (Task 0 Step 3) is the feasibility crux — it directly determines YES vs STOP, and the plan handles both branches.
