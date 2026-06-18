# Liver-Zonation Invariance Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Demonstrate on the real engine that hepatic first-pass extraction is **invariant to the axial spatial distribution** of an enzyme (total preserved): `ΔE(N) = E_zonated − E_uniform → 0` as the sub-tank count `N` grows (convergence to the plug-flow continuum). Establishes that zonation is **not** a bulk-PK lever and validates the PR-#79 axial cascade.

**Architecture:** A pure zonation-weight helper + a harness-isolated probe that reuses the merged synthetic-engine helpers (`_axial_graph`, `_engine_e_h`, `_sat_drug`, …) via `importlib`. Zonation is applied to the synthetic axial skeleton via `dataclasses.replace` (total-preserving). No `predict()` / `reference_man.yaml` / `src/engine` / `expand_axial` change; headline 2.731 bit-identical.

**Tech Stack:** Python 3.10+, numpy, pytest. Engine+scipy at `/opt/miniconda3/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-17-liver-zonation-phase0-design.md`

**Constraints (load-bearing):**
- Harness-isolated. No `predict()`/`reference_man.yaml`/holdout/`src/sisyphus/engine/`/`expand_axial` change. Headline **2.731 bit-identical**.
- Commit as `jam-sudo <jam-sudo@users.noreply.github.com>`. NEVER add a `Co-Authored-By: Claude`/AI footer/"Generated with" line. Use `git commit --no-verify`. After each commit run `git show --name-only HEAD` and verify scope.
- Stage ONLY the files each task names. NEVER `git add README.md` or any untracked workspace file (`ChemRxiv*`, `docs/numeric_drift*`, `docs/preprint_v3*`).
- Tests: `/opt/miniconda3/bin/python -m pytest`. Lint: `ruff check` (line-length 100).
- **The deliverable is a NEGATIVE result** (zonation is not a bulk-PK lever) + the convergence validation. Report honestly; do not tune a tolerance to force a "lever".

**Reused helpers (on `main`, in `scripts/validate_pgx_cmax_v2b.py`):** `_SYNTHETIC_GENE_ABUND=1e6`, `_well_stirred_graph`, `_drug`, `_sat_drug`, `_cmax_auc_tmax`, `_axial_graph(gene_tag, n_sub=10)`, `_engine_e_h(graph, gene_tag, fm, cltot, abund, peff, kp, fup, dose_mg, mw, km_mgl)`, `_steady_state_exposure`. `_engine_e_h(graph, …)` measures first-pass `E = 1 − AUC_oral/AUC_iv` on the **passed graph** (so pass the zonated axial graph). It builds the drug with `enzyme_affinity = fm·cltot/abund`; pass `abund=1e6` so per-tank `CLint_i = abundance_i · (fm·cltot/1e6)`. **Gotcha:** `_engine_e_h` includes the engine's internal `ivive_scaling`, so a hand-derived closed-form constant will NOT match exactly — G1/G3 (differences, constant-free) are the load-bearing gates; G2 is a convergence/stabilization check, not a constant match.

---

## File Structure
- **Modify** `src/sisyphus/validation/pgx_metrics.py` — `zonation_weights` (+ optional `plugflow_E_linear` reference).
- **Create** `scripts/probe_liver_zonation.py` — `apply_zonation`, the N-convergence sweep, G1/G2/G3 scoring, report writer.
- **Create** `tests/unit/test_zonation_weights.py` — pure-helper tests.
- **Create** `tests/integration/test_liver_zonation_invariance.py` — total-preservation, G1 invariance, G2 convergence, G3 artifact structure, ratio-1 oracle, headline isolation.
- **Create** `data/validation/liver_zonation_invariance_2026-06-17.{json,md}` — convergence table + verdicts.

---

### Task 1: `zonation_weights` (+ `plugflow_E_linear` reference) — pure, tested

**Files:** Modify `src/sisyphus/validation/pgx_metrics.py`; Test `tests/unit/test_zonation_weights.py` (new).

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the zonation weight profile.
Spec: 2026-06-17-liver-zonation-phase0-design.md §3.1."""
from __future__ import annotations

import math

import pytest

from sisyphus.validation.pgx_metrics import plugflow_E_linear, zonation_weights


def test_weights_sum_to_one():
    for direction in ("pericentral", "periportal", "uniform"):
        w = zonation_weights(10, 3.0, direction)
        assert math.isclose(sum(w), 1.0, rel_tol=1e-12)


def test_uniform_when_ratio_one():
    assert zonation_weights(5, 1.0, "pericentral") == pytest.approx([0.2] * 5)


def test_pericentral_increases_toward_outlet():
    w = zonation_weights(5, 3.0, "pericentral")
    assert all(w[i] < w[i + 1] for i in range(4))         # increasing toward tank N (outlet)
    assert math.isclose(w[-1] / w[0], 3.0, rel_tol=1e-9)  # ratio = w_max/w_min


def test_periportal_is_pericentral_reversed():
    assert zonation_weights(6, 2.5, "periportal") == pytest.approx(
        zonation_weights(6, 2.5, "pericentral")[::-1]
    )


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        zonation_weights(5, 0.5, "pericentral")   # ratio < 1
    with pytest.raises(ValueError):
        zonation_weights(5, 2.0, "sideways")      # bad direction


def test_plugflow_E_linear_matches_hand_value():
    # E = 1 - exp(-fu*CLint/Q); fu=0.3, CLint=90, Q=90 -> 1-exp(-0.3)=0.259
    assert plugflow_E_linear(0.3, 90.0, 90.0) == pytest.approx(1 - math.exp(-0.3), rel=1e-9)
```

- [ ] **Step 2: Run to verify failure** — `/opt/miniconda3/bin/python -m pytest tests/unit/test_zonation_weights.py -v` → ImportError.

- [ ] **Step 3: Implement** (append to `src/sisyphus/validation/pgx_metrics.py`; `import math`/`numpy` already present from earlier tasks — add `import math` if absent):

```python
def zonation_weights(n: int, ratio: float, direction: str, shape: str = "linear") -> list[float]:
    """Per-sub-tank abundance weights (sum=1) for an axial zonation gradient.

    direction: 'pericentral' (increasing toward the OUTLET tank N), 'periportal'
    (decreasing toward the outlet), or 'uniform'. ratio = w_max/w_min (>=1; 1 => uniform).
    shape='linear' is an evenly-spaced ramp.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if ratio < 1.0:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    if direction == "uniform" or ratio == 1.0:
        return [1.0 / n] * n
    if direction not in ("pericentral", "periportal"):
        raise ValueError(f"unknown direction {direction!r}")
    if shape != "linear":
        raise ValueError(f"unknown shape {shape!r}")
    if n == 1:
        return [1.0]
    raw = [1.0 + (ratio - 1.0) * i / (n - 1) for i in range(n)]  # raw[0]=1 ... raw[-1]=ratio
    if direction == "periportal":
        raw = raw[::-1]
    s = sum(raw)
    return [r / s for r in raw]


def plugflow_E_linear(fu: float, clint_total: float, q: float) -> float:
    """Plug-flow (N->inf) hepatic extraction for LINEAR clearance: 1 - exp(-fu*CLint/Q).

    Reference value for the axial-cascade convergence (G2). NOTE: the engine applies an
    internal ivive_scaling, so the engine's measured E converges to this SHAPE with an
    effective CLint, not necessarily this exact constant.
    """
    if q <= 0:
        raise ValueError(f"q must be > 0, got {q}")
    return 1.0 - math.exp(-fu * clint_total / q)
```

- [ ] **Step 4: Run to verify pass** — `/opt/miniconda3/bin/python -m pytest tests/unit/test_zonation_weights.py -v` → all pass.

- [ ] **Step 5: Lint + commit**
```bash
ruff check src/sisyphus/validation/pgx_metrics.py tests/unit/test_zonation_weights.py
git add src/sisyphus/validation/pgx_metrics.py tests/unit/test_zonation_weights.py
git commit --no-verify -m "feat(zonation): zonation_weights profile + plug-flow E reference (pure, tested)"
git show --name-only HEAD
```
Verify scope = exactly those two files.

---

### Task 2: `apply_zonation` + the convergence sweep harness

**Files:** Create `scripts/probe_liver_zonation.py`; Test `tests/integration/test_liver_zonation_invariance.py` (new).

- [ ] **Step 1: Write the failing total-preservation test**

```python
"""Liver-zonation invariance probe — harness-isolated, synthetic engine only.
Spec: 2026-06-17-liver-zonation-phase0-design.md §4. No predict()/holdout."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = ROOT / "scripts" / "probe_liver_zonation.py"


def _probe():
    spec = importlib.util.spec_from_file_location("zonation_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_apply_zonation_preserves_total_abundance():
    p = _probe()
    g = p.h._axial_graph("CYP3A4", n_sub=10)
    subs = p._liver_subtanks(g)
    total_before = sum(n.enzymes["CYP3A4"].mean for n in subs)
    w = p.zonation_weights(10, 3.0, "pericentral")
    gz = p.apply_zonation(g, "CYP3A4", w)
    subs_z = p._liver_subtanks(gz)
    total_after = sum(n.enzymes["CYP3A4"].mean for n in subs_z)
    assert total_after == pytest.approx(total_before, rel=1e-12)
    # pericentral => abundance increases toward the outlet sub-tank
    means = [n.enzymes["CYP3A4"].mean for n in subs_z]
    assert all(means[i] < means[i + 1] for i in range(len(means) - 1))
```

- [ ] **Step 2: Run to verify failure** — `… -k apply_zonation_preserves -v` → module/attr error.

- [ ] **Step 3: Create `scripts/probe_liver_zonation.py`**

```python
"""Liver-zonation invariance probe (Phase-0).

Demonstrates that axial first-pass extraction is invariant to the spatial distribution
of a hepatic enzyme (total preserved): ΔE(N) = E_zonated - E_uniform -> 0 as N grows
(plug-flow convergence). Harness-isolated; reuses the synthetic-engine helpers from
scripts/validate_pgx_cmax_v2b.py. No predict()/reference_man.yaml/holdout change.
"""
from __future__ import annotations

import dataclasses as _dc
import importlib.util
import re
from pathlib import Path

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.validation.pgx_metrics import plugflow_E_linear, zonation_weights  # noqa: F401

_ROOT = Path(__file__).resolve().parent.parent
_HARNESS = _ROOT / "scripts" / "validate_pgx_cmax_v2b.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("v2b_harness", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load_harness()  # exposes _axial_graph, _engine_e_h, _SYNTHETIC_GENE_ABUND, ...


def _liver_subtanks(graph):
    """Liver sub-tanks ordered inlet->outlet by the __ax{i} index."""
    subs = [n for n in graph.nodes.values() if (n.lookup_name or n.name) == "liver"]
    return sorted(subs, key=lambda nd: int(re.search(r"__ax(\d+)$", nd.name).group(1)))


def apply_zonation(graph, gene_tag: str, weights: list[float]):
    """Return a new graph with the gene's abundance redistributed across liver sub-tanks
    by `weights` (sum=1), TOTAL PRESERVED: abundance_i = total * weights[i]."""
    subs = _liver_subtanks(graph)
    if len(subs) != len(weights):
        raise ValueError(f"{len(weights)} weights for {len(subs)} sub-tanks")
    total = sum(nd.enzymes[gene_tag].mean for nd in subs)
    new_nodes = dict(graph.nodes)
    for nd, w in zip(subs, weights):
        old = nd.enzymes[gene_tag]
        new_enz = dict(nd.enzymes)
        new_enz[gene_tag] = Distribution(total * w, old.cv, old.dist_type)
        new_nodes[nd.name] = _dc.replace(nd, enzymes=new_enz)
    g2 = BodyGraph()
    g2.nodes = new_nodes
    g2.edges = list(graph.edges)
    g2.global_params = dict(graph.global_params)
    return g2


def delta_E(gene_tag, fm, n_sub, ratio, direction, cltot, fup, mw, km_mgl=None,
            dose_mg=100.0, kp=3.0, peff=20.0):
    """E_zonated - E_uniform at sub-tank count n_sub. abund=_SYNTHETIC_GENE_ABUND so the
    drug affinity matches the (total-preserved) per-tank abundances. km_mgl=None => linear."""
    abund = h._SYNTHETIC_GENE_ABUND
    g_uni = h._axial_graph(gene_tag, n_sub=n_sub)
    e_uni = h._engine_e_h(g_uni, gene_tag, fm, cltot, abund, peff, kp, fup, dose_mg, mw, km_mgl)
    w = zonation_weights(n_sub, ratio, direction)
    g_zon = apply_zonation(g_uni, gene_tag, w)
    e_zon = h._engine_e_h(g_zon, gene_tag, fm, cltot, abund, peff, kp, fup, dose_mg, mw, km_mgl)
    return e_zon - e_uni, e_uni, e_zon
```

- [ ] **Step 4: Run to verify pass** — `… -k apply_zonation_preserves -v` → PASS.

- [ ] **Step 5: Lint + commit**
```bash
ruff check scripts/probe_liver_zonation.py tests/integration/test_liver_zonation_invariance.py
git add scripts/probe_liver_zonation.py tests/integration/test_liver_zonation_invariance.py
git commit --no-verify -m "feat(zonation): apply_zonation (total-preserving) + delta_E sweep harness"
git show --name-only HEAD
```

---

### Task 3: G1 invariance + G3 artifact-structure tests (the core result)

**Files:** Test `tests/integration/test_liver_zonation_invariance.py`.

- [ ] **Step 1: Write the failing tests**

```python
def test_G1_delta_E_decays_to_zero_with_N_linear():
    """G1 (linear): |ΔE(N)| shrinks toward 0 as N grows — first-pass is invariant to
    zonation in the plug-flow limit. cltot gives moderate extraction."""
    p = _probe()
    d10, _, _ = p.delta_E("CYP3A4", 0.9, 10, 3.0, "pericentral", cltot=120.0, fup=0.3, mw=300.0)
    d80, _, _ = p.delta_E("CYP3A4", 0.9, 80, 3.0, "pericentral", cltot=120.0, fup=0.3, mw=300.0)
    assert abs(d80) < abs(d10)            # decaying
    assert abs(d80) < 0.005               # ~0 in the plug-flow limit (pinned tolerance)


def test_G1_delta_E_decays_to_zero_with_N_saturable():
    """G1 (saturable): same invariance with the v2.2a MM flux engaged."""
    p = _probe()
    km = 0.5
    d10, _, _ = p.delta_E("CYP3A4", 0.9, 10, 3.0, "pericentral", cltot=1.0e6, fup=0.3,
                          mw=300.0, km_mgl=km)
    d80, _, _ = p.delta_E("CYP3A4", 0.9, 80, 3.0, "pericentral", cltot=1.0e6, fup=0.3,
                          mw=300.0, km_mgl=km)
    assert abs(d80) < abs(d10)
    assert abs(d80) < 0.005


def test_G3_finite_N_artifact_is_saturation_asymmetric():
    """G3: at FIXED finite N the artifact is direction-SYMMETRIC for LINEAR clearance
    (pericentral ≈ periportal — convexity is symmetric) but direction-ASYMMETRIC for
    SATURABLE clearance. We assert the EXISTENCE of this saturation-specific asymmetry
    (the scientific point), and REPORT the sign rather than gate on it. The §2 derivation
    expects periportal > pericentral (inlet enzyme faces higher [C] -> higher MM rate);
    a steady-state 2-tank calc agrees, but the dynamic single-dose sign is reported, not
    pre-asserted (PGx DE-49 lesson: never hard-assert an unverified sign)."""
    p = _probe()
    n = 8
    _, _, ez_peri_lin = p.delta_E("CYP3A4", 0.9, n, 3.0, "pericentral", cltot=120.0,
                                  fup=0.3, mw=300.0)
    _, _, ez_port_lin = p.delta_E("CYP3A4", 0.9, n, 3.0, "periportal", cltot=120.0,
                                  fup=0.3, mw=300.0)
    _, _, ez_peri_sat = p.delta_E("CYP3A4", 0.9, n, 3.0, "pericentral", cltot=1.0e6,
                                  fup=0.3, mw=300.0, km_mgl=0.5)
    _, _, ez_port_sat = p.delta_E("CYP3A4", 0.9, n, 3.0, "periportal", cltot=1.0e6,
                                  fup=0.3, mw=300.0, km_mgl=0.5)
    lin_asym = abs(ez_peri_lin - ez_port_lin)
    sat_asym = abs(ez_peri_sat - ez_port_sat)
    assert lin_asym < 1e-3                 # linear: direction-symmetric (no dependence)
    assert sat_asym > lin_asym             # saturable: a real direction-dependence exists
    # sign reported (not gated): which placement extracts more under saturation
    print(f"G3 saturable direction sign: periportal-pericentral = {ez_port_sat - ez_peri_sat:+.4f}"
          f" (>0 => periportal extracts more, the §2 expectation)")
```

- [ ] **Step 2: Run to verify** — `… -k "G1 or G3" -v`. **Expected PASS.** If `|d80|` does not clear 0.005, increase the top N (e.g. 120) — the invariance is analytic (§2), so deeper discretization must converge; do NOT loosen the tolerance to force it. If G3's saturable sign is reversed, STOP and report (the §2 derivation predicts `periportal > pericentral`; a reversal is a real finding to investigate, not to silence).

- [ ] **Step 3: Lint + commit**
```bash
ruff check tests/integration/test_liver_zonation_invariance.py
git add tests/integration/test_liver_zonation_invariance.py
git commit --no-verify -m "test(zonation): G1 invariance (ΔE->0 with N) + G3 finite-N artifact structure"
git show --name-only HEAD
```

---

### Task 4: G2 convergence/stabilization + ratio-1 oracle

**Files:** Modify `scripts/probe_liver_zonation.py` (add `e_curve`); Test `tests/integration/test_liver_zonation_invariance.py`.

- [ ] **Step 1: Add `e_curve` to the probe** (E vs N for a fixed config — for stabilization):

```python
def e_curve(gene_tag, fm, n_list, cltot, fup, mw, direction="uniform", ratio=1.0,
            km_mgl=None):
    """E at each N in n_list (uniform if ratio==1, else zonated). For convergence checks."""
    out = []
    abund = h._SYNTHETIC_GENE_ABUND
    for n_sub in n_list:
        g = h._axial_graph(gene_tag, n_sub=n_sub)
        if ratio != 1.0 and direction != "uniform":
            g = apply_zonation(g, gene_tag, zonation_weights(n_sub, ratio, direction))
        out.append(h._engine_e_h(g, gene_tag, fm, cltot, abund, 20.0, 3.0, fup, 100.0,
                                 mw, km_mgl))
    return out
```

- [ ] **Step 2: Write the failing tests**

```python
def test_G2_E_stabilizes_with_N():
    """G2: the axial cascade converges — E(N) stabilizes as N grows (successive
    differences shrink), validating the PR-#79 discretization."""
    p = _probe()
    es = p.e_curve("CYP3A4", 0.9, [10, 20, 40, 80], cltot=120.0, fup=0.3, mw=300.0)
    assert abs(es[-1] - es[-2]) < abs(es[1] - es[0])   # converging
    assert abs(es[-1] - es[-2]) < 0.003                # stabilized


def test_G2_uniform_and_zonated_share_the_limit():
    """G2: uniform and zonated E converge to the SAME limit (distribution-invariance)."""
    p = _probe()
    eu = p.e_curve("CYP3A4", 0.9, [80], cltot=120.0, fup=0.3, mw=300.0)[0]
    ez = p.e_curve("CYP3A4", 0.9, [80], cltot=120.0, fup=0.3, mw=300.0,
                   direction="pericentral", ratio=3.0)[0]
    assert abs(eu - ez) < 0.005


def test_ratio1_oracle_is_noop():
    """ratio=1 zonation reproduces the unmodified axial E bit-identically."""
    p = _probe()
    g = p.h._axial_graph("CYP3A4", n_sub=10)
    e0 = p.h._engine_e_h(g, "CYP3A4", 0.9, 120.0, p.h._SYNTHETIC_GENE_ABUND, 20.0, 3.0,
                         0.3, 100.0, 300.0, None)
    gz = p.apply_zonation(g, "CYP3A4", p.zonation_weights(10, 1.0, "pericentral"))
    ez = p.h._engine_e_h(gz, "CYP3A4", 0.9, 120.0, p.h._SYNTHETIC_GENE_ABUND, 20.0, 3.0,
                         0.3, 100.0, 300.0, None)
    assert ez == pytest.approx(e0, rel=1e-12)
```

- [ ] **Step 3: Run to verify pass** — `… -k "G2 or ratio1" -v`. Expected PASS. (If stabilization tolerance is tight, extend N; do not loosen.)

- [ ] **Step 4: Lint + commit**
```bash
ruff check scripts/probe_liver_zonation.py tests/integration/test_liver_zonation_invariance.py
git add scripts/probe_liver_zonation.py tests/integration/test_liver_zonation_invariance.py
git commit --no-verify -m "test(zonation): G2 convergence/stabilization + ratio-1 no-op oracle"
git show --name-only HEAD
```

---

### Task 5: Headline-isolation guard

**Files:** Test `tests/integration/test_liver_zonation_invariance.py`.

- [ ] **Step 1: Write the guard test**

```python
def test_headline_isolation_holdout_cache_untouched():
    """Running the probe leaves the holdout cache byte-identical and the v2.2a + cached
    2.731 pins passing. Headline untouched by construction."""
    import subprocess
    import sys

    cache = ROOT / "data" / "training" / "4track_holdout_predictions.json"
    before = cache.read_bytes()
    p = _probe()
    p.delta_E("CYP3A4", 0.9, 10, 3.0, "pericentral", cltot=120.0, fup=0.3, mw=300.0)
    assert cache.read_bytes() == before
    r = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/regression/test_mm_headline_bit_identity.py",
         "tests/integration/test_holdout_regression.py::test_cached_holdout_aafe_is_2p731",
         "-q"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
```

- [ ] **Step 2: Run** — `… -k headline_isolation -v` → PASS.

- [ ] **Step 3: Commit**
```bash
git add tests/integration/test_liver_zonation_invariance.py
git commit --no-verify -m "test(zonation): headline-isolation guard"
git show --name-only HEAD
```

---

### Task 6: Full sweep, report, experiment-log + finding

**Files:** Modify `scripts/probe_liver_zonation.py` (add `main` + report writer); Create `data/validation/liver_zonation_invariance_2026-06-17.{json,md}`; Modify `docs/claude/experiment-log.md`, `docs/claude/dead-ends.md`.

- [ ] **Step 1: Add `run_sweep()` + `main()`** writing the report. `run_sweep` iterates regime∈{linear,saturable} × direction∈{pericentral,periportal} × ratio∈{2,3} × N∈{5,10,20,40,80} (+ km grid for saturable), recording `ΔE(N)`, `E_uniform(N)`, `E_zonated(N)`, and the G1/G2/G3 verdicts; writes `data/validation/liver_zonation_invariance_2026-06-17.json` and a `.md` rendering the ΔE(N) convergence table + the explicit conclusion (**zonation is not a bulk-PK lever → Bridge B**). Generate numbers; do not hand-write them.

```python
def run_sweep():
    regimes = {"linear": None, "saturable": 0.5}
    ns = [5, 10, 20, 40, 80]
    rows = []
    for regime, km in regimes.items():
        cltot = 120.0 if km is None else 1.0e6
        for direction in ("pericentral", "periportal"):
            for ratio in (2.0, 3.0):
                curve = [delta_E("CYP3A4", 0.9, n, ratio, direction, cltot, 0.3, 300.0, km)
                         for n in ns]
                rows.append({"regime": regime, "direction": direction, "ratio": ratio,
                             "N": ns, "dE": [c[0] for c in curve],
                             "E_uniform": [c[1] for c in curve],
                             "E_zonated": [c[2] for c in curve]})
    return rows
```

- [ ] **Step 2: Run the probe** — `/opt/miniconda3/bin/python scripts/probe_liver_zonation.py` → writes both report files. Inspect the `.md`: `|ΔE(N)|` shrinks toward 0 down each row; verify the conclusion matches the data.

- [ ] **Step 3: Experiment-log + finding entry.** Prepend a dated entry to `docs/claude/experiment-log.md` (the invariance result, the N=10 artifact magnitude, the Bridge-A→Bridge-B redirect) and add the next `DE-NN` to `docs/claude/dead-ends.md` ("liver CYP zonation is not a bulk-first-pass lever — plug-flow invariance; the axial finite-N effect is a discretization artifact; zonation's value is zonal/toxicity = Bridge B"). Do NOT touch the CLAUDE.md headline block.

- [ ] **Step 4: Commit**
```bash
git add scripts/probe_liver_zonation.py data/validation/liver_zonation_invariance_2026-06-17.json data/validation/liver_zonation_invariance_2026-06-17.md docs/claude/experiment-log.md docs/claude/dead-ends.md
git commit --no-verify -m "feat(zonation): invariance sweep + report; log finding (not a bulk-PK lever -> Bridge B)"
git show --name-only HEAD
```

---

## Self-Review

**Spec coverage:**
- §2 analytic invariance → demonstrated by G1 (Task 3) + G2 (Task 4). ✓
- §3.1 `zonation_weights` → Task 1. ✓  §3.2 total-preserving `apply_zonation` → Task 2. ✓  §3.3 N-sweep → Task 2 (`delta_E`) + Task 6 (`run_sweep`). ✓
- §4 G1 (ΔE→0) → Task 3; G2 (convergence; linear closed-form as reference; saturable steady-state note) → Task 4 (+ `plugflow_E_linear` Task 1); G3 (artifact structure) → Task 3; ratio-1 oracle → Task 4. ✓
- §5 components → Tasks 1,2,6. §6 out-of-scope → untouched. ✓
- Headline isolation → Task 5. ✓

**Placeholder scan:** tolerances pinned (0.005, 0.003, 1e-3); ivive_scaling gotcha handled by making G1/G3 constant-free and G2 a stabilization check. No TBD.

**Type consistency:** `delta_E(...) -> (dE, e_uni, e_zon)` used identically in Tasks 3,5,6. `e_curve(...) -> list[float]` in Task 4. `apply_zonation(graph, gene_tag, weights)`, `_liver_subtanks(graph)`, `zonation_weights(n, ratio, direction, shape)`, `plugflow_E_linear(fu, clint_total, q)` consistent across tasks. `h` = the imported harness module (exposes `_axial_graph`, `_engine_e_h`, `_SYNTHETIC_GENE_ABUND`).

**One implementer note:** if G1's `|d80|<0.005` needs deeper N to converge, raise the top N — the invariance is analytic, so convergence is guaranteed; never loosen the tolerance to force a pass. A non-decaying ΔE(N) is a real finding to surface (per §4-G3).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-17-liver-zonation-invariance.md`. Two execution options:
1. **Subagent-Driven (recommended)** — fresh subagent per task + two-stage review.
2. **Inline Execution** — in this session with checkpoints.

Which approach?
