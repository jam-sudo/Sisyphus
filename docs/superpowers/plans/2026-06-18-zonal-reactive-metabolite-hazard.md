# Zonal Reactive-Metabolite Hazard Probe (B1 Phase-0) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Demonstrate on the real axial engine that a per-zone reactive-metabolite hazard (local bioactivation exceeding local saturable detox) localizes by zonation, is strongly zonation-dependent while the bulk parent PK is invariant (DE-50 closure), and shows a saturable-detox dose-threshold — qualitatively reproducing the acetaminophen zone-3 pattern.

**Architecture:** A pure hazard post-processor + a harness-isolated probe that reuses the merged synthetic-engine helpers (`_axial_graph`, `_engine_e_h`) via `importlib` and `zonation_weights`. The reactive metabolite is NOT an engine species — hazard is computed from the parent's per-sub-tank concentration profile. No `predict()`/`reference_man.yaml`/`src/engine`/`expand_axial` change; headline 2.731 bit-identical.

**Tech Stack:** Python 3.10+, numpy, pytest. Engine+scipy at `/opt/miniconda3/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-18-zonal-reactive-metabolite-hazard-design.md`

**Constraints (load-bearing):**
- Harness-isolated. No `predict()`/`reference_man.yaml`/holdout/`src/sisyphus/engine/`/`expand_axial` change. Headline **2.731 bit-identical**.
- Commit as `jam-sudo <jam-sudo@users.noreply.github.com>`. NEVER add a `Co-Authored-By: Claude`/AI footer/"Generated with" line. `git commit --no-verify`. After each commit `git show --name-only HEAD` and verify scope.
- Stage ONLY the files each task names. NEVER `git add README.md` or untracked workspace files (`ChemRxiv*`, `docs/numeric_drift*`, `docs/preprint_v3*`).
- Tests: `/opt/miniconda3/bin/python -m pytest`. Lint: `ruff check` (line-length 100).
- **Qualitative mechanism demonstration, not a calibrated tox number.** Synthetic-param selection (dose range, Vmax_bio/detox, Km) to make the threshold/zone-specificity VISIBLE is allowed and is NOT fitting to clinical data; document it. Honest-negative (hazard near-invariant / no threshold) is a first-class outcome — report, never tune to manufacture.

**Reused helpers** (`main`, `scripts/validate_pgx_cmax_v2b.py`): `_axial_graph(gene_tag, n_sub=10)`, `_well_stirred_graph`, `_drug`, `_sat_drug`, `_engine_e_h`, `_cmax_auc_tmax`, `_SYNTHETIC_GENE_ABUND=1e6`, `_T_EVAL`, `solve`, `ODECompiler`, `ResolvedParams`. Per-sub-tank concentration: `res.concentrations["liver__ax{i}"]` (time series on `res.time_h == _T_EVAL`); unbound `C_u,i = fup · c_node` (matching `_peak_liver_cu`'s convention). `zonation_weights(n, ratio, direction)` is in `src/sisyphus/validation/pgx_metrics.py` (from DE-50).

---

## File Structure
- **Modify** `src/sisyphus/validation/pgx_metrics.py` — `mm_rate`, `zonal_hazard` (pure).
- **Create** `scripts/probe_zonal_hazard.py` — `_parent_profile_by_zone`, zonation + dose sweeps, G1/G2/G3 scoring, report.
- **Create** `tests/unit/test_zonal_hazard.py` — pure-helper tests.
- **Create** `tests/integration/test_zonal_hazard_probe.py` — parent-profile sanity, G1/G2/G3, headline isolation.
- **Create** `data/validation/zonal_hazard_probe_2026-06-18.{json,md}`.

---

### Task 1: `mm_rate` + `zonal_hazard` (pure, tested)

**Files:** Modify `src/sisyphus/validation/pgx_metrics.py`; Test `tests/unit/test_zonal_hazard.py` (new).

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the zonal reactive-metabolite hazard post-processor.
Spec: 2026-06-18-zonal-reactive-metabolite-hazard-design.md §3.2."""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.validation.pgx_metrics import mm_rate, zonal_hazard


def test_mm_rate_basic():
    assert mm_rate(1.0, 10.0, 1.0) == pytest.approx(5.0)   # c=Km -> half Vmax
    assert mm_rate(0.0, 10.0, 1.0) == pytest.approx(0.0)


def test_zonal_hazard_threshold_zero_below_capacity():
    # constant C_u; formation MM(c)=Vmax_bio*c/(Km+c) < detox capacity => hazard 0
    time = np.linspace(0.0, 10.0, 101)
    c = [np.full_like(time, 1.0)]                # one zone, c=1
    # MM(1; vmax=4, km=1) = 2.0; detox capacity 3.0 > 2.0 -> no excess
    h = zonal_hazard(c, [4.0], 1.0, [3.0], time)
    assert h[0] == pytest.approx(0.0)


def test_zonal_hazard_positive_above_capacity():
    time = np.linspace(0.0, 10.0, 101)
    c = [np.full_like(time, 1.0)]
    # MM(1; vmax=8, km=1) = 4.0; detox 1.0 -> excess 3.0 over T=10 -> 30
    h = zonal_hazard(c, [8.0], 1.0, [1.0], time)
    assert h[0] == pytest.approx(30.0, rel=1e-6)


def test_zonal_hazard_monotonic_decreasing_in_detox():
    time = np.linspace(0.0, 10.0, 101)
    c = [np.full_like(time, 1.0)]
    h_lo = zonal_hazard(c, [8.0], 1.0, [1.0], time)[0]
    h_hi = zonal_hazard(c, [8.0], 1.0, [3.0], time)[0]
    assert h_hi < h_lo


def test_zonal_hazard_per_zone_shape():
    time = np.linspace(0.0, 5.0, 51)
    c = [np.full_like(time, 2.0), np.full_like(time, 0.5)]   # 2 zones
    h = zonal_hazard(c, [10.0, 10.0], 1.0, [2.0, 2.0], time)
    assert len(h) == 2 and h[0] > h[1]      # higher-conc zone has more hazard
```

- [ ] **Step 2: Run to verify failure** — `/opt/miniconda3/bin/python -m pytest tests/unit/test_zonal_hazard.py -v` → ImportError.

- [ ] **Step 3: Implement** (append to `src/sisyphus/validation/pgx_metrics.py`; `numpy as np` already imported):

```python
def mm_rate(c: float, vmax: float, km: float) -> float:
    """Michaelis-Menten rate Vmax*c/(Km+c)."""
    return vmax * c / (km + c)


def zonal_hazard(c_u_by_zone, vmax_bio_by_zone, km_bio, vmax_detox_by_zone, time):
    """Per-zone reactive-metabolite hazard = time-integral of bioactivation rate that
    EXCEEDS local saturable detox capacity: H_i = ∫ max(0, MM(C_u,i; Vmax_bio,i, Km_bio)
    − Vmax_detox,i) dt. The covalent-binding / toxicity proxy (spec §2).

    c_u_by_zone: list of per-zone unbound-conc arrays (each len == len(time)).
    vmax_bio_by_zone / vmax_detox_by_zone: per-zone scalars. Returns a per-zone list.
    """
    t = np.asarray(time, dtype=float)
    trapz = getattr(np, "trapezoid", np.trapz)
    out = []
    for c_u, vmax_bio, vmax_detox in zip(c_u_by_zone, vmax_bio_by_zone, vmax_detox_by_zone):
        c_arr = np.asarray(c_u, dtype=float)
        form = vmax_bio * c_arr / (km_bio + c_arr)          # MM formation rate
        excess = np.maximum(0.0, form - vmax_detox)         # reactive escaping detox
        out.append(float(trapz(excess, t)))
    return out
```

- [ ] **Step 4: Run to verify pass** — expect all pass.

- [ ] **Step 5: Lint + commit**
```bash
ruff check src/sisyphus/validation/pgx_metrics.py tests/unit/test_zonal_hazard.py
git add src/sisyphus/validation/pgx_metrics.py tests/unit/test_zonal_hazard.py
git commit --no-verify -m "feat(bridge-b): zonal hazard post-processor (mm_rate, zonal_hazard) — pure, tested"
git show --name-only HEAD
```
Verify scope = exactly those two files.

---

### Task 2: `_parent_profile_by_zone` harness + parent-profile sanity

**Files:** Create `scripts/probe_zonal_hazard.py`; Test `tests/integration/test_zonal_hazard_probe.py` (new).

- [ ] **Step 1: Write the failing sanity test**

```python
"""Zonal reactive-metabolite hazard probe — harness-isolated, synthetic engine only.
Spec: 2026-06-18-zonal-reactive-metabolite-hazard-design.md. No predict()/holdout."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = ROOT / "scripts" / "probe_zonal_hazard.py"


def _probe():
    spec = importlib.util.spec_from_file_location("zonal_hazard_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parent_profile_by_zone_decreasing_inlet_to_outlet():
    """Sanity: the parent is extracted as it flows, so per-zone peak C_u decreases
    inlet(ax1)->outlet(axN) for a meaningfully-extracted drug."""
    p = _probe()
    c_by_zone, time = p._parent_profile_by_zone(
        "CYP3A4", fm=0.9, n_sub=10, cltot=1.0e6, fup=0.3, mw=300.0, km_mgl=0.5,
        dose_mg=100.0,
    )
    assert len(c_by_zone) == 10 and len(time) == len(c_by_zone[0])
    peaks = [float(np.max(c)) for c in c_by_zone]
    assert peaks[0] > peaks[-1] > 0          # extracted along the tube
```

- [ ] **Step 2: Run to verify failure** — module/attr error.

- [ ] **Step 3: Create `scripts/probe_zonal_hazard.py`**

```python
"""Zonal reactive-metabolite hazard probe (Bridge B / B1, Phase-0).

Computes a per-zone reactive-metabolite hazard as a POST-PROCESSOR on the axial
parent concentration profile (the reactive metabolite is NOT an engine species).
Demonstrates: hazard localizes by zonation; bulk parent PK is invariant to that
zonation (DE-50) while per-zone hazard is not; a saturable-detox dose-threshold with
zone-specificity (acetaminophen pattern). Harness-isolated; reuses the synthetic-engine
helpers from scripts/validate_pgx_cmax_v2b.py. No predict()/reference_man.yaml/holdout.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np

from sisyphus.validation.pgx_metrics import zonal_hazard, zonation_weights  # noqa: F401

_ROOT = Path(__file__).resolve().parent.parent
_HARNESS = _ROOT / "scripts" / "validate_pgx_cmax_v2b.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("v2b_harness", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load_harness()


def _subtank_names(graph):
    subs = [n.name for n in graph.nodes.values() if (n.lookup_name or n.name) == "liver"]
    return sorted(subs, key=lambda nm: int(re.search(r"__ax(\d+)$", nm).group(1)))


def _parent_profile_by_zone(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, dose_mg=100.0,
                            kp=3.0, peff=20.0):
    """Solve a single oral dose on the synthetic axial liver; return
    (c_u_by_zone, time): per-sub-tank UNBOUND parent conc arrays (C_u = fup*c_node,
    matching _peak_liver_cu) ordered inlet(ax1)->outlet(axN), and the time grid."""
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve
    g = h._axial_graph(gene_tag, n_sub=n_sub)
    abund = h._SYNTHETIC_GENE_ABUND
    drug = h._sat_drug(gene_tag, fm, cltot, abund, peff, kp, km_mgl, fup, dose_mg, mw)
    rg, rd = g.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, float(h._T_EVAL[-1])), t_eval=h._T_EVAL)
    names = _subtank_names(g)
    c_u_by_zone = [fup * np.asarray(res.concentrations[nm]) for nm in names]
    return c_u_by_zone, np.asarray(res.time_h)
```

- [ ] **Step 4: Run to verify pass** — sanity test passes.

- [ ] **Step 5: Lint + commit**
```bash
ruff check scripts/probe_zonal_hazard.py tests/integration/test_zonal_hazard_probe.py
git add scripts/probe_zonal_hazard.py tests/integration/test_zonal_hazard_probe.py
git commit --no-verify -m "feat(bridge-b): per-zone parent-profile harness for zonal hazard"
git show --name-only HEAD
```

---

### Task 3 (CONTROLLER-CALIBRATED): G1 localization + G2 bulk-invariant/hazard-variant

> **Calibration gate (controller does this BEFORE writing the gate tests).** After Task 2, the controller runs `_parent_profile_by_zone` + `zonal_hazard` over candidate (dose, Vmax_bio, Vmax_detox, Km_bio, zonation) params to find a regime where the threshold and zone-specificity are VISIBLE (like the DE-50 cltot calibration). The exact constants in the tests below are placeholders `<CAL_*>` the controller replaces with calibrated values; the SHAPE of the assertions is fixed. This is synthetic-param selection for mechanism visibility, NOT clinical fitting.

**Files:** Modify `scripts/probe_zonal_hazard.py` (add `zone_hazard_profile`, `bulk_E`); Test `tests/integration/test_zonal_hazard_probe.py`.

- [ ] **Step 1: Add helpers to the probe**

```python
def zone_hazard_profile(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, dose_mg,
                        bio_direction, bio_ratio, detox_direction, detox_ratio,
                        vmax_bio_total, vmax_detox_total, km_bio):
    """Per-zone hazard for a given bioactivation- and detox-zonation. vmax_bio/detox are
    distributed across zones by zonation_weights (independently). Returns the per-zone
    hazard list (inlet->outlet)."""
    c_by_zone, time = _parent_profile_by_zone(gene_tag, fm, n_sub, cltot, fup, mw,
                                              km_mgl, dose_mg)
    wbio = zonation_weights(n_sub, bio_ratio, bio_direction)
    wdet = zonation_weights(n_sub, detox_ratio, detox_direction)
    vmax_bio = [vmax_bio_total * w for w in wbio]
    vmax_detox = [vmax_detox_total * w for w in wdet]
    return zonal_hazard(c_by_zone, vmax_bio, km_bio, vmax_detox, time)


def bulk_E(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, bio_direction, bio_ratio):
    """Bulk parent extraction with the bioactivation enzyme zonated (for the G2
    invariance arm). Reuses apply_zonation semantics via the DE-50 path."""
    from scripts.probe_liver_zonation import apply_zonation  # noqa
    g = h._axial_graph(gene_tag, n_sub=n_sub)
    g = apply_zonation(g, gene_tag, zonation_weights(n_sub, bio_ratio, bio_direction))
    return h._engine_e_h(g, gene_tag, fm, cltot, h._SYNTHETIC_GENE_ABUND, 20.0, 3.0,
                         fup, 100.0, mw, km_mgl)
```

> Note: `from scripts.probe_liver_zonation import apply_zonation` may not resolve under importlib loading. If it fails, the controller will inline a local `_apply_zonation` copy (3 lines) in `probe_zonal_hazard.py` instead — decide at calibration time. (Documented so the implementer doesn't guess.)

- [ ] **Step 2: Write the gate tests** (controller fills `<CAL_*>` from calibration)

```python
def _aceta_cfg():
    """Acetaminophen-like config: bioactivation pericentral-high, detox pericentral-LOW.
    All <CAL_*> set by controller calibration to a regime where the threshold is visible."""
    return dict(gene_tag="CYP3A4", fm=0.9, n_sub=10, cltot=<CAL_CLTOT>, fup=0.3,
                mw=300.0, km_mgl=<CAL_KM_MGL>, vmax_bio_total=<CAL_VBIO>,
                vmax_detox_total=<CAL_VDET>, km_bio=<CAL_KMBIO>)


def test_G1_hazard_localizes_pericentral_for_aceta_config():
    """G1 (sanity): acetaminophen config (bio pericentral-high, detox pericentral-low)
    -> hazard peaks at the OUTLET zone (zone 3)."""
    p = _probe()
    cfg = _aceta_cfg()
    haz = p.zone_hazard_profile(**cfg, dose_mg=<CAL_DOSE_HI>, bio_direction="pericentral",
                                bio_ratio=3.0, detox_direction="periportal",
                                detox_ratio=3.0)   # detox periportal-high == pericentral-low
    assert int(np.argmax(haz)) >= cfg["n_sub"] - 3       # peak in outlet zones


def test_G2_bulk_E_invariant_while_hazard_profile_varies():
    """G2 (centerpiece, DE-50 closure): holding total bioactivation fixed, varying its
    zonation leaves bulk parent E ~invariant, while the per-zone hazard profile (peak
    zone) moves materially."""
    p = _probe()
    cfg = _aceta_cfg()
    e_peri = p.bulk_E(cfg["gene_tag"], cfg["fm"], cfg["n_sub"], cfg["cltot"], cfg["fup"],
                      cfg["mw"], cfg["km_mgl"], "pericentral", 3.0)
    e_port = p.bulk_E(cfg["gene_tag"], cfg["fm"], cfg["n_sub"], cfg["cltot"], cfg["fup"],
                      cfg["mw"], cfg["km_mgl"], "periportal", 3.0)
    assert abs(e_peri - e_port) < <CAL_E_TOL>            # bulk ~invariant (DE-50)
    haz_bio_peri = p.zone_hazard_profile(**cfg, dose_mg=<CAL_DOSE_HI>,
                                         bio_direction="pericentral", bio_ratio=3.0,
                                         detox_direction="uniform", detox_ratio=1.0)
    haz_bio_port = p.zone_hazard_profile(**cfg, dose_mg=<CAL_DOSE_HI>,
                                         bio_direction="periportal", bio_ratio=3.0,
                                         detox_direction="uniform", detox_ratio=1.0)
    assert int(np.argmax(haz_bio_peri)) != int(np.argmax(haz_bio_port))   # peak moves
```

- [ ] **Step 3: Run** `-k "G1 or G2" -v` → PASS with calibrated constants.

- [ ] **Step 4: Lint + commit**
```bash
ruff check scripts/probe_zonal_hazard.py tests/integration/test_zonal_hazard_probe.py
git add scripts/probe_zonal_hazard.py tests/integration/test_zonal_hazard_probe.py
git commit --no-verify -m "test(bridge-b): G1 localization + G2 bulk-invariant/hazard-variant"
git show --name-only HEAD
```

---

### Task 4: G3 saturable-detox dose-threshold + zone-specificity + protective lever

**Files:** Test `tests/integration/test_zonal_hazard_probe.py`.

- [ ] **Step 1: Write the gate tests** (calibrated `<CAL_DOSE_LO>` below threshold, `<CAL_DOSE_HI>` above)

```python
def test_G3_dose_threshold_and_zone_specificity():
    """G3 (the mechanism): below the threshold dose, NO zone has hazard; above it, the
    pericentral (high-bio/low-detox) zone crosses FIRST. Raising detox raises the
    threshold (protective lever)."""
    p = _probe()
    cfg = _aceta_cfg()
    kw = dict(bio_direction="pericentral", bio_ratio=3.0,
              detox_direction="periportal", detox_ratio=3.0)   # detox pericentral-low
    haz_lo = p.zone_hazard_profile(**cfg, dose_mg=<CAL_DOSE_LO>, **kw)
    haz_hi = p.zone_hazard_profile(**cfg, dose_mg=<CAL_DOSE_HI>, **kw)
    assert max(haz_lo) == pytest.approx(0.0)             # below threshold: no hazard
    assert max(haz_hi) > 0.0                             # above threshold: hazard
    assert int(np.argmax(haz_hi)) >= cfg["n_sub"] - 3    # ...in the pericentral zone

    # protective lever: 3x detox capacity removes the hazard at the same high dose
    cfg_protected = dict(cfg, vmax_detox_total=cfg["vmax_detox_total"] * 3.0)
    haz_protected = p.zone_hazard_profile(**cfg_protected, dose_mg=<CAL_DOSE_HI>, **kw)
    assert max(haz_protected) < max(haz_hi)              # detoxification protects
```

Add `import pytest` to the test file imports if not present.

- [ ] **Step 2: Run** `-k G3 -v` → PASS. If the threshold/zone-specificity is not visible, the controller re-calibrates the param regime (deeper bio/detox contrast, dose span bracketing the threshold) — synthetic visibility, not fitting. A genuine absence of threshold is an honest-negative to report.

- [ ] **Step 3: Lint + commit**
```bash
ruff check tests/integration/test_zonal_hazard_probe.py
git add tests/integration/test_zonal_hazard_probe.py
git commit --no-verify -m "test(bridge-b): G3 saturable-detox dose-threshold + zone-specificity + protective lever"
git show --name-only HEAD
```

---

### Task 5: Headline-isolation guard

**Files:** Test `tests/integration/test_zonal_hazard_probe.py`.

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
    p._parent_profile_by_zone("CYP3A4", 0.9, 8, 1.0e6, 0.3, 300.0, 0.5, dose_mg=100.0)
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

- [ ] **Step 2: Run** `-k headline_isolation -v` → PASS.

- [ ] **Step 3: Commit**
```bash
git add tests/integration/test_zonal_hazard_probe.py
git commit --no-verify -m "test(bridge-b): headline-isolation guard"
git show --name-only HEAD
```

---

### Task 6: Sweep + report + experiment-log

**Files:** Modify `scripts/probe_zonal_hazard.py` (add `run_sweep` + `main`); Create `data/validation/zonal_hazard_probe_2026-06-18.{json,md}`; Modify `docs/claude/experiment-log.md`.

- [ ] **Step 1: Add `run_sweep()` + `main()`** writing the report: (a) the G2 contrast (bulk E across bio-zonation vs hazard peak-zone across bio-zonation); (b) the G3 dose-threshold curve (per-zone hazard vs dose for the acetaminophen config) + the protective-lever arm; (c) G1 localization. Generate numbers; don't hand-write them. The `.md` states the conclusion: **the per-zone hazard is a real surface orthogonal to bulk PK (DE-50 closure); the acetaminophen zone-3 dose-threshold pattern is reproduced qualitatively; this is the first concrete Bridge-B endpoint.**

- [ ] **Step 2: Run** `/opt/miniconda3/bin/python scripts/probe_zonal_hazard.py` → writes both report files; inspect the `.md` (numbers match the `.json`, conclusion matches the data).

- [ ] **Step 3: Experiment-log entry.** Prepend a dated entry to `docs/claude/experiment-log.md`: Bridge B / B1 Phase-0 — the zonal-hazard surface, G1/G2/G3 outcomes, the DE-50 closure (local matters, bulk doesn't), the acetaminophen qualitative reproduction, and the B1.x follow-ups (transported-metabolite, GSH-pool dynamics, quantitative PoD). This is a POSITIVE demonstration (a new surface), not a dead-end — no `DE-NN` unless a gate honest-negatives. Do NOT touch the CLAUDE.md headline block.

- [ ] **Step 4: Commit**
```bash
git add scripts/probe_zonal_hazard.py data/validation/zonal_hazard_probe_2026-06-18.json data/validation/zonal_hazard_probe_2026-06-18.md docs/claude/experiment-log.md
git commit --no-verify -m "feat(bridge-b): zonal-hazard sweep + report; log B1 Phase-0 (first Bridge-B endpoint)"
git show --name-only HEAD
```

---

## Self-Review

**Spec coverage:**
- §2 mechanism (formation-exceeds-detox, both zonated) → `zonal_hazard` (Task 1) + `zone_hazard_profile` (Task 3). ✓
- §3.1 per-zone parent profile → `_parent_profile_by_zone` (Task 2). §3.2 hazard post-processor → Task 1. §3.3 sweeps → Tasks 3,4,6. ✓
- §4 G1 (localization) → Task 3; G2 (bulk-invariant/hazard-variant) → Task 3; G3 (dose-threshold + zone-specificity + protective lever) → Task 4. ✓
- §5 components → Tasks 1,2,6. §6 out-of-scope → untouched. ✓
- Headline isolation → Task 5.

**Placeholder scan:** `<CAL_*>` are INTENTIONAL controller-calibration placeholders (resolved before the gate-test commits, like the DE-50 cltot calibration) — flagged explicitly, not left for the implementer to guess. The `from scripts.probe_liver_zonation import apply_zonation` import risk is flagged with a fallback. No other placeholders.

**Type consistency:** `zonal_hazard(c_u_by_zone, vmax_bio_by_zone, km_bio, vmax_detox_by_zone, time) -> list` used identically Tasks 1,3. `_parent_profile_by_zone(...) -> (c_by_zone, time)` Tasks 2,3,5. `zone_hazard_profile(...)`/`bulk_E(...)` consistent Tasks 3,4,6. `h` = imported harness module; `zonation_weights` from pgx_metrics.

**One controller action (between Task 2 and Task 3):** calibrate `<CAL_*>` to a regime where the G3 threshold + zone-specificity are visible (and confirm `apply_zonation` import resolves or inline it) — exactly the DE-50 calibration discipline. This de-risks the gate tests before the subagent writes them.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-18-zonal-reactive-metabolite-hazard.md`. Two execution options:
1. **Subagent-Driven (recommended)** — fresh subagent per task + two-stage review (with a controller calibration step between Task 2 and Task 3).
2. **Inline Execution** — in this session with checkpoints.

Which approach?
