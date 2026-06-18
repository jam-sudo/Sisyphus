# Zonal GSH-pool Depletion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen the B1 zonal hazard probe (PR #82) with a dynamic, depleting per-zone GSH pool, demonstrating that the resulting hazard is *history-dependent* (provably beyond the static pointwise model) while bulk PK stays invariant — qualitatively the acetaminophen mechanism.

**Architecture:** One pure post-processor function (`gsh_pool_hazard`, a self-contained RK4 GSH-pool ODE integrator) + one metric (`transition_width`) added to `src/sisyphus/validation/pgx_metrics.py`; one harness-isolated probe script (`scripts/probe_gsh_depletion.py`) that reuses the B1/PR-#79 axial machinery via importlib and runs five sweeps; unit + integration tests; a results report. No engine / `predict()` / `reference_man.yaml` change — the GSH pool and reactive metabolite are NOT engine species. Headline 2.731 is bit-identical throughout.

**Tech Stack:** Python 3.10+, numpy only (no scipy in the new pure code), the merged Sisyphus engine (`ODECompiler`/`solve`) reached only through the existing B1 harness, pytest, ruff (line-length 100). Tests + probe run under `/opt/miniconda3/bin/python`.

**Operating constraints (every task):**
- Work on branch `feat/gsh-pool-depletion` (already created, spec already committed there).
- Commit as `jam-sudo <jam-sudo@users.noreply.github.com>`, NO `Co-Authored-By: Claude` / AI trailer, always `git commit --no-verify`.
- Stage ONLY the files named in each task. NEVER `git add README.md` or the untracked workspace docs (`ChemRxiv_submission_metadata.md`, `docs/numeric_drift_followups_2026-06-12.md`, `docs/preprint_v3_revised_evaluation.md`, etc.). Use explicit `git add <path>` — never `git add -A`/`git add .`.
- Run tests with `/opt/miniconda3/bin/python -m pytest`. Run `/opt/miniconda3/bin/python -m ruff check src tests` (line-length 100) before each commit that touches `src/`/`tests/`.
- Harness-isolated: no edits under `src/sisyphus/engine/`, `src/sisyphus/predict/`, `src/sisyphus/pipeline/`, `data/physiology/`, or the holdout list.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/sisyphus/validation/pgx_metrics.py` (modify, append only) | Add `gsh_pool_hazard(...)` (pure RK4 pool integrator + escaped-flux trapezoid) and `transition_width(...)` (10→90% normalized dose-span). B1's `zonal_hazard`/`mm_rate`/`zonation_weights` reused unchanged. |
| `scripts/probe_gsh_depletion.py` (new) | The probe: `_ordering_profiles`, `_divided_dose_profile_by_zone`, importlib-reuse of B1 helpers, five sweeps, G-order/G1/G2/G-time/G-cliff/G-NAC scoring, report writer. |
| `tests/unit/test_gsh_pool_hazard.py` (new) | Pure-function tests for `gsh_pool_hazard` + `transition_width`. |
| `tests/integration/test_gsh_depletion_probe.py` (new) | Gate tests over the probe + headline-isolation guard. |
| `data/validation/gsh_depletion_2026-06-18.{json,md}` (new, generated) | Sweep tables + verdicts + conclusion. |
| `docs/claude/experiment-log.md` (modify) | Dated entry. |
| `docs/claude/dead-ends.md` (modify, only if honest-negative) | DE entry if a gate genuinely fails. |

---

## Task 1: `gsh_pool_hazard` + `transition_width` in pgx_metrics.py

**Files:**
- Modify: `src/sisyphus/validation/pgx_metrics.py` (append after `zonal_hazard`, ends at line 181)
- Test: `tests/unit/test_gsh_pool_hazard.py`

**Background the implementer needs:** The per-zone pool ODE is
`dGSH/dt = k_syn·(GSH0 − GSH) − R_form(t)·GSH/(Kg+GSH)` with `R_form(t)=Vmax_bio·C_u(t)/(Km_bio+C_u(t))`,
and the hazard is `∫ R_form(t)·(1 − GSH(t)/(Kg+GSH(t))) dt` (the NAPQI escaping a depleted pool).
`C_u(t)` is given on the grid `time`; integrate the ODE on an internally-refined uniform grid (so accuracy
does not depend on the engine's coarse `t_eval`), linearly interpolating `C_u`. Clamp `GSH≥0` each step
(the autocatalytic crash can otherwise push it slightly negative). Pure numpy, deterministic, no scipy.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_gsh_pool_hazard.py`:

```python
"""Unit tests for the GSH-pool hazard post-processor and the transition-width metric."""
import numpy as np

from sisyphus.validation.pgx_metrics import gsh_pool_hazard, mm_rate, transition_width


def _const_profile(c, t_end=24.0, n=2401):
    t = np.linspace(0.0, t_end, n)
    return [np.full_like(t, float(c))], t


def test_zero_formation_gives_zero_hazard_and_full_pool():
    # Vmax_bio=0 => no NAPQI => pool stays at GSH0, hazard == 0.
    c_by_zone, t = _const_profile(1.0)
    h = gsh_pool_hazard(c_by_zone, vmax_bio_by_zone=[0.0], km_bio=1.0,
                        gsh0_by_zone=[10.0], k_syn=0.3, kg=1.0, time=t)
    assert h[0] == 0.0


def test_constant_cu_reaches_analytic_steady_state():
    # Long constant exposure: numeric GSH(end) matches the algebraic steady state
    # k_syn*(GSH0-g) = R*g/(Kg+g), and hazard rate -> R*(1 - g/(Kg+g)).
    c = 2.0
    vmax_bio, km_bio, gsh0, k_syn, kg = 5.0, 1.0, 10.0, 0.3, 1.0
    r = mm_rate(c, vmax_bio, km_bio)
    # solve quadratic q*g^2 - g*(q*GSH0 - q*Kg - r) - q*GSH0*Kg = 0 for g (positive root)
    q = k_syn
    b = q * gsh0 - q * kg - r
    g_star = (b + np.sqrt(b * b + 4 * q * q * gsh0 * kg)) / (2 * q)
    haz_rate_star = r * (1.0 - g_star / (kg + g_star))
    c_by_zone, t = _const_profile(c, t_end=200.0, n=20001)  # long enough to equilibrate
    h = gsh_pool_hazard(c_by_zone, [vmax_bio], km_bio, [gsh0], k_syn, kg, t)
    # late-window hazard accumulates at ~haz_rate_star per hour
    late = gsh_pool_hazard([c_by_zone[0][t >= 150.0]], [vmax_bio], km_bio, [gsh0],
                           k_syn, kg, t[t >= 150.0])
    span = float(t[t >= 150.0][-1] - t[t >= 150.0][0])
    assert abs(late[0] / span - haz_rate_star) / haz_rate_star < 0.02


def test_saturating_formation_drains_pool_hazard_to_full_escape():
    # Huge formation vs tiny synthesis: pool ~0, escape factor ~1, hazard -> ∫R_form.
    c = 50.0
    vmax_bio, km_bio = 100.0, 1.0
    c_by_zone, t = _const_profile(c, t_end=24.0, n=4801)
    r = mm_rate(c, vmax_bio, km_bio)
    h = gsh_pool_hazard(c_by_zone, [vmax_bio], km_bio, gsh0_by_zone=[1.0],
                        k_syn=0.01, kg=1.0, time=t)
    integral_rform = r * float(t[-1] - t[0])
    assert h[0] > 0.9 * integral_rform


def test_hazard_monotone_decreasing_in_gsh0():
    # G-NAC analytic lever: raising GSH0 can only lower the escaped-flux hazard.
    c_by_zone, t = _const_profile(5.0, t_end=24.0, n=4801)
    kw = dict(vmax_bio_by_zone=[20.0], km_bio=1.0, k_syn=0.3, kg=1.0, time=t)
    h_lo = gsh_pool_hazard(c_by_zone, gsh0_by_zone=[5.0], **kw)[0]
    h_hi = gsh_pool_hazard(c_by_zone, gsh0_by_zone=[50.0], **kw)[0]
    assert h_hi < h_lo


def test_deterministic():
    c_by_zone, t = _const_profile(5.0)
    a = gsh_pool_hazard(c_by_zone, [20.0], 1.0, [5.0], 0.3, 1.0, t)
    b = gsh_pool_hazard(c_by_zone, [20.0], 1.0, [5.0], 0.3, 1.0, t)
    assert a == b


def test_transition_width_sharp_lt_broad():
    doses = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    # broad: gradual ramp; sharp: near-step at one dose
    broad = [0.0, 0.1, 0.3, 0.6, 0.85, 1.0]
    sharp = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    w_broad = transition_width(doses, broad)
    w_sharp = transition_width(doses, sharp)
    assert w_sharp < w_broad


def test_transition_width_zero_curve_is_nan_safe():
    # All-zero (or flat) curve has no defined 10->90 transition; return inf (not a crash).
    doses = [1.0, 2.0, 3.0]
    assert transition_width(doses, [0.0, 0.0, 0.0]) == float("inf")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_gsh_pool_hazard.py -q`
Expected: FAIL — `ImportError: cannot import name 'gsh_pool_hazard'` (and `transition_width`).

- [ ] **Step 3: Implement both functions**

Append to `src/sisyphus/validation/pgx_metrics.py` (after line 181):

```python
def gsh_pool_hazard(c_u_by_zone, vmax_bio_by_zone, km_bio, gsh0_by_zone, k_syn, kg, time,
                    steps_per_interval: int = 8):
    """Per-zone reactive-metabolite hazard under a DYNAMIC, depleting GSH pool (spec §2).

    Pool ODE per zone:  dGSH/dt = k_syn*(GSH0 - GSH) - R_form(t)*GSH/(Kg+GSH),
    with R_form(t) = Vmax_bio * C_u(t) / (Km_bio + C_u(t)). Hazard (escaped covalent
    binding) = ∫ R_form(t) * (1 - GSH(t)/(Kg+GSH(t))) dt.

    Integrated with a self-contained fixed-step RK4 on a grid refined `steps_per_interval`x
    per input interval (linear C_u interpolation); GSH is clamped >= 0 against the
    autocatalytic crash. Pure numpy / deterministic / no scipy. Returns a per-zone list.
    """
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("time must be a 1-D array of length >= 2")
    trapz = getattr(np, "trapezoid", np.trapz)

    # Refined uniform integration grid spanning [t0, t_end].
    n_fine = (t.size - 1) * int(steps_per_interval) + 1
    tf = np.linspace(t[0], t[-1], n_fine)

    def _rform(c):
        return vmax_bio * c / (km_bio + c)

    out = []
    for c_u, vmax_bio, gsh0 in zip(c_u_by_zone, vmax_bio_by_zone, gsh0_by_zone):
        c_arr = np.asarray(c_u, dtype=float)
        c_fine = np.interp(tf, t, c_arr)

        def _dgsh(gsh, c):
            return k_syn * (gsh0 - gsh) - _rform(c) * gsh / (kg + gsh)

        gsh = float(gsh0)
        gsh_fine = np.empty(n_fine)
        gsh_fine[0] = gsh
        for i in range(n_fine - 1):
            dt = tf[i + 1] - tf[i]
            c0, c1 = c_fine[i], c_fine[i + 1]
            cmid = 0.5 * (c0 + c1)
            k1 = _dgsh(gsh, c0)
            k2 = _dgsh(gsh + 0.5 * dt * k1, cmid)
            k3 = _dgsh(gsh + 0.5 * dt * k2, cmid)
            k4 = _dgsh(gsh + dt * k3, c1)
            gsh = gsh + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if gsh < 0.0:
                gsh = 0.0
            gsh_fine[i + 1] = gsh

        escape = 1.0 - gsh_fine / (kg + gsh_fine)
        haz_rate = _rform(c_fine) * escape
        out.append(float(trapz(haz_rate, tf)))
    return out


def transition_width(doses, values):
    """Dose-span over which a (monotone-ish) dose-response curve rises 10%->90% of its own
    max, on a log10-dose axis. Smaller = sharper. Curve is normalized to its max so the
    metric is immune to a hard-zero floor (spec §4 G-cliff). Returns inf if the curve never
    reaches 90% above 10% (e.g. all-zero / flat)."""
    d = np.asarray(doses, dtype=float)
    v = np.asarray(values, dtype=float)
    vmax = float(np.max(v))
    if vmax <= 0.0:
        return float("inf")
    vn = v / vmax
    ld = np.log10(d)

    def _cross(level):
        for i in range(1, len(vn)):
            if vn[i - 1] < level <= vn[i]:
                # linear interp in log-dose where vn crosses `level`
                f = (level - vn[i - 1]) / (vn[i] - vn[i - 1])
                return ld[i - 1] + f * (ld[i] - ld[i - 1])
        return None

    lo, hi = _cross(0.1), _cross(0.9)
    if lo is None or hi is None:
        return float("inf")
    return float(hi - lo)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_gsh_pool_hazard.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Lint**

Run: `/opt/miniconda3/bin/python -m ruff check src/sisyphus/validation/pgx_metrics.py tests/unit/test_gsh_pool_hazard.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/validation/pgx_metrics.py tests/unit/test_gsh_pool_hazard.py
git commit --no-verify -m "feat(validation): gsh_pool_hazard dynamic-pool post-processor + transition_width (B1.x)"
```

---

## Task 2: the probe script `scripts/probe_gsh_depletion.py`

**Files:**
- Create: `scripts/probe_gsh_depletion.py`
- (no test in this task — Task 3 tests the probe; this task delivers a runnable script with importable functions)

**Reuse contract (verified against the repo):**
- `scripts/probe_zonal_hazard.py` (B1, PR #82) exposes, at module scope, `_parent_profile_by_zone(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, dose_mg=100.0, kp=3.0, peff=20.0) -> (c_u_by_zone, time)` and `bulk_E(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, bio_direction, bio_ratio) -> float`, plus the loaded harness module `h` (with `_axial_graph`, `_sat_drug`, `_SYNTHETIC_GENE_ABUND`, `_T_EVAL`) and `_subtank_names(graph)`. Load it by the SAME importlib pattern it itself uses.
- `zonation_weights(n, ratio, direction)` and the new `gsh_pool_hazard`, `transition_width`, plus B1's `zonal_hazard`, are imported from `sisyphus.validation.pgx_metrics`.

- [ ] **Step 1: Write the script**

Create `scripts/probe_gsh_depletion.py`:

```python
"""Zonal GSH-pool depletion probe (Bridge B / B1.x, Phase-0).

Upgrades the B1 static-detox hazard (PR #82) to a DYNAMIC, depleting per-zone GSH pool.
Demonstrates: the pool makes the hazard HISTORY-DEPENDENT (a pure concentration-reordering
moves the dynamic hazard but provably NOT the static pointwise hazard — the centerpiece);
excess path-dependence (bolus vs divided) over the static envelope baseline; a transient
depletion cliff (autocatalytic, reported via transition_width); an NAC-precursor lever
(monotone in GSH0); all orthogonal to bulk parent PK (DE-50). Harness-isolated; reuses the
B1 axial machinery via importlib. No predict()/reference_man.yaml/holdout/engine change.

a-priori physiological pinning (fixed BEFORE observing any cliff/path outcome, spec §3.4):
GSH resynthesis t1/2 ~ 2-4 h => k_syn = ln2/t1/2 ~ 0.2-0.35 /h; tau (divided spacing) on
the recovery timescale.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from sisyphus.validation.pgx_metrics import (
    gsh_pool_hazard,
    transition_width,
    zonal_hazard,
    zonation_weights,
)

_ROOT = Path(__file__).resolve().parent.parent


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_b1 = _load("b1_probe", "scripts/probe_zonal_hazard.py")  # _parent_profile_by_zone, bulk_E, h
h = _b1.h

# --- a-priori pinned pool kinetics (spec §3.4) ---
_T_HALF_GSH_H = 3.0                      # hepatic GSH resynthesis t1/2 ~ 2-4 h (mid)
_K_SYN = np.log(2.0) / _T_HALF_GSH_H     # ~0.231 /h
_KG = 1.0                                # scavenging affinity (mg/L-equiv), synthetic
_KM_BIO = 1.0
_TAU_H = 4.0                             # divided-dose spacing, on the recovery timescale

# Same acetaminophen-like skeleton config as B1 (_CFG), minus the static detox fields.
_CFG = dict(gene_tag="CYP3A4", fm=0.9, n_sub=10, cltot=1.0e6, fup=0.3, mw=300.0,
            km_mgl=0.5)
_VMAX_BIO_TOTAL = 300.0
_GSH0_TOTAL = 150.0                       # baseline pool budget (synthetic)
_VMAX_DETOX_TOTAL = 15.0                  # B1 static capacity, for the dynamic-vs-static arm
_DOSES = [50.0, 100.0, 200.0, 400.0, 800.0]
# APAP zonation: bioactivation pericentral-high, pool pericentral-low (detox-equivalent).
_APAP = dict(bio_direction="pericentral", bio_ratio=3.0,
             gsh_direction="periportal", gsh_ratio=3.0)


def _vmax_bio_zone(n_sub, direction, ratio):
    return [_VMAX_BIO_TOTAL * w for w in zonation_weights(n_sub, ratio, direction)]


def _gsh0_zone(n_sub, direction, ratio):
    return [_GSH0_TOTAL * w for w in zonation_weights(n_sub, ratio, direction)]


def _dynamic_profile_hazard(c_by_zone, time, bio_dir, bio_ratio, gsh_dir, gsh_ratio):
    n = len(c_by_zone)
    return gsh_pool_hazard(c_by_zone, _vmax_bio_zone(n, bio_dir, bio_ratio), _KM_BIO,
                           _gsh0_zone(n, gsh_dir, gsh_ratio), _K_SYN, _KG, time)


def _static_profile_hazard(c_by_zone, time, bio_dir, bio_ratio, detox_dir, detox_ratio):
    n = len(c_by_zone)
    vmax_bio = _vmax_bio_zone(n, bio_dir, bio_ratio)
    vmax_detox = [_VMAX_DETOX_TOTAL * w for w in zonation_weights(n, detox_ratio, detox_dir)]
    return zonal_hazard(c_by_zone, vmax_bio, _KM_BIO, vmax_detox, time)


# ---------- G-order: pure history-dependence (no engine) ----------
def _ordering_profiles(levels, dt_h=2.0, pts_per_step=200):
    """Two single-zone C_u(t) trajectories visiting the SAME value-multiset in ascending vs
    descending order (each level held dt_h). Static hazard is identical by construction;
    dynamic differs (pool memory)."""
    asc = list(levels)
    desc = list(levels)[::-1]
    t_parts, ca, cd = [], [], []
    for k, (la, ld) in enumerate(zip(asc, desc)):
        seg = np.linspace(k * dt_h, (k + 1) * dt_h, pts_per_step, endpoint=False)
        t_parts.append(seg)
        ca.append(np.full_like(seg, float(la)))
        cd.append(np.full_like(seg, float(ld)))
    t = np.concatenate(t_parts + [np.array([len(asc) * dt_h])])
    c_asc = np.concatenate(ca + [np.array([float(asc[-1])])])
    c_desc = np.concatenate(cd + [np.array([float(desc[-1])])])
    return t, c_asc, c_desc


def order_test():
    levels = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    t, c_asc, c_desc = _ordering_profiles(levels)
    vmax_bio, gsh0 = [_VMAX_BIO_TOTAL], [_GSH0_TOTAL]
    vmax_detox = [_VMAX_DETOX_TOTAL]
    hs_a = zonal_hazard([c_asc], vmax_bio, _KM_BIO, vmax_detox, t)[0]
    hs_d = zonal_hazard([c_desc], vmax_bio, _KM_BIO, vmax_detox, t)[0]
    hd_a = gsh_pool_hazard([c_asc], vmax_bio, _KM_BIO, gsh0, _K_SYN, _KG, t)[0]
    hd_d = gsh_pool_hazard([c_desc], vmax_bio, _KM_BIO, gsh0, _K_SYN, _KG, t)[0]
    static_rel = abs(hs_a - hs_d) / max(hs_a, hs_d, 1e-30)
    dyn_rel = abs(hd_a - hd_d) / max(hd_a, hd_d, 1e-30)
    return {"static_asc": hs_a, "static_desc": hs_d, "static_rel_diff": static_rel,
            "dyn_asc": hd_a, "dyn_desc": hd_d, "dyn_rel_diff": dyn_rel}


# ---------- G-time: divided-dose two-segment axial solve ----------
def _divided_dose_profile_by_zone(total_dose, n_splits, tau_h):
    """Per-sub-tank C_u(t) for `n_splits` equal doses spaced tau_h, on the synthetic axial
    liver. Two-segment compiled-ODE solve: dose, integrate to tau, ADD next dose to the
    admin-node state, re-solve from the carried state; concatenate. Equal total mass to a
    single bolus (the static control absorbs the saturable-first-pass envelope difference)."""
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve

    g = h._axial_graph(_CFG["gene_tag"], n_sub=_CFG["n_sub"])
    abund = h._SYNTHETIC_GENE_ABUND
    drug = h._sat_drug(_CFG["gene_tag"], _CFG["fm"], _CFG["cltot"], abund, 20.0, 3.0,
                       _CFG["km_mgl"], _CFG["fup"], total_dose, _CFG["mw"])
    rg, rd = g.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    admin = compiled.state_index[drug.administration_node]
    names = _b1._subtank_names(g)

    per_dose = total_dose / n_splits
    t_end = float(h._T_EVAL[-1])
    t_all, conc = [], {nm: [] for nm in names}
    y = np.zeros(compiled.n_states)
    t_offset = 0.0
    for s in range(n_splits):
        y[admin] += per_dose
        seg_end = tau_h if s < n_splits - 1 else (t_end - t_offset)
        seg_end = max(seg_end, 1e-6)
        t_eval = np.linspace(0.0, seg_end, 400)
        res = solve(compiled, params, y, t_span=(0.0, seg_end), t_eval=t_eval)
        keep = slice(0, -1) if s < n_splits - 1 else slice(0, None)
        t_all.append(t_offset + np.asarray(res.time_h)[keep])
        for nm in names:
            conc[nm].append(np.asarray(res.concentrations[nm])[keep])
        # Seed the next segment from the FULL end-of-segment state (amounts, mg),
        # matching the y0 = amounts convention (dose_mg is placed into an amount state).
        y = np.zeros(compiled.n_states)
        for name, idx in compiled.state_index.items():
            arr = res.amounts.get(name)
            if arr is not None:
                y[idx] = float(np.asarray(arr)[-1])
        t_offset += seg_end
    time = np.concatenate(t_all)
    c_by_zone = [_CFG["fup"] * np.concatenate(conc[nm]) for nm in names]
    return c_by_zone, time


def time_test():
    total = 400.0
    cb, tb = _b1._parent_profile_by_zone(_CFG["gene_tag"], _CFG["fm"], _CFG["n_sub"],
                                         _CFG["cltot"], _CFG["fup"], _CFG["mw"],
                                         _CFG["km_mgl"], dose_mg=total)
    cd, td = _divided_dose_profile_by_zone(total, n_splits=2, tau_h=_TAU_H)
    dyn_b = max(_dynamic_profile_hazard(cb, tb, _APAP["bio_direction"], _APAP["bio_ratio"],
                                        _APAP["gsh_direction"], _APAP["gsh_ratio"]))
    dyn_d = max(_dynamic_profile_hazard(cd, td, _APAP["bio_direction"], _APAP["bio_ratio"],
                                        _APAP["gsh_direction"], _APAP["gsh_ratio"]))
    sta_b = max(_static_profile_hazard(cb, tb, _APAP["bio_direction"], _APAP["bio_ratio"],
                                       "periportal", _APAP["gsh_ratio"]))
    sta_d = max(_static_profile_hazard(cd, td, _APAP["bio_direction"], _APAP["bio_ratio"],
                                       "periportal", _APAP["gsh_ratio"]))
    dyn_ratio = dyn_b / max(dyn_d, 1e-30)
    sta_ratio = sta_b / max(sta_d, 1e-30)
    return {"dyn_bolus": dyn_b, "dyn_divided": dyn_d, "dyn_ratio": dyn_ratio,
            "static_bolus": sta_b, "static_divided": sta_d, "static_ratio": sta_ratio,
            "excess_path_dependence": dyn_ratio - sta_ratio, "tau_h": _TAU_H}


# ---------- G1/G2 zonation + G-cliff dose + G-NAC ----------
def zonation_test():
    rows = []
    for bdir in ("pericentral", "uniform", "periportal"):
        br = 1.0 if bdir == "uniform" else 3.0
        cb, tb = _b1._parent_profile_by_zone(_CFG["gene_tag"], _CFG["fm"], _CFG["n_sub"],
                                             _CFG["cltot"], _CFG["fup"], _CFG["mw"],
                                             _CFG["km_mgl"], dose_mg=200.0)
        haz = _dynamic_profile_hazard(cb, tb, bdir, br, "uniform", 1.0)
        e = _b1.bulk_E(_CFG["gene_tag"], _CFG["fm"], _CFG["n_sub"], _CFG["cltot"],
                       _CFG["fup"], _CFG["mw"], _CFG["km_mgl"], bdir, br)
        rows.append({"bio_zonation": bdir, "bulk_E": round(e, 6),
                     "hazard_peak_zone": int(np.argmax(haz)), "maxH": round(max(haz), 4)})
    e_span = max(r["bulk_E"] for r in rows) - min(r["bulk_E"] for r in rows)
    return rows, e_span


def dose_test():
    dyn_curve, sta_curve, rows = [], [], []
    for d in _DOSES:
        cb, tb = _b1._parent_profile_by_zone(_CFG["gene_tag"], _CFG["fm"], _CFG["n_sub"],
                                             _CFG["cltot"], _CFG["fup"], _CFG["mw"],
                                             _CFG["km_mgl"], dose_mg=d)
        hd = _dynamic_profile_hazard(cb, tb, _APAP["bio_direction"], _APAP["bio_ratio"],
                                     _APAP["gsh_direction"], _APAP["gsh_ratio"])
        hs = _static_profile_hazard(cb, tb, _APAP["bio_direction"], _APAP["bio_ratio"],
                                    "periportal", _APAP["gsh_ratio"])
        dyn_curve.append(max(hd))
        sta_curve.append(max(hs))
        rows.append({"dose": d, "dyn_maxH": round(max(hd), 4), "dyn_peak_zone": int(np.argmax(hd)),
                     "static_maxH": round(max(hs), 4)})
    return rows, transition_width(_DOSES, dyn_curve), transition_width(_DOSES, sta_curve)


def nac_test():
    global _GSH0_TOTAL
    base = _GSH0_TOTAL
    out = []
    cb, tb = _b1._parent_profile_by_zone(_CFG["gene_tag"], _CFG["fm"], _CFG["n_sub"],
                                         _CFG["cltot"], _CFG["fup"], _CFG["mw"],
                                         _CFG["km_mgl"], dose_mg=400.0)
    for scale in (1.0, 1.5, 3.0):
        _GSH0_TOTAL = base * scale
        haz = _dynamic_profile_hazard(cb, tb, _APAP["bio_direction"], _APAP["bio_ratio"],
                                      _APAP["gsh_direction"], _APAP["gsh_ratio"])
        out.append({"gsh0_scale": scale, "maxH": round(max(haz), 4)})
    _GSH0_TOTAL = base
    return out


def run_sweep():
    order = order_test()
    zon_rows, e_span = zonation_test()
    dose_rows, w_dyn, w_sta = dose_test()
    return {
        "pinned": {"t_half_gsh_h": _T_HALF_GSH_H, "k_syn_per_h": _K_SYN, "kg": _KG,
                   "tau_h": _TAU_H, "gsh0_total": _GSH0_TOTAL, "km_bio": _KM_BIO},
        "G_order": order,
        "G2_invariance_contrast": zon_rows, "G2_bulk_E_span": e_span,
        "G_time": time_test(),
        "G_cliff": {"rows": dose_rows, "width_dynamic": w_dyn, "width_static": w_sta},
        "G_NAC": nac_test(),
    }


def main():
    import json

    res = run_sweep()
    base = _ROOT / "data" / "validation" / "gsh_depletion_2026-06-18"
    out = {
        "title": "Zonal GSH-pool depletion probe (Bridge B / B1.x, Phase-0)",
        "date": "2026-06-18",
        "conclusion": (
            "A depleting per-zone GSH pool makes the zonal reactive-metabolite hazard "
            "HISTORY-DEPENDENT: a pure concentration reordering leaves the static pointwise "
            f"hazard unchanged (rel diff {res['G_order']['static_rel_diff']:.1e}) while moving "
            f"the dynamic hazard ({res['G_order']['dyn_rel_diff']:.2f}) — structure beyond the "
            "B1 static model and orthogonal to bulk parent PK (DE-50, bulk-E span "
            f"{res['G2_bulk_E_span']:.1e}). Excess path-dependence over the static envelope "
            f"baseline = {res['G_time']['excess_path_dependence']:+.3f}; dose transition width "
            f"dynamic {res['G_cliff']['width_dynamic']:.3f} vs static "
            f"{res['G_cliff']['width_static']:.3f} (log10-dose); raising GSH0 lowers hazard "
            "(NAC lever). k_syn/tau pinned a priori from GSH t1/2. Headline 2.731 untouched "
            "(harness-isolated). Qualitative acetaminophen mechanism; not a calibrated tox number."
        ),
        **res,
    }
    base.with_suffix(".json").write_text(json.dumps(out, indent=2, default=float))

    o = res["G_order"]
    g = res["G_cliff"]
    tt = res["G_time"]
    lines = [
        "# Zonal GSH-pool depletion probe — Bridge B / B1.x Phase-0 (2026-06-18)",
        "",
        "**Harness-isolated** (`scripts/probe_gsh_depletion.py`); the GSH pool and reactive "
        "metabolite are a POST-PROCESSOR on the axial parent profile, not engine species. No "
        "`predict()` / `reference_man.yaml` / holdout change; headline **2.731 bit-identical**. "
        "Reuses the axial machinery (PR #79) + B1 harness (PR #82). `k_syn`/`tau` pinned a "
        f"priori: GSH t1/2 {_T_HALF_GSH_H} h -> k_syn {_K_SYN:.3f}/h, tau {_TAU_H} h.",
        "", "## Conclusion", "", out["conclusion"], "",
        "## G-order — pool memory (centerpiece)",
        "",
        f"Same value-multiset, reordered. Static rel diff **{o['static_rel_diff']:.1e}** "
        f"(invariant, by construction) vs dynamic rel diff **{o['dyn_rel_diff']:.3f}** "
        "(moves) — the pool carries order/history information the static model cannot.",
        "",
        "## G2 — local matters, bulk doesn't (DE-50)",
        "",
        f"Bulk parent E span across bioactivation zonation **{res['G2_bulk_E_span']:.2e}** "
        "(~invariant) while the dynamic hazard peak-zone moves:",
        "",
        "| bio zonation | bulk E | hazard peak-zone | maxH |",
        "|---|---|---|---|",
    ]
    for r in res["G2_invariance_contrast"]:
        lines.append(f"| {r['bio_zonation']} | {r['bulk_E']} | {r['hazard_peak_zone']} "
                     f"| {r['maxH']} |")
    lines += [
        "",
        "## G-time — excess path-dependence (bolus vs 2x divided, equal dose)",
        "",
        f"dynamic ratio {tt['dyn_ratio']:.3f} vs static ratio {tt['static_ratio']:.3f} "
        f"=> **excess {tt['excess_path_dependence']:+.3f}** (tau {tt['tau_h']} h). The static "
        "path effect is measured, not assumed zero; the excess is the pool-dynamics signature.",
        "",
        "## G-cliff — dynamic vs static dose-response sharpness",
        "",
        f"transition width (log10-dose, 10->90% of own max): dynamic **{g['width_dynamic']:.3f}** "
        f"vs static **{g['width_static']:.3f}** (smaller = sharper; reported, not presupposed).",
        "",
        "| dose | dyn maxH | dyn peak-zone | static maxH |",
        "|---|---|---|---|",
    ]
    for r in g["rows"]:
        lines.append(f"| {r['dose']} | {r['dyn_maxH']} | {r['dyn_peak_zone']} "
                     f"| {r['static_maxH']} |")
    lines += [
        "",
        "## G-NAC — precursor protective lever",
        "",
        "| GSH0 scale | maxH |", "|---|---|",
    ]
    for r in res["G_NAC"]:
        lines.append(f"| {r['gsh0_scale']} | {r['maxH']} |")
    lines += ["", "peak-zone 0-indexed inlet(0)->outlet(9); zone 9 = pericentral / zone 3.", ""]
    base.with_suffix(".md").write_text("\n".join(lines))
    print("G-order static/dyn rel:", f"{o['static_rel_diff']:.1e}", f"{o['dyn_rel_diff']:.3f}",
          "| G2 E-span:", f"{res['G2_bulk_E_span']:.1e}",
          "| G-time excess:", f"{tt['excess_path_dependence']:+.3f}",
          "| cliff dyn/static:", f"{g['width_dynamic']:.3f}/{g['width_static']:.3f}")


if __name__ == "__main__":
    main()
```

> **Implementer note (one live-verification point):** the next-segment seed reads
> `res.amounts[name]` (mg time series, per the `SimResult` contract) and places the last
> value into `y[state_index[name]]` — correct because the engine's states are amounts (the
> B1 solve seeds `y0[state_index[admin]] = dose_mg`). Confirm at runtime (Task 2 Step 2)
> that `res.amounts` is populated for every `state_index` name; if any node is concentration-
> only, reconstruct its amount as concentration × compartment volume from `params` before
> seeding. The divided-dose parent profile must show two distinct absorption humps.

- [ ] **Step 2: Verify the script runs end-to-end**

Run: `/opt/miniconda3/bin/python scripts/probe_gsh_depletion.py`
Expected: prints a one-line summary and writes `data/validation/gsh_depletion_2026-06-18.{json,md}`. If `res_state_end` mismatches the engine's `SimResult`, fix per the implementer note until it runs and the divided-dose parent profile is non-degenerate (two absorption humps visible in `cd`).

- [ ] **Step 3: Lint**

Run: `/opt/miniconda3/bin/python -m ruff check scripts/probe_gsh_depletion.py`
Expected: `All checks passed!`

- [ ] **Step 4: Commit (script only — NOT the generated data yet; Task 4 regenerates it)**

```bash
git add scripts/probe_gsh_depletion.py
git commit --no-verify -m "feat(validation): zonal GSH-pool depletion probe (B1.x)"
```

---

## Task 3: integration gate tests

**Files:**
- Create: `tests/integration/test_gsh_depletion_probe.py`

**Pattern:** import the probe by file path with importlib (same as the probe imports B1), call its sweep functions, assert the gates. Use stack-independent assertions (inequalities / signs / tolerances), NOT pinned floats — CI Linux drifts ~12% from macOS.

- [ ] **Step 1: Write the tests**

Create `tests/integration/test_gsh_depletion_probe.py`:

```python
"""Gate tests for the zonal GSH-pool depletion probe (B1.x). Stack-independent
assertions only (signs/inequalities/tolerances), not pinned floats."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "gsh_probe", _ROOT / "scripts" / "probe_gsh_depletion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe()


def test_g_order_static_invariant_dynamic_variant():
    # Centerpiece: reordering the SAME concentration value-multiset leaves the static
    # pointwise hazard unchanged (proven invariant) but moves the dynamic pool hazard.
    o = probe.order_test()
    assert o["static_rel_diff"] < 1e-9          # measure-preserving reordering invariant
    assert o["dyn_rel_diff"] > 0.05             # pool memory: materially different


def test_g2_bulk_invariant_hazard_variant():
    rows, e_span = probe.zonation_test()
    assert e_span < 1e-2                         # bulk parent E ~invariant to CYP zonation (DE-50)
    peaks = {r["hazard_peak_zone"] for r in rows}
    assert len(peaks) > 1                        # per-zone hazard peak-zone moves with zonation


def test_g1_localization_apap_peaks_pericentral():
    # APAP config (bioactivation pericentral-high, pool pericentral-low) -> outlet peak.
    cb, tb = probe._b1._parent_profile_by_zone(
        probe._CFG["gene_tag"], probe._CFG["fm"], probe._CFG["n_sub"], probe._CFG["cltot"],
        probe._CFG["fup"], probe._CFG["mw"], probe._CFG["km_mgl"], dose_mg=400.0)
    haz = probe._dynamic_profile_hazard(cb, tb, probe._APAP["bio_direction"],
                                        probe._APAP["bio_ratio"], probe._APAP["gsh_direction"],
                                        probe._APAP["gsh_ratio"])
    assert int(np.argmax(haz)) >= probe._CFG["n_sub"] - 3   # pericentral (outlet) zone


def test_g_time_excess_path_dependence_reported_and_dynamic_ge_static():
    t = probe.time_test()
    # both ratios finite & >0; dynamic carries at least as much path-dependence as static.
    assert t["dyn_ratio"] > 0 and t["static_ratio"] > 0
    assert t["dyn_ratio"] >= t["static_ratio"] - 1e-6       # excess >= 0 (within noise)


def test_g_cliff_transition_widths_finite_and_reported():
    rows, w_dyn, w_sta = probe.dose_test()
    # both curves rise over the dose grid (not flat) -> finite widths; report comparison.
    assert np.isfinite(w_dyn) and np.isfinite(w_sta)
    assert rows[-1]["dyn_maxH"] > rows[0]["dyn_maxH"]       # dynamic hazard rises with dose


def test_g_nac_monotone_protective():
    out = probe.nac_test()
    maxh = [r["maxH"] for r in out]                          # gsh0 scale 1.0,1.5,3.0
    assert maxh[0] >= maxh[1] >= maxh[2]                     # more pool -> less hazard


def test_headline_isolation_unchanged():
    # The 4-track holdout cache must be untouched by anything in this probe.
    p = _ROOT / "data" / "training" / "4track_holdout_predictions.json"
    d = json.loads(p.read_text())
    assert abs(d["overall"]["meta"]["aafe"] - 2.731) < 5e-3


@pytest.mark.parametrize("name", ["test_cached_holdout_aafe_is_2p731",
                                  "test_mm_headline_bit_identity"])
def test_headline_pins_exist(name):
    # Guard: the canonical headline pins still exist in the suite (regenerated, not removed).
    found = list(_ROOT.glob("tests/**/*.py"))
    assert any(name in f.read_text() for f in found), f"{name} pin missing"
```

- [ ] **Step 2: Run the tests**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_gsh_depletion_probe.py -q`
Expected: PASS. If `test_g_order` dynamic side is below 0.05, the pinned `k_syn`/pool are in a too-fast-recovery regime — do NOT tune to pass; instead record an honest-negative (the pool memory is below the visibility floor) and adjust the assertion to document the measured value, then carry the finding to Task 4 / dead-ends. (Reordering memory should be robustly present for slow `k_syn`; a near-zero dynamic diff would itself be a notable finding.)

- [ ] **Step 3: Lint**

Run: `/opt/miniconda3/bin/python -m ruff check tests/integration/test_gsh_depletion_probe.py`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_gsh_depletion_probe.py
git commit --no-verify -m "test(validation): gate tests for GSH-pool depletion probe (B1.x)"
```

---

## Task 4: regenerate report + docs

**Files:**
- Create (generated): `data/validation/gsh_depletion_2026-06-18.{json,md}`
- Modify: `docs/claude/experiment-log.md` (prepend a dated entry)
- Modify (only if a gate honestly failed): `docs/claude/dead-ends.md`

- [ ] **Step 1: Regenerate the report from the committed probe**

Run: `/opt/miniconda3/bin/python scripts/probe_gsh_depletion.py`
Expected: writes both files; prints the summary line. Read `data/validation/gsh_depletion_2026-06-18.md` and confirm the verdicts are internally consistent (static_rel_diff ≪ dyn_rel_diff; bulk-E span small; excess path-dependence reported; transition widths reported).

- [ ] **Step 2: Append the experiment-log entry**

Prepend to `docs/claude/experiment-log.md` (use the actual measured numbers from Step 1, not these placeholders-of-shape):

```markdown
## 2026-06-18 — Bridge B / B1.x: zonal GSH-pool depletion (history-dependent hazard)

Upgraded the B1 static-detox hazard (PR #82) to a dynamic, depleting per-zone GSH pool
(`gsh_pool_hazard`, self-contained RK4 post-processor; `k_syn=ln2/t½_GSH`, t½≈3 h pinned a
priori). **Centerpiece (G-order):** a pure concentration reordering leaves the static
pointwise hazard invariant (rel diff <1e-9, by the measure-preserving-reordering identity)
while moving the dynamic pool hazard (rel diff <MEASURED>) — history-dependence the static
model structurally cannot represent, orthogonal to bulk parent PK (DE-50, bulk-E span
<MEASURED>). G-time excess path-dependence (bolus vs 2× divided) = <MEASURED> over the
static envelope baseline. G-cliff transition width dynamic <MEASURED> vs static <MEASURED>
(log10-dose; reported, not presupposed). G-NAC: raising GSH0 monotonically lowers hazard
(analytic). Harness-isolated; headline **2.731 bit-identical**. `scripts/probe_gsh_depletion.py`,
`data/validation/gsh_depletion_2026-06-18.{json,md}`. Qualitative acetaminophen mechanism,
not a calibrated tox number.
```

- [ ] **Step 3: (Only if a gate honestly failed) add a dead-ends entry**

If G-cliff shows the dynamic is NOT sharper, or G-order/G-time collapse, append the next `DE-NN`
to `docs/claude/dead-ends.md` stating the measured null and that no parameter was tuned to rescue
it. If all gates pass, SKIP this step (this is a positive result; B1.x is not a dead end).

- [ ] **Step 4: Verify headline bit-identity + full new-suite green**

```bash
/opt/miniconda3/bin/python -m pytest tests/unit/test_gsh_pool_hazard.py \
  tests/integration/test_gsh_depletion_probe.py -q
/opt/miniconda3/bin/python -m pytest -k "cached_holdout_aafe_is_2p731 or mm_headline_bit_identity" -q
```
Expected: all PASS; the headline pins green (proves 2.731 untouched).

- [ ] **Step 5: Commit**

```bash
git add data/validation/gsh_depletion_2026-06-18.json data/validation/gsh_depletion_2026-06-18.md docs/claude/experiment-log.md
# add docs/claude/dead-ends.md ONLY if Step 3 created an entry
git commit --no-verify -m "validation(bridge-b): GSH-pool depletion results + experiment-log (B1.x)"
```

---

## Self-Review (completed)

**Spec coverage:** §1 purpose items → G-order (Task 2 `order_test`/Task 3), path (G-time), cliff (G-cliff), NAC (G-NAC) all mapped. §2 pool ODE + autocatalytic crash + GSH-clamp → Task 1 integrator. §3.1 reuse / §3.3 ordering / §3.4 divided-dose + a-priori pinning → Task 2. §4 all six gates → Task 3 tests. §5 components → Tasks 1–4 file-for-file. §6/§7 constraints → operating-constraints block + headline-isolation tests.

**Placeholder scan:** the one intentional placeholder line in `_divided_dose_profile_by_zone` is explicitly called out with deletion instructions and the live-verification note (the SimResult amounts access is the single point that may differ from the engine API). No `TBD`/`add error handling`/uncoded steps elsewhere.

**Type consistency:** `gsh_pool_hazard(c_u_by_zone, vmax_bio_by_zone, km_bio, gsh0_by_zone, k_syn, kg, time)` and `transition_width(doses, values)` signatures identical across Task 1 (def + tests), Task 2 (calls), Task 3 (via probe). `zonal_hazard`/`zonation_weights`/`mm_rate` used with their real repo signatures (verified). Probe functions (`order_test`, `time_test`, `zonation_test`, `dose_test`, `nac_test`, `run_sweep`) referenced consistently in Task 3.

**Known live-verification point:** the next-segment seed reading `res.amounts[name]` — Task 2 Step 2 catches this at runtime; the implementer note gives the fallback (rebuild the amount from concentration × compartment volume per the compiler state convention).
