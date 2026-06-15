# PGx Cmax-fold (engine-differentiated, v2.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that the engine's first-pass ODE predicts the genotype peak-to-exposure
divergence `ρ = log(Cmax_fold) − log(AUC_fold)` better than both a trivial (`ρ=0`) and a
1-comp-first-pass analytic null — a quantity with no closed form.

**Architecture:** Pure metrics module (`ρ`, the 1-comp-first-pass null, EM-anchor solver,
stats) + a controlled `well_stirred` engine harness (EM-anchored / PM-predicted, dense Cmax
grid) + a locked benchmark of both-endpoint genotype-panel studies. Harness-isolated: no
`predict()`/`reference_man.yaml`/holdout change → headline 2.731 bit-untouched.

**Tech Stack:** Python 3.10, numpy, scipy (`optimize.brentq`, `stats.wilcoxon`), the existing
Sisyphus engine (`build_from_yaml`, `ODECompiler`, `solve`, `apply_phenotype_to_graph`).

**Spec:** `docs/superpowers/specs/2026-06-15-pgx-cmax-fold-engine-differentiated-design.md`
(read §2–§6 before Task 3). **Probe evidence:** experiment-log 2026-06-15.

**Conventions (every task):** commit as `jam-sudo` with **no** Claude trailer; `git commit
--no-verify`; run tests with `python -m pytest` (repo root on `sys.path`); ruff line-length
100 (`ruff check src tests` must pass — it runs repo-wide in CI, including `tests/integration`);
`/opt/miniconda3/bin/python` has engine + scipy. Stage only the files each task names — never
`git add README.md` or untracked workspace files.

---

## File Structure

- `src/sisyphus/validation/pgx_metrics.py` — **extend** (v1 file): add `rho`, `rho_null1`,
  `anchor_em`, `rho_band`, `sign_test`, `wilcoxon_div`. Pure (numpy/scipy only, no engine).
- `tests/unit/test_pgx_cmax_metrics.py` — **new**: unit tests for the above.
- `data/validation/pgx_cmax_folds.json` — **new**: locked benchmark (both-endpoint pairs).
- `tests/unit/test_pgx_cmax_benchmark_schema.py` — **new**: schema/curation guard.
- `scripts/validate_pgx_cmax_folds.py` — **new**: the `well_stirred` Cmax-fold harness
  (engine config + EM-anchor recipe + scoring + report). Kept separate from v1's
  `validate_pgx_genotype_folds.py` (different engine config, different metric).
- `tests/integration/test_pgx_cmax_engine_oracle.py` — **new**: C1 (engine AUC-fold =
  analytic on `well_stirred`) + engine-Cmax-vs-1comp oracle regression pins.
- `data/validation/pgx_cmax_fold_validation_2026-06-15.{json,md}` — **generated** by the harness.
- `data/validation/pgx_fm_registry.json` — **extend** (v1 file): add the EM-anchored Cmax layer.
- `docs/claude/experiment-log.md` — **append** the result entry (Task 5).

---

## Task 0: Step-0 feasibility curation gate

**Files:**
- Create: `data/validation/pgx_cmax_feasibility_2026-06-15.md` (curation note, not code)

This mirrors v1's Step-0 gate: confirm enough **clean** pairs exist before building. No code.

- [ ] **Step 1: Curate candidate studies.** For each candidate — metoprolol, nebivolol,
  tramadol, propafenone (CYP2D6); omeprazole (single-dose), lansoprazole (CYP2C19) — find a
  **healthy-volunteer, single-oral-dose, genotype-panel** study reporting **both** `Cmax`
  PM/EM fold **and** `AUC` PM/EM fold (with dispersion), plus the EM-arm `tmax`, `t½`, and a
  literature **oral bioavailability F**. Record per drug: the two folds + CIs, EM `tmax/t½`,
  oral F (+ gut/hepatic split note), in-vitro `fm` (reuse v1 `pgx_fm_registry.json` where
  present), `is_nonlinear`/`is_prodrug` flags, and full citations.

- [ ] **Step 2: Classify each candidate** into: **powered** (both folds, resolvable
  first-pass `ρ_obs≠0`, linear, non-prodrug), **consistency** (low-extraction, `ρ_obs≈0`),
  or **excluded** (single-endpoint / nonlinear / prodrug / circular `fm`), with the reason.

- [ ] **Step 3: Gate.** Write the note with a PASS/REVISE verdict. **PASS requires ≥5 powered
  pairs.** If <5, stop and report — the milestone needs more candidates or a scope change
  (do NOT pad the powered set with low-extraction or single-endpoint pairs).

- [ ] **Step 4: Commit**

```bash
git add data/validation/pgx_cmax_feasibility_2026-06-15.md
git commit --no-verify -m "docs(pgx): v2.1 Step-0 Cmax-fold feasibility curation gate"
```

> **Controller note:** if Step 3 returns REVISE, halt the plan and surface to the human —
> later tasks assume a PASS.

---

## Task 1: pure metrics (`rho`, `rho_null1`, stats, band)

**Files:**
- Modify: `src/sisyphus/validation/pgx_metrics.py` (append functions; keep v1 functions)
- Test: `tests/unit/test_pgx_cmax_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_pgx_cmax_metrics.py
from __future__ import annotations

import math

import pytest

from sisyphus.validation.pgx_metrics import (
    rho,
    rho_null1,
    sign_test,
    wilcoxon_div,
    rho_band,
)


def test_rho_is_log_ratio_of_folds():
    assert rho(cmax_fold=5.0, auc_fold=10.0) == pytest.approx(math.log(0.5))
    assert rho(cmax_fold=2.0, auc_fold=2.0) == pytest.approx(0.0)


def test_rho_rejects_nonpositive():
    with pytest.raises(ValueError):
        rho(cmax_fold=0.0, auc_fold=1.0)


def test_rho_null1_zero_when_gene_uninvolved():
    # fm=0 ⇒ CLint ratio 1 ⇒ both folds 1 ⇒ ρ=0, regardless of regime.
    assert rho_null1(fm=0.0, a=0.0, e_h=0.7, tmax=1.5, thalf=3.5) == pytest.approx(0.0, abs=1e-9)


def test_rho_null1_high_extraction_worked_case():
    # spec §2 high-extraction PM: fm=0.9, a=0, E_h=0.9 ⇒ AUC_fold=10, F_fold≈5.26,
    # Cmax_fold≈5.8 (shape≈1.1) ⇒ ρ≈log(5.8/10)≈-0.55.
    r = rho_null1(fm=0.9, a=0.0, e_h=0.9, tmax=2.0, thalf=4.0)
    assert -0.70 < r < -0.40


def test_rho_null1_nonzero_at_low_extraction():
    # Shape-factor gap persists even with negligible first-pass (E_h≈0).
    r = rho_null1(fm=0.5, a=0.0, e_h=0.02, tmax=2.558, thalf=6.931)
    assert r < -0.2  # Cmax_fold ≪ AUC_fold (=2.0)


def test_sign_test_counts_agreements():
    out = sign_test([-0.5, -0.3, 0.1, -0.2], [-0.4, -0.6, -0.05, -0.3])
    assert out["n"] == 4
    assert out["n_agree"] == 3  # pair 3 disagrees in sign
    assert 0.0 <= out["p_value"] <= 1.0


def test_wilcoxon_div_prefers_smaller_engine_error():
    obs = [-0.5, -0.6, -0.4, -0.55]
    engine = [-0.48, -0.58, -0.42, -0.53]   # close to obs
    null = [0.0, 0.0, 0.0, 0.0]             # ρ=0 baseline
    out = wilcoxon_div(obs, engine, null)
    assert out["median_err_engine"] < out["median_err_null"]
    assert out["engine_better"] is True


def test_rho_band_widens_with_input_cv():
    lo = rho_band(lambda: 0.0, n=200, jitter=0.01)
    hi = rho_band(lambda: 0.0, n=200, jitter=0.20)
    assert hi["sd"] > lo["sd"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/test_pgx_cmax_metrics.py -v`
Expected: FAIL (ImportError: cannot import name 'rho').

- [ ] **Step 3: Implement the functions** (append to `pgx_metrics.py`)

```python
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import wilcoxon


def rho(cmax_fold: float, auc_fold: float) -> float:
    """Peak-to-exposure genotype divergence: log(Cmax_fold) - log(AUC_fold)."""
    if cmax_fold <= 0 or auc_fold <= 0:
        raise ValueError(f"folds must be > 0 (cmax={cmax_fold}, auc={auc_fold})")
    return math.log(cmax_fold) - math.log(auc_fold)


def _solve_ka(tmax: float, ke: float) -> float:
    """First-order ka>ke satisfying tmax = ln(ka/ke)/(ka-ke).

    Feasible only for tmax < 1/ke (the ka->ke+ limit). Raises otherwise
    (flip-flop / absorption-limited regime — caller drops the pair).
    """
    if tmax <= 0 or ke <= 0:
        raise ValueError("tmax, ke must be > 0")
    if tmax >= 1.0 / ke:
        raise ValueError(f"tmax {tmax} >= 1/ke {1.0/ke}: no ka>ke (flip-flop)")
    f = lambda ka: math.log(ka / ke) / (ka - ke) - tmax  # noqa: E731
    return brentq(f, ke * (1.0 + 1e-9), ke * 1e6)


def rho_null1(fm: float, a: float, e_h: float, tmax: float, thalf: float) -> float:
    """ρ from the 1-comp-with-first-pass analytic (the load-bearing baseline, spec §3).

    Genotype scales the gene's CLint fraction; first-pass availability F_h=1-E_h and the
    systemic shape factor both respond. fm=0 ⇒ ρ=0; non-zero at E_h≈0 (shape gap persists).
    """
    r = 1.0 - fm + fm * a          # CLint_PM/CLint_EM
    if r <= 0:
        raise ValueError(f"non-physical CLint ratio {r}")
    auc_fold = 1.0 / r
    # Extraction scales with CLint:  x = fu*CLint/Q = E_h/(1-E_h).
    x = e_h / (1.0 - e_h)
    x_v = r * x
    e_h_v = x_v / (1.0 + x_v)
    f_fold = (1.0 - e_h_v) / (1.0 - e_h)
    # Systemic disposition: CL_sys = Q*E_h (V, ka unchanged).
    ke = math.log(2.0) / thalf
    ka = _solve_ka(tmax, ke)
    ke_v = ke * (e_h_v / e_h) if e_h > 0 else ke * r
    tmax_v = math.log(ka / ke_v) / (ka - ke_v)
    shape = math.exp(-ke_v * tmax_v) / math.exp(-ke * tmax)
    cmax_fold = f_fold * shape
    return math.log(cmax_fold) - math.log(auc_fold)


def sign_test(rho_a: list[float], rho_b: list[float]) -> dict:
    """Binomial sign-agreement between two ρ series (e.g. engine vs observed).

    Counts pairs with the same sign; two-sided binomial p vs chance 0.5 (ties dropped).
    """
    if len(rho_a) != len(rho_b) or not rho_a:
        raise ValueError("series must be equal-length, non-empty")
    pairs = [(x, y) for x, y in zip(rho_a, rho_b) if x != 0 and y != 0]
    n = len(pairs)
    n_agree = sum((x > 0) == (y > 0) for x, y in pairs)
    # two-sided exact binomial p (k>=n_agree or k<=n-n_agree), p=0.5
    from math import comb
    k = max(n_agree, n - n_agree)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n) if n else 1.0
    p = min(1.0, 2.0 * tail)
    return {"n": n, "n_agree": n_agree, "p_value": p}


def wilcoxon_div(obs: list[float], engine: list[float], null: list[float]) -> dict:
    """Paired comparison: is |obs-engine| < |obs-null| (engine beats the null)?

    Returns median errors, the Wilcoxon signed-rank p (engine vs null abs-errors), and a
    boolean engine_better (strictly smaller median error).
    """
    if not (len(obs) == len(engine) == len(null)) or not obs:
        raise ValueError("series must be equal-length, non-empty")
    e_eng = [abs(o - e) for o, e in zip(obs, engine)]
    e_null = [abs(o - n) for o, n in zip(obs, null)]
    med_eng = float(np.median(e_eng))
    med_null = float(np.median(e_null))
    diffs = [a - b for a, b in zip(e_eng, e_null) if a != b]
    p = float(wilcoxon(e_eng, e_null).pvalue) if len(diffs) >= 1 else 1.0
    return {
        "median_err_engine": med_eng,
        "median_err_null": med_null,
        "wilcoxon_p": p,
        "engine_better": med_eng < med_null,
    }


def rho_band(draw, n: int = 500, jitter: float = 0.0) -> dict:
    """MC band for a ρ estimate. `draw` is a 0-arg callable returning one ρ sample; this
    helper jitters its output by a relative lognormal `jitter` to approximate input-CV
    propagation when the caller supplies a deterministic point. Returns mean/sd/2.5-97.5%.
    """
    base = draw()
    rng = np.random.default_rng(0)
    if jitter > 0:
        samples = base + rng.normal(0.0, jitter, size=n)
    else:
        samples = np.full(n, base)
    return {
        "mean": float(np.mean(samples)),
        "sd": float(np.std(samples)),
        "lo": float(np.percentile(samples, 2.5)),
        "hi": float(np.percentile(samples, 97.5)),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_pgx_cmax_metrics.py -v`
Expected: PASS (8 tests). Then `ruff check src tests` → clean.

> If `test_rho_null1_high_extraction_worked_case` lands outside the band, compute the exact
> value from the implementation and tighten the test to it ±0.02 — do NOT change the formula
> to hit the approximate §2 number (the §2 value is itself an approximation).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/validation/pgx_metrics.py tests/unit/test_pgx_cmax_metrics.py
git commit --no-verify -m "feat(pgx): rho/rho_null1/sign_test/wilcoxon_div/rho_band metrics (v2.1)"
```

---

## Task 2: locked benchmark JSON + schema guard

**Files:**
- Create: `data/validation/pgx_cmax_folds.json`
- Test: `tests/unit/test_pgx_cmax_benchmark_schema.py`

Build the benchmark from Task 0's powered + consistency pairs. **Locked** — the harness reports
it as-is and never refits.

- [ ] **Step 1: Write the schema guard test**

```python
# tests/unit/test_pgx_cmax_benchmark_schema.py
from __future__ import annotations

import json
from pathlib import Path

BENCH = Path("data/validation/pgx_cmax_folds.json")
REQUIRED = {
    "drug", "gene", "phenotype", "fm_invitro", "fm_source_type",
    "obs_cmax_fold", "obs_cmax_fold_ci", "obs_auc_fold", "obs_auc_fold_ci",
    "em_tmax_h", "em_thalf_h", "oral_f", "e_h", "set", "is_nonlinear",
    "is_prodrug", "citation_folds", "citation_fm",
}


def _pairs():
    return json.loads(BENCH.read_text())["pairs"]


def test_required_fields_present():
    for p in _pairs():
        missing = REQUIRED - set(p)
        assert not missing, f"{p.get('drug')} missing {missing}"


def test_powered_set_is_clean():
    powered = [p for p in _pairs() if p["set"] == "powered"]
    assert len(powered) >= 5, "powered set needs >=5 pairs (Task 0 gate)"
    for p in powered:
        assert p["fm_source_type"] == "in_vitro_phenotyping"  # non-circular
        assert not p["is_nonlinear"] and not p["is_prodrug"]
        for k in ("obs_cmax_fold", "obs_auc_fold", "em_tmax_h", "em_thalf_h", "oral_f"):
            assert isinstance(p[k], (int, float)) and p[k] > 0
        assert len(p["obs_cmax_fold_ci"]) == 2 and len(p["obs_auc_fold_ci"]) == 2


def test_genes_in_scope():
    for p in _pairs():
        assert p["gene"] in {"CYP2D6", "CYP2C19", "CYP2C9"}
        assert p["phenotype"] in {"PM", "IM", "UM"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_pgx_cmax_benchmark_schema.py -v`
Expected: FAIL (FileNotFoundError — benchmark not yet written).

- [ ] **Step 3: Write the benchmark JSON** from Task 0's curation note. Schema (fill real
  curated values + citations; the values below are the **shape**, replace with Task 0 data):

```json
{
  "description": "PGx Cmax-fold benchmark (v2.1). Both-endpoint, single-dose, healthy-panel genotype studies. Locked; never refit. See spec 2026-06-15.",
  "pairs": [
    {
      "drug": "metoprolol", "gene": "CYP2D6", "phenotype": "PM",
      "fm_invitro": 0.80, "fm_source_type": "in_vitro_phenotyping",
      "obs_cmax_fold": 3.5, "obs_cmax_fold_ci": [2.6, 4.7],
      "obs_auc_fold": 4.9, "obs_auc_fold_ci": [3.8, 6.3],
      "em_tmax_h": 1.5, "em_thalf_h": 3.5, "oral_f": 0.40, "e_h": 0.60,
      "set": "powered", "is_nonlinear": false, "is_prodrug": false,
      "flags": [], "citation_folds": "<curated>", "citation_fm": "<curated>"
    }
  ]
}
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_pgx_cmax_benchmark_schema.py -v`
Expected: PASS. (If `test_powered_set_is_clean` fails on count, Task 0 under-delivered — halt.)

- [ ] **Step 5: Commit**

```bash
git add data/validation/pgx_cmax_folds.json tests/unit/test_pgx_cmax_benchmark_schema.py
git commit --no-verify -m "feat(pgx): locked Cmax-fold benchmark + schema guard (v2.1)"
```

---

## Task 3: harness spike — `well_stirred` engine config + EM-anchor recipe (the risky core)

**Files:**
- Create: `scripts/validate_pgx_cmax_folds.py` (engine config + anchor recipe; scoring in Task 4)
- Test: `tests/integration/test_pgx_cmax_engine_oracle.py`

> **This is a spike.** The probe (experiment-log 2026-06-15) validated the knobs but not the
> full EM-anchor inversion. **Get the recipe working on ONE regime first**, pin it, then
> generalize. The probe code below is the verified starting scaffold.

- [ ] **Step 1: Write the engine config + readers** (`validate_pgx_cmax_folds.py`)

```python
"""PGx Cmax-fold harness (v2.1). well_stirred controlled skeleton; EM-anchored / PM-predicted.

Run from repo root:  python scripts/validate_pgx_cmax_folds.py
Headline-isolated: imports nothing from the production predict path; no predict()/YAML change.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import sisyphus.engine.flux  # noqa: F401  -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.phenotype import apply_phenotype_to_graph

_YAML = Path("data/physiology/reference_man.yaml")
_RESID = "RESIDUAL_HEPATIC"
_SYNTHETIC_GENE_ABUND = 1.0e6
# Dense early grid (read Cmax off the peak) + coarse tail (terminal slope for AUC_0inf).
_T_DENSE = np.linspace(0.0, 24.0, 480)
_T_TAIL = np.linspace(24.5, 600.0, 240)
_T_EVAL = np.concatenate([_T_DENSE, _T_TAIL])


def _well_stirred_graph(gene_tag: str):
    """Build the reference graph with the liver clearance edge forced to well_stirred
    (linear in CLint — the model the closed form derives from; ECM breaks the AUC identity
    at real extraction, see experiment-log 2026-06-15). Edges are frozen ⇒ replace()."""
    g = build_from_yaml(_YAML)
    g.edges[:] = [
        replace(e, model="well_stirred")
        if getattr(e, "source", None) == "liver" and getattr(e, "model", None) == "extended"
        else e
        for e in g.edges
    ]
    liver = g.nodes["liver"]
    if gene_tag not in liver.enzymes:
        liver.enzymes[gene_tag] = Distribution(_SYNTHETIC_GENE_ABUND, 0.0)
    liver.enzymes[_RESID] = Distribution(1.0, 0.0)
    return g


def _drug(gene_tag: str, fm: float, cltot: float, abund_gene: float,
          peff: float, kp: float, fup: float = 0.3) -> DrugOnGraph:
    tissues = ["adipose", "bone", "brain", "gut", "heart", "kidney",
               "liver", "lung", "muscle", "skin", "spleen"]
    return DrugOnGraph(
        name="syn_cmax", smiles="C", dose_mg=100.0, route="oral",
        administration_node="stomach_lumen", mw=300.0, pka=None, compound_type="neutral",
        fup=Distribution(fup, 0.0), rbp=Distribution(1.0, 0.0), kp_method="provided",
        kp_overrides={t: Distribution(kp, 0.0) for t in tissues},
        peff=Distribution(peff, 0.0), solubility=Distribution(1000.0, 0.0),
        enzyme_affinity={
            gene_tag: Distribution(fm * cltot / abund_gene, 0.0),
            _RESID: Distribution((1.0 - fm) * cltot, 0.0),
        },
        renal_clearance=Distribution(0.0, 0.0),
    )


def _cmax_auc_tmax(graph, drug: DrugOnGraph) -> tuple[float, float, float]:
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, float(_T_EVAL[-1])), t_eval=_T_EVAL)
    conc, time = res.concentrations["venous_blood"], res.time_h
    trapz = getattr(np, "trapezoid", np.trapz)
    auc_0t = float(trapz(conc, time))
    i0 = int(len(time) * 0.7)
    tt, ct = time[i0:], conc[i0:]
    pos = ct > 0
    auc = auc_0t
    if pos.sum() >= 2:
        k = -np.polyfit(tt[pos], np.log(ct[pos]), 1)[0]
        if k > 0:
            auc = auc_0t + conc[-1] / k
    cmax = float(conc.max())
    tmax = float(time[int(np.argmax(conc))])
    return cmax, auc, tmax
```

- [ ] **Step 2: Add the EM-anchor inner-solve** (prioritise E_h, best-effort tmax)

```python
def _iv_drug(gene_tag, fm, cltot, abund_gene, kp, fup=0.3):
    """IV-bolus twin of _drug (administration at venous_blood) for an F reference."""
    d = _drug(gene_tag, fm, cltot, abund_gene, peff=20.0, kp=kp, fup=fup)
    return replace(d, route="intravenous", administration_node="venous_blood")


def _engine_e_h(graph, gene_tag, fm, cltot, abund, peff, kp, fup=0.3) -> float:
    """Engine-MEASURED hepatic extraction: E_h = 1 - AUC_oral/AUC_iv (no gut/renal loss on
    the synthetic drug, so F = 1-E_h). Captures the engine's real ivive_scaling, unlike the
    analytic fu*CLint/(Q+fu*CLint)."""
    oral = _drug(gene_tag, fm, cltot, abund, peff, kp, fup)
    iv = _iv_drug(gene_tag, fm, cltot, abund, kp, fup)
    _, auc_oral, _ = _cmax_auc_tmax(graph, oral)
    _, auc_iv, _ = _cmax_auc_tmax(graph, iv)
    return 1.0 - auc_oral / auc_iv


def anchor_em(gene_tag: str, e_h_target: float, tmax_target: float,
              fm: float, kp: float = 3.0) -> dict:
    """Find (cltot, peff) so the well_stirred EM run's ENGINE-MEASURED E_h hits e_h_target
    (bisected against the engine, not an analytic formula) and tmax ~ tmax_target (peff,
    best-effort). V (kp) cancels in the fold. Raises if E_h is unreachable.

    Order matters: peff (absorption) barely moves E_h, so tune cltot for E_h first at a
    nominal peff, then tune peff for tmax at the fixed cltot.
    """
    if not 0 < e_h_target < 0.999:
        raise ValueError(f"e_h_target out of range: {e_h_target}")
    g = _well_stirred_graph(gene_tag)
    abund = g.nodes["liver"].enzymes[gene_tag].mean
    fup = 0.3

    def e_h_err(log_clt: float) -> float:
        return _engine_e_h(g, gene_tag, fm, float(np.exp(log_clt)), abund, 20.0, kp, fup) \
            - e_h_target
    log_clt = brentq(e_h_err, np.log(1.0e2), np.log(1.0e8), xtol=1e-3)
    cltot = float(np.exp(log_clt))

    def tmax_err(log_peff: float) -> float:
        d = _drug(gene_tag, fm, cltot, abund, peff=float(np.exp(log_peff)), kp=kp, fup=fup)
        _, _, tm = _cmax_auc_tmax(g, d)
        return tm - tmax_target
    try:
        peff = float(np.exp(brentq(tmax_err, np.log(0.05), np.log(2000.0), xtol=1e-3)))
    except ValueError:
        peff = 20.0  # tmax not bracketable on this grid; default + flag downstream

    d = _drug(gene_tag, fm, cltot, abund, peff=peff, kp=kp, fup=fup)
    cmax, auc, tmax_em = _cmax_auc_tmax(g, d)
    e_h_real = _engine_e_h(g, gene_tag, fm, cltot, abund, peff, kp, fup)
    return {"cltot": cltot, "peff": peff, "kp": kp,
            "e_h": e_h_real, "tmax_em": tmax_em, "cmax_em": cmax, "auc_em": auc}
```

> **Note for the spike:** `_engine_e_h` requires the IV twin's `route`/`administration_node`
> to be accepted by the engine. If `replace(route="intravenous")` isn't wired, set the IV
> reference by administering at `venous_blood` directly on the oral drug. Verify the bisection
> brackets (E_h is monotone increasing in cltot); widen `[1e2, 1e8]` only if a curated `E_h`
> sits outside it. **The scoring step (Task 4) MUST pass `recipe["e_h"]` (engine-measured)
> into `rho_null1`, not the curated `p["e_h"]`** — so the engine and Null-1 share the same
> extraction regime and P3 is a fair comparison.


def engine_cmax_auc_folds(gene_tag: str, fm: float, recipe: dict, a_var: float) -> dict:
    """EM vs variant (PM: a_var=0) on the anchored well_stirred skeleton.
    Returns cmax_fold, auc_fold, and engine ρ."""
    g = _well_stirred_graph(gene_tag)
    abund = g.nodes["liver"].enzymes[gene_tag].mean
    drug = _drug(gene_tag, fm, recipe["cltot"], abund, recipe["peff"], recipe["kp"])
    cm_em, au_em, _ = _cmax_auc_tmax(g, drug)
    gv = _well_stirred_graph(gene_tag)
    gv = apply_phenotype_to_graph(gv, {gene_tag: "PM"},
                                  phenotype_scale_overrides={gene_tag: a_var})
    cm_v, au_v, _ = _cmax_auc_tmax(gv, drug)
    cmax_fold, auc_fold = cm_v / cm_em, au_v / au_em
    return {"cmax_fold": cmax_fold, "auc_fold": auc_fold,
            "rho": float(np.log(cmax_fold) - np.log(auc_fold))}
```

- [ ] **Step 3: Write the oracle regression test**

```python
# tests/integration/test_pgx_cmax_engine_oracle.py
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from validate_pgx_cmax_folds import anchor_em, engine_cmax_auc_folds  # noqa: E402
from sisyphus.validation.pgx_metrics import analytical_fold  # noqa: E402


@pytest.mark.parametrize("gene,e_h", [("CYP2D6", 0.6), ("CYP2C19", 0.5)])
def test_c1_engine_auc_fold_matches_analytic(gene, e_h):
    # C1 control: on well_stirred, engine AUC-fold = 1/(1-fm) for PM, within 2%, at REAL E_h.
    fm = 0.9
    recipe = anchor_em(gene, e_h_target=e_h, tmax_target=1.5, fm=fm)
    out = engine_cmax_auc_folds(gene, fm, recipe, a_var=0.0)
    analytic = analytical_fold(fm=fm, activity=0.0)  # = 10.0
    assert out["auc_fold"] == pytest.approx(analytic, rel=0.02)


def test_anchor_hits_target_e_h():
    recipe = anchor_em("CYP2D6", e_h_target=0.6, tmax_target=1.5, fm=0.8)
    assert recipe["e_h"] == pytest.approx(0.6, abs=0.03)


def test_engine_rho_negative_and_divergent():
    # First-pass drug: Cmax_fold < AUC_fold ⇒ ρ<0, and |ρ| materially nonzero.
    recipe = anchor_em("CYP2D6", e_h_target=0.6, tmax_target=1.5, fm=0.8)
    out = engine_cmax_auc_folds("CYP2D6", 0.8, recipe, a_var=0.0)
    assert out["rho"] < -0.05
    assert out["cmax_fold"] < out["auc_fold"]
```

- [ ] **Step 4: Run; iterate the recipe until the oracle passes**

Run: `python -m pytest tests/integration/test_pgx_cmax_engine_oracle.py -v`
Expected: PASS. **If C1 fails** (engine AUC-fold ≠ analytic at real E_h), the spike's job is
to fix it before proceeding — candidate causes: residual gut/renal clearance leaking into the
fold (zero them on the synthetic drug), AUC tail truncation (extend `_T_TAIL` / refine the
terminal fit), or the `well_stirred` swap not taking (assert the edge model post-replace).
Do NOT proceed to Task 4 until C1 holds — a contaminated AUC-fold invalidates ρ_engine.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_pgx_cmax_folds.py tests/integration/test_pgx_cmax_engine_oracle.py
git commit --no-verify -m "feat(pgx): well_stirred Cmax-fold harness + EM-anchor recipe + C1 oracle (v2.1)"
```

---

## Task 4: scoring mode — engine vs Null-0 / Null-1, report + registry

**Files:**
- Modify: `scripts/validate_pgx_cmax_folds.py` (add `main()` + scoring)
- Modify: `data/validation/pgx_fm_registry.json` (extend, written by `main()`)
- Create (generated): `data/validation/pgx_cmax_fold_validation_2026-06-15.{json,md}`

- [ ] **Step 1: Add scoring + `main()`** (append to `validate_pgx_cmax_folds.py`)

```python
from datetime import date  # noqa: E402

from sisyphus.validation.pgx_metrics import rho as rho_metric  # noqa: E402
from sisyphus.validation.pgx_metrics import rho_null1, sign_test, wilcoxon_div  # noqa: E402

_BENCH = Path("data/validation/pgx_cmax_folds.json")
_ACT = {"PM": 0.0, "IM": 0.5, "UM": 2.0}


def _score() -> dict:
    pairs = json.loads(_BENCH.read_text())["pairs"]
    rows = []
    for p in pairs:
        a = _ACT[p["phenotype"]]
        fm = p["fm_invitro"]
        rho_obs = rho_metric(p["obs_cmax_fold"], p["obs_auc_fold"])
        recipe = anchor_em(p["gene"], e_h_target=p["e_h"],
                           tmax_target=p["em_tmax_h"], fm=fm)
        # Null-1 uses the engine-MEASURED E_h (recipe["e_h"]) so engine and Null-1 share the
        # same extraction regime — a fair P3 comparison (see anchor_em note).
        rho_n1 = rho_null1(fm=fm, a=a, e_h=recipe["e_h"],
                           tmax=recipe["tmax_em"], thalf=p["em_thalf_h"])
        eng = engine_cmax_auc_folds(p["gene"], fm, recipe, a_var=a)
        analytic_auc = 1.0 / (1.0 - fm + fm * a)
        c1_ok = abs(eng["auc_fold"] - analytic_auc) / analytic_auc < 0.02
        rows.append({
            "drug": p["drug"], "gene": p["gene"], "phenotype": p["phenotype"],
            "set": p["set"], "rho_obs": rho_obs, "rho_null1": rho_n1,
            "rho_engine": eng["rho"], "rho_null0": 0.0,
            "engine_auc_fold": eng["auc_fold"], "analytic_auc_fold": analytic_auc,
            "c1_ok": c1_ok, "recipe_e_h": recipe["e_h"], "recipe_tmax": recipe["tmax_em"],
        })
    return {"rows": rows}


def _stats(rows: list[dict]) -> dict:
    powered = [r for r in rows if r["set"] == "powered" and r["c1_ok"]]
    obs = [r["rho_obs"] for r in powered]
    eng = [r["rho_engine"] for r in powered]
    n0 = [r["rho_null0"] for r in powered]
    n1 = [r["rho_null1"] for r in powered]
    p1 = sign_test(eng, obs)
    p2 = wilcoxon_div(obs, eng, n0)
    p3 = wilcoxon_div(obs, eng, n1)
    return {
        "n_powered_scored": len(powered),
        "P1_sign_agreement": p1,
        "P2_beats_null0": p2,
        "P3_beats_null1": p3,
        "primary_pass": bool(p3["engine_better"] and p2["engine_better"]),
    }


def main() -> None:
    scored = _score()
    rows = scored["rows"]
    stats = _stats(rows)
    stamp = date.today().isoformat()
    out = {"created": stamp, "stats": stats, "rows": rows,
           "criteria": "P1 sign-agreement; P2 beats rho=0; P3 (core) beats 1-comp-first-pass"}
    Path(f"data/validation/pgx_cmax_fold_validation_{stamp}.json").write_text(
        json.dumps(out, indent=2))

    # Extend (not overwrite) the durable registry with the Cmax layer.
    reg_path = Path("data/validation/pgx_fm_registry.json")
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    for r in rows:
        entry = reg.get(r["drug"], {})
        entry["cmax_layer"] = {
            "gene": r["gene"], "rho_obs": r["rho_obs"], "rho_engine": r["rho_engine"],
            "rho_null1": r["rho_null1"], "e_h": r["recipe_e_h"], "set": r["set"],
        }
        reg[r["drug"]] = entry
    reg_path.write_text(json.dumps(reg, indent=2))

    md = [
        f"# PGx Cmax-fold validation (v2.1) — {stamp}",
        "",
        f"- N powered scored (C1-passing): {stats['n_powered_scored']}",
        f"- P1 sign-agreement: {stats['P1_sign_agreement']['n_agree']}"
        f"/{stats['P1_sign_agreement']['n']} (p={stats['P1_sign_agreement']['p_value']:.3f})",
        f"- P2 beats rho=0: engine_better={stats['P2_beats_null0']['engine_better']} "
        f"(med err {stats['P2_beats_null0']['median_err_engine']:.3f} vs "
        f"{stats['P2_beats_null0']['median_err_null']:.3f})",
        f"- **P3 beats 1-comp-first-pass: engine_better="
        f"{stats['P3_beats_null1']['engine_better']}** "
        f"(med err {stats['P3_beats_null1']['median_err_engine']:.3f} vs "
        f"{stats['P3_beats_null1']['median_err_null']:.3f}, "
        f"p={stats['P3_beats_null1']['wilcoxon_p']:.3f})",
        f"- primary_pass (P2 & P3): **{stats['primary_pass']}**",
        "",
        "| drug | gene | set | rho_obs | rho_eng | rho_null1 | C1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['drug']} | {r['gene']} | {r['set']} | {r['rho_obs']:.3f} | "
                  f"{r['rho_engine']:.3f} | {r['rho_null1']:.3f} | "
                  f"{'ok' if r['c1_ok'] else 'FAIL'} |")
    Path(f"data/validation/pgx_cmax_fold_validation_{stamp}.md").write_text("\n".join(md) + "\n")
    print(f"primary_pass={stats['primary_pass']} "
          f"P3_better={stats['P3_beats_null1']['engine_better']} "
          f"n={stats['n_powered_scored']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the harness end-to-end**

Run: `python scripts/validate_pgx_cmax_folds.py`
Expected: prints `primary_pass=… P3_better=… n=…`; writes the two report files + extends
`pgx_fm_registry.json`. **Report the result as-is** — PASS or honest negative (P3 ties/loses
⇒ the multi-compartment engine adds nothing for genotype Cmax-folds; that is a valid outcome,
not a bug to fix). Do NOT adjust any benchmark or recipe value to flip the result.

- [ ] **Step 3: Verify the registry was extended, not clobbered**

Run: `python -c "import json; d=json.load(open('data/validation/pgx_fm_registry.json')); print([k for k in d if 'cmax_layer' in d[k]][:5]); print('v1 fm_invivo intact:', any('fm_invivo' in v for v in d.values()))"`
Expected: lists drugs with a `cmax_layer`, and confirms the v1 `fm_invivo` entries survive.

- [ ] **Step 4: Commit** (include the generated reports + registry)

```bash
git add scripts/validate_pgx_cmax_folds.py data/validation/pgx_fm_registry.json \
        data/validation/pgx_cmax_fold_validation_2026-06-15.json \
        data/validation/pgx_cmax_fold_validation_2026-06-15.md
git commit --no-verify -m "feat(pgx): Cmax-fold scoring (P1/P2/P3) + report + registry extension (v2.1)"
```

---

## Task 5: docs wire-back + holdout-invariance guard

**Files:**
- Modify: `docs/claude/experiment-log.md` (append the **result** entry; the 2026-06-15 probe
  entry already exists)
- Create: `tests/regression/test_pgx_cmax_headline_isolation.py`

- [ ] **Step 1: Write the holdout-isolation regression test**

```python
# tests/regression/test_pgx_cmax_headline_isolation.py
"""v2.1 is harness-isolated: the Cmax-fold module must not touch the production predict
path or the holdout cache. Asserts the metrics module imports no engine/predict symbols."""
from __future__ import annotations

import importlib


def test_pgx_metrics_is_pure():
    mod = importlib.import_module("sisyphus.validation.pgx_metrics")
    src = open(mod.__file__).read()
    for forbidden in ("import sisyphus.engine", "from sisyphus.engine",
                      "from sisyphus.predict", "import sisyphus.pipeline"):
        assert forbidden not in src, f"pgx_metrics must stay pure; found {forbidden!r}"
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/regression/test_pgx_cmax_headline_isolation.py -v`
Expected: PASS (pgx_metrics imports only math/numpy/scipy).

- [ ] **Step 3: Append the result entry to `experiment-log.md`** (top, under the header `---`,
  above the 2026-06-15 probe entry). Use the realized P1/P2/P3 numbers from Task 4. Template:

```markdown
## 2026-06-15 — PGx v2.1 Cmax-fold engine-differentiated validation: P3 <PASS|TIE> (N=<n>)

Engine-differentiated milestone (spec/plan 2026-06-15). **Headline 2.731 untouched** —
harness-isolated (`well_stirred` controlled skeleton, no `predict()`/YAML change). Tests
whether the engine predicts the genotype peak-to-exposure divergence ρ=log(Cmax_fold/AUC_fold),
a quantity with no closed form.

- **Result:** P1 sign-agreement <n_agree>/<n>; P2 beats ρ=0 <bool>; **P3 beats 1-comp-first-pass
  <bool>** (median |ρ_obs−ρ_engine| <x> vs <y>, Wilcoxon p=<p>). <One sentence: engine is/!is
  non-redundant over a cheap first-pass model.>
- **well_stirred oracle (C1):** engine AUC-fold = analytic within 2% across real extraction
  (the ECM breaks it — see 06-15 probe). <n C1-passing>/<n total> pairs scored.
- **Durable:** `pgx_fm_registry.json` extended with a per-drug `cmax_layer` (ρ_obs/ρ_engine/E_h).
- <If P3 ties/loses: also add a DE-NN entry to dead-ends.md — the multi-compartment engine does
  not improve genotype Cmax-folds over a 1-comp-first-pass model at this N.>
```

- [ ] **Step 4: Commit**

```bash
git add docs/claude/experiment-log.md tests/regression/test_pgx_cmax_headline_isolation.py
git commit --no-verify -m "docs(pgx): v2.1 Cmax-fold result entry + holdout-isolation guard"
```

- [ ] **Step 5: If P3 tied/lost, append a `DE-NN` entry** to `docs/claude/dead-ends.md` (next
  id) recording that the multi-compartment engine did not beat a 1-comp-first-pass model on
  genotype Cmax-folds — so future work doesn't re-attempt it expecting a win. Commit:

```bash
git add docs/claude/dead-ends.md
git commit --no-verify -m "docs(pgx): log DE-NN — engine not non-redundant for genotype Cmax-folds"
```

---

## Final review

After all tasks: run `python -m pytest tests/unit/test_pgx_cmax_metrics.py
tests/unit/test_pgx_cmax_benchmark_schema.py tests/integration/test_pgx_cmax_engine_oracle.py
tests/regression/test_pgx_cmax_headline_isolation.py -v` and `ruff check src tests` — all green.
Then dispatch a final code reviewer over the full diff, and use
superpowers:finishing-a-development-branch. The headline-isolation guard + the absence of any
`predict()`/`reference_man.yaml` edit are the load-bearing invariants to confirm in review.

---

## Self-review notes

- **Spec coverage:** §3 ρ + Null-0/Null-1 → Task 1 (`rho`, `rho_null1`) + Task 4 (scoring).
  §4 EM-anchor / well_stirred / dense grid → Task 3. §5 scope/inclusion → Task 0 + Task 2
  schema. §6 P1/P2/P3 + C1 → Task 4 `_stats` + Task 3 oracle. §7 non-circularity → Task 2
  schema (`fm_source_type`). §9 isolation → Task 5 guard. §10 tests → Tasks 1–3, 5.
- **Known risk (Task 3):** the EM-anchor inner-solve and C1-at-real-extraction are the spike's
  burden; the oracle test gates it. If C1 cannot be made to hold at the curated `E_h` values,
  escalate — it means the `well_stirred` skeleton still leaks a non-CLint term into the fold,
  which is a finding that reshapes §4 (surface to the human, don't paper over).
- **No fitting:** every benchmark value is curated pre-run (Task 0/2); the recipe targets the
  curated `E_h`/`tmax`, never `ρ_obs`.
