# PGx v2.2b — Nonlinear Saturable Genotype-Fold Validation Implementation Plan

> **⚠ SUPERSEDED (2026-06-16) — halted at Task 1 (the feasibility spike).** Single-dose
> saturation is not robustly engaged (only propafenone@low-`Km`, a cherry-pick). Pivoted to a
> multi-dose/steady-state milestone. See the spec banner +
> `data/validation/pgx_cmax_v2b_spike_2026-06-16.md` + experiment-log 2026-06-16. Tasks 2–6
> below were never executed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that the engine's saturable metabolism (v2.2a `enzyme_km`) reproduces the
observed dose-dependent genotype Cmax/AUC folds of nonlinear drugs better than the same engine
with saturation off.

**Architecture:** A controlled `well_stirred` EM-anchored skeleton (the gene fraction carries a
literature `Km` → saturable; residual stays linear). Per drug per dose, run three engines —
EM-saturable, PM-saturable, and a linear null (`Km=∞`, re-anchored) — and compare predicted
folds to observed. **The first task is a gating feasibility spike** on propafenone; if saturation
is not engaged at therapeutic dose, HALT.

**Tech Stack:** Python 3.10, numpy, scipy (`optimize.brentq`/`brenth`, `stats.wilcoxon`), the
Sisyphus engine (now MM-capable via v2.2a). **Spec:**
`docs/superpowers/specs/2026-06-15-pgx-cmax-v2b-nonlinear-genotype-design.md` (read §2, §3, §5,
§7 before Task 1).

**Reuse / starting scaffold:** v2.1's Cmax-harness code was authored in
`docs/superpowers/plans/2026-06-15-pgx-cmax-fold-engine-differentiated.md` (Task 3) but **never
executed** (v2.1 halted at its gate). That file's `_well_stirred_graph`, `_drug`,
`_cmax_auc_tmax` (dense `t_eval`), `_engine_e_h`, and `anchor_em` are the **base linear**
scaffold — port them, then add saturation. v2.2a shipped `DrugOnGraph.enzyme_km` + the saturable
`well_stirred` flux.

**Conventions (every task):** commit as `jam-sudo`, NO Claude/AI trailer, `git commit
--no-verify`; tests via `python -m pytest`; ruff line-length 100 (`ruff check src tests` — CI
runs it repo-wide). `/opt/miniconda3/bin/python` has engine+scipy. Stage only the files each task
names; never `git add README.md` or untracked workspace files. **Benchmark is locked — never
refit `Km` to improve a fold.** Headline 2.731 must stay bit-identical (harness-isolated; the
saturable flux is bit-identical when `enzyme_km` is empty — proven in v2.2a).

---

## File Structure

- `scripts/validate_pgx_cmax_v2b.py` — **new**: the three-engine saturable harness (base linear
  scaffold ported from the v2.1 plan + saturable anchor + enzyme_km on the gene) + scoring + report.
- `src/sisyphus/validation/pgx_metrics.py` — **extend**: `km_uM_to_unbound_mgL`, the saturable
  EM-anchor solver, the dose-dependence statistic (pure; numpy/scipy only).
- `data/validation/pgx_cmax_v2b_folds.json` — **new**: locked benchmark.
- `tests/unit/test_pgx_v2b_metrics.py` — **new**: unit tests for the pure functions.
- `tests/unit/test_pgx_v2b_benchmark_schema.py` — **new**: schema/curation guard.
- `tests/integration/test_pgx_v2b_harness.py` — **new**: spike checks as regression pins.
- `tests/regression/test_pgx_v2b_headline_isolation.py` — **new**: holdout isolation.
- `data/validation/pgx_cmax_v2b_validation_2026-06-16.{json,md}`, `pgx_fm_registry.json` —
  **generated/extended** by the harness.
- `docs/claude/experiment-log.md` — **append** the result entry.

---

## Task 1: Feasibility spike on propafenone (GATING — HALT if saturation not engaged)

**Files:**
- Create: `scripts/validate_pgx_cmax_v2b.py` (prototype: base harness + saturable anchor)
- Create: `data/validation/pgx_cmax_v2b_spike_2026-06-16.md` (spike report — verdict + numbers)

> This is a SPIKE. Build the minimum to answer the four §7 gating questions on ONE drug
> (propafenone). Iterate freely. Do NOT build the full benchmark/scoring yet.

- [ ] **Step 1: Port the base linear Cmax harness.** From
  `docs/superpowers/plans/2026-06-15-pgx-cmax-fold-engine-differentiated.md` (Task 3, Steps 1–2),
  copy into `scripts/validate_pgx_cmax_v2b.py`: `_well_stirred_graph(gene_tag)` (liver edge →
  `well_stirred` via `dataclasses.replace`; inject synthetic gene abundance for CYP2C19/absent
  genes; `RESIDUAL_HEPATIC` at abundance 1.0), `_drug(...)` (synthetic `DrugOnGraph`, oral),
  `_cmax_auc_tmax(graph, drug)` (dense `_T_EVAL = concat(linspace(0,24,480), linspace(24.5,600,240))`,
  AUC_0inf with terminal-slope tail, returns `(cmax, auc, tmax)`), `_iv_drug`/`_engine_e_h`
  (oral-vs-IV E_h), and `anchor_em(gene_tag, e_h_target, tmax_target, fm, kp)` (bisect `cltot`
  against engine-measured E_h, then `peff` for `tmax`). Confirm it imports and runs on a linear
  synthetic propafenone-like drug (no `enzyme_km`).

- [ ] **Step 2: Add the saturable drug builder + saturable anchor (prototype).** Propafenone
  reference: CYP2D6 `fm≈0.8`, `Km≈5.3 µM`, MW 341.4, dose 300 mg, oral F≈0.10 (EM, high
  first-pass), `tmax≈2.5 h`, `t½≈5.5 h`. Convert `Km`: `Km_mgL = 5.3e-6 * 341.4 * 1e3 * fu_mic`;
  use `fu_mic≈0.5` as the spike default (sensitivity later) → `Km_mgL ≈ 9.0e-4` (record the exact
  value you compute).

```python
def _sat_drug(gene_tag, fm, cltot, abund_gene, peff, kp, km_mgL, dose, fup=0.3):
    """Synthetic drug with the GENE fraction saturable (enzyme_km) and residual linear."""
    d = _drug(gene_tag, fm, cltot, abund_gene, peff, kp, fup=fup)  # base builder
    from dataclasses import replace
    from sisyphus.core import Distribution
    return replace(d, dose_mg=dose,
                   enzyme_km={gene_tag: Distribution(km_mgL, 0.0)})  # residual tag: NO km ⇒ linear
```

  **Saturable EM-anchor (prototype):** with `Km` fixed, find `cltot` so the SATURABLE EM run at
  the reference dose reproduces the target EM `E_h` (engine-measured, as the linear anchor) — i.e.
  re-bisect `cltot` using `_sat_drug` instead of `_drug`. The intrinsic `fm` split is preserved
  by the `fm`/(1−fm) abundance×affinity construction (gene gets `fm·cltot`, residual `(1−fm)·cltot`).
  Verify the bisection converges (the saturable EM E_h is monotone in `cltot`).

- [ ] **Step 3: The four gating checks (propafenone, dose 300 mg unless noted).**

```python
# (1) anchor converges: saturable anchor returns finite cltot/peff, EM run reproduces target E_h.
# (2) saturation ENGAGED: at the anchored skeleton, compute the PM/EM fold with the SATURABLE
#     gene vs with the LINEAR gene (km=inf, re-anchored). They must differ MATERIALLY.
# (3) dose-dependence: saturable AUC-fold at 300 mg vs 400 mg must DIFFER (fold shrinks with dose).
# (4) oracle: with km=inf the engine AUC-fold == analytic 1/(1-fm) within 2% (v2.2a/v2.1 C2).
```

  Compute and PRINT, for propafenone at the anchored skeleton:
  - `fold_sat_300` (Cmax & AUC), `fold_lin_300` (linear null, re-anchored), `fold_sat_400`.
  - **Gate (2) PASS** if `|log(fold_sat_300_AUC) − log(fold_lin_300_AUC)| > 0.10` (saturation
    moves the AUC-fold by >~10%); **Gate (3) PASS** if `|log(fold_sat_300_AUC) −
    log(fold_sat_400_AUC)| > 0.03` (dose moves it); **Gate (4) PASS** if linear AUC-fold matches
    analytic within 2%.
  - If `_engine_e_h` for propafenone's curated `E_h` (high first-pass) is unreachable on the
    bisection bracket, widen it; if the saturable conc never approaches `Km` (gate 2 fails),
    that is the HALT signal.

- [ ] **Step 4: Write the spike report + decide.** Write
  `data/validation/pgx_cmax_v2b_spike_2026-06-16.md` with the four checks' numbers and a
  PASS/HALT verdict. **If gate (2) [saturation engaged] FAILS, STOP — report to the controller
  that the engine does not reach liver `C_u ~ Km` at therapeutic dose, which reshapes the
  milestone (e.g. needs the first-pass liver-conc examined, or the powered set reconsidered).**
  Do NOT proceed to Task 2 on a HALT.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_pgx_cmax_v2b.py data/validation/pgx_cmax_v2b_spike_2026-06-16.md
git commit --no-verify -m "spike(pgx): v2.2b saturable harness feasibility on propafenone"
```

> **Controller note:** treat Task 1 like v1/v2.1's gate. Read the spike report; only proceed if
> all four gates pass (esp. gate 2). A HALT here is a legitimate, valuable outcome.

---

## Task 2: pure metrics — `Km` conversion, saturable-anchor solver, dose-dependence stat

**Files:**
- Modify: `src/sisyphus/validation/pgx_metrics.py`
- Test: `tests/unit/test_pgx_v2b_metrics.py`

Extract the spike's validated logic into pure, tested functions.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_pgx_v2b_metrics.py
from __future__ import annotations

import math

import pytest

from sisyphus.validation.pgx_metrics import (
    km_uM_to_unbound_mgL,
    dose_dependence,
)


def test_km_conversion_worked_example():
    # propafenone Km 5.3 µM, MW 341.4, fu_mic 0.5:
    #   5.3 µmol/L * 341.4 g/mol = 1809.4 µg/L = 1.8094 mg/L total; * 0.5 = 0.9047 mg/L unbound.
    val = km_uM_to_unbound_mgL(km_uM=5.3, mw=341.4, fu_mic=0.5)
    assert val == pytest.approx(0.9047, rel=1e-3)


def test_km_conversion_fu_mic_one_is_total():
    assert km_uM_to_unbound_mgL(km_uM=10.0, mw=200.0, fu_mic=1.0) == pytest.approx(2.0, rel=1e-9)


def test_km_conversion_rejects_bad_inputs():
    with pytest.raises(ValueError):
        km_uM_to_unbound_mgL(km_uM=-1.0, mw=200.0, fu_mic=0.5)
    with pytest.raises(ValueError):
        km_uM_to_unbound_mgL(km_uM=5.0, mw=200.0, fu_mic=1.5)


def test_dose_dependence_flat_for_equal_folds():
    # no dose-dependence ⇒ delta 0
    out = dose_dependence(folds=[5.0, 5.0], doses=[300.0, 400.0])
    assert out["delta_log"] == pytest.approx(0.0)


def test_dose_dependence_shrinking_fold_is_negative():
    out = dose_dependence(folds=[5.0, 4.0], doses=[300.0, 400.0])
    assert out["delta_log"] == pytest.approx(math.log(4.0) - math.log(5.0))
    assert out["shrinks_with_dose"] is True
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/unit/test_pgx_v2b_metrics.py -v`
  (ImportError).

- [ ] **Step 3: Implement** (append to `pgx_metrics.py`)

```python
def km_uM_to_unbound_mgL(km_uM: float, mw: float, fu_mic: float) -> float:
    """Convert a literature Michaelis Km to the engine's unbound mg/L basis.

    km_uM: total (microsomal) Km in µmol/L. mw: g/mol. fu_mic: microsomal unbound fraction
    (total→unbound at the enzyme; well_stirred unbound-at-enzyme ≈ unbound plasma). Use
    fu_mic=1.0 when the literature Km is already unbound.
        Km[mg/L] = km_uM(µmol/L) * mw(g/mol) / 1000 * fu_mic
    (µmol/L * g/mol = µg/L; /1000 ⇒ mg/L.)
    """
    if km_uM <= 0 or mw <= 0:
        raise ValueError(f"km_uM, mw must be > 0 (got {km_uM}, {mw})")
    if not 0 < fu_mic <= 1.0:
        raise ValueError(f"fu_mic must be in (0, 1], got {fu_mic}")
    return km_uM * mw / 1000.0 * fu_mic


def dose_dependence(folds: list[float], doses: list[float]) -> dict:
    """Dose-dependence of a genotype fold. folds/doses are paired, dose-ascending.

    Returns delta_log = log(fold_high) - log(fold_low) (negative ⇒ fold shrinks with dose, the
    saturation signature) and shrinks_with_dose.
    """
    if len(folds) != len(doses) or len(folds) < 2:
        raise ValueError("need >=2 paired (fold, dose) points")
    order = sorted(range(len(doses)), key=lambda i: doses[i])
    lo, hi = order[0], order[-1]
    if folds[lo] <= 0 or folds[hi] <= 0:
        raise ValueError("folds must be > 0")
    import math
    delta = math.log(folds[hi]) - math.log(folds[lo])
    return {"delta_log": delta, "shrinks_with_dose": delta < 0.0}
```

  The **saturable EM-anchor solver** validated in the spike: lift it from the harness into a pure
  function `saturable_anchor(...)` ONLY if it can be made engine-free; the spike's version calls
  the engine, so keep it in the harness (`validate_pgx_cmax_v2b.py`) and unit-test it at the
  integration layer (Task 4). `pgx_metrics` stays engine-free.

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/unit/test_pgx_v2b_metrics.py -v`
  (5 pass). `ruff check src tests` clean.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/validation/pgx_metrics.py tests/unit/test_pgx_v2b_metrics.py
git commit --no-verify -m "feat(pgx): km_uM_to_unbound_mgL + dose_dependence metrics (v2.2b)"
```

---

## Task 3: locked benchmark + schema guard

**Files:**
- Create: `data/validation/pgx_cmax_v2b_folds.json`
- Test: `tests/unit/test_pgx_v2b_benchmark_schema.py`

- [ ] **Step 1: Write the schema guard test**

```python
# tests/unit/test_pgx_v2b_benchmark_schema.py
from __future__ import annotations

import json
from pathlib import Path

BENCH = Path("data/validation/pgx_cmax_v2b_folds.json")
REQUIRED = {
    "drug", "gene", "set", "is_nonlinear", "is_mbi", "km_uM", "km_basis",
    "fu_mic", "mw", "fm_invitro", "fm_source_type", "oral_f", "em_tmax_h",
    "em_thalf_h", "doses", "citation_km", "citation_folds",
}


def _rows():
    return json.loads(BENCH.read_text())["drugs"]


def test_required_fields_present():
    for r in _rows():
        assert not (REQUIRED - set(r)), f"{r.get('drug')} missing {REQUIRED - set(r)}"


def test_powered_rows_have_literature_km_and_folds():
    powered = [r for r in _rows() if r["set"] == "powered"]
    assert len(powered) >= 3
    for r in powered:
        assert isinstance(r["km_uM"], (int, float)) and r["km_uM"] > 0  # no "Km not found"
        assert r["fm_source_type"] == "in_vitro_phenotyping"            # non-circular
        assert not r["is_mbi"]                                          # MBI excluded (omeprazole)
        for d in r["doses"]:
            for k in ("dose_mg", "obs_cmax_fold", "obs_auc_fold",
                      "obs_cmax_fold_ci", "obs_auc_fold_ci"):
                assert k in d, f"{r['drug']} dose row missing {k}"


def test_dose_ranging_pairs_have_two_doses():
    rows = {r["drug"]: r for r in _rows()}
    for drug in ("voriconazole", "propafenone"):
        assert len(rows[drug]["doses"]) >= 2, f"{drug} must be dose-ranging"


def test_anchor_is_linear():
    rows = {r["drug"]: r for r in _rows()}
    assert rows["metoprolol"]["is_nonlinear"] is False
```

- [ ] **Step 2: Run to verify failure** — FileNotFoundError.

- [ ] **Step 3: Write the benchmark** from `pgx_cmax_v2b_feasibility_2026-06-15.md`. Shape (fill
  curated values + citations; raw `Km` stays raw — conversion happens in the harness):

```json
{
  "description": "PGx v2.2b nonlinear saturable genotype-fold benchmark. Raw literature Km (µM); harness converts to unbound mg/L. Locked; never refit Km. Spec 2026-06-15.",
  "drugs": [
    {
      "drug": "propafenone", "gene": "CYP2D6", "set": "powered",
      "is_nonlinear": true, "is_mbi": false,
      "km_uM": 5.3, "km_basis": "total_microsomal_HLM_5OH", "fu_mic": 0.5,
      "mw": 341.4, "fm_invitro": 0.80, "fm_source_type": "in_vitro_phenotyping",
      "oral_f": 0.10, "em_tmax_h": 2.5, "em_thalf_h": 5.5,
      "doses": [
        {"dose_mg": 300, "obs_cmax_fold": 2.4, "obs_cmax_fold_ci": [1.7, 3.4],
         "obs_auc_fold": 11.0, "obs_auc_fold_ci": [8.0, 15.0]},
        {"dose_mg": 400, "obs_cmax_fold": 2.0, "obs_cmax_fold_ci": [1.4, 2.9],
         "obs_auc_fold": 8.5, "obs_auc_fold_ci": [6.0, 12.0]}
      ],
      "flags": [], "citation_km": "Kroemer 1991 PMID 1857335",
      "citation_folds": "Tran 2022 PMID 35890339"
    }
  ]
}
```

  Include: propafenone, voriconazole (≥2 doses), lansoprazole, atomoxetine (set=powered);
  metoprolol (set=anchor, is_nonlinear=false); perhexiline (set=secondary). Use the gate note's
  values; where a value was flagged "to extract/pull" (lansoprazole `Km`, atomoxetine fold+CI),
  curate it from the cited primary before locking — do NOT invent. If a powered drug's `Km` truly
  cannot be obtained, demote it to `set="secondary"` and note why.

- [ ] **Step 4: Run to verify pass** — 4 tests pass. (If `test_powered_rows...` count <3, the
  curation under-delivered — surface to controller.)

- [ ] **Step 5: Commit**

```bash
git add data/validation/pgx_cmax_v2b_folds.json tests/unit/test_pgx_v2b_benchmark_schema.py
git commit --no-verify -m "feat(pgx): locked v2.2b nonlinear benchmark + schema guard"
```

---

## Task 4: production three-engine harness

**Files:**
- Modify: `scripts/validate_pgx_cmax_v2b.py` (generalize the spike to all drugs/doses)
- Test: `tests/integration/test_pgx_v2b_harness.py`

- [ ] **Step 1: Generalize the harness.** Wrap the spike machinery into:

```python
def evaluate_drug(row: dict) -> list[dict]:
    """Per dose: saturable EM-anchor → {EM,PM}-saturable + linear-null folds vs observed."""
    km_mgL = km_uM_to_unbound_mgL(row["km_uM"], row["mw"], row["fu_mic"])
    out = []
    for dose in row["doses"]:
        recipe = saturable_anchor(row["gene"], row["fm_invitro"], km_mgL,
                                  e_h_target=_e_h_from_oral_f(row["oral_f"]),
                                  tmax_target=row["em_tmax_h"], dose=dose["dose_mg"])
        sat = engine_folds(row["gene"], row["fm_invitro"], recipe, km_mgL, dose["dose_mg"])
        lin = engine_folds(row["gene"], row["fm_invitro"], recipe_linear(recipe),
                           km_mgL=float("inf"), dose=dose["dose_mg"])  # re-anchored, km=inf
        out.append({"dose": dose["dose_mg"],
                    "obs_cmax_fold": dose["obs_cmax_fold"], "obs_auc_fold": dose["obs_auc_fold"],
                    "sat": sat, "lin": lin})
    return out
```

  `engine_folds` runs EM and PM (gene→0) on the (possibly saturable) skeleton and returns
  `{cmax_fold, auc_fold, rho}`. `recipe_linear` re-anchors with `km=inf` (the null). Keep helpers
  in the harness (engine-coupled).

- [ ] **Step 2: Write the harness regression test** (the spike's gates, locked)

```python
# tests/integration/test_pgx_v2b_harness.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from validate_pgx_cmax_v2b import evaluate_drug  # noqa: E402

_PROP = {  # propafenone reference row (mirrors the benchmark)
    "drug": "propafenone", "gene": "CYP2D6", "km_uM": 5.3, "km_basis": "x", "fu_mic": 0.5,
    "mw": 341.4, "fm_invitro": 0.80, "oral_f": 0.10, "em_tmax_h": 2.5, "em_thalf_h": 5.5,
    "doses": [{"dose_mg": 300, "obs_cmax_fold": 2.4, "obs_auc_fold": 11.0},
              {"dose_mg": 400, "obs_cmax_fold": 2.0, "obs_auc_fold": 8.5}],
}


def test_saturation_engaged_and_dose_dependent():
    rows = evaluate_drug(_PROP)
    import math
    # saturation moves the AUC-fold vs the linear null (gate 2)
    r300 = rows[0]
    assert abs(math.log(r300["sat"]["auc_fold"]) - math.log(r300["lin"]["auc_fold"])) > 0.10
    # dose-dependence: saturable AUC-fold shrinks 300 → 400 (gate 3)
    assert math.log(rows[1]["sat"]["auc_fold"]) < math.log(rows[0]["sat"]["auc_fold"])
    # linear null is ~flat across dose
    assert abs(math.log(rows[1]["lin"]["auc_fold"])
               - math.log(rows[0]["lin"]["auc_fold"])) < 0.03
```

- [ ] **Step 3: Run** — `python -m pytest tests/integration/test_pgx_v2b_harness.py -v` (PASS;
  these reproduce the spike gates). `ruff` clean.

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_pgx_cmax_v2b.py tests/integration/test_pgx_v2b_harness.py
git commit --no-verify -m "feat(pgx): v2.2b three-engine saturable harness + gate pins"
```

---

## Task 5: scoring (P1/P2/C1/C2) + report + registry

**Files:**
- Modify: `scripts/validate_pgx_cmax_v2b.py` (add `main()` + scoring)
- Modify: `data/validation/pgx_fm_registry.json` (extend)
- Create (generated): `data/validation/pgx_cmax_v2b_validation_2026-06-16.{json,md}`

- [ ] **Step 1: Scoring + `main()`.** Over the benchmark: collect per-drug per-dose `sat`/`lin`
  folds; compute
  - **P1:** paired Wilcoxon of `|log(obs)−log(sat)|` vs `|log(obs)−log(lin)|` over all powered
    (drug,dose,endpoint) rows; `sat_better = median(err_sat) < median(err_lin)`.
  - **P2:** for voriconazole/propafenone, `dose_dependence(sat_auc_folds, doses)` vs the observed
    `dose_dependence(obs_auc_folds, doses)` — sign agreement + the linear null's flatness.
  - **C1:** metoprolol — assert `|log(sat_auc)−log(lin_auc)| < 0.05` and no dose-dependence.
  - **C2:** linear-null AUC-fold == analytic `1/(1−fm)` within 2% on a low-extraction synthetic.
  Write `pgx_cmax_v2b_validation_2026-06-16.{json,md}` (record the pre-registered §4 criteria
  verbatim + realized stats) and extend `pgx_fm_registry.json` with a per-drug `saturable_layer`
  (`km_uM`, `km_mgL`, `regime`, `sat`/`lin`/`obs` folds). **Report the result as-is — PASS or
  honest negative (P1 ties ⇒ saturation no better than linear at this N/Km uncertainty). Never
  refit `Km`.**

- [ ] **Step 2: Run the harness end-to-end** — `python scripts/validate_pgx_cmax_v2b.py` prints
  `P1_sat_better=… P2_dose_dep=… n_powered=…`; writes report + registry. Verify the registry
  extension did not clobber the v1/existing layers (`python -c "import json;
  d=json.load(open('data/validation/pgx_fm_registry.json')); print(any('fm_invivo' in v for v in
  d.values()))"` → True).

- [ ] **Step 3: Commit** (include generated report + registry)

```bash
git add scripts/validate_pgx_cmax_v2b.py data/validation/pgx_fm_registry.json \
        data/validation/pgx_cmax_v2b_validation_2026-06-16.json \
        data/validation/pgx_cmax_v2b_validation_2026-06-16.md
git commit --no-verify -m "feat(pgx): v2.2b scoring (P1/P2/C1/C2) + report + registry extension"
```

---

## Task 6: headline-isolation guard + docs wire-back

**Files:**
- Create: `tests/regression/test_pgx_v2b_headline_isolation.py`
- Modify: `docs/claude/experiment-log.md`

- [ ] **Step 1: Isolation test**

```python
# tests/regression/test_pgx_v2b_headline_isolation.py
"""v2.2b is harness-isolated. pgx_metrics stays engine-free; the harness sets enzyme_km only on
synthetic skeleton drugs (production predict path never does), so the holdout cache is untouched."""
from __future__ import annotations

import importlib


def test_pgx_metrics_is_pure():
    mod = importlib.import_module("sisyphus.validation.pgx_metrics")
    src = open(mod.__file__).read()
    for forbidden in ("import sisyphus.engine", "from sisyphus.engine",
                      "from sisyphus.predict", "import sisyphus.pipeline"):
        assert forbidden not in src, f"pgx_metrics must stay pure; found {forbidden!r}"
```

- [ ] **Step 2: Run it** — PASS. Also run `python -m pytest
  tests/regression/test_mm_headline_bit_identity.py
  tests/integration/test_holdout_regression.py -q` to confirm the v2.2a bit-identity pin and the
  holdout cache pin still hold (headline 2.731 untouched).

- [ ] **Step 3: Append the experiment-log entry** (top, under the header `---`), using the
  realized P1/P2 numbers from Task 5:

```markdown
## 2026-06-16 — PGx v2.2b: nonlinear saturable genotype-fold validation (P1 <PASS|TIE>, N=<n>)

Engine-differentiated validation consuming the v2.2a MM flux. **Headline 2.731 untouched**
(harness-isolated; `enzyme_km` only on synthetic skeletons). Saturable `well_stirred` skeleton,
gene fraction carries literature `Km` (residual linear), EM-anchored / PM-predicted, three engines
(EM/PM-saturable + linear null `Km=∞`). Powered: voriconazole/propafenone (dose-ranging),
lansoprazole, atomoxetine; anchor metoprolol; secondary perhexiline.

- **P1 (saturable beats linear on single-dose folds):** sat better=<bool>, median |Δlog| sat <x>
  vs lin <y> (paired Wilcoxon p=<p>).
- **P2 (dose-dependence, voriconazole/propafenone):** saturable reproduces the observed fold
  shrinkage with dose (Δlog sign agreement); linear null flat. <one sentence>.
- **C1 metoprolol anchor:** saturable ≈ linear ≈ observed, no spurious dose-dependence.
- Durable: `pgx_fm_registry.json` `saturable_layer` (Km/regime/fold deltas). <If P1 ties: log a
  DE-NN — saturation not separable from linear at this N / Km uncertainty.>
```

- [ ] **Step 4: Commit**

```bash
git add tests/regression/test_pgx_v2b_headline_isolation.py docs/claude/experiment-log.md
git commit --no-verify -m "docs(pgx): v2.2b result entry + headline-isolation guard"
```

- [ ] **Step 5: If P1 tied,** append a `DE-NN` to `docs/claude/dead-ends.md` (saturation not
  separable from the linear engine for genotype folds at this N/`Km` uncertainty) and commit.

---

## Final review

`python -m pytest tests/unit/test_pgx_v2b_metrics.py tests/unit/test_pgx_v2b_benchmark_schema.py
tests/integration/test_pgx_v2b_harness.py tests/regression -q` and `ruff check src tests` — all
green. Dispatch a final reviewer over the diff; confirm: (1) no `predict()`/`reference_man.yaml`
edit; (2) `pgx_metrics` engine-free; (3) `Km` never tuned to a fold; (4) the saturable anchor and
linear null are both EM-anchored (fair contrast). Then superpowers:finishing-a-development-branch.

---

## Self-review notes

- **Spec coverage:** §3.1 saturable anchor → Task 1 (prototype) + Task 4 (`saturable_anchor`).
  §3.2 three engines → Task 4. §4 P1/P2/C1/C2 → Task 5. §5 `Km` conversion → Task 2. §6
  scope/schema → Task 3. §7 spike → Task 1 (gating). §8 tests → Tasks 2–6. §9 components → all.
- **Gating risk:** Task 1 is the make-or-break — if saturation isn't engaged at therapeutic dose
  (gate 2), HALT and surface; do not build Tasks 2–6 on a failed spike.
- **Type consistency:** `km_uM_to_unbound_mgL(km_uM, mw, fu_mic)`, `dose_dependence(folds, doses)
  -> {delta_log, shrinks_with_dose}`, `evaluate_drug(row) -> [{dose,obs_*,sat,lin}]`,
  `engine_folds(...) -> {cmax_fold, auc_fold, rho}` — consistent across tasks.
- **No fitting:** `Km`/`fm`/`fu_mic` curated pre-run (Task 3), locked; the anchor targets EM PK,
  never the genotype fold.
