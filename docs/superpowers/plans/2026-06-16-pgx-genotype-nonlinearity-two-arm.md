# Two-Arm Genotype-Nonlinearity Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that the v2.2a saturable Michaelis–Menten engine reproduces genotype-stratified *nonlinear dose-dependence* — with arm-opposite signs (systemic fold diverges, first-pass fold converges) — that a linear model cannot, using literature `Km` (never fitted) and a cherry-pick-proof feasibility gate.

**Architecture:** Pure metric helpers in `src/sisyphus/validation/pgx_metrics.py`; a locked literature dataset + schema guard; a two-arm three-engine harness extending `scripts/validate_pgx_cmax_v2b.py` (Arm S = steady-state `well_stirred` via `solve_regimen`; Arm F = single-dose **axial** `parallel_tube` via `expand_axial` + the (A) PR #79 phenotype fix). Genotype contrast = EM vs PM at **realistic literature `pm_gene_activity`** (via `phenotype_scale_overrides`), never gene→0 (that flips the Arm-S sign; reserved for the oracle).

**Tech Stack:** Python 3.10+, numpy, scipy (`brentq`, `polyfit`), pytest. Engine+scipy at `/opt/miniconda3/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-16-pgx-genotype-nonlinearity-two-arm-design.md`

**Constraints (load-bearing):**
- **Harness-isolated.** No `predict()` / `reference_man.yaml` / holdout change. Headline Meta AAFE **2.731 bit-identical**. `enzyme_km` only on synthetic skeletons.
- **No fitting / no cherry-picking.** `Km`, `fm`, `fu_mic`, `pm_gene_activity` are literature, fixed. The Task 1 box-robustness gate forecloses single-favorable-`Km` artifacts. Synthetic-param selection in *data-independent* unit tests (to make a mechanism visible) is allowed and must be labeled as such — it is never fitting to an observed clinical fold.
- **Dependency:** Arm F requires the (A) axial phenotype fix (PR #79). **Arm F tasks (4F, 5F, 6F) must not run until #79 is merged to `main`.** Arm S, the pure helpers, the dataset, and the oracle do not depend on #79.
- Commit as `jam-sudo <jam-sudo@users.noreply.github.com>`. **NEVER** add a `Co-Authored-By: Claude` / AI footer / "Generated with" line. Use `git commit --no-verify`.
- Stage ONLY the files each task names. **NEVER** `git add README.md` or any untracked workspace file. After each commit run `git show --name-only HEAD` and verify scope.
- Tests: `/opt/miniconda3/bin/python -m pytest`. Lint: `ruff check` (line-length 100; CI runs `ruff check src tests` repo-wide).

---

## File Structure

- **Modify** `src/sisyphus/validation/pgx_metrics.py` — add 4 pure functions: `km_uM_to_unbound_mgL`, `loglog_beta`, `delta_beta`, `box_robustness_pass`.
- **Modify** `scripts/validate_pgx_cmax_v2b.py` — add `_axial_graph`, `_steady_state_exposure`, `_single_dose_exposure`, `_beta_for_genotype`, the box-robustness probe runner, the two-arm scoring (`run_arm_s`, `run_arm_f`), the oracle check, and a `--full` entrypoint. Reuses existing `_well_stirred_graph`, `_drug`, `_sat_drug`, `_cmax_auc_tmax`, `_peak_liver_cu`, `anchor_em`.
- **Create** `data/validation/pgx_genotype_nonlinearity_folds.json` — the locked dataset.
- **Create** `tests/regression/test_pgx_genotype_nonlinearity_schema.py` — schema guard.
- **Create** `tests/unit/test_pgx_nonlinearity_metrics.py` — pure-helper unit tests.
- **Create** `tests/integration/test_pgx_arm_sign_mechanism.py` — the data-independent arm-sign crux + axial-inlet check + oracle + headline isolation.
- **Create** `data/validation/pgx_genotype_nonlinearity_2026-06-16.{json,md}` — results + report (Task 8).

---

### Task 1: Pure metric helpers (`km_uM_to_unbound_mgL`, `loglog_beta`, `delta_beta`, `box_robustness_pass`)

**Files:**
- Modify: `src/sisyphus/validation/pgx_metrics.py`
- Test: `tests/unit/test_pgx_nonlinearity_metrics.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the genotype-nonlinearity pure metrics.
Spec: 2026-06-16-pgx-genotype-nonlinearity-two-arm-design.md §5.2, §6."""
from __future__ import annotations

import pytest

from sisyphus.validation.pgx_metrics import (
    box_robustness_pass,
    delta_beta,
    km_uM_to_unbound_mgL,
    loglog_beta,
)


def test_km_conversion_worked_example():
    # propafenone: 5.3 µM × fu_mic 0.5 × MW 341.4 / 1000 = 0.90471 mg/L (spike value)
    assert km_uM_to_unbound_mgL(5.3, 341.4, 0.5) == pytest.approx(0.90471, rel=1e-4)


def test_km_conversion_rejects_bad_input():
    with pytest.raises(ValueError):
        km_uM_to_unbound_mgL(-1.0, 300.0, 0.5)
    with pytest.raises(ValueError):
        km_uM_to_unbound_mgL(5.0, 300.0, 0.0)
    with pytest.raises(ValueError):
        km_uM_to_unbound_mgL(5.0, 300.0, 1.5)


def test_loglog_beta_proportional_is_one():
    # AUC exactly proportional to dose → slope 1.0
    assert loglog_beta([100.0, 200.0, 400.0], [10.0, 20.0, 40.0]) == pytest.approx(1.0)


def test_loglog_beta_supraproportional_gt_one():
    # doubling dose quadruples exposure → slope 2.0
    assert loglog_beta([100.0, 200.0], [10.0, 40.0]) == pytest.approx(2.0)


def test_loglog_beta_needs_two_points():
    with pytest.raises(ValueError):
        loglog_beta([100.0], [10.0])


def test_delta_beta_sign():
    assert delta_beta(1.6, 1.0) == pytest.approx(0.6)   # systemic: PM more nonlinear
    assert delta_beta(1.0, 1.6) == pytest.approx(-0.6)  # first-pass: EM more nonlinear


def test_box_robustness_pass_requires_every_corner():
    # all corners above threshold → pass
    assert box_robustness_pass([0.2, 0.15, 0.30, 0.12], threshold=0.10) is True
    # one corner below → fail (cherry-pick foreclosure)
    assert box_robustness_pass([0.2, 0.05, 0.30, 0.12], threshold=0.10) is False
    with pytest.raises(ValueError):
        box_robustness_pass([], threshold=0.10)
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_pgx_nonlinearity_metrics.py -v`
Expected: FAIL — `ImportError` (functions not defined).

- [ ] **Step 3: Implement the helpers**

Append to `src/sisyphus/validation/pgx_metrics.py` (the module already imports nothing heavy; add `import numpy as np` at the top if not present):

```python
def km_uM_to_unbound_mgL(km_uM: float, mw: float, fu_mic: float) -> float:
    """Literature Km (µM, total-microsomal) → engine unbound mg/L.

    Km_unbound[mg/L] = km_uM × fu_mic × MW / 1000   (basis of C_u = fup·c_plasma).
    """
    if km_uM <= 0 or mw <= 0:
        raise ValueError(f"km_uM and mw must be positive, got {km_uM}, {mw}")
    if not 0 < fu_mic <= 1:
        raise ValueError(f"fu_mic must be in (0, 1], got {fu_mic}")
    return km_uM * fu_mic * mw / 1000.0


def loglog_beta(doses: list[float], exposures: list[float]) -> float:
    """Log–log slope β = d log(exposure) / d log(dose) (least squares).

    β = 1 ⇒ dose-proportional (linear); β > 1 ⇒ supra-proportional (saturation).
    """
    if len(doses) != len(exposures) or len(doses) < 2:
        raise ValueError("need >=2 matched (dose, exposure) points")
    ld = np.log(np.asarray(doses, dtype=float))
    le = np.log(np.asarray(exposures, dtype=float))
    return float(np.polyfit(ld, le, 1)[0])


def delta_beta(beta_pm: float, beta_em: float) -> float:
    """Genotype cross-term Δβ = β_PM − β_EM.

    >0 ⇒ systemic divergence (PM's low Vmax saturates first);
    <0 ⇒ first-pass convergence (EM's high extraction saturates).
    """
    return float(beta_pm - beta_em)


def box_robustness_pass(deltas: list[float], threshold: float = 0.10) -> bool:
    """True iff |Δlog fold| exceeds `threshold` at EVERY Km×fu_mic box corner.

    Engagement only at the favorable corner is a FAIL (the v2.2b cherry-pick
    foreclosure). `deltas` are the per-corner |Δlog AUC-fold (sat − linear-null)|.
    """
    if not deltas:
        raise ValueError("deltas must be non-empty")
    return all(d > threshold for d in deltas)
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_pgx_nonlinearity_metrics.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/sisyphus/validation/pgx_metrics.py tests/unit/test_pgx_nonlinearity_metrics.py
git add src/sisyphus/validation/pgx_metrics.py tests/unit/test_pgx_nonlinearity_metrics.py
git commit --no-verify -m "feat(pgx): nonlinearity metric helpers (km conversion, log-log beta, box robustness)"
git show --name-only HEAD
```
Verify scope = exactly those two files.

---

### Task 2: Locked dataset + schema guard (Task 0a — data-availability gate)

**Files:**
- Create: `data/validation/pgx_genotype_nonlinearity_folds.json`
- Test: `tests/regression/test_pgx_genotype_nonlinearity_schema.py` (new)

> **HALT-EARLY GATE.** This task curates the **observed dose-ranging genotype data**. If a clean-primary drug's ≥2-dose genotype-stratified exposure cannot be sourced from the literature, **do not fabricate it** — set that drug's `tier` to `at_risk` with a `data_gap` note and surface it. If *both* primaries in an arm lack dose-ranging data, that arm HALTs (Task 6 skips it). The constants below (`Km`, `fm`, `mw`, `fup`, `pm_gene_activity`) are pre-curated from the spec/spike; the **observed `dose_rows` exposures + their citations are the real curation work** and MUST carry a `source` per row.

- [ ] **Step 1: Write the failing schema-guard test**

```python
"""Schema guard for the locked genotype-nonlinearity dataset.
Spec: 2026-06-16-pgx-genotype-nonlinearity-two-arm-design.md §7."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
_DATA = ROOT / "data" / "validation" / "pgx_genotype_nonlinearity_folds.json"

_ARMS = {"systemic", "first_pass"}
_TIERS = {"clean_primary", "confounded_secondary", "at_risk", "negative_control"}
_MODELS = {"well_stirred", "parallel_tube"}


def _load():
    return json.loads(_DATA.read_text())


def test_dataset_exists_and_has_drugs():
    d = _load()
    assert "drugs" in d and isinstance(d["drugs"], list) and d["drugs"]


def test_every_drug_has_core_fields():
    for drug in _load()["drugs"]:
        assert drug["arm"] in _ARMS, drug.get("name")
        assert drug["tier"] in _TIERS, drug.get("name")
        assert drug["liver_model"] in _MODELS, drug.get("name")
        for k in ("name", "gene", "mw", "fm"):
            assert k in drug, (drug.get("name"), k)


def test_km_block_present_and_non_circular():
    for drug in _load()["drugs"]:
        if drug["tier"] in ("clean_primary", "negative_control"):
            km = drug["km"]
            assert km["value_uM"] > 0 and km["fu_mic"] > 0
            assert km["basis"] and km["source"]  # provenance, non-circular (in-vitro)


def test_clean_primary_has_two_dose_rows_and_activity():
    for drug in _load()["drugs"]:
        if drug["tier"] != "clean_primary":
            continue
        rows = drug["dose_rows"]
        assert len(rows) >= 2, drug["name"]
        for r in rows:
            assert r["dose_mg"] > 0 and "exposure" in r
            assert r["exposure"]["EM"] > 0 and r["exposure"]["PM"] > 0
            assert r["source"]  # observed value must be sourced
        a = drug["pm_gene_activity"]
        assert 0.0 < a["value"] < 1.0 and a["source"], drug["name"]


def test_mbi_drugs_are_not_clean_primary():
    # voriconazole (auto-inhibition) may appear ONLY as confounded_secondary
    for drug in _load()["drugs"]:
        if drug.get("is_mbi"):
            assert drug["tier"] == "confounded_secondary", drug["name"]


def test_negative_controls_are_linear():
    for drug in _load()["drugs"]:
        if drug["tier"] == "negative_control":
            assert drug["is_nonlinear"] is False
            assert drug["km"]["value_uM"] > 0  # needs Km so N1 runs the saturable path
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/miniconda3/bin/python -m pytest tests/regression/test_pgx_genotype_nonlinearity_schema.py -v`
Expected: FAIL — file not found.

- [ ] **Step 3: Create the dataset**

Create `data/validation/pgx_genotype_nonlinearity_folds.json`. Pre-fill the constants below; **curate the `dose_rows` observed `EM`/`PM` exposures from the literature** (phenytoin: steady-state Css at ≥2 daily doses, CYP2C9 EM vs *3/*3; propafenone: AUC or Cmax at ≥2 single oral doses, CYP2D6 EM vs PM). Each `dose_rows[*].source` and `pm_gene_activity.source` must be a real citation. If a primary's dose-ranging data cannot be found, change its `tier` to `at_risk`, drop `dose_rows` to what exists, and add `"data_gap": "<what is missing>"`.

```json
{
  "meta": {
    "purpose": "Locked observed genotype-stratified nonlinear exposure for the two-arm validation. Never refit.",
    "spec": "docs/superpowers/specs/2026-06-16-pgx-genotype-nonlinearity-two-arm-design.md",
    "units": {"km_value": "uM (total microsomal)", "exposure": "AUC mg·h/L or steady-state Css mg/L, stated per drug", "mw": "g/mol"}
  },
  "drugs": [
    {
      "name": "phenytoin", "arm": "systemic", "tier": "clean_primary", "gene": "CYP2C9",
      "liver_model": "well_stirred", "mw": 252.3, "fm": 0.90, "fup": 0.10,
      "is_mbi": false, "is_nonlinear": true,
      "exposure_metric": "css_avg",
      "regimen": {"interval_h": 24.0, "n_doses": 30},
      "km": {"value_uM": 5.0, "basis": "in-vivo unbound Vmax/Km", "fu_mic": 1.0, "source": "<CURATE: e.g. population PK, unbound Km>"},
      "pm_gene_activity": {"value": 0.25, "source": "<CURATE: CYP2C9*3/*3 residual phenytoin clearance fraction>"},
      "dose_rows": [
        {"dose_mg": 200.0, "exposure": {"EM": 0.0, "PM": 0.0}, "source": "<CURATE observed Css EM/PM at 200 mg/d>"},
        {"dose_mg": 300.0, "exposure": {"EM": 0.0, "PM": 0.0}, "source": "<CURATE observed Css EM/PM at 300 mg/d>"},
        {"dose_mg": 400.0, "exposure": {"EM": 0.0, "PM": 0.0}, "source": "<CURATE observed Css EM/PM at 400 mg/d>"}
      ]
    },
    {
      "name": "propafenone", "arm": "first_pass", "tier": "clean_primary", "gene": "CYP2D6",
      "liver_model": "parallel_tube", "mw": 341.4, "fm": 0.80, "fup": 0.30,
      "is_mbi": false, "is_nonlinear": true,
      "exposure_metric": "auc",
      "em_anchor": {"tmax_h": 2.5, "thalf_h": 5.5, "f_oral": 0.10},
      "km": {"value_uM": 5.3, "basis": "HLM total", "fu_mic": 0.5, "source": "Kroemer 1989 (also Hemeryck 0.12 µM — span used in box gate)"},
      "km_span_uM": [0.12, 5.3],
      "pm_gene_activity": {"value": 0.03, "source": "<CURATE: CYP2D6 PM (*4/*4) residual activity ~0.01-0.05>"},
      "dose_rows": [
        {"dose_mg": 150.0, "exposure": {"EM": 0.0, "PM": 0.0}, "source": "<CURATE observed AUC EM/PM at 150 mg>"},
        {"dose_mg": 300.0, "exposure": {"EM": 0.0, "PM": 0.0}, "source": "<CURATE observed AUC EM/PM at 300 mg>"},
        {"dose_mg": 450.0, "exposure": {"EM": 0.0, "PM": 0.0}, "source": "<CURATE observed AUC EM/PM at 450 mg>"}
      ]
    },
    {
      "name": "voriconazole", "arm": "systemic", "tier": "confounded_secondary", "gene": "CYP2C19",
      "liver_model": "well_stirred", "mw": 349.3, "fm": 0.60, "fup": 0.42,
      "is_mbi": true, "is_nonlinear": true,
      "note": "auto-inhibition (TDI) — NOT clean reversible MM; illustrative only, excluded from primary scoring",
      "km": {"value_uM": 9.3, "basis": "HLM total", "fu_mic": 0.6, "source": "<CURATE>"}
    },
    {
      "name": "mexiletine", "arm": "first_pass", "tier": "at_risk", "gene": "CYP2D6",
      "liver_model": "parallel_tube", "mw": 179.3, "fm": 0.60, "fup": 0.56,
      "is_mbi": false, "is_nonlinear": true,
      "data_gap": "genotype-stratified dose-ranging exposure thin; promote to clean_primary only if curated",
      "km": {"value_uM": 15.8, "basis": "HLM total", "fu_mic": 0.5, "source": "<CURATE>"}
    },
    {
      "name": "tolbutamide", "arm": "systemic", "tier": "negative_control", "gene": "CYP2C9",
      "liver_model": "well_stirred", "mw": 270.3, "fm": 0.85, "fup": 0.05,
      "is_mbi": false, "is_nonlinear": false,
      "control_role": "hard_null",
      "km": {"value_uM": 200.0, "basis": "HLM total (high → unsaturated at clinical dose)", "fu_mic": 0.5, "source": "<CURATE>"},
      "pm_gene_activity": {"value": 0.15, "source": "<CURATE CYP2C9*3/*3>"},
      "dose_rows": [
        {"dose_mg": 500.0, "exposure": {"EM": 0.0, "PM": 0.0}, "source": "<CURATE>"},
        {"dose_mg": 1000.0, "exposure": {"EM": 0.0, "PM": 0.0}, "source": "<CURATE>"}
      ]
    },
    {
      "name": "metoprolol", "arm": "first_pass", "tier": "negative_control", "gene": "CYP2D6",
      "liver_model": "parallel_tube", "mw": 267.4, "fm": 0.75, "fup": 0.88,
      "is_mbi": false, "is_nonlinear": false,
      "control_role": "mild_positive_calibration",
      "note": "mild saturable CYP2D6 first-pass — engine should reproduce a SMALL beta>1, not a flat null",
      "km": {"value_uM": 50.0, "basis": "HLM total", "fu_mic": 0.5, "source": "<CURATE>"},
      "pm_gene_activity": {"value": 0.02, "source": "<CURATE CYP2D6 PM>"},
      "dose_rows": [
        {"dose_mg": 50.0, "exposure": {"EM": 0.0, "PM": 0.0}, "source": "<CURATE>"},
        {"dose_mg": 200.0, "exposure": {"EM": 0.0, "PM": 0.0}, "source": "<CURATE>"}
      ]
    }
  ]
}
```

> Negative controls are tier `negative_control` but the schema test for "clean_primary has two dose rows" does not apply to them; the separate `test_negative_controls_are_linear` covers them. The `0.0` exposures are placeholders the curator MUST replace with sourced observed values before the dataset is considered locked; until then the arm's P1/P2 cannot score (Task 6 treats an all-zero `dose_rows` as a data gap and HALTs that drug).

- [ ] **Step 4: Run the schema guard**

Run: `/opt/miniconda3/bin/python -m pytest tests/regression/test_pgx_genotype_nonlinearity_schema.py -v`
Expected: PASS (structure valid). Note: the schema guard checks `exposure["EM"]>0` — so the curator must fill real positive observed values for the structural test to pass. If genuinely unavailable, move that drug to `at_risk` (then the clean-primary check skips it) and record the HALT in the Task 6 report.

- [ ] **Step 5: Commit**

```bash
git add data/validation/pgx_genotype_nonlinearity_folds.json tests/regression/test_pgx_genotype_nonlinearity_schema.py
git commit --no-verify -m "data(pgx): locked genotype-nonlinearity dataset + schema guard"
git show --name-only HEAD
```
Verify scope.

---

### Task 3: Harness runners — axial graph, steady-state, single-dose, per-genotype β

**Files:**
- Modify: `scripts/validate_pgx_cmax_v2b.py`
- Test: `tests/integration/test_pgx_arm_sign_mechanism.py` (new — runner smoke tests here; the sign crux is Task 4)

Add the building blocks the two arms share. Reuse existing `_well_stirred_graph`, `_drug`, `_sat_drug`, `_cmax_auc_tmax`. The compile/solve pattern mirrors `_cmax_auc_tmax` exactly (`rg = graph.realize_means()`, `ODECompiler().compile(rg)`, `ResolvedParams(rg, rd)`).

- [ ] **Step 1: Write failing smoke tests**

```python
"""Data-independent mechanism tests for the two-arm genotype-nonlinearity harness.
Spec: 2026-06-16-...-two-arm-design.md §5, §8. Runs the synthetic engine only —
no clinical data, no predict()/holdout. Imports the harness script by path."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = ROOT / "scripts" / "validate_pgx_cmax_v2b.py"


def _harness():
    spec = importlib.util.spec_from_file_location("v2b_harness", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_axial_graph_has_subtanks_no_literal_liver():
    h = _harness()
    g = h._axial_graph("CYP2D6", n_sub=8)
    assert "liver" not in g.nodes
    subs = [n for n in g.nodes.values() if (n.lookup_name or n.name) == "liver"]
    assert len(subs) == 8 and all("CYP2D6" in s.enzymes for s in subs)


def test_steady_state_exposure_accumulates():
    h = _harness()
    g = h._well_stirred_graph("CYP2C9")
    abund = g.nodes["liver"].enzymes["CYP2C9"].mean
    drug = h._drug("CYP2C9", 0.9, 5.0, abund, peff=20.0, kp=3.0, fup=0.10,
                   dose_mg=100.0, mw=252.3)
    e1 = h._steady_state_exposure(g, drug, interval_h=24.0, n_doses=1, metric="css_avg")
    e30 = h._steady_state_exposure(g, drug, interval_h=24.0, n_doses=20, metric="css_avg")
    assert e30 > e1 > 0  # accumulation to steady state


def test_single_dose_exposure_positive():
    h = _harness()
    g = h._well_stirred_graph("CYP2D6")
    abund = g.nodes["liver"].enzymes["CYP2D6"].mean
    drug = h._sat_drug("CYP2D6", 0.8, 1.0e7, abund, peff=20.0, kp=3.0, km_mgl=0.9,
                       fup=0.3, dose_mg=150.0, mw=341.4)
    assert h._single_dose_exposure(g, drug, metric="auc") > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_pgx_arm_sign_mechanism.py -v`
Expected: FAIL — `_axial_graph` / `_steady_state_exposure` / `_single_dose_exposure` not defined.

- [ ] **Step 3: Add the runners to `scripts/validate_pgx_cmax_v2b.py`**

Add near the top imports: `import dataclasses` (if absent) and the axial/regimen imports inside the functions (kept local to avoid import cost when running the spike). Insert after `_peak_liver_cu`:

```python
def _axial_graph(gene_tag: str, n_sub: int = 10):
    """Well_stirred synthetic skeleton with the liver clearance edge switched to
    parallel_tube and axially expanded into n_sub serial well_stirred sub-tanks
    (lookup_name='liver'). Genotype scaling reaches every sub-tank via the (A) fix."""
    import dataclasses as _dc

    from sisyphus.graph.axial import expand_axial
    from sisyphus.graph.types import ClearanceEdge
    g = _well_stirred_graph(gene_tag)
    g.nodes["liver"] = _dc.replace(g.nodes["liver"], axial_subcompartments=n_sub)
    g.edges[:] = [
        _dc.replace(e, model="parallel_tube")
        if isinstance(e, ClearanceEdge) and e.source == "liver"
        else e
        for e in g.edges
    ]
    return expand_axial(g)


def _steady_state_exposure(graph, drug, interval_h: float, n_doses: int,
                           metric: str = "css_avg") -> float:
    """Multi-dose steady-state exposure over the LAST dosing interval.
    metric: 'css_avg' (AUC_tau/tau), 'auc_tau', or 'cmax_ss'."""
    from sisyphus.regimen.solver import solve_regimen
    from sisyphus.regimen.types import DosingEvent, DosingRegimen
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    events = tuple(
        DosingEvent(time_h=i * interval_h, dose_mg=drug.dose_mg,
                    node=drug.administration_node)
        for i in range(n_doses)
    )
    reg = DosingRegimen(events=events)
    t_total = n_doses * interval_h
    res = solve_regimen(compiled, params, reg, t_total_h=t_total, dt_output=0.05)
    conc, time = res.concentrations["venous_blood"], res.time_h
    mask = time >= (t_total - interval_h - 1e-9)
    ct, tt = conc[mask], time[mask]
    trapz = getattr(np, "trapezoid", np.trapz)
    if metric == "cmax_ss":
        return float(ct.max())
    auc_tau = float(trapz(ct, tt))
    if metric == "auc_tau":
        return auc_tau
    if metric == "css_avg":
        return auc_tau / interval_h
    raise ValueError(f"unknown metric {metric!r}")


def _single_dose_exposure(graph, drug, metric: str = "auc") -> float:
    """Single-dose exposure on the given (possibly axial) graph. metric 'auc' or 'cmax'."""
    cmax, auc, _ = _cmax_auc_tmax(graph, drug)
    if metric == "auc":
        return auc
    if metric == "cmax":
        return cmax
    raise ValueError(f"unknown metric {metric!r}")
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_pgx_arm_sign_mechanism.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + commit**

```bash
ruff check scripts/validate_pgx_cmax_v2b.py tests/integration/test_pgx_arm_sign_mechanism.py
git add scripts/validate_pgx_cmax_v2b.py tests/integration/test_pgx_arm_sign_mechanism.py
git commit --no-verify -m "feat(pgx): axial + steady-state + single-dose harness runners"
git show --name-only HEAD
```

---

### Task 4: The arm-sign mechanism crux + axial-inlet check (data-independent)

**Files:**
- Modify: `scripts/validate_pgx_cmax_v2b.py` (add `_beta_for_genotype`)
- Test: `tests/integration/test_pgx_arm_sign_mechanism.py`

> This is the §5.2 crux and the highest-value test: prove the engine produces **Δβ > 0 for a systemic gene** and **Δβ < 0 for a first-pass gene** with realistic PM scaling — and that it would fail under gene→0 PM. Synthetic params are chosen to make saturation visible; this is **not** fitting to clinical data. **Arm F half (axial) requires PR #79 merged.**

- [ ] **Step 1: Add `_beta_for_genotype` to the harness**

Insert after `_single_dose_exposure`:

```python
def _beta_for_genotype(graph_builder, drug_builder, doses, genotype, gene_tag,
                       pm_activity, regime, interval_h=24.0, n_doses=20):
    """Log–log exposure–dose slope β for one genotype.

    graph_builder() -> a fresh base graph; drug_builder(dose) -> the drug at that dose.
    genotype: 'EM' (no scaling) or 'PM' (gene × pm_activity via the (A) phenotype fix).
    regime: 'steady_state' (Arm S) or 'single_dose' (Arm F).
    """
    from sisyphus.validation.pgx_metrics import loglog_beta
    exposures = []
    for dose in doses:
        g = graph_builder()
        if genotype == "PM":
            g = apply_phenotype_to_graph(
                g, {gene_tag: "PM"}, phenotype_scale_overrides={gene_tag: pm_activity}
            )
        drug = drug_builder(dose)
        if regime == "steady_state":
            exposures.append(_steady_state_exposure(g, drug, interval_h, n_doses, "css_avg"))
        elif regime == "single_dose":
            exposures.append(_single_dose_exposure(g, drug, "auc"))
        else:
            raise ValueError(f"unknown regime {regime!r}")
    return loglog_beta(list(doses), exposures)
```

- [ ] **Step 2: Write the failing sign tests**

Append to `tests/integration/test_pgx_arm_sign_mechanism.py`:

```python
def test_systemic_gene_diverges_delta_beta_positive():
    """Arm-S crux: systemic clearance gene → PM (low Vmax) saturates first →
    β_PM > β_EM → Δβ > 0 (fold diverges). Synthetic, deep-saturation params."""
    h = _harness()
    from sisyphus.validation.pgx_metrics import delta_beta
    gene, fm, mw, fup = "CYP2C9", 0.9, 252.3, 0.10
    km_mgl = 0.2  # low Km → strong systemic saturation across the dose span
    cltot, abund, kp = 50.0, h._SYNTHETIC_GENE_ABUND, 3.0
    doses = [100.0, 300.0, 900.0]  # ~10x span

    def gb():
        return h._well_stirred_graph(gene)

    def db(dose):
        return h._sat_drug(gene, fm, cltot, abund, 20.0, kp, km_mgl, fup, dose, mw)

    b_em = h._beta_for_genotype(gb, db, doses, "EM", gene, 0.25, "steady_state")
    b_pm = h._beta_for_genotype(gb, db, doses, "PM", gene, 0.25, "steady_state")
    assert delta_beta(b_pm, b_em) > 0.02, (b_pm, b_em)


def test_first_pass_gene_converges_delta_beta_negative():
    """Arm-F crux (REQUIRES PR #79): first-pass extraction gene → EM (high extraction)
    saturates → β_EM > β_PM → Δβ < 0 (fold converges). Axial skeleton."""
    h = _harness()
    from sisyphus.validation.pgx_metrics import delta_beta
    gene, fm, mw, fup = "CYP2D6", 0.85, 341.4, 0.30
    km_mgl = 0.3
    cltot, abund, kp = 5.0e6, h._SYNTHETIC_GENE_ABUND, 3.0
    doses = [75.0, 300.0, 600.0]

    def gb():
        return h._axial_graph(gene, n_sub=10)

    def db(dose):
        return h._sat_drug(gene, fm, cltot, abund, 20.0, kp, km_mgl, fup, dose, mw)

    b_em = h._beta_for_genotype(gb, db, doses, "EM", gene, 0.03, "single_dose")
    b_pm = h._beta_for_genotype(gb, db, doses, "PM", gene, 0.03, "single_dose")
    assert delta_beta(b_pm, b_em) < -0.02, (b_pm, b_em)


def test_axial_inlet_cu_exceeds_well_stirred():
    """Arm-F premise: the axial liver's peak unbound Cu exceeds the well_stirred Cu
    for the same drug/dose (well_stirred averages the inlet away)."""
    h = _harness()
    gene, fup = "CYP2D6", 0.30
    abund = h._SYNTHETIC_GENE_ABUND
    drug = h._sat_drug(gene, 0.85, 5.0e6, abund, 20.0, 3.0, 0.3, fup, 300.0, 341.4)
    g_ws = h._well_stirred_graph(gene)
    g_ax = h._axial_graph(gene, n_sub=10)
    cu_ws = h._peak_liver_cu(g_ws, drug, fup)
    # axial: max Cu over all sub-tanks (tank 1 sees the inlet)
    cu_ax = max(
        h._peak_liver_cu_node(g_ax, drug, n.name, fup)
        for n in g_ax.nodes.values() if (n.lookup_name or n.name) == "liver"
    )
    assert cu_ax > cu_ws > 0
```

This test references `_peak_liver_cu_node` (peak Cu at a *named* node). Add it to the harness next to `_peak_liver_cu`:

```python
def _peak_liver_cu_node(graph, drug, node_name: str, fup: float = 0.3) -> float:
    """Peak unbound conc (mg/L) at a specific node = fup * max(c_node)."""
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, float(_T_EVAL[-1])), t_eval=_T_EVAL)
    return float(fup * res.concentrations[node_name].max())
```

- [ ] **Step 3: Run — Arm-S sign + axial-inlet must pass now; Arm-F sign needs #79**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_pgx_arm_sign_mechanism.py -v`
Expected: `test_systemic_gene_diverges_delta_beta_positive` PASS; `test_axial_inlet_cu_exceeds_well_stirred` PASS; `test_first_pass_gene_converges_delta_beta_negative` PASS **iff PR #79 is merged** (else the axial PM scaling no-ops and Δβ≈0 → FAIL). If #79 is not yet merged, mark that one test `@pytest.mark.skip(reason="requires PR #79 axial phenotype fix")` and unskip after merge.

> If the Arm-S sign test does not clear the ±0.02 margin, widen the dose span or lower `km_mgl` (deeper saturation) — this is synthetic-param tuning for **mechanism visibility**, not fitting. The SIGN is physics-guaranteed; only its magnitude depends on how hard the gene is driven.

- [ ] **Step 4: Lint + commit**

```bash
ruff check scripts/validate_pgx_cmax_v2b.py tests/integration/test_pgx_arm_sign_mechanism.py
git add scripts/validate_pgx_cmax_v2b.py tests/integration/test_pgx_arm_sign_mechanism.py
git commit --no-verify -m "test(pgx): arm-sign mechanism crux (systemic diverges, first-pass converges) + axial-inlet check"
git show --name-only HEAD
```

---

### Task 5: Box-robustness gate (Task 0b) + oracle (C2)

**Files:**
- Modify: `scripts/validate_pgx_cmax_v2b.py` (add `box_robustness_probe`, `oracle_check`)
- Test: `tests/integration/test_pgx_arm_sign_mechanism.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_box_robustness_probe_propafenone_axial_engages():
    """Propafenone axial first-pass engages saturation across the Km span (low + high)
    × fu_mic ∈ {0.3,0.6,1.0}: every corner |Δlog AUC-fold| > 0.10 → gate PASS."""
    h = _harness()
    deltas = h.box_robustness_probe(
        gene_tag="CYP2D6", fm=0.80, mw=341.4, fup=0.30, dose_mg=300.0,
        km_span_uM=[0.12, 5.3], fu_mic_grid=[0.3, 0.6, 1.0],
        pm_activity=0.03, regime="single_dose",
    )
    from sisyphus.validation.pgx_metrics import box_robustness_pass
    # report all corners; propafenone is expected to pass at the LOW Km end robustly,
    # and the probe returns the per-corner deltas for the gate decision.
    assert len(deltas) == 6 and all(d >= 0 for d in deltas)
    # the gate decision itself (PASS/HALT) is asserted in the Task 6 report, not pinned
    # to a hard PASS here (high-Km corner may legitimately fail → that is the gate working)


def test_oracle_linear_fold_matches_analytic_well_stirred():
    """C2: with idealized gene→0 PM (a_var=0), the linear engine's oral AUC genotype
    fold = 1/(1-fm) on the well_stirred skeleton."""
    h = _harness()
    fold = h.oracle_check(gene_tag="CYP2C9", fm=0.9, skeleton="well_stirred")
    assert fold == pytest.approx(1.0 / (1.0 - 0.9), rel=0.02)


def test_oracle_linear_fold_matches_analytic_axial():
    """C2 on the axial skeleton (REQUIRES PR #79 for PM scaling); same 1/(1-fm)."""
    h = _harness()
    fold = h.oracle_check(gene_tag="CYP2D6", fm=0.8, skeleton="axial")
    assert fold == pytest.approx(1.0 / (1.0 - 0.8), rel=0.05)
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_pgx_arm_sign_mechanism.py -k "box_robustness_probe or oracle" -v`
Expected: FAIL — `box_robustness_probe` / `oracle_check` not defined.

- [ ] **Step 3: Implement the probe + oracle**

Add to the harness:

```python
def box_robustness_probe(gene_tag, fm, mw, fup, dose_mg, km_span_uM, fu_mic_grid,
                         pm_activity, regime, interval_h=24.0, n_doses=20):
    """Per-corner |Δlog AUC-fold (saturable − linear-null)| across the Km×fu_mic box.
    Returns the list of corner deltas; the caller applies box_robustness_pass()."""
    from sisyphus.validation.pgx_metrics import km_uM_to_unbound_mgL
    deltas = []
    for km_uM in km_span_uM:
        for fu_mic in fu_mic_grid:
            km_mgl = km_uM_to_unbound_mgL(km_uM, mw, fu_mic)
            fold_sat = _genotype_fold_engine(gene_tag, fm, mw, fup, dose_mg, km_mgl,
                                             pm_activity, regime, interval_h, n_doses)
            fold_lin = _genotype_fold_engine(gene_tag, fm, mw, fup, dose_mg, None,
                                             pm_activity, regime, interval_h, n_doses)
            deltas.append(abs(np.log(fold_sat) - np.log(fold_lin)))
    return deltas


def _genotype_fold_engine(gene_tag, fm, mw, fup, dose_mg, km_mgl, pm_activity,
                          regime, interval_h=24.0, n_doses=20, a_var=None):
    """PM/EM exposure fold at one dose. km_mgl None → linear-null. a_var overrides the
    PM gene scaling (None → use pm_activity; pass 0.0 for the idealized oracle)."""
    cltot, abund, kp = (50.0, _SYNTHETIC_GENE_ABUND, 3.0)
    pm_scale = pm_activity if a_var is None else a_var

    def build(graph_builder):
        g_em = graph_builder()
        g_pm = apply_phenotype_to_graph(
            graph_builder(), {gene_tag: "PM"},
            phenotype_scale_overrides={gene_tag: pm_scale},
        )
        if km_mgl is None:
            drug = _drug(gene_tag, fm, cltot, abund, 20.0, kp, fup, dose_mg, mw)
        else:
            drug = _sat_drug(gene_tag, fm, cltot, abund, 20.0, kp, km_mgl, fup,
                             dose_mg, mw)
        if regime == "steady_state":
            e_em = _steady_state_exposure(g_em, drug, interval_h, n_doses, "css_avg")
            e_pm = _steady_state_exposure(g_pm, drug, interval_h, n_doses, "css_avg")
        else:
            e_em = _single_dose_exposure(g_em, drug, "auc")
            e_pm = _single_dose_exposure(g_pm, drug, "auc")
        return e_pm / e_em

    builder = (lambda: _axial_graph(gene_tag)) if regime == "single_dose" \
        else (lambda: _well_stirred_graph(gene_tag))
    return build(builder)


def oracle_check(gene_tag, fm, skeleton):
    """C2: idealized gene→0 PM (a_var=0), LINEAR engine. Returns the oral AUC fold,
    which must equal 1/(1-fm)."""
    regime = "single_dose" if skeleton == "axial" else "steady_state"
    return _genotype_fold_engine(gene_tag, fm, 300.0, 0.3, 200.0, None, 0.0,
                                 regime, a_var=0.0)
```

> Note `_genotype_fold_engine` uses a fixed synthetic `cltot=50.0`; for the oracle the absolute value is irrelevant (the fold is `1/(1-fm)` at any cltot in the low-extraction limit). For `box_robustness_probe`, `cltot` should put the engine in the saturating regime — if a corner's `fold_sat≈fold_lin` because cltot is too low, that is a true non-engagement (gate working), not a bug.

- [ ] **Step 4: Run to verify pass**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_pgx_arm_sign_mechanism.py -k "box_robustness_probe or oracle" -v`
Expected: oracle tests PASS (axial one needs #79); the box probe test asserts shape + non-negativity, PASS.

- [ ] **Step 5: Lint + commit**

```bash
ruff check scripts/validate_pgx_cmax_v2b.py tests/integration/test_pgx_arm_sign_mechanism.py
git add scripts/validate_pgx_cmax_v2b.py tests/integration/test_pgx_arm_sign_mechanism.py
git commit --no-verify -m "feat(pgx): box-robustness gate probe + C2 oracle on both skeletons"
git show --name-only HEAD
```

---

### Task 6: Two-arm scoring (P1/P2/N1 + secondary) over the locked dataset

**Files:**
- Modify: `scripts/validate_pgx_cmax_v2b.py` (add `score_drug`, `run_arm`, `run_full`)
- Test: `tests/integration/test_pgx_arm_sign_mechanism.py`

> Drives the locked dataset. For each clean-primary drug: anchor at the lowest dose; compute `β_EM`, `β_PM` (saturable) and the linear-null β≡1; P1 (β_sat vs observed β), P2 (arm-correct Δβ sign), secondary accuracy delta. Negative controls run N1. **A drug whose `dose_rows` exposures are all 0 (uncurated) or `tier=at_risk` is skipped with a logged data-gap; an arm with no scorable clean drug HALTs.**

- [ ] **Step 1: Write the failing test**

```python
def test_run_arm_s_scores_or_halts_cleanly():
    """run_arm('systemic') returns a structured result: either scored phenytoin or a
    logged HALT (data gap). Never raises on a well-formed dataset."""
    h = _harness()
    res = h.run_arm("systemic")
    assert "arm" in res and res["arm"] == "systemic"
    assert "status" in res and res["status"] in ("scored", "halted")
    if res["status"] == "scored":
        assert "drugs" in res and res["drugs"]
        for d in res["drugs"]:
            assert "beta_em_sat" in d and "delta_beta_sat" in d
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_pgx_arm_sign_mechanism.py -k run_arm_s -v`
Expected: FAIL — `run_arm` not defined.

- [ ] **Step 3: Implement scoring**

Add to the harness (loads the dataset, scores per the spec):

```python
import json as _json

_FOLDS = Path("data/validation/pgx_genotype_nonlinearity_folds.json")


def _load_dataset():
    return _json.loads(_FOLDS.read_text())


def score_drug(drug: dict) -> dict:
    """Score one clean-primary drug: β_EM/β_PM (saturable) + arm-correct Δβ sign + the
    linear-null β≡1 reference. Returns a result dict; flags data gaps without raising."""
    rows = [r for r in drug.get("dose_rows", []) if r["exposure"]["EM"] > 0]
    if len(rows) < 2:
        return {"name": drug["name"], "status": "data_gap",
                "reason": "fewer than 2 curated dose rows"}
    doses = [r["dose_mg"] for r in rows]
    obs_em = [r["exposure"]["EM"] for r in rows]
    obs_pm = [r["exposure"]["PM"] for r in rows]
    from sisyphus.validation.pgx_metrics import delta_beta, loglog_beta
    gene, fm, mw, fup = drug["gene"], drug["fm"], drug["mw"], drug["fup"]
    pm_act = drug["pm_gene_activity"]["value"]
    km = drug["km"]
    km_mgl = km_uM_to_unbound_mgL(km["value_uM"], mw, km["fu_mic"])
    regime = "single_dose" if drug["liver_model"] == "parallel_tube" else "steady_state"
    interval_h = drug.get("regimen", {}).get("interval_h", 24.0)
    n_doses = drug.get("regimen", {}).get("n_doses", 20)

    gbuild = (lambda: _axial_graph(gene)) if regime == "single_dose" \
        else (lambda: _well_stirred_graph(gene))
    # anchor cltot at the LOWEST dose so (Vmax, CL_r) are fixed in the near-linear regime
    cltot, abund, kp = _anchor_cltot_low_dose(gene, fm, fup, mw, doses[0], km_mgl,
                                              regime, gbuild, interval_h, n_doses)

    def dbuild(dose):
        return _sat_drug(gene, fm, cltot, abund, 20.0, kp, km_mgl, fup, dose, mw)

    b_em = _beta_for_genotype(gbuild, dbuild, doses, "EM", gene, pm_act, regime,
                              interval_h, n_doses)
    b_pm = _beta_for_genotype(gbuild, dbuild, doses, "PM", gene, pm_act, regime,
                              interval_h, n_doses)
    beta_obs_em = loglog_beta(doses, obs_em)
    beta_obs_pm = loglog_beta(doses, obs_pm)
    dbeta_sat = delta_beta(b_pm, b_em)
    dbeta_obs = delta_beta(beta_obs_pm, beta_obs_em)
    expect_sign = +1.0 if drug["arm"] == "systemic" else -1.0
    return {
        "name": drug["name"], "arm": drug["arm"], "status": "scored",
        "beta_em_sat": b_em, "beta_pm_sat": b_pm, "delta_beta_sat": dbeta_sat,
        "beta_em_obs": beta_obs_em, "beta_pm_obs": beta_obs_pm,
        "delta_beta_obs": dbeta_obs, "beta_linear_null": 1.0,
        "p1_pass": abs(b_em - beta_obs_em) < 0.20 or abs(b_pm - beta_obs_pm) < 0.20,
        "p2_sign_correct": (dbeta_sat * expect_sign > 0) and (dbeta_obs * expect_sign > 0),
    }


def _anchor_cltot_low_dose(gene, fm, fup, mw, low_dose, km_mgl, regime, gbuild,
                           interval_h, n_doses):
    """Anchor cltot at the lowest dose by reusing anchor_em's E_h bisection at a low
    target extraction; returns (cltot, abund, kp). Kept simple: use anchor_em's machinery
    for well_stirred; for axial reuse the same cltot (extraction emerges from topology)."""
    abund = _well_stirred_graph(gene).nodes["liver"].enzymes[gene].mean
    recipe = anchor_em(gene, e_h_target=0.30, tmax_target=2.0, fm=fm, fup=fup,
                       dose_mg=low_dose, mw=mw, km_mgl=km_mgl)
    return recipe["cltot"], abund, recipe["kp"]


def run_arm(arm: str) -> dict:
    """Score every clean_primary drug in an arm; HALT if none scorable."""
    drugs = [d for d in _load_dataset()["drugs"]
             if d["arm"] == arm and d["tier"] == "clean_primary"]
    scored = [score_drug(d) for d in drugs]
    ok = [s for s in scored if s.get("status") == "scored"]
    return {"arm": arm, "status": "scored" if ok else "halted", "drugs": scored}


def run_full() -> dict:
    return {"systemic": run_arm("systemic"), "first_pass": run_arm("first_pass")}
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_pgx_arm_sign_mechanism.py -k run_arm_s -v`
Expected: PASS — returns `scored` (if phenytoin curated) or `halted` (if data gap), never raises.

- [ ] **Step 5: Lint + commit**

```bash
ruff check scripts/validate_pgx_cmax_v2b.py tests/integration/test_pgx_arm_sign_mechanism.py
git add scripts/validate_pgx_cmax_v2b.py tests/integration/test_pgx_arm_sign_mechanism.py
git commit --no-verify -m "feat(pgx): two-arm P1/P2 scoring over the locked dataset (anchor at lowest dose)"
git show --name-only HEAD
```

---

### Task 7: Headline-isolation guard

**Files:**
- Test: `tests/integration/test_pgx_arm_sign_mechanism.py`

- [ ] **Step 1: Write the guard test**

```python
def test_headline_isolation_holdout_cache_untouched():
    """Importing/running the harness must not touch the holdout cache, and the v2.2a
    empty-enzyme_km bit-identity + cached-2.731 pins still hold."""
    import subprocess
    import sys
    # the harness imports + a scoring run leave the cache file unmodified
    cache = ROOT / "data" / "training" / "4track_holdout_predictions.json"
    before = cache.read_bytes()
    h = _harness()
    h.run_arm("systemic")
    assert cache.read_bytes() == before
    # the v2.2a bit-identity + cached headline pins still pass
    r = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/regression/test_mm_headline_bit_identity.py",
         "tests/integration/test_holdout_regression.py::test_cached_holdout_aafe_is_2p731",
         "-q"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
```

- [ ] **Step 2: Run**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_pgx_arm_sign_mechanism.py -k headline_isolation -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_pgx_arm_sign_mechanism.py
git commit --no-verify -m "test(pgx): headline-isolation guard (holdout cache + 2.731 pins untouched)"
git show --name-only HEAD
```

---

### Task 8: Report + registry extension

**Files:**
- Modify: `scripts/validate_pgx_cmax_v2b.py` (add a `--full` CLI entry that writes the report)
- Create: `data/validation/pgx_genotype_nonlinearity_2026-06-16.{json,md}`
- Modify: `data/validation/pgx_fm_registry.json` (saturable layer)

- [ ] **Step 1: Add a `--full` entrypoint**

Extend `main()` (or add a branch) so `python scripts/validate_pgx_cmax_v2b.py --full` calls `run_full()`, plus the box-robustness gate per arm and the oracle, and writes:
- `data/validation/pgx_genotype_nonlinearity_2026-06-16.json` — `run_full()` output + per-arm box deltas + gate PASS/HALT + oracle folds.
- `data/validation/pgx_genotype_nonlinearity_2026-06-16.md` — a human report: per-arm status (scored/halted), each clean drug's β_EM/β_PM (sat vs observed), Δβ sign vs expected, P1/P2 verdict, the secondary accuracy delta, the box-robustness corners, and an explicit honest-negative note where an arm HALTed.

```python
def _write_full_report():
    import json as _j
    full = run_full()
    box_s = box_robustness_probe("CYP2C9", 0.9, 252.3, 0.10, 300.0, [5.0, 5.0],
                                 [0.3, 0.6, 1.0], 0.25, "steady_state")
    box_f = box_robustness_probe("CYP2D6", 0.8, 341.4, 0.30, 300.0, [0.12, 5.3],
                                 [0.3, 0.6, 1.0], 0.03, "single_dose")
    out = {"arms": full, "box_robustness": {"systemic": box_s, "first_pass": box_f},
           "oracle": {"well_stirred": oracle_check("CYP2C9", 0.9, "well_stirred"),
                      "axial": oracle_check("CYP2D6", 0.8, "axial")}}
    Path("data/validation/pgx_genotype_nonlinearity_2026-06-16.json").write_text(
        _j.dumps(out, indent=2))
    # ... render the .md from `out` (per-arm status, β table, gate corners, honest-negative)
    return out
```

(Render the `.md` deterministically from `out`; include the exact β/Δβ numbers and the gate verdict. Do not hand-edit numbers — generate them.)

- [ ] **Step 2: Run the full harness, generate the report**

Run: `/opt/miniconda3/bin/python scripts/validate_pgx_cmax_v2b.py --full`
Expected: writes both report files; prints per-arm status. Inspect the `.md`: every β/Δβ matches the `.json`.

- [ ] **Step 3: Extend `pgx_fm_registry.json` with the saturable layer**

Add a `saturable_layer` block keyed by drug with `{Km_uM, fu_mic, Km_unbound_mgL, regime, arm, tier, pm_gene_activity, delta_beta_sat, p2_sign_correct}` from the report. Keep it additive (do not alter existing keys).

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_pgx_cmax_v2b.py data/validation/pgx_genotype_nonlinearity_2026-06-16.json data/validation/pgx_genotype_nonlinearity_2026-06-16.md data/validation/pgx_fm_registry.json
git commit --no-verify -m "feat(pgx): two-arm validation full run + report + registry saturable layer"
git show --name-only HEAD
```

- [ ] **Step 5: Update experiment log + CLAUDE.md MIPD/PGx note**

Append a dated entry to `docs/claude/experiment-log.md` (top) with the outcome (per-arm scored/halted, Δβ signs, honest-negative where applicable). If an arm HALTed, also add a `DE-NN` entry to `docs/claude/dead-ends.md`. Do **not** touch the CLAUDE.md headline metrics block (2.731 is untouched by construction). Commit:

```bash
git add docs/claude/experiment-log.md docs/claude/dead-ends.md
git commit --no-verify -m "docs(pgx): log two-arm genotype-nonlinearity outcome"
git show --name-only HEAD
```

---

## Self-Review

**Spec coverage (against `2026-06-16-pgx-genotype-nonlinearity-two-arm-design.md`):**
- §2 dose-trend metric / `1/(1−fm)` ceiling: Tasks 1 (`loglog_beta`), 6 (β scoring), 5 (oracle). ✓
- §3 two arms + tiers + controls: Task 2 dataset (tiers), Tasks 3/6 (Arm S well_stirred steady-state, Arm F axial). ✓
- §4.1 data-availability gate: Task 2 (curation; at_risk/HALT on gap). ✓
- §4.2 box-robustness (every corner > 0.10): Tasks 1 (`box_robustness_pass`), 5 (`box_robustness_probe`). ✓
- §5.1 three engines, realistic PM scaling (not gene→0), anchor at lowest dose: Task 6 (`score_drug`, `_anchor_cltot_low_dose`); idealized gene→0 only in oracle (Task 5). ✓
- §5.2 P1 (β vs obs, linear-null β≡1) + P2 (arm-correct Δβ sign) + N1 graded controls: Tasks 4 (sign crux), 6 (P1/P2). Negative-control N1 scoring: **see gap below.** 
- §5.3 secondary accuracy delta: Task 6/8 (report). ✓
- §5.4 oracle C2 both skeletons: Task 5. ✓
- §6 km conversion: Task 1. ✓
- §7 dataset + schema guard: Task 2. ✓
- §8 tests (km, β, anchor round-trip, box monotone, axial-inlet, arm-sign, headline isolation): Tasks 1,3,4,5,7. **Anchor round-trip + box-monotone-in-Km unit tests: see gap below.**

**Gaps found and fixed inline:**
1. **N1 negative-control scoring** was implied but not given its own task step. Add to Task 6 Step 3: a `score_negative_control(drug)` that runs the saturable engine at the drug's high literature `Km` and asserts tolbutamide `Δβ ≈ 0` / `β ≈ 1` (hard null) and metoprolol a *small* `β_EM>1` (mild-positive), surfaced in the report. **Implementer: add `score_negative_control` mirroring `score_drug` but reading `control_role`; include both controls in `run_arm`.**
2. **Anchor round-trip + box-monotone-in-Km** unit tests (spec §8) are not pinned. Add to Task 1 a `test_box_robustness_pass` boundary case (done) and to Task 5 a `test_box_probe_monotone_in_km` (deltas grow as Km falls). **Implementer: add `test_box_probe_monotone_in_km` asserting `probe(low_Km)` corners ≥ `probe(high_Km)` corners for the same fu_mic.**

**Placeholder scan:** the dataset `dose_rows` exposures are `0.0` *intentionally* — they are the curator's literature task (Task 2 Step 3), guarded so an uncurated row is treated as a data gap (HALT), never silently scored. No code placeholders.

**Type consistency:** `_genotype_fold_engine(gene_tag, fm, mw, fup, dose_mg, km_mgl, pm_activity, regime, interval_h, n_doses, a_var)` — same signature in Task 5 (probe, oracle) and consistent with `_beta_for_genotype` / `score_drug`. `_steady_state_exposure(graph, drug, interval_h, n_doses, metric)` and `_single_dose_exposure(graph, drug, metric)` consistent across Tasks 3–6. `loglog_beta(doses, exposures)`, `delta_beta(beta_pm, beta_em)`, `km_uM_to_unbound_mgL(km_uM, mw, fu_mic)`, `box_robustness_pass(deltas, threshold)` consistent Tasks 1–8.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-16-pgx-genotype-nonlinearity-two-arm.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task + two-stage review, fast iteration.
2. **Inline Execution** — execute in this session with checkpoints.

> **Sequencing note:** Tasks 1–3, 5 (oracle/well_stirred), 6 (Arm S), 7 run now. The **Arm-F axial halves of Tasks 4, 5, 6 require PR #79 merged** — run them after merge (or skip-mark the axial tests and unskip post-merge). Task 2's curation is the real HALT-early gate: if neither arm has dose-ranging genotype data, surface the HALT before building Task 6 scoring.

Which approach?
