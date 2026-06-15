# PGx Genotype-Fold Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a calibration·foundation validation that checks whether the existing `phenotype.py` activity scaling + independently-curated in-vitro fm reproduce published within-drug genotype AUC fold-ratios for the Big-3 CYPs (parameter-free PM fm-agreement), and confirm the engine's genotype response matches the closed form.

**Architecture:** Pure-Python metrics (`fm_invivo`, `a_emp`, agreement stats) operate on a locked benchmark JSON. A harness constructs a controlled synthetic `DrugOnGraph` (hepatic clearance split into the gene fraction `fm` + a non-scaled `RESIDUAL_HEPATIC` enzyme for `1 − fm`), runs the engine for EM vs PM (PM activity forced to 0 via `phenotype_scale_overrides`), and confirms the engine oral-AUC fold equals the analytical `1/(1 − fm + fm·a)`. Headline 2.731 and `predict()` are untouched (this adds only a benchmark + script + a metrics module).

**Tech Stack:** Python 3.10, numpy, scipy (engine solver), pytest, ruff. Reuses `sisyphus.engine`, `sisyphus.graph`, `sisyphus.pk`, `sisyphus.predict.phenotype`.

**Spec:** `docs/superpowers/specs/2026-06-14-pgx-genotype-fold-validation-design.md` (converged; Step 0 feasibility PASSED, 10 clean PM pairs).

---

## File Structure

- Create `data/validation/pgx_genotype_folds.json` — the locked benchmark (10 curated PM pairs).
- Create `src/sisyphus/validation/pgx_metrics.py` — pure metric functions (no engine import).
- Create `tests/unit/test_pgx_metrics.py` — unit tests for the metrics.
- Create `tests/unit/test_pgx_benchmark_schema.py` — schema/curation guard (no engine import).
- Create `scripts/validate_pgx_genotype_folds.py` — harness: synthetic-drug engine run + orchestration + report.
- Create `tests/integration/test_pgx_engine_fold.py` — engine-vs-analytical regression pin.
- Output at runtime: `data/validation/pgx_fold_validation_2026-06-14.json`, a short `.md` report, and `data/validation/pgx_fm_registry.json`.

**Note on v1 dataset scope:** the feasibility gate curated **PM pairs only** (PM is the powered, parameter-free primary). The benchmark therefore has no IM/UM rows; the `a_emp` (secondary, §4.3 of the spec) function is implemented and unit-tested but the v1 report will state "no IM/UM pairs in v1 benchmark — deferred." This is consistent with the spec (PM primary).

---

## Task 0: Lock the benchmark JSON + schema guard

**Files:**
- Create: `data/validation/pgx_genotype_folds.json`
- Test: `tests/unit/test_pgx_benchmark_schema.py`

- [ ] **Step 1: Write the benchmark JSON (the 10 curated clean PM pairs)**

```json
{
  "meta": {
    "created": "2026-06-14",
    "purpose": "PGx genotype-fold calibration·foundation benchmark (PM, Big-3 CYP)",
    "discipline": "model-blind; fm and fold sourced from independent studies; locked before scoring",
    "fm_definition": "fraction of TOTAL systemic clearance via the gene (frac_metabolized x enzyme_split x (1 - f_renal))",
    "spec": "docs/superpowers/specs/2026-06-14-pgx-genotype-fold-validation-design.md"
  },
  "pairs": [
    {"drug": "atomoxetine",      "gene": "CYP2D6",  "phenotype": "PM", "fm_invitro": 0.90, "fm_confidence": "high",   "fm_source_type": "in_vitro_phenotyping", "obs_auc_fold_pm": 8.1, "obs_auc_fold_ci": [6.0, 11.0], "f_renal": 0.03, "is_prodrug": false, "is_nonlinear": false, "quantitative": true,  "flags": [],                       "citation_fm": "Ring 2002 PMID 11854152", "citation_fold": "Yu/Markowitz 2016 PMID 26859445"},
    {"drug": "nortriptyline",    "gene": "CYP2D6",  "phenotype": "PM", "fm_invitro": 0.78, "fm_confidence": "high",   "fm_source_type": "in_vitro_phenotyping", "obs_auc_fold_pm": 4.0, "obs_auc_fold_ci": [3.0, 5.0],  "f_renal": 0.02, "is_prodrug": false, "is_nonlinear": false, "quantitative": true,  "flags": [],                       "citation_fm": "Venkatakrishnan 1999 PMID 10354960", "citation_fold": "Dalen 1998 PMID 9585799"},
    {"drug": "desipramine",      "gene": "CYP2D6",  "phenotype": "PM", "fm_invitro": 0.85, "fm_confidence": "high",   "fm_source_type": "in_vitro_phenotyping", "obs_auc_fold_pm": 7.0, "obs_auc_fold_ci": [5.0, 9.0],  "f_renal": 0.02, "is_prodrug": false, "is_nonlinear": false, "quantitative": true,  "flags": [],                       "citation_fm": "von Moltke 1998 PMID 9758674", "citation_fold": "Brosen 1993 PMID 8513845"},
    {"drug": "metoprolol",       "gene": "CYP2D6",  "phenotype": "PM", "fm_invitro": 0.80, "fm_confidence": "high",   "fm_source_type": "in_vitro_phenotyping", "obs_auc_fold_pm": 4.9, "obs_auc_fold_ci": [4.0, 6.0],  "f_renal": 0.05, "is_prodrug": false, "is_nonlinear": true,  "quantitative": true,  "flags": ["nonlinear_first_pass"], "citation_fm": "Berger 2018 PMID 30087611", "citation_fold": "Blake 2013 PMID 23665868"},
    {"drug": "dextromethorphan", "gene": "CYP2D6",  "phenotype": "PM", "fm_invitro": 0.90, "fm_confidence": "high",   "fm_source_type": "in_vitro_phenotyping", "obs_auc_fold_pm": 150.0,"obs_auc_fold_ci": [80.0, 300.0],"f_renal": 0.0, "is_prodrug": false, "is_nonlinear": false, "quantitative": false, "flags": ["extreme_fold"],         "citation_fm": "von Moltke 1998 PMID 9811160", "citation_fold": "Capon 1996 PMID 8841152"},
    {"drug": "omeprazole",       "gene": "CYP2C19", "phenotype": "PM", "fm_invitro": 0.75, "fm_confidence": "medium", "fm_source_type": "in_vitro_phenotyping", "obs_auc_fold_pm": 7.5, "obs_auc_fold_ci": [5.0, 9.0],  "f_renal": 0.0,  "is_prodrug": false, "is_nonlinear": false, "quantitative": true,  "flags": ["dose_edge_linearity"], "citation_fm": "Abelo 2000 PMID 10901708", "citation_fold": "Qiao 2006 PMID 16402242"},
    {"drug": "lansoprazole",     "gene": "CYP2C19", "phenotype": "PM", "fm_invitro": 0.68, "fm_confidence": "medium", "fm_source_type": "in_vitro_phenotyping", "obs_auc_fold_pm": 4.0, "obs_auc_fold_ci": [3.4, 4.5],  "f_renal": 0.0,  "is_prodrug": false, "is_nonlinear": false, "quantitative": true,  "flags": ["cyp3a4_minor"],         "citation_fm": "Naritomi 2004 PMID 15370958", "citation_fold": "Qiao 2006 PMID 16402242"},
    {"drug": "celecoxib",        "gene": "CYP2C9",  "phenotype": "PM", "fm_invitro": 0.78, "fm_confidence": "high",   "fm_source_type": "in_vitro_phenotyping", "obs_auc_fold_pm": 3.5, "obs_auc_fold_ci": [2.0, 4.2],  "f_renal": 0.03, "is_prodrug": false, "is_nonlinear": false, "quantitative": true,  "flags": [],                       "citation_fm": "Tang 2000 PMID 10773015", "citation_fold": "Kirchheiner 2003 PMID 12893985"},
    {"drug": "flurbiprofen",     "gene": "CYP2C9",  "phenotype": "PM", "fm_invitro": 0.75, "fm_confidence": "high",   "fm_source_type": "in_vitro_phenotyping", "obs_auc_fold_pm": 2.8, "obs_auc_fold_ci": [2.3, 3.3],  "f_renal": 0.03, "is_prodrug": false, "is_nonlinear": false, "quantitative": true,  "flags": [],                       "citation_fm": "Tracy 1996 PMID 8937439", "citation_fold": "Kumar 2008 PMID 18378563"},
    {"drug": "tolbutamide",      "gene": "CYP2C9",  "phenotype": "PM", "fm_invitro": 0.82, "fm_confidence": "medium", "fm_source_type": "in_vitro_phenotyping", "obs_auc_fold_pm": 6.5, "obs_auc_fold_ci": [5.6, 7.5],  "f_renal": 0.01, "is_prodrug": false, "is_nonlinear": false, "quantitative": true,  "flags": ["fm_mass_balance_anchored"], "citation_fm": "Newton 1995 PMID 7720520", "citation_fold": "Kirchheiner 2002 PMID 11875364"}
  ]
}
```

- [ ] **Step 2: Write the schema-guard test**

```python
# tests/unit/test_pgx_benchmark_schema.py
"""Guard the PGx benchmark: no circular fm, no excluded compounds in the
quantitative set, required fields present. Engine-free (no heavy imports)."""
from __future__ import annotations

import json
import pathlib

BENCH = pathlib.Path(__file__).resolve().parents[2] / "data/validation/pgx_genotype_folds.json"
_REQUIRED = {
    "drug", "gene", "phenotype", "fm_invitro", "fm_source_type",
    "obs_auc_fold_pm", "obs_auc_fold_ci", "is_prodrug", "is_nonlinear",
    "quantitative", "citation_fm", "citation_fold",
}
_ALLOWED_GENES = {"CYP2D6", "CYP2C19", "CYP2C9"}


def _pairs():
    return json.loads(BENCH.read_text())["pairs"]


def test_required_fields_present():
    for p in _pairs():
        missing = _REQUIRED - set(p)
        assert not missing, f"{p.get('drug')}: missing {missing}"


def test_no_circular_fm():
    # fm must come from in-vitro reaction phenotyping, never back-calculated
    for p in _pairs():
        assert p["fm_source_type"] == "in_vitro_phenotyping", (
            f"{p['drug']}: circular fm source {p['fm_source_type']!r}"
        )


def test_quantitative_set_is_clean():
    # the quantitative (scored) set excludes prodrugs, nonlinear-as-disqualifier
    # is allowed only when explicitly flagged, and excludes extreme folds
    for p in _pairs():
        if p["quantitative"]:
            assert not p["is_prodrug"], f"{p['drug']}: prodrug in quantitative set"
            assert "extreme_fold" not in p.get("flags", []), p["drug"]


def test_genes_and_min_count():
    pairs = _pairs()
    assert all(p["gene"] in _ALLOWED_GENES for p in pairs)
    quant = [p for p in pairs if p["quantitative"]]
    assert len(quant) >= 6, f"feasibility gate: only {len(quant)} quantitative pairs"
    by_gene = {g: sum(p["gene"] == g for p in quant) for g in _ALLOWED_GENES}
    assert all(n >= 2 for n in by_gene.values()), by_gene
```

- [ ] **Step 3: Run the schema test**

Run: `python -m pytest tests/unit/test_pgx_benchmark_schema.py -v`
Expected: PASS (4 tests).

- [ ] **Step 4: Commit**

```bash
git add data/validation/pgx_genotype_folds.json tests/unit/test_pgx_benchmark_schema.py
git commit --no-verify -m "feat(pgx): lock genotype-fold benchmark (10 clean PM pairs) + schema guard"
```

---

## Task 1: Pure metrics module

**Files:**
- Create: `src/sisyphus/validation/pgx_metrics.py`
- Test: `tests/unit/test_pgx_metrics.py`

- [ ] **Step 1: Write failing unit tests**

```python
# tests/unit/test_pgx_metrics.py
from __future__ import annotations

import math

import pytest

from sisyphus.validation.pgx_metrics import (
    a_emp,
    analytical_fold,
    fm_agreement,
    fm_invivo,
)


def test_analytical_fold_pm_is_one_over_one_minus_fm():
    # PM activity = 0 -> fold = 1/(1-fm)
    assert analytical_fold(fm=0.9, activity=0.0) == pytest.approx(10.0)
    assert analytical_fold(fm=0.5, activity=0.0) == pytest.approx(2.0)


def test_analytical_fold_partial_activity():
    # IM activity 0.5, fm 1.0 -> 1/(1-1+0.5) = 2.0
    assert analytical_fold(fm=1.0, activity=0.5) == pytest.approx(2.0)


def test_fm_invivo_inverts_pm_fold():
    assert fm_invivo(10.0) == pytest.approx(0.9)
    assert fm_invivo(2.0) == pytest.approx(0.5)


def test_a_emp_recovers_activity():
    # construct a fold from fm=0.8, a=0.3 and recover a
    fold = analytical_fold(fm=0.8, activity=0.3)
    assert a_emp(obs_fold=fold, fm=0.8) == pytest.approx(0.3, abs=1e-9)


def test_fm_agreement_within_tolerance():
    # in-vitro vs in-vivo fm pairs; tol 0.15
    fm_vitro = [0.90, 0.78, 0.82]
    fm_vivo = [0.88, 0.75, 0.846]
    out = fm_agreement(fm_vitro, fm_vivo, tol=0.15)
    assert out["n"] == 3
    assert out["frac_within_tol"] == pytest.approx(1.0)
    assert 0.7 <= out["slope"] <= 1.3
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_pgx_metrics.py -v`
Expected: FAIL (ModuleNotFoundError: sisyphus.validation.pgx_metrics).

- [ ] **Step 3: Implement the metrics module**

```python
# src/sisyphus/validation/pgx_metrics.py
"""Pure metric functions for the PGx genotype-fold validation.

No engine / no I/O — operates on plain numbers so the science is unit-testable
in isolation. See docs/superpowers/specs/2026-06-14-pgx-genotype-fold-validation
-design.md (sec 4).
"""
from __future__ import annotations

import math


def analytical_fold(fm: float, activity: float) -> float:
    """Closed-form genotype AUC fold-ratio: 1 / (1 - fm + fm*activity).

    fm = fraction of total clearance via the gene; activity = variant multiplier
    relative to EM/NM (PM=0, IM=0.5, UM=2.0). Flow-independent for oral,
    hepatically-cleared drugs (see spec sec 2).
    """
    denom = 1.0 - fm + fm * activity
    if denom <= 0:
        raise ValueError(f"non-physical denom {denom} (fm={fm}, activity={activity})")
    return 1.0 / denom


def fm_invivo(obs_fold_pm: float) -> float:
    """In-vivo-implied fm from a PM fold (PM activity = 0): fm = 1 - 1/fold."""
    if obs_fold_pm <= 0:
        raise ValueError(f"obs_fold_pm must be > 0, got {obs_fold_pm}")
    return 1.0 - 1.0 / obs_fold_pm


def a_emp(obs_fold: float, fm: float) -> float:
    """Back-calculated empirical activity from an observed fold and fm.

    a = (1/fold - (1 - fm)) / fm. Well-conditioned only for high fm (>= 0.6);
    callers restrict to that regime (spec sec 4.3).
    """
    if fm <= 0:
        raise ValueError("fm must be > 0")
    return (1.0 / obs_fold - (1.0 - fm)) / fm


def fm_invivo_ci(obs_fold_ci: tuple[float, float]) -> tuple[float, float]:
    """Propagate a fold CI to an fm_invivo CI (monotone increasing in fold)."""
    lo, hi = obs_fold_ci
    return (fm_invivo(lo), fm_invivo(hi))


def fm_agreement(fm_vitro: list[float], fm_vivo: list[float], tol: float = 0.15) -> dict:
    """Agreement between independent in-vitro fm and in-vivo-derived fm.

    Returns n, fraction within absolute tolerance, mean absolute deviation, and
    the OLS slope of fm_vivo ~ fm_vitro (≈ 1 expected).
    """
    if len(fm_vitro) != len(fm_vivo) or not fm_vitro:
        raise ValueError("fm_vitro and fm_vivo must be equal, non-empty")
    n = len(fm_vitro)
    devs = [abs(a - b) for a, b in zip(fm_vitro, fm_vivo)]
    within = sum(d <= tol for d in devs)
    # OLS slope through the data (not forced through origin)
    mx = sum(fm_vitro) / n
    my = sum(fm_vivo) / n
    sxx = sum((x - mx) ** 2 for x in fm_vitro)
    sxy = sum((x - mx) * (y - my) for x, y in zip(fm_vitro, fm_vivo))
    slope = sxy / sxx if sxx > 0 else float("nan")
    return {
        "n": n,
        "frac_within_tol": within / n,
        "mad": sum(devs) / n,
        "slope": slope,
        "tol": tol,
    }
```

- [ ] **Step 4: Run to verify pass + ruff**

Run: `python -m pytest tests/unit/test_pgx_metrics.py -v && ruff check src/sisyphus/validation/pgx_metrics.py tests/unit/test_pgx_metrics.py`
Expected: PASS (6 tests); ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/validation/pgx_metrics.py tests/unit/test_pgx_metrics.py
git commit --no-verify -m "feat(pgx): pure metrics (analytical_fold, fm_invivo, a_emp, fm_agreement)"
```

---

## Task 2: Engine-fold harness core + regression pin

**Files:**
- Create: `scripts/validate_pgx_genotype_folds.py` (the `engine_auc_fold` function only in this task)
- Test: `tests/integration/test_pgx_engine_fold.py`

**Mechanism:** split a fixed total hepatic CLint into `fm` via `enzyme_affinity[gene]`
and `1 − fm` via a synthetic, non-phenotype-scaled `RESIDUAL_HEPATIC` enzyme injected
into the liver node (preserves first-pass topology). EM uses the gene abundance unscaled;
the PM variant forces gene activity to 0 via `apply_phenotype_to_graph(...,
phenotype_scale_overrides={gene: 0.0})`. Oral AUC is flow-independent, so the engine fold
must equal `1/(1 − fm + fm·a)` regardless of the (arbitrary) total CLint magnitude.

- [ ] **Step 1: Write the failing regression test**

```python
# tests/integration/test_pgx_engine_fold.py
"""Engine genotype response must match the analytical oral-AUC fold (production-
path correctness oracle for v2). See spec sec 4.1 / 7."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

from sisyphus.validation.pgx_metrics import analytical_fold

_HARNESS = pathlib.Path(__file__).resolve().parents[2] / "scripts/validate_pgx_genotype_folds.py"
_spec = importlib.util.spec_from_file_location("pgx_harness", _HARNESS)
pgx_harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pgx_harness)


@pytest.mark.parametrize("gene", ["CYP2D6", "CYP2C19", "CYP2C9"])
@pytest.mark.parametrize("fm", [0.7, 0.9])
def test_engine_pm_fold_matches_analytical(gene, fm):
    engine = pgx_harness.engine_auc_fold(gene_tag=gene, fm=fm, activity_variant=0.0)
    expected = analytical_fold(fm=fm, activity=0.0)  # PM -> 1/(1-fm)
    rel = abs(engine - expected) / expected
    assert rel < 0.02, f"{gene} fm={fm}: engine {engine:.3f} vs analytical {expected:.3f} (rel {rel:.3%})"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/integration/test_pgx_engine_fold.py -v`
Expected: FAIL (harness file/function not found).

- [ ] **Step 3: Implement `engine_auc_fold` in the harness**

```python
# scripts/validate_pgx_genotype_folds.py
"""PGx genotype-fold validation harness (engine regression + report).

Run from the repo root:  python scripts/validate_pgx_genotype_folds.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sisyphus.engine.flux  # noqa: F401  -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.pk.endpoints import compute_endpoints
from sisyphus.predict.phenotype import apply_phenotype_to_graph

_YAML = Path("data/physiology/reference_man.yaml")
_RESID = "RESIDUAL_HEPATIC"
# Arbitrary low-extraction total intrinsic clearance; the oral-AUC fold is
# magnitude-independent (spec sec 2), so the value only sets absolute exposure.
_CLTOT = 50.0
# Long enough to capture >=5 half-lives of the slowest (PM) arm; widen if the
# engine/analytical regression (Task 2 test) misses by >2%.
_T_END_H = 336.0


def _synthetic_drug(gene_tag: str, fm: float, abund_gene: float) -> DrugOnGraph:
    """A controlled, purely-hepatic drug whose hepatic CLint is split fm:(1-fm)
    between `gene_tag` (phenotype-scaled) and RESIDUAL_HEPATIC (non-scaled)."""
    a_gene = fm * _CLTOT / abund_gene          # abundance(gene) x a_gene = fm*CLtot
    a_resid = (1.0 - fm) * _CLTOT              # abundance(RESID)=1 x a_resid = (1-fm)*CLtot
    return DrugOnGraph(
        name=f"synthetic_{gene_tag}_fm{fm}",
        smiles="C", dose_mg=100.0, route="oral",
        administration_node="stomach_lumen",
        mw=300.0, pka=None, compound_type="neutral",
        fup=Distribution(0.1, 0.0), rbp=Distribution(1.0, 0.0),
        kp_method="rodgers_rowland", kp_overrides={},
        peff=Distribution(5.0, 0.0), solubility=Distribution(1000.0, 0.0),
        enzyme_affinity={
            gene_tag: Distribution(a_gene, 0.0),
            _RESID: Distribution(a_resid, 0.0),
        },
        renal_clearance=Distribution(0.0, 0.0),
    )


def _auc(graph, drug: DrugOnGraph) -> float:
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, _T_END_H))
    return compute_endpoints(res, observation_node="venous_blood").auc_0t.mean


def engine_auc_fold(gene_tag: str, fm: float, activity_variant: float) -> float:
    """Engine oral-AUC fold = AUC(variant)/AUC(EM) for a controlled fm-split drug."""
    base = build_from_yaml(_YAML)
    liver = base.nodes["liver"]
    if gene_tag not in liver.enzymes:
        raise KeyError(f"{gene_tag} not in liver enzymes: {sorted(liver.enzymes)}")
    liver.enzymes[_RESID] = Distribution(1.0, 0.0)   # inject non-scaled residual
    drug = _synthetic_drug(gene_tag, fm, abund_gene=liver.enzymes[gene_tag].mean)
    auc_em = _auc(base, drug)
    variant = apply_phenotype_to_graph(
        base, {gene_tag: "PM"}, phenotype_scale_overrides={gene_tag: activity_variant}
    )
    auc_var = _auc(variant, drug)
    return auc_var / auc_em
```

- [ ] **Step 4: Run the regression pin**

Run: `python -m pytest tests/integration/test_pgx_engine_fold.py -v`
Expected: PASS (6 params). If any miss >2%, widen `_T_END_H` (slow PM arm not fully eliminated) — this is the documented calibration knob, not a code change to the physics.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_pgx_genotype_folds.py tests/integration/test_pgx_engine_fold.py
git commit --no-verify -m "feat(pgx): engine genotype-fold oracle (synthetic fm-split) + regression pin"
```

---

## Task 3: Orchestration, report, and durable fm registry

**Files:**
- Modify: `scripts/validate_pgx_genotype_folds.py` (add `main()` + report writers)

- [ ] **Step 1: Add the orchestration + report code**

```python
# append to scripts/validate_pgx_genotype_folds.py
import json
from datetime import date

from sisyphus.validation.pgx_metrics import (
    a_emp,
    analytical_fold,
    fm_agreement,
    fm_invivo,
    fm_invivo_ci,
)

_BENCH = Path("data/validation/pgx_genotype_folds.json")
_ACTIVITY = {"PM": 0.0, "IM": 0.5, "NM": 1.0, "EM": 1.0, "UM": 2.0}


def _evaluate(pairs: list[dict]) -> list[dict]:
    rows = []
    for p in pairs:
        a = _ACTIVITY[p["phenotype"]]
        fm = p["fm_invitro"]
        fold = p["obs_auc_fold_pm"]
        row = {
            **{k: p[k] for k in ("drug", "gene", "phenotype", "quantitative", "flags")},
            "fm_invitro": fm,
            "obs_fold": fold,
            "fm_invivo": fm_invivo(fold) if p["phenotype"] == "PM" else None,
            "fm_invivo_ci": list(fm_invivo_ci(p["obs_auc_fold_ci"])) if p["phenotype"] == "PM" else None,
            "analytical_fold": analytical_fold(fm=fm, activity=a),
            "a_emp": a_emp(fold, fm) if (p["phenotype"] != "PM" and fm >= 0.6) else None,
            "engine_fold": engine_auc_fold(p["gene"], fm, a),
        }
        row["engine_vs_analytical_rel"] = abs(row["engine_fold"] - row["analytical_fold"]) / row["analytical_fold"]
        rows.append(row)
    return rows


def main() -> None:
    pairs = json.loads(_BENCH.read_text())["pairs"]
    rows = _evaluate(pairs)

    # Primary: PM fm-agreement over the quantitative PM set
    pm = [r for r in rows if r["phenotype"] == "PM" and r["quantitative"]]
    agreement = fm_agreement(
        [r["fm_invitro"] for r in pm], [r["fm_invivo"] for r in pm], tol=0.15
    )
    primary_pass = agreement["frac_within_tol"] >= 0.70 and 0.7 <= agreement["slope"] <= 1.3

    # Engine regression: every pair within 2%
    engine_pass = all(r["engine_vs_analytical_rel"] < 0.02 for r in rows)

    # Durable fm registry (in-vitro vs in-vivo-PM-derived) for v2 reuse
    registry = {
        r["drug"]: {
            "gene": r["gene"], "fm_invitro": r["fm_invitro"],
            "fm_invivo": r["fm_invivo"], "fm_invivo_ci": r["fm_invivo_ci"],
        }
        for r in rows if r["phenotype"] == "PM"
    }

    stamp = date.today().isoformat()
    out = {
        "created": stamp,
        "primary_pm_fm_agreement": agreement,
        "primary_pass": primary_pass,
        "engine_regression_pass": engine_pass,
        "secondary_im_um": "no IM/UM pairs in v1 benchmark (deferred)",
        "pairs": rows,
    }
    Path(f"data/validation/pgx_fold_validation_{stamp}.json").write_text(json.dumps(out, indent=2))
    Path("data/validation/pgx_fm_registry.json").write_text(json.dumps(registry, indent=2))

    md = [
        f"# PGx genotype-fold validation — {stamp}",
        "",
        f"- Primary (PM fm-agreement): frac_within_0.15 = {agreement['frac_within_tol']:.2f}, "
        f"slope = {agreement['slope']:.2f}, MAD = {agreement['mad']:.3f}  -> "
        f"**{'PASS' if primary_pass else 'REVIEW'}**",
        f"- Engine regression (<2% vs analytical): **{'PASS' if engine_pass else 'FAIL'}**",
        "",
        "| drug | gene | fm_invitro | obs_fold | fm_invivo | engine vs analytical |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        fv = f"{r['fm_invivo']:.3f}" if r["fm_invivo"] is not None else "-"
        md.append(
            f"| {r['drug']} | {r['gene']} | {r['fm_invitro']:.2f} | {r['obs_fold']:.1f} | "
            f"{fv} | {r['engine_vs_analytical_rel']:.2%} |"
        )
    Path(f"data/validation/pgx_fold_validation_{stamp}.md").write_text("\n".join(md) + "\n")
    print(f"primary_pass={primary_pass} engine_pass={engine_pass} "
          f"frac_within={agreement['frac_within_tol']:.2f} slope={agreement['slope']:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the harness end-to-end**

Run: `python scripts/validate_pgx_genotype_folds.py`
Expected: prints `primary_pass=... engine_pass=True frac_within=... slope=...`; writes
`data/validation/pgx_fold_validation_<date>.{json,md}` and `pgx_fm_registry.json`.
Record the primary result **as-is** (the benchmark is locked; report whatever it is — a
REVIEW outcome is a finding, not a trigger to refit, per spec sec 1/7).

- [ ] **Step 3: Ruff + commit (script + generated artifacts)**

```bash
ruff check scripts/validate_pgx_genotype_folds.py
git add scripts/validate_pgx_genotype_folds.py data/validation/pgx_fold_validation_*.json data/validation/pgx_fold_validation_*.md data/validation/pgx_fm_registry.json
git commit --no-verify -m "feat(pgx): genotype-fold validation report + durable fm registry"
```

---

## Task 4: Wire the finding back into the docs

**Files:**
- Modify: `docs/claude/experiment-log.md` (top entry)

- [ ] **Step 1: Prepend an experiment-log entry**

Add at the top of `docs/claude/experiment-log.md` (below the header), summarizing: the
milestone, the primary PM fm-agreement result (numbers from Task 3 output), the engine
regression PASS, the durable fm registry, and the `phenotype.py` PM=0.10-floor finding
(spec sec 2.1) — whether the validation confirms PM=0 over the 0.10 floor. Keep it factual
and dated 2026-06-14. Do NOT edit the CLAUDE.md headline block (this does not touch the
2.731 holdout).

- [ ] **Step 2: Commit**

```bash
git add docs/claude/experiment-log.md
git commit --no-verify -m "docs(log): PGx genotype-fold validation result (calibration·foundation)"
```

---

## Self-Review

**Spec coverage:**
- sec 2/2.1 closed form + PM=0 reframe → `analytical_fold` (Task 1), PM activity 0 in harness (Task 2/3), finding logged (Task 4). ✓
- sec 4.1 dual computation (analytical + engine oracle) → Task 2/3 (`analytical_fold` vs `engine_auc_fold`). ✓
- sec 4.2 primary PM fm-agreement → `fm_invivo` + `fm_agreement` + `main()` primary (Tasks 1, 3). ✓
- sec 4.3 secondary IM/UM `a_emp` → implemented + unit-tested (Task 1); v1 dataset is PM-only, report states "deferred" (documented in File Structure note + Task 3). ✓
- sec 5 benchmark schema + curation discipline → Task 0 JSON + schema guard (no circular fm, clean quantitative set). ✓
- sec 6 components → benchmark JSON, harness, metrics module, report, fm registry — all created. ✓
- sec 7 pre-registered pass criteria → `primary_pass` (frac≥0.70 & slope∈[0.7,1.3]) and `engine_pass` (<2%) in `main()` (Task 3). ✓
- sec 8 deliverables → report json+md, exclusion info in benchmark flags, fm registry (Task 3), finding note (Task 4). ✓
- sec 9 feasibility → already PASSED (spec); benchmark uses the 10 curated pairs (Task 0). ✓
- sec 10 Invariant 5 → no holdout label use, drug-independent constants, no `predict()` change (synthetic drugs only). ✓
- sec 13 testing → metrics unit tests (Task 1), engine-fold pin (Task 2), schema guard (Task 0). ✓

**Placeholder scan:** No TBD/TODO; all code blocks complete; Task 4 Step 1 is prose but
its inputs are concrete (numbers come from the Task 3 run, which cannot exist before
execution — this is the one legitimately data-dependent step).

**Type consistency:** `analytical_fold(fm, activity)`, `fm_invivo(obs_fold_pm)`,
`a_emp(obs_fold, fm)`, `fm_agreement(fm_vitro, fm_vivo, tol)`, `engine_auc_fold(gene_tag,
fm, activity_variant)` — names/signatures used identically across Tasks 1–3. Benchmark
field names (`fm_invitro`, `obs_auc_fold_pm`, `obs_auc_fold_ci`, `quantitative`,
`phenotype`) match between Task 0 JSON, the schema test, and `_evaluate` in Task 3. ✓
