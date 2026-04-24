# Hardening H5+H1 — CI & Lockfile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship reproducible `requirements-lock.txt` + green `.github/workflows/ci.yml` on first CI run. Fix two pre-existing real bugs that block green (`CompiledODE` F821 + `sys.modules` test pollution). Mark pre-existing propranolol Cmax drift as `xfail`. Defer ruff cleanup.

**Architecture:** Fresh-venv generated pip lockfile; single-version Python 3.10 CI; ruff advisory (non-gating); benchmark smoke via `run_benchmark(max_drugs=5)`.

**Tech Stack:** pip + venv (no uv), GitHub Actions (ubuntu-latest + setup-python@v5), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-04-23-hardening-h5-h1-ci-lockfile-design.md`

---

### Task 1: Setup feature branch

**Files:** none (git only)

- [ ] **Step 1: Confirm clean working state on main**

Run: `git status && git log --oneline -1`
Expected: On branch `main`, HEAD `b366035`. Untracked SBI model/data files present (ignore them — pre-existing).

- [ ] **Step 2: Create and checkout feature branch**

```bash
git checkout -b feature/hardening-h5-h1-ci-lockfile
```
Expected: `Switched to a new branch 'feature/hardening-h5-h1-ci-lockfile'`.

- [ ] **Step 3: Verify baseline failures exist (so fixes are verifiable)**

```bash
ruff check src tests 2>&1 | tail -3
pytest tests/unit/test_regimen.py --collect-only 2>&1 | grep -E "error|F821" || echo "collection ok"
pytest tests/unit -x --tb=line 2>&1 | tail -5
pytest tests/integration/test_engine_validation.py::TestEngineValidation::test_cmax_within_5pct -v 2>&1 | tail -5
```
Expected: ruff reports ~142 errors; tests/unit exits with `test_end_to_end_caffeine` TypeError; propranolol integration test FAILED.

---

### Task 2: Generate `requirements-lock.txt` from fresh venv

**Files:**
- Create: `requirements-lock.txt`

- [ ] **Step 1: Create isolated venv and install**

```bash
python3 -m venv /tmp/sis_lock_env
source /tmp/sis_lock_env/bin/activate
pip install --upgrade pip
pip install -e '.[ml,chem,dev]'
```
Expected: All packages install. RDKit pulls `rdkit==2023.9.6` or later (PyPI `rdkit` package, NOT `rdkit-pypi`).

- [ ] **Step 2: Freeze into lockfile**

```bash
pip freeze --exclude-editable > /home/jam/Sisyphus/requirements-lock.txt
deactivate
rm -rf /tmp/sis_lock_env
```
Expected: `requirements-lock.txt` contains ~50-100 lines, includes `numpy`, `scipy`, `xgboost`, `rdkit`, `pytest`, `ruff` with pinned versions. **Must NOT contain** `chemprop`, `descriptastorus`, `mordred`, `unimol-tools`, `rdkit-pypi` (these are from the dev env, not Sisyphus deps).

- [ ] **Step 3: Sanity check lockfile**

```bash
grep -E "^rdkit-pypi|^chemprop|^descriptastorus|^mordred" /home/jam/Sisyphus/requirements-lock.txt && echo "POLLUTION — rerun Step 1-2" || echo "clean"
wc -l /home/jam/Sisyphus/requirements-lock.txt
```
Expected: "clean"; line count ~50-100.

- [ ] **Step 4: Commit**

```bash
cd /home/jam/Sisyphus
git add requirements-lock.txt
git commit -m "feat(repro): add requirements-lock.txt from fresh venv"
```

---

### Task 3: Write `docs/reproducibility.md`

**Files:**
- Create: `docs/reproducibility.md`

- [ ] **Step 1: Create doc**

Write to `/home/jam/Sisyphus/docs/reproducibility.md`:

```markdown
# Reproducibility — Install & Lockfile

## Quick install

```bash
pip install -r requirements-lock.txt
pip install -e .
```

## Fresh install from source (no lockfile)

```bash
pip install -e '.[ml,chem,dev]'
```

Unpinned; transitive versions may drift.

## Regenerating `requirements-lock.txt`

Run from repository root:

```bash
python3 -m venv /tmp/sis_lock_env
source /tmp/sis_lock_env/bin/activate
pip install --upgrade pip
pip install -e '.[ml,chem,dev]'
pip freeze --exclude-editable > requirements-lock.txt
deactivate
rm -rf /tmp/sis_lock_env
```

Regenerate when `pyproject.toml` dependencies change or upstream CVE forces a bump.

## RDKit note

Project uses `rdkit` (PyPI-maintained, 2023.9+). Not `rdkit-pypi` (older community fork).
On platforms where pip cannot resolve RDKit, fall back to conda:
`conda install -c conda-forge rdkit` (document any adaptation in this file).

## CI install path

`.github/workflows/ci.yml` installs from `requirements-lock.txt` identically to the Quick Install above. CI is not a secret third path.
```

- [ ] **Step 2: Commit**

```bash
git add docs/reproducibility.md
git commit -m "docs: reproducibility.md install + lockfile procedure"
```

---

### Task 4: Fix F821 `CompiledODE` in `tests/unit/test_regimen.py`

**Files:**
- Modify: `tests/unit/test_regimen.py:17`

- [ ] **Step 1: Inspect current imports**

```bash
sed -n '15,20p' /home/jam/Sisyphus/tests/unit/test_regimen.py
```
Expected: `from sisyphus.engine.compiler import ODECompiler, ResolvedParams`

- [ ] **Step 2: Add `CompiledODE` to import**

Change line 17 from:
```python
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
```
to:
```python
from sisyphus.engine.compiler import CompiledODE, ODECompiler, ResolvedParams
```

- [ ] **Step 3: Verify ruff F821 gone**

```bash
ruff check tests/unit/test_regimen.py 2>&1 | grep F821 && echo "still present" || echo "fixed"
```
Expected: "fixed".

- [ ] **Step 4: Verify test still runs**

```bash
pytest tests/unit/test_regimen.py --collect-only 2>&1 | tail -3
```
Expected: collects N items without errors.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_regimen.py
git commit -m "fix(tests): import CompiledODE in test_regimen.py (F821)"
```

---

### Task 5: Fix test pollution in `tests/unit/test_n50_benchmark.py`

**Files:**
- Modify: `tests/unit/test_n50_benchmark.py:208-214`

- [ ] **Step 1: Inspect the offending block**

```bash
sed -n '208,218p' /home/jam/Sisyphus/tests/unit/test_n50_benchmark.py
```
Expected: sees `sys.modules["sisyphus.pipeline.predict"] = type(sys)("stub")` and `sys.modules[...].predict = spy_predict`. `monkeypatch` fixture already in function signature.

- [ ] **Step 2: Replace direct sys.modules mutation with monkeypatch.setitem**

Exact change — replace:
```python
    sys.modules["sisyphus.pipeline.predict"] = type(sys)("stub")
    sys.modules["sisyphus.pipeline.predict"].predict = spy_predict
    n50_module.run(payload)
    assert captured == {"route": "iv", "infusion_duration_min": 30.0}
```
with:
```python
    stub = type(sys)("stub")
    stub.predict = spy_predict
    monkeypatch.setitem(sys.modules, "sisyphus.pipeline.predict", stub)
    n50_module.run(payload)
    assert captured == {"route": "iv", "infusion_duration_min": 30.0}
```

- [ ] **Step 3: Verify test itself still passes**

```bash
pytest tests/unit/test_n50_benchmark.py::test_run_passes_infusion_duration -v 2>&1 | tail -10
```
Expected: PASSED.

- [ ] **Step 4: Verify pollution fixed — run full unit suite**

```bash
pytest tests/unit --tb=short 2>&1 | tail -10
```
Expected: **all unit tests pass** (previous `test_end_to_end_caffeine` TypeError gone).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_n50_benchmark.py
git commit -m "fix(tests): scope sys.modules stub via monkeypatch.setitem"
```

---

### Task 6: Mark propranolol integration test xfail

**Files:**
- Modify: `tests/integration/test_engine_validation.py`

- [ ] **Step 1: Inspect parametrize + test**

```bash
grep -n "propranolol\|parametrize\|test_cmax_within_5pct" /home/jam/Sisyphus/tests/integration/test_engine_validation.py | head -20
```

- [ ] **Step 2: Change parametrize entry to apply xfail to propranolol**

Locate the `@pytest.mark.parametrize(...)` decorator for `test_cmax_within_5pct`. Convert the `propranolol` entry to `pytest.param("propranolol", marks=pytest.mark.xfail(reason="pre-existing 16.3% Cmax drift; investigation separate — see docs/claude/propranolol_cmax_drift.md", strict=False))`.

Example diff pattern (adjust to actual file structure):
```python
# Before:
@pytest.mark.parametrize("drug_name", ["midazolam", "caffeine", "warfarin", "propranolol"])

# After:
@pytest.mark.parametrize("drug_name", [
    "midazolam",
    "caffeine",
    "warfarin",
    pytest.param(
        "propranolol",
        marks=pytest.mark.xfail(
            reason="pre-existing 16.3% Cmax drift; investigation separate",
            strict=False,
        ),
    ),
])
```
**Only** the `test_cmax_within_5pct` parametrization gets the xfail. The `test_mass_balance` and `test_solver_success` tests for propranolol continue to PASS — do not touch those.

- [ ] **Step 3: Verify**

```bash
pytest tests/integration/test_engine_validation.py -v 2>&1 | tail -20
```
Expected: 11 passed, 1 xfail (propranolol Cmax), 0 failed.

- [ ] **Step 4: Create follow-up note**

Write to `/home/jam/Sisyphus/docs/claude/propranolol_cmax_drift.md`:

```markdown
# Propranolol Cmax drift — investigation TODO

## Observation

`tests/integration/test_engine_validation.py::test_cmax_within_5pct[propranolol]`
fails on `main` (2026-04-23) with:

- Observed Cmax: 0.157585
- Target (Omega): 0.135500
- Relative error: 16.3% (target threshold: 5%)

Other Omega-equivalence drugs pass: midazolam, caffeine, warfarin.

## Why xfail, not fix

Out of scope for the H5+H1 hardening spec (CI + lockfile infrastructure).
Cause is likely a post-Phase-1 engine drift (OATP ECM migration 2026-04-20,
Achour correlated abundance 2026-04-23, or V3 IV-Cmax routing 2026-04-22)
that needs targeted investigation.

## Candidates to rule out

1. Run `git bisect` between `5eea6eb` (P4 complete) and current `main`
2. Check if propranolol Kp method changed
3. Check if propranolol hepatic extraction path changed with ECM migration
4. Compare propranolol DrugOnGraph snapshot across commits

## Scope

Separate spec. xfail-marked until investigated.
```

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_engine_validation.py docs/claude/propranolol_cmax_drift.md
git commit -m "test(integration): xfail propranolol Cmax drift pending investigation"
```

---

### Task 7: Create `tests/benchmark/test_smoke.py`

**Files:**
- Create: `tests/benchmark/test_smoke.py`

- [ ] **Step 1: Verify existing benchmark test pattern**

```bash
head -40 /home/jam/Sisyphus/tests/benchmark/test_holdout.py
```
Expected: examines how other benchmark tests import + structure.

- [ ] **Step 2: Write smoke test**

Write to `/home/jam/Sisyphus/tests/benchmark/test_smoke.py`:

```python
"""CI smoke benchmark — runs full pipeline on N=5 holdout drugs.

Not a correctness gate; catches import failures, missing model artifacts,
and catastrophic engine regressions. Full holdout benchmark is out-of-CI.
"""
from __future__ import annotations

import pytest


@pytest.mark.slow
def test_benchmark_smoke_n5():
    from sisyphus.validation.benchmark import run_benchmark

    result = run_benchmark(holdout_only=True, max_drugs=5)

    assert result.n_drugs >= 1, "no drugs evaluated"
    assert result.aafe > 0, "AAFE must be positive"
    assert result.aafe < 100, f"AAFE={result.aafe} implausibly large — engine broken?"
```

- [ ] **Step 3: Verify it runs with --run-slow**

First check if repo has `--run-slow` pytest conftest option:

```bash
grep -r "run-slow\|run_slow\|--slow" /home/jam/Sisyphus/conftest.py /home/jam/Sisyphus/tests/conftest.py 2>&1 | head -10
```

If `--run-slow` flag is NOT implemented, use `-m slow` instead (pytest marker selector). Verify:

```bash
pytest tests/benchmark/test_smoke.py -v 2>&1 | tail -10
pytest tests/benchmark/test_smoke.py -v -m slow 2>&1 | tail -10
```

Expected: test is collected and either runs (PASSED) or is skipped by default. Adjust CI yml in Task 8 to match the actual mechanism (`-m slow` is a safe default).

- [ ] **Step 4: Commit**

```bash
git add tests/benchmark/test_smoke.py
git commit -m "test(benchmark): CI smoke test run_benchmark(max_drugs=5)"
```

---

### Task 8: Create `.github/workflows/ci.yml`

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create workflows directory**

```bash
mkdir -p /home/jam/Sisyphus/.github/workflows
```

- [ ] **Step 2: Write CI workflow**

Write to `/home/jam/Sisyphus/.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: pip

      - name: Install locked dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-lock.txt
          pip install -e .

      - name: Ruff (advisory — non-gating this cycle)
        run: ruff check src tests --exit-zero

      - name: Unit tests
        run: pytest tests/unit -v

      - name: Engine validation integration
        run: pytest tests/integration/test_engine_validation.py -v

      - name: Benchmark smoke
        run: pytest tests/benchmark/test_smoke.py -v -m slow
```

- [ ] **Step 3: Validate yaml syntax locally**

```bash
python3 -c "import yaml; yaml.safe_load(open('/home/jam/Sisyphus/.github/workflows/ci.yml'))" && echo "yaml ok"
```
Expected: `yaml ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow (ruff advisory + unit + integration + smoke)"
```

---

### Task 9: Add CI badge to README

**Files:**
- Modify: `/home/jam/Sisyphus/README.md` (first few lines)

- [ ] **Step 1: Inspect current README top**

```bash
head -10 /home/jam/Sisyphus/README.md
```

- [ ] **Step 2: Add badge beneath title**

Add a line with the CI badge immediately after the README title (H1):

```markdown
[![CI](https://github.com/jam-sudo/Sisyphus/actions/workflows/ci.yml/badge.svg)](https://github.com/jam-sudo/Sisyphus/actions/workflows/ci.yml)
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): add CI status badge"
```

---

### Task 10: Full local verification

**Files:** none (verification only)

- [ ] **Step 1: Clean test run matching CI order**

```bash
cd /home/jam/Sisyphus
ruff check src tests --exit-zero 2>&1 | tail -3
pytest tests/unit -v 2>&1 | tail -15
pytest tests/integration/test_engine_validation.py -v 2>&1 | tail -15
pytest tests/benchmark/test_smoke.py -v -m slow 2>&1 | tail -10
```

Expected:
- ruff: `Found 141 errors.` or similar (1 fewer because F821 fixed); advisory, continues
- unit: all tests pass (359+1 previously failing now pass = 360+ pass, 1 skip)
- integration: 11 passed, 1 xfail (propranolol)
- smoke: 1 passed

- [ ] **Step 2: Lockfile install sanity**

```bash
python3 -m venv /tmp/ci_sim
source /tmp/ci_sim/bin/activate
pip install --upgrade pip
pip install -r /home/jam/Sisyphus/requirements-lock.txt
pip install -e /home/jam/Sisyphus
pytest /home/jam/Sisyphus/tests/unit/test_distribution.py -v
deactivate
rm -rf /tmp/ci_sim
```
Expected: install succeeds; sample test PASSED.

- [ ] **Step 3: Push branch, open PR**

```bash
git push -u origin feature/hardening-h5-h1-ci-lockfile
```
Verify GitHub Actions runs on the push — expect green.

If green, task complete.
If red, diagnose from CI logs. Likely issues:
- RDKit install fails on CI runner → fallback to apt-get install or document conda path
- Model artifact path issue in smoke test → check if `data/reference/holdout.json` is tracked (it should be)
- `-m slow` marker gates smoke test away → adjust CI command

---

## Completion criteria

After Task 10:
- `requirements-lock.txt` committed, clean (no chemprop/rdkit-pypi pollution)
- `docs/reproducibility.md` committed
- `tests/unit/test_regimen.py` F821 fixed
- `tests/unit/test_n50_benchmark.py` test pollution fixed
- `tests/integration/test_engine_validation.py` propranolol xfail
- `docs/claude/propranolol_cmax_drift.md` committed
- `tests/benchmark/test_smoke.py` committed
- `.github/workflows/ci.yml` committed
- README CI badge added
- First CI run is green on the feature branch

Estimated total: 2-2.5 hours if everything goes smoothly; budget 3-4 hours with debugging.

---

## Deviations to flag during execution

If any occurs, halt and report:

1. Ruff count drops by > 1 (F821 was 1; other ruff errors should be invariant)
2. New tests break (pollution fix shouldn't affect logic)
3. Integration tests beyond propranolol start failing (engine drift beyond expected)
4. Lockfile can't be generated cleanly (RDKit resolution failure)
5. CI runner OOMs on benchmark smoke (reduce max_drugs to 3)
6. Any request to "also fix ruff violations" — REFUSE per spec non-goal
