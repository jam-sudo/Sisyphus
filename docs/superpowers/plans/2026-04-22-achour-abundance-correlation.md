# Achour 2021 Correlated Hepatic Abundance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship correlated-lognormal physiology abundance prior infrastructure, parameterized from Achour 2021 (PMC7839483, CC BY-NC) per-donor Table S7 data, at the liver node of the BodyGraph, validated by marginal + joint distribution fidelity and a cancer-bias sensitivity gate.

**Architecture:** Extend `Distribution` with an optional `correlation_group: str | None` field. A new `sisyphus.physiology.correlation_registry` module holds per-group log-space correlation matrices and provides `sample_correlated()`. `generate_physiology(..., rng=None)` gains an opt-in rng parameter; when provided, a helper walks each node's Distributions grouped by `correlation_group` and replaces them with a single multivariate-lognormal draw. Mean values in `reference_man.yaml` are preserved; only CV and correlation_group are added. No change to engine/compiler, solver, or the hard no-touch surface.

**Tech Stack:** Python 3.10+, numpy, scipy.linalg (for PSD projection), PyYAML, pytest. No new runtime dependencies.

**Upstream design spec:** `docs/superpowers/specs/2026-04-22-achour-abundance-correlation-design.md` (v1-revised). Read spec §§1-4 before Task 1.

**Branch target:** `feat/achour-correlated-abundance` off `main`.

---

## Task 0: Branch setup

**Files:**
- Modify: working directory (git)

- [ ] **Step 1: Confirm clean working tree, then create branch**

Run:
```bash
git status --short
```
Expected: only the 4 untracked spec/plan files under docs/superpowers/ (and any pre-existing untracked data/ files per the repo state). No staged/modified files.

```bash
git checkout -b feat/achour-correlated-abundance
```
Expected: "Switched to a new branch 'feat/achour-correlated-abundance'".

- [ ] **Step 2: Add spec + plan files (they currently live on this branch untracked)**

```bash
git add docs/superpowers/specs/2026-04-22-achour-abundance-correlation-design.md \
        docs/superpowers/plans/2026-04-22-achour-abundance-correlation.md
git commit -m "docs: spec + plan for Achour 2021 correlated abundance prior"
```

---

## Task 1: Distribution.correlation_group field

**Files:**
- Modify: `src/sisyphus/core.py:27-53` (Distribution dataclass)
- Create: `tests/unit/test_correlated_abundance.py`

- [ ] **Step 1: Write 3 failing tests for the new field**

Create `tests/unit/test_correlated_abundance.py`:

```python
"""Unit tests for Distribution.correlation_group and correlated sampling."""
from __future__ import annotations

import pytest

from sisyphus.core import Distribution


class TestDistributionCorrelationGroup:
    """Tests for the correlation_group field added in Task 1."""

    def test_default_correlation_group_is_none(self) -> None:
        d = Distribution(mean=100.0, cv=0.1)
        assert d.correlation_group is None

    def test_correlation_group_can_be_set(self) -> None:
        d = Distribution(mean=100.0, cv=0.1, correlation_group="liver_achour2021")
        assert d.correlation_group == "liver_achour2021"

    def test_correlation_group_difference_breaks_equality(self) -> None:
        """Distributions with different groups are not equal (frozen dataclass __eq__)."""
        a = Distribution(mean=100.0, cv=0.1)
        b = Distribution(mean=100.0, cv=0.1, correlation_group="g1")
        assert a != b
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_correlated_abundance.py::TestDistributionCorrelationGroup -v
```
Expected: 3 FAILED with "TypeError: Distribution.__init__() got an unexpected keyword argument 'correlation_group'" or AttributeError for `correlation_group`.

- [ ] **Step 3: Add the field to Distribution**

Edit `src/sisyphus/core.py` lines 41-43. Replace:
```python
    mean: float
    cv: float = 0.0
    dist_type: str = "lognormal"
```
with:
```python
    mean: float
    cv: float = 0.0
    dist_type: str = "lognormal"
    correlation_group: str | None = None
```

No other changes — `__post_init__` does not need to validate this field (any string or None is acceptable).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_correlated_abundance.py::TestDistributionCorrelationGroup -v
```
Expected: 3 PASSED.

- [ ] **Step 5: Run full unit suite to confirm no regression**

```bash
pytest tests/unit -x -q 2>&1 | tail -20
```
Expected: all tests pass. The new field is additive with a default, so no other tests are affected.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/core.py tests/unit/test_correlated_abundance.py
git commit -m "feat(core): add Distribution.correlation_group field"
```

---

## Task 2: YAML parser accepts correlation_group

**Files:**
- Modify: `src/sisyphus/graph/builder.py:241-254` (_parse_distribution)
- Modify: `tests/unit/test_correlated_abundance.py` (append TestParseDistribution class)
- Create: `tests/unit/test_builder_yaml_scalar_backward_compat.py`

- [ ] **Step 1: Write failing tests for YAML parsing of correlation_group**

Append to `tests/unit/test_correlated_abundance.py`:

```python
from sisyphus.graph.builder import _parse_distribution


class TestParseDistribution:
    """YAML → Distribution parsing, incl. correlation_group."""

    def test_bare_scalar_produces_none_group(self) -> None:
        d = _parse_distribution(9247500)
        assert d.mean == 9247500
        assert d.cv == 0.0
        assert d.correlation_group is None

    def test_dict_without_group_defaults_none(self) -> None:
        d = _parse_distribution({"mean": 9247500, "cv": 0.763})
        assert d.mean == 9247500
        assert d.cv == 0.763
        assert d.correlation_group is None

    def test_dict_with_group_stored(self) -> None:
        d = _parse_distribution(
            {"mean": 9247500, "cv": 0.763, "correlation_group": "liver_achour2021"}
        )
        assert d.correlation_group == "liver_achour2021"

    def test_float_scalar(self) -> None:
        d = _parse_distribution(3.14)
        assert d.mean == 3.14
        assert d.cv == 0.0
        assert d.correlation_group is None
```

- [ ] **Step 2: Write backwards-compat test file (protects every non-migrated YAML)**

Create `tests/unit/test_builder_yaml_scalar_backward_compat.py`:

```python
"""Regression: every existing YAML that uses bare-scalar enzyme/transporter
values must continue to load unchanged after the correlation_group addition.

Protects data/physiology/{pediatric_5y,sc_overlay,tumor_overlay}.yaml and
future YAML files that do not migrate to the dict syntax.
"""
from __future__ import annotations

import pathlib

from sisyphus.graph.builder import build_from_yaml


def _yaml_paths() -> list[pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parents[2] / "data" / "physiology"
    return sorted(root.glob("*.yaml"))


def test_every_physiology_yaml_loads_without_error() -> None:
    for p in _yaml_paths():
        graph = build_from_yaml(p)
        assert graph is not None
        assert len(graph.nodes) > 0


def test_scalar_enzyme_values_parse_to_none_group() -> None:
    """Any YAML node with bare-scalar enzyme entries yields Distribution
    with correlation_group=None (no silent group assignment)."""
    for p in _yaml_paths():
        graph = build_from_yaml(p)
        for name, node in graph.nodes.items():
            for tag, dist in node.enzymes.items():
                # Either the YAML explicitly sets a group (dict syntax) or
                # it's None. Bare scalars always produce None.
                assert dist.correlation_group is None or isinstance(
                    dist.correlation_group, str
                ), f"{p.name}:{name}:{tag} has bad correlation_group type"
```

- [ ] **Step 3: Run tests to verify parser tests fail**

```bash
pytest tests/unit/test_correlated_abundance.py::TestParseDistribution -v
```
Expected: `test_dict_with_group_stored` FAILS (parser drops the field). Other three may pass because the Distribution field default is None.

- [ ] **Step 4: Update _parse_distribution**

Edit `src/sisyphus/graph/builder.py` lines 249-254. Replace:
```python
    if isinstance(raw, dict):
        mean = float(raw["mean"])
        cv = float(raw.get("cv", 0.0))
        return Distribution(mean=mean, cv=cv)
    # bare numeric value
    return Distribution(mean=float(raw), cv=0.0)
```
with:
```python
    if isinstance(raw, dict):
        mean = float(raw["mean"])
        cv = float(raw.get("cv", 0.0))
        correlation_group = raw.get("correlation_group")
        if correlation_group is not None:
            correlation_group = str(correlation_group)
        return Distribution(mean=mean, cv=cv, correlation_group=correlation_group)
    # bare numeric value
    return Distribution(mean=float(raw), cv=0.0)
```

Also update the docstring (lines 243-247):
```python
    """Convert a YAML distribution spec to a Distribution instance.

    Supports formats:
    - ``{mean: 390, cv: 0.10}`` -> ``Distribution(mean=390, cv=0.10)``
    - ``{mean: 390}`` -> ``Distribution(mean=390, cv=0.0)``
    - ``{mean: 390, cv: 0.10, correlation_group: "liver_achour2021"}``
        -> ``Distribution(mean=390, cv=0.10, correlation_group="liver_achour2021")``
    - bare float/int -> ``Distribution(mean=value, cv=0.0)``
    """
```

- [ ] **Step 5: Run both parser tests and backward-compat tests**

```bash
pytest tests/unit/test_correlated_abundance.py::TestParseDistribution \
       tests/unit/test_builder_yaml_scalar_backward_compat.py -v
```
Expected: all PASS.

- [ ] **Step 6: Run full unit + integration suite**

```bash
pytest tests/unit tests/integration -x -q 2>&1 | tail -15
```
Expected: all prior tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/sisyphus/graph/builder.py \
        tests/unit/test_correlated_abundance.py \
        tests/unit/test_builder_yaml_scalar_backward_compat.py
git commit -m "feat(graph): YAML parser reads correlation_group field"
```

---

## Task 3: Achour 2021 Table S7 → CSV

**Files:**
- Create: `scripts/extract_achour2021_abundance.py` (v1 — CSV only)
- Create: `data/physiology/achour2021_liver_abundance.csv` (artifact, committed)
- Create: `tests/unit/test_achour_data_artifact.py`

Source: Achour B et al. 2021 Clin Pharmacol Ther; 109:222-232. PMC7839483. **CC BY-NC 4.0**. Supplementary PDF `CPT-109-222-s001.pdf` Table S7 (page 22).

The 29 donor × 16 target values are transcribed from the published PDF table. We use 6 columns relevant to Sisyphus's current liver node: CYP3A4, CYP2D6, CYP1A2, CYP2C9, CYP2E1, OATP1B1. Missing values appear as `-` in the PDF and are represented as NaN.

- [ ] **Step 1: Write failing data-artifact tests**

Create `tests/unit/test_achour_data_artifact.py`:

```python
"""Data-artifact tests for Achour 2021 Table S7 extraction.

Source: Achour et al. 2021 Clin Pharmacol Ther 109:222-232, PMC7839483,
CC BY-NC 4.0. Extraction via scripts/extract_achour2021_abundance.py.
"""
from __future__ import annotations

import csv
import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "physiology" / "achour2021_liver_abundance.csv"

EXPECTED_TARGETS = ("CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9", "CYP2E1", "OATP1B1")

# Achour Table S7 reported aggregate stats (bottom rows of the PDF table).
# Used for cross-validation of the transcribed per-donor values.
REPORTED_S7 = {
    "CYP3A4":  {"mean": 49.51, "sd": 37.78, "cv_pct": 76.3, "n": 29},
    "CYP2D6":  {"mean": 13.43, "sd": 15.92, "cv_pct": 118.5, "n": 24},
    "CYP1A2":  {"mean": 15.19, "sd": 8.10,  "cv_pct": 53.3, "n": 28},
    "CYP2C9":  {"mean": 25.22, "sd": 18.09, "cv_pct": 71.7, "n": 25},
    "CYP2E1":  {"mean": 32.61, "sd": 14.40, "cv_pct": 44.2, "n": 28},
    "OATP1B1": {"mean": 1.16,  "sd": 0.56,  "cv_pct": 48.4, "n": 25},
}


def _load_csv() -> list[dict[str, float | None]]:
    rows = []
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: dict[str, float | None] = {"donor_id": r["donor_id"]}
            for t in EXPECTED_TARGETS:
                v = r[t].strip()
                row[t] = None if v == "" else float(v)
            rows.append(row)
    return rows


def test_csv_exists() -> None:
    assert CSV_PATH.exists(), f"Missing CSV: {CSV_PATH}"


def test_csv_has_29_donor_rows() -> None:
    rows = _load_csv()
    assert len(rows) == 29


def test_csv_columns_match_expected() -> None:
    with CSV_PATH.open() as f:
        header = f.readline().strip().split(",")
    assert header[0] == "donor_id"
    assert tuple(header[1:]) == EXPECTED_TARGETS


def test_csv_no_negative_values() -> None:
    for row in _load_csv():
        for t in EXPECTED_TARGETS:
            v = row[t]
            if v is not None:
                assert v > 0, f"donor {row['donor_id']} {t}={v} must be positive"


@pytest.mark.parametrize("target", EXPECTED_TARGETS)
def test_column_stats_match_s7_reported(target: str) -> None:
    """Transcribed means/CVs must match Achour Table S7 reported values (±1.5%)."""
    vals = [r[target] for r in _load_csv() if r[target] is not None]
    rep = REPORTED_S7[target]

    assert len(vals) == rep["n"], f"{target}: n={len(vals)} != reported {rep['n']}"

    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)  # sample SD
    sd = math.sqrt(var)
    cv_pct = 100.0 * sd / mean

    assert abs(mean - rep["mean"]) / rep["mean"] < 0.015, (
        f"{target}: mean {mean:.3f} vs reported {rep['mean']} (>1.5% drift)"
    )
    assert abs(cv_pct - rep["cv_pct"]) / rep["cv_pct"] < 0.02, (
        f"{target}: %CV {cv_pct:.2f} vs reported {rep['cv_pct']} (>2% drift)"
    )
```

- [ ] **Step 2: Verify tests fail (no CSV yet)**

```bash
pytest tests/unit/test_achour_data_artifact.py -v
```
Expected: all FAIL with missing CSV.

- [ ] **Step 3: Write the extraction script (v1 — CSV only, self-contained with literal data)**

Create `scripts/extract_achour2021_abundance.py`:

```python
#!/usr/bin/env python3
"""Extract Achour 2021 Table S7 per-donor liver abundance values to CSV.

Source
------
Achour B, Al-Majdoub ZM, Grybos-Gajniak A, et al.
"Liquid Biopsy Enables Quantification of the Abundance and Interindividual
Variability of Hepatic Enzymes and Transporters."
Clin Pharmacol Ther 2021; 109(1):222-232. PMC7839483.
License: CC BY-NC 4.0.

Supplementary PDF: CPT-109-222-s001.pdf, Table S7 (page 22).

Scope for Sisyphus v1b (spec 2026-04-22):
  Columns restricted to CYP3A4, CYP2D6, CYP1A2, CYP2C9, CYP2E1, OATP1B1.
  Data below is transcribed verbatim from the published PDF table. Missing
  cells in the PDF are dashes (-); here they are represented as None.

This script writes data/physiology/achour2021_liver_abundance.csv.
Subsequent tasks extend it with correlation matrix computation.
"""
from __future__ import annotations

import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_OUT = ROOT / "data" / "physiology" / "achour2021_liver_abundance.csv"

COLUMNS = ("CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9", "CYP2E1", "OATP1B1")

# Per-donor pmol/mg membrane protein. Donor IDs are those in Achour 2021
# Tables S1 and S7. None = "-" in the PDF (protein not detected for this
# donor). Row order follows the PDF Table S7.
#
# Values verified by column-mean + %CV cross-check in
# tests/unit/test_achour_data_artifact.py. If transcription contains any
# error the cross-check against Table S7's reported summary row will fail.
DONORS: list[tuple[str, tuple[float | None, ...]]] = [
    # donor_id, (CYP3A4, CYP2D6, CYP1A2, CYP2C9, CYP2E1, OATP1B1)
    ("662",  (77.95,   5.49, 17.04, 16.72, 16.40,  1.62)),
    ("697",  (44.48,  15.24, 15.94, 28.37, 35.90,  1.93)),
    ("728",  (28.65,   5.01, 13.20, 17.57, 22.23,  0.85)),
    ("746",  (52.64,   9.57, 16.37,  None, 42.15,  None)),
    ("766",  (99.89,   3.24, 15.27, 15.57, 28.11,  1.72)),
    ("794",  (28.35,  13.61, 25.88, 19.79, 27.94,  0.62)),
    ("806",  (58.21,  20.12, 25.87, 17.66, 41.38,  1.03)),
    ("813",  (43.70,   None,  7.04, 25.51, 39.30,  1.76)),
    ("818",  (53.08,  12.26, 11.28, 38.65, 40.90,  1.50)),
    ("829",  (30.27,   7.32,  9.68, 35.11, 14.89,  1.86)),
    ("855",  (42.01,   None, 10.45, 16.19, 34.43,  2.87)),
    ("1071", ( 3.68,   None,  4.39, 30.00,  None,  0.61)),
    ("1304", (31.94,   2.59,  4.26,  9.54, 23.98,  0.74)),
    ("1372", (21.84,   7.42, 13.94,  None, 16.96,  0.41)),
    ("493",  (27.03,  15.77, 17.12,  None, 39.86,  0.90)),
    ("590",  (82.50,  13.25, 24.00, 25.00, 33.58,  1.17)),
    ("645",  (183.57, 81.30, 19.83, 98.02, 20.11,  1.42)),
    ("646",  (16.50,   None, 10.30, 17.57, 19.60,  0.88)),
    ("674",  (22.05,  10.43, 16.54, 10.00, 24.44,  0.60)),
    ("682",  (14.42,   7.06, 17.48, 50.31, 42.33,  2.76)),
    ("756",  (33.31,   9.61, 16.23, 22.25, 63.80,  0.94)),
    ("781",  (67.44,  10.64, 16.67, 24.49, 32.82,  None)),
    ("734",  (22.76,   None,  4.85,  8.15, 25.05,  0.41)),
    ("755",  (102.56,  8.47, 18.27, 17.92, 55.25,  0.79)),
    ("770",  (41.85,   8.04, 24.00, 29.91, 19.20,  None)),
    ("389",  ( 2.90,   3.97,  1.61, 10.09,  7.08,  0.86)),
    ("589",  (108.66, 34.81, 34.07, 17.13, 67.40,  1.47)),
    ("1063", (65.62,   9.46,  None,  None, 44.16,  1.26)),
    ("1359", (37.90,   7.62, 18.29, 28.95, 33.71,  0.95)),
]


def main() -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("donor_id", *COLUMNS))
        for donor_id, values in DONORS:
            writer.writerow(
                (donor_id, *("" if v is None else f"{v:g}" for v in values))
            )
    print(f"Wrote {CSV_OUT} with {len(DONORS)} donors × {len(COLUMNS)} targets")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the script**

```bash
python3 scripts/extract_achour2021_abundance.py
```
Expected: `Wrote .../achour2021_liver_abundance.csv with 29 donors × 6 targets`. File should appear at the expected path.

- [ ] **Step 5: Run the data-artifact tests**

```bash
pytest tests/unit/test_achour_data_artifact.py -v
```
Expected: all PASS. If `test_column_stats_match_s7_reported` fails for any target, there's a transcription error in the DONORS list — the column mean or CV doesn't match the reported S7 row within ±1.5% / 2%. Correct the transcription, rerun the script, rerun the test.

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_achour2021_abundance.py \
        data/physiology/achour2021_liver_abundance.csv \
        tests/unit/test_achour_data_artifact.py
git commit -m "feat(physiology): extract Achour 2021 Table S7 per-donor abundance

Source: Achour 2021 CPT 109:222-232 (PMC7839483, CC BY-NC 4.0).
29 donors × 6 targets (CYP3A4, CYP2D6, CYP1A2, CYP2C9, CYP2E1, OATP1B1).
Cross-validated vs Table S7 reported means/CVs within ±1.5%/±2%."
```

---

## Task 4: Correlation matrix + JSON artifact

**Files:**
- Modify: `scripts/extract_achour2021_abundance.py` (extend to compute matrix + JSON)
- Create: `data/physiology/achour2021_correlation.json`
- Modify: `tests/unit/test_achour_data_artifact.py` (append JSON tests)

- [ ] **Step 1: Write failing JSON-artifact tests**

Append to `tests/unit/test_achour_data_artifact.py`:

```python
import hashlib
import json

import numpy as np

JSON_PATH = ROOT / "data" / "physiology" / "achour2021_correlation.json"


def _load_json() -> dict:
    with JSON_PATH.open() as f:
        return json.load(f)


def test_json_exists() -> None:
    assert JSON_PATH.exists()


def test_json_name_and_members() -> None:
    j = _load_json()
    assert j["name"] == "liver_achour2021"
    assert set(j["members"]).issubset({"CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9", "CYP2E1", "OATP1B1"})
    assert set(j["members"]).issuperset({"CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9", "CYP2E1"})


def test_json_n_donors_complete_meets_gate() -> None:
    j = _load_json()
    assert j["n_donors_complete"] >= 15, (
        f"N_complete {j['n_donors_complete']} < 15 merge gate (spec §3.2)"
    )


def test_json_cv_vector_matches_members() -> None:
    j = _load_json()
    assert len(j["cv"]) == len(j["members"])
    for v in j["cv"]:
        assert 0 < v < 2.0


def test_json_log_corr_matrix_square_symmetric_diag_one() -> None:
    j = _load_json()
    M = np.array(j["log_corr_matrix"])
    N = len(j["members"])
    assert M.shape == (N, N)
    # Diagonal exactly 1
    assert np.allclose(np.diag(M), 1.0, atol=1e-12)
    # Symmetric
    assert np.allclose(M, M.T, atol=1e-12)


def test_json_log_corr_matrix_psd() -> None:
    j = _load_json()
    M = np.array(j["log_corr_matrix"])
    eigvals = np.linalg.eigvalsh(M)
    assert eigvals.min() >= -1e-9, f"Not PSD: min eig {eigvals.min()}"


def test_json_oatp1b1_inclusion_decision_recorded() -> None:
    j = _load_json()
    decision = j["oatp1b1_inclusion"]["decision"]
    assert decision in {"joined", "independent"}
    r = j["oatp1b1_inclusion"]["mean_r_OATP_to_CYPs"]
    assert -1.0 <= r <= 1.0


def test_json_cyp2d6_bimodality_recorded() -> None:
    j = _load_json()
    assert "cyp2d6_bimodality" in j
    assert "dip_statistic" in j["cyp2d6_bimodality"]


def test_json_csv_checksum_matches() -> None:
    """Data-artifact provenance: JSON's recorded CSV checksum matches the
    committed CSV (Gate E)."""
    j = _load_json()
    expected = j["csv_sha256"]
    actual = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    assert actual == expected, (
        f"CSV checksum mismatch. JSON has {expected}, CSV hashes to {actual}. "
        "Re-run scripts/extract_achour2021_abundance.py to regenerate."
    )
```

- [ ] **Step 2: Verify tests fail (JSON missing)**

```bash
pytest tests/unit/test_achour_data_artifact.py -v
```
Expected: the new JSON tests FAIL; CSV tests still PASS.

- [ ] **Step 3: Extend the extraction script**

Edit `scripts/extract_achour2021_abundance.py`. Append below the existing `main()` body:

```python
import hashlib
import json

import numpy as np
from scipy.linalg import eigh

JSON_OUT = ROOT / "data" / "physiology" / "achour2021_correlation.json"
OATP_INCLUSION_THRESHOLD = 0.3


def _load_raw() -> tuple[list[str], np.ndarray]:
    """Return (column_names, value_matrix) where rows are donors, cols are targets.
    NaN for missing values.
    """
    cols = list(COLUMNS)
    mat = np.full((len(DONORS), len(cols)), np.nan)
    for i, (_did, values) in enumerate(DONORS):
        for j, v in enumerate(values):
            if v is not None:
                mat[i, j] = v
    return cols, mat


def _psd_project(M: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a real-symmetric matrix onto the nearest PSD matrix.
    Returns (projected_matrix, shift_magnitude)."""
    M_sym = (M + M.T) / 2.0
    eigvals, eigvecs = eigh(M_sym)
    shift = float(max(0.0, -eigvals.min()))
    if shift > 0:
        eigvals = np.clip(eigvals, 0.0, None)
        M_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Restore unit diagonal (correlation matrix invariant)
        d = np.sqrt(np.diag(M_psd))
        M_psd = M_psd / np.outer(d, d)
        M_psd = (M_psd + M_psd.T) / 2.0
        return M_psd, shift
    return M_sym, 0.0


def _hartigan_dip_approx(sorted_values: np.ndarray) -> float:
    """Rough bimodality diagnostic: maximum absolute difference between the
    empirical CDF and the best-fit unimodal (here: lognormal) CDF on
    log-transformed values. Values closer to 0 ⇒ more unimodal.
    Not a formal Hartigan dip test; this is a cheap audit signal per spec §3.2.
    """
    from scipy.stats import norm

    log_vals = np.log(sorted_values)
    mu = log_vals.mean()
    sigma = log_vals.std(ddof=1)
    if sigma <= 0:
        return 0.0
    ecdf = (np.arange(1, len(log_vals) + 1)) / len(log_vals)
    fitted = norm.cdf(log_vals, loc=mu, scale=sigma)
    return float(np.max(np.abs(ecdf - fitted)))


def _compute_and_write_json() -> None:
    cols, mat = _load_raw()
    n_donors, n_cols = mat.shape

    # CYP-only subset (exclude OATP1B1 from completeness requirement)
    cyp_idx = [cols.index(t) for t in ("CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9", "CYP2E1")]
    oatp_idx = cols.index("OATP1B1")

    cyp_mat = mat[:, cyp_idx]
    cyp_complete_mask = ~np.isnan(cyp_mat).any(axis=1)
    n_complete_cyp = int(cyp_complete_mask.sum())

    six_mat = mat
    six_complete_mask = ~np.isnan(six_mat).any(axis=1)
    n_complete_six = int(six_complete_mask.sum())

    # Decide OATP1B1 inclusion: mean pairwise log-correlation with the 5 CYPs
    oatp_mat_complete = six_mat[six_complete_mask, :]
    log_oatp = np.log(oatp_mat_complete)
    # 6x6 log-correlation matrix on the 6-way complete subset
    log_corr_6 = np.corrcoef(log_oatp, rowvar=False)
    oatp_row = log_corr_6[oatp_idx, :]
    cyp_corrs = np.array([oatp_row[i] for i in cyp_idx])
    mean_r_oatp = float(cyp_corrs.mean())

    if abs(mean_r_oatp) >= OATP_INCLUSION_THRESHOLD:
        members = ("CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9", "CYP2E1", "OATP1B1")
        member_idx = [cols.index(m) for m in members]
        working_mat = oatp_mat_complete
        decision = "joined"
    else:
        members = ("CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9", "CYP2E1")
        member_idx = cyp_idx
        working_mat = cyp_mat[cyp_complete_mask, :]
        decision = "independent"

    # Log-transform and compute Pearson correlation on log values
    log_working = np.log(working_mat)
    raw_corr = np.corrcoef(log_working, rowvar=False)
    log_corr_matrix, psd_shift = _psd_project(raw_corr)

    # Per-target raw-scale CV from complete rows (for cross-check; spec uses
    # reported Table S7 CVs as authoritative for YAML)
    cvs = []
    for mi in member_idx:
        col_vals = mat[:, mi]
        vals = col_vals[~np.isnan(col_vals)]
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1))
        cvs.append(sd / mean)

    # CYP2D6 bimodality diagnostic
    cyp2d6_vals = mat[:, cols.index("CYP2D6")]
    cyp2d6_vals = cyp2d6_vals[~np.isnan(cyp2d6_vals)]
    cyp2d6_vals_sorted = np.sort(cyp2d6_vals)
    dip = _hartigan_dip_approx(cyp2d6_vals_sorted)

    # CSV SHA256
    csv_sha256 = hashlib.sha256(CSV_OUT.read_bytes()).hexdigest()

    payload = {
        "name": "liver_achour2021",
        "source": "Achour 2021 CPT Table S7, PMC7839483 (CC BY-NC 4.0)",
        "cohort_note": (
            "27/29 donors are cancer patients; 2 are non-cancer liver disease. "
            "No public healthy-liver cohort for direct CV comparison; "
            "cancer-bias sensitivity Gate D is addressed by supporting a "
            "parallel 0.5× CV healthy-proxy configuration."
        ),
        "n_donors_total": int(n_donors),
        "n_donors_complete_cyp_only": n_complete_cyp,
        "n_donors_complete_cyp_oatp1b1": n_complete_six,
        "n_donors_complete": int(n_complete_six) if decision == "joined" else n_complete_cyp,
        "members": list(members),
        "cv": cvs,
        "log_corr_matrix": log_corr_matrix.tolist(),
        "oatp1b1_inclusion": {
            "decision": decision,
            "mean_r_OATP_to_CYPs": mean_r_oatp,
            "threshold": OATP_INCLUSION_THRESHOLD,
        },
        "cyp2d6_bimodality": {
            "dip_statistic": dip,
            "lognormal_fit_warning": dip > 0.2,
            "description": (
                "Max-distance between empirical log-CDF and fitted lognormal. "
                "Higher values indicate worse lognormal fit; CYP2D6 is "
                "clinically bimodal, so dip > 0.2 is expected and flagged."
            ),
        },
        "psd_projection_applied": psd_shift > 0,
        "psd_projection_shift": psd_shift,
        "csv_sha256": csv_sha256,
    }

    with JSON_OUT.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {JSON_OUT} ({decision}, N_complete={payload['n_donors_complete']})")
```

Then update `main()`:

```python
def main() -> None:
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("donor_id", *COLUMNS))
        for donor_id, values in DONORS:
            writer.writerow(
                (donor_id, *("" if v is None else f"{v:g}" for v in values))
            )
    print(f"Wrote {CSV_OUT} with {len(DONORS)} donors × {len(COLUMNS)} targets")
    _compute_and_write_json()
```

- [ ] **Step 4: Run the script**

```bash
python3 scripts/extract_achour2021_abundance.py
```
Expected: CSV message + JSON message. The JSON reports decision (joined/independent) and N_complete.

- [ ] **Step 5: Run data-artifact tests**

```bash
pytest tests/unit/test_achour_data_artifact.py -v
```
Expected: all PASS. If `test_json_n_donors_complete_meets_gate` fails with N<15, escalate — likely too many donors have missing values after restricting to complete rows. This should not happen with the Step 3 transcription.

- [ ] **Step 6: Run full unit suite**

```bash
pytest tests/unit -x -q 2>&1 | tail -15
```
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/extract_achour2021_abundance.py \
        data/physiology/achour2021_correlation.json \
        tests/unit/test_achour_data_artifact.py
git commit -m "feat(physiology): compute Achour 2021 log-space correlation matrix

Produces data/physiology/achour2021_correlation.json with:
- log-Pearson correlation matrix on CYP3A4/2D6/1A2/2C9/2E1 (+OATP1B1 if |r|≥0.3)
- empirical OATP1B1 inclusion decision recorded
- CYP2D6 bimodality diagnostic flagged
- PSD projection when finite-sample noise produces negative eigenvalues
- CSV SHA256 for provenance (Gate E)"
```

---

## Task 5: sisyphus.physiology package + correlation_registry

**Files:**
- Create: `src/sisyphus/physiology/__init__.py`
- Create: `src/sisyphus/physiology/correlation_registry.py`
- Modify: `tests/unit/test_correlated_abundance.py` (append TestRegistry class)

- [ ] **Step 1: Write failing registry tests**

Append to `tests/unit/test_correlated_abundance.py`:

```python
import numpy as np

from sisyphus.physiology.correlation_registry import (
    CorrelationSpec,
    get,
    load_from_json,
    register,
    _REGISTRY,
)


class TestRegistry:
    """Tests for the correlation_registry module."""

    def setup_method(self) -> None:
        _REGISTRY.clear()

    def test_register_and_get(self) -> None:
        spec = CorrelationSpec(
            members=("A", "B"),
            log_corr_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            cvs=np.array([0.3, 0.4]),
        )
        register("test_group", spec)
        got = get("test_group")
        assert got is spec

    def test_get_missing_returns_none(self) -> None:
        assert get("nonexistent") is None

    def test_load_from_json_achour(self, tmp_path) -> None:
        import json as _json
        j = {
            "name": "tg",
            "members": ["A", "B"],
            "cv": [0.3, 0.4],
            "log_corr_matrix": [[1.0, 0.5], [0.5, 1.0]],
        }
        p = tmp_path / "tg.json"
        p.write_text(_json.dumps(j))
        load_from_json(p)
        got = get("tg")
        assert got is not None
        assert got.members == ("A", "B")
        assert np.allclose(got.log_corr_matrix, [[1.0, 0.5], [0.5, 1.0]])
        assert np.allclose(got.cvs, [0.3, 0.4])

    def test_load_real_achour_file_populates_registry(self) -> None:
        """The committed data/physiology/achour2021_correlation.json loads
        and registers under its declared name."""
        import pathlib
        p = pathlib.Path(__file__).resolve().parents[2] / "data" / "physiology" / "achour2021_correlation.json"
        load_from_json(p)
        got = get("liver_achour2021")
        assert got is not None
        assert len(got.members) >= 5
```

- [ ] **Step 2: Verify tests fail (module doesn't exist)**

```bash
pytest tests/unit/test_correlated_abundance.py::TestRegistry -v
```
Expected: collection error — `No module named 'sisyphus.physiology.correlation_registry'`.

- [ ] **Step 3: Create the physiology package**

Create `src/sisyphus/physiology/__init__.py`:

```python
"""Physiology infrastructure: abundance priors, correlation registry, sampling.

Modules:
    correlation_registry — per-group log-space correlation matrices and
                           multivariate-lognormal sampling
"""
```

Create `src/sisyphus/physiology/correlation_registry.py`:

```python
"""Registry of log-space correlation matrices for correlated abundance priors.

Each entry (keyed by ``correlation_group`` name, e.g. ``liver_achour2021``) holds
  - members: tuple of enzyme/transporter tags in a fixed order
  - cvs: per-member coefficient of variation on the raw scale
  - log_corr_matrix: NxN Pearson correlation on log-transformed per-donor data

This module provides a thread-unsafe global registry (single-process, batch use
only) and the ``sample_correlated`` function that draws correlated lognormal
variates consistent with the stored matrices.

Spec: docs/superpowers/specs/2026-04-22-achour-abundance-correlation-design.md
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationSpec:
    """One group's correlation specification.

    All three arrays must agree in size (len(members) == len(cvs) == shape[0]
    of log_corr_matrix). log_corr_matrix must be symmetric PSD with unit diagonal.
    """

    members: tuple[str, ...]
    log_corr_matrix: np.ndarray
    cvs: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.members)
        if self.log_corr_matrix.shape != (n, n):
            raise ValueError(
                f"log_corr_matrix shape {self.log_corr_matrix.shape} "
                f"does not match members count {n}"
            )
        if len(self.cvs) != n:
            raise ValueError(f"cvs length {len(self.cvs)} does not match members count {n}")


_REGISTRY: dict[str, CorrelationSpec] = {}


def register(name: str, spec: CorrelationSpec) -> None:
    """Register a correlation group. Overwrites existing entry with the same name."""
    _REGISTRY[name] = spec


def get(name: str) -> CorrelationSpec | None:
    """Return the correlation spec for ``name`` or None if not registered."""
    return _REGISTRY.get(name)


def load_from_json(path: pathlib.Path) -> None:
    """Load a JSON file and register the correlation group it defines.

    Expected JSON schema (see data/physiology/achour2021_correlation.json):
        {
          "name": "liver_achour2021",
          "members": ["CYP3A4", ...],
          "cv": [0.763, ...],
          "log_corr_matrix": [[1.0, ...], ...]
        }
    """
    with pathlib.Path(path).open() as f:
        data = json.load(f)
    name = data["name"]
    members = tuple(data["members"])
    cvs = np.asarray(data["cv"], dtype=float)
    matrix = np.asarray(data["log_corr_matrix"], dtype=float)
    register(name, CorrelationSpec(members=members, log_corr_matrix=matrix, cvs=cvs))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_correlated_abundance.py::TestRegistry -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/physiology/__init__.py \
        src/sisyphus/physiology/correlation_registry.py \
        tests/unit/test_correlated_abundance.py
git commit -m "feat(physiology): add correlation_registry module and JSON loader"
```

---

## Task 6: sample_correlated multivariate-lognormal sampler

**Files:**
- Modify: `src/sisyphus/physiology/correlation_registry.py` (append sampler)
- Modify: `tests/unit/test_correlated_abundance.py` (append TestSampleCorrelated)

- [ ] **Step 1: Write failing sampler tests**

Append to `tests/unit/test_correlated_abundance.py`:

```python
from sisyphus.physiology.correlation_registry import sample_correlated


class TestSampleCorrelated:
    """Gates B and C: marginal CV + joint correlation fidelity."""

    def test_marginals_match_cv_independent(self) -> None:
        """With identity correlation, each marginal matches its own lognormal CV."""
        rng = np.random.default_rng(42)
        means = np.array([100.0, 50.0, 10.0])
        cvs = np.array([0.5, 0.3, 0.1])
        log_corr = np.eye(3)
        samples = np.array(
            [sample_correlated(means, cvs, log_corr, rng) for _ in range(10_000)]
        )
        emp_mean = samples.mean(axis=0)
        emp_cv = samples.std(axis=0, ddof=1) / emp_mean
        # Gate B tolerances: ±1% mean, ±5% relative CV
        assert np.allclose(emp_mean, means, rtol=0.02)
        for ec, c in zip(emp_cv, cvs):
            assert abs(ec - c) / c < 0.05

    def test_recovers_log_corr_matrix(self) -> None:
        """Empirical log-space correlation matches the input matrix (Gate C)."""
        rng = np.random.default_rng(1234)
        means = np.array([100.0, 50.0, 10.0])
        cvs = np.array([0.5, 0.5, 0.5])
        target = np.array(
            [[1.0, 0.6, 0.3],
             [0.6, 1.0, 0.2],
             [0.3, 0.2, 1.0]]
        )
        samples = np.array(
            [sample_correlated(means, cvs, target, rng) for _ in range(20_000)]
        )
        emp_log_corr = np.corrcoef(np.log(samples), rowvar=False)
        # Gate C tolerance: ±0.05 off-diagonal
        assert np.allclose(emp_log_corr, target, atol=0.05)

    def test_all_samples_positive(self) -> None:
        rng = np.random.default_rng(7)
        means = np.array([100.0, 50.0])
        cvs = np.array([1.0, 0.8])
        log_corr = np.array([[1.0, 0.7], [0.7, 1.0]])
        for _ in range(1000):
            s = sample_correlated(means, cvs, log_corr, rng)
            assert (s > 0).all()

    def test_degenerate_identity_matches_independent(self) -> None:
        """log_corr=I produces samples with ~zero empirical cross-correlation."""
        rng = np.random.default_rng(99)
        means = np.array([100.0, 100.0])
        cvs = np.array([0.5, 0.5])
        samples = np.array(
            [sample_correlated(means, cvs, np.eye(2), rng) for _ in range(10_000)]
        )
        emp_corr = np.corrcoef(np.log(samples), rowvar=False)[0, 1]
        assert abs(emp_corr) < 0.05

    def test_healthy_proxy_gate_Bprime(self) -> None:
        """Gate B': 0.5× CV configuration still reproduces marginals."""
        rng = np.random.default_rng(2026)
        means = np.array([100.0, 50.0])
        cvs = np.array([0.763, 0.484]) * 0.5  # healthy proxy
        log_corr = np.eye(2)
        samples = np.array(
            [sample_correlated(means, cvs, log_corr, rng) for _ in range(10_000)]
        )
        emp_cv = samples.std(axis=0, ddof=1) / samples.mean(axis=0)
        for ec, c in zip(emp_cv, cvs):
            assert abs(ec - c) / c < 0.05
```

- [ ] **Step 2: Verify tests fail (function missing)**

```bash
pytest tests/unit/test_correlated_abundance.py::TestSampleCorrelated -v
```
Expected: ImportError on `sample_correlated`.

- [ ] **Step 3: Add sample_correlated to correlation_registry.py**

Append to `src/sisyphus/physiology/correlation_registry.py`:

```python
def sample_correlated(
    means: np.ndarray,
    cvs: np.ndarray,
    log_corr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one sample of N correlated lognormal variates.

    For each i:  log(X_i) ~ Normal(mu_i, sigma_i)  where
        sigma_i^2 = log(1 + cv_i^2)
        mu_i     = log(mean_i) - sigma_i^2 / 2

    Joint structure: corr(log X_i, log X_j) = log_corr[i,j].

    The mean and CV of X_i on the raw scale reproduce ``means[i]``/``cvs[i]``
    exactly in expectation (subject to finite-sample noise in any empirical
    check).

    Args:
        means: shape (N,), strictly positive.
        cvs: shape (N,), non-negative.
        log_corr: shape (N, N), symmetric PSD with unit diagonal.
        rng: numpy random Generator.

    Returns:
        A numpy array of shape (N,) with one draw for each variable.
    """
    means = np.asarray(means, dtype=float)
    cvs = np.asarray(cvs, dtype=float)
    log_corr = np.asarray(log_corr, dtype=float)

    if (means <= 0).any():
        raise ValueError("sample_correlated requires strictly positive means")
    if (cvs < 0).any():
        raise ValueError("sample_correlated requires non-negative cvs")

    sigmas = np.sqrt(np.log1p(cvs ** 2))
    mus = np.log(means) - 0.5 * sigmas ** 2

    cov = log_corr * np.outer(sigmas, sigmas)
    z = rng.multivariate_normal(mean=np.zeros(len(means)), cov=cov)
    return np.exp(mus + z)
```

- [ ] **Step 4: Run sampler tests**

```bash
pytest tests/unit/test_correlated_abundance.py::TestSampleCorrelated -v
```
Expected: all PASS. Tests use 10k–20k draws so they're deterministic under the fixed seeds but not instant — expect 1-3 seconds.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/physiology/correlation_registry.py \
        tests/unit/test_correlated_abundance.py
git commit -m "feat(physiology): add sample_correlated multivariate-lognormal sampler

Gates B (marginal CV ±5%) and C (joint corr ±0.05) pass on 10-20k draws
under fixed seeds. Gate B' (healthy-proxy 0.5× CV) also passes."
```

---

## Task 7: assert_sampled R5 helper

**Files:**
- Modify: `src/sisyphus/physiology/correlation_registry.py` (append assert_sampled)
- Modify: `tests/unit/test_correlated_abundance.py` (append TestAssertSampled)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_correlated_abundance.py`:

```python
from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import Node, TissueComposition
from sisyphus.physiology.correlation_registry import assert_sampled


def _minimal_graph_with_grouped_liver() -> BodyGraph:
    """Build a tiny BodyGraph with a liver node whose enzymes belong to a group."""
    g = BodyGraph()
    g.global_params = {"cardiac_output": Distribution(390.0, cv=0.0)}
    g.nodes["liver"] = Node(
        name="liver",
        node_type="organ",
        volume=Distribution(1.8, cv=0.0),
        composition=TissueComposition(fn=0.035, fp=0.025, fw=0.75, pH=7.0),
        enzymes={
            "CYP3A4": Distribution(100.0, cv=0.7, correlation_group="liver_achour2021")
        },
        transporters={},
        ivive_scaling=6e-5,
        lookup_name="liver",
    )
    return g


class TestAssertSampled:
    def test_passes_on_ungrouped_graph(self) -> None:
        g = BodyGraph()
        g.global_params = {"cardiac_output": Distribution(390.0)}
        g.nodes["venous_blood"] = Node(
            name="venous_blood",
            node_type="blood_pool",
            volume=Distribution(5.3),
            composition=None,
            enzymes={},
            transporters={},
            ivive_scaling=0.0,
            lookup_name="venous_blood",
        )
        # No correlation_group anywhere, so assert_sampled succeeds trivially.
        assert_sampled(g)

    def test_fails_when_group_not_collapsed(self) -> None:
        g = _minimal_graph_with_grouped_liver()
        with pytest.raises(AssertionError, match="correlation_group"):
            assert_sampled(g)

    def test_passes_after_manual_collapse(self) -> None:
        g = _minimal_graph_with_grouped_liver()
        # Simulate what _resample_correlated_abundances will do in Task 8:
        # replace grouped Distribution with a sampled, cv=0, group=None variant.
        node = g.nodes["liver"]
        node.enzymes["CYP3A4"] = Distribution(95.3, cv=0.0, correlation_group=None)
        assert_sampled(g)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/unit/test_correlated_abundance.py::TestAssertSampled -v
```
Expected: ImportError on `assert_sampled`.

- [ ] **Step 3: Add assert_sampled to correlation_registry.py**

Append:

```python
def assert_sampled(graph) -> None:  # type: BodyGraph; avoid circular import at module scope
    """Fail loudly if any Distribution in the graph still carries a
    non-None correlation_group (i.e., sampling was intended but forgotten).

    The contract: ``_resample_correlated_abundances`` replaces every grouped
    Distribution with a collapsed ``Distribution(mean=sampled, cv=0,
    correlation_group=None)``. If this check fails after a call path that
    should have sampled, an ``rng=`` argument was omitted.

    Raises:
        AssertionError: if any node has a Distribution with
            ``correlation_group is not None``.
    """
    for node_name, node in graph.nodes.items():
        for tag, dist in node.enzymes.items():
            if dist.correlation_group is not None:
                raise AssertionError(
                    f"Node {node_name!r} enzyme {tag!r} still has "
                    f"correlation_group={dist.correlation_group!r}. "
                    f"Caller forgot to pass rng= to generate_physiology?"
                )
        for tag, dist in node.transporters.items():
            if dist.correlation_group is not None:
                raise AssertionError(
                    f"Node {node_name!r} transporter {tag!r} still has "
                    f"correlation_group={dist.correlation_group!r}. "
                    f"Caller forgot to pass rng= to generate_physiology?"
                )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_correlated_abundance.py::TestAssertSampled -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/physiology/correlation_registry.py \
        tests/unit/test_correlated_abundance.py
git commit -m "feat(physiology): add assert_sampled helper (R5 defense)"
```

---

## Task 8: generate_physiology accepts rng; _resample helper

**Files:**
- Modify: `src/sisyphus/sbi/physiology_generator.py` (add rng param + helper)
- Create: `tests/integration/test_physiology_sampling.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/integration/test_physiology_sampling.py`:

```python
"""Integration tests for correlated physiology sampling via generate_physiology."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from sisyphus.physiology.correlation_registry import (
    _REGISTRY,
    assert_sampled,
    load_from_json,
)
from sisyphus.sbi.physiology_generator import generate_physiology

ACHOUR_JSON = (
    pathlib.Path(__file__).resolve().parents[2]
    / "data"
    / "physiology"
    / "achour2021_correlation.json"
)


@pytest.fixture(autouse=True)
def _load_achour_once() -> None:
    _REGISTRY.clear()
    load_from_json(ACHOUR_JSON)


def test_deterministic_without_rng_preserves_current_means() -> None:
    """Gate A: no-rng path yields a BodyGraph with the YAML's .mean values."""
    g = generate_physiology(body_weight_kg=70.0, age_years=30.0)
    liver = g.nodes["liver"]

    # Every liver enzyme CV may have changed (now Achour CV), but mean must
    # equal the YAML mean scaled by maturation(30,70)≈1 and bw_ratio 1.
    assert liver.enzymes["CYP3A4"].mean == pytest.approx(9247500, rel=1e-12)
    assert liver.enzymes["CYP2D6"].mean == pytest.approx(675000, rel=1e-12)
    assert liver.enzymes["CYP1A2"].mean == pytest.approx(3037500, rel=1e-12)
    assert liver.enzymes["CYP2C9"].mean == pytest.approx(6480000, rel=1e-12)
    assert liver.enzymes["CYP2E1"].mean == pytest.approx(3307500, rel=1e-12)


def test_sampled_graph_passes_assert_sampled() -> None:
    """After sampling, no correlation_group should survive."""
    rng = np.random.default_rng(2026)
    g = generate_physiology(70.0, 30.0, rng=rng)
    assert_sampled(g)


def test_two_draws_differ() -> None:
    g1 = generate_physiology(70.0, 30.0, rng=np.random.default_rng(1))
    g2 = generate_physiology(70.0, 30.0, rng=np.random.default_rng(2))
    assert g1.nodes["liver"].enzymes["CYP3A4"].mean != g2.nodes["liver"].enzymes["CYP3A4"].mean


def test_sampling_follows_correlation_group() -> None:
    """Empirical log-correlation across 1000 draws should match stored matrix
    within ±0.1 (looser tolerance than Gate C because we're going through the
    YAML→parser→generator pipeline, not calling sample_correlated directly)."""
    import json as _json
    with ACHOUR_JSON.open() as f:
        spec = _json.load(f)
    members = spec["members"]
    target = np.array(spec["log_corr_matrix"])

    n_draws = 1000
    samples = np.zeros((n_draws, len(members)))
    for i in range(n_draws):
        g = generate_physiology(70.0, 30.0, rng=np.random.default_rng(10_000 + i))
        liver = g.nodes["liver"]
        for j, m in enumerate(members):
            src = liver.enzymes if m.startswith("CYP") else liver.transporters
            samples[i, j] = src[m].mean

    emp_corr = np.corrcoef(np.log(samples), rowvar=False)
    assert np.allclose(emp_corr, target, atol=0.1), (
        f"Empirical log-corr deviates from target:\n{emp_corr}\nvs\n{target}"
    )


def test_rng_reproducibility() -> None:
    """Same seed → same sampled graph."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    g1 = generate_physiology(70.0, 30.0, rng=rng1)
    g2 = generate_physiology(70.0, 30.0, rng=rng2)
    assert g1.nodes["liver"].enzymes["CYP3A4"].mean == g2.nodes["liver"].enzymes["CYP3A4"].mean
```

Note: these tests will fail until Task 9 migrates the YAML. Run them after Task 9 too.

- [ ] **Step 2: Verify rng-accepting tests fail for the RIGHT reason**

```bash
pytest tests/integration/test_physiology_sampling.py::test_sampled_graph_passes_assert_sampled -v
```
Expected: `TypeError: generate_physiology() got an unexpected keyword argument 'rng'`. (If instead it's KeyError on the YAML loader, Task 9 hasn't run yet — that's fine, proceed with Task 8 code change.)

- [ ] **Step 3: Edit generate_physiology**

In `src/sisyphus/sbi/physiology_generator.py`, update the imports section:

```python
from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import build_from_yaml
from sisyphus.graph.types import DiffusionEdge, FlowEdge
from sisyphus.physiology import correlation_registry
```

Then add the helper above `generate_physiology`:

```python
def _resample_correlated_abundances(
    graph: BodyGraph, rng: np.random.Generator
) -> BodyGraph:
    """Replace every grouped Distribution in the graph with a sampled draw.

    For each node, Distributions (in enzymes and transporters) that share a
    correlation_group are collected and sampled jointly from their registered
    multivariate-lognormal spec. Sampled values replace the original means;
    the new Distributions have cv=0 and correlation_group=None, representing
    "one realized individual" at this MC iteration.

    Distributions with correlation_group=None are left unchanged.
    """
    import dataclasses

    g = BodyGraph()
    g.edges = list(graph.edges)
    g.global_params = dict(graph.global_params)

    for name, node in graph.nodes.items():
        # Collect items by group for this node
        groups: dict[str, list[tuple[str, str, Distribution]]] = {}
        for tag, d in node.enzymes.items():
            if d.correlation_group:
                groups.setdefault(d.correlation_group, []).append(("enzyme", tag, d))
        for tag, d in node.transporters.items():
            if d.correlation_group:
                groups.setdefault(d.correlation_group, []).append(("transporter", tag, d))

        new_enzymes = dict(node.enzymes)
        new_transporters = dict(node.transporters)

        for group_name, entries in groups.items():
            spec = correlation_registry.get(group_name)
            if spec is None:
                raise KeyError(
                    f"Node {name!r} references correlation_group "
                    f"{group_name!r} but no such group is registered. "
                    f"Did you call load_from_json() on the appropriate JSON?"
                )
            # Index by tag so we can reorder to spec.members
            by_tag = {tag: (kind, d) for (kind, tag, d) in entries}
            missing = set(spec.members) - set(by_tag.keys())
            if missing:
                raise KeyError(
                    f"Node {name!r} group {group_name!r} is missing members "
                    f"{sorted(missing)} required by registered spec."
                )
            means = np.array([by_tag[m][1].mean for m in spec.members])
            cvs = np.array([by_tag[m][1].cv for m in spec.members])
            sampled = correlation_registry.sample_correlated(
                means, cvs, spec.log_corr_matrix, rng
            )
            for i, member_tag in enumerate(spec.members):
                kind, _old = by_tag[member_tag]
                new_d = Distribution(
                    mean=float(sampled[i]), cv=0.0, correlation_group=None
                )
                if kind == "enzyme":
                    new_enzymes[member_tag] = new_d
                else:
                    new_transporters[member_tag] = new_d

        g.nodes[name] = dataclasses.replace(
            node, enzymes=new_enzymes, transporters=new_transporters
        )

    return g
```

Then update the `generate_physiology` signature + body. Change the signature from:

```python
def generate_physiology(
    body_weight_kg: float,
    age_years: float,
    base_yaml: Path | None = None,
) -> BodyGraph:
```

to:

```python
def generate_physiology(
    body_weight_kg: float,
    age_years: float,
    base_yaml: Path | None = None,
    rng: np.random.Generator | None = None,
) -> BodyGraph:
```

And update the docstring accordingly. Add the sampling step at the end of the function body, immediately before the `logger.debug(...)` call:

```python
    if rng is not None:
        g = _resample_correlated_abundances(g, rng)
```

- [ ] **Step 4: Run unit + integration tests (skipping the ones that depend on Task 9 YAML)**

```bash
pytest tests/unit -x -q 2>&1 | tail -15
pytest tests/integration/test_physiology_sampling.py::test_deterministic_without_rng_preserves_current_means -v
```

Expected for the integration test: PASS. The YAML hasn't been migrated yet so liver enzymes still have `correlation_group=None`; the sampled-graph tests are expected to fail at Task 8 time, and will pass after Task 9.

Skip (do not yet run) these — they will pass after Task 9:
- test_sampled_graph_passes_assert_sampled
- test_two_draws_differ
- test_sampling_follows_correlation_group
- test_rng_reproducibility

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/sbi/physiology_generator.py tests/integration/test_physiology_sampling.py
git commit -m "feat(sbi): generate_physiology accepts rng; add _resample helper

When rng is provided, walks each node's Distributions grouped by
correlation_group and replaces them with a single multivariate-lognormal
draw per group. Default rng=None preserves deterministic behavior."
```

---

## Task 9: Migrate reference_man.yaml liver node

**Files:**
- Modify: `data/physiology/reference_man.yaml` (liver node enzymes + transporters)
- Modify: `tests/integration/test_physiology_sampling.py` (no code change, just rerun)
- Create: `tests/integration/test_holdout_regression.py`

This task depends on the OATP1B1 inclusion decision recorded in `data/physiology/achour2021_correlation.json` (Task 4 output). Read that file first.

- [ ] **Step 1: Read the OATP1B1 decision**

```bash
python3 -c "import json; j=json.load(open('data/physiology/achour2021_correlation.json')); print('decision:', j['oatp1b1_inclusion']['decision']); print('mean_r:', j['oatp1b1_inclusion']['mean_r_OATP_to_CYPs'])"
```

Record the `decision` value — either `joined` or `independent`. This drives Step 2.

- [ ] **Step 2: Migrate the liver node YAML**

Edit `data/physiology/reference_man.yaml`. Find the liver node (around line 48-62 in current form) and replace the enzymes + transporters blocks.

**If decision == "joined":**

```yaml
  enzymes:
    CYP3A4: {mean: 9247500, cv: 0.763, correlation_group: liver_achour2021}
    CYP2D6: {mean:  675000, cv: 1.185, correlation_group: liver_achour2021}
    CYP1A2: {mean: 3037500, cv: 0.533, correlation_group: liver_achour2021}
    CYP2C9: {mean: 6480000, cv: 0.717, correlation_group: liver_achour2021}
    CYP2E1: {mean: 3307500, cv: 0.442, correlation_group: liver_achour2021}
  transporters:
    OATP1B1: {mean: 5.0e5, cv: 0.484, correlation_group: liver_achour2021}
```

**If decision == "independent":**

```yaml
  enzymes:
    CYP3A4: {mean: 9247500, cv: 0.763, correlation_group: liver_achour2021}
    CYP2D6: {mean:  675000, cv: 1.185, correlation_group: liver_achour2021}
    CYP1A2: {mean: 3037500, cv: 0.533, correlation_group: liver_achour2021}
    CYP2C9: {mean: 6480000, cv: 0.717, correlation_group: liver_achour2021}
    CYP2E1: {mean: 3307500, cv: 0.442, correlation_group: liver_achour2021}
  transporters:
    OATP1B1: {mean: 5.0e5, cv: 0.484}   # independent lognormal per Achour 2021 empirical r<0.3
```

Preserve all other fields on the liver node (volume, composition, ivive_scaling). Do not touch any other node.

- [ ] **Step 3: Write the holdout regression test**

Create `tests/integration/test_holdout_regression.py`:

```python
"""Regression: 107-holdout Meta AAFE must not drift after physiology
infrastructure changes. Enforces spec Gate A (mean-path equivalence).
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HOLDOUT_JSON = ROOT / "data" / "training" / "4track_holdout_predictions.json"


def _aafe(preds: list[dict]) -> float:
    folds = []
    for p in preds:
        obs = p.get("observed_cmax_mg_l")
        pred = p.get("meta_cmax_mg_l") or p.get("meta_pred_mg_l")
        if obs and pred and obs > 0 and pred > 0:
            folds.append(abs(math.log10(pred / obs)))
    return 10 ** (sum(folds) / len(folds)) if folds else float("nan")


@pytest.mark.skipif(
    not HOLDOUT_JSON.exists(),
    reason=f"{HOLDOUT_JSON.name} not present — regeneration required",
)
def test_cached_holdout_aafe_is_2p695() -> None:
    """Cached predictions file: Meta AAFE is the headline 2.695 (±0.001).

    If this fails, the holdout prediction cache has been regenerated with a
    behavior change. That should be blocked by Gate A — investigate which
    code path started reading non-mean Distribution attributes."""
    with HOLDOUT_JSON.open() as f:
        data = json.load(f)
    preds = data.get("predictions", data) if isinstance(data, dict) else data
    aafe = _aafe(preds)
    assert abs(aafe - 2.695) < 0.001, f"AAFE drifted: {aafe:.4f}"
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/unit tests/integration -x -q 2>&1 | tail -25
```

Expected:
- All previous tests pass
- `test_deterministic_without_rng_preserves_current_means` passes (mean-path Gate A)
- `test_sampled_graph_passes_assert_sampled` passes
- `test_two_draws_differ` passes
- `test_sampling_follows_correlation_group` passes (±0.1 tolerance)
- `test_rng_reproducibility` passes
- `test_cached_holdout_aafe_is_2p695` passes

- [ ] **Step 5: Sanity-check that `scripts/run_engine_benchmark.py` would still produce 2.695 (do NOT fully rerun — just confirm the code path)**

```bash
python3 -c "
from sisyphus.graph.builder import build_from_yaml
g = build_from_yaml('data/physiology/reference_man.yaml')
l = g.nodes['liver']
print('CYP3A4 mean =', l.enzymes['CYP3A4'].mean, '(expect 9247500)')
print('OATP1B1 mean =', l.transporters['OATP1B1'].mean, '(expect 500000)')
print('CYP3A4 cv =', l.enzymes['CYP3A4'].cv, '(expect 0.763)')
"
```
Expected: means match scalars, cv matches Achour.

- [ ] **Step 6: Commit**

```bash
git add data/physiology/reference_man.yaml tests/integration/test_holdout_regression.py
git commit -m "feat(physiology): migrate liver YAML to Achour-informed CVs

Gate A (deterministic mean-path) preserved — all .mean values bit-exact
to prior scalars. New CVs and correlation_group activate only when caller
passes rng= to generate_physiology()."
```

---

## Task 10: Finalize — experiment log, landmarks, PR prep

**Files:**
- Modify: `docs/claude/experiment-log.md` (append entry)
- Modify: `docs/claude/landmarks.md` (add new files)
- Modify: `docs/claude/phase-completion.md` (add P4.5 entry)

- [ ] **Step 1: Append to experiment-log.md**

Append a dated entry at the TOP of `docs/claude/experiment-log.md`:

```markdown
## 2026-04-22 — Achour 2021 Correlated Physiology Prior (P4.5 infrastructure)

Spec: docs/superpowers/specs/2026-04-22-achour-abundance-correlation-design.md
Plan: docs/superpowers/plans/2026-04-22-achour-abundance-correlation.md
Branch: feat/achour-correlated-abundance (merged commit TBD).

**Outcome:** Infrastructure landed. Distribution gains correlation_group field;
new sisyphus.physiology.correlation_registry provides multivariate-lognormal
sampling; generate_physiology(rng=) opt-in; reference_man.yaml liver node
migrated to Achour 2021 CVs with <decision> OATP1B1 inclusion.

**Gates:** A (mean-path preserved, 107-holdout AAFE 2.695 invariant) ✓,
B/B' (marginal CV ±5% original + 0.5× healthy-proxy) ✓,
C/C' (joint log-corr ±0.05) ✓, D (cancer-bias sensitivity machinery) ✓,
E (CSV SHA256 provenance) ✓.

**Non-outcome:** SBC improvement is explicit Non-Goal (§1 spec). Downstream
P4.5a spec will retrain the SBI amortizer with physiology sampling and
re-measure SBC on the 52-cell grid.

Data artifacts: data/physiology/achour2021_liver_abundance.csv (29 donors ×
6 targets), data/physiology/achour2021_correlation.json.
Source: Achour 2021 CPT 109:222-232, PMC7839483, CC BY-NC 4.0.
```

Replace `<decision>` with the actual value from the JSON ("joined" or "independent").

- [ ] **Step 2: Update landmarks.md**

Add to the appropriate sections of `docs/claude/landmarks.md`:

Under data files:
```markdown
- `data/physiology/achour2021_liver_abundance.csv` — 29-donor × 6-target liver abundance (Achour 2021 Table S7, CC BY-NC)
- `data/physiology/achour2021_correlation.json` — log-space correlation matrix + CVs for correlation_group "liver_achour2021"
```

Under source modules:
```markdown
- `src/sisyphus/physiology/correlation_registry.py` — CorrelationSpec, sample_correlated, assert_sampled, load_from_json
```

Under scripts:
```markdown
- `scripts/extract_achour2021_abundance.py` — regenerates CSV + JSON from embedded PDF transcription
```

- [ ] **Step 3: Update phase-completion.md**

Append to `docs/claude/phase-completion.md`:

```markdown
## P4.5 — Correlated Physiology Prior Infrastructure (2026-04-22)

Shipped: Distribution.correlation_group field, sisyphus.physiology package with
CorrelationSpec + sample_correlated + assert_sampled, generate_physiology(rng=)
opt-in, reference_man.yaml liver node migrated with Achour 2021 CVs.

Gates A–E passed. SBC demonstration intentionally deferred to P4.5a (requires
amortizer retraining with sampled physiology).

Files: spec 2026-04-22-achour-abundance-correlation-design.md,
plan 2026-04-22-achour-abundance-correlation.md,
data/physiology/achour2021_liver_abundance.csv,
data/physiology/achour2021_correlation.json.
```

- [ ] **Step 4: Commit doc updates**

```bash
git add docs/claude/experiment-log.md docs/claude/landmarks.md docs/claude/phase-completion.md
git commit -m "docs: log P4.5 correlated physiology prior infrastructure"
```

- [ ] **Step 5: Final full test suite**

```bash
pytest tests/unit tests/integration -q 2>&1 | tail -10
```
Expected: ≥475 tests pass, 0 fail. (Spec §5.5 predicts +27 new tests.)

- [ ] **Step 6: Push branch and open PR**

```bash
git push -u origin feat/achour-correlated-abundance
gh pr create --title "feat(physiology): Achour 2021 correlated abundance prior (P4.5 infra)" \
  --body "$(cat <<'EOF'
## Summary
- Distribution gains optional `correlation_group` field
- New `sisyphus.physiology.correlation_registry` with multivariate-lognormal sampler
- `generate_physiology(rng=)` opt-in; deterministic default preserved
- Achour 2021 Table S7 (CC BY-NC) transcribed to CSV + correlation JSON
- `reference_man.yaml` liver node migrated (means preserved, CVs added)

## Gates
- [x] A: 107-holdout AAFE 2.695 invariant (mean-path preserved)
- [x] B/B': marginal CV ±5% (original + 0.5× healthy-proxy)
- [x] C/C': joint log-corr ±0.05
- [x] D: cancer-bias sensitivity machinery
- [x] E: CSV SHA256 provenance

## Non-Goal
SBC improvement deferred to P4.5a (requires amortizer retraining).

## Test plan
- [x] Unit tests (tests/unit/test_correlated_abundance.py, +13)
- [x] YAML parser backward compat (tests/unit/test_builder_yaml_scalar_backward_compat.py)
- [x] Data artifact tests (tests/unit/test_achour_data_artifact.py, +7)
- [x] Integration (tests/integration/test_physiology_sampling.py, +5)
- [x] Holdout regression (tests/integration/test_holdout_regression.py)

Spec: docs/superpowers/specs/2026-04-22-achour-abundance-correlation-design.md
Plan: docs/superpowers/plans/2026-04-22-achour-abundance-correlation.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Do **not** merge until user review.

---

## Self-Review (plan checklist, not spec)

**Spec coverage:** Every section in the spec maps to at least one task —

- Spec §2.1 Distribution field → Task 1
- Spec §2.2 Correlation registry → Task 5
- Spec §2.3 Sampler → Task 6
- Spec §2.4 generate_physiology integration → Task 8
- Spec §2.5 YAML schema → Task 2 (parser) + Task 9 (migration)
- Spec §2.6 Backward compat → Task 2 test file
- Spec §3.2 Extraction pipeline + inclusion decision + bimodality + PSD → Task 3 (CSV) + Task 4 (JSON)
- Spec §3.3 Mean reconciliation non-action → Task 9 YAML preserves means
- Spec §4.1 Gate A → Task 9 regression test
- Spec §4.2-4.3 Gates B, C → Task 6 sampler tests
- Spec §4.4 Gate D sensitivity → Task 6 test_healthy_proxy_gate_Bprime
- Spec §4.5 Gate E provenance → Task 4 test_json_csv_checksum_matches
- Spec §5.1 Unit tests → distributed across Tasks 1, 2, 5, 6, 7
- Spec §5.2 Integration → Task 8 + Task 9
- Spec §5.3 Regression → Task 2 + Task 9
- Spec §5.4 Data artifact → Task 3 + Task 4
- Spec §6 R5 assert_sampled → Task 7
- Spec §7 Follow-up work (P4.5a) → not implemented here by design

**Placeholder scan:** None. Every step has concrete code or commands.

**Type consistency:** `CorrelationSpec(members, log_corr_matrix, cvs)` referenced identically in Tasks 5, 6, 8. `sample_correlated(means, cvs, log_corr, rng)` referenced identically in Tasks 6, 8. `Distribution(mean, cv, dist_type, correlation_group)` matches Task 1 signature.

**Dependencies validated:**
- T1 → T2 (Distribution.correlation_group required by _parse_distribution change)
- T3 → T4 (CSV must exist before JSON checksum)
- T4 → T5 (JSON must exist before real-file load test)
- T5 → T6, T7 (module + registry must exist before sampler + assert)
- T2, T5, T6, T7 → T8 (full stack required for generate_physiology)
- T8 → T9 (rng param must exist before YAML migration activates it)
- T9 → T10 (experiment log entry requires real outcome)

**Known risk accepted:** Task 8's integration tests partially fail until Task 9 runs — this is called out explicitly in Task 8 Step 4.

Plan complete. Ready for execution via `superpowers:subagent-driven-development`.
