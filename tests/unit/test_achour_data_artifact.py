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
