"""Data-artifact tests for Achour 2021 Table S7 extraction.

Source: Achour et al. 2021 Clin Pharmacol Ther 109:222-232, PMC7839483,
CC BY-NC 4.0. Extraction via scripts/extract_achour2021_abundance.py.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib

import numpy as np
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


def test_csv_donor_ids_are_unique() -> None:
    """Guard against a transcription mistake that duplicates one donor and
    drops another — row count alone wouldn't catch this."""
    rows = _load_csv()
    ids = [r["donor_id"] for r in rows]
    assert len(set(ids)) == 29, f"Duplicate donor ids: {ids}"


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
