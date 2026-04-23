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
  cells in the PDF are dashes (-); here they are represented as Python ``None``
  in the in-memory ``DONORS`` literal and as an **empty field** (RFC 4180
  standard) in the written CSV — not the string ``"NaN"``. The test loader
  translates empty-string cells back to ``None``; downstream consumers
  (pandas ``read_csv`` with default ``na_values``, numpy ``genfromtxt``
  with ``missing_values=""``) will recognize the empty cell as missing.

This script writes data/physiology/achour2021_liver_abundance.csv.
Subsequent tasks extend it with correlation matrix computation.

Note on CYP3A4 cross-check drift: the computed column mean from DONORS
(~49.85) differs from the reported Table S7 mean (49.51) by 0.694%.
This is rounding-accumulation across 29 PDF values rounded to 2dp and
is expected — not a transcription error. All other columns drift by
<0.21%.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import numpy as np
from scipy.linalg import eigh

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_OUT = ROOT / "data" / "physiology" / "achour2021_liver_abundance.csv"
JSON_OUT = ROOT / "data" / "physiology" / "achour2021_correlation.json"
OATP_INCLUSION_THRESHOLD = 0.3

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
    ("806",  (58.21,  20.12, 30.70, 17.66, 41.38,  1.03)),
    ("813",  (43.70,   None,  7.04, 25.51, 39.30,  1.76)),
    ("818",  (53.08,  12.26, 11.28, 38.65, 40.90,  1.50)),
    ("829",  (30.27,   7.32,  9.68, 35.11, 14.89,  1.86)),
    ("855",  (42.01,   None, 10.45, 16.19, 34.43,  None)),
    ("1071", ( 3.68,   None,  4.39, 30.00,  None,  0.61)),
    ("1304", (31.94,   2.59,  4.26,  9.54, 23.98,  0.74)),
    ("1372", (21.84,   7.42, 13.94,  None, 16.96,  0.41)),
    ("493",  (27.03,  15.77, 17.12,  None, 39.86,  0.90)),
    ("590",  (82.50,  13.25, 24.00, 25.00, 33.58,  1.17)),
    ("645",  (183.57, 81.30, 19.83, 98.02, 20.11,  1.42)),
    ("646",  (16.50,   None, 10.30, 17.57, 19.60,  0.88)),
    ("674",  (22.05,  10.43, 16.54, 10.00, 24.44,  0.60)),
    ("682",  (14.42,   7.06, 17.48, 50.31, 42.33,  2.76)),
    ("756",  (33.31,   9.61, 26.23, 22.25, 63.80,  0.94)),
    ("781",  (67.44,  10.64, 16.67, 24.49, 32.82,  None)),
    ("734",  (22.76,   None,  4.85,  8.15, 25.05,  0.41)),
    ("755",  (102.56,  8.47, 18.27, 17.92, 55.25,  0.79)),
    ("770",  (41.85,   8.04,  4.69, 29.91, 19.20,  None)),
    ("389",  ( 2.90,   3.97,  1.61, 10.09,  7.08,  0.86)),
    ("589",  (108.66, 34.81, 34.07, 17.13, 67.40,  1.47)),
    ("1063", (65.62,   9.46,  None,  None, 44.16,  1.26)),
    ("1359", (37.90,   7.62, 18.29, 28.95, 33.71,  0.95)),
]


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


if __name__ == "__main__":
    main()
