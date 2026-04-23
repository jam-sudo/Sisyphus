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
