#!/usr/bin/env python3
"""Predeclared source-consistency ablation of the fitted direct Cmax training set.

Maps every fitted row in ``data/training/cmax_omega_public_clean.csv`` back to its
arm group in the two pinned Omega source workbooks, records the row-level source
evidence for dose mass basis and formulation, then compares the existing
full-data XGBoost against a filtered-data XGBoost on identical scaffold folds and
identical scored rows. A size-matched random-subset control separates "source
consistency" from "less training data".

Features, hyperparameters and the target log10(Cmax/dose) are frozen: they are
read from the shipped model metadata and the shipped dataset. Nothing in
``models/`` or the fitted dataset is written.

Usage:
    .venv/bin/python scripts/cmax_source_consistency_ablation.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.retrain_cmax_public import (  # noqa: E402
    DATASET,
    META,
    SOURCE,
    SOURCE_SHA,
    SOURCE_WORKBOOKS,
)
from scripts.train_clint_expanded import scaffold_split_indices  # noqa: E402
from sisyphus.descriptors import compute_features  # noqa: E402

LEDGER = ROOT / "data/validation/cmax_source_consistency_ledger_2026-09-25.csv"
RESULTS = ROOT / "data/validation/cmax_source_consistency_ablation_2026-09-25.json"
DEFAULT_CACHE = Path.home() / ".cache/sisyphus/omega_mmpk"

# The model has no formulation input, so reported modified-release arms are
# excluded. Other labels are only "non-modified reported", not proven IR.
MODIFIED_RELEASE = {"tablet er", "capsule er", "er", "enteric-coated tablet", "ocas tablet"}
N_BOOTSTRAP = 2000
N_CONTROL_SEEDS = 10
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

log = logging.getLogger("cmax_ablation")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_xlsx(path: Path) -> list[dict[str, str | int | None]]:
    """Read sheet1 of an xlsx into dicts keyed by header text (stdlib only)."""
    with zipfile.ZipFile(path) as archive:
        shared = [
            "".join(node.text or "" for node in si.iter(f"{NS}t"))
            for si in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall(f"{NS}si")
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    def value(cell: ET.Element) -> str | None:
        node = cell.find(f"{NS}v")
        if node is None or node.text is None:
            return None
        return shared[int(node.text)] if cell.get("t") == "s" else node.text

    def column(cell: ET.Element) -> str:
        return "".join(char for char in (cell.get("r") or "") if char.isalpha())

    rows = sheet.findall(f".//{NS}row")
    header = {column(cell): value(cell) for cell in rows[0].findall(f"{NS}c")}
    table = []
    for row in rows[1:]:
        record: dict[str, str | int | None] = {
            header.get(column(cell)): value(cell) for cell in row.findall(f"{NS}c")
        }
        record["_row"] = int(row.get("r") or 0)
        table.append(record)
    return table


def load_workbooks(cache: Path) -> list[dict[str, str | int | None]]:
    """Fetch (if absent) and SHA-verify the pinned source workbooks."""
    cache.mkdir(parents=True, exist_ok=True)
    arms: list[dict[str, str | int | None]] = []
    for pin in SOURCE_WORKBOOKS:
        name = pin["url"].rsplit("/", 1)[-1]
        path = cache / name
        if not path.exists():
            raw = pin["url"].replace("github.com", "raw.githubusercontent.com")
            raw = raw.replace("/blob/", "/")
            log.info("downloading pinned workbook %s", name)
            with urllib.request.urlopen(raw, timeout=120) as response:  # noqa: S310
                path.write_bytes(response.read())
        if sha256(path) != pin["sha256"]:
            raise ValueError(f"{name} SHA256 does not match the pinned source workbook")
        for arm in read_xlsx(path):
            arm["_workbook"] = name
            arms.append(arm)
    return arms


def arm_groups(arms: list[dict]) -> dict[tuple[str, float], list[dict]]:
    """Group source arms that carry a Cmax by (analyte name, reported dose)."""
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for arm in arms:
        if not arm.get("name") or not arm.get("dose (mg)") or not arm.get("cmax (ng/ml)"):
            continue
        groups[(str(arm["name"]).strip().lower(), round(float(arm["dose (mg)"]), 6))].append(arm)
    return groups


def build_ledger(groups: dict[tuple[str, float], list[dict]]) -> list[dict]:
    """One ledger row per fitted row, carrying its verified source evidence."""
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("Omega Cmax source SHA256 changed")
    source = {row["name"]: row for row in csv.DictReader(SOURCE.open(newline=""))}
    fitted = list(csv.DictReader(DATASET.open(newline="")))
    expected = json.loads(META.read_text())["trained_on"]
    if sha256(DATASET) != expected["sha256"] or len(fitted) != expected["n_drugs_clean"]:
        raise ValueError("Fitted dataset does not match the shipped model metadata")

    ledger = []
    for row in fitted:
        aggregate = source[row["name"]]
        source_dose = float(aggregate["dose_mg"])
        group = groups.get((row["name"].strip().lower(), round(source_dose, 6)), [])
        # Same identity check the administered-analyte audit used: the arm count must
        # equal n_studies and their geometric mean must reproduce the fitted aggregate.
        cmax = [float(arm["cmax (ng/ml)"]) / 1000 for arm in group]
        if len(group) != int(aggregate["n_studies"]) or not cmax:
            raise ValueError(f"{row['name']}: source arm count does not match n_studies")
        if not np.isclose(np.exp(np.mean(np.log(cmax))), float(aggregate["cmax_mg_L"]), rtol=1e-6):
            raise ValueError(f"{row['name']}: source arm geometric mean differs from aggregate")

        salts = sorted({str(arm.get("salt") or "").strip() for arm in group} - {""})
        forms = sorted({str(arm.get("formulation") or "").strip().lower() for arm in group} - {""})
        # A fitted dose that differs from the source dose is an adjudicated
        # active-moiety conversion performed by the production recipe.
        converted = not np.isclose(float(row["dose_mg"]), source_dose, rtol=1e-9)
        if not salts:
            dose_basis = "no_salt_reported"
        elif converted:
            dose_basis = "salt_converted_to_parent"
        else:
            # The source has no dose-basis field: a reported salt leaves salt vs
            # parent-equivalent mass unverifiable at row level.
            dose_basis = "salt_reported_basis_unverified"

        if not all(str(arm.get("formulation") or "").strip() for arm in group):
            formulation_class = "unreported"
        elif set(forms) & MODIFIED_RELEASE:
            formulation_class = "modified_release"
        else:
            formulation_class = "reported_non_modified"

        reasons = []
        if dose_basis == "salt_reported_basis_unverified":
            reasons.append("dose_mass_basis_unverified")
        if formulation_class == "unreported":
            reasons.append("formulation_not_source_reported")
        if formulation_class == "modified_release":
            reasons.append("modified_release_formulation")

        food = sorted(
            {
                str(arm.get("comments")).strip()
                for arm in group
                if arm.get("comments")
                and any(k in str(arm["comments"]).lower()
                        for k in ("fasted", "fasting", "fed ", "food", "meal"))
            }
        )
        ledger.append(
            {
                "name": row["name"],
                "fitted_dose_mg": row["dose_mg"],
                "fitted_cmax_mg_L": row["cmax_mg_L"],
                "log_cmax_per_dose": row["log_cmax_per_dose"],
                "n_studies": aggregate["n_studies"],
                "workbook_rows": " ".join(
                    f"{arm['_workbook']}:{arm['_row']}" for arm in group
                ),
                "pmids": " ".join(
                    sorted({str(arm["pmid"]) for arm in group if arm.get("pmid")})
                ),
                "salt_forms": "|".join(salts),
                "dose_mass_basis": dose_basis,
                "formulations": "|".join(forms),
                "formulation_class": formulation_class,
                "food_context_comment": " || ".join(food),
                # Neither workbook has a matrix or analyte column; plasma parent is
                # a dataset-level curation property, not row-level evidence.
                "matrix_evidence": "not_recorded_in_source",
                "include": not reasons,
                "exclude_reasons": ";".join(reasons),
            }
        )
    return ledger


def fold_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    err = pred - obs
    return {
        "aafe": float(10 ** np.mean(np.abs(err))),
        "geometric_bias": float(10 ** np.mean(err)),
        "pct_within_2fold": float(100 * np.mean(np.abs(err) <= np.log10(2))),
        "pct_within_3fold": float(100 * np.mean(np.abs(err) <= np.log10(3))),
        "r2": float(1 - np.sum(err**2) / np.sum((obs - obs.mean()) ** 2)),
        "n": int(len(obs)),
    }


def out_of_fold(
    X: np.ndarray, y: np.ndarray, folds: list[list[int]], scored: np.ndarray,
    train_pool: np.ndarray, params: dict
) -> np.ndarray:
    """OOF predictions for `scored` rows, training only on `train_pool` minus the fold.

    Folds are whole Murcko scaffolds, so dropping a fold drops its scaffolds from
    training for every arm.
    """
    pred = np.full(len(y), np.nan)
    for fold in folds:
        test = np.intersect1d(np.asarray(fold), scored)
        if not len(test):
            continue
        train = np.setdiff1d(train_pool, np.asarray(fold))
        model = xgb.XGBRegressor(**params).fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ledger = build_ledger(arm_groups(load_workbooks(args.workbook_dir)))
    with LEDGER.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ledger)
    log.info("ledger: %s rows -> %s", len(ledger), LEDGER.relative_to(ROOT))

    fitted = list(csv.DictReader(DATASET.open(newline="")))
    params = json.loads(META.read_text())["hyperparameters"]
    X = np.asarray([compute_features(row["smiles"]) for row in fitted])
    y = np.asarray([float(row["log_cmax_per_dose"]) for row in fitted])
    folds = scaffold_split_indices([row["smiles"] for row in fitted])

    full_pool = np.arange(len(y))
    keep = np.asarray([i for i, entry in enumerate(ledger) if entry["include"]])
    log.info("metadata-filtered subset: %s of %s rows", len(keep), len(y))

    arms = {
        "full_data": out_of_fold(X, y, folds, keep, full_pool, params),
        "filtered_data": out_of_fold(X, y, folds, keep, keep, params),
    }
    control = []
    for seed in range(N_CONTROL_SEEDS):
        pool = np.sort(
            np.random.default_rng(seed).choice(full_pool, size=len(keep), replace=False)
        )
        control.append(out_of_fold(X, y, folds, keep, pool, params))

    results = {
        "scored_rows": len(keep),
        "arms": {name: fold_metrics(pred[keep], y[keep]) for name, pred in arms.items()},
        "random_subset_control": {
            "n_seeds": N_CONTROL_SEEDS,
            "size": int(len(keep)),
            "per_seed_aafe": [float(fold_metrics(p[keep], y[keep])["aafe"]) for p in control],
        },
    }
    aafe = results["random_subset_control"]["per_seed_aafe"]
    results["random_subset_control"].update(
        mean_aafe=float(np.mean(aafe)), min_aafe=float(np.min(aafe)), max_aafe=float(np.max(aafe))
    )

    # Paired bootstrap over scored rows: the arms share rows, so resample rows once.
    rng = np.random.default_rng(42)
    full_err = np.abs(arms["full_data"][keep] - y[keep])
    filt_err = np.abs(arms["filtered_data"][keep] - y[keep])
    ratios = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(keep), len(keep))
        ratios.append(10 ** np.mean(filt_err[idx]) / 10 ** np.mean(full_err[idx]))
    results["paired_aafe_ratio_filtered_over_full"] = {
        "point": float(10 ** np.mean(filt_err) / 10 ** np.mean(full_err)),
        "ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
        "n_bootstrap": N_BOOTSTRAP,
    }
    worst = np.argsort(-(filt_err - full_err))
    results["largest_per_drug_changes"] = [
        {
            "name": fitted[keep[i]]["name"],
            "observed_log_cmax_per_dose": float(y[keep[i]]),
            "full_error_log10": float(arms["full_data"][keep[i]] - y[keep[i]]),
            "filtered_error_log10": float(arms["filtered_data"][keep[i]] - y[keep[i]]),
        }
        for i in list(worst[:10]) + list(worst[-10:])
    ]
    results["per_drug_oof_log10_error"] = [
        {
            "name": fitted[i]["name"],
            "observed_log_cmax_per_dose": float(y[i]),
            "full_error_log10": float(arms["full_data"][i] - y[i]),
            "filtered_error_log10": float(arms["filtered_data"][i] - y[i]),
            "control_mean_error_log10": float(np.mean([p[i] for p in control]) - y[i]),
        }
        for i in keep
    ]
    results["provenance"] = {
        "dataset_sha256": sha256(DATASET),
        "source_sha256": SOURCE_SHA,
        "source_workbooks": SOURCE_WORKBOOKS,
        "ledger_sha256": sha256(LEDGER),
        "hyperparameters": params,
        "n_features": int(X.shape[1]),
    }
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")

    for name, metric in results["arms"].items():
        log.info(
            "%-14s AAFE=%.4f bias=%.4f %%2f=%.1f R2=%.3f N=%s",
            name, metric["aafe"], metric["geometric_bias"],
            metric["pct_within_2fold"], metric["r2"], metric["n"],
        )
    ctrl = results["random_subset_control"]
    log.info("random control  AAFE mean=%.4f range=[%.4f, %.4f] (%s seeds, size %s)",
             ctrl["mean_aafe"], ctrl["min_aafe"], ctrl["max_aafe"], ctrl["n_seeds"], ctrl["size"])
    ratio = results["paired_aafe_ratio_filtered_over_full"]
    log.info("paired AAFE ratio filtered/full = %.4f  95%% CI [%.4f, %.4f]",
             ratio["point"], *ratio["ci95"])
    log.info("results: %s", RESULTS.relative_to(ROOT))


if __name__ == "__main__":
    main()
