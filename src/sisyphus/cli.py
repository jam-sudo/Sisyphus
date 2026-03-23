"""Sisyphus CLI entry point.

Usage::

    sisyphus predict --smiles "CC(=O)Oc1ccccc1C(=O)O" --dose 500
    sisyphus benchmark --holdout
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Sisyphus PBPK Prediction")
    subparsers = parser.add_subparsers(dest="command")

    # predict command
    pred_parser = subparsers.add_parser("predict", help="Predict PK for a SMILES")
    pred_parser.add_argument("--smiles", required=True, help="SMILES string")
    pred_parser.add_argument("--dose", type=float, required=True, help="Dose in mg")
    pred_parser.add_argument("--route", default="oral", choices=["oral", "iv"])
    pred_parser.add_argument("--verbose", "-v", action="store_true")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run holdout benchmark")
    bench_parser.add_argument("--holdout", action="store_true", help="Run on holdout set only")
    bench_parser.add_argument("--max-drugs", type=int, default=None, help="Limit number of drugs")
    bench_parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.command == "predict":
        _run_predict(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_predict(args: argparse.Namespace) -> None:
    """Run a single drug prediction and print results."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from sisyphus.pipeline.predict import predict

    result = predict(args.smiles, args.dose, args.route)

    print(f"Drug: {result.drug_name}")
    print(f"Method: {result.method}")
    print(f"Confidence: {result.confidence}")
    print(f"Cmax: {result.pk.cmax.mean:.4f} mg/L")
    if result.pk.tmax:
        print(f"Tmax: {result.pk.tmax.mean:.2f} h")
    if result.pk.auc_0t:
        print(f"AUC: {result.pk.auc_0t.mean:.4f} mg*h/L")
    if result.pk.t_half:
        print(f"t½: {result.pk.t_half.mean:.2f} h")
    if result.engine_pk:
        print(f"  Engine Cmax: {result.engine_pk.cmax.mean:.4f} mg/L")
    if result.ml_pk:
        print(f"  ML Cmax: {result.ml_pk.cmax.mean:.4f} mg/L")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


def _run_benchmark(args: argparse.Namespace) -> None:
    """Run holdout benchmark and print summary metrics."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from sisyphus.validation.benchmark import run_benchmark

    result = run_benchmark(holdout_only=args.holdout, max_drugs=args.max_drugs)

    print("\nBenchmark Results:")
    print(f"  Drugs evaluated: {result.n_drugs}")
    print(f"  AAFE: {result.aafe:.3f}")
    print(f"  %2-fold: {result.pct_2fold:.1f}%")
    if result.pi_coverage_90 is not None:
        print(f"  90% PI coverage: {result.pi_coverage_90:.1%}")


if __name__ == "__main__":
    main()
