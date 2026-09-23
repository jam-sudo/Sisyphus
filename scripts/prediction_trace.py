#!/usr/bin/env python3
"""Emit a stagewise, hashable prediction trace for numerical-drift diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from sisyphus.pipeline.predict import predict
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile


def _hash_array(values: tuple[float, ...]) -> str:
    arr = np.asarray(values, dtype="<f8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def trace(smiles: str, dose_mg: float, route: str) -> dict:
    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    result = predict(smiles, dose_mg, route, strict=True)
    cmax = result.cmax_prediction
    engine = result.engine_simulation
    if cmax is None:
        raise RuntimeError("CmaxPrediction missing")
    return {
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "resource_profile": result.resource_profile,
        },
        "input": {"smiles": profile.smiles, "dose_mg": dose_mg, "route": route},
        "descriptors": {
            "mw": profile.mw,
            "logp": profile.logp,
            "tpsa": profile.tpsa,
            "compound_type": profile.compound_type,
            "ad_flags": list(profile.ad_flags),
        },
        "adme": {
            "fup": adme.fup.mean,
            "clint": adme.clint.mean,
            "peff": adme.peff.mean,
            "solubility": adme.solubility.mean,
            "vdss": adme.vdss.mean,
            "rbp": adme.rbp.mean,
        },
        "cmax": {
            "final": cmax.cmax.mean,
            "tracks": dict(cmax.tracks),
            "weights": dict(cmax.weights),
            "interval_90": cmax.interval_90,
            "interval_source": cmax.interval_source,
        },
        "engine": None
        if engine is None
        else {
            "cmax": engine.endpoints.cmax.mean,
            "tmax": engine.endpoints.tmax.mean,
            "auc_0t": engine.endpoints.auc_0t.mean,
            "mass_balance_error": engine.mass_balance_error,
            "time_sha256": _hash_array(engine.time_h),
            "concentration_sha256": _hash_array(engine.concentration_mg_l),
            "n_time": len(engine.time_h),
        },
        "artifact_provenance": dict(result.artifact_provenance),
        "execution_status": result.execution_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--dose", type=float, required=True)
    parser.add_argument("--route", choices=["oral", "iv"], default="oral")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = trace(args.smiles, args.dose, args.route)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
