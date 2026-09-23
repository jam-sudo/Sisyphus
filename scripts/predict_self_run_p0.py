#!/usr/bin/env python3
"""Seal P0 predictions from the label-free manifest before opening outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

MANIFEST_SHA = "e304a1e57275576f3de65b515a9392ab74194529fc68f0924574d1298a44ab3c"
MODEL_SHA = "618106b53b0c9ce3c5b8a8fe62c5adf8f02b2308"


def run(manifest_path: Path, output: Path) -> dict:
    if os.environ.get("SISYPHUS_PROFILE") != "public":
        raise ValueError("P0 requires SISYPHUS_PROFILE=public")
    container_digest = os.environ.get("SISYPHUS_CONTAINER_DIGEST")
    if not container_digest or not container_digest.startswith("sha256:"):
        raise ValueError("P0 requires a recorded container digest")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("P0 manifest differs from the committed candidate list")
    if b"pk_cmax_value" in raw:
        raise ValueError("P0 manifest contains an outcome field")
    manifest = json.loads(raw)
    if manifest["model_git_sha"] != MODEL_SHA:
        raise ValueError("P0 model SHA mismatch")

    from sisyphus.pipeline.predict import predict
    from sisyphus.resources import artifact_provenance

    rows = []
    for compound in manifest["candidates"]:
        for arm in compound["arms"]:
            row = {"candidate_id": compound["candidate_id"], "arm_id": arm["arm_id"]}
            try:
                result = predict(
                    compound["smiles"],
                    float(arm["dose_mg"]),
                    route="oral",
                    n_mc_samples=0,
                    strict=True,
                )
                meta = result.cmax_prediction
                if result.execution_status != "ok" or meta is None or result.ml_pk is None:
                    raise RuntimeError("incomplete strict prediction")
                row.update(
                    status="ok",
                    meta_cmax_mg_l=meta.cmax.mean,
                    ml_cmax_mg_l=result.ml_pk.cmax.mean,
                    interval_90=meta.residual_interval_90,
                    interval_source=meta.residual_interval_source,
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
            rows.append(row)

    payload = {
        "protocol": "docs/validation/self_run_pilot_p0.md",
        "manifest_sha256": MANIFEST_SHA,
        "model_git_sha": MODEL_SHA,
        "container_digest": container_digest,
        "artifact_provenance": dict(artifact_provenance("public")),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"arms": len(rows), "failed": sum(row["status"] != "ok" for row in rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output), sort_keys=True))
