#!/usr/bin/env python3
"""Score prospective candidate drugs through the 4-track pipeline in
PUBLIC-CLONE state (DrugBank + logp_correction temporarily hidden), matching
the canonical prospective cache provenance.

Input : JSON list of {name, smiles, dose_mg, obs_cmax_mg_L, [cmax_basis,
        is_prodrug]}
Output: {n, drugs:[{name, smiles, dose_mg, obs, meta, engine, ml, meta_fold,
        engine_fold, ml_fold, in_ad, ad_flags, ...}]} in the public_only schema.

Usage:
    python scripts/score_prospective_candidates.py candidates.json out.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_HIDE = [ROOT / "data" / "drugbank", ROOT / "models" / "adme" / "logp_correction.json"]
_SUFFIX = ".__pubclone_hidden__"


def _hide() -> list[tuple[Path, Path]]:
    moved = []
    for p in _HIDE:
        if p.exists():
            h = p.parent / (p.name + _SUFFIX)
            p.rename(h)
            moved.append((p, h))
    return moved


def _restore(moved: list[tuple[Path, Path]]) -> None:
    for p, h in moved:
        if h.exists():
            h.rename(p)


def _fold(pred: float | None, obs: float) -> float | None:
    if pred is None or pred <= 0 or obs <= 0:
        return None
    return pred / obs if pred >= obs else obs / pred


def main() -> int:
    cands = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        ROOT / "data" / "validation" / "prospective_expansion_scored.json"

    moved = _hide()
    rows = []
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from rdkit import RDLogger
        RDLogger.logger().setLevel(RDLogger.ERROR)
        from sisyphus.pipeline.predict import predict

        for c in cands:
            name = c["name"]
            smi = c["smiles"]
            dose = float(c["dose_mg"])
            obs = float(c["obs_cmax_mg_L"])
            try:
                r = predict(smi, dose, "oral")
                meta = float(r.pk.cmax.mean)
                eng = float(r.engine_pk.cmax.mean) if r.engine_pk else None
                ml = float(r.ml_pk.cmax.mean) if r.ml_pk else None
                ad = list(getattr(r, "ad_flags", []))
                fm = _fold(meta, obs)
                rows.append({
                    "name": name, "smiles": smi, "dose_mg": dose, "obs": obs,
                    "meta": meta, "engine": eng, "ml": ml,
                    "meta_fold": fm, "engine_fold": _fold(eng, obs), "ml_fold": _fold(ml, obs),
                    "in_ad": not ad, "ad_flags": ad,
                    "cmax_basis": c.get("cmax_basis"), "is_prodrug": c.get("is_prodrug"),
                })
                print(f"  {name:18s} obs={obs:<9.4g} meta={meta:<9.4g} fold={fm:.2f}  ad={ad}")
            except Exception as exc:
                rows.append({"name": name, "error": str(exc)})
                print(f"  {name:18s} ERROR: {exc}")
    finally:
        _restore(moved)

    out_path.write_text(json.dumps({"n": len(rows), "drugs": rows}, indent=2) + "\n")
    print(f"\nWrote {out_path} ({len(rows)} drugs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
