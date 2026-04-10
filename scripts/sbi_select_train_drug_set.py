#!/usr/bin/env python3
"""Select the 50 training drugs for the multi-drug SBI amortizer (Track A).

Sampling protocol (Pre-resolved Decisions, docs/next_steps_plan.md):
  - Source: data/training/mmpk_expanded_v2.csv
  - Exclusion: any drug in data/reference/holdout.json
  - SMILES must be valid, compound_type computable, dose in (0, 5000] mg
  - Stratified random sample over compound_type (neutral / acid / base / zwitterion)
  - Seed: 42

Output: data/sbi/train_drug_set.json — one entry per drug with name, SMILES,
dose_mg, route (always "oral" for this pool), and a handful of descriptor fields
used for conditioning.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

TARGET_N = 50
SEED = 42
POOL_CSV = ROOT / "data" / "training" / "mmpk_expanded_v2.csv"
HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
OUT_JSON = ROOT / "data" / "sbi" / "train_drug_set.json"


def _profile_one(args: tuple[str, str, float]) -> dict | None:
    name, smiles, dose_mg = args
    try:
        from sisyphus.predict.chemistry import compute_profile
        p = compute_profile(smiles)
        return {
            "name": name,
            "smiles": smiles,
            "dose_mg": dose_mg,
            "route": "oral",
            "compound_type": p.compound_type,
            "mw": float(p.mw),
            "logp": float(p.logp),
            "tpsa": float(p.tpsa),
            "hbd": int(p.hbd),
            "hba": int(p.hba),
            "rotatable_bonds": int(p.rotatable_bonds),
            "pka": float(p.pka) if p.pka is not None else None,
        }
    except Exception as exc:
        logging.warning("profile failed for %s: %s", name, exc)
        return None


def load_pool() -> list[tuple[str, str, float]]:
    with open(HOLDOUT_JSON) as f:
        holdout = {h.lower().strip() for h in json.load(f)["holdout"]}

    pool: dict[str, tuple[str, float]] = {}
    with open(POOL_CSV) as f:
        for row in csv.DictReader(f):
            name = row["name"].lower().strip()
            if name in holdout:
                continue
            if row.get("in_holdout", "").lower() == "true":
                continue
            smi = (row.get("canon_smiles") or "").strip()
            if not smi:
                continue
            try:
                dose = float(row.get("dose_mg") or 0)
            except ValueError:
                continue
            if dose <= 0 or dose > 5000:
                continue
            if name not in pool:
                pool[name] = (smi, dose)

    return [(n, smi, dose) for n, (smi, dose) in pool.items()]


def main() -> None:
    print(f"[select] loading pool from {POOL_CSV.name}")
    pool = load_pool()
    print(f"[select] pool size after filters: {len(pool)}")

    print("[select] computing molecular profiles (4 workers)...")
    t0 = time.time()
    profiles: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_profile_one, item): item[0] for item in pool}
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if res is not None:
                profiles.append(res)
            if done % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(pool)} done  {elapsed:.1f}s  "
                      f"({done/elapsed:.1f}/s)", flush=True)
    print(f"[select] profiled {len(profiles)} drugs in {time.time()-t0:.1f}s")

    # Stratify
    by_type: dict[str, list[dict]] = defaultdict(list)
    for p in profiles:
        by_type[p["compound_type"]].append(p)

    print("[select] compound_type distribution in pool:")
    for t, lst in sorted(by_type.items()):
        print(f"  {t}: {len(lst)}")

    # Proportional allocation with minimum floor of 2 per type
    rng = random.Random(SEED)
    n_total = sum(len(lst) for lst in by_type.values())
    alloc = {}
    for t, lst in by_type.items():
        n = max(2, round(len(lst) / n_total * TARGET_N))
        n = min(n, len(lst))
        alloc[t] = n
    # Correct to exactly TARGET_N
    diff = TARGET_N - sum(alloc.values())
    types_sorted = sorted(by_type.keys(), key=lambda t: -len(by_type[t]))
    idx = 0
    while diff != 0 and idx < len(types_sorted) * 4:
        t = types_sorted[idx % len(types_sorted)]
        if diff > 0 and alloc[t] < len(by_type[t]):
            alloc[t] += 1
            diff -= 1
        elif diff < 0 and alloc[t] > 2:
            alloc[t] -= 1
            diff += 1
        idx += 1
    assert sum(alloc.values()) == TARGET_N, (
        f"allocation failed: {alloc} sum {sum(alloc.values())}"
    )

    print(f"[select] target allocation: {alloc}")

    # Deterministic random sample per bucket
    selected: list[dict] = []
    for t in sorted(by_type.keys()):
        lst = sorted(by_type[t], key=lambda d: d["name"])  # stable order
        chosen = rng.sample(lst, k=alloc[t])
        selected.extend(chosen)

    # Sort final list for reproducibility
    selected.sort(key=lambda d: d["name"])

    payload = {
        "source": str(POOL_CSV.relative_to(ROOT)),
        "pool_size": len(pool),
        "profiled_size": len(profiles),
        "target_n": TARGET_N,
        "seed": SEED,
        "allocation": alloc,
        "drugs": selected,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[select] wrote {OUT_JSON}")
    print(f"[select] {len(selected)} drugs selected")
    for p in selected:
        print(f"  {p['name']:>30s}  {p['compound_type']:>11s}  "
              f"dose={p['dose_mg']:>7.1f}mg  mw={p['mw']:.0f}  logp={p['logp']:.1f}")


if __name__ == "__main__":
    main()
