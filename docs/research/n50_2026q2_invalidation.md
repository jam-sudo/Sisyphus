---
last_updated: 2026-07-04
parent: ./cherry_picking_process_v1.md
charter: Record that the N50 secondary-holdout cycle 2026Q2 is invalidated as a never-touch generalization instrument. Binding — the 2026Q2 freeze AAFE (5.25) must not be cited as a generalization result.
---

# N50 2026Q2 — Invalidation Record

**Status: INVALIDATED (2026-07-04).** The N50 secondary permanent holdout, cycle
2026Q2 (`data/reference/holdout_n50.json`, 50/50 curated 2026-04-22; freeze
`data/validation/n50_benchmark_2026Q2.json`, run 2026-04-24), is **not** a valid
never-touch generalization instrument. Its exclusion inventory was incomplete and
name-based, so a large fraction of the set was already inside the training and
tuning corpora. The freeze result **must not** be published or cited as a
generalization AAFE.

The instrument was retired rather than re-run; a clean successor (N50') is
deferred to a separately-authorised curation cycle. Per
[cherry_picking_process_v1.md](./cherry_picking_process_v1.md) §6, a process
failure of a gated instrument is itself a negative experiment and is recorded
here (and as DE-53 in [dead-ends.md](./dead-ends.md)).

## What N50 2026Q2 was supposed to measure

Per `docs/_internal/specs/2026-04-22-n50-secondary-holdout-design.md` and
cherry_picking_process_v1.md §1: an unbiased generalization estimate free of the
cherry-picking exposure the 107-drug holdout carries (the 107-holdout has driven
47+ config decisions — track weights, routing, meta-learner choices). N50 was to
be curated **once**, benchmarked **once**, published, then retired.

## Why it is invalid — the InChIKey-14 exclusion audit

The 2026Q2 exclusion inventory (`scripts/build_n50_exclusion.py`, original form)
checked only **three** sources — the 107-holdout, MMPK, and TDC hepatocyte — by
**drug name**. It missed five further training corpora that feed the Cmax
pipeline, the DrugBank enrichment pool, and every synonym/salt/stereo variant a
name string cannot see.

Re-auditing all 50 drugs on the **InChIKey-14 connectivity block** (stereo- and
salt-insensitive) against every shipped SMILES-bearing artifact:

| Exclusion class | Hits | Detail |
|---|---:|---|
| **Hard training corpora** (fitted-target leakage) | **21 / 50** | VDss training (17), direct Cmax training in MMPK (rifampin = **rifampicin**, torsemide = **torasemide**, paclitaxel), CLint (irinotecan, tebipenem, molidustat), CLF, bioavailability |
| **Meta-weight tuning** | 2 | elafibranor + vimseltinib appear in `data/validation/meta_weight_sweep_cache.json` — a direct violation of the §1 "no track-weight tuning against N50" rule |
| **DrugBank enrichment** (soft E4) | **47 / 50** | only 3 drugs pass the conservative E4 rule |

The synonym misses are the tell: name matching cannot equate *rifampin* with
*rifampicin* or *torsemide* with *torasemide*, but they are the same molecule and
were in the MMPK Cmax training set. InChIKey-14 catches all of them. The audit is
reproducible with `scripts/build_n50_exclusion.py --audit
data/reference/holdout_n50.json` (exits non-zero: "FAIL: 21/50 in hard corpora").

## Composition confound (separate from contamination)

Even setting contamination aside, the 2026Q2 set was enriched for cases the
production (oral-tuned) pipeline is not built to predict, so the overall number
conflates several effects:

- **16 IV drugs (32%)** — the pipeline was tuned and validated on the all-oral
  107-holdout; it under-predicts every IV drug in the set by roughly 10× (a
  route-coverage gap, not a generalization signal).
- **8 deliberately-adversarial OATP1B1 substrates** (spec §5 ECM stress subset,
  subset AAFE 8.45) — a diagnostic subset, not a representative sample.
- **1 prodrug outlier** — sepiapterin, whose parent Cmax the engine cannot model
  (no pre-systemic AKR1-reduction route), gives a ~4900× over-prediction that
  alone lifts the mean.

## The numbers (do not cite as generalization)

The freeze `overall` AAFE is **5.25** [95% CI 3.79, 7.77], N=50. A current-pipeline
smoke test (post FLUX-1 / CLF / UGT, the same headline pipeline) gives **5.27** —
i.e. the result is pipeline-version-stable, so staleness is not the issue; the
curation is.

The cleanest number extractable from the existing set — drugs clean of every hard
corpus **and** oral (N=24 genuinely-novel oral NMEs) — is AAFE **≈4.0**, versus
the 107-holdout on the same local numerics stack (~2.62). That ~1.5× gap is real,
but it **corroborates the already-documented prospective degradation** (the
FDA-NME prospective set is AAFE 3.27, root cause = bioavailability-F
under-prediction on novel chemotypes) rather than adding a new signal: the
N50-clean drugs are 2024–2026 novel scaffolds (out-of-distribution), while the
107-holdout is in-distribution held-out, so part of the gap is OOD-vs-IID, not
pure cherry-picking optimism.

## Corrective actions taken

1. **Exclusion tooling fixed.** `scripts/build_n50_exclusion.py` now keys on
   InChIKey-14 across **all** hard corpora plus DrugBank, and gained an `--audit`
   mode that fails on any hard-corpus hit. This is the gate any future N50' must
   pass **before** curation, not after.
2. **The instrument is marked invalid in-band.** `holdout_n50.json` and the freeze
   JSON carry an `invalidated` block; `scripts/run_n50_benchmark.py` refuses a
   `--freeze-run` on an invalidated file.
3. **The 5.25 result is quarantined** — not promoted to any headline table.

## Requirements for N50' (deferred, when re-curation is authorised)

- Every candidate must pass `build_n50_exclusion.py --audit` (zero hard-corpus
  hits) **and** be absent from DrugBank by InChIKey-14 — i.e. genuinely-novel
  molecules, not names the old inventory missed.
- Match the pipeline's domain: oral-majority. Keep any IV / adversarial-transporter
  drugs in a **separately-reported** subset so the headline generalization number
  is not confounded by a known route-coverage gap.
- Primary-source observed Cmax only (no back-calculation), PubChem-CID-verified
  SMILES — the existing spec's A1–A5 admission rules stand.
