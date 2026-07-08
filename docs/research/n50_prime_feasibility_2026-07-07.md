# N50' Clean Re-Curation — Feasibility Assessment (2026-07-07)

**Status: infeasible from repository data → deferred to a human-led curation cycle.**

Follow-on to the 2026Q2 invalidation ([n50_2026q2_invalidation.md](./n50_2026q2_invalidation.md),
dead-ends.md DE-53). After the exclusion tooling was fixed to key on InChIKey-14 (PR #96), the
next step was to curate a genuinely-clean N50' (the unbiased-generalization instrument the
cherry-picking caveat calls for). This assessment answers the make-or-break question **before**
any curation: *is a clean N=50 even sourceable?* It is not.

## Verdict

A clean N50' secondary permanent holdout **cannot be sourced from the existing repository data**.
The genuinely-clean candidate pool is **0**. N=50 is infeasible from repo sources by a margin of
**−50**; even N=1 is not cleanly achievable. Building N50' requires importing ~50 fresh novel
molecules with primary-source clinical Cmax from outside the current reference set — a
**human-verified** curation effort, not an automated/agent execution.

## The feasibility funnel (`data/reference/clinical_pk.json`)

| Step | Filter | Count |
|---|---|---|
| 0 | Drugs in `clinical_pk.json` | 331 |
| 1 | Non-null clinical `cmax_mg_L` | **177** |
| 2 | After removing the 107-holdout + 76-train (by name **and** InChIKey-14) | **16** |
| 3 | After removing hard-corpus IK14 hits + unparseable SMILES → nominal clean | 2 |
| **3′** | **After inspection → genuinely valid** | **0** |
| — | OATP1B1 non-statin substrates in the clean pool (spec requires ≥3) | **0** |

- **Why step 2 is brutal:** 167 of the 331 `clinical_pk` names are literally in holdout+train —
  the clinical reference and the holdout corpus are nearly the same file. Of the 16 survivors,
  **11 are in hard training corpora** and **3 have unparseable SMILES**.
- **The 2 nominal survivors both fail on inspection:**
  - `guanfacine er` — FALSE CLEAN. Its stored SMILES is the **HCl salt** (IK14 `DGFYECXYGUIODH`),
    whose connectivity block differs from the free base (`INJOMKTZOLKMBF`), so it dodged the
    holdout-side IK14 match — but **guanfacine is explicitly in the holdout list**. This is the exact
    salt/synonym leak class the tooling exists to catch; the counterion in the stored SMILES defeated
    the IK14 gate (a tooling gap — see Follow-ups).
  - `lanthanum carbonate` — inorganic multi-fragment salt (2×La³⁺ + 3×carbonate), not a single-active
    small molecule, and present in DrugBank.

## Why the constraints bind

- **E1–E3** (107-holdout / MMPK-Cmax / TDC-CLint): the clinical reference **is** the training source.
- **E4** (DrugBank-absent): DrugBank covers **14,154** InChIKey-14 blocks; the 2026Q2 attempt had
  **47/50** in DrugBank. "Not in DrugBank" ≈ "not a catalogued drug" → only very-new molecules qualify.
- **E5** (not in any validation file): the prospective N=28 novel-drug set is already consumed (it
  lives in `data/validation/`), so those cannot be reused.

## The E4 / public-clone reframe (a yield lever, not a rescue)

The 2.743 headline is **public-clone** (DrugBank hidden). In that regime a drug being *in* DrugBank
does **not** leak into its prediction — the fup/pKa/logP enrichment is disabled. So E4's
"DrugBank-absent" rule is over-conservative *for a public-clone benchmark*. **If** the N50 benchmark
commits to running public-clone (DrugBank hidden at freeze — note `run_n50_benchmark.py` does **not**
currently hide it), E4 could relax to **hard-corpora-clean only** (the actual fitted-target leakage).
But even with E4 relaxed, the repo pool is only ~2–5 — still far short of 50. The reframe improves a
*fresh* curation's admissible yield; it does not rescue the repo pool.

## What a real N50' requires (deferred, human-led)

1. ~50 fresh novel molecules (likely 2024–2026 NMEs **not** already in the prospective N=28 set), each with:
   - Primary-source observed Cmax — **no back-calculation from AUC + t½** (A2), ≥2 independent sources
     (the prospective adversarial-verification discipline);
   - PubChem-CID-verified canonical SMILES (A3), canonicalized to the free base;
   - Source DOI / table reference (A5);
   - ≥3 **non-statin** OATP1B1 substrates;
   - oral-majority, with any IV / adversarial-transporter drugs quarantined into a **separately-reported
     subset** (the 2026Q2 composition-confound lesson).
2. Each candidate gated **pre-curation** through `scripts/build_n50_exclusion.py --audit` (zero
   hard-corpus hits) + a DrugBank IK14 review.

## Integrity constraint — why this is human-led, not agent-automated

An agent must **not** generate the primary-source Cmax / SMILES / DOI values for a *never-touch*
generalization instrument. A single hallucinated value would invalidate the instrument's entire
purpose — **worse** than the 2026Q2 contamination, because a fabricated "primary source" is
undetectable by the IK14 gate (which checks structure membership, not value provenance). The sourcing
and verification of the actual clinical values must be human-performed. An agent may scaffold (draft
candidate lists, run the IK14 audit, check SMILES parse) but may not be the source of record for the
observed values.

## Follow-ups (tooling, non-blocking)

- **Salt canonicalization in the IK14 gate.** The guanfacine case shows a counterion in a stored
  SMILES yields a different IK14 than the free base, silently defeating both the exclusion match and
  the audit. `build_n50_exclusion.py` should strip salts / take the largest organic fragment before
  computing IK14 on both the corpus and the candidate side. (Low priority — it made a *contaminated*
  drug look clean here, i.e. it only ever produces false-negatives on exclusion, which a human curator
  would catch, but it is a real gap.)

## Status

**Deferred to a separately-authorized, human-led curation cycle.** The tooling (IK14 exclusion +
`--audit` gate, PR #96) and the governance spec
(`docs/_internal/specs/2026-04-22-n50-secondary-holdout-design.md`) are ready. The blocker is the
sourcing of fresh, primary-verified clinical data, which is out of scope for automated curation. Until
then, the cherry-picking caveat stands: the 107-holdout AAFE (2.743) is a point estimate whose
true-generalization value is not independently instrumented.
