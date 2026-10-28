# Meta-Review of Engine Audit — 2026-06-04

**Subject:** `docs/engine_audit_findings_2026-06-04.md`
**Scope:** Review-only. No code was modified. Every finding's code claim was independently
verified by reading the cited source and, where useful, by read-only runtime probes
(in-memory graph construction; no repo files touched). The two highest-severity findings
(P1, P1/P2) were additionally cross-checked by hand.

## Overall Verdict

**Trustworthy and largely accurate.** All 7 items' *factual* claims about the code ("line X
says Y") were confirmed — zero fabricated or misread code claims. Where the audit slips, it is
consistently in *interpretation / scope* (overstating reachability or audience, or
under-stating that a defect is even more dormant than claimed), never in the underlying code
facts. The audit also correctly hedges most findings as latent ("May Report",
"experimental-only", "future-platform risk").

Two adjustments recommended: down-rate the JAX-divergence finding (P2 → P3), and widen the
README finding's scope (3/4 rows stale, RBP-2 omission).

## Per-Finding Verdicts

| Finding | Verdict | Key evidence |
|---|---|---|
| **P1** — blood-vs-plasma reporting basis | **CONFIRMED** | `solver.py:78-83` emits blood-pool `A/V` as "blood concentration"; no `/RBP` conversion in `endpoints.py:23,32` or `predict.py:449-450`. Probe reproduces RBP=2 → reported Cmax 20 vs plasma-expected 10. Severity **P2 right (P3 in current production)** |
| **P1/P2** — extended-ECM drops hepatic `fu_correction` | **CONFIRMED** (most important) | Liver flagged (`reference_man.yaml:80`) but its edge is `model: extended` (`:343`); extended branch (`flux.py:310-360`) uses `fup` directly, while `well_stirred` (`:244-245`), `parallel_tube` (`:279-280`), and `ProdrugActivation` (`:658-659`) apply the correction. Probe: well_stirred scales 5×, extended 1.0× (ignored). Severity **P2 right** |
| **P2** — `parallel_tube` collapses to well_stirred | **CONFIRMED** | Two branches mathematically identical (bit-identical runtime probe); `flux.py:178` docstring still advertises a nonexistent "PT exponential extraction". Severity **P2 right** |
| **P2** — JAX/SciPy `fu_correction` divergence | **OVERSTATED** (factually accurate, severity too high) | Code claims confirmed, but "non-ECM graphs silently drift" implies a reachable config that does not exist (reference liver uses `extended` → fail-louds under JAX; all other nodes unflagged/identity-valued). **Down-rate to P3** |
| **P3** — ActiveTransport target-local vs efflux docstring | **CONFIRMED** | Reads abundance/IVIVE from the *target* node (`flux.py:545/557/576`); docstring advertises efflux/secretion; tests cover uptake only; zero `active_transport` edges ship. Severity **P3 right** |
| **P2** — README Omega-parity staleness | **CONFIRMED** (audit *under-states* scope) | README still frames all 4 drugs as Omega parity. Actually **3 of 4 rows** are stale (warfarin also moved); the dominant mover is **RBP-2**, not FLUX-1, and RBP-2 is absent from the README entirely |
| "Healthy" test run | **CONFIRMED** | "49 passed, 1 skipped" reproduces; the skip is `importorskip("jax")`. Count is environment-dependent (50 passed / 0 skipped with jax installed) |

## Most Important Real Issue

**P1/P2 — the `extended` ECM branch silently drops the hepatic `fu_correction`.** Highest value
because:

- The production liver runs exactly that `extended` model.
- The correction registry is *designed to grow* — `hepatic_fu_correction.py:77-84` enforces a
  `mean >= 1.0` anti-fudge guard, anticipating future literature values.
- No test guards the production path — B-11 covers only `well_stirred`/`parallel_tube`
  (`test_flux_fu_correction_integration.py:11`: "ECM (extended) branch is out of B-11 scope").

The first time someone curates a real correction value, it will **silently no-op on the liver
and pass all tests.** The adversarial pass sharpened this: `fu_correction_applicable` is set on
the liver node *only* (gut_wall is unflagged, `graph/types.py` default 0.0), so the correction
is structurally dead on *every* clearance edge in the reference physiology — the one node that
opts in uses the one model that drops it.

Currently a strict no-op numerically (DrugOnGraph default 1.0; all 19 registry entries
mean=1.0), so the 2.784 headline is unaffected today — but it is a live wiring inconsistency,
not a cosmetic one.

## Where the Audit Is Wrong / Misleading

1. **JAX divergence is over-severity-rated.** "Non-ECM graphs silently drift" implies a
   reachable production config; none exists (reference liver fail-louds under JAX; everything
   else is unflagged or identity-valued). Should be **P3**.
2. **README finding under-states scope.** **3 of 4 rows** stale (not 2), and the dominant
   correctness fix driving the divergence (**RBP-2**) is *entirely absent* from the README —
   the audit attributes the gap to FLUX-1 alone.
3. **`parallel_tube` mis-targets audience.** The public `predict()` API exposes no
   clearance-model selector; the real victim is an internal researcher via
   `run_chain_benchmark.py`, not an external API user. (Also: no type literally named
   `ClearanceModel` exists — it is `ClearanceEdge.model: str`, validated only by docstring.)

## Unverifiable Claim

Only one material claim could not be independently confirmed: the spec's assertion that **all
107 holdout drugs + midazolam empirically realize RBP=1.0 on the production path** (the basis
for P1 being dormant on the headline). The full holdout benchmark was not re-run, and the
`adme.py:208-212` clamp does *not* structurally force RBP=1 — it only resets when
`|RBP-1| > 0.5`, so any drug landing in `[0.5,1.5]\{1.0}` would survive and be mis-reported.
**P1's dormancy is therefore empirical (asserted by the spec, unverified here), not
structural.**

## Methodology Note

Per-finding verification used two independent agents (a verifier and an adversarial challenger),
which caught two reasoning errors in the first verifier that the challenger corrected — both in
the *less-severe* direction:

- **P1:** the claim that the bug "already affects" the RBP≠1 engine-validation goldens was
  refuted — those goldens are self-consistent whole-blood snapshots regenerated from this same
  engine (`test_engine_validation.py:28-48`), so the basis omission produces zero error there.
- **P1/P2:** the claim that a future curated value "would still fire on the gut (well_stirred)"
  was refuted — gut_wall is unflagged, so the correction is dead on every reference edge.

---

*Generated as a review of the audit document; no source files were changed.*
