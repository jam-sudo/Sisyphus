# RBP-2: Blood:Plasma Concentration-Basis Cleanup (the unfinished FLUX-1 class) — Design

**Date:** 2026-06-04
**Status:** IMPLEMENTED (branch `fix/rbp-concentration-basis`, TDD, all tests green) — awaiting review/commit
**Depends on:** FLUX-1 (`2026-06-03-flux1-extraction-double-count-design.md`) — this is the explicitly-deferred "RBP-basis" follow-up (FLUX-1 spec §7 / §2 note line 57).
**Branch:** `fix/rbp-concentration-basis`

**Outcome (2026-06-04):** Implemented per §2 (flow blood-pool gate; clearance/GFR/extended/prodrug
plasma-basis; diffusion unchanged) in `engine/flux.py` + `rhs_jax.py`. **Empirically confirmed
bit-identical on the entire holdout**: all 107 holdout drugs + midazolam realise RBP=1.0 (the RBP model
resets out-of-band predictions to 1.0), so the cache-pin `test_cached_holdout_aafe_is_2p784` passes
unchanged (headline 2.784 untouched) and the reporting basis change was therefore **descoped** (no
RBP≠1 drug reaches the production observation path; deferred as a separate semantic PR — see §7 Out).
The 3 curated engine-validation goldens with RBP≠1 (midazolam 0.66 / warfarin 0.58 / propranolol 0.81)
moved in the correct fu_b direction (lower Cmax: −53% / −31% / −27%) and were updated; caffeine (RBP=1.0)
is the bit-identical no-op witness. Tests: new `tests/unit/test_rbp_concentration_basis.py` (TDD red→green),
764 unit + 139 integration/regression pass, 0 failures.

## 1. Problem — the engine has no single convention for what a node's `A/V` means

FLUX-1 fixed the worst instance of a *class* of dimensional double-application. The post-FLUX-1 audit
(48-agent, adversarially verified) found the same class, unfinished, around the **blood:plasma ratio
`RBP`**. The state vector holds amounts (mg); a node's concentration is `A/V`. The engine's flux
specs disagree about whether `A/V` is *whole-blood*, *plasma*, or *unbound*:

| Site | code | implied meaning of blood-pool `A/V` | correct? |
|---|---|---|---|
| `FlowFluxSpec` (`flux.py:163`, `rhs_jax.py:273-278`) | `c_out = A·RBP/(V·Kp)` | plasma (×RBP→blood) | **wrong for blood pools** |
| Clearance sink ×4 (`flux.py:250-258`/`347-353`, rhs mirrors) | `fup·CLint·c_out`, `c_out=A·RBP/(V·Kp)` | applies *plasma* `fup` to a *blood* conc | **wrong (missing /RBP)** |
| `gfr_filtration` (`flux.py:294-302`, rhs `339-344`) | `renal_cl·(A·RBP/(V·Kp))`, var named `c_plasma` | applies *plasma* `renal_cl` to *blood* conc | **wrong (extra ×RBP)** |
| `DiffusionFluxSpec` (`flux.py:478-481`, rhs `397`) | `cu = fup·(A/V)/RBP` | whole-blood (/RBP→plasma) | **correct** |
| Reporting (`solver.py:79`, `endpoints.py:23`) | venous Cmax = `A/V`, labelled "plasma" | whole-blood, compared to clinical *plasma* | **wrong (missing /RBP)** |

Two consequences the audit verified empirically: (a) systemic CL/realized hepatic E moves **~1.7×**
across RBP 0.7→1.5 (probe), i.e. the same first-pass lever FLUX-1 targeted, still RBP-confounded; and
(b) the `*_vasc` nodes (`adipose_vasc` …) are a blood_pool that is simultaneously the source of a flow
edge (×RBP) and a diffusion edge (/RBP) → an **internal RBP² self-contradiction**. The code also
contradicts its own docstrings (`flux.py:131-135`, `:162` "For blood: RBP cancels or =1" — false) and
the JAX path computes `node_is_blood` (`params_jax.py:53,147`) but never uses it to gate RBP.

**External corroboration:** the canonical well-stirred hepatic clearance uses the **blood unbound
fraction `fu_b = fup/Rb`** as its driving-force unbound concentration (Pang & Han 2019, Biochem
Pharmacol PMID 31398312; standard PBPK). So the clearance-sink error is a deviation from the textbook
equation the engine claims to implement, not merely an internal inconsistency. (Deep-research
2026-06-04, [[external-pbpk-benchmark-bar]].)

## 2. First-principles resolution — `blood_pool A/V ≡ whole-blood C_blood`

⚠ The two audit auditors reached **opposite** conclusions (one: "A/V=plasma ⇒ flow right, diffusion
wrong"; the other: "A/V=blood ⇒ diffusion right, flow wrong"), and the synthesis muddled it. The
ambiguity is settled by the **volume definition**, not opinion:

- `reference_man.yaml`: `venous_blood` (3.7 L) + `arterial_blood` (1.5 L) ≈ **total blood volume**; the
  `*_vasc` volumes are vascular *blood* volumes. So a blood pool's `V` is a **whole-blood** volume ⇒
  `A/V` is unambiguously **whole-blood concentration `C_blood`**.
- Definitions: `C_plasma = C_blood/RBP`; unbound `C_u = fup·C_plasma = fup·C_blood/RBP`.
- Tissue node: `A/(V·Kp) = C_plasma` (Kp is tissue:plasma); blood leaving a perfusion-limited tissue
  is `C_blood,out = RBP·C_plasma = A·RBP/(V·Kp)`.

Under this single convention, each flux's *correct* form is:

| flux | correct expression | change vs current |
|---|---|---|
| Flow, **blood_pool** source | `Q·(A/V)` | **drop ×RBP** (gate on `is_blood_pool`) |
| Flow, **tissue** source | `Q·A·RBP/(V·Kp)` | unchanged ✓ |
| Clearance sink (liver/gut, all 4 models) | `fup·CLint·C_plasma = fup·CLint·A/(V·Kp)` ≡ `fu_b·CLint·c_out` with `fu_b=fup/RBP` | **÷RBP** (use blood unbound) |
| GFR filtration (kidney) | `renal_cl·C_plasma = renal_cl·A/(V·Kp)` | **drop ×RBP** |
| Diffusion (vasc↔tissue) | `cu = fup·(A/V)/RBP` (blood_pool) / `fup·A/(V·Kp)` (tissue) | **unchanged ✓** (synthesis was backwards) |
| Reporting (venous Cmax) | `C_plasma = (A/V)/RBP` | **÷RBP** or document blood-basis |

The diffusion edge is **already correct**; fixing FLOW (not diffusion) resolves the RBP² contradiction
on the `*_vasc` nodes (both then treat `A/V` as blood).

**Implementation note (cleanliness):** rather than scatter `/RBP`, drive every *unbound/plasma* consumer
off the **plasma** concentration `A/(V·Kp)` and every *convective* consumer off **blood**
`A·RBP/(V·Kp)` (blood_pool: Kp=1, RBP=1). A single helper (`c_plasma(node)`, `c_blood(node)`) gated on
`is_blood_pool` removes the ambiguity at the source and prevents recurrence.

## 3. Bit-identity guarantee (why the headline barely moves)

Production RBP is heavily clamped: `predict/adme.py:208` clips to `[0.5, 3.0]` and `:210-212` **resets
to exactly 1.0 whenever `|RBP−1| > 0.5`** (RBP model R²=−0.08 — mostly resets). So:
- For every drug with realized RBP = 1.0 (the majority), **all four fixes are exact no-ops** ⇒ the
  `realize_means` headline path is **bit-identical** for those drugs.
- Only drugs with surviving RBP ∈ [0.5, 1.5]\{1.0} change, and by ≤ ~1.5× per affected flux.
- The flow/GFR/reporting fixes are second-order on Cmax; the **clearance-sink `fu_b`** fix is the one
  with first-pass leverage (it scales hepatic+gut extraction by `1/RBP`), so it is the only part that
  can move the holdout — bounded and only for RBP≠1 drugs.

This is testable as an acceptance gate: **assert the regenerated `4track_holdout_predictions.json` is
bit-identical for every holdout drug whose realized RBP = 1.0**, and enumerate the RBP≠1 movers.

## 4. Recalibration — hold midazolam `E_gut` invariant (only if midazolam RBP ≠ 1)

Liver affinities are XGBoost-decomposed (physiological in-vitro CLint) ⇒ no liver recal. The gut
CYP3A4 abundance is the midazolam back-fit (re-anchored by FLUX-1 to 1.384e7). The `fu_b` change scales
gut extraction by `1/RBP_mdz`; if midazolam's *realized* RBP ≠ 1.0 (curated value 0.66 survives the
clamp), re-anchor gut CYP3A4 by `k = RBP_mdz` to hold `E_gut = 0.2582` invariant — same mechanism and
midazolam-train-not-holdout justification as FLUX-1 §4. If midazolam's realized RBP = 1.0 in the
benchmark path, **no re-anchor is needed** (the fix is a no-op for it). Verify empirically before
touching the YAML.

## 5. Validation, regen, honest report (FLUX-1 playbook)

1. **Failing test first:** `tests/unit/test_rbp_concentration_basis.py` — at RBP≠1 on a blood_pool IV
   bolus: (a) convective venous `A/V` is RBP-independent; (b) reported plasma Cmax = blood/RBP; (c)
   hepatic `E` matches `fu_b·CLint/(Q+fu_b·CLint)` with `fu_b=fup/RBP`; (d) GFR rate uses plasma. Each
   currently fails; pins the corrected physics.
2. numpy↔JAX parity assert on a metabolized + renally-cleared drug at RBP≠1.
3. Update formula-encoding unit tests (`test_flux.py` flow/clearance/gfr references) to the corrected
   basis — verify each is *closer to correct*, not blindly rewritten.
4. Re-anchor gut CYP3A4 only if §4 condition holds; verify midazolam `E_gut` empirically unchanged.
5. **Regenerate** `4track_holdout_predictions.json`, refresh CI bootstrap, re-run prospective N=28.
   Acceptance: RBP=1 drugs bit-identical; report the true Meta/Engine/In-domain/Prospective AAFE
   **whatever it is** (correctness-first; the fix may regress — ship anyway per [[correctness-over-benchmark]]).
6. Reconcile CLAUDE.md metrics block against the regenerated cache; experiment-log entry; if it
   regresses with no compensating correctness story beyond "matches the canonical equation," that *is*
   the story (cf. FLUX-1).

## 6. Risks

- Headline may move for RBP≠1 drugs (bounded by the clamp; accepted). The `fu_b` fix is the only first-
  pass lever; flow/GFR/reporting are second-order.
- The **reporting** `÷RBP` (venous → plasma) changes the *comparison basis* for RBP≠1 drugs even where
  internal dynamics are unchanged — this is a real Cmax shift and must be in the same regen, not split.
- Stiffer ODE only if extraction rises materially (RBP<1 drugs); watch `solver_success` + mass balance.

## 7. Scope

- **In:** FlowFluxSpec blood-pool RBP gate; `fu_b=fup/RBP` in all 4 clearance models; GFR `c_plasma`
  basis; venous→plasma reporting; numpy+JAX; conditional gut re-anchor; regen; docs. Optional: the
  `c_plasma()/c_blood()` helper refactor to prevent recurrence.
- **Out:** SBI/TDM posterior re-validation (separate spec); MC enzyme-correlation + gut CYP3A4 cv
  (separate, PI-path); DDI `fu_perpetrator`; NCA λz. Not bundled — each is its own correctness unit.

## 8. Hard-constraint check

- #1 identity-blind — fix is node-*type*-gated math (`is_blood_pool`), no name matching. ✓
- #5 holdout inviolable — re-anchor (if any) uses midazolam (train). ✓
- #6 no drug-specific branches — single global convention; no `if drug==`. ✓
- #8 hard no-touch — touches `engine/flux.py`, `rhs_jax.py`, reporting in `solver.py`/`endpoints.py`
  (allowed); NOT `compiler.py`/`solver.py` *integrator*, NOT DrugOnGraph fields, NOT holdout list;
  re-anchor is a physiological anchor, not a Cmax-loss fudge. ⚠ Confirm the `solver.py:79`/`endpoints.py`
  reporting edit does not touch the no-touch solver internals — it is a post-solve concentration label
  conversion, not integrator logic. ✓ (verify at implementation)
