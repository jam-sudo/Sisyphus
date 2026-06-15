# Michaelis–Menten saturable metabolic clearance — engine capability (v2.2a)

> **Why now.** The PGx v2.1 Cmax-fold milestone halted at its feasibility gate
> (`data/validation/pgx_cmax_feasibility_2026-06-15.md`): the high-first-pass drugs whose
> genotype Cmax-fold diverges from the AUC-fold are almost all **saturable**, which the
> engine's **linear** clearance flux cannot represent. This spec adds that missing capability.
> It is the foundation for **v2.2b** (the nonlinear genotype Cmax/AUC validation), but is a
> standalone, generally-useful engine feature (high-dose PK, DDI victim saturation,
> nonlinear drugs broadly).

## 1. Purpose & non-goals

**Purpose.** Give the engine saturable (Michaelis–Menten) metabolic clearance —
`CLint = Σᵢ Vmaxᵢ/(Kmᵢ + C_u)` — that **reduces exactly to the current linear flux** when no
`Km` is supplied. The capability is identity-blind (the engine reads `Km` from the drug
contract, not by name) and off the production path by default.

**Non-goals (explicit).**
- **No production-path activation.** `predict()` never populates `enzyme_km`; production stays
  linear. Headline **2.731 is bit-identical** (regression-pinned).
- **`well_stirred` only.** ECM/`extended` intracellular saturation needs a different
  concentration basis (PS_active concentrates the drug) — deferred.
- **No mechanism-based inhibition.** Time-dependent enzyme inactivation (omeprazole
  auto-inhibition, `kinact/KI`) is a separate capability (v2.3); omeprazole is therefore out
  of the v2.2b set.
- **No saturable protein binding / distribution.** Only the enzyme saturates; `fup` and `Kp`
  stay concentration-independent (stated assumption, §3).
- **No genotype / validation here.** That is v2.2b (consumes this capability, its own spec +
  data-feasibility gate).

## 2. Background: the model (load-bearing identities)

Single-enzyme Michaelis–Menten: `v = Vmax·C_u/(Km + C_u)`, `Vmax = kcat·[E]`, with `C_u` the
**unbound substrate at the enzyme**. For `well_stirred` (no concentrative uptake) the canonical
assumption is `C_u = fup · c_plasma`.

Summed over enzymes at a node:

```
CLint_organ(C_u) = Σᵢ Vmaxᵢ/(Kmᵢ + C_u) = Σᵢ (Vmaxᵢ/Kmᵢ) · 1/(1 + C_u/Kmᵢ)
```

The current engine computes the **linear limit** `Σᵢ Vmaxᵢ/Kmᵢ = Σᵢ abundanceᵢ·affinityᵢ`,
i.e. `affinityᵢ = CLintᵢ/abundanceᵢ = kcatᵢ/Kmᵢ`. Therefore the **existing `affinity` keeps its
meaning**, and the only new per-enzyme datum is `Kmᵢ`, with `Vmaxᵢ = abundanceᵢ·affinityᵢ·Kmᵢ`
emerging implicitly. The saturable model is the linear model with each enzyme term multiplied
by `1/(1 + C_u/Kmᵢ)`; `Kmᵢ → ∞` recovers the linear term exactly.

## 3. Method: per-enzyme saturation inside `well_stirred`

Saturable metabolism is a **drug–enzyme property**, not an organ property, so it lives inside
the existing `well_stirred` clearance branch as a per-enzyme factor — **not** a new `model=`
string (which is selected per edge at build time and could not express "drug X saturates on
the standard liver edge"). The engine stays identity-blind: it dispatches by edge model as
today and reads `Km` from the drug.

When `drug.enzyme_km` is non-empty, the `well_stirred` branch computes `c_plasma` first, then:

```
C_u = fup · c_plasma                         # fup already includes the B-11 liver correction
clint_organ = Σ  abundance · affinity · ivive / (1 + C_u / Km_tag)     # Km_tag = +inf if absent
rate = fup · clint_organ · c_plasma          # FLUX-1 form unchanged
```

The separate convective `Q·c_out` FlowEdge is untouched, so flow-limited extraction still
emerges from the ODE (`E = fu_b·CLint/(Q + fu_b·CLint)`); saturation only **reduces CLint as
`C_u` rises**, making extraction (and first-pass `F`) concentration- and dose-dependent.

**Assumption (in scope):** linear distribution and concentration-independent `fup`/`Kp` — only
the enzyme saturates. **`C_u` basis:** unbound plasma `fup·c_plasma`, valid for `well_stirred`
(the ECM intracellular basis is the §1 deferral).

## 4. Contract & plumbing

- **`DrugOnGraph`** (`src/sisyphus/core.py`, frozen): add
  `enzyme_km: dict[str, Distribution] = field(default_factory=dict)` — per-enzyme-tag Km on the
  **unbound-concentration basis, mg/L**. Appended after the last field (`renal_clearance`) so
  all existing construction sites (25) are unaffected; empty by default. Parallels the existing
  `enzyme_affinity_for_conversion` dict. Invariant 8 holds — no existing field is modified.
  **`DrugOnGraph.sample()` and `realize_means()` rebuild the dataclass field-by-field, so both
  must be updated to carry `enzyme_km`** (else realization silently drops it and the flux never
  sees it). The empty-dict comprehension makes zero `.sample()` calls ⇒ RNG order — hence the
  headline — is preserved. `__post_init__` rejects a non-positive Km mean.
- **`ResolvedParams`** (`src/sisyphus/engine/compiler.py`): add
  `drug_enzyme_km(tag: str) -> float` returning `self._drug.enzyme_km[tag].mean` when present,
  else `float("inf")` (⇒ saturation factor `1/(1+C_u/∞) = 1`).
- **`ClearanceFluxSpec.apply`** (`src/sisyphus/engine/flux.py`, `well_stirred` branch only):
  the guarded fork in §5.

## 5. Headline safety (the guard)

Refactoring the linear sum into per-enzyme MM terms changes floating-point associativity, so a
`factor == 1` path is **not** bit-identical. The `well_stirred` branch therefore forks:

```
if not <drug has any enzyme_km>:
    <existing linear block, verbatim>        # byte-for-byte identical → 2.731 untouched
else:
    <the §3 saturable block>
```

A cheap "has any enzyme_km" predicate is threaded through `ResolvedParams` (e.g.
`drug_has_enzyme_km() -> bool`) so the flux avoids per-call dict construction. Production
`predict()` never sets `enzyme_km`, so it always takes the verbatim branch.

## 6. Testing

- **Bit-identity (headline):** (a) regression test — the 107-holdout cache
  `4track_holdout_predictions.json` is unchanged by this work; (b) a sampled drug with empty
  `enzyme_km` produces an identical `SimResult` (same Cmax/AUC to full float precision) before
  vs after the change.
- **MM rate oracle (the RHS check):** with a finite `Km`, the `well_stirred` flux's computed
  elimination rate equals the analytic `fup·Σ abundanceᵢ·affinityᵢ·ivive/(1+C_u/Kmᵢ)·c_plasma`
  (= the `−Vmax·C/(Km+C)` RHS, `Vmaxᵢ = abundanceᵢ·affinityᵢ·Kmᵢ`) for a constructed state —
  asserted directly on `ClearanceFluxSpec.apply`, independent of accumulation (scipy's
  integrator is already trusted). A full single-compartment conc-time match is **not** used —
  the production graph is multi-compartment, so an isolated 1-comp analytic would not match.
- **Saturation behaviour:** at `C_u ≫ Km`, dose-proportionality breaks — doubling the dose
  yields **>2×** AUC (supra-proportional); the effect scales with `dose/Km`.
- **Linear-limit pin:** `Km → ∞` (and `Km ≫ C_u`) reproduces the linear-model AUC within
  tolerance — the capability degrades gracefully to current behaviour.
- **Identity-blindness:** the existing engine identity-blindness test still passes (no name
  matching added); `Km` flows only through the drug contract.

## 7. Out of scope (→ later)

ECM/intracellular saturation (different `C_u` basis); mechanism-based inhibition (`kinact/KI`,
omeprazole) → **v2.3**; saturable protein binding; gut-wall saturation; any genotype scaling or
clinical validation → **v2.2b** (its own spec + a dose-ranging-data feasibility gate; powered
set = propafenone / atomoxetine / lansoprazole + metoprolol anchor, omeprazole excluded pending
MBI); any production-path `enzyme_km` population (a future DDI/high-dose registry could populate
it, separately specced).
