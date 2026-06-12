# MIPD Dose Recommendation (Target Attainment) — Design

**Date:** 2026-06-12
**Status:** Approved (design); pending implementation plan
**Module:** `src/sisyphus/mipd/dosing.py` (new)
**Builds on:** `mipd/tdm.py` (IV steady-state TDM), `mipd/renal_grid.py` (renal-CL grid forward model), `regimen/solver.py`

---

## 1. Purpose

The MIPD arc inverts the engine-as-prior into a posterior PK given sparse measured data:
F latent → clint latent → CrCl individualization → steady-state IV TDM (renal-CL latent) →
weight/age covariates. `predict_tdm` produces a posterior over the patient's renal clearance
from an observed steady-state trough — but stops at *describing* the patient. It does not close
the clinical loop: **"so what dose do I give next?"**

This layer adds **dose recommendation / target attainment**: given the conditioned posterior and
a clinical target (constraints on trough, peak Cmax, and/or AUC24), recommend the
**(dose, interval)** that maximizes the probability of meeting the target under the posterior.

This is the un-walled lever. SMILES-only headline accuracy is empirically walled
(`smiles-only-cmax-empirically-walled`); the measured-data / individualization regime is the one
direction that still moves the clinical value, and dose recommendation is its natural endpoint.

## 2. Key structural fact: the engine is linear in dose (LTI)

The PBPK clearance models are concentration-independent (well-stirred
`E = fup·CLint/(Q + fup·CLint)`; no saturable Michaelis–Menten in this path). Therefore, at a
**fixed disposition** (fixed `r`, fixed interval τ), every steady-state exposure quantity scales
**linearly with dose**:

```
trough(D, τ) = (D / D_ref) · trough(D_ref, τ)
cmax(D, τ)   = (D / D_ref) · cmax(D_ref, τ)
auc24(D, τ)  = (D / D_ref) · auc24(D_ref, τ)
```

This is exact for the current engine. Consequence: the **dose** knob is inverted analytically
from one solve per interval — no per-dose re-solve. The **interval** knob is nonlinear
(accumulation factor `1/(1 − e^{−kτ})`), so it costs one engine re-solve per candidate interval.

A guard: were saturable kinetics ever added to the engine, LTI would break. The load-bearing
test (§7.1) asserts LTI holds in the actual engine, so a future nonlinearity would fail loudly.

## 3. Approach (chosen: A — interval re-solve + analytic dose)

The patient's renal-CL posterior `r` is a **patient property**, invariant to the dosing regimen.
It is inferred **once** from the observed trough under the *current* regimen, then propagated
forward to each candidate regimen.

- **A (chosen):** infer `r` once (reuse `predict_tdm`); per candidate interval τ_k, build one
  `build_renal_cl_grid` at τ_k; solve the optimal dose analytically within τ_k via LTI; compare
  intervals. Cost: 1 inference grid + K interval grids. Exact, reuses the forward model verbatim.
- **B (rejected):** full 2D (dose × interval) brute grid — re-solves per dose despite LTI, zero
  accuracy gain, K_τ × K_D cost.
- **C (rejected):** closed-form 1-compartment accumulation — fastest but discards the platform's
  graph-derived multi-compartment ODE and would diverge from `predict_tdm`'s own posterior.

## 4. Contracts

New file `src/sisyphus/mipd/dosing.py`. New public names exported from `mipd/__init__.py`.
**No changes to `PosteriorPK`, `engine/`, or `predict()`** — the recommendation is self-contained.

```python
@dataclass(frozen=True)
class Constraint:
    """A bound on one steady-state PK quantity, evaluated under the posterior."""
    quantity: str                 # "trough" | "cmax" | "auc24"
    low: float | None = None      # mg/L (trough/cmax); mg·h/L (auc24)
    high: float | None = None
    # __post_init__: quantity in the allowed set; at least one of low/high set;
    # low <= high when both are set. Else ValueError.

@dataclass(frozen=True)
class DoseTarget:
    """A set of constraints. Attainment = P(ALL constraints satisfied) under the posterior."""
    constraints: tuple[Constraint, ...]
    # __post_init__: non-empty. Else ValueError.

@dataclass(frozen=True)
class CandidateEval:
    """One (dose, interval) row of the recommendation search — for transparency."""
    dose_mg: float
    interval_h: float
    attainment_prob: float
    trough_median: float
    cmax_median: float
    auc24_median: float

@dataclass(frozen=True)
class DoseRecommendation:
    dose_mg: float                # recommended per-dose amount
    interval_h: float             # recommended interval
    attainment_prob: float        # P(all constraints satisfied) at the recommendation
    cmax: Posterior               # exposure posteriors AT the recommended regimen
    trough: Posterior
    auc24: Posterior
    target: DoseTarget
    candidates: tuple[CandidateEval, ...]
    renal_scale: Posterior        # inferred patient renal-CL posterior (carried through)
    n_eff: float                  # ESS of the inferred posterior (degeneracy diagnostic)
    warnings: tuple[str, ...]
```

**Quantity definitions** (steady state, final dosing interval `[last, last+τ]`):
- `trough` = concentration at `t = last + τ` (just before the next dose) = `grid.conc_at(r, last+τ)`.
- `cmax` = max over the final interval (the grid's `cmax`).
- `auc24` = (∫ over the final interval, the grid's `auc`) × `24/τ`.

## 5. Public function

```python
def recommend_dose(
    smiles: str,
    regimen: DosingRegimen,        # the CURRENT regimen the observation was measured under
    observations,                  # MeasuredConc trough(s) under the current regimen (may be empty)
    target: DoseTarget,
    *,
    covariates: Covariates | None = None,
    candidate_intervals: tuple[float, ...] | None = None,  # default (8.0, 12.0, 24.0) ∪ {current τ}
    dose_step_mg: float | None = None,     # round analytic dose to this increment; None = continuous
    dose_bounds_mg: tuple[float, float] | None = None,  # clamp recommended dose; None = (0, ∞)
    renal_prior_cv: float = 1.0,
    n_samples: int = 20000,
    n_grid: int = 13,
    seed: int = 0,
    kp_method: str = "rodgers_rowland",
) -> DoseRecommendation
```

IV-only (matches `predict_tdm`; oral steady-state TDM is a separate future layer).

## 6. Algorithm (data flow)

1. **Validate.** IV-only regimen (every event targets `DEFAULT_IV_NODE`, else `ValueError` mirroring
   `predict_tdm`'s message). `target` non-empty; each `Constraint` valid (enforced in `__post_init__`).
2. **Infer the renal posterior once.** Call `predict_tdm(smiles, regimen, observations,
   covariates=covariates, renal_prior_cv=..., n_samples=..., n_grid=..., seed=..., kp_method=...)`.
   Take `r_samples = post.renal_scale.samples`, `n_eff = post.n_eff`, and `post.warnings` (covariate
   warnings). Empty `observations` ⇒ posterior == prior == the a-priori engine (allowed; documented).
3. **Candidate intervals.** `candidate_intervals or (8.0, 12.0, 24.0)`, unioned with the current
   regimen's interval (`renal_grid._regimen_interval_h`), sorted and de-duplicated.
4. **Per interval τ_k (one engine re-solve each):**
   a. Construct a reference regimen at τ_k: same infusion duration as the current regimen, reference
      dose `D_ref` = the current per-dose amount, `n_doses_k = max(2, round(current_last_dose_time_h
      / τ_k) + 1)` (keeps the candidate's time-to-final-dose ≥ the original, so steady state is
      reached at least as well).
   b. `grid_k = build_renal_cl_grid(smiles, regimen_τk, n_grid=n_grid, renal_factor=renal_factor,
      body_weight_kg=..., age_years=..., kp_method=kp_method)` (covariate-derived `renal_factor` and
      weight/age threaded exactly as `predict_tdm` does).
   c. Per `r`-sample look up the reference exposures: `trough_ref[i] = grid_k.conc_at(r[i], last+τ_k)`;
      `cmax_ref[i]` and `auc_int_ref[i]` from `RenalCLForward(grid_k)`; `auc24_ref[i] = auc_int_ref[i]
      · (24/τ_k)`.
   d. **Analytic dose solve (LTI).** Each quantity ∝ D, so for the dose multiplier `m = D/D_ref`,
      sample *i* satisfies all constraints iff `m ∈ [m_lo[i], m_hi[i]]`, where
      `m_lo[i] = max over lower-bounded constraints c of (low_c / q_ref^c[i])` (0 if none) and
      `m_hi[i] = min over upper-bounded constraints c of (high_c / q_ref^c[i])` (+∞ if none).
      `attainment(m) = (1/N) · #{ i : m_lo[i] ≤ m ≤ m_hi[i] }`. The max-attainment m-region is found
      exactly by a **max-interval-overlap sweep** over the 2N sorted breakpoints.
   e. Pick `m*` in the max-overlap region by the centering rule (§6.1), apply `dose_step_mg`
      rounding and `dose_bounds_mg` clamp to get `D = m*·D_ref`, then **re-evaluate** attainment at
      the actual rounded/clamped dose so the reported number matches the recommended dose.
   f. Record a `CandidateEval(dose=D, interval=τ_k, attainment, trough/cmax/auc24 medians)`.
5. **Pick the winner across intervals** (§6.2).
6. Assemble `DoseRecommendation`: exposure posteriors scaled to the chosen D (`q_ref · D/D_ref`),
   the full candidate table, `renal_scale`, `n_eff`, and warnings.

### 6.1 Centering rule (within an interval's max-overlap m-region `[a, b]`)

- **Bounded** (a window constrains both sides): `m* = sqrt(a·b)` — geometric midpoint, maximizing the
  margin to falling out of attainment as the patient's true `r` varies.
- **Unbounded above** (only lower bounds, e.g. AUC24 ≥ 400): `m* = a` — the **smallest** dose meeting
  all floors (minimize exposure).
- **Unbounded below** (only upper bounds, e.g. Cmax ≤ X): `m* = b` — the **largest** dose under all
  ceilings (maximize efficacy under the safety ceiling).

### 6.2 Cross-interval objective (the user-chosen rule)

Primary: maximize `attainment_prob`. Tie within `ε = 1e-3` → prefer the **longest** interval (fewest
administrations/day; the point of extended-interval dosing). Remaining tie → the candidate whose
centering left the largest margin (geometric center of the widest max-overlap region).

## 7. Testing (TDD; load-bearing first)

1. **LTI exactness (load-bearing).** `build_renal_cl_grid` at `2·D_ref` equals the `D_ref` grid's
   `cmax`/`auc` scaled ×2 within solver tolerance — validates the engine-is-linear premise the whole
   layer rests on. A future saturable nonlinearity fails here.
2. Feasible trough window → recommended median trough ∈ window and attainment high.
3. Longer-interval tie-break: two intervals both attain ~1.0 → the longer τ is chosen.
4. Centering: recommended median sits central in the window, not at an edge.
5. Joint peak+trough (aminoglycoside): Cmax window + trough ceiling → recommender selects a longer
   interval to drop the trough under its ceiling while holding Cmax in window; both constraints attained.
6. Cmax-ceiling-only → dose pushed just under the ceiling. AUC24-floor-only → smallest dose meeting it.
7. Oral regimen → `ValueError` ("IV"); invalid targets (empty, bad quantity, no bound, low>high) → `ValueError`.
8. Covariate warning surfaces in `recommendation.warnings`; `dose_step_mg` respected (recommended dose
   is a multiple of the step); infeasible target → soft low-attainment warning present; `renal_scale`
   shifts off 1.0 when an observation is supplied.

**Stack-independence discipline.** Directional tests reference the engine's *own* r=1 predictions
(no absolute-concentration magic numbers), so assertions hold across the ~12% macOS/CI numerics-stack
drift — the same fix applied to the IV-TDM directional test (`df4492c`).

## 8. Error handling

`ValueError` (hard) for: oral regimen, empty `target`, invalid `quantity`, a `Constraint` with neither
bound, `low > high`. `build_renal_cl_grid`'s all-grid-points failure propagates its existing
`ValueError`. **Soft warning** (returned in `warnings`, not raised) when best attainment `< 0.5`:
`"best attainment {x:.2f} < 0.50; target may be infeasible for this patient"`.

## 9. Invariants preserved

- `engine/` and `predict()` untouched; reuses `build_renal_cl_grid` / `solve_regimen` verbatim.
- `PosteriorPK` contract unchanged — `DoseRecommendation` is a new, self-contained type.
- IV-only (matches `predict_tdm`).
- Holdout inviolable; the 2.731 Meta headline untouched (new module, off the 4-track path —
  bit-identical headline guaranteed because no production prediction path changes).
- No drug-specific branches; the recommender is identity-blind (operates on engine outputs only).
- All PK quantities are `Posterior`/`Distribution`.

## 10. Defaults (confirmed)

- Candidate intervals default `(8.0, 12.0, 24.0)` ∪ {current τ}.
- Dose output continuous analytic, with optional `dose_step_mg` rounding and `dose_bounds_mg` clamp.
- Infeasibility soft-warning threshold = attainment `< 0.5`.

## 11. Out of scope (future layers)

- Oral steady-state TDM (F + clint latent jointly) — `recommend_dose` is IV-only, mirroring `predict_tdm`.
- Discrete vial-size optimization beyond `dose_step_mg` rounding.
- Loading-dose / non-steady-state titration schedules.
- PK/PD (effect-compartment) targets — constraints are on exposure quantities only.
