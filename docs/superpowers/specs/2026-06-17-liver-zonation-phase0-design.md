# Phase-0 zonated-liver probe — does CYP zonation change first-pass?

> **Context.** Bridge A from the 2026-06-17 virtual-cell fusion research: cell/spatial atlases parameterize the **axial liver** (`liver__ax1..N` serial sub-tanks from `expand_axial`, PR #79) with **zone-specific** enzyme abundances instead of the uniform `1/N` split. Real liver is zonated — CYP3A4/CYP2E1/CYP1A2 are pericentral (zone 3 = perivenous = the **outlet** sub-tanks), ~2–3× over periportal (Halpern & Itzkovitz 2017; 2024–25 human spatial atlases; magnitudes from Achour-style targeted proteomics). This Phase-0 **gates the mechanism on the real engine before any productization**: is the effect material and defensible, or within noise (an honest negative, like the PGx systemic arm)?

## 1. Purpose & non-goals

**Purpose.** Measure whether redistributing a hepatic enzyme's abundance along the axial liver according to a literature zonation gradient — **preserving the organ total** — changes first-pass extraction `E` / Cmax versus the uniform (`1/N`) bulk split, and whether the change is (a) **material** at a physiological gradient and (b) **correctly-signed and defensible** in the saturable regime.

**Non-goals (explicit).**
- **Sensitivity/extensibility demonstration, NOT an accuracy claim.** Zone-resolved human CYP *proteomics* is sparse (spatial atlases give mRNA *patterns*; protein *magnitudes* are limited targeted-MS data); this probe does not claim a Cmax accuracy gain. An honest-negative magnitude is a first-class outcome.
- **Harness-isolated.** No `predict()` / `reference_man.yaml` / holdout / engine (`src/sisyphus/engine/`) / `expand_axial` change. Headline **2.731 bit-identical**. Zonation is applied to a **synthetic** axial skeleton via `dataclasses.replace`, exactly as the PGx harness forces models.
- **No productization here.** A YAML `zonation_profile` on the `Node` contract + teaching `expand_axial` to consume it is a *follow-up*, gated on this probe.
- **Single-enzyme synthetic skeleton.** Not the real multi-enzyme ECM liver (which would confound the mechanism).
- **No fitting / no cherry-picking.** The gradient steepness is swept across the plausible literature range and reported in full (anti-cherry-pick, per the PGx box gate).

## 2. Background: why zonation can change first-pass (the derivable expectations)

The axial liver is N serial well-stirred sub-tanks, flow `Q`, full dual hepatic inflow into tank 1 (periportal, zone 1) and outflow from tank N (pericentral, zone 3 → venous). With the v2.2a intrinsic-clearance flux, each tank's survival fraction is `s_i = Q/(Q + fup·CLint_i) = 1/(1+a_i)`, `a_i = fup·CLint_i/Q`; total survival `S = Π s_i`, `E = 1−S`. Redistribution holds `Σ a_i = A` fixed (organ total preserved).

- **Linear regime (no `Km`).** `Σ log(1+a_i)` is concave in the `a_i`, so by Jensen it is **maximized at uniform** → `S` minimized → **`E` maximized at uniform**. Therefore **any zonation (unevenness) reduces `E`** → higher F → higher Cmax. The reduction grows with gradient variance. **No dependence on direction** (the product is symmetric in the `a_i`): pericentral ≈ periportal.
- **Saturable regime (v2.2a `enzyme_km`).** `a_i` now depends on the *local* concentration via `1/(1+C_u,i/Km)`. Inlet tanks see higher `C_u` → more saturation → locally *less effective* enzyme. So placing enzyme **pericentrally (outlet, lower `C_u`, less saturated)** retains **more** extraction than placing it **periportally (inlet, higher `C_u`, more saturated)** at equal unevenness: **`E_pericentral > E_periportal`**. This direction-dependence is a **pure saturation effect** — absent in the linear regime.

**The defensible, falsifiable signature:** linear shows **no** direction-dependence; saturable shows `E_pericentral > E_periportal`. The contrast isolates a saturation-specific spatial effect tied directly to the v2.2a flux. Both effects (convexity magnitude, saturation direction) are monotonic and well-behaved — expected to be **more robust** than the regime-knife-edge PGx systemic sign (DE-49).

## 3. Method

### 3.1 The zonation weight profile (pure, tested)
`zonation_weights(n, ratio, direction, shape="linear") -> list[float]` returns `n` weights summing to **1.0**:
- `direction="pericentral"` → monotonically **increasing** toward tank N (outlet); `"periportal"` → decreasing; `"uniform"` → all `1/n`.
- `ratio = w_max / w_min` (the periportal:pericentral fold; `ratio=1` ⇒ uniform).
- `shape="linear"` (evenly-spaced) for the primary probe; the function is shape-extensible.

### 3.2 Applying zonation (total-preserving)
`apply_zonation(axial_graph, gene_tag, weights)` sets each sub-tank `i` abundance `= total × weights[i]` (where `total` = the summed gene abundance across the uniform sub-tanks = the parent organ abundance), via `dataclasses.replace`. **Invariant: `Σ abundance_i` is identical to the uniform graph** (pure redistribution). Sub-tank order follows the axial inlet→outlet order (`liver__ax1` … `liver__axN`).

### 3.3 The sweep (both gates, robustness built in)
For each **regime** ∈ {linear, saturable}, **direction** ∈ {uniform, pericentral, periportal}, over a grid of **ratio** ∈ {1.5, 2, 3, 4} × **N** ∈ {6, 10} × (saturable only) **Km** spanning the engaging range:
- build the synthetic axial skeleton (reuse `_axial_graph`); anchor `cltot` to a target first-pass `E` (reuse `_engine_e_h` / `anchor_em`);
- apply the zonation weights; measure first-pass `E` (`_engine_e_h`) and Cmax (`_cmax_auc_tmax`);
- record `ΔE = E_zonated − E_uniform`, relative `ΔCmax`, and (saturable) `E_pericentral − E_periportal`.

## 4. Metric & pre-registered gates

- **G1 — magnitude (material?).** At a physiological gradient (ratio 2–3), `|relative ΔCmax|` (and `|ΔE|`) exceeds a pre-registered threshold (**plan-pinned, e.g. ≥5% relative ΔCmax**), and the sign/size is **stable across ratio ∈ [1.5,4], N, Km** (reported in full — no favorable-corner pick). If `|ΔCmax|` is within noise across the range → **honest negative** (zonation is second-order; log it; productization not warranted on magnitude alone).
- **G2 — saturable direction (defensible signature).** `E_pericentral > E_periportal` for the saturable enzyme (margin > a pinned tolerance), robust across the Km×ratio grid; and the **linear** control shows `E_pericentral ≈ E_periportal` (no direction-dependence) — confirming G2 is saturation-specific. A direction-dependence in the *linear* control would falsify the mechanism (fail).
- **Oracle (sanity).** Uniform zonation (`ratio=1`) reproduces the unmodified axial `E` bit-identically (the redistribution is a no-op at ratio 1).
- **Honest-negative path:** report G1/G2 as-is; if magnitude ties, say so (cf. DE-49). No threshold tuned to force a pass.

## 5. Components
- **Extend** `src/sisyphus/validation/pgx_metrics.py`: `zonation_weights` (pure, tested).
- **New** `scripts/probe_liver_zonation.py`: `apply_zonation`, the sweep, G1/G2 scoring, report writer. Reuses the synthetic-engine helpers from `scripts/validate_pgx_cmax_v2b.py` (`_axial_graph`, `_well_stirred_graph`, `_drug`, `_sat_drug`, `_engine_e_h`, `_cmax_auc_tmax`, `anchor_em`) via `importlib` load (the pattern the PGx tests already use) — no edit to the merged harness.
- **New** `tests/unit/test_zonation_weights.py`: sum-to-1, monotonic per direction, `ratio` correct, uniform at `ratio=1`.
- **New** `tests/integration/test_liver_zonation_probe.py`: total-abundance preserved (exact); linear → zonation reduces `E`, pericentral≈periportal (no direction); saturable → pericentral > periportal (direction signature); ratio-1 oracle no-op; headline-isolation guard (holdout cache + `test_mm_headline_bit_identity` + `test_cached_holdout_aafe_is_2p731`).
- **New** `data/validation/liver_zonation_probe_2026-06-17.{json,md}`: sweep table + G1/G2 verdicts + honest-negative note.

## 6. Out of scope (→ follow-up if gated in)
A productized `zonation_profile` on the `Node` contract consumed by `expand_axial`; real-substrate (midazolam) zonation on the reference liver; multi-enzyme zonation; sourcing zone-resolved CYP proteomics; non-CYP zonation (e.g., periportal UGT). MIPD/PD coupling (Bridge B).
