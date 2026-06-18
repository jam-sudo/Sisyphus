# Bridge B / B1 Phase-0 — zonal reactive-metabolite hazard probe

> **Context.** First sub-project of **Bridge B** (the PD/toxicity surface) from the 2026-06-17 virtual-cell fusion research, and the scientific closure of **DE-50**: zonation is invariant for *bulk* first-pass, but the *per-zone* reactive-metabolite hazard is exactly where it matters. Phase-0 is a **mechanistic, qualitative** demonstration — harness-isolated, headline **2.731 untouched** — anchored on the canonical acetaminophen zone-3 (centrilobular) necrosis mechanism. Decided fidelity (brainstorm): a **post-processor on the parent's per-sub-tank concentration profile** (the reactive metabolite is NOT an engine species), with **zonated, saturable detox** and a **dose-threshold** endpoint.

## 1. Purpose & non-goals

**Purpose.** Demonstrate, on the real axial engine, that a per-zone **reactive-metabolite hazard** — local bioactivation exceeding local detox capacity — (a) **localizes** to the zone set by the bioactivation×detox zonation, (b) is **strongly zonation-dependent** even though the *bulk* parent PK is invariant to that same zonation (DE-50), and (c) exhibits a **saturable-detox dose-threshold** with zone-specificity, reproducing the acetaminophen pattern (pericentral CYP2E1 + pericentral-low GSH → zone-3 vulnerability above a dose threshold).

**Non-goals (explicit).**
- **Mechanistic/qualitative demonstration, NOT a calibrated toxicity number.** No DILI probability, no quantitative threshold dose vs clinical data (data-sparsity lessons: PGx, zonation). An honest-negative (e.g. the hazard profile turns out near-invariant) is a first-class outcome.
- **Harness-isolated.** No `predict()` / `reference_man.yaml` / holdout / `src/sisyphus/engine/` / `expand_axial` change. Headline **2.731 bit-identical**. Built on a synthetic axial skeleton.
- **The reactive metabolite is NOT an engine species.** Phase-0 computes per-zone bioactivation as a **post-processor** on the parent profile. Modeling the reactive metabolite as a transported species (downstream-detox fidelity) is a follow-up (B1.x) that would need its own feasibility spike (does the engine convect a formed metabolite downstream?).
- **Steady-state saturable detox is a proxy for GSH-pool depletion dynamics.** The finite-pool, time-depleting GSH mechanism is deferred (B1.x).
- **Single bioactivation enzyme + single detox path** on the synthetic skeleton (not the real multi-enzyme ECM liver).

## 2. Background: the zonal-hazard mechanism (and why bulk PK misses it)

The axial liver (PR #79) gives the parent's **per-sub-tank** unbound concentration profile `C_u,i(t)` (zone i, inlet→outlet = periportal→pericentral). The reactive metabolite (NAPQI-analog) is *formed* locally at rate set by the local bioactivation enzyme and *removed* locally by a saturable detox (GSH-analog):

- **per-zone formation** `R_form,i(t) = Vmax_bio,i · C_u,i(t) / (Km_bio + C_u,i(t))` (saturable MM; `Vmax_bio,i ∝ local CYP abundance`);
- **per-zone detox capacity** `Vmax_detox,i` (a *zonatable* finite capacity);
- **per-zone hazard** `H_i = ∫ max(0, R_form,i(t) − Vmax_detox,i) dt` — the reactive flux that **exceeds** local detox and would bind covalently (the covalent-binding / toxicity proxy).

**Two zonated factors, co-localized (the real acetaminophen mechanism).** Bioactivation (CYP2E1) is pericentral-**high**; GSH detox is pericentral-**low**. Both zonated toward the outlet → the pericentral zone has the highest `R_form` and the lowest `Vmax_detox` → it crosses the `R_form > Vmax_detox` threshold first → zone-3 necrosis. The hazard is governed by the **interplay** of the two gradients, not bioactivation alone.

**Why bulk PK cannot see this (DE-50 closure).** Total parent bioactivated = total hepatic extraction, which is **invariant** to the CYP spatial distribution (DE-50, plug-flow). So the same redistribution that leaves the bulk parent Cmax/AUC unchanged moves the per-zone hazard profile substantially. The zonal surface carries information **orthogonal** to the (walled) bulk headline — the Bridge-B value proposition.

## 3. Method

### 3.1 Per-zone parent profile (reuse)
Build the synthetic axial liver (`_axial_graph`, PR #79 machinery); the bioactivation enzyme abundance is distributed across sub-tanks by `zonation_weights` (the DE-50 helper). Solve a single oral dose; read the per-sub-tank parent concentration time-course from `SimResult.concentrations["liver__ax{i}"]` (unbound `C_u,i = fup · c_plasma-basis`). Bulk parent `E` via the existing `_engine_e_h` (for the G2 invariance arm).

### 3.2 The hazard post-processor (pure, tested)
`zonal_hazard(c_u_by_zone, vmax_bio_by_zone, km_bio, vmax_detox_by_zone) -> list[float]`: per-zone `H_i = ∫ max(0, MM(C_u,i; Vmax_bio,i, Km_bio) − Vmax_detox,i) dt`. Pure (numpy); no engine. `vmax_bio_by_zone` and `vmax_detox_by_zone` come from `zonation_weights × total` (independently zonatable — bioactivation pericentral-high, detox pericentral-low for the acetaminophen config).

### 3.3 The sweeps
- **Zonation sweep** (G1/G2): vary the bioactivation and detox zonation (uniform / pericentral / periportal × ratio), holding totals fixed; record the per-zone hazard profile (peak zone + magnitude) AND the bulk parent `E`.
- **Dose sweep** (G3): for the acetaminophen config (bioactivation pericentral-high, detox pericentral-low), vary the dose; record per-zone hazard vs dose, the threshold dose (first dose with any `H_i > 0`), and the first-vulnerable zone; plus a protective-lever arm (uniformly raise `Vmax_detox`) → threshold shifts up.

## 4. Metric & pre-registered gates

- **G1 — localization tracks the zonation (sanity).** For the acetaminophen config the hazard peaks at the **outlet** zone (zone 3); flipping the bioactivation/detox gradients moves the peak. (Near-trivial — a sanity check, not the result.)
- **G2 — local matters, bulk doesn't (the centerpiece, DE-50 closure).** Holding total bioactivation enzyme fixed, varying its zonation leaves bulk parent `E` ~invariant (`|ΔE/E|` below a pinned small tolerance, reusing the DE-50 result) **while** the per-zone hazard peak-zone and peak magnitude change materially (above a pinned threshold). Quantifies orthogonal information the bulk headline structurally misses.
- **G3 — saturable-detox dose-threshold + zone-specificity (the mechanism).** A dose-threshold exists: below it, `R_form,i < Vmax_detox,i` in **all** zones (hazard ≡ 0); above it, the highest-bioactivation/lowest-detox zone crosses **first** (zone 3 for the acetaminophen config). Raising detox capacity raises the threshold monotonically (the protective lever). Correctly-signed; the threshold/zone depend on the bioactivation×detox interplay.
- **Honest-negative path:** if the hazard profile is near-invariant to zonation (G2 fails) or no threshold/zone-specificity emerges (G3), report it (cf. DE-49/DE-50). No parameter tuned to manufacture a threshold.

## 5. Components
- **Extend** `src/sisyphus/validation/pgx_metrics.py`: `zonal_hazard(...)` (+ a small `mm_rate(c, vmax, km)` helper), pure, tested.
- **New** `scripts/probe_zonal_hazard.py`: `_parent_profile_by_zone(...)` (solve axial, return per-sub-tank `C_u,i(t)`), the zonation + dose sweeps, G1/G2/G3 scoring, report writer. Reuses `_axial_graph`/`_engine_e_h` (importlib, the PGx pattern) + `zonation_weights`.
- **New** `tests/unit/test_zonal_hazard.py`: `zonal_hazard` worked example, threshold behavior (zero below capacity, positive above), monotonic in detox.
- **New** `tests/integration/test_zonal_hazard_probe.py`: G1 localization, **G2** bulk-`E`-invariant-while-hazard-variant, **G3** dose-threshold + zone-specificity + protective lever, parent-profile sanity (per-zone `C_u` decreasing inlet→outlet), headline-isolation guard.
- **New** `data/validation/zonal_hazard_probe_2026-06-18.{json,md}`: sweep tables + G1/G2/G3 verdicts + the explicit conclusion (zonal hazard is a real surface orthogonal to bulk PK; acetaminophen pattern reproduced qualitatively).

## 6. Out of scope (→ B1.x / Bridge B follow-ups)
Transported reactive-metabolite as an engine species (downstream-detox fidelity; needs the convection feasibility spike). GSH-pool **depletion dynamics** (finite, time-depleting pool). Quantitative toxicity threshold/PoD vs clinical data (DOSE-L1000 calibration, B3). Productization into `pkpd.py` / a tox module / `reference_man`. Multi-enzyme real liver. Tumor/efficacy PD (B4).
