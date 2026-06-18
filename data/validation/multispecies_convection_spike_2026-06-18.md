# Multi-species engine-convection spike — Bridge B / B1.x (2026-06-18)

**Harness-isolated** (`scripts/spike_multispecies_convection.py`); a synthetic two-species axial graph built via the graph API. No `predict()` / `reference_man.yaml` / holdout / `src/engine` change; headline **2.731 bit-identical**. Reuses the axial machinery (PR #79) + the engine's active-metabolite species + `zonation_weights`.

## Verdict

**YES** — FEASIBLE (YES). A synthetic TWO-species axial graph (parent liver__ax1..N + metabolite__ax1..N + shared metabolite_sink) compiles and solves on the stock engine with ZERO src/engine change: T1 compiles, T2 solves, T3 mass-conserves (mass_balance_error 3.55e-15; species balance closes, formed 52.47 = chain_end 3.378e-17 + sink_end 52.47). T4: per-zone formation is spatially resolved and inter-zone convection (TransitEdge) moves the metabolite center-of-mass toward the outlet. T5: a Damkohler sweep (Da = N*k_detox/k_conv, k_conv fixed) shows the convection-vs-local shift is large at low Da (convected/stable regime) and ~0 at high Da (local/reactive regime); monotone-decreasing = True. Production already ships well-mixed multi-species, so this validates AXIAL composition; the high-Da agreement (shift->0) validates the local-only post-processor (scripts/probe_zonal_hazard.py) as the Da>>1 limit. Harness-isolated; headline 2.731 bit-identical.

## T1–T5 confirmation

- **T1 compiles / T2 solves:** the N=10 two-species graph compiles to an ODE skeleton and solves on the stock solver.
- **T3 mass-conserves:** mass_balance_error **3.55e-15**; species balance closes — formed **52.466** = chain_end **3.37839e-17** + sink_end **52.466**.
- **T4 spatial resolution + convection:** per-zone formation; inter-zone TransitEdge convects the metabolite toward the outlet (higher center-of-mass at low Da).
- **T5 Damkohler decision map:** below.

k_conv = **5.556 /h** (proxy 1/V_zone, V_zone = 0.18 L; FIXED across the sweep). Da = N·k_detox/k_conv via k_detox = Da·k_conv/N.

## Da decision map (APAP-like pericentral formation)

| Da | shift ⟨z⟩(convected − control) | uniform-formation shift |
|---|---|---|
| 0.03 | +4.3504 | +4.4105 |
| 0.1 | +4.2956 | +4.3545 |
| 0.3 | +4.1414 | +4.1969 |
| 1.0 | +3.6343 | +3.6783 |
| 3.0 | +2.5095 | +2.5299 |
| 10.0 | +0.9878 | +0.9890 |
| 30.0 | +0.3333 | +0.3333 |

- **shift_low_da** (Da=0.03): **+4.3504** zones — convection-dominated, metabolite swept toward the outlet.
- **shift_high_da** (Da=30.0): **+0.3333** zones — reaction-dominated, profile ~ local (validates the post-processor as the Da>>1 limit).
- **monotone-decreasing:** True.
- **crossover bracket** (shift past half its low-Da value): (3.0, 10.0).

Production already ships well-mixed multi-species; this spike validates AXIAL multi-species composition (per-zone formation + inter-zone convection), conserving mass with zero engine change. The high-Da agreement (shift→0) confirms the local-only post-processor (`scripts/probe_zonal_hazard.py`) is the correct Da≫1 limit; the low-Da regime is where a transported-metabolite species is needed.
