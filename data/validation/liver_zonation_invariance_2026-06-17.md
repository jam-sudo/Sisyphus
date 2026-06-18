# Liver-zonation invariance probe — Phase-0 (2026-06-17)

**Harness-isolated** (`scripts/probe_liver_zonation.py`); no `predict()` / `reference_man.yaml` / holdout change; headline **2.731 bit-identical**. Reuses the merged axial machinery (PR #79) + v2.2a saturable flux.

## Verdict

First-pass extraction is INVARIANT to the axial spatial distribution of a hepatic enzyme (total preserved): |ΔE(N)| -> 0 as N grows (plug-flow convergence, spec §2). The finite-N effect is a CSTR-discretization artifact (linear ~1e-5, saturable ~1e-3 at N=10), not physiology. Zonation is NOT a bulk-first-pass / Cmax lever; its modeling value is zonal/local (Bridge B, zonal toxicity). Headline 2.731 untouched (harness-isolated). See DE-50.

- **G1 invariance holds:** True (|ΔE(N)| decays toward 0 and clears 5e-3 for every regime/direction/ratio).
- **G3 saturation-specific direction:** True — saturable asymmetry 1.69e-04 vs linear 3.83e-10; sign (periportal−pericentral) = +1.69e-04 (periportal extracts more — the §2 prediction: inlet enzyme faces higher [C] → higher MM rate).

## ΔE(N) convergence (the invariance demonstration)

| regime | direction | ratio | ΔE(N=5) | ΔE(N=10) | ΔE(N=20) | ΔE(N=40) | ΔE(N=80) |
|---|---|---|---|---|---|---|---|
| linear | pericentral | 2.0 | -8.29e-05 | -3.47e-05 | -1.59e-05 | -7.63e-06 | -3.74e-06 |
| linear | pericentral | 3.0 | -1.87e-04 | -7.82e-05 | -3.58e-05 | -1.72e-05 | -8.41e-06 |
| linear | periportal | 2.0 | -8.29e-05 | -3.47e-05 | -1.59e-05 | -7.62e-06 | -3.74e-06 |
| linear | periportal | 3.0 | -1.87e-04 | -7.82e-05 | -3.58e-05 | -1.72e-05 | -8.41e-06 |
| saturable | pericentral | 2.0 | -4.15e-04 | -2.06e-04 | -1.05e-04 | -5.38e-05 | -2.72e-05 |
| saturable | pericentral | 3.0 | -8.78e-04 | -4.27e-04 | -2.15e-04 | -1.09e-04 | -5.47e-05 |
| saturable | periportal | 2.0 | -2.58e-04 | -1.08e-04 | -4.75e-05 | -2.15e-05 | -1.01e-05 |
| saturable | periportal | 3.0 | -6.49e-04 | -2.82e-04 | -1.30e-04 | -6.09e-05 | -2.93e-05 |

Every row's `|ΔE|` shrinks ~monotonically toward 0 as N grows — the plug-flow invariance. The N=10 artifact magnitude (linear ~1e-5, saturable ~1e-3) is the discretization bias to keep in mind for any future axial work.
