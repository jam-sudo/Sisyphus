# Zonal GSH-pool depletion probe — Bridge B / B1.x Phase-0 (2026-06-18)

**Harness-isolated** (`scripts/probe_gsh_depletion.py`); the GSH pool and reactive metabolite are a POST-PROCESSOR on the axial parent profile, not engine species. No `predict()` / `reference_man.yaml` / holdout change; headline **2.731 bit-identical**. Reuses the axial machinery (PR #79) + B1 harness (PR #82). `k_syn`/`tau` pinned a priori: GSH t1/2 3.0 h -> k_syn 0.231/h, tau 4.0 h.

## Conclusion

A depleting per-zone GSH pool makes the zonal reactive-metabolite hazard HISTORY-DEPENDENT: a pure concentration reordering leaves the static pointwise hazard unchanged (rel diff 0.0e+00) while moving the dynamic hazard (0.03) — structure beyond the B1 static model and orthogonal to bulk parent PK (DE-50, bulk-E span 4.3e-04). The CLEAN signature of pool memory is this ordering test, not the physical bolus-vs-divided arm: there the excess path-dependence over the static envelope baseline = -1.091 (honest-negative — the pool's escape-saturation CAPS the bolus hazard and so COMPRESSES the dynamic ratio below the static envelope ratio, rather than amplifying it). Dose transition width dynamic 0.263 vs static 0.471 (log10-dose, smaller=sharper); raising GSH0 lowers hazard (NAC lever). k_syn/tau pinned a priori from GSH t1/2. Headline 2.731 untouched (harness-isolated). Qualitative acetaminophen mechanism; not a tox number.

## G-order — pool memory (centerpiece)

Same value-multiset, reordered. Static rel diff **0.0e+00** (invariant, by construction) vs dynamic rel diff **0.029** (moves) — the pool carries order/history information the static model cannot.

## G2 — local matters, bulk doesn't (DE-50)

Bulk parent E span across bioactivation zonation **4.27e-04** (~invariant) while the dynamic hazard peak-zone moves:

| bio zonation | bulk E | hazard peak-zone | maxH |
|---|---|---|---|
| pericentral | 0.957487 | 4 | 0.189 |
| uniform | 0.957914 | 0 | 0.3264 |
| periportal | 0.957631 | 0 | 0.5273 |

## G-time — excess path-dependence (bolus vs 2x divided, equal dose)

dynamic ratio 4.355 vs static ratio 5.446 => **excess -1.091** (tau 4.0 h). The static path effect is measured, not assumed zero. **Honest-negative:** the excess is *negative* — the pool's escape-saturation caps the bolus hazard and compresses the dynamic bolus/divided ratio BELOW the static envelope ratio. So the physical divided-dose arm is not where pool memory shows up cleanly; the G-order test is.

## G-cliff — dynamic vs static dose-response sharpness

transition width (log10-dose, 10->90% of own max): dynamic **0.263** vs static **0.471** (smaller = sharper; reported, not presupposed).

| dose | dyn maxH | dyn peak-zone | static maxH |
|---|---|---|---|
| 50.0 | 0.0513 | 9 | 0.0 |
| 100.0 | 0.1172 | 9 | 0.1105 |
| 200.0 | 0.3229 | 9 | 1.2642 |
| 400.0 | 1.7233 | 9 | 6.8857 |
| 800.0 | 20.2813 | 9 | 29.8434 |

## G-NAC — precursor protective lever

| GSH0 scale | maxH |
|---|---|
| 1.0 | 1.7233 |
| 1.5 | 0.9911 |
| 3.0 | 0.4281 |

peak-zone 0-indexed inlet(0)->outlet(9); zone 9 = pericentral / zone 3.
