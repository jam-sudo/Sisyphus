# Zonal reactive-metabolite hazard probe — Bridge B / B1 Phase-0 (2026-06-18)

**Harness-isolated** (`scripts/probe_zonal_hazard.py`); the reactive metabolite is a POST-PROCESSOR on the axial parent profile, not an engine species. No `predict()` / `reference_man.yaml` / holdout change; headline **2.731 bit-identical**. Reuses the axial machinery (PR #79) + `zonation_weights` (DE-50).

## Conclusion

The per-zone reactive-metabolite hazard (local bioactivation exceeding local saturable detox) is a real surface ORTHOGONAL to bulk PK: varying bioactivation zonation leaves bulk parent extraction ~invariant (DE-50, span 4.3e-04) while the per-zone hazard peak-zone moves. A saturable-detox DOSE-THRESHOLD with pericentral (zone-3) specificity emerges, and raising detox capacity protects — qualitatively reproducing the acetaminophen centrilobular pattern. First concrete Bridge-B endpoint; headline 2.731 untouched (harness-isolated). Post-processor fidelity; transported-metabolite + GSH-pool dynamics + quantitative PoD are B1.x.

## G2 — local matters, bulk doesn't (DE-50 closure)

Bulk parent extraction span across bioactivation zonation: **4.27e-04** (~invariant), while the hazard peak-zone moves:

| bio zonation | bulk E | hazard peak-zone | maxH |
|---|---|---|---|
| pericentral | 0.957487 | 5 | 0.8428 |
| uniform | 0.957914 | 0 | 2.1769 |
| periportal | 0.957631 | 0 | 4.1374 |

## G3 — saturable-detox dose-threshold + zone-specificity (acetaminophen config)

Bioactivation pericentral-high, detox pericentral-low. Below the threshold no zone has hazard; above it the pericentral (zone-3) zone crosses first; 3× detox protects.

| dose | maxH | peak-zone | maxH (3× detox) |
|---|---|---|---|
| 50.0 | 0 | 0 | 0 |
| 100.0 | 0.1105 | 9 | 0 |
| 200.0 | 1.264 | 9 | 0.2104 |
| 400.0 | 6.886 | 9 | 5.081 |
| 800.0 | 29.84 | 9 | 27.17 |

peak-zone is 0-indexed inlet(0)→outlet(9); zone 9 = pericentral / zone 3 / centrilobular. The dose=50 row (maxH 0) is below threshold.
