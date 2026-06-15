# PGx v2.1 Cmax-fold — Step-0 feasibility curation gate

**Verdict: REVISE (gate not met).** Powered pairs confirmed: **1** (metoprolol). Gate
requires ≥5. Per the plan, the milestone halts here for a scope decision — the powered set
is **not** padded with consistency/single-endpoint pairs.

Curation: three independent, model-blind Opus literature curators (PubMed primary sources +
CPIC), each finding folds and fm from **independent** sources (non-circular). No CIs or PMIDs
fabricated; every absent value marked "not found".

## The finding (why the gate fails)

The regime where a genotype **Cmax-fold diverges from the AUC-fold** — high hepatic
first-pass extraction — overlaps almost entirely with the **saturable / nonlinear** regime
the engine's **linear** `well_stirred` clearance model cannot represent. Every high-divergence
substrate found is nonlinear; the linear substrates that would be powered report only
AUC-fold, are paywalled, or are prodrug/low-fm confounded. **v2.1's premise (validate engine
first-pass genotype Cmax on LINEAR drugs) is data-starved precisely because the interesting
drugs need v2.2's Michaelis–Menten flux.**

## Classification (12 + 4 candidates assessed)

### Powered (first-pass, linear, both endpoints, non-circular fm) — N=1
| drug | gene | obs_cmax_fold | obs_auc_fold | em_tmax_h | em_thalf_h | oral_F | fm_invitro | source |
|---|---|---|---|---|---|---|---|---|
| **metoprolol** | CYP2D6 | 2.3× | 4.9× | ~1.5 | ~3.3 | ~0.50 | ~0.75 | Blake 2013 PMID 23665868 (folds); Frontiers PMID 30087611 (fm) |

### Upgradeable (clean AUC-fold + non-circular fm, but per-group Cmax paywalled) — N=2
| drug | gene | obs_auc_fold | em_thalf_h | oral_F | fm_invitro | blocker |
|---|---|---|---|---|---|---|
| nortriptyline | CYP2D6 | 3.32× | ~20 | ~0.61 | ~0.80 | Cmax table Dalén 1998 PMID 9585799 (Wiley, not in PMC) |
| risperidone | CYP2D6 | ~6.2× [5.05–7.62] | ~3–5 | ~0.68 | ~0.75 | Cmax Novalbos 2010 PMID 20814331 (paywall) + active-moiety confound |

### Consistency / falsification anchors (low first-pass, ρ_obs≈small, linear) — reusable
| drug | gene | obs_cmax_fold | obs_auc_fold | fm_invitro | note |
|---|---|---|---|---|---|
| **celecoxib** | CYP2C9 | 1.8× (*3/*3) | 7.7× (*3/*3) | major 2C9 | Prieto-Pérez 2013 PMID 23996211 — textbook low-first-pass divergence (peak flat, exposure explodes); the sharpest "engine must NOT over-predict ρ" test |
| diazepam | CYP2C19 | ≈1.0 | (t½ only) | ~0.33 | Jung 1997 PMID 9029042 — negative control (Cmax unchanged) |
| flurbiprofen | CYP2C9 | flat | ~1.6× | **0.71** | clean numeric in-vitro fm |
| tolbutamide | CYP2C9 | (CL) | ~6.5× | ~0.8 | Kirchheiner 2002 PMID 11875364; AUC inferred from CL |
| citalopram / sertraline | CYP2C19 | not found | ~2.2× / ~1.41× | ~0.15 / ~0.13 | low fm, modest fold |

### Excluded (nonlinear / single-endpoint / prodrug / circular fm)
- **Nonlinear (→ v2.2):** propafenone (Cmax 11.2×/AUC 2.4×, F 0.30→0.81), nebivolol
  (F 0.12→0.96), atomoxetine (Cmax ~3.6×/AUC ~11.4×, Tmax 1.5→4.5 h), omeprazole
  (saturable + MBI auto-inhibition), lansoprazole, moclobemide, venlafaxine. **These have the
  richest both-endpoint genotype data and are the natural v2.2 powered set.**
- **Prodrug / active-moiety:** tramadol, codeine (fm≈0.1), tolterodine, venlafaxine, risperidone.
- **Single-endpoint / no panel:** desipramine, dextromethorphan, mexiletine, donepezil,
  escitalopram, clobazam, glipizide.

## Recommendation

**Reorder: build v2.2 (Michaelis–Menten metabolic `ClearanceFluxSpec`) first, then run the
genotype Cmax-fold validation on the nonlinear powered set** (omeprazole, propafenone,
atomoxetine, lansoprazole, …), which is data-rich. v2.1-as-specced is blocked by a data
scarcity that is itself caused by the missing engine capability. Alternative narrow paths:
(b) a consistency-only v2.1 (celecoxib + diazepam + flurbiprofen) testing the weaker claim
"engine does not over-predict ρ on low-extraction drugs" — but that does not prove
multi-compartment non-redundancy; (c) obtain the two paywalled Cmax tables (Wiley Scholar
Gateway MCP, interactive OAuth) → 3 powered pairs, still below the gate.

Curated data above is preserved for reuse regardless of path.
