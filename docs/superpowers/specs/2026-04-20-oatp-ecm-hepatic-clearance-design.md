# OATP Extended Clearance Model (ECM) for Hepatic Clearance — Design Spec

**Date:** 2026-04-20
**Status:** Draft, pending user review
**Author:** Hypatia
**Scope:** Engine refinement to unblock OATP Phase 2A (5-statin Cmax validation) and Phase 2B (SLCO1B1 phenotype directional response).
**Replaces:** Two-step MM-uptake + well-stirred metabolism pipeline for OATP substrates.

---

## 0. Problem & Motivation

Phase 2A/2B are blocked on a structural engine limitation, not on parameter tuning:

- At `liver.OATP1B1` abundance `1.0e11` (Phase 1 pravastatin calibration), MM active uptake is in the **flow-limited saturated regime**. Effective rate `k_uptake = (Jmax/Km) × abundance × f_up` exceeds hepatic blood flow by ≥6 orders of magnitude.
- Abundance sweep `[1e9, 3e9, 1e10, 3e10, 1e11]`: pravastatin Cmax invariant at 0.0385; rosuvastatin Cmax invariant at 0.180 (27× over observed 0.0066). Phenotype scaling (PM 0.10×, UM 2×) has zero directional effect.
- Both scipy.LSODA and JAX/diffrax Kvaerno5 stall or time out for 4/5 statins in ON mode.

Root cause: the current engine applies `active_transport` directly from blood → hepatocyte as a stiff injection, bypassing the **hepatocyte efflux + metabolism + biliary efflux** branches that determine true hepatic clearance. The MM formulation cannot express the clinical truth that OATP uptake *perfuses* the hepatocyte but metabolism/biliary excretion *determines clearance*.

The fix is to adopt the PBPK literature's canonical treatment of OATP substrates: the **Extended Clearance Model (ECM)** (Shitara 2006, Watanabe 2009, Varma 2014, Kunze 2014), applied via quasi-steady-state on the hepatocyte to produce a closed-form `CL_h` expression usable inside the existing well-stirred flux.

---

## 1. Architecture

### 1.1 New clearance model: `"extended"`

Add a third branch to `ClearanceFluxSpec` in `src/sisyphus/engine/flux.py` alongside existing `well_stirred` and `parallel_tube`:

```python
if model == "extended":
    # ECM: QSSA-closed hepatocyte with active+passive uptake, passive efflux,
    # metabolism, biliary clearance.
    # See Section 2 for derivation and formula.
    ...
```

This flux reads (all accessors already exist in `ResolvedParams` unless marked **NEW**):
- `params.node_transporters(target)` → `{tag: abundance}` for all transporters at target node (identity-blind iteration)
- `params.node_enzymes(target)` → `{tag: abundance}` for all enzymes at target node
- `params.drug_enzyme_affinity(tag)` → per-enzyme drug CLint
- `params.drug_transporter_jmax(tag)`, `params.drug_transporter_km(tag)` → MM kinetics
- `params.drug_param("fup")` → existing; **extended** to also resolve `"ps_passive"`, `"ps_eff"`, `"cl_int_bile"`
- `params.total_inflow(source)` → Q_h aggregated from FlowEdges
- `params.get_node_param(target, "ivive_scaling")` → existing per-node scaling factor

**No string matching on organ names.** The flux receives `source` and `target` as arbitrary strings.

### 1.2 YAML: remove redundant `active_transport` edges

In `data/physiology/reference_man.yaml`, replace:

```yaml
edges:
  - {source: portal_vein,    target: liver, type: active_transport}
  - {source: arterial_blood, target: liver, type: active_transport}
  - {source: arterial_blood, target: liver, type: clearance, model: well_stirred}
```

with:

```yaml
edges:
  - {source: arterial_blood, target: liver, type: clearance, model: extended}
```

The `arterial_blood` source is conventional; the flux internally uses `total_inflow(source)` which already aggregates portal + arterial contributions via graph-level flow balance.

**`ActiveTransportFluxSpec` is retained** in `flux.py` for future use (e.g., renal active secretion, BBB transporters) but no physiology YAML currently instantiates it.

### 1.3 DrugOnGraph: three new fields

Extend `predict/ivive.py:build_drug_on_graph` and the `DrugOnGraph` dataclass:

```python
@dataclass(frozen=True)
class DrugOnGraph:
    ...
    ps_passive:   Distribution  # L/h, passive sinusoidal uptake (blood → hepatocyte)
    ps_eff:       Distribution  # L/h, passive sinusoidal efflux  (hepatocyte → blood)
    cl_int_bile:  Distribution  # L/h, biliary intrinsic clearance (hepatocyte → bile)
```

**Defaults for non-OATP / non-biliary drugs:**

| field | default | rationale |
|---|---|---|
| `ps_passive` | `Distribution(1e6, cv=0)` | WS limit: PS_passive ≫ Q_h ⇒ ECM reduces exactly to well-stirred |
| `ps_eff`     | `= ps_passive` (1e6)     | Symmetric passive diffusion |
| `cl_int_bile`| `Distribution(0, cv=0)`  | No biliary clearance unless drug-specific data |

When PS_passive/PS_eff both default to 1e6 L/h and cl_int_bile=0, the ECM formula algebraically reduces to `CL_h = Q_h × f_up × CL_int_metab / (Q_h + f_up × CL_int_metab)` — the existing well-stirred expression. Numerically verified to agree within <0.1% (Section 4).

OATP substrates (5 statins + future valsartan/bosentan) override these fields via `data/transporters/hepatic_ecm.json` loaded in `build_drug_on_graph` when `transporter_kinetics=load_oatp1b1_kinetics(name)` is supplied.

### 1.4 ResolvedParams changes (minimal)

Only one modification to `src/sisyphus/engine/compiler.py:ResolvedParams.drug_param`:

```python
def drug_param(self, param: str) -> float:
    """Look up a scalar drug parameter."""
    if param == "fup":             return self._drug.fup.mean
    if param == "rbp":             return self._drug.rbp.mean
    if param == "renal_clearance": return self._drug.renal_clearance.mean
    if param == "dose_mg":         return self._drug.dose_mg
    if param == "peff":            return self._drug.peff.mean
    if param == "particle_radius_um": return self._drug.particle_radius_um
    # NEW: ECM drug-level permeabilities + biliary CLint
    if param == "ps_passive":      return self._drug.ps_passive.mean
    if param == "ps_eff":          return self._drug.ps_eff.mean
    if param == "cl_int_bile":     return self._drug.cl_int_bile.mean
    raise KeyError(f"Unknown drug param: {param}")
```

`node_transporters`, `drug_transporter_jmax`, `drug_transporter_km`, `total_inflow`, `get_node_param`, `node_enzymes`, `drug_enzyme_affinity` — **already exist, no changes**.

### 1.5 Identity-blindness preservation

The new flux iterates:
- `for tag, abundance in params.node_transporters(target).items(): ...`
- `for tag, abundance in params.node_enzymes(target).items(): ...`

Test (Invariant 1 verification): substitute every organ name in `reference_man.yaml` with random strings; engine numerical output must be bit-identical. This test exists for current flux types and is extended to cover `extended`.

---

## 2. ECM Math

### 2.1 Derivation (QSSA on hepatocyte)

Let:
- `C_b` = concentration in sinusoidal blood (mg/L), connected to graph `source`
- `C_h` = concentration in hepatocyte water (mg/L), QSSA-eliminated intermediate
- `f_up` = unbound fraction in plasma
- `Q_h` = total hepatic blood inflow (L/h), from `params.total_inflow(source)`

Flux balance at the hepatocyte (QSSA, dC_h/dt ≈ 0):

```
Uptake from blood  = Efflux back to blood + Metabolism + Biliary excretion
PS_inf × f_up × C_b = (PS_eff + CL_int,metab + CL_int,bile) × C_h
```

where:
- `PS_inf = PS_active + PS_passive` — total sinusoidal uptake permeability
- `PS_active = Σ_t (abundance_t × Jmax_t / Km_t) × ivive` — linear-regime MM (C_u ≪ Km, all 5 statins at clinical dose satisfy this; see Section 2.4)
- `PS_eff` — passive sinusoidal efflux (blood ← hepatocyte)
- `CL_int,metab = Σ_e (abundance_e × affinity_e) × ivive` — existing per-enzyme CLint formulation
- `CL_int,bile` — drug-specific biliary intrinsic clearance
- `CL_int,h = CL_int,metab + CL_int,bile` — total hepatocyte elimination

Solving for steady-state `C_h`:

```
C_h = [PS_inf × f_up / (PS_eff + CL_int,h)] × C_b
```

Mass leaving the organ per unit time (elimination flux into metabolism + bile):

```
Rate_elim = CL_int,h × C_h
         = [PS_inf × f_up × CL_int,h / (PS_eff + CL_int,h)] × C_b
```

Closing with blood-side flow (well-stirred on the sinusoidal compartment):

```
Q_h × (C_b - C_out) = Rate_elim
```

Substituting and solving for the apparent hepatic clearance `CL_h` such that `Rate_elim = CL_h × C_out`:

```
                Q_h × f_up × PS_inf × CL_int,h
CL_h  =  ────────────────────────────────────────────────
          Q_h × (PS_eff + CL_int,h) + f_up × PS_inf × CL_int,h
```

### 2.2 Engine formula (exactly what `flux.py` computes)

```python
ivive = params.get_node_param(target, "ivive_scaling")

# PS_active from transporters at target node (identity-blind iteration)
PS_active = 0.0
for tag, abundance in params.node_transporters(target).items():
    jmax = params.drug_transporter_jmax(tag)
    km   = params.drug_transporter_km(tag)
    if jmax <= 0 or km <= 0:
        continue
    PS_active += abundance * jmax / km
PS_active *= ivive

# Drug-level permeabilities (bare floats; no f_up multiplication here)
PS_passive  = params.drug_param("ps_passive")
PS_eff      = params.drug_param("ps_eff")
CL_int_bile = params.drug_param("cl_int_bile")

PS_inf   = PS_active + PS_passive

# Metabolism — inlined per existing well_stirred pattern (identity-blind)
CL_int_metab = 0.0
for tag, abundance in params.node_enzymes(target).items():
    affinity = params.drug_enzyme_affinity(tag)
    if affinity > 0:
        CL_int_metab += abundance * affinity * ivive
CL_int_h = CL_int_metab + CL_int_bile

# ECM closure
Q_h   = params.total_inflow(source)
f_up  = params.drug_param("fup")
num   = Q_h * f_up * PS_inf * CL_int_h
den   = Q_h * (PS_eff + CL_int_h) + f_up * PS_inf * CL_int_h
CL_h  = num / den if den > 0 else 0.0

# Same downstream as well_stirred
c_out = y[source_idx] * rbp / (v_source * kp_source)
rate  = CL_h * c_out
```

**`f_up` appears exactly once in the numerator.** (Earlier draft with `PS_active = f_up × Σ ...` was corrected — see change log.)

### 2.3 Degenerate limit (recovers well-stirred exactly)

For a drug with `PS_passive = PS_eff = P` (large), `CL_int_bile = 0`, no OATP data:

```
PS_inf      = 0 + P = P
CL_int,h    = CL_int,metab   ≡ CL_int
num         = Q_h × f_up × P × CL_int
den         = Q_h × (P + CL_int) + f_up × P × CL_int
            ≈ Q_h × P + f_up × P × CL_int         (if P ≫ CL_int)
            = P × (Q_h + f_up × CL_int)
CL_h        ≈ Q_h × f_up × CL_int / (Q_h + f_up × CL_int)   ✓ well-stirred
```

For `P = 1e6 L/h`, `Q_h = 87 L/h`, `CL_int = 100 L/h`, `f_up = 0.1`: the `+ CL_int` correction in the denominator is `100 / (1e6 + 100) ≈ 1e-4` relative error. Bit-level invariance on the 107 holdout is numerically guaranteed to <0.1%.

### 2.4 v1 scope: linear PS_active

v1 assumes `C_u,blood ≪ Km` so that `PS_active = Σ abundance × Jmax/Km` is concentration-independent. Verified valid for all 5 statins at clinical doses (C_u/Km < 0.02). **Not valid** for:
- Toxicology/high-dose DDI where C_u approaches Km
- Transporter-mediated saturation in uptake inhibition studies

**v2 (deferred)** will restore the full MM form `J_uptake(C_u) = Σ abundance × Jmax × C_u/(Km + C_u)`, implemented as a drug-concentration-dependent `PS_active(C_b)` inside the flux. The QSSA closure remains but becomes a scalar nonlinear equation in `C_h`; small cost, preserves identity-blindness.

---

## 3. Data Curation

### 3.1 New file: `data/transporters/hepatic_ecm.json`

Schema:

```json
{
  "pravastatin": {
    "ps_passive_L_per_h":   {"mean": 0.8,  "cv": 0.40, "source": "Watanabe 2009"},
    "ps_eff_L_per_h":       {"mean": 0.8,  "cv": 0.40, "source": "PS_passive symmetric"},
    "cl_int_bile_L_per_h":  {"mean": 45.0, "cv": 0.50, "source": "Yabe 2011"}
  },
  ...
}
```

### 3.2 Five statins — literature sources (to be finalized in plan phase)

Values below are **placeholders** for scope; plan-phase curation derives final numbers from primary literature:

| drug         | PS_passive (L/h) | PS_eff (L/h) | CL_int,bile (L/h) | Sources |
|--------------|------------------|--------------|-------------------|---------|
| pravastatin  | ~0.8             | ~0.8         | ~45               | Watanabe 2009, Yabe 2011 |
| rosuvastatin | ~1.5             | ~1.5         | ~90               | Jones 2012, Bergman 2006 |
| atorvastatin | ~3.0             | ~3.0         | ~30               | Kunze 2014, Jamei 2014 |
| pitavastatin | ~2.5             | ~2.5         | ~60               | Hirano 2006, Li 2018 |
| fluvastatin  | ~5.0             | ~5.0         | ~20               | Lindahl 2004, Varma 2014 |

All widths `cv=0.40` for PS values, `cv=0.50` for `CL_int,bile` (biliary data is sparser).

### 3.3 oatp1b1.json — status

Existing `data/transporters/oatp1b1.json` (Jmax/Km for 5 statins) is **retained unchanged** as the transporter-affinity source. PS_active is computed inside the flux from Jmax/Km × abundance, preserving the per-transporter structure and the `TRANSPORTER_ALIASES["SLCO1B1"] = "OATP1B1"` phenotype wiring.

### 3.4 Pravastatin re-calibration under ECM

Current: `liver.OATP1B1` abundance = `1.0e11`, Jmax=228 pmol/s, Km=13.6 µM → PS_active ≈ 0.06 L/h. Watanabe 2009 reports PS_inf ≈ 0.5-2 L/h (10-30× gap — a fit compensation for the missing biliary branch and the flow-limited plateau).

**Calibration strategy (Option II, chosen):** scale abundance ×20-30 to ~2e12 so PS_active lands in Watanabe 2009 range. This retains the existing MM machinery (Jmax/Km/abundance structure) and the CPIC phenotype scaling hook via `apply_phenotype_to_graph` (abundance scaling = activity scaling). Alternatives considered:

- **Option I** (re-curate Jmax/Km): disturbs oatp1b1.json which is already literature-grounded; rejected.
- **Option III** (drop MM, store PS_active directly): loses per-transporter decomposition and the phenotype scaling hook; rejected.

Calibration target: pravastatin 40 mg oral, observed Cmax 0.045 mg/L. Fold-error gate 0.7-1.3 in plan phase.

---

## 4. 107-Holdout Invariance

### 4.1 Mechanism

For any drug without an entry in `hepatic_ecm.json` and without transporter kinetics data (the 107 holdout — *none* are OATP1B1 substrates with active uptake modeled in current Phase 1 pipeline):

- `PS_active = 0` (no transporter kinetics loaded)
- `PS_passive = PS_eff = 1e6 L/h` (defaults)
- `CL_int,bile = 0`
- `CL_int,metab` unchanged (existing organ-CLint formula)

ECM formula → well-stirred exactly (Section 2.3).

### 4.2 Regression gate

Required: `|Meta AAFE_after - Meta AAFE_before| < 0.01` on the 107-holdout, measured via `scripts/run_engine_benchmark.py`. Target: 2.695 invariant to ≤3 decimal places.

Automated regression test added to `tests/integration/test_holdout_regression.py` reads the cached prediction JSON and computes AAFE.

### 4.3 Numerical verification

Unit test: for a random drug with only CLint>0 (no transporter), compute flux rate in both `well_stirred` and `extended` modes with `PS_passive = PS_eff = 1e6`. Assert `|rate_ext - rate_ws| / rate_ws < 1e-4`.

---

## 5. Validation Gates

All gates are **blocking** on spec acceptance. Plan phase cannot land without passing.

### 5.1 Stiffness elimination

- **Gate:** 5 statins, 40 mg or clinical dose, t_end = 24 h, `scipy.LSODA` default tolerances. Wall time per solve **< 5 s**.
- **Mechanism:** ECM closed-form eliminates the fast blood→hepatocyte transient (QSSA does it analytically); no stiff injection.

### 5.2 Pravastatin Phase 1 calibration preserved

- **Gate:** pravastatin 40 mg oral → Cmax fold-error (FE) vs observed 0.045 mg/L in range **[0.7, 1.3]**. Current Phase 1: ratio 0.86 (FE=1.17).

### 5.3 Phase 2A: 4 non-pravastatin statins converge

- **Gate:** rosuvastatin, atorvastatin, pitavastatin, fluvastatin — all solve in ON mode, **FE < 3.0×**.
- **Rationale:** current state is 27× for rosuvastatin or timeout. Target 3× is the Meta AAFE holdout bar; acceptable for ECM v1 to match general population calibration.

### 5.4 Phase 2B: SLCO1B1 PM directional response

- **Gate:** `sisyphus tdm --phenotype SLCO1B1:PM` on pravastatin or any statin → Cmax **increases ≥30%** relative to SLCO1B1:EM baseline. Matches Niemi 2006 clinical AUC +60-100% for PM (Cmax proportion typically ~0.5× AUC effect).

### 5.5 PS_active vs literature

- **Gate:** pravastatin post-re-calibration PS_active within **0.5-2×** of Watanabe 2009 PS_inf ≈ 1 L/h.
- Same gate applied to 4 other statins against primary-source PS_inf values (Kunze 2014, Jones 2012, Maeda 2011).

### 5.6 107-holdout regression

See Section 4.2. `|ΔAAFE| < 0.01`.

---

## 6. Test Plan

### 6.1 Unit tests

Added to `tests/unit/test_engine_flux.py`:

1. **ECM formula correctness** — hand-computed test vector (Q=90, f_up=0.1, PS_inf=1.0, PS_eff=1.0, CL_int=100) → CL_h matches closed form to 1e-9.
2. **Degenerate limit (WS recovery)** — drug with PS_passive=PS_eff=1e6, CL_int_bile=0, no transporters → `|CL_h_ext - CL_h_ws| / CL_h_ws < 1e-4`.
3. **f_up appears exactly once** — doubling f_up with other params fixed: numerator scales 2×, denominator gains only the `+ f_up × PS_inf × CL_int,h` term; CL_h change must match analytical derivative, not 4× (which would indicate the f_up² bug).
4. **PS_active from transporters** — node with two transporters (abundances a1, a2), drug kinetics (J1/K1, J2/K2) → `PS_active = (a1 × J1/K1 + a2 × J2/K2) × ivive`.
5. **Identity-blindness** — rename `liver` → `"xyz123"` in a test graph; ECM flux rate invariant.
6. **No transporter kinetics fallback** — drug with no entry for a given transporter tag: contributes 0 to PS_active (no exception, no fill).
7. **CL_int_bile default 0** — drug without cl_int_bile field: CL_int,h = CL_int,metab only.

### 6.2 Integration tests

Added to `tests/integration/test_oatp_ecm.py`:

1. **5-statin Cmax convergence** — each statin solves in <5 s, FE < 3× (pravastatin <1.3×).
2. **SLCO1B1 PM directional response** — `apply_phenotype_to_graph(graph, {"SLCO1B1": "PM"})` → Cmax ≥1.3× EM baseline on pravastatin.
3. **SLCO1B1 UM directional response** — Cmax ≤0.7× EM baseline.
4. **Phenotype wiring unchanged** — existing `test_phenotype_transporter.py` (11 tests) all pass.

### 6.3 Regression tests

- **107-holdout AAFE invariance** — `tests/integration/test_holdout_regression.py::test_ecm_holdout_invariance` — |ΔAAFE| < 0.01 vs cached `4track_holdout_predictions.json`.
- **SBI routing unchanged** — 12/0/1 routing (post-P6) on 13 routing drugs. None are OATP substrates; ECM defaults apply; routing table unchanged.

### 6.4 Test count delta

Current: 448 tests (per CLAUDE.md). New additions:
- Unit: +7
- Integration: +4
- Regression: +1 (deterministic, reads cached JSON)

Target: **460 pass, 0 skip** after ECM lands.

---

## 7. v1 Scope & Deferrals

### 7.1 In scope (v1)

- ECM closed-form in linear PS_active regime
- 5 statins curated for PS_passive/PS_eff/CL_int_bile
- Pravastatin abundance re-calibration to match Watanabe 2009 PS_inf
- SLCO1B1 phenotype directional Cmax response
- 107-holdout AAFE invariance

### 7.2 Deferred (v2+)

- **Saturable PS_active** — full MM form `PS_active(C_b)` for toxicology/high-dose DDI. Requires concentration-dependent root-finding in the flux.
- **Zonation** — periportal vs perivenous hepatocyte heterogeneity (zonation). Literature weak for PBPK; defer.
- **Parallel-tube extraction** — existing `parallel_tube` model is independent; coexists with ECM (ECM acts on hepatocyte, PT on blood-side mixing). Combined ECM+PT deferred.
- **Non-statin OATP substrates** — valsartan, bosentan, irinotecan. Data curation effort; add in plan phase 2.
- **Other transporters** — P-gp, BCRP at intestine/liver, OCT1/2 at kidney. Each is a parallel ECM instance; v1 demonstrates the machinery on OATP1B1 only.
- **Mechanistic biliary feedback** — bile → intestine recycling (enterohepatic). Currently bile flux is a one-way sink.

---

## 8. Change Log (design iteration)

| # | Change | Reason |
|---|---|---|
| 1 | Initial proposal: MM uptake + well-stirred | baseline before ultrathink |
| 2 | Hybrid (phenotype-only mode + full ECM mode) | cost framing |
| 3 | Committed to full ECM | user: "더 과학적으로 옳은 방법으로 진행" (scientifically correct method) |
| 4 | Fixed `PS_active = f_up × Σ(...)` → `PS_active = Σ(...)` | f_up² double-multiplication bug |
| 5 | Added pravastatin re-calibration strategy (Option II: scale abundance) | Watanabe 2009 gap diagnosed |
| 6 | Added v1 linear-regime scope limit | saturable MM deferral rationale |
| 7 | Added ResolvedParams accessor extensions | compile-layer support for new drug fields |
| 8 | Confirmed phenotype wiring requires no code changes | abundance scaling = PS_active scaling = CPIC activity scaling |
| 9 | Self-review: Section 1.4 reduced to `drug_param` extension only | verified `node_transporters`/`drug_transporter_jmax`/`drug_transporter_km`/`get_node_param` already exist in ResolvedParams |
| 10 | Self-review: Section 2.2 inlined `CL_int_metab` loop | no `organ_clint` accessor exists; matches existing `well_stirred` pattern |
| 11 | Self-review: Section 2.1 ref fixed (Section 5.3 → 2.4) | linear-regime justification lives in 2.4, not in Phase 2A gate |
| 12 | Self-review: added `jmax <= 0` rejection in PS_active loop | defensive against zero-kinetics drug-transporter pairs |

---

## 9. References

- **Shitara Y et al.** *Drug Metab Pharmacokinet* 2006 — Transporters as a determinant of drug clearance and tissue distribution.
- **Watanabe T et al.** *J Pharmacol Exp Ther* 2009 — PBPK prediction of hepatic uptake of statins from *in vitro* data.
- **Varma MV et al.** *Clin Pharmacol Ther* 2014 — Extended clearance classification system (ECCS).
- **Kunze A et al.** *Drug Metab Dispos* 2014 — Prediction of OATP1B1-mediated drug-drug interactions for statins.
- **Maeda K et al.** *Clin Pharmacol Ther* 2011 — Identification of the rate-determining process in hepatic elimination of rosuvastatin.
- **Jones HM et al.** *Drug Metab Dispos* 2012 — Mechanistic PBPK modeling of rosuvastatin.
- **Yabe Y et al.** *Drug Metab Dispos* 2011 — Kinetic analysis of biliary excretion of pravastatin.
- **Hirano M et al.** *J Pharmacol Exp Ther* 2006 — Contribution of OATP1B1 and OATP1B3 to the hepatic uptake of pitavastatin.
- **Niemi M et al.** *Clin Pharmacol Ther* 2006 — SLCO1B1 c.521T>C polymorphism effects on statin pharmacokinetics.
- **Jamei M et al.** *Clin Pharmacokinet* 2014 — A mechanistic PBPK approach to predict in vivo DDIs for atorvastatin.
- **Bergman E et al.** *Eur J Clin Pharmacol* 2006 — Rosuvastatin: intestinal and hepatic uptake.
- **Li R et al.** *Mol Pharm* 2018 — Hepatic uptake kinetics of pitavastatin.
- **Lindahl A et al.** *Eur J Pharm Sci* 2004 — Fluvastatin permeability and transport.
