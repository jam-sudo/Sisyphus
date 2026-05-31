# Sisyphus

**Graph-based whole-body PBPK simulation with native uncertainty propagation**

[![CI](https://github.com/jam-sudo/Sisyphus/actions/workflows/ci.yml/badge.svg)](https://github.com/jam-sudo/Sisyphus/actions/workflows/ci.yml)

[Methodology](#methodology) &middot; [Quickstart](#quickstart) &middot; [Validation](#validation) &middot; [Architecture](#architecture) &middot; [Limitations](#limitations)

---

Sisyphus is a physiologically based pharmacokinetic (PBPK) platform that represents the human body as a typed directed multi-graph, derives ordinary differential equation (ODE) systems from graph topology, and propagates parameter uncertainty natively through Monte Carlo sampling.

The platform accepts a SMILES string and dosing regimen as input and produces PK endpoints (C<sub>max</sub>, T<sub>max</sub>, AUC, t<sub>1/2</sub>) with 90% prediction intervals. Beyond single-dose prediction, Sisyphus supports multi-dose regimen simulation with steady-state detection, Bayesian therapeutic drug monitoring (TDM) via dispatched simulation-based / importance / iterative Bayesian methods, model-informed precision dosing (MIPD), drug-drug interaction (DDI) modeling, pharmacogenomic phenotype-aware prediction (SLCO1B1, NAT2, UGT1A1), and PK/PD effect estimation. OATP1B1-mediated hepatic uptake is modeled via an extended clearance model (ECM) with closed-form QSSA hepatocyte kinetics.

```
$ sisyphus predict --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O" --dose 100

Drug: Cn1c(=O)c2c(ncn2C)n(C)c1=O
Method: hybrid
Confidence: high
Cmax: 1.1792 mg/L
Tmax: 0.63 h
t½: 2.07 h
```

## Methodology

### Physiological model

The body is represented as a 34-compartment directed multi-graph comprising blood pools (arterial, venous, portal vein), perfusion-limited organs (11), permeability-limited organs (4, each split into vascular and extravascular sub-compartments), GI lumen segments (8, compartmental absorption and transit model; Yu &amp; Amidon, 1999), and mass-balance sinks (4). Physiological parameters follow the ICRP Reference Man (ICRP, 2002). Tissue compositions for partition coefficient estimation are taken from Rodgers &amp; Rowland (2005). CYP enzyme abundances follow Shimada et al. (1994).

```
                      ┌─────────────────────────────────────────────┐
                      │                                             │
   ┌──────┐    ┌──────┴───┐                                   ┌─────┴────┐
   │ lung │───►│ arterial │─► brain ─────────────────────────►│  venous  │
   └──┬───┘    │  blood   │─► heart ─────────────────────────►│  blood   │
      │        │          │─► kidney ────────────────────────►│          │
      │        │          │                                   │          │
      │        │          │─► gut wall ──┐                    │          │
      │        │          │─► spleen  ───┼─► liver ──────────►│          │
      │        │          │─► pancreas ──┘   (portal,CYP450)  │          │
      │        │          │                                   │          │
      │        │          │─► muscle, adipose, skin, bone ───►│          │
      │        └──────────┘                                   └─────┬────┘
      │                                                             │
      └─────────────────────────────────────────────────────────────┘

   stomach ──► duodenum ──► jejunum ──► ileum ──► colon ──► fecal excretion
                  │            │          │
                  └────────────┴──────────┘
                       absorption ──► gut wall
```

### ODE formulation

The ODE system is derived automatically from graph topology. Each edge type dispatches a flux function:

**Perfusion-limited transport** (FlowFluxSpec):

$$\frac{dA_i}{dt} = Q_i \cdot C_{in} - Q_i \cdot \frac{A_i \cdot R_{B:P}}{V_i \cdot K_{p,i}}$$

**Hepatic clearance — Extended Clearance Model (ECM, default)** (ClearanceFluxSpec, `model="extended"`):

The QSSA-derived effective intrinsic clearance, with $PS_{inf,total} = PS_{inf} + J_{max} \cdot f_u / (K_m + f_u \cdot C_{u,hep})$:

$$CL_{int,eff} = \frac{PS_{inf,total} \cdot CL_{int,h}}{PS_{eff} + CL_{int,h}}$$

is embedded in the well-stirred form:

$$CL_{h} = \frac{Q \cdot f_u \cdot CL_{int,eff}}{Q + f_u \cdot CL_{int,eff}}$$

where $PS_{inf}$ is passive sinusoidal influx, $PS_{eff}$ is sinusoidal efflux, $J_{max}/K_m$ are active uptake (OATP1B1, …) Michaelis–Menten parameters, and $CL_{int,h}$ is the sum of metabolic + biliary intrinsic clearance. Derivation: hepatocyte mass balance $PS_{inf,total} \cdot C_{u,blood} = (PS_{eff} + CL_{int,h}) \cdot C_{u,hep}$ at steady state (QSSA). For drugs without active transporter kinetics, the model reduces to the classical well-stirred form:

$$CL_{int,organ} = \sum_j \left( E_j \cdot k_j \right) \cdot S_{IVIVE}$$

$$CL_{organ} = \frac{Q \cdot f_u \cdot CL_{int,organ}}{Q + f_u \cdot CL_{int,organ}}$$

where $E_j$ is the enzyme abundance (total pmol in organ), $k_j$ is the per-enzyme intrinsic clearance (&mu;L/min/pmol), and $S_{IVIVE}$ converts units (60/10<sup>6</sup>, &mu;L/min &rarr; L/h). This formulation is **enzyme-level**: clearance at any organ is computed from its local enzyme profile, not from organ identity (Houston, 1994; Yang et al., 2007; Shitara et al., 2013 / Yoshikado et al., 2017 for the ECM/QSSA hepatocyte form implemented here).

**Permeability-limited distribution** (DiffusionFluxSpec):

$$\frac{dA_{vasc}}{dt} = Q \cdot C_{art} - Q \cdot C_{vasc} - PS \cdot (C_{u,vasc} - C_{u,tissue})$$

$$\frac{dA_{tissue}}{dt} = PS \cdot (C_{u,vasc} - C_{u,tissue})$$

**GI absorption** (AbsorptionFluxSpec):

$$k_a = \frac{2.88 \cdot P_{eff} \cdot f_{ka}}{r}$$

where $P_{eff}$ is effective permeability (&times;10<sup>&minus;4</sup> cm/s), $f_{ka}$ is the segment-specific absorption fraction, and $r$ is particle radius (&mu;m).

**Tissue:plasma partition coefficients** are computed via the Rodgers &amp; Rowland method (Rodgers &amp; Rowland, 2005, 2006), with the Berezhkovskiy correction for acids (Berezhkovskiy, 2004).

### Prediction pipeline

The full pipeline combines mechanistic simulation with data-driven prediction:

1. **SMILES &rarr; molecular profile**: RDKit descriptors, structural pK<sub>a</sub> classification, applicability domain assessment
2. **ADME prediction**: Pre-trained XGBoost models for f<sub>u,p</sub>, CL<sub>int</sub>, R<sub>B:P</sub>, VD<sub>ss</sub> (trained on TDC datasets; Huang et al., 2021), with DrugBank experimental f<sub>u,p</sub> enrichment where available
3. **IVIVE**: CL<sub>int</sub> decomposition into per-enzyme affinities, Kp calculation
4. **PBPK simulation**: 34-state ODE system solved via LSODA (Petzold, 1983)
5. **ML direct prediction**: XGBoost C<sub>max</sub> model (trained on 1,128 drugs from multi-source clinical PK data)
6. **CL/F analytical track**: closed-form 1-compartment C<sub>max</sub> estimate using XGBoost CL/F + V<sub>d</sub> predictions and k<sub>a</sub> from Engine T<sub>max</sub> / Peff. Decorrelates with Engine+ML residuals via different input channels.
7. **VDss analytical track**: 1-compartment C<sub>max</sub> using XGBoost VDss (volume-of-distribution-at-steady-state) predictor. Conditional activation based on applicability. Added 2026-04 as the orthogonal fourth track.
8. **Meta-learner**: Compound-type-adaptive geometric blend of all four tracks via LOOCV-calibrated weights. Base compounds: engine 0.60 / ML 0.40 / CLF 0.00; non-base: engine 0.35 / ML 0.50 / CLF 0.15. VDss track weight 0.20 when activated; other weights scaled by ×0.80 so the four-track sum remains unity.

### Uncertainty propagation

All parameters are represented as `Distribution(mean, cv, dist_type)`. Monte Carlo sampling draws N realizations from all parameter distributions simultaneously, solves the ODE for each, and aggregates the resulting PK endpoints into distributional summaries with prediction intervals. The graph topology is compiled once; only parameter values change across MC iterations ("compile once, parameterize many").

### Multi-dose regimen simulation

Multi-dose pharmacokinetics are computed by an event-driven solver that wraps the single-dose ODE engine. Dose events are injected into the state vector between integration segments; the ODE right-hand side is not modified. This preserves the identity-blind engine invariant while supporting arbitrary dosing schedules (repeated oral, IV infusion, mixed regimens).

Steady-state detection applies a trough variation criterion (&lt;5%) across the last three dosing intervals. The solver reports C<sub>ss,max</sub>, C<sub>ss,min</sub>, accumulation ratio (AR = C<sub>ss,max</sub> / C<sub>max,first</sub>), and dose number at which steady state is reached.

### Therapeutic drug monitoring

Given observed plasma concentrations, Sisyphus refines population-level parameter distributions into individual posteriors using a **dispatched Bayesian router** (`data/sbi/method_routing.json`) that selects one of three methods per drug:

- **SBI (Simulation-Based Inference, default, 12/13 production drugs).** Amortized neural posterior estimation (Normalizing Spline Flow) conditioned on (simulator output, patient observation); the per-drug posterior is pre-trained offline. Inference is a single forward pass (milliseconds). Used when the Simulation-Based Calibration (SBC) gate passes on the offline validation profile.
- **IBIS (Iterative Bayesian Importance Sampling, 1/13 production drugs).** Used as fallback when SBC fails for a drug (e.g. pravastatin OATP1B1 issues pre-ECM); closed-loop iteration prevents weight degeneracy.
- **IS (classical Importance Sampling, 0/13 production drugs post-routing).** Retained for legacy compatibility; used only for compounds where SBI training data is insufficient AND IBIS has not been validated.

Effective sample size (ESS = $1/\sum w_i^2$) is monitored for importance-based methods. Morphine routes SBI with a likelihood reweighting kernel (P6, 2026-04-19) to compensate for hierarchical variance deficiency. All three methods share a unified output contract: per-PK-parameter posterior Distribution. This mechanism bypasses the CL<sub>int</sub> prediction ceiling (R&sup2; &asymp; 0.24): observed drug levels directly correct inaccurate population priors, reducing posterior CV by &gt;50% in validation (see [TDM validation](#tdm-validation)).

### Model-informed precision dosing

MIPD recommends adjusted doses to achieve a target steady-state concentration:

1. Bayesian update at the observed dose yields a posterior C<sub>ss</sub> distribution.
2. Linear dose scaling: $dose_{new} = dose_{current} \times (C_{ss,target} / C_{ss,posterior})$
3. Clamp to clinical dose range and round to a practical increment (default 25 mg).

Linear scaling assumes non-saturable metabolism, which holds for most drugs at therapeutic concentrations. For drugs with known nonlinear pharmacokinetics (e.g., phenytoin), this approximation should be used with caution.

### Drug-drug interactions

DDI is modeled by adjusting enzyme abundances in the body graph prior to ODE compilation. The engine sees modified abundances and computes clearance as usual &mdash; no engine modifications are required.

**Competitive inhibition:**

$$E_{eff} = \frac{E_{base}}{1 + [I] / K_i}$$

**Enzyme induction (E<sub>max</sub> model):**

$$E_{eff} = E_{base} \cdot \left(1 + \frac{E_{max} \cdot [I]}{EC_{50} + [I]}\right)$$

where $E_{base}$ is baseline enzyme abundance (pmol), $[I]$ is perpetrator plasma concentration, $K_i$ is the inhibition constant, and $E_{max}$/$EC_{50}$ are induction parameters. Preset perpetrators: ketoconazole, fluconazole, quinidine (CYP inhibitors) and rifampin (CYP3A4 inducer).

### PK/PD link

Pharmacodynamic effects are computed from the concentration-time profile via an effect compartment with sigmoid E<sub>max</sub> response:

**Effect-site equilibration:**

$$\frac{dC_e}{dt} = k_{e0} \cdot (C_p - C_e)$$

**Sigmoid E<sub>max</sub> response:**

$$E = E_0 + \frac{E_{max} \cdot C_e^n}{EC_{50}^n + C_e^n}$$

where $k_{e0}$ is the effect-site equilibration rate constant (h<sup>&minus;1</sup>), $E_0$ is the baseline effect, and $n$ is the Hill coefficient. This is implemented as analytical post-processing on the PK solution, not as additional ODE states, preserving engine isolation. Preset PD models are provided for midazolam (sedation) and warfarin (INR response).

## Quickstart

### Installation

```bash
pip install -e ".[dev,ml,chem]"
```

> Pre-trained XGBoost models (f<sub>u,p</sub>, CL<sub>int</sub>, R<sub>B:P</sub>, VD<sub>ss</sub>, C<sub>max</sub>) are required in `models/adme/` and `models/direct_pk/`. Re-training scripts are provided in `scripts/`.

### CLI

Six commands cover the clinical pharmacology workflow:

```bash
# Single-dose PK prediction
sisyphus predict --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O" --dose 100

# Multi-dose regimen simulation (atorvastatin 40 mg QD × 14 days)
sisyphus simulate --smiles "CC(C)c1n(CC(O)CC(O)CC(=O)O)c(=O)..." \
    --dose 40 --interval 24 --doses 14

# TDM Bayesian update (midazolam 5 mg, observed 0.015 mg/L at t=1 h)
sisyphus tdm --smiles "c1ccc2c(c1)C(=NC(=O)N2)c1ccccc1F" \
    --dose 5 --obs "1.0:0.015"

# DDI prediction (midazolam + ketoconazole inhibition)
sisyphus ddi --smiles "c1ccc2c(c1)C(=NC(=O)N2)c1ccccc1F" \
    --dose 5 --inhibitor ketoconazole

# MIPD dose recommendation (target Css,max = 0.02 mg/L)
sisyphus dose-adjust --smiles "c1ccc2c(c1)C(=NC(=O)N2)c1ccccc1F" \
    --dose 5 --obs "1.0:0.015" --target-css 0.02

# Holdout benchmark (add --compute-pi for empirical 90% PI coverage; diagnostic only)
sisyphus benchmark --holdout
```

All commands accept `--verbose` for debug-level logging.

### Python API

```python
from sisyphus.pipeline.predict import predict

result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)

print(result.pk.cmax.mean)    # 1.18 mg/L
print(result.method)          # "hybrid"
print(result.confidence)      # "high"
```

### Engine-only mode (known compound parameters)

For validation or mechanistic studies, the engine can be driven directly from compound YAML files, bypassing ADME prediction:

```python
from pathlib import Path
import numpy as np
from sisyphus.graph.builder import build_from_yaml
from sisyphus.compounds import load_compound
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.pk.endpoints import compute_endpoints
import sisyphus.engine.flux  # register flux functions

graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
drug = load_compound(Path("data/compounds/midazolam.yaml"))

compiled = ODECompiler().compile(graph)
rng = np.random.default_rng(42)
params = ResolvedParams(graph.sample(rng), drug.sample(rng))

y0 = np.zeros(compiled.n_states)
y0[compiled.state_index[drug.administration_node]] = drug.dose_mg

result = solve(compiled, params, y0, t_span=(0, 24))
pk = compute_endpoints(result)

print(f"Cmax: {pk.cmax.mean:.4f} mg/L")  # 0.0069 mg/L (matches Omega ±0.5%)
```

### Monte Carlo uncertainty

```python
from sisyphus.engine.uncertainty import UncertaintyEngine

ue = UncertaintyEngine()
mc = ue.propagate_fast(compiled, graph, drug, n_samples=1000)

print(mc.pk.cmax)                # Distribution(mean≈1.18, cv≈0.3)
print(mc.cmax_90ci)              # (0.57, 1.59) mg/L
print(len(mc.cmax_samples))      # 1000 individual realizations
```

## Validation

### Engine validation against Omega PBPK

Four drugs with known compound parameters were simulated and compared against the [Omega PBPK](https://github.com/jam-sudo/Omega) ODE engine (35-state hardcoded model):

| Drug | Dose | Sisyphus C<sub>max</sub> (mg/L) | Omega C<sub>max</sub> (mg/L) | Relative error |
|------|:----:|:------------:|:----------:|:-----:|
| Midazolam | 2 mg PO | 0.006911 | 0.006943 | 0.5% |
| Caffeine | 100 mg PO | 1.7151 | 1.7139 | 0.1% |
| Warfarin | 10 mg PO | 0.4917 | 0.4922 | 0.1% |
| Propranolol | 80 mg PO | 0.1353 | 0.1355 | 0.1% |

Mass balance error &lt; 10<sup>&minus;12</sup> for all simulations.

### Holdout benchmark (SMILES &rarr; C<sub>max</sub>)

External validation on a Murcko scaffold-stratified holdout set (N=107, seed=42, never used in training or model selection). The holdout set integrates observed concentration&ndash;time profiles from the Open Systems Pharmacology (OSP) repository, curated literature PK data, and FDA DailyMed labels. Performance is reported using AAFE (Absolute Average Fold Error; Obach et al., 1997) with bootstrap 95% confidence intervals (10,000 resamples on |log<sub>10</sub>(fold error)|):

$$AAFE = 10^{\operatorname{mean}\left(\left|\log_{10}\frac{C_{max,pred}}{C_{max,obs}}\right|\right)}$$

| Track | AAFE | 95% CI | %2-fold | %3-fold | N |
|---|:-:|:-:|:-:|:-:|:-:|
| **Meta-learner (production)** | **2.698** | [2.32, 3.17] | 46.7% | 61.7% | 107 |
| Engine only | 3.831 | [3.18, 4.65] | 30.8% | 43.9% | 107 |
| ML only | 3.010 | [2.56, 3.56] | 43.0% | 57.9% | 107 |
| Meta, in-domain | 2.760 | [2.30, 3.34] | 45.6% | 59.5% | 79 |

> **Reproducibility note (2026-05-09 audit-driven update; B-03 refresh 2026-05-20; B-03.x literature-IVIVE 2026-05-25; B-02 Phase 2 UGT registry 2026-05-27).** These numbers reflect a **public-clone deterministic state**: a fresh `git clone` followed by `pip install -r requirements-lock.txt` reproduces the headline AAFE to 3 sig figs. Per-drug Cmax bit-identity additionally requires matching the exact numerics stack (Python minor version, BLAS implementation, libomp build) used to generate the canonical `data/training/4track_holdout_predictions.json`; on differing environments per-drug Cmax can drift up to ~12% on individual drugs (largest observed: trazodone engine 0.291 → 0.256) without materially shifting aggregate AAFE. The previous cache (Meta 2.679 [2.30, 3.14], In-domain 2.733) was generated on a local-developer environment that conditionally loaded two artifacts not present in this repository: a proprietary DrugBank export (`data/drugbank/`, academic license required, gitignored) and a residual-correction logP XGBoost model (`models/adme/logp_correction.json`, gitignored). Both shifted Cmax predictions silently for the drugs they covered. Removing the silent shift moves Meta AAFE from local-artifact values near 2.68 to the public-clone 2.75–2.77 range. The 2026-05-20 B-03 refresh adds clopidogrel to the prodrug registry (parent-observation scoring + per-enzyme CES1/CYP yields) and fixes a parallel double-counting bug in `cyp_clearance_overrides.lookup_metabolic_fraction` (full InChIKey missed against the non-isomeric clinical_pk.json SMILES, leaving the default XGBoost-derived hepatic CL running alongside the explicit ProdrugActivationEdges); the fix shifts Meta AAFE 2.751 → 2.772 (+0.7%) and the In-domain Meta from 2.837 → 2.862, both well inside the bootstrap CI. The 2026-05-25 B-03.x refresh replaces the B-03 placeholder enzyme affinities (0.030 each for CES1/CYP3A4/CYP2C9) with literature-IVIVE values derived from Subash 2025 (Mol Pharm PMC12673578) rCES1 Vmax/Km + Boberg 2017 (PMC5267516) CES1 hepatic abundance + Kazui 2010 (DMD 38:92–99) 85/15 inactive/active fate split partition — CES1 0.0586, CYP3A4 0.0322, CYP2C9 0.0817 μL/min/pmol, preserving the literature 85/15 fate split and yielding a 1.92× total CLint scale-up over the placeholder; the resulting clopidogrel parent Meta fold tightens 5.15× → 4.67×, aggregate Meta AAFE shifts 2.772 → 2.769 (Δ=−0.0025, within bootstrap CI noise), and the clopidogrel registry disposition flips from `ceiling_accepted` to `literature_applied`. Bootstrap 95% CIs (10,000 resamples on |log<sub>10</sub>(fold)|, seed=20260422, computed 2026-05-20 after the B-03 fix) are above; the refreshed Meta CI [2.37, 3.26] overlaps the previous [2.30, 3.14], confirming the cumulative artifact + double-count shift remains within sampling uncertainty. The 2026-05-27 B-02 Phase 2 refresh activates the previously-disabled UGT path via 2 literature-curated substrate registries (`data/enzymes/{ugt2b7,ugt1a9}_substrates.json`, 8 seed drugs: morphine, codeine, ketorolac, indomethacin via UGT2B7; dapagliflozin, etodolac, bexagliflozin, glasdegib via UGT1A9) plus 2 abundance entries in `data/physiology/reference_man.yaml` (UGT2B7 2.43e6 pmol, UGT1A9 8.10e5 pmol; Achour 2017 PMC5328673 / Margaillan 2015 reference range). The cycle ships with **no DrugBank dependency** for the UGT path — closes the DE-36 reproducibility blocker. Gate-D 99-of-107 bit-identical verified (only the 8 seeds shift on the same numerics stack). Meta AAFE shifts 2.692 → 2.698 (Δ = +0.0067, **1.6% of the new bootstrap CI half-width [2.32, 3.17]** — well within sampling noise; see `data/validation/4track_ci_2026-05-27_B02.json`). 6 of 8 seeds improve (under-predicted drugs move toward observation); 2 of 8 (morphine, codeine) worsen because their pre-B-02 over-prediction relied on CYP-default over-extraction that the correct UGT path now partly displaces — a secondary diagnostic finding logged as DE-38, with `docs/_internal/backlog.md §B-13` (local-only) scoping the Phase 2.x abundance/IVIVE recalibration to address it (subsequently shipped as PR #49, metric-neutral). The headline table values above reflect the post-B-02 same-numerics-stack state and supersede the prior 2.772 headline (which was tied to a different numerics stack — Python/BLAS — and produced an aggregate-AAFE drift of ~0.08 unrelated to any cycle change, per the established ~12% per-drug numerics-stack drift). Prospective slice (15 NMEs) refresh is deferred to a follow-up commit (small expected delta; only ~3 of 15 drugs have DrugBank records). Artifacts: `data/validation/4track_ci_2026-05-12_v0.4.json` (B-03 era), `data/validation/4track_ci_2026-05-27_B02.json` (current). Subsequent metric-neutral cycles — B-13 gut-UGT correction (2026-05-29, ΔMeta AAFE ≈ −3×10⁻⁵) and B-14 hepatic-UGT IVIVE (2026-05-30, no-op / DE-40) — leave the 3-significant-figure headline 2.698 unchanged.

The 4-track meta-learner combines mechanistic PBPK (Engine), data-driven XGBoost C<sub>max</sub> (ML), a closed-form CL/F analytical (CLF), and a conditional VDss analytical track. Weights are compound-type-adaptive and LOOCV-calibrated: base compounds blend Engine 0.60 / ML 0.40; other compounds use Engine 0.35 / ML 0.50 / CLF 0.15, with VDss 0.20 added when applicability criteria are satisfied (and the remaining tracks re-scaled by ×0.80 so the total is 1.0). In-domain AAFE (N=79) excludes 28 drugs flagged as out-of-applicability-domain (prodrugs, high-MW, extreme lipophilicity, extended-release formulation mismatch). The current public-clone cache (`data/training/4track_holdout_predictions.json`) carries in-domain N=79; an earlier numerics stack briefly reported N=81 after the logp residual correction pushed two borderline drugs across the high-lipophilicity AD threshold, but that correction is no longer applied in the public-clone state.

**Prospective validation** (FDA NMEs approved 2024–2025, curated 2026-04, never used in training or tuning; refreshed 2026-05-12 under public-clone state; **N=14 after the 2026-05-31 vorasidenib contamination removal** — see note below):

| Slice | AAFE | 95% CI | %2-fold | N | Notes |
|---|:-:|:-:|:-:|:-:|:-:|
| All | 2.319 | [1.59, 3.47] | 57.1% | 14 | vorasidenib removed 2026-05-31 (training-data leak); was 2.402 N=15 |
| In-domain | 2.077 | [1.39, 3.29] | 70.0% | 10 | was 2.200 N=11 (incl. contaminated vorasidenib) |

Prospective AAFE remains **below** retrospective (2.319 &lt; 2.698 overall; 2.077 &lt; 2.760 in-domain), the opposite direction from classical over-fitting on the 107-holdout. N=14 is underpowered, so the prospective CIs are wide (Meta overall [1.59, 3.47], in-domain [1.39, 3.29]) — both overlap the retrospective in-domain Meta CI [2.39, 3.44], so the apparent prospective advantage is **not** statistically distinguishable from sampling noise. **Contamination correction (2026-05-31):** vorasidenib was removed from the prospective set after it was found in the training/reference corpora (`clinical_pk.json` gold-tier reference, `mmpk_expanded_v2.csv`, `vdss_v2_training.csv`, `bioavailability_v1.csv`, and `holdout.json['train']`) — it was never genuinely prospective. The remaining 14 drugs' per-drug predictions are unchanged (removing one drug does not alter the others'), so the corrected aggregates derive directly from the published `prospective_N15_public_only_2026-05-12.json` folds with no numerics-stack regeneration. Bootstrap CIs recomputed 2026-05-31 (10,000 resamples on |log<sub>10</sub>(fold)|, seed=20260422) via `scripts/bootstrap_4track_ci.py`. Artifacts: `data/validation/prospective_N14_public_only_2026-05-31.json` (per-drug folds), `data/validation/prospective_ci_2026-05-31_N14.json` (CI bundle); the superseded `prospective_N15_public_only_2026-05-12.json` / `prospective_ci_2026-05-15.json` are retained for audit trail.

**Cherry-picking caveat.** The 107-holdout has been used for ~47 configuration feedback cycles (track weights, routing, meta-learner variants). A quantitative audit (`docs/claude/cherry_picking_audit_2026-04-22.md`) scores aggregate risk 4.65/10 (moderate). The retrospective-contamination estimate (2.85–3.10 from the audit) overlaps the upper tail of the current public-clone Meta bootstrap CI ([2.32, 3.17], point estimate 2.698), meaning the headline cannot statistically reject the null hypothesis that tuning inflated AAFE. A secondary permanent holdout (N50) is planned per `docs/claude/cherry_picking_process_v1.md`.

### Multi-dose regimen validation

Three drugs were simulated at clinical dosing regimens and compared against FDA-label steady-state C<sub>max</sub> values:

| Drug | Regimen | Predicted C<sub>ss,max</sub> (mg/L) | FDA label C<sub>ss,max</sub> (mg/L) | Fold error |
|------|---------|:---:|:---:|:---:|
| Atorvastatin | 40 mg QD | 0.027 | 0.029 | 0.93 |
| Metformin | 500 mg BID | 0.55 | 1.0 | 0.55 |
| Warfarin | 5 mg QD | 0.34 | 1.4 | 0.24 |

Atorvastatin (CYP3A4-mediated, moderate protein binding) was predicted within 7% of the label value. Metformin (renal elimination-dominant, f<sub>u</sub> &asymp; 1.0) and warfarin (f<sub>u</sub> = 0.01, highly protein-bound) show expected under-prediction consistent with the model&rsquo;s known limitations in renal clearance modeling and CL<sub>int</sub> over-prediction for highly bound compounds. Accumulation ratio direction was correct for all three drugs. Steady-state detection operated correctly in all cases.

### TDM validation

Bayesian update was validated in two stages: a single-drug functional test, then a multi-drug benchmark across diverse pharmacokinetic profiles. The tables below report the **Importance Sampling** baseline (legacy); production now uses a dispatched SBI/IS/IBIS router (`data/sbi/method_routing.json`) with 12/13 production drugs routing to SBI after SBC-gate validation. SBI provides millisecond-scale inference with equivalent or better CV reduction; detailed SBC + per-drug coverage reports are tracked separately.

**Single-drug validation** (midazolam, 5 mg PO, one observation at t = 1 h, 10% assay CV):

| Metric | Prior | Posterior |
|--------|:-----:|:---------:|
| C<sub>max</sub> CV | 44.3% | 19.8% |
| ESS | &mdash; | 586.6 (29.3% of 2,000) |
| CV reduction | &mdash; | 55.4% |

**Multi-drug benchmark** (5 holdout drugs, synthetic patient observations scaled from engine C(t) profiles to observed C<sub>max</sub>, 10% log-normal assay noise, seed = 42):

| Drug | Type | 1-obs CV reduction | 2-obs CV reduction | 3-obs CV reduction | 1-obs ESS |
|------|:----:|:------------------:|:------------------:|:------------------:|:---------:|
| Morphine | base | 76.3% | 77.0% | 74.6% | 428 |
| Amantadine | base | 74.0% | 74.5% | 74.7% | 514 |
| Ketorolac | acid | 87.7% | 92.6% | 90.1% | 2.8 |
| Clozapine | neutral | 68.7% | 76.2% | 76.9% | 482 |
| Rivaroxaban | neutral | 83.8% | 93.3% | 98.3% | 7.1 |

Across all 15 runs (5 drugs &times; 3 observation scenarios): mean CV reduction 81.2%, mean error reduction 79.8%, 90% CI coverage 67%. A single observation suffices to reduce C<sub>max</sub> CV by 70&ndash;88% for drugs where the population prior fold error is below 2.5&times;. For drugs with larger prior errors (ketorolac, fold error 3.25&times;) or multi-observation scenarios, effective sample size degrades below 10, indicating particle degeneracy in the importance sampler. Sequential Bayesian methods (ensemble Kalman filter, particle filter) would be required for these cases.

Timepoint sensitivity analysis (morphine, single observation): t = 1.0 h (near T<sub>max</sub>) yielded maximal CV reduction (76.3%); observations beyond 4 h post-dose provided diminishing information (CV reduction 34%). Seed sensitivity across three random seeds was 0.8%, confirming robustness at N = 2,000 prior samples.

### Performance

| Operation | Time | Configuration |
|-----------|:----:|------|
| Full prediction (SMILES &rarr; C<sub>max</sub>) | 414 ms | Deterministic, single core |
| ODE solve (full fidelity) | 106 ms | LSODA, rtol=10<sup>&minus;8</sup>, atol=10<sup>&minus;10</sup> |
| ODE solve (MC fast path) | 33 ms | LSODA, rtol=10<sup>&minus;4</sup>, atol=10<sup>&minus;6</sup> |
| MC N=1,000 | 33.5 s | Pure Python RHS (no JIT compilation) |
| RHS evaluation | 31 &mu;s | 54 flux specs per call |

Single-patient deterministic prediction completes in &lt;500 ms, compatible with interactive clinical decision support workflows. MC propagation at N=1,000 requires ~34 s due to pure Python ODE evaluation; JIT compilation (e.g., via Numba) is an optimization path not yet pursued.

### Test suite

**856 passed / 0 failed / 27 skipped / 3 xfailed** (886 outcomes, full sweep 2026-05-27, 3:34; skip count includes artifact-conditional markers per `tests/_artifact_helpers.py`) covering graph construction, ODE compilation, flux functions (including ECM + V3 windowed IV-Cmax + ProdrugActivation + OneCompartmentElimination), solver correctness, mass balance (incl. two-species analytical 2C cascade), ADME prediction, meta-learner calibration, multi-dose regimen, TDM Bayesian update via SBI/IBIS/IS dispatch, MIPD dose recommendation, DDI, PK/PD, applicability-domain detection, prodrug-activation registry + pipeline integration, pharmacogenomic phenotype scaling (SLCO1B1, NAT2, UGT1A1), UGT2B7/UGT1A9 public substrate registry (B-02 Phase 2), and holdout benchmark reproducibility.

**Persistent xfails (3):** 3 statin Cmax tests under ECM (rosuvastatin, atorvastatin Peff over-prediction; fluvastatin under-prediction FE 4.79 in the ECM-forced gate / FE 1.54 in the production no-ECM `predict()` path — issue #21 closed post-PR #29 as not-ECM-applicable per Niemi 2009 CYP2C9-dominance; the gate failure is the intended signal that ECM should not be activated for fluvastatin). The 3 prodrug 3-fold clinical validation gates (sepiapterin, tebipenem-pivoxil, fostamatinib) that were previously xfailed per spec &sect; 3.3 mechanistic-A doctrine are now `pytest.skip`-gated under the public-clone state (DrugBank-conditional disposition data not present in this clone), preserving the mechanistic-A semantics without polluting the xfail count. remdesivir was promoted out of xfail in PR #43 (fold 2.78 < 3.0 gate).

**No failing tests.** A previously documented "known failing" entry for `test_irinotecan_returns_active_sn38_cmax` (claimed SN-38 C<sub>max</sub> 9.71 mg/L vs gate &lt; 1.0 mg/L) was a documentation error introduced 2026-05-08 in commit `71be8d0` — the test has always passed since its introduction in PR #34 (verified 2026-05-16: SN-38 Cmax 0.0466 mg/L, within Slatter 2000 clinical 0.03-0.10 mg/L range; the same value is recorded in `docs/claude/experiment-log.md` under the 2026-05-08 entry). The previously listed cached-AAFE assertion and v3 enzyme-leak audit failures were resolved in PR #43 (cached test refreshed to 2.751 under public-clone state; subsequently advanced to 2.772 via the B-03 clopidogrel registry fix, 2.769 via B-03.x literature-IVIVE, and 2.698 via the B-02 Phase 2 UGT public registry activation — current pinned test is `test_cached_holdout_aafe_is_2p698`; v3 leak audit now passes). Headline AAFE (cache 2.698) is re-runnable via `scripts/run_engine_benchmark.py`.

## Architecture

```
SMILES + dose
    │
    ▼
 predict ──► DrugOnGraph (enzyme-level, all values are Distribution)
                  │
                  ▼
             engine ◄── BodyGraph (from YAML)
             (compile graph → ODE → solve → MC propagate)
                  │
                  ▼
               pk (Cmax, AUC, t½ from SimResult)
                  │
    ml ───────────┤
    (direct PK)   │
                  ▼
             pipeline (meta-learner → final PredictionResult)
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
    regimen      ddi       pkpd
   (multi-dose, (enzyme    (effect
    TDM, MIPD)  adj.)     compartment)
```

| Layer | Responsibility | Depends on |
|-------|---------------|------------|
| `graph/` | BodyGraph, node/edge types, YAML builder | `core` |
| `engine/` | ODE compiler, flux registry (including ECM), solver, MC | `core`, `graph` |
| `predict/` | SMILES &rarr; chemistry &rarr; ADME &rarr; DrugOnGraph + transporter DB | `core` |
| `ml/` | XGBoost C<sub>max</sub>, CL/F, VDss predictors, 4-track meta-learner | `core` |
| `pk/` | SimResult &rarr; PKEndpoints (route-aware) | `core` |
| `regimen/` | Multi-dose solver, TDM method dispatch (SBI/IS/IBIS), MIPD | `core`, `engine`, `graph`, `sbi` |
| `sbi/` | Simulation-based inference training + amortized posterior | `core`, `engine` |
| `pipeline/` | Orchestrator wiring all layers | all layers |
| `ddi.py` | Drug-drug interactions (competitive inhibition, E<sub>max</sub> induction) | `core`, `graph` |
| `pkpd.py` | PK/PD effect modeling (effect compartment, sigmoid E<sub>max</sub>) | `core` |

**Layer isolation.** No cross-layer imports outside designated dependencies. `predict` does not import `engine`. `engine` does not import `predict`. `regimen` wraps `engine` without modifying it. Shared data types live in `core.py`.

### Design principles

1. **Identity-blind engine.** The ODE compiler and flux functions operate on node/edge *types*, never on *identities*. No string matching on organ names, enzyme names, or drug names exists in `engine/`. Replacing all organ names with random strings produces identical numerical results.

2. **Distribution-native.** All physiological and drug parameters are `Distribution` objects. Point estimates are represented as `Distribution(mean=x, cv=0)`. The uncertainty system is not an add-on; it is the system&rsquo;s native representation.

3. **Compile once, parameterize many.** Graph topology is compiled into an ODE skeleton once. MC iterations change only parameter values, not structure. 1,000 MC samples = 1 compilation + 1,000 ODE solves.

## Extending the Model

The architecture is designed so that new compartments, routes, populations, interaction models, and clinical workflows require **zero changes to the ODE engine**. This was validated empirically: subcutaneous injection, pediatric physiology, tumor compartment, DDI, PK/PD, multi-dose regimen, and TDM were each implemented with 0 lines changed in `src/sisyphus/engine/`.

### New organ (tumor compartment)

```yaml
nodes:
  - name: tumor
    type: organ
    volume: 0.05
    composition: {fn: 0.013, fp: 0.010, fw: 0.700, pH: 6.8}
edges:
  - {source: arterial_blood, target: tumor, type: flow, flow_fraction: 0.005}
  - {source: tumor, target: venous_blood, type: flow}
```

### New route (subcutaneous injection)

```python
graph.add_node(Node(name="sc_depot", node_type="lumen", volume=Distribution(0.01)))
graph.add_edge(AbsorptionEdge(source="sc_depot", target="venous_blood",
                               ka_fraction=Distribution(1.0)))
```

### New population (pediatric)

Allometrically scaled physiology (cardiac output &prop; BW<sup>0.75</sup>) with ontogeny-adjusted enzyme abundances (e.g., CYP3A4 at 50% of adult at age 5). Same graph structure, different YAML parameters.

### Drug-drug interactions

Competitive CYP inhibition via pre-simulation enzyme abundance adjustment:

```python
from sisyphus.ddi import apply_inhibition, KETOCONAZOLE

inhibited_graph = apply_inhibition(graph, KETOCONAZOLE)
# Midazolam AUC increases 12x (clinical reference: ~15x)
```

### PK/PD modeling

Effect compartment with sigmoid E<sub>max</sub> response, computed as post-processing on the concentration-time profile:

```python
from sisyphus.pkpd import compute_effect, PDModel

pd = PDModel(ke0=0.5, emax=100.0, ec50=0.05, hill=2.0)
effect = compute_effect(sim_result, pd)
```

## Limitations

- **Small-molecule oral PK only.** Biologics (antibodies, ADCs), parenteral formulations beyond SC, and non-oral routes (inhalation, topical) are not validated.
- **Prodrug activation: v3 input-data refresh shipped, validation gate still fails; v0.3.4/B-03 expands registry to 7 substrates.** Iterations: v1 (2026-04-26, first-order conversion `rate = k × A_parent`), v2 (PR #7, 2026-04-30, well-stirred enzyme-abundance extraction parallel to the CYP3A4 elimination pattern), v3 (PR #15, 2026-05-01, input-data quality refresh per spec §3.3 mechanistic-A doctrine — 6 items dispositioned: 2 literature_applied, 4 ceiling_accepted), **v0.3.4** (PR #34, 2026-05-08, registry expansion adding `simvastatin` and `irinotecan`), **B-04** (2026-05-19, optional per-enzyme `yield` field), and **B-03** (2026-05-20, clopidogrel dual-fate CES1 dead-end + CYP bioactivation entry). v2 replaces `conversion_rate_per_h` with `enzyme_affinity_for_conversion: dict[str, Distribution]` and adds SPR/CES1/CES2/ALPI enzyme abundances at liver/gut_wall/kidney; B-04 adds per-enzyme conversion yields for multi-fate prodrugs. The shared 3-drug 3-fold clinical validation gate **still fails under v3** (sepiapterin 4748&times;, tebipenem-pivoxil 9.05&times;, fostamatinib 4.50&times;) — extraction-step rate-limits dominate over active CL/V disposition. Irinotecan SN-38 passes its mechanical correctness gate under public-clone state (0.0466 mg/L, within Slatter 2000 clinical 0.03&ndash;0.10 mg/L range). Clopidogrel remains scored as parent C<sub>max</sub> in the 107-holdout. The B-03.x literature-IVIVE refresh (2026-05-25) replaced the B-03 placeholder affinities (0.030 each, calibrated to the 85/15 fate split only) with Subash 2025 rCES1 Vmax/Km + Boberg 2017 CES1 abundance + Kazui 2010 fate-split partition (CES1 0.0586, CYP3A4 0.0322, CYP2C9 0.0817 μL/min/pmol; disposition_state flipped to `literature_applied`); the parent Meta fold tightened 5.15&times; → 4.67&times; (1.92× total CLint scale-up), within bootstrap CI noise. Active R-130964 disposition remains `ceiling_accepted` because the labile thiol and covalent P2Y12 binding prevent a clean conventional CL/V measurement. Prodrugs continue to be flagged out-of-applicability-domain. Detailed status: `CHANGELOG.md` &sect; Unreleased.
- **Simplified pK<sub>a</sub>.** Ionization state is classified by structural rules (carboxylic acid &rarr; 4.5, aliphatic amine &rarr; 9.0), not computed quantum-mechanically. This limits Kp accuracy for highly ionized compounds.
- **Phase II metabolism &mdash; expanding.** Liver NAT2 (1.0e7 pmol, CV 0.6) and UGT1A1 (1.215e6 pmol, CV 0.5) abundances were added in v0.3.2 (PR #32) with phenotype-aware scaling propagated through `predict()`; empirical PM/EM Cmax ratios verified at v0.3.2 merge (tizanidine CYP1A2 1.518&times;, isoniazid NAT2 1.478&times;, raltegravir UGT1A1 1.419&times;). PR #32 also closed a silent-zero back-solve cancellation bug for CYP/UGT/NAT phenotypes via pre-phenotype abundance snapshotting. **B-02 Phase 2 (2026-05-27) adds UGT2B7 (2.43e6 pmol) and UGT1A9 (8.10e5 pmol) abundances with 8 literature-curated substrate registry entries (no DrugBank dependency); phenotype scaling for UGT2B7/UGT1A9 deferred to Phase 2.x.** Sulfation (SULT) and other UGT isoforms (UGT1A4, UGT2B15 etc.) remain unmodeled; drugs cleared primarily by these routes will be under-predicted.
- **Transporter-mediated disposition: OATP1B1 only.** Hepatic uptake by OATP1B1 is modeled mechanistically via the ECM (closed-form QSSA hepatocyte flux) with per-drug kinetic parameters in `data/transporters/oatp1b1.json`. Other hepatic transporters (OATP1B3, NTCP, BSEP), intestinal transporters (P-gp, BCRP), and renal transporters (OAT1/3, MATE1/2-K) are not mechanistically modeled. P-gp efflux at the gut wall is approximated via a binary permeability correction.
- **CL<sub>int</sub> prediction is the weakest link.** The XGBoost CL<sub>int</sub> model achieves R&sup2; &asymp; 0.24 on TDC Hepatocyte_AZ (scaffold-split CV). This ceiling persists across molecular representations (Morgan FP, MACCS keys, atom-pair FP, MoLFormer, ChemBERTa, Uni-Mol, Chemprop D-MPNN), model architectures (XGBoost, Random Forest, Ridge, GNN), data scales (978&ndash;1,910 compounds), and alternative formulations (classification, BDE reactivity features, direct CL/F bypass, AUC decomposition). The authoritative failed-experiment list (`docs/claude/dead-ends.md`) enumerates **40 distinct approaches** across 13 categories, none of which produced a meaningful reduction in holdout AAFE. The primary bottleneck is assay noise in public hepatocyte clearance data, not model capacity or molecular representation. Bayesian TDM partially mitigates this at the individual patient level: observed drug concentrations correct inaccurate population priors, reducing posterior CV by &gt;50% (see [TDM validation](#tdm-validation)).
- **Error cancellation constrains component-level improvements.** The IVIVE pipeline (f<sub>u,p</sub> &times; CL<sub>int</sub> &times; scaling &rarr; CL<sub>h</sub>) exhibits systematic error cancellation: improving any single ADME component (e.g., CL<sub>int</sub> R&sup2; from 0.21 to 0.33 via data expansion) worsens overall AAFE because the error balance with other components is disrupted. Simultaneous replacement of all ADME models also failed to improve AAFE (+0.023), and post-hoc meta-learner optimization across more than 60 blending strategies (stacking, analog correction, rank aggregation, Bayesian model averaging, ensemble selection, isotonic/LOWESS calibration, substructure correction, disagreement routing, and others) confirmed that all such combinations produce holdout errors correlated at r &gt; 0.95 with the baseline meta-learner. The current compound-type-adaptive geometric blend is provably near-optimal at this sample size. Measured ADME inputs (experimental f<sub>u,p</sub> and CL<sub>int</sub>) reduce engine AAFE from 2.33 to 1.98, confirming that the mechanistic architecture is sound when inputs are accurate.
- **IV-Cmax observation convention.** For intravenous bolus dosing, engine Cmax is extracted as the maximum concentration over `t ≥ 5 min` (windowed max), not the instantaneous `dose / V_venous` spike at `t = 0`. This matches the clinical first-draw convention and is route-conditional; oral drugs use full-interval max (V2-compatible). The 107-drug holdout set is entirely oral, so this methodology has zero impact on the headline AAFE. See `docs/superpowers/specs/2026-04-22-iv-cmax-observation-design.md`.
- **ECM (Extended Clearance Model) generalization unverified for non-statins.** ECM is validated on 5 statins (2/5 strict-pass 3-fold gate: pravastatin (FE 1.066, post-PR #22 metabolic\_fraction reconciliation), pitavastatin; fluvastatin xfail in the ECM-forced gate (FE 4.79 under-prediction) — issue #21 closed post-PR #29 reclassifying fluvastatin as not-ECM-applicable per Niemi 2009 CYP2C9-dominance; production `predict()` correctly skips ECM via the `ecm_applicable=false` flag in `data/transporters/oatp1b1.json` and yields FE 1.54 within the 3-fold gate; rosuvastatin/atorvastatin xfail due to Peff over-prediction). A pre-registered generalization test on valsartan + glimepiride (2026-04-22, N=2) returned Mode C with systematic 2.5× underprediction under V3 methodology. The fup override hypothesis was ruled out (DE-33); candidates remain for Jmax calibration, Vss/Kp over-distribution, and ECM architectural limits for Km &gt; 1 µM substrates. Users should not rely on ECM accuracy for non-statin OATP1B1 substrates without independent verification.
- **R<sub>B:P</sub> defaults to 1.0.** The RBP model (R&sup2; = &minus;0.08 on external data) is effectively disabled; all drugs are assumed to have equal blood and plasma concentrations.
- **90% prediction interval is uncalibrated.** The first empirical coverage measurement (2026-04-24, full N=107 holdout, 1,000 MC samples; `data/validation/holdout_pi_coverage_2026-04-24.json`) yielded **29.9%** at the nominal 90% level. The MC interval reflects only parameter-uncertainty propagation and captures roughly one third of the observed residual spread &mdash; the remaining ~60 percentage points are structural error (IVIVE scaling assumptions, DE-33 OATP underprediction, CL<sub>int</sub> assay noise). The 90% PI is therefore exposed as a **diagnostic** quantity and must not be quoted as a clinically calibrated interval without empirical recalibration.
- **MIPD assumes linear pharmacokinetics.** Dose recommendations use linear scaling, which may be inaccurate for drugs with saturable metabolism (e.g., phenytoin) or nonlinear protein binding.
- **TDM importance sampling degenerates for large prior errors.** When the population prior is far from the individual truth (fold error &gt;3&times;) or multiple observations are used (&ge;3), the effective sample size drops below 10, indicating particle weight degeneracy. Sequential Bayesian methods (EnKF, particle filter) would address this.

## Project Structure

```
src/sisyphus/
├── core.py              # Distribution, TissueComposition, data contracts
├── descriptors.py       # Morgan fingerprints + RDKit descriptors
├── compounds.py         # Compound YAML → DrugOnGraph loader
├── ddi.py               # Drug-drug interaction modeling
├── pkpd.py              # PK/PD effect compartment + Emax
├── cli.py               # Command-line interface (6 commands)
│
├── graph/               # Body graph definition and construction
│   ├── types.py         # Node, Edge type hierarchy (frozen dataclasses)
│   ├── body.py          # BodyGraph (add/remove/validate/sample)
│   ├── builder.py       # YAML → BodyGraph with flow conservation check
│   └── presets.py       # reference_man() (reference_woman YAML not shipped)
│
├── engine/              # ODE compilation and solving (identity-blind)
│   ├── compiler.py      # ODECompiler, CompiledODE, ResolvedParams
│   ├── flux.py          # FluxSpec implementations (8 transport types,
│   │                    #   incl. ECM ActiveTransport, ProdrugActivation,
│   │                    #   OneCompartmentElimination)
│   ├── params_jax.py    # JAX parameter resolution (experimental)
│   ├── result.py        # SimResult dataclass
│   ├── rhs_jax.py       # JAX right-hand side (experimental)
│   ├── solver.py        # LSODA wrapper (solve, solve_mc)
│   ├── solver_jax.py    # JAX solver (experimental)
│   └── uncertainty.py   # Monte Carlo propagation
│
├── predict/             # SMILES → drug parameterization
│   ├── chemistry.py     # Molecular profiling, pKa, AD assessment
│   ├── adme.py          # XGBoost ADME property prediction
│   ├── ivive.py         # In vitro → in vivo extrapolation, Kp
│   ├── drugbank.py      # DrugBank experimental enrichment (fup, logP)
│   ├── phenotype.py     # Pharmacogenomic phenotype (e.g., SLCO1B1)
│   ├── registry.py      # Prodrug activation registry (SMILES-keyed)
│   ├── cyp_clearance_overrides.py # metabolic_fraction registry (OATP1B1 substrates, ECM)
│   ├── non_cyp_substrates.py # NAT2 / UGT1A1 substrate-class registry (SMILES-keyed)
│   └── transporter_db.py # OATP1B1 + hepatic ECM kinetic parameters
│
├── ml/                  # Data-driven PK prediction
│   ├── features.py      # Feature vector construction
│   ├── models.py        # XGBoost Cmax predictor
│   ├── clf_predictor.py # CL/F analytical track
│   ├── registry.py      # Model manifest + feature-schema hash (H2)
│   ├── surrogate.py     # Optional ML surrogate solver (experimental; relocated from engine/ in PR #39)
│   └── ensemble.py      # 4-track meta-learner (_W_VDSS=0.20)
│
├── pk/                  # PK endpoint extraction
│   ├── endpoints.py     # SimResult → PKEndpoints (route-aware t_min_h)
│   ├── nca.py           # Non-compartmental analysis (AUC, t½)
│   └── analytical.py    # Closed-form 1-cpt and 2-cpt solutions
│
├── regimen/             # Multi-dose and clinical pharmacology
│   ├── types.py         # DosingEvent, DosingRegimen (frozen dataclasses)
│   ├── solver.py        # Event-driven multi-dose solver
│   ├── profile.py       # Steady-state detection, Css metrics
│   ├── tdm.py           # Bayesian TDM dispatcher (SBI/IS/IBIS routing)
│   ├── tdm_sbi.py       # SBI (neural posterior) TDM method
│   ├── tdm_ibis.py      # Iterative Bayesian Importance Sampling
│   ├── tdm_enkf.py      # Ensemble Kalman filter (sequential alternative)
│   └── dosing.py        # MIPD dose recommendation
│
├── sbi/                 # Amortized neural posterior estimation
│   ├── priors.py        # Drug/physiology prior distributions
│   ├── simulator.py     # Forward model for SBI training
│   ├── amortizer.py     # Neural posterior trainer (NSF)
│   ├── multi_drug.py    # Multi-drug amortizer + continuous (bw, age)
│   ├── physiology_generator.py  # Achour correlated physiology sampler
│   └── sbc.py           # Simulation-Based Calibration gate
│
├── validation/          # Benchmarking infrastructure
│   ├── reference.py     # Clinical PK reference loader (331 drugs)
│   ├── benchmark.py     # Holdout benchmark runner (--compute-pi from H3)
│   ├── metrics.py       # AAFE, fold error, PI coverage
│   └── oatp_generalization.py  # ECM substrate generalization classifier
│
└── pipeline/            # End-to-end orchestration
    ├── predict.py       # SMILES → PredictionResult (route-conditional V3)
    └── config.py        # Pipeline configuration

data/
├── physiology/          # BodyGraph YAML definitions
├── compounds/           # Curated compound configurations
└── reference/           # Clinical PK reference data, holdout split

models/                  # Pre-trained XGBoost models (committed; ~37MB)
```

## Predecessor

Sisyphus inherits validated data assets from [Omega PBPK](https://github.com/jam-sudo/Omega) (591 commits) but not its architecture:

| Inherited (data) | Not inherited (architecture) |
|-------------------|------------------------------|
| 331-drug clinical reference | 35-state hardcoded ODE system |
| Scaffold-stratified holdout split | Organ-specific CL<sub>int</sub> fields |
| ICRP physiology values | Sequential ADME &rarr; IVIVE chain |
| Pre-trained XGBoost models | Point-estimate pipeline |
| Rodgers &amp; Rowland tissue compositions | Post-hoc hybrid selector |

Key empirical findings from Omega that informed Sisyphus:

- **Data quality dominates model choice.** 14 reference corrections reduced AAFE by 47.5% with zero model changes.
- **Gut CL<sub>int</sub> &gt; hepatic CL<sub>int</sub> for C<sub>max</sub>.** Global sensitivity analysis (Sobol): gut S<sub>T</sub>=0.47, hepatic S<sub>T</sub>=0.00.
- **Meta-learner &gt; fixed ensemble.** Feature importance: ML C<sub>max</sub> 50%, PBPK C<sub>max</sub> 26%.

## References

- Berezhkovskiy, L. M. (2004). Volume of distribution at steady state for a linear pharmacokinetic system with peripheral elimination. *J Pharm Sci*, 93(6), 1628&ndash;1640.
- Houston, J. B. (1994). Utility of in vitro drug metabolism data in predicting in vivo metabolic clearance. *Biochem Pharmacol*, 47(9), 1469&ndash;1479.
- Huang, K., et al. (2021). Therapeutics Data Commons: Machine learning datasets and tasks for drug discovery and development. *NeurIPS Datasets and Benchmarks*.
- ICRP (2002). Basic anatomical and physiological data for use in radiological protection: reference values. *ICRP Publication 89*.
- Obach, R. S., et al. (1997). The prediction of human pharmacokinetic parameters from preclinical and in vitro metabolism data. *J Pharmacol Exp Ther*, 283(1), 46&ndash;58.
- Petzold, L. R. (1983). Automatic selection of methods for solving stiff and nonstiff systems of ordinary differential equations. *SIAM J Sci Stat Comput*, 4(1), 136&ndash;148.
- Poulin, P., &amp; Theil, F. P. (2002). Prediction of pharmacokinetics prior to in vivo studies. *J Pharm Sci*, 91(4), 940&ndash;951.
- Rodgers, T., &amp; Rowland, M. (2005). Physiologically based pharmacokinetic modelling 2: Predicting the tissue distribution of acids, very weak bases, neutrals and zwitterions. *J Pharm Sci*, 95(6), 1238&ndash;1257.
- Rodgers, T., &amp; Rowland, M. (2006). Mechanistic approaches to volume of distribution predictions: Understanding the processes. *Pharm Res*, 24(5), 918&ndash;933.
- Shimada, T., et al. (1994). Interindividual variations in human liver cytochrome P-450 enzymes involved in the oxidation of drugs, carcinogens and toxic chemicals. *J Pharmacol Exp Ther*, 270(1), 414&ndash;423.
- Shitara, Y., Maeda, K., Ikejiri, K., Yoshida, K., Horie, T., &amp; Sugiyama, Y. (2013). Clinical significance of organic anion transporting polypeptides (OATPs) in drug disposition: their roles in hepatic clearance and intestinal absorption. *Biopharm Drug Dispos*, 34(1), 45&ndash;78.
- Yang, J., Jamei, M., Yeo, K. R., Tucker, G. T., &amp; Rostami-Hodjegan, A. (2007). Prediction of intestinal first-pass drug metabolism. *Curr Drug Metab*, 8(7), 676&ndash;684.
- Yoshikado, T., Toshimoto, K., Nakada, T., Ikejiri, K., Kusuhara, H., Maeda, K., &amp; Sugiyama, Y. (2017). Comparison of methods for estimating unbound intracellular-to-medium concentration ratios in rat and human hepatocytes using statins. *Drug Metab Dispos*, 45(7), 779&ndash;789.
- Yu, L. X., &amp; Amidon, G. L. (1999). A compartmental absorption and transit model for estimating oral drug absorption. *Int J Pharm*, 186(2), 119&ndash;125.

## How to Cite

If you use Sisyphus in your research, please cite:

```
Yoon, J. M. (2026). Sisyphus (0.1.0): Graph-based whole-body PBPK
simulation with native uncertainty propagation.
https://github.com/jam-sudo/Sisyphus
```

> The git tag `v1.0.0` (commit `67b7064`) records an earlier feature-branch
> milestone that has been superseded by the current `main` line under a
> more conservative measurement regime. See `CHANGELOG.md` for details.

## Requirements

- Python &ge; 3.10
- numpy, scipy, pyyaml (core)
- rdkit, xgboost, scikit-learn (prediction)

## License

MIT
