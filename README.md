# Sisyphus

**Oral structure + dose → Cmax, with a separate graph-based PBPK research engine**

[![CI](https://github.com/jam-sudo/Sisyphus/actions/workflows/ci.yml/badge.svg)](https://github.com/jam-sudo/Sisyphus/actions/workflows/ci.yml)

[Methodology](#methodology) &middot; [Quickstart](#quickstart) &middot; [Validation](#validation) &middot; [Architecture](#architecture) &middot; [Limitations](#limitations)

**Preprint:** [Yoon, J. M. (2026). *Sisyphus: A Topology-Compiled Physiologically Based Pharmacokinetic Platform with Structure-Only Input and Bayesian Parameter Refinement.* ChemRxiv.](https://doi.org/10.26434/chemrxiv.15004452/v1) &mdash; DOI [10.26434/chemrxiv.15004452/v1](https://doi.org/10.26434/chemrxiv.15004452/v1)

The published v1 preprint and archived `Sisyphus_Preprint.pdf` report an older AAFE of 2.698. The current code's source-screened development benchmark is 2.8405 on 73 scored compounds; the value comes from a different model and reference set. Use the validation section below for current evidence.

---

Sisyphus is an oral structure-only C<sub>max</sub> prediction system with a separate physiologically based pharmacokinetic (PBPK) research engine. The engine represents the human body as a typed directed multi-graph and derives ordinary differential equation (ODE) systems from graph topology.

The production output is C<sub>max</sub> for a canonical parent SMILES and positive oral dose. Engine-derived T<sub>max</sub>, AUC, half-life, multi-dose simulation, TDM, MIPD, DDI, PGx, and PK/PD are experimental research outputs and are not covered by the C<sub>max</sub> accuracy claim. Residual/model-error and parameter-Monte-Carlo intervals are exposed separately.

**Intended use.** Sisyphus targets oral structure-only Cmax prediction (canonical parent SMILES + dose) when measured ADME is unavailable. On the repeatedly accessed retrospective **development benchmark**, Meta AAFE is 2.841 [bootstrap 95% CI 2.31&ndash;3.54, N=73 scored]. This is not an independent holdout result: the original 107-compound cohort has informed repeated system-selection decisions. A source-adjudicated diagnostic P0 pilot on 18 compounds, scored with an earlier model, found Meta AAFE **3.34**, compared with **3.01** for direct ML; its labels were AI-assisted, historical VDss training membership is unverified, and it does not establish Meta superiority. The current system has **no unconsumed independently curated external holdout AAFE**. Error of this scale and the wide development-residual interval (&divide;&times;~9.97-fold; 91.8% coverage on the repeatedly used development set) restrict the tool to **screening, ranking, and uncertainty-aware triage**, not dose setting.

```
$ sisyphus predict --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O" --dose 100

Drug: Cn1c(=O)c2c(ncn2C)n(C)c1=O
Method: hybrid
Execution: ok (public profile)
Applicability: no rule-based warning flags (confidence is not calibrated)
Final oral Cmax: <value> mg/L
Development empirical residual 90% interval: <low>–<high> mg/L
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
The coefficient 2.88 is an empirical absorption-scale parameter for these numerical input units; it carries the units needed to produce h<sup>&minus;1</sup> and is not a dimensionless cm/s-to-&mu;m/h conversion. Its transferability across formulations has not been independently established.

**Tissue:plasma partition coefficients** are computed via the Rodgers &amp; Rowland method (Rodgers &amp; Rowland, 2005, 2006), with the Berezhkovskiy correction for acids (Berezhkovskiy, 2004).

### Prediction pipeline

The full pipeline combines mechanistic simulation with data-driven prediction:

1. **SMILES &rarr; molecular profile**: RDKit descriptors, structural pK<sub>a</sub> classification, applicability domain assessment
2. **ADME prediction**: Pre-trained XGBoost models for f<sub>u,p</sub>, CL<sub>int</sub>, R<sub>B:P</sub>, VD<sub>ss</sub> (trained on TDC datasets; Huang et al., 2021), with DrugBank experimental f<sub>u,p</sub> enrichment where available
3. **IVIVE**: CL<sub>int</sub> decomposition into per-enzyme affinities, Kp calculation
4. **PBPK simulation**: 34-state ODE system solved via LSODA (Petzold, 1983)
5. **ML direct prediction**: XGBoost C<sub>max</sub> model (trained on 1,128 drugs from multi-source clinical PK data)
6. **CL/F analytical track**: closed-form 1-compartment C<sub>max</sub> estimate using XGBoost CL/F + V<sub>d</sub> predictions and k<sub>a</sub> from Engine T<sub>max</sub> / Peff. Decorrelates with Engine+ML residuals via different input channels.
7. **VDss volume proxy**: dose divided by predicted VDss (volume-of-distribution-at-steady-state) for a fixed 70 kg body weight. This is a simple scale estimate, not an absorption/elimination C<sub>max</sub> model. It is included whenever a positive VDss estimate is available; this routing was selected on N=107 and is not an independently validated applicability rule.
8. **Meta-learner**: Compound-type-adaptive geometric blend of all four tracks with weights selected by LOOCV on the repeatedly accessed N=107 development set. Base compounds: engine 0.60 / ML 0.40 / CLF 0.00; non-base: engine 0.35 / ML 0.50 / CLF 0.15. VDss track weight 0.20 when activated; other weights scaled by ×0.80 so the four-track sum remains unity. These weights are frozen pending blinded external evaluation.

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

Effective sample size (ESS = $1/\sum w_i^2$) is monitored for importance-based methods. Morphine routes SBI with a likelihood reweighting kernel (P6, 2026-04-19) to compensate for hierarchical variance deficiency. All three methods share a unified output contract: per-PK-parameter posterior Distribution. This mechanism bypasses the CL<sub>int</sub> prediction ceiling (current scaffold-CV R&sup2; = 0.215): observed drug levels directly correct inaccurate population priors, reducing posterior CV by &gt;50% in validation (see [TDM validation](#tdm-validation)).

### Model-informed precision dosing

MIPD recommends adjusted doses to achieve a target steady-state concentration:

1. Bayesian update at the observed dose yields a posterior C<sub>ss</sub> distribution.
2. Linear dose scaling: $dose_{new} = dose_{current} \times (C_{ss,target} / C_{ss,posterior})$
3. Clamp to clinical dose range and round to a practical increment (default 25 mg).

Linear scaling assumes non-saturable metabolism. The dosing API rejects drugs
with declared saturable enzyme kinetics unless the caller explicitly opts into
an experimental override; production nonlinear dosing requires forward
simulation across candidate doses.

### Experimental engine-as-prior posterior PK (MIPD module)

The `mipd/` module explores using the mechanistic engine as a structural prior updated by measured concentrations. Bioavailability F is one candidate latent because formulation, salt or crystal form, food, particle size, and transporter genetics are absent from the SMILES input. The current experiments do not show that one observation identifies F or corrects all sources of structural error.

True F can be treated experimentally as a latent with a wide prior centered on the engine's emergent F<sub>engine</sub>, then updated from measured data by sampling/importance-resampling (SIR). This research path is not part of the supported structure-only Cmax path, and its posterior intervals have not been independently shown to be clinically calibrated.

The module provides an a-priori-to-posterior path (`predict_posterior`), steady-state IV trough TDM with a renal-clearance latent (`predict_tdm`, vancomycin/aminoglycoside scope), patient covariate individualization (creatinine clearance via measured CrCl or a Cockcroft-Gault estimate; body weight and age via a physiology-generator graph swap), and target-attainment dose recommendation (`recommend_dose`) over the resulting posterior.

> This module is a research hypothesis. Synthetic/simulator validation does not
> establish real-patient predictive accuracy or clinical utility.

### Experimental drug-drug interactions

DDI is modeled by adjusting enzyme abundances in the body graph prior to ODE compilation. The engine sees modified abundances and computes clearance as usual &mdash; no engine modifications are required.

**Competitive inhibition:**

$$E_{eff} = \frac{E_{base}}{1 + [I] / K_i}$$

**Enzyme induction (E<sub>max</sub> model):**

$$E_{eff} = E_{base} \cdot \left(1 + \frac{E_{max} \cdot [I]}{EC_{50} + [I]}\right)$$

where $E_{base}$ is baseline enzyme abundance (pmol), $[I]$ is perpetrator plasma concentration, $K_i$ is the inhibition constant, and $E_{max}$/$EC_{50}$ are induction parameters. Preset perpetrators: ketoconazole, fluconazole, quinidine (CYP inhibitors) and rifampin (CYP3A4 inducer).

### Experimental PK/PD link

Pharmacodynamic effects are computed from the concentration-time profile via an effect compartment with sigmoid E<sub>max</sub> response:

**Effect-site equilibration:**

$$\frac{dC_e}{dt} = k_{e0} \cdot (C_p - C_e)$$

**Sigmoid E<sub>max</sub> response:**

$$E = E_0 + \frac{E_{max} \cdot C_e^n}{EC_{50}^n + C_e^n}$$

where $k_{e0}$ is the effect-site equilibration rate constant (h<sup>&minus;1</sup>), $E_0$ is the baseline effect, and $n$ is the Hill coefficient. This is implemented as analytical post-processing on the PK solution, not as additional ODE states, preserving engine isolation. Preset PD models are provided for midazolam (sedation) and warfarin (INR response).

## Quickstart

### Installation

```bash
pip install -e ".[dev,ml]"
```

> Pre-trained XGBoost models (f<sub>u,p</sub>, CL<sub>int</sub>, R<sub>B:P</sub>, VD<sub>ss</sub>, C<sub>max</sub>) are required in `models/adme/` and `models/direct_pk/`. Re-training scripts are provided in `scripts/`.

### CLI

The oral `predict` command is the production-facing interface. The other commands below are explicitly experimental research interfaces:

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
sisyphus benchmark --development-set
```

All commands accept `--verbose` for debug-level logging.

### Python API

```python
from sisyphus.pipeline.predict import predict

result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)

print(result.cmax_prediction.cmax.mean)       # authoritative Meta Cmax
print(result.cmax_prediction.interval_90)     # residual/conformal interval
print(result.engine_simulation.endpoints)     # coherent engine-only endpoints
print(result.execution_status)                # "ok" / explicit fallback status
print(result.in_applicability_domain)         # rule-based flags, not calibrated accuracy
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

print(f"Cmax: {pk.cmax.mean:.4f} mg/L")  # ~0.0028 mg/L (post FLUX-1 + RBP-2; see Validation)
```

### Monte Carlo uncertainty

```python
from sisyphus.engine.uncertainty import UncertaintyEngine

ue = UncertaintyEngine()
mc = ue.propagate_fast(compiled, graph, drug, n_samples=1000)

print(mc.pk.cmax)                # Distribution(mean≈0.0030, cv≈0.22)
print(mc.cmax_90ci)              # (0.0018, 0.0040) mg/L
print(len(mc.cmax_samples))      # 1000 individual realizations
```

## Validation

### Engine validation: Omega parity and post-correction snapshots

Four drugs with known compound parameters are simulated end-to-end (YAML &rarr; BodyGraph &rarr; compile &rarr; solve &rarr; C<sub>max</sub>). They originally matched the [Omega PBPK](https://github.com/jam-sudo/Omega) ODE engine (35-state hardcoded model) to within ~0.5% — but Omega *shared* two physiological bugs that Sisyphus has since corrected, so cross-engine parity is no longer the right oracle for three of the four:

- **FLUX-1** (2026-06-03): a flow-limitation double-count that capped hepatic/gut extraction at E&rarr;0.5. Moves high-extraction drugs (midazolam, propranolol).
- **RBP-2** (2026-06-04): a blood:plasma concentration-basis correction. Moves any drug with R<sub>B:P</sub> &ne; 1 (midazolam 0.66, warfarin 0.58, propranolol 0.81).

After both fixes, only **caffeine** (R<sub>B:P</sub>=1, low-extraction) remains a true cross-engine Omega-parity check. The other three are now **Sisyphus self-consistency regression snapshots** — their divergence from Omega *is* the correctness fix, not an error. Values are pinned in `tests/integration/test_engine_validation.py` (&plusmn;5% gate; the targets carry documented macOS&harr;CI numerics-stack drift).

| Drug | Dose | Sisyphus C<sub>max</sub> (mg/L) | Omega C<sub>max</sub> (mg/L) | Basis |
|------|:----:|:------------:|:----------:|:------|
| Caffeine (Omega parity) | 100 mg PO | 1.6910 | 1.7139 | **Omega parity** — R<sub>B:P</sub>=1, low-extraction; invariant to both fixes (1.3% &lt; &plusmn;5% gate, macOS-stack drift) |
| Midazolam (Sisyphus snapshot, post-FLUX-1/RBP-2) | 2 mg PO | 0.002800 | 0.006943 | Sisyphus snapshot — FLUX-1 + RBP-2 (R<sub>B:P</sub> 0.66) |
| Warfarin (Sisyphus snapshot, post-FLUX-1/RBP-2) | 10 mg PO | 0.343133 | 0.4922 | Sisyphus snapshot — RBP-2 (R<sub>B:P</sub> 0.58); FLUX-1 no-op (low-extraction) |
| Propranolol (Sisyphus snapshot, post-FLUX-1/RBP-2) | 80 mg PO | 0.059875 | 0.1355 | Sisyphus snapshot — FLUX-1 + RBP-2 (R<sub>B:P</sub> 0.81) |

Mass balance error &lt; 10<sup>&minus;12</sup> for all simulations. **Lesson:** Omega parity is *not* a sufficient correctness oracle — both shared bugs survived for as long as they did precisely because parity held.

> Engine-validation targets: high-extraction drugs (midazolam, propranolol) use post-FLUX-1/RBP-2 Sisyphus regression snapshots, not Omega parity — Omega shared the flow-limitation double-count bug FLUX-1 fixed, so Omega parity is no longer a correctness oracle for them. Warfarin is also a Sisyphus snapshot (RBP-2 only; FLUX-1 was a no-op for this low-extraction drug). The blood:plasma concentration basis is RBP-2 (whole-blood pools reported on a plasma basis); see `docs/_internal/specs/2026-06-04-rbp-concentration-basis-design.md`.

### Retrospective development benchmark (SMILES &rarr; C<sub>max</sub>)

Retrospective evaluation on a Murcko scaffold-stratified development split (107 compounds, seed=42; 73 currently have source-supported scored Cmax). All seven active fitted models now have SHA-pinned public training snapshots with compound-level exclusions, but this cohort has been used for repeated weight, track, routing, and mechanism feedback and therefore is not an independent system holdout. It integrates observed concentration&ndash;time profiles from OSP, curated literature PK data, and FDA DailyMed labels, with mixed formulations and populations. Performance is reported using AAFE with bootstrap confidence intervals conditional on the selected system; those intervals do not include adaptive model-selection bias:

$$AAFE = 10^{\operatorname{mean}\left(\left|\log_{10}\frac{C_{max,pred}}{C_{max,obs}}\right|\right)}$$

| Track | AAFE | 95% CI | %2-fold | %3-fold | N |
|---|:-:|:-:|:-:|:-:|:-:|
| **Meta-learner (production)** | **2.841**† | [2.31, 3.54] | 42.5% | 65.8% | 73 |
| Engine only | 3.872 | [2.99, 5.09] | 38.4% | 50.7% | 73 |
| ML only | 3.034 | [2.44, 3.79] | 43.8% | 58.9% | 73 |
| Meta, in-domain | 2.904 | [2.34, 3.68] | 38.3% | 63.3% | 60 |

The paired compound-bootstrap Meta/ML AAFE ratio is **0.936** (95% CI
**0.827–1.057**, 10,000 resamples, seed 20260422). This conditional interval
does not account for repeated system selection. The development data cannot
establish independent Meta superiority. The in-domain ratio is
1.000 (0.893–1.125).

A reference-curve audit removed 165 synthetic or arm-mixed concentration
profiles, including seven attached to scored development drugs; see
[`development_reference_curve_followup_2026-09-24.md`](docs/validation/development_reference_curve_followup_2026-09-24.md).
The subsequent parent-dose and quinine-arm correction is documented in
[`development_parent_dose_followup_2026-09-24.md`](docs/validation/development_parent_dose_followup_2026-09-24.md).
The codeine reference was subsequently replaced with a directly matched FDA
fasted tablet arm; see
[`development_codeine_fda_arm_followup_2026-09-24.md`](docs/validation/development_codeine_fda_arm_followup_2026-09-24.md).
The implicit-salt follow-up corrected amantadine, fluvoxamine, hydroxyzine,
and trazodone; see
[`development_implicit_salt_followup_2026-09-24.md`](docs/validation/development_implicit_salt_followup_2026-09-24.md).
The dapagliflozin peak is now the original study's directly tabulated Cmax
geometric mean rather than a digitized mean-profile maximum; see
[`development_dapagliflozin_table_followup_2026-09-24.md`](docs/validation/development_dapagliflozin_table_followup_2026-09-24.md).
The OSP silver-arm follow-up quarantined cimetidine and mefenamic acid, and
replaced probenecid with the original-study mean peak; see
[`development_osp_silver_profile_followup_2026-09-24.md`](docs/validation/development_osp_silver_profile_followup_2026-09-24.md).
The original-source check for apixaban, famotidine, and sildenafil corrected
mixed-arm parameters and an approximately rounded Cmax; see
[`development_three_legacy_silver_arms_2026-09-24.md`](docs/validation/development_three_legacy_silver_arms_2026-09-24.md).
The follow-up on acamprosate, ponatinib, posaconazole, and upadacitinib
quarantined an untraceable peak and corrected three primary-study citations;
see [`development_four_more_silver_arms_2026-09-24.md`](docs/validation/development_four_more_silver_arms_2026-09-24.md).
The next source check quarantined alvimopan's five-day twice-daily peak,
converted donepezil's hydrochloride tablet strength to parent mass, and
identified fruquintinib's original oral-suspension arm; see
[`development_alvimopan_donepezil_fruquintinib_2026-09-24.md`](docs/validation/development_alvimopan_donepezil_fruquintinib_2026-09-24.md).
The next source check converted ketorolac tromethamine to free-acid dose,
used the exact single-dose label peak, and identified brincidofovir's unboosted
FDA review control arm; see
[`development_ketorolac_brincidofovir_2026-09-24.md`](docs/validation/development_ketorolac_brincidofovir_2026-09-24.md).
The subsequent check replaced lamivudine's repeated-dose source with a matched
single-dose arm and separated mercaptopurine tablet and suspension results; see
[`development_lamivudine_mercaptopurine_2026-09-24.md`](docs/validation/development_lamivudine_mercaptopurine_2026-09-24.md).
The next check quarantined progesterone's five-day peak and corrected
rifabutin label statistics; see
[`development_progesterone_rifabutin_2026-09-24.md`](docs/validation/development_progesterone_rifabutin_2026-09-24.md).
The direct Cmax model also excludes an indapamide training aggregate traced to
whole-blood measurements; see
[`development_indapamide_training_matrix_2026-09-24.md`](docs/validation/development_indapamide_training_matrix_2026-09-24.md).
The same source-matrix screen quarantined cyclosporine, everolimus, and tacrolimus
training aggregates; see
[`development_immunosuppressant_training_matrix_2026-09-24.md`](docs/validation/development_immunosuppressant_training_matrix_2026-09-24.md).
An administered-drug/analyte identity screen quarantined another 114 Cmax
aggregates pending exact dose-basis adjudication; see
[`development_cmax_administered_identity_2026-09-24.md`](docs/validation/development_cmax_administered_identity_2026-09-24.md).
The pimecrolimus 15 mg label was also traced to blood measurements; see
[`development_pimecrolimus_training_matrix_2026-09-24.md`](docs/validation/development_pimecrolimus_training_matrix_2026-09-24.md).
The voclosporin 0.25 mg/kg label and four exploratory doses were traced to
whole-blood measurements; see
[`development_voclosporin_training_matrix_2026-09-24.md`](docs/validation/development_voclosporin_training_matrix_2026-09-24.md).

> **Reproducibility (2026-09-24).** The table uses public-only TDC fup, Peff, hepatocyte CLint, and Lombardo VDss plus Omega Cmax retrains. Earlier morphine and digoxin reference corrections were followed by an arm-level audit. The current cache excludes unsupported leflunomide and sirolimus arms plus seven prodrug-metabolite labels (adefovir dipivoxil, fesoterodine, molnupiravir, prasugrel, tenofovir disoproxil, valacyclovir, valganciclovir), excludes unsupported abiraterone, atovaquone, clonidine, clozapine, darolutamide, darunavir, glasdegib, itraconazole, pomalidomide, ranolazine, sonidegib, tamsulosin, and vilazodone arms, and uses directly reported paroxetine, nilotinib, clopidogrel, levocetirizine, methylphenidate, norethindrone, carbamazepine, zonisamide, oxybutynin, dasatinib, pindolol, bexagliflozin, sumatriptan, ketoconazole, levofloxacin, and metronidazole parent arms, plus directly measured single-dose cetirizine and febuxostat fasting arms instead of accumulation-adjusted steady-state estimates; the clomipramine label dose is converted from hydrochloride to parent mass. Indomethacin now uses a primary Health Canada fasted 50 mg capsule arm (3.107 mg/L); the former 25 mg value remains unverified. Ketoconazole now uses a primary fasted 200 mg tablet arm (4.22 mg/L) instead of an approximate fed-label value; its 3–4 h postdose meal timing falls short of the external V1 primary rule. Lopinavir, pilocarpine, temozolomide, and venlafaxine are also excluded after exact-arm review; see `docs/validation/development_additional_arm_followup_2026-09-24.md`. A further generic-label screen corrected azithromycin, ciprofloxacin, diclofenac, moxifloxacin, and zolpidem, and excluded isosorbide mononitrate and losartan; see `docs/validation/development_generic_label_arm_followup_2026-09-24.md`. The remaining six generic-label rows had mixed-arm parameters and synthetic curves repaired; see `docs/validation/development_remaining_generic_label_followup_2026-09-24.md`. An OSP source-identity audit corrected cabozantinib, ruxolitinib, and erythromycin; see `docs/validation/development_osp_identity_and_dose_followup_2026-09-24.md`. Acamprosate and phenytoin parent-equivalent doses and phenytoin observed Cmax were corrected; see `docs/validation/development_salt_equivalent_reference_followup_2026-09-24.md`. Per-drug predictions are in `data/training/4track_holdout_predictions.json`; bootstrap intervals are in `data/validation/4track_ci_2026-09-24_audited_reference.json`. Their fitted datasets contain 1,557 fup, 874 Peff, 908 Cmax, 995 CLint, and 1,055 VDss hash-pinned rows. †This repeatedly used development set and its conditional bootstrap CI do not establish independent generalization. Earlier benchmark lineage and numerics-drift measurements are in `docs/research/experiment-log.md`.

The 4-track meta-learner combines mechanistic PBPK (Engine), data-driven XGBoost C<sub>max</sub> (ML), a closed-form CL/F analytical (CLF), and a conditional VDss analytical track. Weights are compound-type-adaptive and were LOOCV-selected on the original N=107 cohort: base compounds blend Engine 0.60 / ML 0.40; other compounds use Engine 0.35 / ML 0.50 / CLF 0.15, with VDss 0.20 added when applicability criteria are satisfied. The current in-domain N=60 slice is descriptive only: applicability flags have not demonstrated reliable error stratification, and neither slice is independent evidence.

**Consumed temporal challenge** (FDA NMEs approved 2024–2025, single-active-ingredient oral small molecules, production-clean at construction, re-scored on the 2026-07-05 engine, N=28). These are historical predictions made before the public-only fup, Peff, Cmax, CLint, and VDss retrains; the set has informed diagnosis and is no longer independent evidence for the current system:

| Slice | AAFE | 95% CI | %2-fold | %3-fold | N |
|---|:-:|:-:|:-:|:-:|:-:|
| All | 3.286 | [2.45, 4.48]† | 25.0% | 50.0% | 28 |
| In-domain | 3.318 | [2.12, 5.39]† | 37.5% | 50.0% | 16 |

On the 2026-07-05 system, the temporal cohort was directionally worse than its same-version development benchmark (3.286 versus 2.743), reversing the earlier favorable N=15 reading. It was assembled by decontaminating previously proposed candidates and expanding FDA-NME discovery before that re-score:

- **Decontaminated.** Three drugs that had leaked into production training were removed: vorasidenib (`clinical_pk.json` gold reference), and aficamten + gepotidacin (`clf_training.csv` → the CLF track, which has no prospective-exclusion filter). A reusable production-aware gate (`scripts/check_prospective_eligibility.py`) now enforces this; it additionally rejected **9 of 26** newly-discovered candidates as already-in-training (e.g. ensartinib in `holdout.json['train']`, deuruxolitinib in `clinical_pk.json`, 7 in `clf_training.csv`). Membership in non-production files (e.g. `mmpk_expanded_*`, `vdss_v2_training` — models the pipeline never loads) is *not* treated as contamination.
- **Expanded.** Exhaustive FDA-NME discovery (101 unique 2024–2025 NMEs across 3 cross-checked sources) → 37 new oral small-molecule candidates → adversarial per-drug Cmax verification (FDA label / EMA EPAR / peer-reviewed PK, ≥2 independent sources agreeing within ~1.5×). Excluded with reasons: 4 verification failures, 7 combination products, 9 production-contaminated, 1 prodrug (sepiapterin, parent-Cmax fold ~3000 — consistent with the prior vadadustat prodrug exclusion). 16 added.

The same-machine re-score moved Meta AAFE 3.208 → 3.286 and Engine AAFE 4.302 → 4.551, while the direct ML track was bit-identical. This supports an absorption/first-pass error diagnosis but does not establish a statistically separated generalization gap. †Compound bootstrap, 100,000 resamples of absolute log-fold error, seed 20260422; diagnostic because the set is consumed. Current artifact: `data/validation/prospective_N28_current_engine_2026-07-05.json`.

**Adaptive-selection caveat.** N=107 has been used for dozens of configuration feedback cycles (track weights, routing, and meta variants), including inspection of the public-only fup, Peff, Cmax, CLint, and VDss candidates. The historical 2.85–3.10 selection-bias sensitivity range has not been recalibrated for this model. The current scored N=73 bootstrap CI ([2.31, 3.54], point estimate 2.841) does not account for adaptive search. The attempted 2026Q2 N50 was invalidated after 21/50 repository-corpus collisions and must not be cited. The replacement strategy is the outcome-blind N=260 protocol in `docs/validation/external_holdout_v1_protocol.md` (N=120 resource-limited fallback; its combined release gate has about 80% pass probability only near a 19% true improvement).

### Historical diagnostic P0 source-adjudicated pilot

This pilot used the earlier DrugBank-trained fup artifact, so its scores do not
measure the current public-only model. All 186 frozen candidates were reviewed against original PK sources before the
sealed predictions were opened. Eighteen compounds (58 arms) met the source
rules; 47 were excluded and 121 remain unresolved. Compound-cluster Meta AAFE
was **3.34** (95% bootstrap CI 2.43–4.79), versus **3.01** (2.26–4.18) for
direct ML. The paired Meta/ML ratio was **1.11** (95% CI 0.93–1.34); only
33.3% of compounds were within twofold for Meta. These results do not show
Meta superiority. Source eligibility was AI-assisted with coordinator checks,
and the clarification of eligibility rules followed first-pass viewing of
source Cmax values. Two scored compounds occur in the current VDss fitted
snapshot; the earlier model's fitted-row membership cannot be reconstructed.
This is diagnostic development evidence, not independent External Holdout V1
or a clinical release gate. See the [P0 result](docs/validation/self_run_p0_result.md)
for hashes, exclusions, and remaining source uncertainty.

### AI-assisted blind P1 diagnostic (earlier public model)

Eight previously unseen compounds were source-screened by a separate Claude
Code worker, and anonymous inputs/predictions were committed before their
Cmax labels were opened. On this small, nonconsecutive, **development-grade**
set, a second source review found one of the eight labels was **whole-blood**
Cmax, not plasma. Excluding that arm, frozen-prediction Meta AAFE is **6.63**
(95% compound-bootstrap CI 3.10–13.84, N=7), versus **5.25** (2.22–13.97)
for direct ML; the paired Meta/ML ratio is **1.26** (0.90–1.75). These
post-label diagnostic values are a serious error signal, not a population
accuracy estimate or a proof of track inferiority. Immediate-release and active-moiety
dose evidence is incomplete, one paper has an inconsistent capsule strength,
and two independent human curators were unavailable. No P1 arm qualifies for
External Holdout V1. See the [P1 diagnostic](docs/validation/blind_p1_result_2026-09-24.md)
for the freeze chain, source cells, sensitivity, and limitations.

### AI-assisted P2 FDA acquisition check

A separate Claude worker consecutively screened all **248 oral FDA NME rows
from 2015–2025** against the repository exclusion union and strict original-arm
criteria. No new eligible compound remained, so no predictions were run.
Repository name collisions removed 235 rows; the two structure-clean parents
that reached original-source arm review lacked stated post-dose meal timing.
A follow-up found an explicit oral-solution arm for one parent but no matching
post-dose fasting statement; both exposed identities are now excluded from a
future blinded cohort. This is a single-agent acquisition check,
not External Holdout V1 evidence; the [P2 screen](docs/validation/blind_p2_fda_acquisition_2026-09-24.md)
records ordered attrition and limitations. A qualifying external cohort needs
other source windows and independent human curation.

### Experimental multi-dose regression checks

Three drugs were simulated at clinical dosing regimens and compared against FDA-label steady-state C<sub>max</sub> values:

| Drug | Regimen | Predicted C<sub>ss,max</sub> (mg/L) | FDA label C<sub>ss,max</sub> (mg/L) | Fold error |
|------|---------|:---:|:---:|:---:|
| Atorvastatin | 40 mg QD | 0.070 | 0.029 | 2.42 |
| Metformin | 500 mg BID | 2.39 | 1.0 | 2.39 |
| Warfarin | 5 mg QD | 0.21 | 1.4 | 0.15 |

(Regenerated on the current engine — post-FLUX-1 + paracellular — via `scripts/verify_v2.py`; supersedes the pre-FLUX-1 values.) Atorvastatin and metformin are each over-predicted by ~2.4×. The historical warfarin comparison suggested severe under-prediction, but its cited Cmax source was later quarantined for a dose/analyte mismatch, so that comparison is not current accuracy evidence. Predicted accumulation ratios tracked the theoretical values (metformin 1.11 vs 1.35; atorvastatin 1.50 vs 1.44; warfarin 1.81 vs 2.94) and steady-state detection operated correctly in all cases. This is a solver-mechanics and limitation-exposure exercise rather than a multi-dose accuracy claim: each compound&rsquo;s single-dose error propagates into its steady-state estimate.

### Experimental TDM regression checks

Bayesian update was tested with a single-drug functional check and a multi-drug synthetic-observation benchmark. The tables below report the **Importance Sampling** baseline (legacy); the experimental SBI/IS/IBIS router (`data/sbi/method_routing.json`) sends 12/13 configured drugs to SBI after simulation-based calibration checks. The reported SBI inference is millisecond-scale and reduced posterior CV in those tests; detailed calibration and per-drug coverage reports are tracked separately. These checks do not establish clinical TDM validity.

The morphine row and pooled summaries below are historical: they used a superseded 30 mg / 18.65 ng/mL reference. See the [reference adjudication](docs/validation/reference_pmid_screen_2026-09-24.md). They require a dose-matched rerun before being used as current TDM evidence.

**Single-drug functional check** (midazolam, 5 mg PO, one observation at t = 1 h, 10% assay CV):

| Metric | Prior | Posterior |
|--------|:-----:|:---------:|
| C<sub>max</sub> CV | 44.3% | 19.8% |
| ESS | &mdash; | 586.6 (29.3% of 2,000) |
| CV reduction | &mdash; | 55.3% |

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
| Full prediction (SMILES &rarr; C<sub>max</sub>) | 414 ms | Prior warm-run benchmark, deterministic, single core |
| ODE solve (full fidelity) | 106 ms | LSODA, rtol=10<sup>&minus;8</sup>, atol=10<sup>&minus;10</sup> |
| ODE solve (MC fast path) | 33 ms | LSODA, rtol=10<sup>&minus;4</sup>, atol=10<sup>&minus;6</sup> |
| MC N=1,000 | 33.5 s | Pure Python RHS (no JIT compilation) |
| RHS evaluation | 31 &mu;s | 54 flux specs per call |

The prior warm-run benchmark supports interactive research screening, but first-call model loading and hardware can change latency. A local macOS/Python 3.10 caffeine check on 2026-09-23 took 1.11 s on the first call and 0.16&ndash;0.17 s on five warm calls. MC propagation at N=1,000 required ~34 s in the prior benchmark due to pure Python ODE evaluation; JIT compilation (e.g., via Numba) is an optimization path not yet pursued.

### Test suite

The full test suite covers graph construction, ODE compilation, flux functions, solver correctness, mass balance, ADME prediction, the Cmax ensemble, dosing research modules, applicability flags, phenotype scaling, leakage guards, and development-benchmark reproducibility. CI output is the authoritative current count; historical pass counts are not used as a scientific validation claim.

**Expected failures (3):** Rosuvastatin and atorvastatin still miss their ECM-forced Cmax gates; the separate axial PGx test deliberately retains a strict expected failure because its well-stirred analytic oracle does not apply to parallel-tube extraction. Fluvastatin now passes its numerical gate, but ECM remains marked not applicable for it in production. Three prodrug clinical gates are skipped in the public clone because their conditional disposition data are absent.

**Test status.** The current public-only fup/Peff/Cmax/CLint/VDss benchmark is pinned by `test_cached_development_aafe_is_2p841`; historical benchmark changes and resolved failures are recorded in `docs/research/experiment-log.md`. The cached headline is reproducible with `scripts/run_engine_benchmark.py` on the pinned public profile.

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
       ┌──────────┼──────────┬──────────┐
       ▼          ▼          ▼          ▼
    regimen      ddi       pkpd       mipd
   (multi-dose, (enzyme    (effect    (engine-as-prior
    TDM)        adj.)     compartment) posterior, MIPD)
```

| Layer | Responsibility | Depends on |
|-------|---------------|------------|
| `graph/` | BodyGraph, node/edge types, YAML builder | `core` |
| `engine/` | ODE compiler, flux registry (including ECM), solver, MC | `core`, `graph` |
| `predict/` | SMILES &rarr; chemistry &rarr; ADME &rarr; DrugOnGraph + transporter DB | `core` |
| `ml/` | XGBoost C<sub>max</sub>, CL/F, VDss predictors, 4-track meta-learner | `core` |
| `pk/` | SimResult &rarr; PKEndpoints (route-aware) | `core` |
| `regimen/` | Multi-dose solver, TDM method dispatch (SBI/IS/IBIS/EnKF), linear-scaling dose adjust | `core`, `engine`, `graph`, `sbi` |
| `sbi/` | Simulation-based inference training + amortized posterior, physiology generator | `core`, `engine` |
| `mipd/` | Engine-as-prior posterior PK: F/CL/renal latents via SIR, covariate individualization (CrCl, weight/age), target-attainment dose recommendation | `core`, `engine`, `graph`, `regimen`, `sbi` |
| `pipeline/` | Orchestrator wiring all layers | all layers |
| `ddi.py` | Drug-drug interactions (competitive inhibition, E<sub>max</sub> induction) | `core`, `graph` |
| `pkpd.py` | PK/PD effect modeling (effect compartment, sigmoid E<sub>max</sub>) | `core` |

Only the structure-only Cmax prediction path is production-supported. `regimen`,
`sbi`, `mipd`, `ddi`, and `pkpd` are experimental research modules and are not
validated for clinical decisions. See `docs/architecture/product_scope.md`.

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

### Experimental drug-drug interactions

Competitive CYP inhibition via pre-simulation enzyme abundance adjustment:

```python
from sisyphus.ddi import apply_inhibition, KETOCONAZOLE

inhibited_graph = apply_inhibition(graph, KETOCONAZOLE)
# Mechanistic scenario only; external clinical-pair validation is required.
```

### Experimental PK/PD modeling

Effect compartment with sigmoid E<sub>max</sub> response, computed as post-processing on the concentration-time profile:

```python
from sisyphus.pkpd import compute_effect, PDModel

pd = PDModel(ke0=0.5, emax=100.0, ec50=0.05, hill=2.0)
effect = compute_effect(sim_result, pd)
```

## Limitations

- **Stereoisomer discrimination is unvalidated.** The direct-ML track uses a chirality-blind Morgan fingerprint, so enantiomers with the same connectivity receive the same ML features. Curated full-InChIKey registries can distinguish specific structures, but their isomer-specific C<sub>max</sub> accuracy has not been independently tested.
- **Evaluated scope is narrower than the engine's route support.** The development C<sub>max</sub> claim concerns oral small-molecule parent drugs. Biologics and non-oral routes have no independent C<sub>max</sub> accuracy evaluation here.
- **Prodrug activation remains experimental.** The seven-substrate registry supports parent-to-active routing, but its three-drug clinical gate still fails (sepiapterin 4748&times;, tebipenem pivoxil 9.05&times;, fostamatinib 4.50&times; in the recorded v3 test). Prodrugs are flagged outside the applicability domain. Clopidogrel parent C<sub>max</sub> remains in the consumed development benchmark; that label does not validate active-metabolite prediction. See `CHANGELOG.md` for mechanism and version history.
- **Simplified pK<sub>a</sub>.** Ionization state is classified by structural rules (carboxylic acid &rarr; 4.5, aliphatic amine &rarr; 9.0), not computed quantum-mechanically. This limits Kp accuracy for highly ionized compounds.
- **Phase II metabolism is incomplete.** NAT2, UGT1A1, UGT1A9, and UGT2B7 have modeled abundances; SULT and other UGT routes remain unmodeled. Parent-drug exposure may be overpredicted when omitted pathways materially contribute to clearance, all else equal. The modeled pathway effects have not been independently validated for clinical dosing.
- **Transporter-mediated disposition: OATP1B1 only.** Hepatic uptake by OATP1B1 is modeled mechanistically via the ECM (closed-form QSSA hepatocyte flux) with per-drug kinetic parameters in `data/transporters/oatp1b1.json`. Other hepatic transporters (OATP1B3, NTCP, BSEP), intestinal transporters (P-gp, BCRP), and renal transporters (OAT1/3, MATE1/2-K) are not mechanistically modeled. P-gp efflux at the gut wall is approximated via a binary permeability correction.
- **CL<sub>int</sub> prediction is limited on the tested data.** The public-only XGBoost model achieved scaffold-CV R&sup2; = 0.215 on 995 TDC Hepatocyte_AZ compounds. Tested representation, dataset, and architecture changes did not reliably improve end-to-end C<sub>max</sub> on the repeatedly used development cohort. This does not prove an intrinsic R&sup2; ceiling or identify assay noise as the sole cause; see the corrected [accuracy diagnosis](docs/research/diagnosis.md). TDM results are internal, observation-conditioned research and do not establish structure-only or clinical dosing accuracy.
- **Novel-drug underprediction has an unresolved mechanism.** Some 2024&ndash;2025 compounds were severely underpredicted in a consumed prospective diagnostic set. The earlier claim that F alone caused these errors is withdrawn: the cited oral CL/F is apparent clearance, not systemic CL, and the ten-drug literature-F comparison mixed incompatible endpoints. Low predicted F and engine&harr;ML divergence did not provide a useful error flag in the tested sets. A matched human oral/IV study and untouched C<sub>max</sub> cohort are needed to distinguish F, clearance, formulation, and other causes; see [diagnosis &sect;8](docs/research/diagnosis.md) and the [F source audit](docs/validation/f_reference_source_audit_2026-09-24.md).
- **Error cancellation constrains component-level improvements.** Development experiments show strong residual correlation among many blending variants and sensitivity to compensating ADME errors. This does not prove the current blend is optimal. Further selection on N=107 would deepen adaptive overfitting; changes must be hypothesis-driven, trained without external labels, and judged once on the blinded external protocol.
- **IV-Cmax observation convention.** For intravenous bolus dosing, engine Cmax is extracted as the maximum concentration over `t ≥ 5 min`, rather than the instantaneous `t = 0` spike. Oral drugs use the full-interval maximum. All 73 currently scored development compounds are oral, so this convention does not affect that AAFE; IV accuracy remains unvalidated.
- **ECM (Extended Clearance Model) generalization unverified for non-statins.** Under the current public profile, pravastatin, pitavastatin, and fluvastatin pass their numerical ECM-forced Cmax gates; rosuvastatin and atorvastatin remain expected failures. Fluvastatin is marked `ecm_applicable=false` in production because its CYP2C9-dominant clearance makes the ECM-forced result biologically inapplicable. A pre-registered generalization test on valsartan + glimepiride (2026-04-22, N=2) returned Mode C with systematic 2.5× underprediction under V3 methodology. The fup override hypothesis was ruled out (DE-33); candidates remain for Jmax calibration, Vss/Kp over-distribution, and ECM architectural limits for Km &gt; 1 µM substrates. Users should not rely on ECM accuracy for non-statin OATP1B1 substrates without independent verification.
- **R<sub>B:P</sub> defaults to 1.0.** The RBP model (R&sup2; = &minus;0.08 on external data) is effectively disabled; all drugs are assumed to have equal blood and plasma concentrations.
- **90% residual interval is not externally calibrated.** The parameter-only Monte Carlo interval covered only 29.9% of N=107 observations at nominal 90%. The user-facing Meta band is a much wider empirical residual quantile whose component predictions are not fully out-of-sample for its calibration records. It is shown only for the public, default-Kp, four-track oral path after its calibration cache and model hashes pass a first-use check; altered partition methods, licensed-profile predictions, track fallbacks, and stale artifacts have no such band. It is not a valid split-conformal guarantee. Independent calibration must use nested OOF predictions or a separate untouched calibration cohort, followed by one-time evaluation on a different blinded holdout.
- **Dose recommendation and TDM are experimental.** Linear dose scaling is rejected automatically for modeled saturable metabolism unless explicitly overridden. That guard prevents a known misuse but does not clinically validate MIPD, TDM, or any posterior interval; these modules remain outside the main product claim.

## Project Structure

```
src/sisyphus/
├── core.py              # Distribution, TissueComposition, data contracts
├── descriptors.py       # Morgan fingerprints + RDKit descriptors
├── compounds.py         # Compound YAML → DrugOnGraph loader
├── ddi.py               # Drug-drug interaction modeling
├── pkpd.py              # PK/PD effect compartment + Emax
├── cli.py               # Cmax CLI + explicitly experimental research commands
├── resources.py         # explicit public/licensed model-data profiles
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
│   ├── result.py        # SimResult dataclass
│   ├── solver.py        # LSODA wrapper (solve, solve_mc)
│   └── uncertainty.py   # Monte Carlo propagation
│
├── predict/             # SMILES → drug parameterization
│   ├── chemistry.py     # Molecular profiling, pKa, AD assessment
│   ├── adme.py          # XGBoost ADME property prediction
│   ├── ivive.py         # In vitro → in vivo extrapolation, Kp
│   ├── hepatic_fu_correction.py # hepatic intracellular fu correction registry
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
│   └── dosing.py        # Linear-scaling dose adjustment (see mipd/ for engine-as-prior MIPD)
│
├── sbi/                 # Amortized neural posterior estimation
│   ├── priors.py        # Drug/physiology prior distributions
│   ├── simulator.py     # Forward model for SBI training
│   ├── amortizer.py     # Neural posterior trainer (NSF)
│   ├── multi_drug.py    # Multi-drug amortizer + continuous (bw, age)
│   ├── physiology_generator.py  # Achour correlated physiology sampler
│   └── sbc.py           # Simulation-Based Calibration gate
│
├── mipd/                # Engine-as-prior posterior PK (model-informed precision dosing)
│   ├── core.py          # F latent + SIR posterior (FPrior, MeasuredF/Cmax/AUC, PosteriorPK)
│   ├── clgrid.py        # CL latent + MeasuredConc via engine clint-scale grid surrogate
│   ├── grid.py          # build_cl_grid (CrCl-aware drug renal_clearance scaling)
│   ├── renal_grid.py    # Renal-CL latent grid (steady-state IV)
│   ├── tdm.py           # predict_tdm — steady-state IV trough TDM (renal-CL latent)
│   ├── covariates.py    # Covariates(crcl_ml_min, ...) + Cockcroft-Gault CrCl estimate
│   ├── dosing.py        # recommend_dose — target-attainment dose recommendation
│   ├── meta.py          # Route the posterior through the meta blend (product interval)
│   ├── api.py           # predict_posterior entry point
│   └── amortizer.py     # (shared amortizer hooks)
│
├── physiology/          # Achour correlated-physiology registry (source package)
│   └── correlation_registry.py  # log-space correlation matrices + correlated-lognormal sampling
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

models/                  # Pre-trained XGBoost models (committed; ~31MB)
```

## Predecessor

Sisyphus inherits curated data assets from [Omega PBPK](https://github.com/jam-sudo/Omega) (591 commits) but not its architecture:

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

If you use Sisyphus in your research, please cite the preprint:

```
Yoon, J. M. (2026). Sisyphus: A Topology-Compiled Physiologically Based
Pharmacokinetic Platform with Structure-Only Input and Bayesian Parameter
Refinement. ChemRxiv. https://doi.org/10.26434/chemrxiv.15004452/v1
```

Software (this repository):

```
Yoon, J. M. (2026). Sisyphus (0.1.0): Graph-based whole-body PBPK
simulation with native uncertainty propagation.
https://github.com/jam-sudo/Sisyphus
```

> The git tag `v1.0.0` (commit `d6ee9a6`) records an earlier feature-branch
> milestone that has been superseded by the current `main` line under a
> more conservative measurement regime. See `CHANGELOG.md` for details.

## Requirements

- Python &ge; 3.10
- numpy, scipy, pyyaml (core)
- rdkit, xgboost, scikit-learn (prediction)

## License

MIT
