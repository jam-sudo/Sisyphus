# Sisyphus Breakthrough Path — Convergent Research Findings

**Date**: 2026-04-04
**Status**: Research complete, implementation pending
**Baseline**: Meta AAFE 2.808 (clean, post-leakage-fix), Oracle 2.076

## Problem Statement

After 33+ methods tested across 24 experiments, Meta AAFE plateaued at the CLint R²=0.24 ceiling. Every post-hoc correction converged to r>0.986 with baseline — **mathematically constrained by the architecture of post-hoc stacking**. The current pipeline factors the problem as:

```
SMILES → CLint_xgb (noisy target, R²=0.24) → engine ODE → Cmax → meta-learner
```

Training ADME models against their own noisy targets guarantees the 0.24 ceiling propagates forward.

## Convergent Evidence (4 independent research clusters)

Four parallel research agents surveyed:
1. **SciML for dynamical systems** (neural ODEs, diff physics, PINNs)
2. **Molecular foundation models** (equivariant GNNs, MACE, MolE, CheMeleon)
3. **Simulation-based inference** (NPE/NLE, BayesFlow, Simformer, flow matching)
4. **Cross-domain simulation** (weather, astro, plasma, robotics, digital twins)

**3 of 4 agents independently converged on the same architectural prescription.**

### Convergent Recommendation: UDE (Universal Differential Equation) over Diffrax

```
dC/dt = engine_RHS(C, θ_physics)  +  NN_θ(drug_embedding, state, t)
         [existing PBPK engine]        [learned residual, trained end-to-end]
```

**Why this works where 33 methods failed**:
- Gradient flows through the stiff ODE solver end-to-end
- θ is optimized against **final Cmax** (the quantity we care about)
- NOT against CLint labels (which have R²=0.24 target noise)
- Mass conservation + monotonicity preserved by ODE structure
- Breaks the "error cancellation lock-in" (post-hoc methods are mathematically constrained; gradient-through-solver is architecturally different class)

**Why this cannot be done with current SciPy LSODA**: SciPy's LSODA is not differentiable. Diffrax provides JAX-native stiff ODE solvers (Kvaerno5, Radau) with full autograd through the adjoint method.

## Negative Convergence: What All 4 Agents Agree to SKIP

- **Foundation models / frozen embeddings for CLint**: MolE (2024 SOTA) gives +1-3% Spearman over XGBoost on TDC ADMET. **Within noise of error-cancellation swings.** Target noise, not representation, is the ceiling.
- **Pure Neural ODE without physics**: Pharos v0 (465K params, 1074 samples) already confirmed data-scarcity at our scale.
- **PINNs / FNO / Neural Operators**: Wrong regime (20-state stiff ODE, not spatial PDE).
- **Docking-based features**: affinity ≠ kcat, already tested (ΔR²=+0.005).
- **Post-hoc meta-learners**: 33 methods already tested, all r>0.986.

## 4-Phase Implementation Roadmap

### Phase 0: Differentiable Engine (Enabler)
**Scope**: Port `engine/solver.py` to JAX/Diffrax Kvaerno5 alongside existing SciPy LSODA.
**Cost**: 2 person-weeks.
**Deliverable**: `engine/solver_jax.py` + validation tests (numerical equivalence to LSODA within 1e-4).
**Success criterion**: All existing integration tests pass with JAX solver as drop-in replacement.
**No changes**: graph topology, compiler, production path.

### Phase 1: UDE / Neural Closure (Primary Breakthrough)
**Scope**: Add learnable residual NN to the CLint computation within the engine ODE. Train end-to-end by backpropagating through Diffrax solver to match holdout Cmax.
**Cost**: 4-5 person-weeks.
**Architecture**:
```
CLint_effective(drug, t) = CLint_xgb(drug) * exp(NN_θ(morgan_fp, state, t))
```
**Regularization** (physiology-informed, from PLOS CB 2024):
- Sign constraint (CLint > 0)
- Magnitude bound (|NN correction| < 2 log units)
- Mass conservation already guaranteed by ODE structure
**Success criterion**: Meta AAFE < 2.5 on 107-drug holdout with proper LOOCV.
**Risk**: Overfit at N=107. Mitigation: start with frozen XGBoost base + small NN residual (<100 params).

### Phase 2: Hierarchical Amortized SBI (Principled Uncertainty)
**Scope**: Train BayesFlow/Simformer to amortize posterior over ADME parameters given SMILES + observed Cmax. Partial pooling across 107 drugs via hierarchical hyperpriors.
**Cost**: 6-8 person-weeks.
**Enabler**: Requires differentiable engine (Phase 0) for training data generation (10⁵-10⁶ synthetic drug simulations).
**Success criterion**: SBC calibration on holdout — 90% posterior predictive CI contains observed Cmax in ~90% of drugs.
**Libraries**: `mackelab/simformer`, BayesFlow 2.x, `sbi` v0.23+.

### Phase 3: EnKF for TDM (Independent, Quick Win)
**Scope**: Replace importance sampling in `regimen/tdm.py` with Ensemble Kalman Filter.
**Cost**: 1-2 person-weeks.
**Independent** of Phases 0-2.
**Addresses documented particle degeneracy**: ketorolac (ESS=2.5), rivaroxaban (ESS=1.0) in v2.1 TDM benchmark.
**Success criterion**: 90% CI coverage ≥90% (vs current 67%), ESS >100 on all holdout drugs.

## Prior Art to Read First (Critical)

Before starting Phase 1, read these 3 papers carefully:

1. **Uni-PK** (Toward Generalizable Data-Driven PK with Interpretable Neural ODEs, JCIM 2025)
   - Closest published analog. If it beats our AAFE 2.81 on similar holdout, strong signal to adopt their architecture.

2. **Dynamic Graph Neural Networks for PBPK Modeling** (arXiv 2510.22096)
   - Literally our graph-native PBPK architecture, learned from data.
   - Risk: direct competitor. Opportunity: if it underperforms, validates our hybrid design.

3. **NeuralGCM** (Nature 632, 2024)
   - Template from weather modeling: differentiable dynamical core + learned closure.
   - Weather ML hit our exact wall in 2018-2020 ("ML doesn't beat numerics"); escaped via this architecture by 2024.

## Key Tools / Libraries

| Tool | Purpose | Phase |
|------|---------|-------|
| Diffrax (Kidger) | JAX stiff ODE + adjoint | 0, 1 |
| JAX | Autograd backbone | 0, 1, 2 |
| sbi (Mackelab) | Amortized SBI (NPE, FMPE, TSNPE) | 2 |
| BayesFlow 2.x | Hierarchical amortized inference | 2 |
| Simformer | All-in-one SBI (transformer+diffusion) | 2 |
| diffeqpy / Stan+Torsten | Validation benchmark for Phase 2 | 2 |

## Go / No-Go Decision Points

### After Phase 0:
- JAX solver matches LSODA within 1e-4 on all test cases? → proceed
- Performance acceptable (< 5x slower than LSODA for single run)? → proceed
- Adjoint gradients computable? → proceed

### After Phase 1 (must hit at least one):
- Meta AAFE < 2.70 (beats baseline by 0.1) → continue to Phase 2
- In-domain AAFE < 2.50 → continue to Phase 2
- Engine AAFE improved while Meta maintained → partial win

### After Phase 2:
- SBC calibration passes → production candidate
- Posterior predictive beats current 90% CI coverage → adopt

### Regardless of Phase 1/2 outcome:
- Phase 3 (EnKF) should be done — it directly fixes a documented bug.

## What This Attempts to Avoid

Lessons from 24 failed experiments encoded in design:
- No swapping individual ADME models (breaks error cancellation)
- No post-hoc scalar corrections (mathematically bounded by r>0.986)
- No foundation model replacements (ceiling is target noise)
- No architectural additions without clear mechanistic justification
- Every new component has a falsifiable success criterion before implementation starts

## Sources

Primary papers (all 2023-2026):
- Uni-PK: Chem. Inf. Model. 2025, `10.1021/acs.jcim.5c02924`
- Dynamic GNN PBPK: arXiv 2510.22096
- NeuralGCM: Kochkov et al., Nature 632 (2024)
- Physiology-informed UDE: Philipps, PLOS Comp Bio 2024
- Stiff Neural ODE implicit solvers: arXiv 2410.05592
- Simformer: arXiv 2404.09636
- Flow Matching Posterior Estimation + simulator feedback: arXiv 2410.22573
- BayesFlow 2.x: bayesflow.org
- Diffrax: github.com/patrick-kidger/diffrax

Code repositories:
- DENG-MIT/KAN-ODEs
- jax-md/jax-md
- Novartis/DeepCt
- mackelab/simformer
