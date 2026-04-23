# Achour 2021 Correlated Hepatic Abundance Priors — Design Spec

**Date:** 2026-04-22
**Status:** Draft, pending user review
**Author:** Hypatia
**Scope:** Replace scalar physiology parameters with correlated population-sampled distributions sourced from Achour et al. 2021 per-donor liver abundance data, for improved SBC calibration.
**Priority:** P3 (pivot from Rank 1 TransPortal after DE-33 architectural blocker identified; approved 2026-04-22).

---

## 0. Problem & Motivation

SBC (simulation-based calibration) currently fails on 11/52 drug-population cells in the continuous hierarchical grid (`project_p4_continuous_complete.md`: 41/52 pass = 78.8%). The SBI simulator (`src/sisyphus/sbi/physiology_generator.py`) produces a **deterministic physiology** given (body_weight, age) — all enzyme/transporter abundances are frozen at their scalar YAML means with `Distribution(..., cv=0)`. MC uncertainty is propagated only through drug-level parameters, not through physiology.

This is biologically implausible. Achour et al. 2021 (PMC7839483, CC BY-NC, 29 matched liver+plasma donors) Table S7 reports hepatic pmol/mg membrane protein abundance with substantial inter-individual variability:

| Enzyme | %CV | | Transporter | %CV |
|---|---|---|---|---|
| CYP3A4 | 76.3 | | P-gp | 60.8 |
| CYP2D6 | 118.5 | | BCRP | 44.7 |
| CYP2C9 | 71.7 | | MRP2 | 47.3 |
| CYP1A2 | 53.3 | | **OATP1B1** | **48.4** |
| CYP2E1 | 44.2 | | | |

Achour 2014 DMD meta-analysis further shows enzymes within the same liver are **correlated** (e.g., CYP3A4 / CYP2C8 r = 0.68), because a donor with high total CYP expression tends to have high expression across most isoforms. Independent MC sampling of each enzyme undercounts joint variance in the physiologically-realistic direction (donors with high clearance across multiple pathways).

The fix is to replace the frozen scalars with a **multivariate lognormal abundance prior**, parameterized from Table S7 per-donor data, activated via an opt-in `rng=` argument to `generate_physiology(...)`.

**Why this addresses SBC failures:** Current SBI simulator draws θ (drug parameters) but ties each simulated posterior to a single physiology. The simulator's simulated posterior is narrower than the data-generating distribution across a real cohort where physiology itself varies. SBC's `cov_dev` metric measures posterior calibration against this implicit oracle; wider simulated-θ coverage per iteration should tighten the discrepancy.

## 1. Goal & Non-Goals

### Goal (revised 2026-04-22 after self-review)

Ship v1b **infrastructure + data-backed prior**: per-enzyme lognormal CV + a correlation matrix over the 5 existing liver CYPs (and OATP1B1 if empirically correlated), at the liver node, sampled when the caller provides an RNG. The primary value of this spec is **enabling** correlated physiology sampling; demonstrating downstream improvements (SBC, MC coverage) is separate work.

**Measurable gates (all blocking on merge):**
1. **Deterministic mean-path equivalence** (Gate A). For every Node.enzyme and Node.transporter in the current YAML, `node.enzymes[tag].mean` after YAML edit is bit-exactly equal to the current scalar value. The 107-holdout engine benchmark, which reads `.mean`, produces AAFE 2.695 ± 0.001.
   - *Note:* `Distribution.cv` does change (0 → Achour CV) — this is by design, so `Distribution` equality is NOT claimed. Only the mean-path and benchmark invariance.
2. **Marginal distribution calibration** (Gate B). 10,000 draws from `sample_correlated` reproduce Achour Table S7 CV within ±5% per target.
3. **Joint correlation calibration** (Gate C). 10,000 draws reproduce the stored log-space correlation matrix within ±0.05 per off-diagonal entry.
4. **Cancer-bias sensitivity robustness** (Gate D, NEW). Run Gates B and C with a secondary configuration where Achour CV is scaled down by 50% (healthy-liver proxy, no public healthy cohort available for direct comparison). Both configurations must pass Gates B and C. Selection of which configuration becomes the operational prior is deferred to the P4.5a follow-up spec (§7.2).

### Non-Goals

- **SBC amortizer retraining** — the existing SBI amortizer was trained with frozen physiology. Changing the simulator to sample physiology requires retraining the amortizer for SBC to be a meaningful metric of prior improvement. That retraining is a separate, larger phase (P4.5 scope) and lives in its own follow-up spec. **This spec does not claim SBC improvement.** (§7 Follow-up work documents the downstream plan.)
- UGTs (not yet modeled in `reference_man.yaml` enzymes; out of scope, deferred to v2)
- Intestinal enzymes (gut node has its own CYP3A4 entry; handled in v2 once liver behavior is validated)
- Efflux transporters at intestine (P-gp, BCRP, MRP2 — Achour has data but not instantiated in Sisyphus)
- Non-CYP CYP3A5, CYP2C19, CYP2A6 (present in Achour S7 but not in Sisyphus reference_man.yaml)
- Jmax/Km correlation within drug transporter kinetics (this is a per-drug concern, separate from per-node abundance)
- Mean recalibration against Achour (document the 13.5× discrepancy; keep current calibrated means)
- DE-33 fix (architecturally separate; Path X targets physiology infrastructure, not AAFE)
- CYP2D6 bimodal/mixture modeling (CYP2D6 CV 118.5% is dominated by PM tail; lognormal is a known misfit for this target specifically — see R8. v1b uses uniform lognormal across all 5 CYPs; mixture distribution is v2 scope.)

---

## 2. Architecture

### 2.1 Distribution class extension

Current (`src/sisyphus/core.py`):

```python
@dataclass(frozen=True)
class Distribution:
    mean: float
    cv: float = 0.0
```

New (additive, zero-breakage):

```python
@dataclass(frozen=True)
class Distribution:
    mean: float
    cv: float = 0.0
    correlation_group: str | None = None  # NEW: opt-in marker
```

Semantics:
- `correlation_group=None` (default): independent sampling, unchanged.
- `correlation_group="<name>"`: sample jointly with all other Distributions sharing the same group, using the log-space correlation matrix registered for `<name>`.

### 2.2 Correlation registry

New module `src/sisyphus/physiology/correlation_registry.py`:

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CorrelationSpec:
    members: tuple[str, ...]            # e.g., ("CYP3A4","CYP2D6","CYP1A2","CYP2C9","CYP2E1","OATP1B1")
    log_corr_matrix: np.ndarray         # shape (N,N), Pearson correlation on log-transformed per-donor data

_REGISTRY: dict[str, CorrelationSpec] = {}

def register(name: str, spec: CorrelationSpec) -> None: ...
def get(name: str) -> CorrelationSpec | None: ...
def load_from_json(path: Path) -> None: ...   # idempotent, called at import-time
```

On import of `sisyphus.physiology`, `load_from_json("data/physiology/achour2021_correlation.json")` populates the registry for group name `"liver_achour2021"`.

### 2.3 Correlated lognormal sampler

New function `sisyphus.physiology.correlation_registry.sample_correlated()`:

```python
def sample_correlated(
    means: np.ndarray,          # shape (N,), positive
    cvs: np.ndarray,            # shape (N,), non-negative
    log_corr: np.ndarray,       # shape (N,N), log-space correlations
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one sample of N correlated lognormal variates.

    For each i:  log(X_i) ~ Normal(mu_i, sigma_i)
    where sigma_i^2 = log(1 + cv_i^2), mu_i = log(mean_i) - sigma_i^2/2.

    Joint structure: corr(log X_i, log X_j) = log_corr[i,j].

    Mean/var of X_i on the raw scale reproduce `means[i]`/cv_i exactly.
    """
    sigmas = np.sqrt(np.log1p(cvs**2))
    mus    = np.log(means) - 0.5 * sigmas**2
    cov    = log_corr * np.outer(sigmas, sigmas)
    z      = rng.multivariate_normal(mean=np.zeros(len(means)), cov=cov)
    return np.exp(mus + z)
```

**Critical detail:** the input correlation matrix must be **on the log scale** (correlation of `log(x)`, not of `x`). Computed directly from per-donor data by taking `corr(log(donor_values + epsilon))`; see §3.2.

### 2.4 generate_physiology integration

Add optional `rng` parameter to `src/sisyphus/sbi/physiology_generator.py::generate_physiology(...)`:

```python
def generate_physiology(
    body_weight_kg: float,
    age_years: float,
    base_yaml: Path | None = None,
    rng: np.random.Generator | None = None,   # NEW
) -> BodyGraph:
    ref = build_from_yaml(base_yaml or _DEFAULT_YAML)
    # ... existing allometric + ontogeny scaling ...

    # NEW: if rng provided, re-sample abundances at each node for every group
    if rng is not None:
        g = _resample_correlated_abundances(g, rng)

    return g
```

The helper `_resample_correlated_abundances(graph, rng)` walks each node, groups Distributions by `correlation_group`, and replaces their means with a single multivariate-lognormal draw per group. Distributions with `correlation_group=None` are left unchanged (their stochasticity comes from other MC machinery — DrugOnGraph.sample — if any).

Pseudo-code (the plan phase expands this into tested code):

```python
def _resample_correlated_abundances(
    graph: BodyGraph, rng: np.random.Generator
) -> BodyGraph:
    g = BodyGraph()  # rebuilt with sampled means
    for name, node in graph.nodes.items():
        # Gather Distributions by correlation_group for this node
        groups: dict[str, list[tuple[str, str, Distribution]]] = {}
        # items are (kind, tag, dist) where kind in {"enzyme", "transporter"}
        for tag, d in node.enzymes.items():
            if d.correlation_group:
                groups.setdefault(d.correlation_group, []).append(("enzyme", tag, d))
        for tag, d in node.transporters.items():
            if d.correlation_group:
                groups.setdefault(d.correlation_group, []).append(("transporter", tag, d))

        new_enzymes = dict(node.enzymes)
        new_transporters = dict(node.transporters)

        for group_name, entries in groups.items():
            spec = correlation_registry.get(group_name)
            if spec is None:
                raise KeyError(f"Unknown correlation_group {group_name!r}")
            # Reorder entries to match spec.members order
            by_tag = {tag: (kind, d) for (kind, tag, d) in entries}
            means = np.array([by_tag[m][1].mean for m in spec.members])
            cvs   = np.array([by_tag[m][1].cv   for m in spec.members])
            sampled = sample_correlated(means, cvs, spec.log_corr_matrix, rng)
            for i, member_tag in enumerate(spec.members):
                kind, old_d = by_tag[member_tag]
                new_d = Distribution(
                    mean=float(sampled[i]), cv=0.0, correlation_group=None
                )
                if kind == "enzyme":
                    new_enzymes[member_tag] = new_d
                else:
                    new_transporters[member_tag] = new_d

        g.nodes[name] = dataclasses.replace(
            node, enzymes=new_enzymes, transporters=new_transporters
        )
    g.edges = list(graph.edges)
    g.global_params = dict(graph.global_params)
    return g
```

Key property: the sampled Distributions have `cv=0` after draw — each iteration represents one physiologically-realized individual. Subsequent MC code paths that read `.mean` get the sampled value; code paths that call `.sample(rng)` get that value back deterministically. This decouples population-level sampling (here) from any within-individual MC (DrugOnGraph.sample).

### 2.5 YAML schema extension

`data/physiology/reference_man.yaml` liver node today:

```yaml
- name: liver
  enzymes:
    CYP3A4: 9247500
    CYP2D6: 675000
    CYP1A2: 3037500
    CYP2C9: 6480000
    CYP2E1: 3307500
  transporters:
    OATP1B1: 5.0e5
```

After this spec:

```yaml
- name: liver
  enzymes:
    CYP3A4: {mean: 9247500, cv: 0.763, correlation_group: liver_achour2021}
    CYP2D6: {mean:  675000, cv: 1.185, correlation_group: liver_achour2021}
    CYP1A2: {mean: 3037500, cv: 0.533, correlation_group: liver_achour2021}
    CYP2C9: {mean: 6480000, cv: 0.717, correlation_group: liver_achour2021}
    CYP2E1: {mean: 3307500, cv: 0.442, correlation_group: liver_achour2021}
  transporters:
    OATP1B1: {mean: 5.0e5, cv: 0.484, correlation_group: liver_achour2021}
```

`_parse_distribution` in `src/sisyphus/graph/builder.py` already handles `{mean, cv}` for every enzyme/transporter parsing call site. Only addition: extract `correlation_group` if present (default None). Zero impact on any YAML that omits the new field.

### 2.6 Backwards compatibility

| Callsite | Before | After |
|---|---|---|
| `generate_physiology(bw, age)` | returns deterministic graph | returns identical deterministic graph (rng=None) |
| `generate_physiology(bw, age, rng=rng)` | (new) | returns sampled graph |
| `build_from_yaml(...)` with scalar enzyme YAML | returns `Distribution(mean=X, cv=0)` | unchanged |
| `build_from_yaml(...)` with `{mean, cv}` YAML | returns `Distribution(mean, cv, correlation_group=None)` | unchanged |
| `DrugOnGraph.sample(rng)` | samples drug distributions independently | unchanged — drug correlations out of scope |
| 107-holdout benchmark Meta AAFE | 2.695 | 2.695 ± 0.001 (mean-path preserved; see Gate A §4.1) |

**Invariant 3 (compile once) preserved:** topology unchanged, only parameter means shuffle across MC draws. Engine compilation is untouched.

**Invariant 1 (identity-blindness) preserved:** correlation_group is a string tag; engine never inspects it. All sampling logic lives in `sisyphus.physiology`, not in `sisyphus.engine`.

---

## 3. Data Curation

### 3.1 Source

**Achour et al. 2021 Clinical Pharmacology & Therapeutics, "Liquid Biopsy Enables Quantification of the Abundance and Interindividual Variability of Hepatic Enzymes and Transporters," PMC7839483, DOI 10.1002/cpt.2013.**

License: **CC BY-NC 4.0** (per PMC OA API, 2024-08-05 update). Attribution required; commercial use restricted (compatible with Sisyphus's research posture).

Supplementary file `CPT-109-222-s001.pdf` obtained via Europe PMC REST API (`https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7839483/supplementaryFiles`).

### 3.2 Extraction pipeline

New script `scripts/extract_achour2021_abundance.py`:

1. Parse Table S7 (page 22) from the supplementary PDF — 29 donors × 16 targets in pmol/mg membrane protein. Missing values are `-` in PDF (undetected proteins).
2. Restrict to **candidate targets** matching Sisyphus's current liver node inventory: CYP3A4, CYP2D6, CYP1A2, CYP2C9, CYP2E1 (5 CYPs) + OATP1B1.
3. Construct two datasets:
   - **CYP-only** (5 targets): drop rows where any of the 5 CYPs is missing. Primary correlation matrix for the 5-enzyme group.
   - **CYP+OATP1B1** (6 targets): drop rows where any of the 6 is missing. Secondary matrix for empirical OATP1B1 inclusion test.
4. **Report N_complete for both datasets** — plan phase and data-artifact tests must record these values. Minimum acceptable: N ≥ 15 for 5-way, N ≥ 15 for 6-way. If either < 15, plan phase escalates (spec pre-commit: N is reported in `data/physiology/achour2021_correlation.json`; merge is gated on N ≥ 15).
5. Write per-donor CSV `data/physiology/achour2021_liver_abundance.csv` with ALL 6 candidate columns + donor_id (rows with any value populated, dashes preserved as NaN — let downstream code decide completeness). CC BY-NC allows redistribution with attribution.
6. Compute `log_values = log(per_donor_values)` separately on the CYP-only (5 cols) and CYP+OATP1B1 (6 cols) complete subsets.
7. Compute Pearson correlation on each `log_values` matrix.
8. **OATP1B1 inclusion decision rule** (empirical, evaluated in extraction script):
   - Compute `mean_r_OATP_to_CYPs` = mean of 5 pairwise log-Pearson correlations between OATP1B1 and each CYP (using the CYP+OATP1B1 complete subset).
   - If `|mean_r_OATP_to_CYPs| ≥ 0.3`: OATP1B1 joins `liver_achour2021` group → final matrix is 6×6, YAML marks OATP1B1 with `correlation_group: liver_achour2021`.
   - If `|mean_r_OATP_to_CYPs| < 0.3`: OATP1B1 is independent → final registered matrix is 5×5, YAML sets OATP1B1 `{mean: 5.0e5, cv: 0.484}` (no group → independent lognormal sampling).
   - The decision + computed `mean_r_OATP_to_CYPs` value is recorded in the JSON artifact's `"oatp1b1_inclusion"` object.
9. Compute per-column mean and CV on raw scale (for cross-check against Table S7's reported means/CVs; Gate B verifies).
10. **CYP2D6 bimodality check** — compute log(CYP2D6) per-donor, run a simple bimodality diagnostic (Hartigan's dip test or the trivial `Δ(empirical CDF, lognormal CDF)` max-distance test at donors ordered log-ascending). Record test statistic in the JSON. This is an audit signal, not a blocker — v1b proceeds with lognormal regardless, but v2 will revisit if bimodality is confirmed.
11. **PSD projection** — if the log correlation matrix has any eigenvalue < 0 from finite-sample noise, project to the nearest PSD matrix (shift eigenvalues by a small epsilon), record the projection magnitude in the JSON.
12. Write `data/physiology/achour2021_correlation.json`:
   ```json
   {
     "name": "liver_achour2021",
     "source": "Achour 2021 CPT Table S7, PMC7839483 (CC BY-NC 4.0)",
     "cohort_note": "27/29 donors are cancer patients; 2 are non-cancer liver disease. No healthy-liver cohort; cancer-bias sensitivity Gate D addresses this.",
     "n_donors_complete": <int, computed>,
     "n_donors_complete_cyp_only": <int, computed>,
     "members": ["CYP3A4","CYP2D6","CYP1A2","CYP2C9","CYP2E1", "OATP1B1"?],
     "cv": [0.763, 1.185, 0.533, 0.717, 0.442, 0.484?],
     "log_corr_matrix": [[1.0, r12, ...], ...],
     "oatp1b1_inclusion": {
       "decision": "joined" | "independent",
       "mean_r_OATP_to_CYPs": <computed float>,
       "threshold": 0.3
     },
     "cyp2d6_bimodality": {
       "dip_statistic": <float>,
       "lognormal_fit_warning": <bool>
     },
     "psd_projection_applied": <bool>,
     "psd_projection_shift": <float>
   }
   ```

The CSV and JSON are both authoritative artifacts committed to the repo. Regeneration is deterministic from the unchanged Achour PDF.

### 3.3 Mean reconciliation — explicit non-action

Achour 2021 Table S8 reports CYP3A4 tissue content mean = 0.455 fmol/μg liver tissue → 0.455 pmol/mg × 1500 g = 682,500 pmol per liver.

Sisyphus current: 9,247,500 pmol per liver (13.5× higher), derived from Rodgers/Rowland literature defaults and pravastatin+midazolam calibration.

**Decision:** keep Sisyphus means. Swapping means would invalidate Phase 1-2A calibrations, break the 107-holdout AAFE 2.695 gate, and expand scope. The 13.5× gap is partly explained by:
- Achour cohort: 27/29 are cancer patients (adjacent liver, pathology may suppress expression)
- Microsomal vs membrane-fraction methodology differences (Achour QconCAT membrane; Sisyphus uses microsomal meta-analytic values)

This decision is explicitly recorded in `data/physiology/achour2021_correlation.json` as a `"notes"` field. A later spec can revisit mean recalibration; this one does not.

### 3.4 CV source choice

Use Achour Table S7 %CV values **as-is** for the 6 targets. These are measured on the raw scale (not log). For lognormal sampling we need log-space sigma:

```
sigma_log = sqrt(log(1 + cv_raw^2))
```

For CYP3A4 (CV=0.763): `sigma_log = sqrt(log(1+0.582)) = sqrt(0.459) = 0.678`. The sampler in §2.3 does this conversion internally; YAML stores raw CV for transparency.

### 3.5 Correlation matrix stability

With N≈25 complete donors (after dropping rows with missing values) and 6 targets, the 15 unique off-diagonal entries have standard error ~1/√N ≈ 0.20 per entry. Entries with |r| < 0.25 should not be treated as distinguishable from zero; the data artifact will report bootstrap 95% CI per entry for auditability. Spec does not require any single entry to be significant, only that the global matrix pattern is used.

---

## 4. Validation Gates

All gates are blocking on merge.

### 4.1 Deterministic mean-path equivalence (Gate A)

Regression test: for every Node.enzyme tag and Node.transporter tag in the current `reference_man.yaml`, after the YAML edit, `node.enzymes[tag].mean` (resp. `node.transporters[tag].mean`) equals the pre-edit scalar value exactly (bit-identical float).

Note: `Distribution.cv` is intentionally allowed to change (0 → Achour CV); `Distribution.correlation_group` is allowed to change (None → `"liver_achour2021"`). The `Distribution` object is NOT equal pre/post; this is by design.

Regression test: 107-holdout engine benchmark (`scripts/run_engine_benchmark.py`) with `rng=None` path (deterministic mean-path) → Meta AAFE 2.695 within ±0.001. Automated via cached prediction JSON diff.

This gate catches any downstream code that accidentally starts sampling from the new Distribution.cv > 0 under deterministic-mode assumptions.

### 4.2 Marginal distribution calibration (Gate B)

Unit test: draw N=10,000 samples from `sample_correlated(...)` for the registered liver group (5 or 6 members depending on OATP1B1 inclusion). For each target:
- Empirical mean within ±1% of Distribution.mean
- Empirical CV within ±5% relative (of Distribution.cv)
- All samples positive (lognormal property)

Also runs with the **cancer-bias sensitivity configuration** (CV × 0.5) as Gate B'.

### 4.3 Joint correlation calibration (Gate C)

Unit test: N=10,000 draws, compute empirical log-space correlation matrix, compare to stored `log_corr_matrix`:
- Every off-diagonal entry within ±0.05 absolute
- Diagonal exactly 1.0

Also runs with the cancer-bias sensitivity configuration as Gate C'.

### 4.4 Cancer-bias sensitivity (Gate D, NEW)

Achour 2021 cohort is 27/29 cancer patients (Table S1). Cancer liver may have altered expression variance relative to healthy population. There is no public per-donor healthy-liver abundance dataset with inter-enzyme correlations at matched granularity; Achour 2014 meta-analysis provides per-enzyme means but not per-donor values.

In lieu of a direct healthy-cohort test, we require the implementation to support a **sensitivity configuration**: a parallel `liver_achour2021_healthy_proxy` registration with CV scaled by 0.5 (a conservative healthy-proxy assumption based on the fact that pathology tends to inflate variance) and the same correlation matrix. Both configurations must pass Gates B and C.

Gate passes if:
- Both configurations register successfully.
- Gates B and C pass for both.
- The JSON artifact documents the healthy-proxy configuration.

This is explicitly NOT a claim that 0.5× CV is the correct healthy CV. It is a robustness check ensuring the machinery handles different CV levels without arithmetic pathology. Downstream follow-up work (the SBC retraining spec) will determine which configuration is the correct prior.

### 4.5 Data-artifact provenance (Gate E, NEW — replaces old Gate E forward-compat test which moves to §5.1)

The extraction script `scripts/extract_achour2021_abundance.py` is deterministic on the downloaded supplementary PDF (same PDF hash in, same JSON/CSV out). A checksum of the CSV is recorded in a comment line in the JSON. Any future re-run that produces a different CSV invalidates the committed JSON — a CI test confirms this correspondence.

---

## 5. Test Plan

### 5.1 Unit tests (new file `tests/unit/test_correlated_abundance.py`)

1. `test_distribution_default_correlation_group_none` — backward compat (replaces old Gate E test #1).
2. `test_parse_distribution_reads_correlation_group` — YAML parsing (replaces old Gate E test #2).
3. `test_parse_distribution_bare_scalar_unchanged` — bare int/float still parses to `Distribution(mean, 0, None)`.
4. `test_sample_correlated_marginals_match_cv` — Gate B.
5. `test_sample_correlated_marginals_healthy_proxy` — Gate B' (CV × 0.5 configuration).
6. `test_sample_correlated_recovers_log_corr_matrix` — Gate C.
7. `test_sample_correlated_recovers_log_corr_healthy_proxy` — Gate C'.
8. `test_sample_correlated_all_positive` — lognormal property.
9. `test_sample_correlated_degenerate_identity` — log_corr=I → independent sampling.
10. `test_registry_load_and_get` — registry loading.
11. `test_registry_missing_group_raises` — `_resample_correlated_abundances` raises `KeyError` on unknown group (defensive).
12. `test_assert_sampled_passes_after_resample` — R5 helper: asserts no correlation_group survives sampling.
13. `test_assert_sampled_fails_on_unsampled_group` — R5 helper: detects forgotten `rng=` argument.

### 5.2 Integration tests (new file `tests/integration/test_physiology_sampling.py`)

1. `test_generate_physiology_deterministic_without_rng` — Gate A mean-path check.
2. `test_generate_physiology_sampling_with_rng_different_draws` — two seeds differ, both positive, both obey `assert_sampled`.
3. `test_generate_physiology_sampling_scales_with_bw_age` — sampled graph for (45kg, 5y pediatric) has means consistent with maturation × BW scaling applied to sampled values.
4. `test_sampled_enzymes_follow_correlation_group` — empirical corr matches stored matrix within tol across 1000 draws end-to-end through `generate_physiology`.
5. `test_oatp1b1_inclusion_honored` — if JSON records `decision: joined`, OATP1B1 is sampled in same group; if `independent`, OATP1B1 sampling is empirically uncorrelated with CYPs across 1000 draws.

### 5.3 Regression tests (append to existing)

1. `tests/integration/test_holdout_regression.py` (new or extend): Meta AAFE 2.695 ± 0.001 with deterministic physiology. This is Gate A's benchmark side.
2. `tests/unit/test_builder_yaml_scalar_backward_compat.py`: loading a YAML with bare-scalar enzymes (no mean/cv dict) still produces identical Node structure (protects every other physiology YAML that has not been migrated).

### 5.4 Data artifact tests (new file `tests/unit/test_achour_data_artifact.py`)

1. CSV has ≥29 total donor rows (one row per donor in Table S1, incl. rows with NaNs).
2. JSON `n_donors_complete` ≥ 15 (minimum for a 5- or 6-way Pearson matrix; merge gate per §3.2).
3. JSON log_corr_matrix is symmetric, PSD (eigvals ≥ -1e-9 pre-projection; ≥ 0 post-projection), diagonal = 1.
4. JSON raw-scale CVs match Table S7 reported values within ±2%.
5. JSON records `oatp1b1_inclusion.decision` ∈ {"joined", "independent"}.
6. JSON records `cyp2d6_bimodality.dip_statistic` (may flag misfit; not a merge blocker).
7. JSON CSV checksum matches the committed CSV (data-artifact provenance per Gate E).

### 5.5 Test count delta

New unit: +13 · New integration: +5 · Regression: +2 · Data: +7 = **+27 tests**. Target: **≥475 pass** (current 448).

---

## 6. Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| R1 | SBC amortizer was trained with frozen physiology; changing simulator to sample physiology invalidates amortizer. | **Removed as risk** — now explicitly a Non-Goal (§1). SBC retraining is a follow-up spec (§7). This spec does not claim SBC improvement. |
| R2 | Cancer-patient donor bias in Achour inflates/distorts CV relative to the healthy population our SBI is supposed to calibrate against. | Medium | **High** | (1) §3.3 documents cohort composition. (2) Gate D requires the machinery to support a 0.5× CV healthy-proxy configuration in parallel. (3) Downstream SBC retraining spec selects which configuration becomes the operational prior. |
| R3 | N<30 donor correlation matrix has wide entry-wise SE (~1/√N ≈ 0.20) and may not be PSD after finite-sample estimation. | Medium (PSD) / Certain (wide SE) | Medium | (1) Extraction script uses `scipy.linalg.eigh` + nearest-PSD projection if any eigenvalue < 0. Projection shift reported in JSON. (2) Entries with \|r\| < 0.25 should be read as "not distinguishable from zero" — plan phase documents this caveat in the data artifact. (3) Minimum N ≥ 15 is a merge gate (§3.2 step 4). |
| R4 | 13.5× mean discrepancy between Achour and Sisyphus signals deeper calibration issue. | Known | Blocked | Documented in §3.3 as deliberate non-action. Mean recalibration is Non-Goal. |
| R5 | `rng` argument is forgotten at some SBI callsite, producing silent determinism (sampling intended but not actually performed). | Medium | Medium | Add `sisyphus.physiology.assert_sampled(graph)` helper that inspects a BodyGraph and asserts that no node has a Distribution with `correlation_group != None AND cv > 0` (i.e., all grouped entries have been collapsed to sampled `cv=0` entries). SBI simulator entrypoint calls this after physiology generation; test coverage in §5. |
| R6 | "Byte-identical" in Gate A was impossible due to Distribution.cv and correlation_group changes. | Known | Resolved | Gate A rewritten (§4.1) to require mean-path equivalence only. |
| R7 | OATP1B1 biologically does not co-regulate with CYPs; forcing it into the same correlation group produces a biologically implausible joint prior. | Medium | Low | Extraction script empirically decides via `|mean r| ≥ 0.3` threshold (§3.2 step 8). If OATP1B1 fails the threshold, it is registered as independent. Threshold value and decision recorded in JSON. |
| R8 | CYP2D6 is clinically bimodal (PM 5-10% of Caucasians, near-zero activity); single lognormal undersamples the PM tail. | High (known clinical fact) | Medium | (1) §3.2 step 10 runs a bimodality diagnostic in extraction and records it as an audit flag. (2) CYP2D6 mixture modeling is explicit v2 scope (§1 Non-Goals). v1b proceeds with lognormal knowing this is a misfit; the bimodality diagnostic makes the misfit measurable. (3) The cancer-bias sensitivity configuration (Gate D) partially compensates by offering a narrower CV alternative. |
| R9 | CV is assumed invariant with age/BW (same %CV for a 5-year-old pediatric liver and a 30-year-old adult). Achour measured only adults. | Low | Low | Documented assumption. Pediatric SBC failures that are specifically CV-driven would invalidate this; no explicit gate, but `project_p4_continuous_complete.md` tracks per-pop SBC so pediatric-vs-adult pattern is observable as auxiliary output. |

---

## 7. Rollback Plan & Follow-up Work

### 7.1 Rollback

All changes are additive and opt-in:
- `correlation_group` field defaults to None (zero effect without YAML change)
- `generate_physiology(..., rng=None)` is the default (zero sampling)
- Data files `achour2021_liver_abundance.csv` and `achour2021_correlation.json` are leaf artifacts; removing them disables sampling with a clean KeyError on registry lookup

Rollback = revert the YAML edit in `reference_man.yaml` (delete `cv` and `correlation_group` fields), delete the two data files. Code changes remain dormant.

Branch target: `feat/achour-correlated-abundance` off `main`. Merge after Gates A-E all pass automated.

### 7.2 Follow-up work (separate spec, not this one)

This spec ships the **prior** and the **sampling machinery**. It does not demonstrate a downstream benefit metric. The expected follow-up sequence:

1. **P4.5a — SBI amortizer retrain with correlated physiology.** New spec in `docs/superpowers/specs/` that:
   - Regenerates SBI training data with `generate_physiology(..., rng=sbi_rng)` injected into the simulator.
   - Trains a new amortizer on the enriched simulator.
   - Re-runs SBC on the 52 drug-population grid.
   - Gate: SBC pass-rate improves from baseline 41/52. Magnitude of improvement is a research output of that spec, not a prediction here.
   - Decision point: use 1× CV (Achour raw) or 0.5× CV (healthy-proxy) as operational prior, per Gate D' results.

2. **P4.5b — Expansion** (only if P4.5a succeeds):
   - UGTs (add to `reference_man.yaml`, add Achour rows for UGT1A1/1A9/2B4/2B7 to the group).
   - CYP2C19/2A6/3A5 (add nodes, add to group).
   - Intestinal enzymes + transporters.
   - CYP2D6 mixture modeling (address R8).

This section exists so that the scope restriction is paired with an explicit continuation plan; it is NOT part of v1b implementation.

---

## 8. File Touch Summary

| File | Change | Lines (est) |
|---|---|---|
| `src/sisyphus/core.py` | Distribution: add `correlation_group` field | +3 |
| `src/sisyphus/graph/builder.py` | `_parse_distribution`: read correlation_group | +4 |
| `src/sisyphus/physiology/__init__.py` | new package | +5 |
| `src/sisyphus/physiology/correlation_registry.py` | new: registry + sampler | +120 |
| `src/sisyphus/sbi/physiology_generator.py` | accept `rng`; call `_resample_correlated_abundances` | +25 |
| `data/physiology/reference_man.yaml` | liver enzymes/transporters → dict with cv + group | +12 |
| `data/physiology/achour2021_liver_abundance.csv` | new data artifact | +30 |
| `data/physiology/achour2021_correlation.json` | new data artifact | +30 |
| `scripts/extract_achour2021_abundance.py` | new extraction pipeline | +120 |
| `tests/unit/test_correlated_abundance.py` | new | +220 |
| `tests/integration/test_physiology_sampling.py` | new | +140 |
| `tests/unit/test_achour_data_artifact.py` | new | +90 |
| `tests/integration/test_holdout_regression.py` | extend or new | +40 |
| `tests/unit/test_builder_yaml_scalar_backward_compat.py` | new | +40 |

**Total: ~880 lines across 14 files.**

---

## 9. Self-Review

### 9.1 Revision history (v1 → v1-revised, 2026-04-22)

Critical self-review identified 3 critical flaws in the initial v1 draft; this v1-revised addresses them:

| Flaw | v1 (original) | v1-revised |
|---|---|---|
| **F1 SBC hypothesis** | Gate D claimed SBC pass-rate would improve 41→≥45/52. | SBC improvement is now an explicit **Non-Goal** (§1). Gate D removed. Follow-up SBC retraining spec is referenced in §7.2. Primary value of spec reframed as infrastructure + prior. |
| **F2 Gate A "byte-identical"** | Claimed BodyGraph would be byte-identical in deterministic mode. | Literally false — Distribution.cv and correlation_group change. Gate A rewritten to "deterministic mean-path equivalence": `.mean` values bit-exact, 107-holdout AAFE invariant, Distribution equality explicitly NOT claimed. |
| **F3 Cancer-bias impact** | R2 "impact: low". | R2 "impact: HIGH" — 27/29 cancer donors. New Gate D (cancer-bias sensitivity) requires machinery to support 0.5× CV healthy-proxy configuration. Selection of operational CV deferred to P4.5a follow-up spec. |

Also addressed:
- **F4 CYP2D6 bimodal**: added as explicit Non-Goal (mixture modeling v2), with extraction-time bimodality diagnostic recorded in JSON (R8).
- **F5 Complete-donor N**: `n_donors_complete ≥ 15` is now a merge gate (§3.2 step 4).
- **F6 OATP1B1 inclusion**: empirical, `|mean r| ≥ 0.3` threshold decided in extraction script (§3.2 step 8).
- **F7 CV age/weight invariance**: documented as R9 with observable pediatric-vs-adult SBC pattern.
- **F8 helper pseudo-code**: added to §2.4.
- **F9 redundant Gate E**: old forward-compat gate moved into §5.1 unit tests; new Gate E (data-artifact provenance) introduced.

### 9.2 Placeholder scan

None. All values concrete except the N_complete and OATP1B1-inclusion decision, both of which are parameterized outputs of the extraction script with explicit merge gates.

### 9.3 Internal consistency

- §2.3 input `log_corr` matches §3.2 step 7 computation (log-transformed per-donor → Pearson) ✓
- §2.5 YAML CVs match §0 table which match Table S7 (verified against extracted PDF) ✓
- §3.2 OATP1B1 inclusion decision matches §2.5 YAML syntax (grouped vs ungrouped) ✓
- §4 Gates (A, B, B', C, C', D, E) match §5 Test Plan (deterministic, marginal, joint, sensitivity, provenance) ✓
- §6 R1 marked "Removed" consistent with §1 SBC as Non-Goal ✓
- §7.2 Follow-up P4.5a links back to SBC retraining referenced in §1 ✓

### 9.4 Scope check

Single subsystem (physiology abundance prior + infrastructure). Correctly scoped now that SBC demonstration is excluded. Orthogonal to TransPortal / DE-33 (deliberate — §1 Non-Goals).

### 9.5 Ambiguity check

- "Keep current Sisyphus means" — explicit in §3.3 ✓
- "Sampling mode" — explicit: caller passes `rng=rng` ✓
- "Correlation matrix is on log-scale" — repeated in §2.3 and §3.2 ✓
- "Byte-identical NOT claimed" — explicit in §4.1 ✓
- "SBC improvement NOT claimed" — explicit in §1 Non-Goals and §7.2 ✓

### 9.6 Architecture invariants preserved

- Invariant 1 (identity-blindness): correlation_group is a string the engine never reads ✓
- Invariant 2 (all parameters are Distribution): Distribution gains an optional field, remains the only abundance container ✓
- Invariant 3 (compile once): topology unchanged ✓
- Invariant 5 (holdout inviolable): Gate A enforces 107-holdout AAFE invariance ✓
- Invariant 8 (hard no-touch): no change to `engine/compiler.py`, `engine/solver.py`, `DrugOnGraph` fields, holdout drug list, or Cmax loss ✓

### 9.7 Remaining known risks

Accepted and documented in §6:
- R8 CYP2D6 lognormal misfit (v2 mixture modeling)
- R9 CV invariance with age/weight (Achour adult-only)
- R3 wide SE on correlation entries at N≈15-25 (reported with caveats in JSON)

### 9.8 Handoff

Ready for user review. On approval, next step is `superpowers:writing-plans` → `docs/superpowers/plans/2026-04-22-achour-abundance-correlation.md`.

---

## 10. References

- Achour B, Al-Majdoub ZM, Grybos-Gajniak A, et al. *Clin Pharmacol Ther* 2021; 109:222-232. "Liquid Biopsy Enables Quantification of the Abundance and Interindividual Variability of Hepatic Enzymes and Transporters." PMC7839483. DOI 10.1002/cpt.2013. **CC BY-NC 4.0**.
- Achour B, Barber J, Rostami-Hodjegan A. *Drug Metab Dispos* 2014; 42:1349-56. "Expression of hepatic drug-metabolizing cytochrome P450 enzymes and their intercorrelations: a meta-analysis." PMID 24879845. (Cross-reference for correlation pattern; not itself a data source in this spec.)
- Rodgers T, Rowland M. *Pharm Res* 2006; 23:1286-303. (Baseline Kp + abundance calibration, for historical context of §3.3 mean decision.)
- Sisyphus internal: `project_p4_continuous_complete.md` (SBC 41/52 baseline); `DESIGN.md` (invariant list); `docs/superpowers/specs/2026-04-20-oatp-ecm-hepatic-clearance-design.md` (full ECM — adjacent work, not amended by this spec).
