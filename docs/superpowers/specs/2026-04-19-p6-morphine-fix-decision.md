# P6 Morphine SBI Bias — Decision Package

**Status:** awaiting user decision
**Source diagnosis:** `data/validation/morphine_sbi_bias_diagnosis.json` (2026-04-17)
**Production state:** morphine routes to IS via `data/sbi/method_routing.json` (zero regression)

## The problem in one paragraph

Morphine's SBI posterior has CV=47% (barely tighter than prior CV=41%). Non-identifiability in theta-space — IS finds a tight low-CL mode (peff=1.99, enzyme_aff=1.08), SBI spans a wider high-CL mode (peff=2.57, enzyme_aff=2.60). Forward-simulating a wide theta distribution produces a right-skewed lognormal-like Cmax distribution; **arithmetic mean of a right-skewed distribution overshoots** the observation. SBC passes because marginal rank uniformity over the prior is correct — but individual-patient identifiability is not guaranteed by SBC. Morphine posterior_cmax_mean = 0.0284 vs obs 0.01865 → **+52% bias**. IS on the same drug → +3% bias.

## Three options

| # | Name | Code LOC | Risk | Expected morphine bias | Affects |
|---|------|:--:|:--:|:--:|---|
| 1 | Posterior median | ~2 | **Medium** | ~+20% | All 11 SBI-routed drugs |
| 2 | Likelihood reweighting | ~30 | Medium | ~+3% (match IS) | SBI path only; 2-5× SBI wall-time |
| 3 | Status quo (IS override) | 0 | **None** | N/A (morphine→IS) | Nothing (already in production) |

---

## Option 1 — Posterior median point estimate

**Change** (`src/sisyphus/regimen/tdm_sbi.py:628`):

```python
# Before
post_mean = float(np.mean(post_arr))

# After
post_mean = float(np.median(post_arr))  # robust to lognormal-like skew
```

Plus the multi-obs branch (line 609) — weighted median instead of weighted mean.

**Why it might work:** Median is the MLE-ish point estimate for log-normal family; unaffected by right-tail. Morphine CV=47% skew would drop to ~20% bias (posterior median ≈ exp(mean(log(cmax))) ≈ 0.021, closer to 0.01865).

**Why it's risky:**
- Convention break: IS/IBIS/EnKF all use mean (`tdm_ibis.py:537` `np.average`, `tdm_enkf.py:391` `cmax_post.mean()`). Having SBI alone use median creates asymmetric comparison in TDM tournament.
- Changes all 11 SBI-routed drugs' posterior point estimate. Drugs with tight posterior (CV<15%) will see median≈mean (ΔCmax <2%). But clozapine (SBI posterior CV=30%+ in some scenarios) could shift materially.
- **Must re-run TDM tournament** (`scripts/benchmark_tdm_methods.py`, ~2h for 5 anchor drugs × 4 methods). Routing decisions may change if median bias crosses SBI's advantage threshold.
- **Not principled for bimodal posteriors**: if posterior is truly bimodal with equal modes, median picks an arbitrary trough point between them.

**Test plan:**
1. Unit test: morphine synthetic observation → post_median within 25% of obs (currently post_mean is +52%).
2. Regression test: clozapine, rivaroxaban, amantadine post_median must be within ±10% of current post_mean (tight-posterior drugs should be unaffected).
3. Full tournament: `benchmark_tdm_methods.py --drugs morphine,clozapine,amantadine,ketorolac,rivaroxaban --methods is,ibis,enkf,sbi` — SBI bias must improve on morphine without worsening others >5pp.

**Rollback:** trivial, one-line revert.

---

## Option 2 — SBI posterior likelihood reweighting

**Change** (`src/sisyphus/regimen/tdm_sbi.py`, add ~30 LOC):

After SBI draws posterior samples and forward-simulates Cmax, apply a physical Gaussian likelihood over the **observation** and reweight:

```python
# After SBI forward simulation produces post_arr (Cmax samples)
# Add importance weights from physical likelihood on the observed timepoint
if _extra_obs is None and obs is not None:  # only for single-obs case
    log_w = _gaussian_log_likelihood(
        pred_conc=post_conc_at_t_obs,   # already simulated per sample
        obs_conc=obs.concentration,
        sigma_log=0.0414,  # same noise model as training simulator
    )
    log_w -= log_w.max()  # numerical stability
    weights = np.exp(log_w)
    weights /= weights.sum()
    post_mean = float(np.sum(weights * post_arr))
    # Same weighted variance / CI pattern as multi-obs branch (lines 609-622)
```

**Why it works:** SBI gives an *amortized* posterior approximation over the whole prior-predictive space; adding physical likelihood on the actual observation collapses wide modes. This is mathematically equivalent to IS with SBI posterior as proposal distribution (vs prior as proposal).

**Why it's risky:**
- **Partial loss of SBI amortization speed**: The existing forward simulation at t_obs is already done for `post_arr`, so the likelihood computation is cheap. But if ESS collapses (<30%), we need to resample, which adds another forward simulation pass. Expected 2-5× wall-time vs status quo SBI (~57s → 114-285s), still much faster than IS/IBIS.
- New code path needs test coverage (unit + dispatch + tournament).
- Principled but **effectively converges to IS** on morphine — we'd be paying SBI overhead for IS-level accuracy. Defeats SBI's purpose for this drug class.

**Test plan:**
1. Unit test: reweighted morphine posterior_cmax matches IS within 10%.
2. Numerical test: for a drug where SBI is already accurate (clozapine), reweighting must not degrade (Δbias ≤ 5pp).
3. ESS monitoring: log ESS_reweight; warn if <30%.
4. Tournament: full 5-drug × 4-method, compare SBI vs SBI+reweight.

**Rollback:** flag-gated (`sbi_reweight=False` default initially), easy revert.

---

## Option 3 — Status quo (IS override)

**Change:** None. `data/sbi/method_routing.json` already routes morphine → IS.

**State:**
- Production: 11 SBI / 1 IS (morphine) / 1 IBIS (pravastatin).
- Auto dispatch bias: 13% (SBI 19% + IS 3% + IBIS ~0%, weighted).
- Zero code risk; zero test regression.

**Only downside:** morphine-specific override is a "stamp" — if a *new* drug with similar non-identifiability pattern enters validation, we'd need another override. Detection heuristic (SBI posterior CV > 30% → route to IS) would make this systematic but isn't implemented.

---

## My recommendation: **Option 3 (status quo) + defer Option 2**

**Reasoning:**
1. Option 3 is already in production and working. Morphine TDM is correct via IS.
2. Option 1 breaks convention for an N=1 symptom. Median-vs-mean mismatch across methods complicates future tournament analyses.
3. Option 2 is the *principled* fix but has weak ROI today: morphine is the only known case, and it's already mitigated. If we later find 2+ drugs with the same pattern, Option 2's test + review cost pays off.
4. **Better intermediate**: add a **posterior-CV runtime gate** — if SBI returns CV > 30% after a forward pass, log a warning and auto-route to IS for that call. Costs ~5 LOC, makes the override systematic without a model change. (I can draft this as a 4th option if you want.)

**Red flag I want to flag honestly:** If Option 1 (median) is an acceptable convention change (some Bayesian papers do prefer median-of-posterior), the code change is truly 2 lines. The cost is a 2h tournament re-run, and the gain is morphine-bias drop without per-drug routing overrides. It's a reasonable pragmatic choice if you don't mind the convention.

---

## Decision

Please pick one:

- [ ] **Option 1** (median) — I'll implement + run tournament + commit results
- [ ] **Option 2** (likelihood reweighting) — I'll implement behind `sbi_reweight=False` flag, add tests, benchmark
- [ ] **Option 3** (status quo) — close P6 as "resolved via production routing"; no code change
- [ ] **Option 4** (systematic CV gate — if you want this drafted) — runtime SBI→IS auto-fallback when posterior CV > 30%
