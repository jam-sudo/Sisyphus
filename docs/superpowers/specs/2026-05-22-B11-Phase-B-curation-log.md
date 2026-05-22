# B-11 Phase B Curation Log

**Date started**: 2026-05-22
**Cycle**: Phase B mechanism triage + literature search
**Pre-B-11 baseline**: Meta AAFE 2.7715
**Branch**: feat/b11-phase-b-curation

---

## T11 — Predicted fup table

Run via `sisyphus.predict.chemistry.compute_profile` + `sisyphus.predict.adme.predict_adme` using SMILES from `data/reference/clinical_pk.json`. All 19 drugs route=oral, holdout predictions from `data/training/4track_holdout_predictions.json`.

| name | compound_type | predicted fup | logp | mw | meta_fold | obs Cmax (mg/L) | pred Cmax (mg/L) |
|---|---|---|---|---|---|---|---|
| lenacapavir | neutral | 0.0020 | 5.57 | 968.3 | 37.98 | 0.02340 | 0.88866 |
| acamprosate | acid | 0.5802 | −1.80 | 181.2 | 33.82 | 0.18000 | 6.08845 |
| methylphenidate | base | 0.9500 | 2.16 | 233.3 | 24.73 | 0.00910 | 0.22503 |
| selegiline | base | 0.4179 | 2.70 | 187.3 | 20.22 | 0.00100 | 0.02022 |
| budesonide | neutral | 0.1250 | 1.91 | 430.5 | 14.95 | 0.00150 | 0.02243 |
| paroxetine | base | 0.0500 | 2.53 | 329.4 | 13.22 | 0.00500 | 0.06612 |
| abiraterone | neutral | 0.1469 | 4.76 | 349.5 | 9.60 | 0.07300 | 0.70045 |
| progesterone | neutral | 0.0400 | 3.87 | 314.5 | 9.27 | 0.01730 | 0.16034 |
| oxybutynin | base | 0.2004 | 4.30 | 357.5 | 8.16 | 0.00100 | 0.00816 |
| fesoterodine | base | 0.0299 | 5.29 | 411.6 | 7.66 | 0.00189 | 0.01447 |
| fluvoxamine | base | 0.2150 | 3.20 | 318.3 | 7.46 | 0.01500 | 0.11194 |
| vonoprazan | base | 0.1500 | 2.40 | 345.4 | 7.38 | 0.02500 | 0.18448 |
| ramelteon | neutral | 0.1800 | 2.40 | 259.3 | 6.23 | 0.01160 | 0.07227 |
| clopidogrel | base | 0.2175 | 3.34 | 321.8 | 5.15 | 0.30000 | 1.54604 |
| posaconazole | neutral | 0.0200 | 5.50 | 700.8 | 4.92 | 0.15100 | 0.74348 |
| sumatriptan | base | 0.8600 | 0.93 | 295.4 | 4.80 | 0.01650 | 0.07924 |
| azacitidine | neutral | 0.9786 | −3.50 | 244.2 | 4.01 | 0.14500 | 0.58086 |
| venlafaxine | base | 0.7300 | 3.13 | 277.4 | 3.80 | 0.03550 | 0.13482 |
| morphine | zwitterion | 0.6500 | 0.87 | 285.3 | 3.17 | 0.01865 | 0.05910 |

Note: predicted fup values are XGBoost model outputs. DrugBank fup override applied where available: paroxetine=0.05 (DB), progesterone=0.04 (DB), fesoterodine=0.03 (DB); selegiline used XGBoost 0.418 (DB value 0.005 was 83× discrepant, XGBoost preferred), oxybutynin used XGBoost 0.200 (DB 0.030 discrepant), clopidogrel used XGBoost 0.218 (DB 0.500 discrepant).

---

## T11 — Per-drug mechanism triage

### lenacapavir

- **meta_fold**: 37.98 | **predicted fup**: 0.0020 | **type**: neutral
- **Clinical info source**: FDA label SUNLENCA® 2022 (https://www.accessdata.fda.gov/drugsatfda_docs/label/2022/215973s000lbl.pdf); Mathias et al. 2024, JPET doi:10.1124/jpet.124.002090
- **Major clearance pathway**: Intestinal secretion via P-gp is the primary clearance route (31–60% unchanged drug recovered in bile-duct-cannulated animal feces). CYP3A4 and UGT1A1 are minor contributors. Negligible renal (<0.24%). MW 968 Da triggers `HIGH_MW` + `EXTREME_LIPOPHILIC` AD flags; out-of-applicability-domain.
- **Mechanism hypothesis**: transporter (P-gp intestinal secretion) — the well-stirred model has no P-gp secretion term; all clearance is attributed to hepatic CL → massive over-prediction of plasma accumulation.
- **Preliminary disposition for T12**: not_applicable — out of AD, non-hepatic clearance mechanism (P-gp secretion), PPB irrelevant here.
- **Rationale**: Lenacapavir is flagged as AD-out with `HIGH_MW` and `EXTREME_LIPOPHILIC` and `PGP_EFFLUX_RISK`. The 38× fold error is driven by the engine lacking intestinal P-gp secretion: bile duct cannulation studies confirm 31–60% unchanged drug in feces. Even if fup were corrected, the dominant clearance route is absent from the model entirely. Fu correction would not address the root cause.

---

### acamprosate

- **meta_fold**: 33.82 | **predicted fup**: 0.5802 | **type**: acid
- **Clinical info source**: FDA label Campral® (https://www.accessdata.fda.gov/drugsatfda_docs/label/2010/021431s013lbl.pdf); Pelc 1998, Clin Pharmacokinet PMID:9839087
- **Major clearance pathway**: Pure renal excretion. Acamprosate is not metabolized — no CYP or UGT involvement. Major route is kidney excretion unchanged. t½ 20–33h. Dose adjustment required for renal impairment. Not hepatically cleared at all.
- **Mechanism hypothesis**: renal — model routes all clearance through the hepatic compartment. Acamprosate has ~fup 0.58 (mostly unbound, low binding), logP −1.8 (highly hydrophilic), MW 181. The 34× fold error reflects entirely absent renal clearance modeling as primary route + possible dose/bioavailability issue (666 mg TID but clinical Cmax is at steady state; reference dose may be mismatched).
- **Preliminary disposition for T12**: not_applicable
- **Rationale**: Acamprosate's over-prediction is a structural engine failure — all clearance for this purely renally-excreted drug flows through the hepatic node. There is no hepatic CL pathway to apply fu correction to. fup=0.58 further excludes this from PPB-related hypothesis. The correct fix would be a dedicated renal clearance pathway for this drug class, which is outside B-11 scope.

---

### methylphenidate

- **meta_fold**: 24.73 | **predicted fup**: 0.9500 | **type**: base
- **Clinical info source**: FDA label Ritalin® (https://www.accessdata.fda.gov/drugsatfda_docs/label/2013/021284s020lbl.pdf); Kimko et al. 1999 PMID:10319298
- **Major clearance pathway**: Primarily hydrolyzed by carboxylesterase 1 (CES1) → inactive ritalinic acid. Minor oxidative metabolism. Protein binding is low (10–33%). Systemic CL ~0.40 L/h/kg (d-isomer).
- **Mechanism hypothesis**: enzyme — CES1 is poorly represented in the engine. The model likely relies on CYP-class clearance terms and lacks CES1 hydrolysis as a major clearance pathway. Additionally, the reference Cmax (0.0091 mg/L at 72 mg) implies very high apparent CL; this may be an extended-release formulation dose vs. immediate-release Cmax reference mismatch.
- **Preliminary disposition for T12**: not_applicable
- **Rationale**: Methylphenidate has high fup (~0.90), excluding the PPB-related mechanism. The dominant pathway (CES1 hydrolysis) is systematically absent from the engine's CYP-oriented enzyme_affinity schema. The 25× over-prediction likely reflects that CES1 hydrolysis accounts for ~85% of clearance but is either absent or underweighted. Fu correction would not address the root cause. Separate CES1 registration is the appropriate fix.

---

### selegiline

- **meta_fold**: 20.22 | **predicted fup**: 0.4179 | **type**: base
- **Clinical info source**: StatPearls NBK526094; PBPK PMC7600566; Drugs.com monograph
- **Major clearance pathway**: Extensive first-pass hepatic metabolism by CYP2B6 (major), CYP1A2, CYP3A4 → l-desmethylselegiline + l-methylamphetamine → l-amphetamine. Oral bioavailability only 4–10% (extremely high first-pass). Protein binding reported as 85–90% in StatPearls/PMC7600566 (macroglobulin + albumin); DrugBank lists fup=0.005 (>99.5% bound) — sources disagree by ~10×. High hepatic extraction ratio drug.
- **Mechanism hypothesis**: formulation / extreme first-pass — oral F=4–10% is known but extremely difficult to model accurately. The engine likely assigns normal gut-wall + hepatic extraction but misses the near-complete presystemic loss. Plasma protein binding is reported inconsistently (StatPearls 85–90% bound → fup ~0.10–0.15; DrugBank fup=0.005 → 99.5% bound); the XGBoost prediction (fup=0.42) is above both reported ranges.
- **Preliminary disposition for T12**: PPB-related candidate (secondary mechanism) — predicted fup (0.42) overestimates literature ranges (whether 0.10–0.15 per StatPearls or 0.005 per DrugBank). Primary mechanism is extreme first-pass under-representation. Fup correction alone insufficient; flag as ceiling_accepted unless fup correction can be combined with better F.
- **Rationale**: The 20× fold error combines at least two mechanisms: (1) extreme first-pass metabolism that pushes F to 4–10%, which is hard to capture accurately; (2) possible fup over-estimation by ~3–80× depending on which literature source is used. The primary driver is the extreme oral bioavailability problem. Classify as not_applicable for PPB correction; the fix requires improving first-pass/F modeling, which is not B-11 scope.

  **Revised disposition**: not_applicable (primary mechanism: extreme first-pass F=4–10%, PPB correction cannot address this)

---

### budesonide

- **meta_fold**: 14.95 | **predicted fup**: 0.1250 | **type**: neutral
- **Clinical info source**: FDA label TARPEYO® 2021 (https://www.accessdata.fda.gov/drugsatfda_docs/label/2021/215935s000lbl.pdf); PMID:10518844; PMC10535222
- **Major clearance pathway**: CYP3A4 hepatic first-pass 80–90%, producing 6β-hydroxybudesonide and 16α-hydroxyprednisolone. Oral F = 9–21% (enteric-coated delayed-release formulation used in clinical studies). High clearance compound; elimination solely metabolic (no renal).
- **Mechanism hypothesis**: formulation / first-pass — budesonide uses enteric-coated delayed-release formulations that release drug in the terminal ileum for local IBD action. The reference Cmax is from an EC capsule with very low systemic F. The engine likely models standard oral absorption without the formulation-specific delayed release and reduced systemic bioavailability (F~9%). Additionally, predicted fup=0.125 vs. literature ~0.10–0.15 (protein binding 85–90%) is borderline but close to threshold.
- **Preliminary disposition for T12**: not_applicable — formulation over-prediction dominates. The 15× error primarily reflects that the reference drug is an enteric-coated local-acting formulation (F~9%) not represented in the engine as standard oral absorption.
- **Rationale**: While fup=0.125 places budesonide near the PPB threshold and hepatic CYP3A4 dominates clearance, the 15× fold error is disproportionate to what a fup correction alone could fix. The root cause is formulation: budesonide enteric capsules have F≈9% engineered to be low for local GI action. Without a formulation-specific bioavailability override, applying fu correction is cosmetic. Classify not_applicable for B-11; formulation override is the appropriate fix.

---

### paroxetine

- **meta_fold**: 13.22 | **predicted fup**: 0.0500 | **type**: base
- **Clinical info source**: FDA label PAXIL® 2021 (https://www.accessdata.fda.gov/drugsatfda_docs/label/2021/020031s077lbl.pdf)
- **Major clearance pathway**: CYP2D6-mediated hepatic metabolism (primary at low doses, rapidly saturates). At steady state, alternative P450 isoforms take over. High protein binding (clinical fup ~0.01–0.05; "highly bound to plasma proteins"). Essentially negligible renal excretion. Nonlinear kinetics due to CYP2D6 autoinhibition — paroxetine inhibits its own clearance.
- **Mechanism hypothesis**: PPB-related + CYP2D6 saturation/autoinhibition — fup 0.05 and hepatic-CL-dominant. Additionally, paroxetine is a potent CYP2D6 mechanism-based inhibitor (autoinhibitor), which means the effective in-vivo Cmax is 2–5× higher than predicted by naive hepatic extraction because CL decreases over time. This is a complex scenario: PPB correction would raise CL (lower predicted Cmax), but autoinhibition works in the opposite direction. Both effects may partially cancel.
- **Preliminary disposition for T12**: PPB-related candidate — fup 0.05 AND hepatic CL dominant AND clinical literature fup may be even lower than predicted (FDA label indicates "highly protein bound" consistent with fup < 0.05). However, CYP2D6 saturation/autoinhibition is a confounding mechanism that could independently cause over-prediction at reference doses. Priority: literature_applied candidate if fu,inc/fu,plasma ratio is available; ceiling_accepted otherwise.
- **Rationale**: Paroxetine meets PPB criteria (fup=0.05 < 0.10, hepatic-CL-dominant). The complication is nonlinear PK from CYP2D6 autoinhibition — at clinical doses, paroxetine self-inhibits its primary metabolic pathway, so steady-state Cmax is substantially higher than single-dose predictions suggest. The reference likely is steady-state. The 13× over-prediction may reflect both PPB (under-predicting CL because fup is low) and the engine ignoring CYP2D6 saturation kinetics. B-11 can address the PPB component; the autoinhibition mechanism is separate.

---

### abiraterone

- **meta_fold**: 9.60 | **predicted fup**: 0.1469 | **type**: neutral
- **Clinical info source**: FDA label Zytiga® 2021 (https://www.accessdata.fda.gov/drugsatfda_docs/label/2021/202379s035lbl.pdf); Goldwater 2017 PMID:28107519; Bohnert 2013 PBPK SULT2A1 disposition
- **Major clearance pathway**: Hepatic, primarily SULT2A1 sulfation and CYP3A4 oxidation to inactive metabolites. Protein binding >99% (clinical fup <0.01 vs. predicted 0.147 — XGBoost overshoot of ~15×). ~88% fecal excretion (55% unchanged abiraterone acetate + 22% abiraterone), confirming incomplete absorption/high first-pass.
- **Mechanism hypothesis**: PPB-related (CYP3A4 + SULT2A1, fup<0.10) — predicted fup=0.147 is borderline above the strict 0.10 cutoff, but clinical fup<0.01 clearly satisfies the PPB criterion (highly protein-bound). The B-11 mechanism multiplies the *predicted* fup (0.147) by `fu_correction_liver ≥ 1.0`, raising `fup_effective` → raising hepatic CL → lowering Cmax. Direction is correct (mirrors oxybutynin: predicted fup 0.20, clinical ~0.01, also over-predicted 8×). Food effect (17× fasted→fed) and abiraterone-acetate prodrug conversion are confounders for absolute fold error but do not invalidate the PPB-correction direction.
- **Preliminary disposition for T12**: PPB-related candidate (literature_applied target) — abiraterone is well-studied in PBPK literature (Bohnert 2013 SULT2A1 dispositions; multiple PBPK papers). Search fu,inc/fu,plasma data for SULT2A1 substrates and highly-albumin-bound steroidal scaffolds. If no literature is found, fall through to ceiling_accepted at `fu_correction_liver=1.0` (matches oxybutynin/progesterone fallback).
- **Rationale**: Abiraterone meets the PPB criteria once the upstream XGBoost overshoot is recognized: predicted fup=0.147 vs. clinical fup<0.01, hepatic CYP3A4 + SULT2A1 dominant. The B-11 correction direction (`fup_eff = fup × fu_corr` with `fu_corr ≥ 1`) raises effective CL → reduces Cmax → moves the 9.6× over-prediction toward the observed value, parallel to oxybutynin. The XGBoost over-prediction of bound-drug fup (0.147 vs. <0.01) is an upstream engine concern (model under-represents binding for steroidal/SULT2A1 substrates) that is independent of B-11 scope — B-11 operates on whatever fup predict_adme returned. Food effect and prodrug activation are separate dispositional gaps not addressed here.

---

### progesterone

- **meta_fold**: 9.27 | **predicted fup**: 0.0400 | **type**: neutral
- **Clinical info source**: FDA label PROMETRIUM® 2024 (https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/019781s025lbl.pdf); PMID:8513955
- **Major clearance pathway**: CYP3A4 hepatic first-pass (major), with 10% absolute oral bioavailability due to extensive first-pass metabolism. ~96–99% bound to albumin (50–54%) and transcortin/SHBG (43–48%). Fecal excretion predominates; negligible renal.
- **Mechanism hypothesis**: formulation / extreme first-pass + PPB — oral micronized progesterone (PROMETRIUM) has F≈10% with extreme intraindividual variability (Cmax CV >100% in pivotal studies). Protein binding 96–99% (fup 0.01–0.04). Predicted fup=0.04 is at the lower end of literature range, which is borderline but within range.
- **Preliminary disposition for T12**: PPB-related candidate (borderline) — fup=0.04 < 0.10, hepatic CL dominant. The 9.3× fold error is large but the extreme oral bioavailability variability (F=10%, CV>100%) is likely a larger contributor than PPB alone. Fu correction direction: at fup=0.04, hepatic extraction is low, drug accumulates → Cmax over-predicted. With fu,inc correction >1.0 (higher effective intracellular fup), CL would increase → Cmax decrease → correction in right direction. However, the ~10× error is more consistent with a bioavailability issue (engine doesn't know F=10%). Classify as ceiling_accepted unless specific fu,inc/fu,plasma ratio found.
- **Rationale**: Progesterone meets PPB criteria marginally (fup=0.04, hepatic CL dominant). However, the extreme oral first-pass (F=10%) and extreme inter-individual variability (Cmax CV>100%) are likely the dominant drivers. The 9× over-prediction partially reflects the engine assuming standard oral absorption when the actual clinical Cmax is from a highly variable, low-F formulation. Fu correction can contribute but is unlikely to be the primary fix. Candidate for ceiling_accepted if no direct fu,inc literature found.

---

### oxybutynin

- **meta_fold**: 8.16 | **predicted fup**: 0.2004 | **type**: base
- **Clinical info source**: DailyMed OXYBUTYNIN CHLORIDE ER; StatPearls NBK499985; Lukkari 1998 PMID:9524014
- **Major clearance pathway**: CYP3A4-dominated hepatic first-pass metabolism (>95% first-pass). Oral F only ~10% due to extensive gut-wall + hepatic CYP3A4. Protein binding ~99% (fup ~0.01 in clinical data vs. predicted 0.20 — a 20× discrepancy). Less than 0.1% excreted unchanged in urine.
- **Mechanism hypothesis**: PPB-related — predicted fup=0.20 vs. clinical 0.01 is a 20× discrepancy, the second largest in the set. The WS model using fup=0.20 assigns substantially lower hepatic extraction than at fup=0.01. Wait — same direction analysis as abiraterone: at fup=0.01 (true value), CL would be even lower than at fup=0.20 (because fup×CLint is smaller), so the over-prediction would be WORSE with true fup. Re-examine direction: predicted Cmax 0.00816 vs. obs 0.001 mg/L → engine over-predicts. High fup in model → less hepatic extraction → drug accumulates → over-prediction. With true fup=0.01 → even less hepatic extraction → even more accumulation → even worse. So PPB correction (raising fu,inc) would help: fu,inc/fu,p > 1 means effective intracellular fup is higher → higher CL → lower Cmax → better. This is the correct direction!
- **Preliminary disposition for T12**: PPB-related candidate (strong) — predicted fup=0.20, clinical fup≈0.01, hepatic CYP3A4-dominant. Fu correction direction: fu,inc/fu,p > 1 would increase effective CL → reduce Cmax → fix over-prediction. Search Watanabe 2009, Yamazaki 2010, Riccardi 2017 for oxybutynin or anticholinergic base class. Also note F=10% (enteric gut wall + hepatic CYP3A4) may independently contribute.
- **Rationale**: Oxybutynin's predicted fup (0.20) is approximately 20× above literature clinical value (~0.01), and the engine over-predicts Cmax by 8×. The fu correction mechanism applies correctly here: raising fu,inc above plasma fup increases effective hepatic CL under WS formula, reducing predicted Cmax. Additionally, high first-pass (F=10%) may compound the error. Strong B-11 PPB candidate; search for measured fu,inc ratio in the anticholinergic/antispasmodic class.

---

### fesoterodine

- **meta_fold**: 7.66 | **predicted fup**: 0.0299 | **type**: base — OUT OF AD (PRODRUG flag)
- **Clinical info source**: FDA label TOVIAZ® 2021 (https://www.accessdata.fda.gov/drugsatfda_docs/label/2021/022030s019lbl.pdf)
- **Major clearance pathway**: Fesoterodine is a prodrug hydrolyzed by esterases (ubiquitous, not CES1-specific) to 5-HMT (active metabolite) immediately after absorption. 5-HMT is the circulating entity. 5-HMT protein binding is ~50% (albumin + α1-acid glycoprotein). 5-HMT CL via CYP2D6 + CYP3A4 (further to inactive metabolites). Renal clearance ~250 mL/min for 5-HMT (substantial — ~40% renal for the active metabolite).
- **Mechanism hypothesis**: prodrug — fesoterodine is flagged `PRODRUG` and is out of AD. The engine's prodrug modeling predicts the active metabolite's Cmax. Clinical reference Cmax (0.00189 mg/L) may be for fesoterodine parent (which barely exists in plasma — virtually all converts to 5-HMT) or for 5-HMT. The fup=0.03 is for the prodrug parent; the metabolite fup=0.50 is very different. Additional complexity: the model is predicting wrong entity vs. reference.
- **Preliminary disposition for T12**: not_applicable — PRODRUG flag, out of AD. Reference Cmax entity is unclear (parent vs. 5-HMT). The PPB correction target would be 5-HMT (fup 0.50, not PPB-related by criteria).
- **Rationale**: Fesoterodine is a registered prodrug (out of AD). The predicted fup of 0.03 is for the parent molecule, but essentially none of the parent reaches systemic circulation — 5-HMT is the active form with fup≈0.50. The mismatch between parent entity fup and active metabolite fup means the PPB criterion (fup<0.1, hepatic-dominant) doesn't apply to the relevant circulating species. B-11 cannot help here without resolving the prodrug Cmax reference entity mismatch.

---

### fluvoxamine

- **meta_fold**: 7.46 | **predicted fup**: 0.2150 | **type**: base
- **Clinical info source**: Drugs.com monograph; Springer 1995 doi:10.2165/00003088-199500291-00003; FDA label LUVOX® 2017
- **Major clearance pathway**: Hepatic oxidative metabolism by CYP2D6 (minor) and CYP1A2 (major for self-inhibition context). F~50% (not complete despite full GI absorption — significant first-pass). Elimination ~entirely hepatic metabolites in urine. Protein binding ~77% (fup ~0.23). t½ 12–15h.
- **Mechanism hypothesis**: uncertain — fup=0.215 is above the PPB threshold of 0.10, excluding PPB-related hypothesis. The 7.5× fold error at fup=0.22 is puzzling. Contributing factors: (1) CYP1A2 autoinhibition (fluvoxamine potently inhibits CYP1A2, its own metabolizer) → self-inhibitory kinetics make single-dose vs. steady-state diverge; (2) the reference Cmax is for 50 mg single dose (0.015 mg/L), but fluvoxamine has substantial first-pass (F~50%); (3) possible fup over-estimation (literature PPB 77%, predicted fup=0.215 is consistent with ~77% binding, so prediction seems accurate).
- **Preliminary disposition for T12**: not_applicable — fup=0.215 > 0.10, not PPB-candidate. Primary mechanism likely CYP1A2 autoinhibition and/or first-pass modeling inaccuracy.
- **Rationale**: Fluvoxamine's predicted fup (0.22) is above the PPB threshold. The 7.5× over-prediction is better explained by CYP1A2 autoinhibition — fluvoxamine is a strong CYP1A2 inhibitor that inhibits its own metabolism, reducing apparent CL vs. non-inhibitor substrate assumptions used in in-vitro CLint scaling. The engine uses naive in-vitro CLint without accounting for time-dependent CYP1A2 inhibition. This mechanism is outside B-11 scope.

---

### vonoprazan

- **meta_fold**: 7.38 | **predicted fup**: 0.1500 | **type**: base
- **Clinical info source**: FDA label 2024 (https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/215151s006,218710s000lbl.pdf); PMC10088082
- **Major clearance pathway**: CYP3A4 dominant hepatic metabolism, with contributions from CYP2B6, CYP2D6, and SULT. Oral bioavailability ~70%. Protein binding 80% (clinical fup ~0.20). t½ 6.9–8.6h. Hepatic impairment increases exposure 1.2–2.6× depending on severity.
- **Mechanism hypothesis**: unknown — predicted fup=0.15 is close to clinical fup≈0.20 (not a dramatic discrepancy). CYP3A4-dominant, hepatic clearance adequate for PPB check, but fup=0.15 > 0.10 (borderline, fails strict criterion). The 7.4× over-prediction at fup=0.15 is harder to explain by PPB. Contributing factors may include: vonoprazan is strongly basic (pKa ~9.0, pyrrole K-CAB type), potentially sequestered in acidic organelles via lysosomal trapping → apparent Vd much larger than predicted → distribution, not clearance, is the issue; alternatively, the CLint may be systematically over-estimated for this structural class.
- **Preliminary disposition for T12**: not_applicable — fup=0.15 borderline but above 0.10 threshold; clinical fup~0.20 even further above threshold. Mechanism is more likely lysosomal trapping causing larger Vd (basic drug, pKa 9) or CYP3A4 CLint over-estimation.
- **Rationale**: Vonoprazan fails the strict PPB criterion (fup=0.15 ≥ 0.10; clinical fup≈0.20 ≥ 0.10). The 7.4× over-prediction is better attributed to either Vd mis-estimation (strongly basic drug with potential lysosomal trapping inflating real Vd) or CLint over-prediction. Neither mechanism is addressable by B-11 fu correction.

---

### ramelteon

- **meta_fold**: 6.23 | **predicted fup**: 0.1800 | **type**: neutral
- **Clinical info source**: FDA label Rozerem® 2018 (https://www.accessdata.fda.gov/drugsatfda_docs/label/2018/021782s021lbl.pdf)
- **Major clearance pathway**: CYP1A2-dominant hepatic metabolism (major), CYP2C and CYP3A4 minor. Extensive first-pass: oral F ~1.8% (very rapid and extensive first-pass metabolism). Protein binding 82% (fup ~0.18). Active metabolite M-II has 1/10–1/5 binding affinity. Negligible renal clearance.
- **Mechanism hypothesis**: extreme first-pass + CYP1A2 dominant — oral F=1.8% is among the lowest in the drug class (per FDA label, "rapid, high first-pass"). The reference Cmax (0.0116 mg/L at 16 mg) corresponds to an F ~1.8%. The engine likely assigns a much higher F (5–15%) based on standard logP/Peff assumptions, explaining 6× over-prediction. fup=0.18 is above threshold.
- **Preliminary disposition for T12**: not_applicable — fup=0.18 > 0.10; primary mechanism is extreme oral first-pass (F=1.8%), not PPB.
- **Rationale**: Ramelteon's oral bioavailability of 1.8% means the engine must correctly model ~98% first-pass elimination. With fup=0.18 (above threshold) and the extreme first-pass as dominant mechanism, PPB correction is irrelevant. The over-prediction is a Peff/F estimation problem.

---

### clopidogrel

- **meta_fold**: 5.15 | **predicted fup**: 0.2175 | **type**: base
- **Clinical info source**: FDA label Plavix® (https://www.accessdata.fda.gov/drugsatfda_docs/label/2009/020839s044lbl.pdf); PMC5677184; PMC8934724
- **Major clearance pathway**: CES1 hydrolyzes ~85% of absorbed clopidogrel to inactive carboxylic acid. Only ~15% is bioactivated by CYP2C19, CYP2B6, CYP1A2 to the active thiol metabolite. The reference Cmax for clopidogrel parent is from a 300 mg loading dose. Protein binding ~98% for active metabolite, parent ~50% (predicted 0.22 is for the parent molecule).
- **Mechanism hypothesis**: prodrug + CES1 hydrolysis — clopidogrel is being modeled as a prodrug in the engine (it is in the clopidogrel prodrug design spec 2026-05-20). The 5.15× fold error may reflect incomplete CES1 clearance modeling for the parent. Predicted fup=0.22 for the parent (>0.10 threshold) — not PPB-related for the parent. If the reference is active metabolite Cmax (which is much lower), then the mismatch could reflect prodrug activation fraction modeling.
- **Preliminary disposition for T12**: not_applicable — fup=0.22 > 0.10 for the parent; CES1 hydrolysis mechanism dominates. The clopidogrel prodrug pathway is being addressed in the separate B-03 spec (2026-05-20). B-11 fu correction does not address CES1 activity.
- **Rationale**: Clopidogrel's fold error at 5.15× is at the lower end of the over-prediction group. The mechanism is primarily CES1-dominant (85% hydrolysis) and CYP2C19 bioactivation (15%), not PPB-related. Predicted fup=0.22 is above the B-11 threshold. This is being addressed by the separate clopidogrel prodrug design spec. No B-11 action.

---

### posaconazole

- **meta_fold**: 4.92 | **predicted fup**: 0.0200 | **type**: neutral — OUT OF AD (HIGH_MW, PGP_EFFLUX_RISK)
- **Clinical info source**: FDA label Noxafil® (https://www.accessdata.fda.gov/drugsatfda_docs/label/2015/022003s018s020,0205053s002s004,0205596s001s003lbl.pdf); PMC2950982
- **Major clearance pathway**: UGT1A4 glucuronidation (~20–30% of dose) as primary metabolic route; CYP450 involvement is minimal. P-gp substrate (efflux). Renal clearance <1 mL/h (negligible). Fecal elimination predominant. Protein binding >98% (fup ~0.01).
- **Mechanism hypothesis**: formulation / absorption + P-gp efflux — posaconazole oral suspension has highly erratic absorption strongly dependent on gastric pH and food. The reference Cmax is likely from the suspension formulation. Precipitation in the small intestine at pH shift from stomach (high lipophilicity, weakly basic property) causes dramatic absorption variability. Additionally, `PGP_EFFLUX_RISK` AD flag is set and MW=700.8 triggers `HIGH_MW`. Out of AD. The ~5× fold error combines absorption variability with P-gp efflux not modeled.
- **Preliminary disposition for T12**: not_applicable — out of AD (HIGH_MW + PGP_EFFLUX_RISK). Mechanism is absorption/formulation + P-gp efflux, not hepatic-CL-dominant PPB. Although fup=0.02 meets the PPB threshold, UGT-dominated clearance is not hepatic CYP-mediated, and the dominant error mechanism is absorption.
- **Rationale**: Posaconazole is flagged out of AD with two flags. While fup<0.10 and hepatic UGT clearance theoretically meet PPB criteria, the dominant over-prediction mechanism is absorption: the oral suspension has gastric-pH-dependent precipitation, very low and erratic F, and strong food effect. P-gp efflux also reduces systemic exposure. The UGT1A4 clearance pathway is not affected by albumin-facilitated uptake in the same way as CYP substrates. Not applicable for B-11.

---

### sumatriptan

- **meta_fold**: 4.80 | **predicted fup**: 0.8600 | **type**: base
- **Clinical info source**: FDA label Imitrex® (https://www.accessdata.fda.gov/drugsatfda_docs/label/2012/020132s024s026lbl.pdf)
- **Major clearance pathway**: MAO-A oxidative deamination → indole acetic acid (IAA) and IAA glucuronide (inactive). ~60% urinary, ~40% fecal. Protein binding only 14–21% (fup ~0.79–0.86). High apparent clearance. Hepatic first-pass contributes to only ~68% bioavailability.
- **Mechanism hypothesis**: unknown — sumatriptan has very high fup (0.86, clinical 0.79–0.86). The 4.8× fold error cannot be PPB-related (far above threshold). The primary clearance is MAO-A oxidation. Contributing factors: (1) the reference dose is 25 mg oral, clinical Cmax 0.0165 mg/L — this may reflect single-dose and the reference could be from specific populations; (2) the engine likely lacks MAO-A as a clearance enzyme (it uses CYP-based enzyme_affinity), meaning MAO-A clearance is either entirely missing or misrouted to a CYP with wrong CLint; (3) ~32% first-pass loss.
- **Preliminary disposition for T12**: not_applicable — fup=0.86, well above PPB threshold. MAO-A is the primary clearance enzyme, which is not represented in the CYP-based enzyme_affinity framework. Structural engine limitation.
- **Rationale**: Sumatriptan's fup=0.86 completely excludes PPB mechanism. The 4.8× over-prediction is entirely attributable to the engine lacking MAO-A as a registered clearance enzyme. MAO-A-metabolized drugs (serotonin system) are systematically mis-predicted because their primary clearance route is not in the enzyme_affinity registry. Separate MAO-A enzyme registration initiative needed.

---

### azacitidine

- **meta_fold**: 4.01 | **predicted fup**: 0.9786 | **type**: neutral
- **Clinical info source**: FDA label ONUREG® 2020 (https://www.accessdata.fda.gov/drugsatfda_docs/label/2020/214120s000lbl.pdf); DailyMed VIDAZA
- **Major clearance pathway**: Spontaneous chemical hydrolysis and cytidine deaminase enzymatic deamination — not CYP or UGT mediated. No CYP450 involvement. 85% urinary excretion (renal dominant, largely as metabolites). t½ ~4h. Protein binding 6–12% (fup ~0.88–0.94).
- **Mechanism hypothesis**: enzyme/route — azacitidine is cleared by spontaneous hydrolysis and cytidine deaminase (a cytosolic enzyme present in liver and intestinal mucosa), not hepatic CYP. The reference Cmax (0.145 mg/L at 300 mg oral ONUREG tablet) involves a high dose, but oral formulation of azacitidine has substantial variability in absorption. The 4× over-prediction at fup=0.98 cannot be PPB-related. The likely explanation is (1) cytidine deaminase activity is not in the enzyme_affinity schema; (2) the oral tablet formulation of azacitidine has complex absorption (nucleoside analog, requires transport); (3) the reference dose (300 mg oral) is from the ONUREG approved label, not the parenteral formulation, and oral bioavailability for this formulation is approximately 11–13%.
- **Preliminary disposition for T12**: not_applicable — fup=0.98, not PPB-related. Mechanism: cytidine deaminase clearance not modeled + oral bioavailability ~11-13% not captured.
- **Rationale**: Azacitidine has nearly complete fup (0.98), no hepatic CYP metabolism, and is cleared primarily by spontaneous hydrolysis and cytidine deaminase. The oral formulation (ONUREG) has F≈11–13%, contributing to the 4× over-prediction. No B-11 action possible.

---

### venlafaxine

- **meta_fold**: 3.80 | **predicted fup**: 0.7300 | **type**: base
- **Clinical info source**: FDA label Effexor® (https://www.accessdata.fda.gov/drugsatfda_docs/label/2012/020151s059lbl.pdf); PMID:11098412
- **Major clearance pathway**: CYP2D6-mediated conversion to ODV (active metabolite, ~active equivalent) as primary pathway; CYP3A4 and CYP2C19 minor. Protein binding: venlafaxine ~27% (fup ~0.73), ODV ~30% (fup ~0.70). High total CL (~72 L/h). Hepatic impairment extends half-life ~40%. Good oral bioavailability.
- **Mechanism hypothesis**: unknown — fup=0.73, well above PPB threshold. The 3.8× fold error is modest compared to other drugs in the list. Contributing factors: (1) reference Cmax may be for immediate-release vs. engine using standard absorption model; (2) the active metabolite ODV may dominate clinical Cmax at some sampling time points; (3) CYP2D6 variability (extensive metabolizers vs. poor metabolizers have 2-3× different exposure). The 75 mg dose is moderate; Cmax from this reference is 0.035 mg/L.
- **Preliminary disposition for T12**: not_applicable — fup=0.73, not PPB-related. Modest 3.8× fold error may reflect CYP2D6 population variability in the reference study or IR vs. modified-release formulation mismatch.
- **Rationale**: Venlafaxine has high fup (0.73), modest first-pass, and is reasonably well-modeled except for CYP2D6 population variability and possible formulation effects. Not a B-11 candidate.

---

### morphine

- **meta_fold**: 3.17 | **predicted fup**: 0.6500 | **type**: zwitterion
- **Clinical info source**: FDA label MORPHINE SULFATE (https://www.accessdata.fda.gov/drugsatfda_docs/label/2006/019916s004lbl.pdf); PMC3276770
- **Major clearance pathway**: UGT2B7-dominant glucuronidation (liver + kidney + intestine) to M3G (inactive) and M6G (active). Minor CYP3A4 N-demethylation. Protein binding ~36% (fup ~0.64); muscle binding ~54%. Total plasma CL ~0.9–1.2 L/kg/h (high, not limited by fup). Routes via SBI+reweight override in production (morphine has a special routing override per CLAUDE.md).
- **Mechanism hypothesis**: UGT2B7 enzyme representation + routing — morphine is already handled with a special SBI routing override. The 3.17× fold error at fup=0.65 is not PPB-related (fup well above threshold). UGT2B7 is the primary enzyme and is likely not well-characterized in the enzyme_affinity registry. The production routing override (SBI+reweight) partially compensates for the engine limitation.
- **Preliminary disposition for T12**: not_applicable — fup=0.65, not PPB-related. UGT2B7 CLint representation is the engine limitation.
- **Rationale**: Morphine's fup=0.65 excludes PPB mechanism entirely. The 3.17× fold error is the smallest in the over-prediction group. The morphine routing override in production is already partially addressing the engine limitation. Not a B-11 target.

---

## T11 Summary

### PPB-related candidates (fup < 0.10 AND hepatic-CL-dominant AND correct direction)

After direction analysis (confirming fu correction would reduce Cmax to correct over-prediction):

| drug | fup | meta_fold | disposition | rationale |
|---|---|---|---|---|
| **paroxetine** | 0.05 | 13.22 | PPB candidate | fup=0.05, hepatic CYP2D6, correct direction; confounded by autoinhibition |
| **oxybutynin** | 0.20→0.01 | 8.16 | PPB candidate (strong) | clinical fup≈0.01 vs predicted 0.20, hepatic CYP3A4, correct direction |
| **abiraterone** | 0.147→<0.01 | 9.60 | PPB candidate | clinical fup<0.01 vs predicted 0.147, hepatic CYP3A4+SULT2A1, correct direction (parallel to oxybutynin) |
| **progesterone** | 0.04 | 9.27 | PPB candidate (borderline) | fup=0.04, hepatic CYP3A4, correct direction; extreme first-pass confound |
| **fesoterodine** | 0.03 | 7.66 | PPB but PRODRUG AD-out | parent fup 0.03 but active metabolite (5-HMT) fup=0.50; out of AD |

**PPB-related candidates for T12 literature search**: **4 drugs** — paroxetine, oxybutynin, abiraterone, progesterone

(fesoterodine excluded: PRODRUG flag; active metabolite fup=0.50 not PPB-related)

### Excluded from PPB despite fup < 0.10

| drug | fup | meta_fold | reason excluded |
|---|---|---|---|
| lenacapavir | 0.002 | 37.98 | P-gp intestinal secretion is primary CL; out of AD; correction direction wrong |
| posaconazole | 0.02 | 4.92 | Out of AD (HIGH_MW); absorption/P-gp mechanism; UGT not hepatic-CYP |
| budesonide | 0.125 | 14.95 | Formulation (EC) F=9% dominates; fu correction insufficient |
| selegiline | 0.418 | 20.22 | Predicted fup >0.10; extreme first-pass F=4-10% dominates |

### Not applicable — non-PPB mechanisms

| drug | fup | meta_fold | primary mechanism |
|---|---|---|---|
| lenacapavir | 0.002 | 37.98 | P-gp intestinal secretion; out of AD |
| acamprosate | 0.580 | 33.82 | Renal excretion; no hepatic CL pathway |
| methylphenidate | 0.950 | 24.73 | CES1 hydrolysis; high fup; enzyme not in registry |
| selegiline | 0.418 | 20.22 | Extreme first-pass (F=4–10%); primary mechanism |
| budesonide | 0.125 | 14.95 | Formulation: enteric-coated F=9%; first-pass |
| fesoterodine | 0.030 | 7.66 | PRODRUG (AD-out); active metabolite fup=0.50 |
| fluvoxamine | 0.215 | 7.46 | CYP1A2 autoinhibition; fup above threshold |
| vonoprazan | 0.150 | 7.38 | Lysosomal trapping (basic drug); fup above threshold |
| ramelteon | 0.180 | 6.23 | Extreme first-pass F=1.8%; fup above threshold |
| clopidogrel | 0.218 | 5.15 | CES1 hydrolysis (85%); prodrug; addressed by B-03 |
| posaconazole | 0.020 | 4.92 | Absorption/formulation; P-gp; out of AD |
| sumatriptan | 0.860 | 4.80 | MAO-A clearance not in enzyme registry; high fup |
| azacitidine | 0.979 | 4.01 | Cytidine deaminase + spontaneous hydrolysis; F=11–13%; high fup |
| venlafaxine | 0.730 | 3.80 | CYP2D6 variability; high fup |
| morphine | 0.650 | 3.17 | UGT2B7; high fup; special routing already applied |

### Totals

- **PPB-related candidates (T12 targets)**: 4 drugs — paroxetine, oxybutynin, abiraterone, progesterone
- **Not applicable**: 15 drugs
- **Total audited**: 19

### Key patterns in not_applicable group

1. **Enzyme not in registry (4 drugs)**: methylphenidate (CES1), clopidogrel (CES1/CYP2C19), sumatriptan (MAO-A), morphine (UGT2B7) — systematic gap in the enzyme_affinity schema
2. **Extreme first-pass / formulation (3 drugs)**: selegiline (F=4–10%), budesonide (EC F=9%), ramelteon (F=1.8%) — F estimation is a second systematic gap
3. **Wrong clearance route (2 drugs)**: acamprosate (renal), azacitidine (cytidine deaminase) — structural absence of alternative CL routes
4. **Absorption/formulation + P-gp (2 drugs)**: lenacapavir (P-gp secretion), posaconazole (gastric-pH-dependent absorption)
5. **Autoinhibition kinetics (2 drugs)**: fluvoxamine (CYP1A2), paroxetine (CYP2D6) — time-dependent inhibition not modeled
6. **Distribution issue (1 drug)**: vonoprazan (lysosomal trapping)
7. **Prodrug entity mismatch (1 drug)**: fesoterodine (parent vs. metabolite Cmax)

---

## T11 Direction Verification Note

For drugs with fup < 0.10 and hepatic-CL-dominant, the B-11 correction direction must be verified:
- The WS model: `CL_hepatic = Q × fup × CLint / (Q + fup × CLint)`
- At low fup (drug is mostly bound), `fup × CLint << Q` → `CL_hepatic ≈ fup × CLint` → low CL → drug accumulates → Cmax HIGH
- Engine over-predicts Cmax → this means CL is UNDER-predicted → which means `fup × CLint` in model < true
- B-11 applies `fup_effective = fup × fu_correction_liver` where `fu_correction_liver > 1`
- This RAISES `fup_effective × CLint` → RAISES CL_hepatic → REDUCES Cmax → correction in right direction ✓
- For abiraterone and oxybutynin: predicted fup overshoots clinical (0.147 vs. <0.01; 0.20 vs. ~0.01). The B-11 mechanism multiplies *predicted* fup by `fu_correction_liver ≥ 1` — it does NOT swap predicted fup for clinical fup. Starting from predicted fup, raising `fup_effective` increases CL → reduces Cmax → moves over-prediction toward observed. Direction is therefore correct for both drugs. The XGBoost fup overshoot for highly-bound drugs is a separate upstream concern that does not invalidate B-11 candidacy.

---

*Sources consulted: FDA Prescribing Information for all 19 drugs via accessdata.fda.gov and DailyMed; PubMed/PMC for mechanism reviews; search queries and results archived above in session transcript.*
