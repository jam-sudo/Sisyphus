# Prodrug Activation v3 — Literature Deliverable (T1 Pattern)

**Date**: 2026-04-29
**Spec**: `docs/superpowers/specs/2026-04-29-prodrug-activation-v3-design.md`
**Output of**: research/literature phase only; no code, registry, yaml, or test changes.
**Doctrine**: §4 (mean-value, CV, exhaustiveness, template) + §5 (per-item gates).

## Disposition Summary

| # | Item | Disposition | Outcome |
|---|---|---|---|
| 1 | BH4 active CL/Vd (sepiapterin) | **ceiling_accepted** | F_sapropterin absolute primary not located (FDA Kuvan label: "Absolute bioavailability… is not known"). Per §5.1 step 2 fallback: v1 Vd=150 L retained as known-wrong placeholder; v1 CL=40 L/h retained. Uncertainty bound documented. |
| 2 | GS-441524 active CL/Vd (remdesivir) | **literature_applied** | 3 eligible human IV-remdesivir popPK sources; geometric mean CL=17.4 L/h, V=535 L; CV via max(BSV, inter-study GSD). |
| 3 | R406 active CL/Vd (fostamatinib) | **literature_applied** | IV microdose human study reported in Matsukane 2022 review of Baluom/Rigel data; CL=15.7 L/h, Vss=256 L; CV from Vss SD (≈36%). |
| 4 | tebipenem active CL/Vd (tebipenem_pivoxil) | **ceiling_accepted** | Eckburg 2019 reports oral V/F=46.2 L, CL/F=39.1 L/h; F absolute not located in primary literature (50-60% bracket in Eckburg is urinary-recovery surrogate, not IV-comparator F). Per §4.1 Gap 5 strict, V/F substitution is rejected. v1 Vd=50, CL=17 retained as least-bad placeholder; v1 CL value flagged as inconsistent with Eckburg CL/F (factor ~2.3 discrepancy). |
| 5 | SPR primary proteomic abundance | **ceiling_accepted** | No quantitative MS-based human SPR abundance (pmol/mg microsomal protein or equivalent) located in PubMed, Google Scholar, Human Protein Atlas, or proteomics atlas literature. Human Protein Atlas: SPR is "low tissue specificity" Tau=0.35, "non-tissue-enriched", broad cytoplasmic. Available data is enzyme activity in rat (Wu 2020 review citing 130 pmol/h/mg liver — same-entity fail and activity not abundance). v2 class-estimated values retained (liver 1e5, gut 3e3, kidney 3e4). |
| 6 | CES2/tebipenem direct Vmax/Km | **ceiling_accepted** | No in vitro tebipenem-pivoxil hydrolysis Vmax/Km with recombinant or human-tissue CES2 located. Gupta 2023 ADME study identifies hydrolysis in "enterocytes of the gastrointestinal tract via intestinal esterases" without isoform identification (CES1 vs CES2 not distinguished in any primary source). v2 class-extrapolated CES2 affinity in registry retained. |

**Net code-change scope (if v3 PR is opened)**: Items 2 and 3 only update `data/sbi/prodrug_activation_registry.json` (R406 and GS-441524 entries). All other items contribute documentation-only entries (citation, doctrine_path, disposition_state, ceiling_rationale) without numeric value changes.

---

## Item 1 — BH4 active CL/Vd (sepiapterin)

- **v1/v2 state**:
  - Vd_L = 150 L (cv 0.3); CL_per_h = 40 L/h (cv 0.35); fup = 0.23 (cv 0.3)
  - Source: `data/sbi/prodrug_activation_registry.json` (BH4 entry)
  - Citation field: Gao 2024 PMC11597218 — but this is a clinical Cmax observation, not popPK CL/Vd primary.
- **T1 flag**: 1.5–50× off literature per v2 T1 caution table; v1 placeholder unsupported by primary popPK source.
- **Search**:
  - terms: ["tetrahydrobiopterin BH4 popPK pharmacokinetics human IV Vss", "sapropterin oral bioavailability F human", "sapropterin Feillet 2008", "sepiapterin pharmacokinetics human", "Kuvan absolute oral bioavailability FDA label", "plasma tetrahydrobiopterin pharmacokinetics oral administration"]
  - databases: PubMed, Google Scholar, FDA accessdata, EMA EPAR, DrugBank, Wikipedia/UniProt
  - N candidates reviewed: 6 (Feillet 2008 Clin Pharmacokinet 47:817-825; Kuvan FDA label rev 2014/2020; Kuvan EMA EPAR; Fiege 2004 Mol Genet Metab 81:45-51; LactMed Sapropterin monograph; Schircks 2009 sapropterin review)
- **Selected source(s)**:
  - Feillet F, Clarke L, Meli C, et al. Pharmacokinetics of sapropterin in patients with phenylketonuria. *Clin Pharmacokinet*. 2008;47(12):817–825. DOI: 10.2165/0003088-200847120-00006. (78 PKU patients, oral 5/10/20 mg/kg, 2-comp first-order input/elim with baseline endogenous BH4. **Apparent CL/F = 2100 L/h/70 kg; central V/F = 8350 L/70 kg; t½ = 6.69 ± 2.29 h.** BSV not extracted from abstract; full table not accessed.)
  - Kuvan FDA Highlights of Prescribing Information (2014, 2020 revisions, accessdata.fda.gov labels 022181s013/s020). Quote: "Absolute bioavailability or bioavailability for humans after oral administration is not known."
  - Kuvan EMA EPAR (consistent with FDA label; explicit "absolute bioavailability not known").
  - Fiege B, Ballhausen D, Kierat L, et al. Plasma tetrahydrobiopterin and its pharmacokinetic following oral administration. *Mol Genet Metab*. 2004;81(1):45–51. DOI:10.1016/j.ymgme.2003.09.014. (4 healthy adults, BH4 not sapropterin salt; reports Tmax 1–4 h, Cmax 258.7–259.0 nmol/L biopterin at 10 mg/kg, t½ 3.3–5.1 h, AUC 1708–1958 nmol·h/L. **No IV data, no F estimate.**)
- **Doctrine application**:
  - Mean rule attempt (§4.1, oral popPK): V/F ÷ F → central V. Feillet V/F = 8350 L/70 kg = ~119 L/kg apparent; if F = 0.20 (hypothetical) → V = 1670 L/kg = enormous; if F = 0.05 → 6680 L/kg. **F primary citation NOT found**. Per §4.1 Gap 5 + §5.1 step 3, F substitution from estimates is rejected.
  - CV rule: not applied — value not updated.
  - Same-entity check: Feillet measures sapropterin (BH4 dihydrochloride salt). Per §4.1 same-entity, salt forms accepted. **Pass.**
  - Sub-decision (§5.1 fallback chain):
    1. Primary F citation found → ✗ NOT FOUND. FDA + EMA both state F is unknown.
    2. Strict downgrade to ceiling_accepted: ✓ APPLIED.
    3. Geometric mean / CV inflation: rejected (would violate §4.2 Q4).
- **Sub-decisions resolved**:
  - F_sapropterin: **NOT located in primary literature**; FDA/EMA explicit "not known".
  - 2-comp model selection (Vss vs Vc): moot (no recovery possible without F).
  - Fiege 2004 BH4 oral: cannot be used — no F either.
- **Final values**: **no change — v1 retained**.
  - Vd_L: 150 L (cv 0.3) [retained, KNOWN INCORRECT per T1 1.5–50× off Feillet apparent V/F = 8350 L; BUT cannot be replaced without F]
  - CL_per_h: 40 L/h (cv 0.35) [retained; same gap — Feillet apparent CL/F = 2100 L/h is upper bound; central CL = 2100 × F]
- **Disposition**: **ceiling_accepted**
- **Ceiling rationale (per §5.1 + §6.2 dual-location requirement)**:
  - (a) Primary popPK V/F + CL/F values located: Feillet 2008 V/F = 8350 L/70 kg, CL/F = 2100 L/h/70 kg.
  - (b) Primary F citation NOT located. FDA Kuvan label (2014, 2020) and EMA Kuvan EPAR both explicitly state "Absolute bioavailability… is not known". No primary IV BH4 / sapropterin human disposition study identified within the §4.3 source corpus (Fiege 2004 oral-only, no IV crossover).
  - (c) Uncertainty bound: central V is somewhere in [150 L (v1 placeholder), 8350 L (apparent V/F upper bound)], which spans 56× — rationalizes T1 caution "1.5–50× off literature".
  - (d) v1 Vd=150 retained as least-bad placeholder, NOT as endorsed value. Acknowledged known-wrong.
- **Sub-decisions explicit**: F substitution from estimates, CV inflation, F geometric-mean across animal studies — **all REJECTED** per §4.1 Gap 5 + §4.2 Q4.

---

## Item 2 — GS-441524 active CL/Vd (remdesivir)

- **v1/v2 state**:
  - Vd_L = 35 L (cv 0.3); CL_per_h = 10 L/h (cv 0.3); fup = 0.5 (cv 0.3)
  - Source: `data/sbi/prodrug_activation_registry.json` (GS-441524 entry)
  - Citation field: Humeniuk 2020 PMC8007387.
- **T1 flag**: 2.5× off; per §5.2 verification of Sukeishi 2022 species/route required (T1 cited but eligibility unverified).
- **Search**:
  - terms: ["GS-441524 human pharmacokinetics popPK", "remdesivir GS-441524 metabolite plasma kinetics IV human", "Sukeishi 2022 GS-441524", "Tamura 2023 GS-441524 popPK COVID", "Leegwater 2022 remdesivir COVID popPK", "Humeniuk 2020 remdesivir healthy subjects PK"]
  - databases: PubMed, Google Scholar, ASCPT/Wiley journals, ASM journals, PMC
  - N candidates reviewed: 7 (Sukeishi 2022; Tamura 2023; Leegwater 2022; Humeniuk 2020; Rasmussen 2022 cross-species; Yan 2020 ACS Med Chem Lett comparison; Frontiers 2022 preclinical PK)
- **Selected source(s)** (3 eligible human IV-remdesivir popPK; species GS-441524 measured):
  1. **Sukeishi A, Itohara K, Yonezawa A, et al.** Population pharmacokinetic modeling of GS-441524, the active metabolite of remdesivir, in Japanese COVID-19 patients with renal dysfunction. *CPT Pharmacometrics Syst Pharmacol*. 2022;11(1):94–103. DOI: 10.1002/psp4.12736. (n=37 Japanese COVID-19, IV remdesivir, 1-comp model for serum GS-441524; 190 concentration points. Numerical CL/V values not in abstract; eligible by route/species verification.)
  2. **Tamura R, Irie K, Nakagawa A, et al.** Population pharmacokinetics and exposure-clinical outcome relationship of remdesivir major metabolite GS-441524 in patients with moderate and severe COVID-19. *CPT Pharmacometrics Syst Pharmacol*. 2023;12(4):513–521. DOI: 10.1002/psp4.12936. (n=39, IV remdesivir 200/100 mg, 1-comp model. **CL = 11.0 L/h (ISV 43.0%); V = 271 L (ISV 58.1%); t½ = 17.1 h.**)
  3. **Leegwater E, Moes DJAR, Bosma LBE, et al.** Population Pharmacokinetics of Remdesivir and GS-441524 in Hospitalized COVID-19 Patients. *Antimicrob Agents Chemother*. 2022;66(6):e00254-22. DOI: 10.1128/aac.00254-22. (n=17 hospitalized, IV remdesivir 1–2 h infusion, 1-comp model both compounds. **GS-441524 CL = 27.6 L/h (IIV 47.4%); V = 1060 L (IIV 42.9%).**)
  - Humeniuk 2020 (Clin Transl Sci, healthy-subject phase 1 IV remdesivir): supportive, but populated as v1 citation; replaced by 3 popPK sources for primary mean.
- **Doctrine application**:
  - **Mean rule (§4.1)**: All 3 sources are 1-comp models for IV remdesivir → metabolite GS-441524 plasma; total CL and Vd unambiguous (1-comp). Geometric mean across 2 quantitative sources (Tamura 2023 + Leegwater 2022; Sukeishi 2022 numerical not extracted from abstract): CL = √(11.0 × 27.6) = **17.4 L/h**; V = √(271 × 1060) = **535 L**.
  - **CV rule (§4.2)**: max(BSV, inter-study GSD). Inter-study GSD on log scale: log(CL) range 2.40 – 3.32 → log-σ ≈ 0.46 → CV_inter-study ≈ 0.49. BSV (within-study, max across 2 studies) = max(43.0%, 47.4%) = **0.474**. Inter-study GSD wins: **CV = 0.49** (CL); for V: log-range 5.60 – 6.97 → log-σ ≈ 0.69 → **CV_V = 0.79**; max(BSV V, inter-study) = max(58.1%, 79%) = **0.79**.
  - **Same-entity check**: human-only (rejected: Rasmussen 2022 cross-species; cat/feline FIP studies; Frontiers 2022 preclinical mouse/dog/monkey). All 3 selected sources are human + IV remdesivir + GS-441524 species. **Pass.**
  - Sukeishi 2022 §5.2 verification: confirmed human, IV remdesivir, GS-441524 species, 1-comp. **Eligible.** (Numerical values not extracted from abstract; if v3 implementer accesses full text, could enter as 3rd point and update geometric mean.)
- **Sub-decisions resolved**:
  - 1-comp vs 2-comp: all 3 sources 1-comp; Vss-equivalent acceptable per §4.1 Gap 2.
  - Inter-study CV vs BSV: max() rule applied; inter-study GSD dominates for CL.
  - Sukeishi numerical inclusion: deferred (full-text access; if obtained, geometric mean updates marginally — both v1 and Tamura/Leegwater bound it).
- **Final values** (geometric mean of 2 quantified popPK sources; round per existing registry precision):
  - **Vd_L: mean = 535 L, cv = 0.79**
  - **CL_per_h: mean = 17.4 L/h, cv = 0.49**
  - fup unchanged (not in scope of v3 — Item 2 is CL/Vd only)
- **Disposition**: **literature_applied**
- **Citations to enter in registry** (`citation` + `source_dbs_searched`):
  - Primary: Tamura 2023 (DOI 10.1002/psp4.12936) + Leegwater 2022 (DOI 10.1128/aac.00254-22). Sukeishi 2022 (DOI 10.1002/psp4.12736) eligible same-entity but not numerically incorporated pending full-text.
  - `source_dbs_searched`: ["PubMed", "Google Scholar", "ASCPT/Wiley", "ASM/PMC"]
  - `n_candidates_reviewed`: 7

---

## Item 3 — R406 active CL/Vd (fostamatinib)

- **v1/v2 state**:
  - Vd_L = 250 L (cv 0.3); CL_per_h = 28 L/h (cv 0.35); fup = 0.02 (cv 0.3)
  - Source: `data/sbi/prodrug_activation_registry.json` (R406 entry)
  - Citation field: Baluom 2013 PMC3703230.
- **T1 flag**: 1.8× off; expected cleanest item — IV microdose human R406 PK is reported in Matsukane 2022 review of Rigel/Baluom data.
- **Search**:
  - terms: ["R406 fostamatinib metabolite IV pharmacokinetics human", "PMC9250994 R406", "tamatinib R406 popPK", "Baluom 2013 fostamatinib PK healthy subjects", "Matsukane fostamatinib clinical pharmacokinetics review"]
  - databases: PubMed, Google Scholar, Springer/Clin Pharmacokinet, BJCP, PMC
  - N candidates reviewed: 5 (Matsukane 2022 review = PMC9250994; Baluom 2013 BJCP oral phase-1; Sweeny 2010 metabolism paper; renal/hepatic-impairment phase-1 studies; digoxin-DDI study)
- **Selected source(s)**:
  - **Matsukane R, Suetsugu K, Hirota T, Ieiri I.** Clinical Pharmacokinetics and Pharmacodynamics of Fostamatinib and Its Active Moiety R406. *Clin Pharmacokinet*. 2022;61(7):955–972. DOI: 10.1007/s40262-022-01135-0 (PMC9250994). Reports R406 IV microdose data (Rigel-conducted study): "Absolute bioavailability was estimated at 55% according to the results of oral administration of fostamatinib 150 mg and intravenous administration of R406 100 µg. In this micro-dose intravenous study, t1/2 and clearance were 15.3 h and 15.7 L/h, respectively." Vss reported as **256 ± 92 L** in figure/table 2 of review. Citation traces to Baluom 2013 BJCP for the oral arm + the Rigel IV-microdose unpublished/internal data tabulated in the review.
  - Baluom M, Grossbard EB, Mant T, Lau DT-W. Pharmacokinetics of fostamatinib, a spleen tyrosine kinase (SYK) inhibitor, in healthy human subjects following single and multiple oral dosing in three phase I studies. *Br J Clin Pharmacol*. 2013;76(1):78–88. DOI: 10.1111/bcp.12048 (PMC3703230). Provides oral fostamatinib + R406 data (already v1 citation).
- **Doctrine application**:
  - **Mean rule (§4.1)**: IV microdose direct R406 administration → central CL and Vss from peer-reviewed Clin Pharmacokinet review with explicit numerical values. **CL = 15.7 L/h; Vss = 256 L.** No F division needed (IV direct).
  - **CV rule (§4.2)**: BSV from Vss SD: 92/256 = 0.36 (CV of Vss across IV-microdose subjects). For CL: explicit BSV not in extract; class default 0.30 applied (§4.2 priority 3). For V, CV = 0.36 from reported SD.
  - **Same-entity check**: human IV microdose R406, peer-reviewed Clin Pharmacokinet review tracing to Rigel-conducted clinical study. **Pass.**
- **Sub-decisions resolved**:
  - 1-comp vs 2-comp: review reports terminal t½ = 15.3 h with CL = 15.7 L/h and Vss = 256 L → consistent with multi-compartment but Vss is reported (§4.1 prefers Vss when 2-comp).
  - Source provenance: review (Matsukane 2022) is peer-reviewed and traces IV microdose data to a primary clinical study; per §4.1 Gap 3 acceptance "peer-reviewed journal article" — Clin Pharmacokinet review is eligible. Direct primary IV-microdose paper (Rigel filing) not located in PubMed; review citation is the cleanest available source.
- **Final values**:
  - **Vd_L: mean = 256 L, cv = 0.36**
  - **CL_per_h: mean = 15.7 L/h, cv = 0.30** (class default, BSV not in extract)
- **Disposition**: **literature_applied**
- **Citations to enter in registry**:
  - Primary: Matsukane 2022 (DOI 10.1007/s40262-022-01135-0); supplementary Baluom 2013 (DOI 10.1111/bcp.12048).
  - `source_dbs_searched`: ["PubMed", "Google Scholar", "Springer", "BJCP/Wiley", "PMC"]
  - `n_candidates_reviewed`: 5

---

## Item 4 — tebipenem active CL/Vd (tebipenem_pivoxil)

- **v1/v2 state**:
  - Vd_L = 50 L (cv 0.3); CL_per_h = 17 L/h (cv 0.3); fup = 0.5 (cv 0.3)
  - Source: `data/sbi/prodrug_activation_registry.json` (tebipenem entry)
  - Citation field: Eckburg 2019 PMC6709501 (tebipenem oral PK).
- **T1 flag**: "mostly OK"; but per §5.4 full §4.3 exhaustiveness applies — no lazy ceiling.
- **Search**:
  - terms: ["tebipenem pivoxil pharmacokinetics popPK human", "tebipenem oral bioavailability F human", "tebipenem IV human disposition", "SPR-994 tebipenem popPK", "tebipenem pivoxil hydrobromide phase 1", "Eckburg tebipenem food effect", "Gupta tebipenem mass balance ADME", "tebipenem pediatric popPK ME1211", "Sato Kijima tebipenem"]
  - databases: PubMed, Google Scholar, ASM/PMC, Wiley/ASCPT, NEJM, ResearchGate, accessdata.fda.gov (FDA label not yet issued — application complete-response 2022)
  - N candidates reviewed: 9 (Eckburg 2019 SAD/MAD healthy adults; Gupta 2023 ADME mass balance; Gupta 2022 bioequivalence; Eckburg 2022 renal impairment; Pediatric pop PK Sato 2009 = ME1211; Kato 2010 OATP intestinal absorption; Yu 2021 Chinese phase 1 fine granules; food-effect study; complicated UTI phase-3 NEJM 2022 efficacy only)
- **Selected source(s)**:
  - **Eckburg PB, Jain A, Walpole S, et al.** Safety, Pharmacokinetics, and Food Effect of Tebipenem Pivoxil Hydrobromide after Single and Multiple Ascending Oral Doses in Healthy Adult Subjects. *Antimicrob Agents Chemother*. 2019;63(10):e00618-19. DOI: 10.1128/AAC.00618-19. (Healthy adults, SAD 100–900 mg + MAD 300/600 mg q8h IR formulation. **300 mg fasted (n=6): CL/F = 39.1 L/h (CV 36.3%); V/F = 46.2 L (CV 49.6%); t½ = 0.8 h (CV 21.0%).** Extensive dose/formulation/food-state table reported. Quoted in Discussion: "TBPM-PI-HBr provides high tebipenem bioavailability (50% to 60%)" — but this is a **urinary-excretion derived F estimate** (35.0–59.2% urine recovery as unchanged tebipenem in fasted SAD), **not an absolute IV-comparator F**.)
  - **Gupta VK, Maier G, Gasink L, et al.** Absorption, Metabolism, and Excretion of [14C]-Tebipenem Pivoxil Hydrobromide (TBP-PI-HBr) in Healthy Male Subjects. *Antimicrob Agents Chemother*. 2023;67(4):e01509-22. DOI: 10.1128/aac.01509-22 (PMC10112213). (n=8 male subjects, single 600 mg [14C]-TBP-PI-HBr oral; tebipenem 54% of plasma radioactivity AUC; LJC 11562 ring-open inactive metabolite >10%; intact prodrug ~0.58% in feces. **No IV human data; no absolute F estimate.**)
  - Sato R, Kijima K, Tagi K, et al. Population pharmacokinetics of tebipenem pivoxil (ME1211) in pediatric patients. (Pediatric — same-entity human, but pediatric covariate; not preferred for adult Vd/CL central estimate.)
  - **No human IV tebipenem study located.** Confirmed via search exhaustion.
- **Doctrine application**:
  - **Mean rule (§4.1)**: All eligible human studies are oral. Per §4.1 oral acceptance "iff F is separable" — strict F primary citation required. **Search outcome: F absolute primary citation NOT located.**
    - Eckburg 2019 50–60% F is urinary-recovery surrogate (no IV comparator in that study or any companion study).
    - Animal F (mouse 71.4%, rat 59.1%, dog 34.8%, monkey 44.9%) cited in earlier reviews — same-entity strict (§4.1 Gap 1) **rejects** species extrapolation.
  - Per §5.4 + §4.1 Gap 5: V/F substitution rejected. Strict downgrade to ceiling_accepted.
  - **Same-entity check**: salt forms OK (TBPM-PI-HBr → TBPM); active species = tebipenem. **Pass.** But oral-route gating (F primary required) unfulfilled.
- **Sub-decisions resolved**:
  - F_tebipenem absolute: **NOT located** in primary literature. 50–60% urinary-recovery bracket in Eckburg 2019 is not absolute F.
  - "Mostly OK" baseline: rejected as lazy ceiling per §5.4. Full §4.3 exhaustiveness applied, ceiling reached on doctrine grounds (F gap), not source gap.
  - v1 CL=17 L/h vs Eckburg CL/F=39 L/h: v1 is ~2.3× lower than CL/F → consistent with implicit F=0.43 division at v1 authoring time (without primary citation). Per §4.1 Gap 5, this implicit substitution would be ad-hoc and is rejected; v1 CL=17 retained as placeholder NOT endorsed.
  - v1 Vd=50 L vs Eckburg V/F=46.2 L: coincidentally close but cannot be endorsed as central V (would require F=1.0, contradicting urinary 50–60%).
- **Final values**: **no change — v1 retained**.
  - Vd_L: 50 L (cv 0.3) [retained as placeholder; numerically close to Eckburg V/F=46.2 by coincidence, not endorsement]
  - CL_per_h: 17 L/h (cv 0.3) [retained as placeholder; differs from Eckburg CL/F by ~2.3× — implicit F~0.43 division, ad-hoc, not endorsed]
- **Disposition**: **ceiling_accepted**
- **Ceiling rationale**:
  - (a) Primary apparent (V/F, CL/F) values located: Eckburg 2019, Gupta 2022 bioequivalence (consistent ranges).
  - (b) F absolute primary citation NOT located. No human IV tebipenem study exists in the §4.3 corpus. Eckburg 2019's "50–60% bioavailability" sentence is a urinary-recovery summary derived from urine fractions (35.0–59.2% across SAD doses), not an IV-vs-oral comparison.
  - (c) Uncertainty bound: central V is in [V/F × F_low, V/F × F_high] = [46.2 × 0.35, 46.2 × 0.60] = [16.2 L, 27.7 L] — this assumes urinary recovery proxies F. v1 Vd=50 L sits 1.8–3.1× above this rough bound; T1 "mostly OK" is consistent with this proximity.
  - (d) v1 retained as placeholder pending an IV human tebipenem study or an absolute-F crossover study being conducted.
- **Sub-decisions explicit**: F substitution from urinary-recovery surrogate, animal F values, geometric mean of CR formulation F estimates — **all REJECTED** per §4.1 strict.

---

## Item 5 — SPR primary proteomic abundance

- **v1/v2 state** (per `data/physiology/reference_man.yaml` planned v2 entries; current main does not yet have SPR — these are values planned for v2 PR #7 per §5.5 quoted in v3 spec):
  - liver SPR = 1e5 pmol (organ-aggregate, post-IVIVE convention)
  - gut_wall SPR = 3e3 pmol
  - kidney SPR = 3e4 pmol
  - cv = class default 0.5
  - Source: class-estimated from CYP-class abundance pattern × tissue-distribution heuristic
- **T1 flag**: class-estimated (no primary proteomic measurement); per §5.5 T1 caution.
- **Search**:
  - terms: ["sepiapterin reductase SPR human liver proteomic abundance MS", "SPR enzyme expression kidney gut human quantitative", "Wegler enzyme atlas SPR", "ProteomicsDB SPR human", "BH4 biosynthesis enzyme abundance pmol/mg", "sepiapterin reductase tissue distribution liver kidney expression mRNA protein", "SPR sepiapterin reductase quantitative proteomic abundance human liver fmol picomol", "human protein atlas SPR sepiapterin reductase", "SPR 28 kDa human liver microsome cytosol pmol abundance"]
  - databases: PubMed, Google Scholar, Human Protein Atlas (proteinatlas.org), UniProt (P35270), GeneCards, OMIM, ProspecBio, ScienceDirect/Wegler reviews, J Cell Mol Med (Wu 2020)
  - N candidates reviewed: 10 (Wu 2020 J Cell Mol Med review; Human Protein Atlas SPR tissue page; UniProt P35270; OMIM 182125; GeneCards SPR; Wegler 2017 TXP methodology; Prasad 2013/2018 hepatic non-CYP proteomic; Achour 2017 hepatic UGT/SULT atlas — does not include SPR; Vildhede 2018 hepatocyte proteome; Frontiers 2023 peripheralized SPR inhibition pharmacology paper)
- **Selected source(s)**:
  - **Wu Y, Chen P, Sun L, et al.** Sepiapterin reductase: Characteristics and role in diseases. *J Cell Mol Med*. 2020;24:9495–9506. DOI: 10.1111/jcmm.15608 (PMC7520308). Tissue distribution review. **Reports rat SPR enzyme activity 130 pmol/h/mg liver and 80 pmol/h/mg erythrocyte; no detected activity in intestine and muscle. Human distribution noted "high expression in liver, kidney, and colon" (GTEx); no quantitative pmol/mg human values.**
  - **Human Protein Atlas, SPR (ENSG00000116096) tissue expression page.** https://www.proteinatlas.org/ENSG00000116096-SPR/tissue. **"Low tissue specificity" Tau = 0.35; "non-tissue-enriched" cluster 70 (Protein processing, housekeeping); broad cytoplasmic granular pattern in most tissues. No tissue-specific nTPM values quoted in the summary; no protein abundance in pmol/mg.**
  - **No quantitative MS-based human SPR abundance (pmol/mg microsomal or cytosolic protein, or pmol/g organ) located** in PubMed/Google Scholar/Wegler enzyme atlas literature/Achour 2021 (which does not list SPR).
- **Doctrine application (§4.5)**:
  - **Mean rule**: primary MS-based proteomic measurement required. **Not located.** Available data is (a) rat enzyme activity, not human abundance (same-entity fail); (b) qualitative tissue-distribution descriptors (HPA "low specificity"); (c) GTEx mRNA TPM (transcript ≠ protein abundance, not eligible per §4.5 doctrine that requires proteomic primary measurement).
  - **Unit conversion**: not applicable (no primary value to convert).
  - **CV rule**: not applied — value not updated. Class default cv=0.5 retained.
  - **Same-entity check**: human SPR isoform required. Wu 2020 rat data → fail. HPA human qualitative → no quantitative value. **Quantitative same-entity human source NOT located.**
- **Sub-decisions resolved**:
  - Unit basis (per mg microsomal protein vs per g organ): moot.
  - Inter-individual CV vs inter-study GSD: moot.
  - mRNA-as-proxy substitution: REJECTED per §4.5 (mass-spec proteomic primary required; mRNA does not satisfy doctrine).
- **Final values**: **no change — v2-planned class-estimated values retained**.
  - liver SPR = 1e5 pmol (cv 0.5)
  - gut_wall SPR = 3e3 pmol (cv 0.5)
  - kidney SPR = 3e4 pmol (cv 0.5)
  - (Note: these enter `reference_man.yaml` only when v2 PR #7 merges; v3 retains them unchanged.)
- **Disposition**: **ceiling_accepted**
- **Ceiling rationale**:
  - (a) §4.3 source corpus exhaustively searched (PubMed, Google Scholar, Human Protein Atlas, UniProt, Wegler 2017+ enzyme atlas literature, Achour 2021 hepatic non-CYP atlas, ProteomicsDB).
  - (b) No quantitative MS-based human SPR abundance (pmol/mg or equivalent) located in primary literature. Human Protein Atlas provides qualitative descriptors only (Tau=0.35, "low tissue specificity").
  - (c) Available numerical data (Wu 2020 rat 130 pmol/h/mg liver enzyme **activity**) is rejected on two doctrine grounds: (i) same-entity strict fails (rat ≠ human), (ii) activity ≠ abundance.
  - (d) v2-planned class-estimated abundances retained; T1 caution closure documented as data-quality frontier — appropriate v4 follow-up: targeted MS-based SPR proteomic study or curated extract from a future enzyme-atlas update including SPR.

---

## Item 6 — CES2/tebipenem direct Vmax/Km

- **v1/v2 state**:
  - `enzyme_affinity_for_conversion["CES2"]` for tebipenem_pivoxil = class-extrapolated affinity (Vmax/Km) per v2 spec §10. Specific numerical mean/CV not yet committed in main; v2 plan Task 0 deliverable was "TBD".
  - Source: class extrapolation from generic CES2-substrate kinetics (e.g., dabigatran etexilate Km=5.5 µM, Vmax=71.1 pmol/min/mg from Laizure 2014 was a candidate analog).
- **T1 flag**: class-extrapolated; per §5.6 T1 caution requires primary in vitro CES2 + tebipenem-pivoxil kinetics OR strict ceiling.
- **Search**:
  - terms: ["tebipenem pivoxil hydrolysis CES2 in vitro Vmax Km recombinant", "tebipenem pivoxil esterase kinetics human", "carboxylesterase CES2 tebipenem CLint", "SPR-994 hydrolysis CES isoform", "tebipenem activation human liver microsomes", "tebipenem pivoxil enterocyte CES2 carboxylesterase isoform identification", "tebipenem ME1211 hydrolysis intestinal microsome kinetics in vitro Vmax Km", "Kamiya Imai tebipenem"]
  - databases: PubMed, Google Scholar, ASM/PMC, ACS Pubs, MDPI, ChEMBL, DrugBank
  - N candidates reviewed: 10 (Gupta 2023 ADME mass balance PMC10112213; Imai 2010 review of intestinal CES; Kamiya/Imai-style reviews — no specific tebipenem-pivoxil entry; Laizure 2014 dabigatran + CES2 analog; Wang 2014 CES2 substrate review; Eckburg 2019 oral PK doesn't address in vitro; Zhang 2018 CES2 expression; Acta Pharmacologica Sinica 2024 CES2 KO mouse rescue; β-lactamase tebipenem hydrolysis Mycobacterium — not human esterase; Kato 2010 OATP intestinal absorption — not hydrolysis kinetics)
- **Selected source(s)**:
  - **Gupta VK, Maier G, Gasink L, et al.** Absorption, Metabolism, and Excretion of [14C]-Tebipenem Pivoxil Hydrobromide (TBP-PI-HBr) in Healthy Male Subjects. *Antimicrob Agents Chemother*. 2023;67(4):e01509-22. DOI: 10.1128/aac.01509-22 (PMC10112213). **Identifies hydrolysis as occurring "in the enterocytes of the gastrointestinal tract via intestinal esterases" — generic; CES1 vs CES2 isoform NOT distinguished. No in vitro kinetic data.**
  - **No primary in vitro Vmax/Km for tebipenem pivoxil + recombinant CES2 (or CES1) located** in PubMed, Google Scholar, ChEMBL, DrugBank.
  - Imai/Kamiya reviews on intestinal carboxylesterases discuss CES2 substrate landscape generally but do not list tebipenem pivoxil-specific kinetics.
  - Acta Pharm Sin 2024 (CES2 KO rescue mouse) does not include tebipenem pivoxil.
  - β-lactamase studies (Mycobacterium tuberculosis BlaC, Km=0.8 µM, kcat=0.03 min⁻¹) are bacterial enzyme — same-entity fail (CES2 is human carboxylesterase, not bacterial β-lactamase; mechanism distinct).
- **Doctrine application (§4.5)**:
  - **Mean rule**: CLint = Vmax/Km from in vitro CES2 incubation (recombinant or human liver/intestinal CES2). **Not located** for tebipenem pivoxil. Class extrapolation (e.g., dabigatran etexilate-CES2 kinetics) is NOT primary for tebipenem pivoxil.
  - **Unit chain**: not applicable (no primary value).
  - **CV rule**: class default cv=0.5 retained.
  - **Same-entity check**: human CES2 isoform required (not CES1, not bacterial, not animal). **No human CES2 + tebipenem pivoxil source identified.**
  - **Isoform specificity**: §5.6 explicitly requires CES2 not CES1; even if a generic "intestinal esterase" source were located (it isn't), CES1/CES2 distinction is unfulfilled.
- **Sub-decisions resolved**:
  - Vmax/Km units → CLint per pmol enzyme: moot.
  - CES2 vs CES1 isoform identification: literature generic ("intestinal esterases"); no isoform attribution. By tissue/pattern, intestinal carboxylesterase activity is dominated by CES2 (vs CES1 dominance in liver), but this is inference, not measurement — not eligible per §4.5 strict.
  - Class extrapolation from dabigatran etexilate or other CES2 substrates: REJECTED per §4.5 (in vitro source must be the prodrug itself).
- **Final values**: **no change — v2 class-extrapolated retained**.
  - `enzyme_affinity_for_conversion["CES2"]`: v2 placeholder retained (Task 0 class extrapolation; specific numerical TBD per v2 plan; v3 does not adjudicate v2-internal value).
- **Disposition**: **ceiling_accepted**
- **Ceiling rationale**:
  - (a) §4.3 corpus exhaustively searched (PubMed, Google Scholar, ASM/PMC, ACS Pubs, MDPI, ChEMBL, DrugBank, Imai/Kamiya intestinal-esterase reviews).
  - (b) No primary in vitro tebipenem-pivoxil + recombinant CES2 (or human-tissue CES2) Vmax/Km located. Gupta 2023 ADME — the most recent and detailed human study — only describes hydrolysis as "via intestinal esterases" without isoform identification or kinetic parameters.
  - (c) Inferential isoform attribution (intestinal-localized → CES2-dominant by tissue pattern) is REJECTED as ad-hoc per §4.5 strict (isoform-specific same-entity).
  - (d) v2 class-extrapolated affinity retained as placeholder; T1 caution closure documented. Appropriate v4 follow-up: in vitro CES1 vs CES2 incubation with TBP-PI substrate (commercially feasible: Cayman/Corning recombinant CES2 + LC-MS readout).

---

## Cross-Item Notes (audit trail)

- **Mechanistic-A discipline**: no values were back-fit to clinical observations. Items 2 and 3 numerical updates are sourced from peer-reviewed popPK/clinical PK literature on the merits of source eligibility under §4.1 + §4.3, not because they help any drug pass a 3-fold gate.
- **Items closing as ceiling_accepted (4 of 6: Items 1, 4, 5, 6)** are valid outcomes per §2 success criterion. They demarcate the v3 → v4 data-quality frontier.
- **All-ceiling avoided**: Items 2 and 3 close as `literature_applied`, satisfying §10 success criterion #1 (≥1 non-ceiling disposition).
- **Net registry change scope**: GS-441524 entry (V, CL, CV); R406 entry (V, CL, CV). Sepiapterin (BH4) and tebipenem entries: no value change, only documentation field additions (`disposition_state`, `ceiling_rationale`, `source_dbs_searched`, `n_candidates_reviewed`).
- **Net physiology change scope**: zero. SPR abundance retained at v2-planned class-estimated values.
- **Per §6.1 test impact**: only sepiapterin/remdesivir/fostamatinib/tebipenem prodrug 3-fold gate test, plus snapshot regen, will see numerical movement. Of those 4, only remdesivir + fostamatinib have value updates that could affect their fold-error. Sepiapterin and tebipenem unchanged numerically (their ceiling rationale is documentation-only).
- **Per §6.2 leak-audit**: `CHANGED_ENZYME_ABUNDANCES = ∅` (Item 5 ceiling — no physiology change → no cross-drug leak); `DRUG_SPECIFIC_CHANGES = {remdesivir, fostamatinib}` (Items 2, 3); 105 of 107 holdout drugs must be byte-identical Cmax under deterministic point-estimate.
- **Per §7.2 contingency**: scenario "all-ceiling" not triggered; v3 is a code+docs PR, not docs-only.

---

## Sources Cited (audit appendix)

- Feillet 2008 *Clin Pharmacokinet* (sapropterin popPK) — DOI 10.2165/0003088-200847120-00006 — https://pubmed.ncbi.nlm.nih.gov/19026037/
- Fiege 2004 *Mol Genet Metab* (BH4 oral PK) — DOI 10.1016/j.ymgme.2003.09.014 — https://pubmed.ncbi.nlm.nih.gov/14728990/
- Kuvan FDA Highlights of Prescribing Information — https://www.accessdata.fda.gov/drugsatfda_docs/label/2014/022181s013lbl.pdf
- Sukeishi 2022 *CPT Pharmacometrics Syst Pharmacol* (GS-441524 Japanese popPK) — DOI 10.1002/psp4.12736 — https://ascpt.onlinelibrary.wiley.com/doi/10.1002/psp4.12736
- Tamura 2023 *CPT Pharmacometrics Syst Pharmacol* (GS-441524 popPK COVID) — DOI 10.1002/psp4.12936 — https://pubmed.ncbi.nlm.nih.gov/36798006/
- Leegwater 2022 *AAC* (remdesivir/GS-441524 hospitalized COVID) — DOI 10.1128/aac.00254-22 — https://pmc.ncbi.nlm.nih.gov/articles/PMC9211420/
- Matsukane 2022 *Clin Pharmacokinet* (Fostamatinib R406 review, IV microdose) — DOI 10.1007/s40262-022-01135-0 — https://pmc.ncbi.nlm.nih.gov/articles/PMC9250994/
- Baluom 2013 *BJCP* (fostamatinib oral phase 1) — DOI 10.1111/bcp.12048 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3703230/
- Eckburg 2019 *AAC* (tebipenem SAD/MAD healthy adults) — DOI 10.1128/AAC.00618-19 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6709501/
- Gupta 2023 *AAC* (tebipenem ADME mass balance) — DOI 10.1128/aac.01509-22 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10112213/
- Wu 2020 *J Cell Mol Med* (SPR review) — DOI 10.1111/jcmm.15608 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7520308/
- Human Protein Atlas — SPR — https://www.proteinatlas.org/ENSG00000116096-SPR/tissue

---

## Summary Table — Item × Disposition (Final)

| # | Item | Disposition | Citation | v2 → v3 Δ |
|---|---|---|---|---|
| 1 | BH4 CL/Vd (sepiapterin) | ceiling_accepted | Feillet 2008 + FDA Kuvan + EMA EPAR | no change (v1 placeholder retained) |
| 2 | GS-441524 CL/Vd (remdesivir) | literature_applied | Tamura 2023 + Leegwater 2022 (geomean) | CL 10→17.4 (cv 0.3→0.49); V 35→535 (cv 0.3→0.79) |
| 3 | R406 CL/Vd (fostamatinib) | literature_applied | Matsukane 2022 (IV microdose) | CL 28→15.7 (cv 0.35→0.30); V 250→256 (cv 0.3→0.36) |
| 4 | tebipenem CL/Vd | ceiling_accepted | Eckburg 2019 (V/F surrogate rejected) | no change (v1 placeholder retained) |
| 5 | SPR primary proteomic abundance | ceiling_accepted | HPA + Wu 2020 review (animal-only) | no change (v2 class-estimated retained: liver 1e5, gut 3e3, kidney 3e4) |
| 6 | CES2/tebipenem direct CLint | ceiling_accepted | Gupta 2023 (generic intestinal esterases) | no change (v2 class-extrapolated retained) |

**Net code-change**: 2 registry entries updated (remdesivir, fostamatinib). 4 items contribute documentation-only `v3_metadata` blocks without numeric changes. Physiology YAML unchanged.

## Per-prodrug Cmax fold-error progression

| Drug | v1 | v2 | v3 | gate (3-fold)? |
|---|---|---|---|---|
| sepiapterin | 5356× over | 4692× over | 4748× over | xfail (Item 1 ceiling) |
| remdesivir | 4.45× under | 4.43× under | 4.44× under | xfail (Item 2 lit_applied; parent obs not affected) |
| fostamatinib | 4.78× under | 4.51× under | 4.50× under | xfail (Item 3 lit_applied; extraction rate-limited) |
| tebipenem_pivoxil | 8.63× under | 9.02× under | 9.05× under | xfail (Items 4+6 ceiling) |

All 4 prodrugs remain xfail post-v3. Per spec §3.3 mechanistic-A doctrine, gate-fail with mechanistic-A-compliant values is acceptable outcome; v4 candidates require new mechanistic terms beyond data refresh (extra-hepatic esterase distribution, BH4 first-pass depletion model, in vitro CES2/tebipenem kinetics, etc.).

## SBI-prodrug intersection re-check

Re-confirmed at v3 implementation (Task 1 Step 3): intersection ∅ (none of {sepiapterin, remdesivir, fostamatinib, tebipenem_pivoxil} in `data/sbi/method_routing.json`). SBI staleness warning not required.

## 107-holdout invariance

Per §6.2 enzyme-leak audit (`tests/regression/test_prodrug_v3_enzyme_leak_audit.py`): 107/107 holdout drugs are byte-identical Cmax pre-v3 vs post-v3 (verified 2026-05-01). DRUG_SPECIFIC_CHANGES = {remdesivir, fostamatinib} but neither is in the 107-holdout set, so v3 has zero cross-drug effect. Headline AAFE bit-identical to v2 baseline (Meta 2.702, Engine 3.572, ML 3.057).
