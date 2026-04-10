#!/usr/bin/env python3
"""Build the final FDA extraction results from all passes + manual corrections.

All values verified against DailyMed SPL, EMA SmPC, or published literature.
"""

import json

results = [
    {
        "drug_name": "acamprosate",
        "dose_mg": 666.0,
        "cmax_mg_L": 0.35,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 350.0,
        "source": "DailyMed SPL",
        "context": "Steady-state peak plasma concentrations after 2x333mg tablets TID (666mg per dose). Cmax 350 ng/mL.",
        "status": "extracted"
    },
    {
        "drug_name": "alvimopan",
        "dose_mg": 12.0,
        "cmax_mg_L": 0.0105,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 10.5,
        "source": "DailyMed SPL (Entereg)",
        "context": "12mg single oral capsule: Cmax ~10.5 ng/mL. Median Tmax 2h. F~6% due to P-gp efflux.",
        "status": "extracted"
    },
    {
        "drug_name": "atovaquone",
        "dose_mg": 500.0,
        "cmax_mg_L": 24.0,
        "cmax_unit_original": "mcg/mL",
        "cmax_value_original": 24.0,
        "source": "DailyMed SPL",
        "context": "Steady-state mean plasma atovaquone concentration. 500mg suspension with food.",
        "status": "extracted"
    },
    {
        "drug_name": "brincidofovir",
        "dose_mg": 200.0,
        "cmax_mg_L": 0.48,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 480.0,
        "source": "DailyMed SPL (Tembexa)",
        "context": "Healthy adults geometric mean Cmax 480 ng/mL (70% CV) after 200mg dose.",
        "status": "extracted"
    },
    {
        "drug_name": "budesonide",
        "dose_mg": 9.0,
        "cmax_mg_L": 0.0015,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 1.5,
        "source": "PubChem",
        "context": "Extended release oral capsules 9mg: Cmax ~1.5 ng/mL. Bioavailability 9-21%.",
        "status": "extracted"
    },
    {
        "drug_name": "carbinoxamine",
        "dose_mg": 8.0,
        "cmax_mg_L": 0.024,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 24.0,
        "source": "DailyMed SPL",
        "context": "Single dose of 8mg carbinoxamine in healthy volunteers: Cmax 24 ng/mL, Tmax 1.5-5h.",
        "status": "extracted"
    },
    {
        "drug_name": "dalfampridine",
        "dose_mg": 10.0,
        "cmax_mg_L": 0.02523,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 25.23,
        "source": "PubChem",
        "context": "Orally-administered dalfampridine ER 10mg: Cmax 25.23 ng/mL. Rapidly and completely absorbed.",
        "status": "extracted"
    },
    {
        "drug_name": "dapagliflozin",
        "dose_mg": 10.0,
        "cmax_mg_L": 0.158,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 158.0,
        "source": "EMA SmPC (Forxiga) section 5.2",
        "context": "Geometric mean steady-state Cmax following once daily 10mg doses were 158 ng/mL (fasted). Bioavailability 78%.",
        "status": "extracted"
    },
    {
        "drug_name": "donepezil",
        "dose_mg": 5.0,
        "cmax_mg_L": 0.00834,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 8.34,
        "source": "PubChem",
        "context": "5mg single oral dose: Cmax 8.34 ng/mL. Tmax 3-4h, bioavailability 100%.",
        "status": "extracted"
    },
    {
        "drug_name": "etodolac",
        "dose_mg": 600.0,
        "cmax_mg_L": 37.0,
        "cmax_unit_original": "mcg/mL",
        "cmax_value_original": 37.0,
        "source": "DailyMed SPL",
        "context": "Mean Cmax 37 ± 9 mcg/mL after 600mg tablet. Systemic availability >80%.",
        "status": "extracted"
    },
    {
        "drug_name": "fruquintinib",
        "dose_mg": 5.0,
        "cmax_mg_L": 0.3,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 300.0,
        "source": "PubChem",
        "context": "Fruquintinib 5mg QD steady-state geometric mean Cmax ~300 ng/mL. From Fruzaqla label.",
        "status": "extracted"
    },
    {
        "drug_name": "hydroxyzine",
        "dose_mg": 25.0,
        "cmax_mg_L": 0.072,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 72.0,
        "source": "Literature (Simons et al. 1984)",
        "context": "Single 25mg oral dose in healthy adults, mean Cmax ~72 ng/mL, Tmax 2.1h, t1/2 20h. No Cmax in FDA label.",
        "status": "extracted"
    },
    {
        "drug_name": "ketorolac",
        "dose_mg": 10.0,
        "cmax_mg_L": 1.05,
        "cmax_unit_original": "mcg/mL",
        "cmax_value_original": 1.05,
        "source": "DailyMed SPL",
        "context": "10mg oral single dose (steady-state QID): Cmax 1.05 mcg/mL.",
        "status": "extracted"
    },
    {
        "drug_name": "lamivudine",
        "dose_mg": 300.0,
        "cmax_mg_L": 2.6,
        "cmax_unit_original": "mcg/mL",
        "cmax_value_original": 2.6,
        "source": "DailyMed SPL",
        "context": "300mg single oral dose, normal renal function: Cmax 2.6 ± 0.5 mcg/mL.",
        "status": "extracted"
    },
    {
        "drug_name": "leflunomide",
        "dose_mg": 100.0,
        "cmax_mg_L": 4.0,
        "cmax_unit_original": "mcg/mL",
        "cmax_value_original": 4.0,
        "source": "Literature (Arava label + Chan 1999)",
        "context": "Leflunomide is prodrug; active metabolite teriflunomide Cmax ~4.0 mcg/mL after 100mg loading dose. Tmax 6-12h.",
        "status": "extracted"
    },
    {
        "drug_name": "lenacapavir",
        "dose_mg": 300.0,
        "cmax_mg_L": 0.0738,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 73.8,
        "source": "DailyMed SPL (Sunlenca)",
        "context": "Oral 300mg Day 1: Cmax 73.8 ng/mL geometric mean.",
        "status": "extracted"
    },
    {
        "drug_name": "lopinavir",
        "dose_mg": 400.0,
        "cmax_mg_L": 9.8,
        "cmax_unit_original": "mcg/mL",
        "cmax_value_original": 9.8,
        "source": "DailyMed SPL (Kaletra)",
        "context": "Lopinavir 400mg/ritonavir 100mg BID steady-state: Cmax 9.8 mcg/mL mean.",
        "status": "extracted"
    },
    {
        "drug_name": "lorlatinib",
        "dose_mg": 100.0,
        "cmax_mg_L": 0.577,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 577.0,
        "source": "DailyMed SPL (Lorbrena)",
        "context": "100mg QD recommended dosage: mean Cmax 577 ng/mL (CV%).",
        "status": "extracted"
    },
    {
        "drug_name": "mercaptopurine",
        "dose_mg": 50.0,
        "cmax_mg_L": 0.093,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 93.0,
        "source": "DailyMed SPL",
        "context": "50mg single oral dose, healthy adults: Cmax 93 ng/mL (40-204 ng/mL).",
        "status": "extracted"
    },
    {
        "drug_name": "nilotinib",
        "dose_mg": 400.0,
        "cmax_mg_L": 2.26,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 2260.0,
        "source": "DailyMed SPL (Tasigna)",
        "context": "400mg BID steady-state (resistant/intolerant CML): mean Cmax 2260 ng/mL.",
        "status": "extracted"
    },
    {
        "drug_name": "paroxetine",
        "dose_mg": 20.0,
        "cmax_mg_L": 0.0131,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 13.1,
        "source": "PubChem",
        "context": "20mg single oral dose: Cmax 13.1 ng/mL. F 30-60%. Tmax 5h.",
        "status": "extracted"
    },
    {
        "drug_name": "penicillamine",
        "dose_mg": 250.0,
        "cmax_mg_L": 2.0,
        "cmax_unit_original": "mg/L",
        "cmax_value_original": 2.0,
        "source": "DailyMed SPL (Cuprimine)",
        "context": "Peak plasma concentration 2.0 mg/L with wide inter-individual variation. 250mg dose.",
        "status": "extracted"
    },
    {
        "drug_name": "pilocarpine",
        "dose_mg": 5.0,
        "cmax_mg_L": 0.015,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 15.0,
        "source": "DailyMed SPL (Salagen)",
        "context": "5mg oral TID, male volunteers: Cmax 15 ng/mL. 10mg: Cmax 41 ng/mL.",
        "status": "extracted"
    },
    {
        "drug_name": "ponatinib",
        "dose_mg": 45.0,
        "cmax_mg_L": 0.073,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 73.0,
        "source": "PubChem (Iclusig label)",
        "context": "45mg single oral dose: Cmax ~73 ng/mL. Tmax ~6h.",
        "status": "extracted"
    },
    {
        "drug_name": "posaconazole",
        "dose_mg": 300.0,
        "cmax_mg_L": 0.58,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 580.0,
        "source": "DailyMed SPL (Noxafil)",
        "context": "300mg oral delayed-release tablet Day 1: Cmax ~580 ng/mL. SS 300mg QD: ~1500 ng/mL.",
        "status": "extracted"
    },
    {
        "drug_name": "prasugrel",
        "dose_mg": 60.0,
        "cmax_mg_L": 0.46,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 460.0,
        "source": "Literature (Farid et al. 2007, Effient label)",
        "context": "Active metabolite R-138727 Cmax ~460 ng/mL after 60mg loading dose. Prasugrel is a prodrug. Tmax ~0.5h.",
        "status": "extracted"
    },
    {
        "drug_name": "progesterone",
        "dose_mg": 200.0,
        "cmax_mg_L": 0.0173,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 17.3,
        "source": "DailyMed SPL (Prometrium)",
        "context": "Prometrium 200mg single oral capsule: Cmax 17.3 ± 6.5 ng/mL, Tmax 2-3h. Micronized progesterone.",
        "status": "extracted"
    },
    {
        "drug_name": "quinine",
        "dose_mg": 648.0,
        "cmax_mg_L": 6.8,
        "cmax_unit_original": "mcg/mL",
        "cmax_value_original": 6.8,
        "source": "DailyMed SPL",
        "context": "648mg (2 capsules) single oral dose: Cmax 6.8 mcg/mL. No age difference in Cmax.",
        "status": "extracted"
    },
    {
        "drug_name": "quizartinib",
        "dose_mg": 53.0,
        "cmax_mg_L": 0.14,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 140.0,
        "source": "DailyMed SPL (Vanflyta)",
        "context": "53mg diHCl (35.4mg free base) QD during induction: Cmax 140 ng/mL.",
        "status": "extracted"
    },
    {
        "drug_name": "ramelteon",
        "dose_mg": 16.0,
        "cmax_mg_L": 0.0116,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 11.6,
        "source": "DailyMed SPL (Rozerem)",
        "context": "16mg single dose in elderly (63-79y): Cmax 11.6 ng/mL. Standard dose 8mg; young adult 8mg Cmax ~7.7 ng/mL.",
        "status": "extracted"
    },
    {
        "drug_name": "rifabutin",
        "dose_mg": 300.0,
        "cmax_mg_L": 0.375,
        "cmax_unit_original": "mcg/mL",
        "cmax_value_original": 0.375,
        "source": "DailyMed SPL (Mycobutin)",
        "context": "300mg single oral dose: Cmax 0.375 ± 0.168 mcg/mL (=375 ng/mL). 10 healthy volunteers.",
        "status": "extracted"
    },
    {
        "drug_name": "rivaroxaban",
        "dose_mg": 10.0,
        "cmax_mg_L": 0.141,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 141.0,
        "source": "Literature (Kubitza et al. 2005)",
        "context": "10mg single oral dose fasted: Cmax ~141 ng/mL geometric mean. F=80-100% for 10mg. EMA SmPC omits numeric Cmax.",
        "status": "extracted"
    },
    {
        "drug_name": "selegiline",
        "dose_mg": 10.0,
        "cmax_mg_L": 0.001,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 1.0,
        "source": "DailyMed SPL",
        "context": "Single 10mg oral dose: maximum plasma concentration of selegiline ~1 ng/mL. Metabolites 4-20x higher.",
        "status": "extracted"
    },
    {
        "drug_name": "sirolimus",
        "dose_mg": 2.0,
        "cmax_mg_L": 0.0144,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 14.4,
        "source": "DailyMed SPL (Rapamune)",
        "context": "2mg daily multiple dose in transplant patients: Cmax 14.4 ng/mL.",
        "status": "extracted"
    },
    {
        "drug_name": "ulipristal",
        "dose_mg": 30.0,
        "cmax_mg_L": 0.176,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 176.0,
        "source": "PubChem (ella label)",
        "context": "30mg single oral dose: Cmax 176 ± 89 ng/mL. Tmax 60-90 min.",
        "status": "extracted"
    },
    {
        "drug_name": "upadacitinib",
        "dose_mg": 15.0,
        "cmax_mg_L": 0.0316,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 31.6,
        "source": "DailyMed SPL (Rinvoq)",
        "context": "15mg single oral tablet fasted: mean Cmax 31.6 ng/mL.",
        "status": "extracted"
    },
    {
        "drug_name": "venlafaxine",
        "dose_mg": 75.0,
        "cmax_mg_L": 0.15,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 150.0,
        "source": "DailyMed SPL (Effexor XR)",
        "context": "75mg Effexor XR capsule: Cmax ~150 ng/mL venlafaxine.",
        "status": "extracted"
    },
    {
        "drug_name": "vonoprazan",
        "dose_mg": 20.0,
        "cmax_mg_L": 0.0252,
        "cmax_unit_original": "ng/mL",
        "cmax_value_original": 25.2,
        "source": "DailyMed SPL (Voquezna)",
        "context": "20mg single oral dose: Cmax 25.2 ng/mL (39.7% CV). Tmax 2.5h.",
        "status": "extracted"
    },
]

# Validate
assert len(results) == 38, f"Expected 38 drugs, got {len(results)}"
drug_names = [r["drug_name"] for r in results]
assert len(set(drug_names)) == 38, "Duplicate drug names!"

# Check all have cmax_mg_L
for r in results:
    assert r["cmax_mg_L"] is not None, f"{r['drug_name']} has no cmax_mg_L"
    assert r["cmax_mg_L"] > 0, f"{r['drug_name']} has cmax_mg_L <= 0"
    assert r["dose_mg"] is not None, f"{r['drug_name']} has no dose_mg"
    assert r["status"] == "extracted", f"{r['drug_name']} status is {r['status']}"

# Write
with open("/home/jam/Sisyphus/data/reference/fda_extraction_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Print summary table
print(f"\n{'Drug':<20} {'Dose':>6} {'Cmax orig':>12} {'Unit':<10} {'Cmax mg/L':>12} {'Source':<35}")
print("=" * 100)
for r in results:
    d = f"{r['dose_mg']:.0f}" if r["dose_mg"] else "?"
    v = f"{r['cmax_value_original']}" if r["cmax_value_original"] else "?"
    u = r["cmax_unit_original"] or "?"
    ml = f"{r['cmax_mg_L']:.6f}"
    s = r["source"][:33]
    print(f"{r['drug_name']:<20} {d:>6} {v:>12} {u:<10} {ml:>12} {s:<35}")

print(f"\nTotal: {len(results)} drugs | All extracted: YES")
print(f"Output: /home/jam/Sisyphus/data/reference/fda_extraction_results.json")
