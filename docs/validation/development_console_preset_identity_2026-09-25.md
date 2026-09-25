# Web console preset identity audit — 2026-09-25

This audits generated **development/demo predictions**, not independent Cmax
accuracy. The eight preset SMILES were compared with their displayed formulas
using RDKit and with the current `clinical_pk.json` parent structures by full
InChIKey. Two presets required correction:

| Preset | Before | Source-supported correction | Predicted Meta Cmax, before → after |
|---|---|---|---|
| Midazolam, 5 mg | Preset SMILES encoded `C14H9FN2O` (`LWGLAEMOQGDJJW` connectivity), despite a `C18H13ClFN3` label. This was a different scaffold. | Use the [PubChem midazolam](https://pubchem.ncbi.nlm.nih.gov/compound/4192) parent (`DDLIGBOFAVUZHB-UHFFFAOYSA-N`), matching the repository's midazolam reference and the [DailyMed formula](https://dailymed.nlm.nih.gov/dailymed/downloadpdffile.cfm?setId=8f453b91-fcc5-4469-a4c7-9636baac96c8). | 0.0395892 → 0.0253661 mg/L |
| Morphine, 10 mg | Connectivity was morphine, but its stereochemistry was omitted. | Use [PubChem morphine](https://pubchem.ncbi.nlm.nih.gov/compound/5288826) (`BQJCRHHNABKAKU-KBQPJGBKSA-N`), matching the source-adjudicated reference. | 0.0185854 → 0.0231585 mg/L |

The incorrect midazolam SMILES also appeared in three README CLI examples and
three tests that called it a CYP3A4 non-UGT substrate. Those inputs now use
actual midazolam. The README's truncated atorvastatin simulation SMILES was
also replaced with its source-verified parent structure.

The console presets and the deployed `app/data` copy were regenerated from
`scripts/gen_console_data.py`. The deployed benchmark copy had also been stale
(Meta AAFE 2.815842 versus the current 2.830019 development cache); it now
matches `web/public/data`. A regression check compares every preset's full
InChIKey, formula, and molecular weight with its displayed identity and
requires the deployed and source JSON copies to be byte-identical. These
changes correct the identity of the displayed inputs; they do not establish
clinical accuracy or change the N=73 development benchmark.
