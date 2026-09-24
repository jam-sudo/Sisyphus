# Reference concentration–time curve audit — 2026-09-24

The `clinical_pk.json` reference file contained 166 `ct_curve` arrays. These
arrays were inspected separately from observed Cmax, dose, and other PK
parameters. Only the following curve was retained: unscored simvastatin from
PK-DB/Harvey 2018, whose arm mapping still needs a separate source review.

| Finding | Count | Action |
| --- | ---: | --- |
| Fifty-point curves beginning at exactly 1 mg/L, generally followed by the same normalized exponential decline irrespective of the drug's observed Cmax | 159 | Remove: these are synthetic templates, not observed oral plasma profiles. Seven of the 159 belonged to currently scored development drugs (amantadine, apixaban, famotidine, fluvoxamine, pravastatin, ranitidine, sildenafil). |
| Other fifty-point smooth analytical curves assembled from PK summary parameters, including a previously rejected sodium-oxybate arm | 5 | Remove: the sources do not report those arrays as measurements. |
| Codeine 26-point array with two concentrations at every time point | 1 | Remove: [Band et al. 1994](https://pubmed.ncbi.nlm.nih.gov/7983238/) compared immediate-release and sustained-release codeine; the array has no treatment labels and cannot represent a single arm. Retain the paper's explicitly reported first-dose immediate-release Cmax of 138.8 ng/mL as 0.139 mg/L, but downgrade the reference from platinum to silver. The abstract does not establish fasting state or the exact base-equivalent normalization of its stated 60 mg dose. |

This removes **165** misleading arrays. The one remaining simvastatin array
has a different, non-template shape and is not scored; retaining it does not
certify its arm-level provenance. A regression assertion prevents the removed
curve fields from silently returning.

`load_reference()` uses dose, parent SMILES, Cmax, and optional AUC, but does
not read `ct_curve`. All other reference values were left unchanged, so the
N=79 development predictions and Meta AAFE **2.9492** are unchanged. The
development-residual artifact was regenerated solely to update its reference
file hash. No independent validation evidence was added.
