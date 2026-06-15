# PGx genotype-fold validation — 2026-06-14

- Primary (PM fm-agreement): frac_within_0.15 = 1.00, slope = 0.74, MAD = 0.050  -> **PASS**
- Engine regression (<2% vs analytical): **PASS**

| drug | gene | fm_invitro | obs_fold | fm_invivo | engine vs analytical |
|---|---|---|---|---|---|
| atomoxetine | CYP2D6 | 0.90 | 8.1 | 0.877 | 0.31% |
| nortriptyline | CYP2D6 | 0.78 | 4.0 | 0.750 | 0.27% |
| desipramine | CYP2D6 | 0.85 | 7.0 | 0.857 | 0.29% |
| metoprolol | CYP2D6 | 0.80 | 4.9 | 0.796 | 0.27% |
| dextromethorphan | CYP2D6 | 0.90 | 150.0 | 0.993 | 0.31% |
| omeprazole | CYP2C19 | 0.75 | 7.5 | 0.867 | 0.26% |
| lansoprazole | CYP2C19 | 0.68 | 4.0 | 0.750 | 0.23% |
| celecoxib | CYP2C9 | 0.78 | 3.5 | 0.714 | 0.27% |
| flurbiprofen | CYP2C9 | 0.75 | 2.8 | 0.643 | 0.26% |
| tolbutamide | CYP2C9 | 0.82 | 6.5 | 0.846 | 0.28% |
