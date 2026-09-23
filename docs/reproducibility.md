# Reproducibility Contract

Official metrics are generated with the `public` resource profile in the locked
release container. The default profile ignores local licensed DrugBank exports
and gitignored residual-correction models. A licensed research run must declare
`SISYPHUS_PROFILE=licensed_research` and cannot produce the public headline.

Installed code locates the versioned model/data bundle through `SISYPHUS_ROOT`.
Runtime results carry the active profile and SHA256 values for every primary Cmax
artifact. The API exposes the same information and a versioned `/model-info`
contract; the web client rejects an incompatible API major/minor version.

Use `scripts/prediction_trace.py` to compare environments stage by stage:

```bash
SISYPHUS_PROFILE=public SISYPHUS_ROOT=/path/to/bundle \
  python scripts/prediction_trace.py --smiles '...' --dose 100 --out trace.json
```

The trace records descriptors, predicted ADME, each Cmax track, effective weights,
engine endpoints, mass balance, time-grid hash, curve hash, environment versions,
and artifact hashes. Compare the first diverging stage rather than treating an
aggregate AAFE change as a model change.

Release acceptance targets are median per-drug Cmax drift below 1%, maximum drift
below 5%, and aggregate AAFE drift below 0.02 across the canonical cross-platform
smoke panel. Until those gates are met, only the release-container numbers are
official; native-platform runs are diagnostic.
