# External Holdout V1 Linux container smoke (2026-09-25)

At commit `60a07571f3b58ed0693169bbf1ad8e05466cd814`, the holdout Dockerfile built with `docker-buildx build --platform linux/amd64 --load -f scripts/Dockerfile.holdout -t sisyphus-holdout:smoke .` under Colima. The loaded local image was `linux/amd64`, ID `sha256:8fcfb31fd9d62fb149454ccdc1c572c2147f2ac7c71fccab4ee338da341f161b`. An amd64 container reported `x86_64` and imported RDKit, SciPy, XGBoost, and jsonschema.

With the repository mounted read-only at `/repo`, `GIT_TEST_ASSUME_DIFFERENT_OWNER=1 git rev-parse HEAD` succeeded. Disabling system Git config with `GIT_CONFIG_NOSYSTEM=1` made the same simulated owner-mismatch command fail, confirming the Dockerfile's `/repo` safe-directory setting is operative. A self-contained clean temporary clone mounted read-only also passed `verify_frozen_checkout` against its actual Git SHA, tracked-file tree hash, dependency-lock hash, and container ID. Temporary clones were removed, Colima was stopped, and Docker context was restored to `default`.

This was an execution smoke, not a one-time External Holdout V1 prediction or score. No eligible independently curated cohort, private labels, or frozen release manifest was available, so the external accuracy and release gates remain untested.
