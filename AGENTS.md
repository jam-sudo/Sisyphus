# AGENTS.md — Sisyphus

This file orients agentic coders (Claude Code, Codex, Aider, etc.) to the Sisyphus codebase. Read this before making changes. For implementation details and current numerical performance, see `README.md`.

> **Note**: This repository's local `CLAUDE.md` is gitignored and may contain session-specific state. The shared, public-facing agent guidance lives here in `AGENTS.md`. Project-specific Claude Code skills are loaded from `.claude/`.

## Project

**Sisyphus** — SMILES + dose → Cmax PBPK platform. The body is modeled as a typed directed multi-graph (organs as nodes; vessels and metabolism as typed edges); the ODE system is auto-derived from graph topology; all parameters carry uncertainty as `Distribution`. Predecessor: [Omega PBPK](https://github.com/jam-sudo/Omega) (data inherited; architecture rebuilt).

## Architecture (one-liner per layer)

```
pipeline  ←  predict, engine, ml, pk    (orchestrator; SMILES → PredictionResult)
engine    ←  graph                      (ODE compile/solve, MC propagation)
predict   ←  (external libs only)       (SMILES → DrugOnGraph)
ml        ←  (external libs only)       (direct PK predictors, meta-learner)
pk        ←  (nothing)                  (SimResult → PKEndpoints)
graph     ←  (nothing)                  (BodyGraph types, YAML builder)
```

**predict does NOT import engine. engine does NOT import predict.** No cross-layer imports outside `pipeline/`.

## Invariants (load-bearing — do not violate)

1. **Engine is identity-blind.** No string matching on node/enzyme/drug names in `src/sisyphus/engine/`. The "rename test" (replace every YAML organ name with random strings) must produce identical numerical results. Surrogate-related code that depends on hepatic-only assumptions lives in `ml/`, not `engine/`.
2. **All physiological/drug parameters are `Distribution`.** No bare floats. `Distribution(mean=x, cv=0)` for deterministic values.
3. **Compile once, parameterize many.** Graph topology is compiled into an ODE skeleton once. MC iterations change parameters, not structure.
4. **Flow conservation is validated at YAML build time.** Invalid topology never reaches the engine.
5. **Holdout is inviolable.** Drugs in `data/reference/holdout.json` never appear in training, tuning, anchoring, or optimization.
6. **No drug-specific branches.** The answer to "drug X gives wrong results" is never `if drug == X`. It's a better pKa model, Kp method, or reference value.
7. **20 files per directory.** Hard ceiling.
8. **Hard no-touch.** Do not modify `engine/compiler.py`, `engine/solver.py`, `DrugOnGraph` existing fields, the holdout drug list, or fudge any parameter to Cmax loss.

## Before proposing accuracy improvements

⚠ **Read `docs/claude/dead-ends.md` first.** It catalogs 32+ enumerated failed attempts (post-hoc meta-learners, CLint R² gains, ADME replacements, foundation models, docking, UDE, E2E Neural PK, etc.). The accuracy ceiling is a combined CLint target-noise floor + pipeline error cancellation. New track proposals must first pass an error-decorrelation gate — see `docs/claude/diagnosis.md` §4.

## Code style

- Python 3.10+, type hints on public signatures.
- `ruff` (line length 100).
- Frozen dataclasses for contracts (`DrugOnGraph`, `SimResult`, `PredictionResult`, etc.).
- `logging`, never `print()`.
- Constants: `UPPER_SNAKE` with unit suffix (`_L_PER_H`, `_PMOL_PER_MG`). Always cite source in comment.
- One logical change per commit: `type(scope): description` — e.g. `feat(engine): implement ClearanceFluxSpec`.
- Unit test for every public function. Write the test first when possible.

## Error handling

- **Invalid SMILES** → `ValueError`. Only hard exception.
- **Graph validation failure** → `ValueError`. YAML authoring error.
- **Everything else** → structured result. `solver_success=False`, `confidence="low"`, `ad_flags=["prodrug"]`, `warnings=[...]`. Never silently drop errors.

## Process discipline

- **Plan-then-execute** for non-trivial work. Use the `superpowers:brainstorming` skill to produce a spec at `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`, then `superpowers:writing-plans` to produce a plan at `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`, then execute with `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
- **Headline AAFE protection.** Production-promoted numbers (Meta AAFE on the 107-holdout) are protected. Never edit the metrics block in `README.md` or local `CLAUDE.md` from session context alone — always reconcile against `data/training/4track_holdout_predictions.json` after re-running `scripts/run_engine_benchmark.py`.
- **Routing changes** reconcile against `data/sbi/method_routing.json`.

## Key references

| Document | Authoritative for |
|---|---|
| `README.md` | Current implementation: ODE/ECM/IVIVE forms, prediction pipeline, applicability domain, validation status |
| `DESIGN.md` | Architectural rationale (graph-as-body, distribution-native, identity-blind engine). **Deprecated for current impl details — see README**. |
| `docs/claude/dead-ends.md` | Failed-experiment catalog. **MUST READ before any accuracy improvement proposal.** |
| `docs/claude/experiment-log.md` | Chronological experiment history |
| `docs/claude/diagnosis.md` | Accuracy ceiling analysis (target-noise floor + error cancellation) |
| `docs/claude/landmarks.md` | File / model / data / script inventory |
| `docs/claude/phase-completion.md` | Shipped phases and tracks |

## Update order after an experiment

1. **Metrics in `README.md`** — only after `scripts/run_engine_benchmark.py` re-run and `4track_holdout_predictions.json` regenerated.
2. **Append entry to `docs/claude/experiment-log.md`** — at top, with date, commit, numeric outcome.
3. **If the experiment failed**, also add to `docs/claude/dead-ends.md` with the next `DE-NN` id.
4. **If the experiment reshapes ceiling analysis**, update `docs/claude/diagnosis.md` directly.
5. **If a new file/model/script shipped**, add it to `docs/claude/landmarks.md`.
