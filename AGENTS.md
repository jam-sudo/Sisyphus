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

# Clinical / downstream layers (wrap the engine; never modify it):
regimen   ←  engine, graph, sbi         (multi-dose solver, TDM dispatch SBI/IS/IBIS/EnKF, dose adjust)
sbi       ←  engine                     (amortized neural posterior + physiology generator)
mipd      ←  engine, graph, regimen, sbi (engine-as-prior posterior PK, covariate individualization, dose recommendation)
ddi.py / pkpd.py                        (DDI enzyme adjustment; PK/PD effect compartment)
```

**predict does NOT import engine. engine does NOT import predict.** No cross-layer imports outside `pipeline/`. The clinical layers (`regimen`, `sbi`, `mipd`, `ddi`, `pkpd`) wrap the engine without modifying it (extensibility proof: each added 0 lines to `engine/`).

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

⚠ **Read `docs/claude/dead-ends.md` first.** It catalogs 40 enumerated failed attempts (DE-01..DE-40: post-hoc meta-learners, CLint R² gains, ADME replacements, foundation models, docking, UDE, E2E Neural PK, UGT IVIVE, etc.). The accuracy ceiling is a combined CLint target-noise floor + pipeline error cancellation. New track proposals must first pass an error-decorrelation gate — see `docs/claude/diagnosis.md` §4.

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

- **Plan-then-execute** for non-trivial work. Use the `superpowers:brainstorming` skill to produce a spec at `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` (specs are public audit-trail and committed), then `superpowers:writing-plans` to produce an implementation plan, then execute with `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
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
| `docs/claude/cherry_picking_process_v1.md` | Pre-registration, holdout discipline, decorrelation gate |
| `docs/claude/cherry_picking_audit_2026-04-22.md` | Quantitative cherry-picking risk audit (4.65/10) |

## Update order after an experiment

1. **Metrics in `README.md`** — only after `scripts/run_engine_benchmark.py` re-run and `4track_holdout_predictions.json` regenerated.
2. **Append entry to `docs/claude/experiment-log.md`** — at top, with date, commit, numeric outcome.
3. **If the experiment failed**, also add to `docs/claude/dead-ends.md` with the next `DE-NN` id.
4. **If the experiment reshapes ceiling analysis**, update `docs/claude/diagnosis.md` directly.

## Artifact gates — do not introduce silent fallbacks

**Lesson from the 2026-05-09 audit cycle.** Two gitignored artifacts silently augmented prediction accuracy for ~4 weeks, anchoring a headline AAFE (2.679) that no public clone could reproduce. The current headline 2.784 is the *honest* public-clone deterministic value (pinned test `test_cached_holdout_aafe_is_2p784`; was 2.698 pre-FLUX-1, regenerated on the canonical CI stack via `.github/workflows/flux1-regen.yml`), and CI is anchored to that state. FLUX-1 (PR #65, 2026-06-04) moved it 2.698 → 2.784 — a correctness-first *regression* (a real flow-limitation bug fixed; the wrong formula had been load-bearing as calibration).

When adding any new data file or model artifact that `predict()` (or any downstream code) loads conditionally via `Path.exists()`:

- **Default to mandatory**: if the artifact materially shifts predictions, commit it. If it cannot be committed (size, license), make the loader **fail loudly** at import or first use rather than silently fall back.
- **If a conditional fallback is genuinely warranted** (e.g., DrugBank with academic license), the loader must `logger.warning(...)` once at activation, and the README `§Validation` must document which headline value reflects which artifact set.
- **Test fixtures that pin numerical values to a specific artifact state** must `pytest.skipif` on the absence of that artifact, with an actionable reason message (see `tests/regression/test_prodrug_v3_enzyme_leak_audit.py` for the pattern).
- **Cross-environment numerical drift** between local dev and CI on the SAME committed inputs is typically ~0.1–3% (BLAS/CPU-SIMD build differences). Pin tests at 5–7% rel-tolerance for cross-env determinism; below that risks flake, above that misses real architectural-leak signal.

Known artifact gates that flip headline numbers if added/removed:

| Artifact | Path | Status | Effect on Meta AAFE |
|---|---|---|---|
| DrugBank CSVs | `data/drugbank/` | gitignored (academic license) | -2.7% AAFE when present |
| logP residual model | `models/adme/logp_correction.json` | gitignored (locally trained) | contributes to the same -2.7% |

If you find yourself benefiting from one of these locally, do not let your local quality leak into the published headline; the headline must match what a fresh clone sees.

## Branch protection

`main` is protected (`required_status_checks: [test]`, `strict: true`, no force-push, no deletion). All changes land via PR with passing CI. `gh pr merge --auto` will queue but only land after the test job passes — do not rely on auto-merge as a tactic for landing failing changes.
