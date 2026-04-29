# Prodrug Activation v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve T1 caution flags from v2 (active species CL/Vd, SPR proteomic abundance, CES2/tebipenem direct kinetics) by literature search and doctrine application; each flag closes as `literature_applied`, `interpretation_resolved`, or documented `ceiling_accepted`.

**Architecture:** Pure data-quality refresh. v2 architecture (well-stirred extraction, identity-blind multi-site discovery) unchanged. Drug-side updates touch `data/sbi/prodrug_activation_registry.json`; physiology-side updates touch `data/physiology/reference_man.yaml`. Validation reuses v2 test infrastructure with xfail-flip + new schema/leak-audit tests.

**Tech Stack:** Python 3.10+, pytest, JSON/YAML registries, WebSearch/WebFetch for literature corpus per spec §4.3.

**Spec:** `docs/superpowers/specs/2026-04-29-prodrug-activation-v3-design.md`

---

## Phase A — Setup & Preconditions

### Task 1: Preconditions verification + v3 branch creation

**Files:** none (verification + branching)

- [ ] **Step 1: Verify v2 PR #7 merged to main**

Run: `gh pr view 7 --json state,mergedAt`
Expected: `state == "MERGED"` and `mergedAt` non-null.

If not merged: STOP. Spec §8.1 forbids v3 implementation pre-v2-merge. Wait for v2 review/merge before continuing.

- [ ] **Step 2: Pull latest main**

```bash
git checkout main
git pull origin main
```

Expected: HEAD includes commits up through v2 merge (e.g., merge commit referencing `aef6f8e`).

- [ ] **Step 3: Re-confirm SBI-prodrug intersection (spec §8.2)**

```bash
jq -r '.routes | to_entries[] | select(.value=="sbi") | .key' data/sbi/method_routing.json | grep -E "^(sepiapterin|remdesivir|fostamatinib|tebipenem_pivoxil)$"
```

Expected: empty output (intersection ∅, matches spec §8.2 resolution from 2026-04-29).

If non-empty: PR body MUST include SBI staleness warning per spec §8.2 fallback. Document the affected drugs in the implementation log.

- [ ] **Step 4: Create v3 branch**

```bash
git checkout -b feat/prodrug-activation-v3
```

Expected: `On branch feat/prodrug-activation-v3` and HEAD matches main.

- [ ] **Step 5: Commit (no-op marker commit not needed; first real commit follows in Task 2)**

No commit at this step.

---

### Task 2: Pre-v3 deterministic baseline capture

**Files:**
- Create: `tests/regression/data/prodrug_v3_pre_baseline.json`
- Create: `scripts/capture_prodrug_v3_baseline.py`

**Why:** the enzyme-leak audit test (Task 12) compares post-v3 Cmax against pre-v3 byte-identical for `expected_unchanged` drugs. We must capture the pre-v3 baseline BEFORE any registry/yaml changes.

- [ ] **Step 1: Write baseline capture script**

Create `scripts/capture_prodrug_v3_baseline.py`:

```python
"""Capture deterministic point-estimate Cmax for all 107 holdout drugs.

Run BEFORE any v3 value changes. Saves baseline used by
tests/regression/test_prodrug_v3_enzyme_leak_audit.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from sisyphus.pipeline import predict
from sisyphus.validation.reference_loader import load_holdout_drugs


def main() -> None:
    drugs = load_holdout_drugs()
    baseline: dict[str, float] = {}
    for drug in drugs:
        # Deterministic point-estimate: sample with cv=0 enforced upstream.
        # Sisyphus pipeline accepts deterministic=True per existing API
        # (see sisyphus/pipeline/__init__.py).
        result = predict(
            smiles=drug.smiles,
            dose_mg=drug.dose_mg,
            route=drug.route,
            drug_name=drug.name,
            deterministic=True,
        )
        cmax = float(result.pk.cmax.mean) if result.pk and result.pk.cmax else math.nan
        baseline[drug.name] = cmax

    out = Path("tests/regression/data/prodrug_v3_pre_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2, sort_keys=True))
    print(f"Saved baseline for {len(baseline)} drugs to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run baseline capture**

```bash
python scripts/capture_prodrug_v3_baseline.py
```

Expected output:
```
Saved baseline for 107 drugs to tests/regression/data/prodrug_v3_pre_baseline.json
```

If `predict()` does not accept `deterministic=True`, inspect `src/sisyphus/pipeline/__init__.py` for the actual API for forcing point-estimate (likely a `mc_samples=1` + seed param, or a `deterministic` wrapper). Adapt the script and re-run.

- [ ] **Step 3: Verify baseline JSON validity**

```bash
python -c "import json; d=json.load(open('tests/regression/data/prodrug_v3_pre_baseline.json')); print(f'{len(d)} drugs, {sum(1 for v in d.values() if v != v)} NaN')"
```

Expected: `107 drugs, 0 NaN`

If NaN count > 0: investigate which drugs failed deterministic prediction; spec §6.4 treats non-finite predictions as hard failures, but at baseline-capture time pre-v3 we accept current state and document. Add a list of NaN drugs to the baseline file's metadata header (or separate file `tests/regression/data/prodrug_v3_pre_baseline_nan.json`).

- [ ] **Step 4: Commit baseline**

```bash
git add scripts/capture_prodrug_v3_baseline.py tests/regression/data/prodrug_v3_pre_baseline.json
git commit -m "feat(scripts): pre-v3 deterministic baseline capture for leak audit"
```

---

### Task 3: registry_schema test (TDD setup)

**Files:**
- Create: `tests/integration/test_prodrug_v3_registry_schema.py`

**Why:** schema test defines the v3 registry contract (citation, doctrine_path, disposition_state, etc.). Writing it first creates a failing test that drives the registry update tasks.

- [ ] **Step 1: Write the failing schema test**

Create `tests/integration/test_prodrug_v3_registry_schema.py`:

```python
"""Structural validation of v3 prodrug activation registry per spec §6.2.

Verifies each registry entry has required v3 fields (citation,
doctrine_path, disposition_state, etc.) and that conditional fields
(ceiling_rationale, interpretation_decision) are present per disposition.

This is structural validation only — no value comparison (avoids tautology
with implementation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REGISTRY_PATH = Path("data/sbi/prodrug_activation_registry.json")
ALLOWED_DISPOSITIONS = {"literature_applied", "interpretation_resolved", "ceiling_accepted"}
REQUIRED_FIELDS = {
    "citation",
    "doctrine_path",
    "disposition_state",
    "source_dbs_searched",
    "n_candidates_reviewed",
}


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


@pytest.fixture(scope="module")
def registry() -> dict:
    return _load_registry()


def test_registry_file_exists() -> None:
    assert REGISTRY_PATH.exists(), f"{REGISTRY_PATH} not found"


def test_each_entry_has_v3_metadata_block(registry: dict) -> None:
    """Each prodrug entry must have a `v3_metadata` dict with required fields."""
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict):
            continue  # skip non-entry top-level keys
        meta = entry.get("v3_metadata")
        assert meta is not None, f"{drug_name}: v3_metadata block missing"
        assert isinstance(meta, dict), f"{drug_name}: v3_metadata must be dict"
        missing = REQUIRED_FIELDS - set(meta.keys())
        assert not missing, f"{drug_name}: v3_metadata missing fields {missing}"


def test_disposition_state_valid(registry: dict) -> None:
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        ds = entry["v3_metadata"]["disposition_state"]
        assert ds in ALLOWED_DISPOSITIONS, f"{drug_name}: invalid disposition_state {ds!r}"


def test_citation_required_for_non_ceiling(registry: dict) -> None:
    """citation must be non-empty string when disposition is literature_applied
    or interpretation_resolved; may be null only for ceiling_accepted."""
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        meta = entry["v3_metadata"]
        ds = meta["disposition_state"]
        citation = meta.get("citation")
        if ds in ("literature_applied", "interpretation_resolved"):
            assert isinstance(citation, str) and citation.strip(), (
                f"{drug_name}: disposition {ds} requires non-empty citation"
            )
        elif ds == "ceiling_accepted":
            # citation may be null OR non-empty (e.g., V/F found but F primary not)
            if citation is not None:
                assert isinstance(citation, str), f"{drug_name}: citation must be str or null"


def test_ceiling_rationale_required(registry: dict) -> None:
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        meta = entry["v3_metadata"]
        if meta["disposition_state"] == "ceiling_accepted":
            rationale = meta.get("ceiling_rationale", "").strip()
            assert rationale, f"{drug_name}: ceiling_accepted requires non-empty ceiling_rationale"


def test_interpretation_decision_required(registry: dict) -> None:
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        meta = entry["v3_metadata"]
        if meta["disposition_state"] == "interpretation_resolved":
            decision = meta.get("interpretation_decision", "").strip()
            assert decision, (
                f"{drug_name}: interpretation_resolved requires non-empty interpretation_decision"
            )


def test_n_candidates_reviewed_int(registry: dict) -> None:
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        n = entry["v3_metadata"]["n_candidates_reviewed"]
        assert isinstance(n, int) and n >= 1, (
            f"{drug_name}: n_candidates_reviewed must be int ≥ 1, got {n!r}"
        )


def test_source_dbs_searched_is_list(registry: dict) -> None:
    allowed_dbs = {"PubMed", "GoogleScholar", "FDA", "EMA", "ChEMBL", "DrugBank", "bioRxiv"}
    for drug_name, entry in registry.items():
        if not isinstance(entry, dict) or "v3_metadata" not in entry:
            continue
        dbs = entry["v3_metadata"]["source_dbs_searched"]
        assert isinstance(dbs, list) and dbs, (
            f"{drug_name}: source_dbs_searched must be non-empty list"
        )
        unknown = set(dbs) - allowed_dbs
        assert not unknown, f"{drug_name}: unknown source dbs {unknown}; allowed = {allowed_dbs}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/integration/test_prodrug_v3_registry_schema.py -v
```

Expected: tests fail because current registry has no `v3_metadata` blocks. The first failing test should be `test_each_entry_has_v3_metadata_block`.

- [ ] **Step 3: Commit failing test (TDD red phase)**

```bash
git add tests/integration/test_prodrug_v3_registry_schema.py
git commit -m "test(prodrug-v3): registry schema validation (failing TDD)"
```

---

## Phase B — Per-Item Literature Resolution

Each Phase-B task follows the same pattern: literature search → doctrine application → registry/yaml update → literature deliverable section. Tasks 4-9 are sequential (single literature deliverable file appended) but item resolutions are independent (no dependencies between items).

### Task 4: Item 1 — BH4 active CL/Vd (sepiapterin)

**Files:**
- Create / append: `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`
- Modify: `data/sbi/prodrug_activation_registry.json` (sepiapterin entry)

**Spec ref:** §5.1 (BH4 fallback chain), §4.1 (popPK doctrine), §4.4 (documentation template).

- [ ] **Step 1: Initialize literature deliverable file (first item to write)**

Create `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`:

```markdown
# Prodrug Activation v3 — Literature Deliverable

**Date:** 2026-04-29
**Spec:** `docs/superpowers/specs/2026-04-29-prodrug-activation-v3-design.md`

This deliverable applies the §4.4 documentation template to each of the 6 T1-flagged items. Each section captures the literature search, doctrine application, and disposition decision per spec §2 acceptance gate.

---
```

Commit this initialization separately:
```bash
git add docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md
git commit -m "docs(spec): initialize v3 literature deliverable"
```

- [ ] **Step 2: Search literature for BH4 active CL/Vd (per §4.3 corpus)**

Search corpus: PubMed, Google Scholar, FDA review docs, EMA assessment, ChEMBL/DrugBank, bioRxiv.

Search terms (run in WebSearch / WebFetch / available bioRxiv MCP tool):
- `"tetrahydrobiopterin BH4 popPK pharmacokinetics human IV Vss"`
- `"sapropterin oral bioavailability F human"`
- `"sapropterin Feillet 2008"` (T1 cited reference — locate full citation + DOI + Vss)
- `"sepiapterin pharmacokinetics human"`
- `"BH4 plasma disposition compartmental"`

Stop when: predefined corpus exhausted OR 30 candidates reviewed (whichever first, per §4.3).

Document for each candidate reviewed:
- Citation (Author Year + DOI/URL if available)
- Whether eligible per §4.1 doctrine (same-entity human, IV or oral with F separable)
- Reported Vd (specify Vc, Vss, or V/F) and CL
- Reported BSV (if any)

- [ ] **Step 3: Apply BH4 fallback chain (spec §5.1)**

Decision tree:
1. **F primary citation found** → divide V/F by F to get central V; disposition = `interpretation_resolved`. interpretation_decision field must record: F value, F citation, V/F source, V (computed central) + CV.
2. **F primary not found** → strict downgrade to `ceiling_accepted` per spec §5.1. ceiling_rationale must include: (a) literature V/F value found (if any) + citation, (b) F primary not located despite §4.3 exhaustive search, (c) uncertainty bound `[150 L, V/F]`, (d) v1 Vd=150 retained as least-bad placeholder NOT endorsed value.

Do NOT use F geometric mean over reported ranges or CV inflation (spec §5.1 step 3 explicitly rejects these).

- [ ] **Step 4: Append BH4 section to literature deliverable**

Append to `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`, following §4.4 template:

```markdown
## Item 1 — BH4 active CL/Vd (sepiapterin)

- **v1/v2 state:** Vd = 150 L, CL = <existing v1/v2 value from registry>, source = `data/sbi/prodrug_activation_registry.json` v1
- **T1 flag:** literature deviation 1.5-50× (per `docs/superpowers/specs/2026-04-27-prodrug-v2-task1-literature.md`)
- **Search:**
  - terms: <list of search strings used>
  - databases: <e.g., ["PubMed", "GoogleScholar", "FDA"]>
  - N candidates reviewed: <int>
- **Selected source(s):** <full citation(s)> OR null
- **Doctrine application:**
  - Mean rule: <Vss / Vc / V÷F + F=...> (per §4.1)
  - CV rule: <BSV / inter-study GSD / class default> (per §4.2)
  - Same-entity check: <pass — sapropterin = BH4 salt form> per §4.1 Gap 1
- **Sub-decisions resolved:**
  - F_sapropterin: <found primary, F=X, citation=Y> OR <NOT found — fallback to ceiling per §5.1 step 2>
  - 2-comp Vss vs Vc: <Vss selected per §4.1>
- **Final values:** mean=<X>, cv=<Y> (or "no change — v1 retained" if ceiling)
- **Disposition:** literature_applied | interpretation_resolved | ceiling_accepted
- **IF ceiling:** <rationale per spec §5.1 fallback step 2; explicit acknowledgment v1 Vd=150 known incorrect>
```

- [ ] **Step 5: Update sepiapterin entry in registry**

Modify `data/sbi/prodrug_activation_registry.json`. Locate the sepiapterin entry. Add a `v3_metadata` block adjacent to existing fields, and update `active_metabolite.cl_per_h` / `vd_l` if disposition is `literature_applied` or `interpretation_resolved`.

Example structure (illustrative — actual values from Step 3):
```json
{
  "sepiapterin": {
    "...existing v2 fields preserved...": "...",
    "active_metabolite": {
      "name": "BH4",
      "cl_per_h": {"mean": <updated>, "cv": <updated>},
      "vd_l": {"mean": <updated>, "cv": <updated>},
      "...": "..."
    },
    "v3_metadata": {
      "citation": "<full citation>" or null,
      "doctrine_path": "§4.1 oral V/F division; §4.5 not applicable; §5.1 fallback step <1|2>",
      "disposition_state": "<literature_applied | interpretation_resolved | ceiling_accepted>",
      "source_dbs_searched": ["PubMed", "..."],
      "n_candidates_reviewed": <int>,
      "ceiling_rationale": "<required if ceiling_accepted; v1 known-wrong acknowledgment>",
      "interpretation_decision": "<required if interpretation_resolved>"
    }
  }
}
```

If disposition is `ceiling_accepted`: do NOT modify `active_metabolite.cl_per_h` or `vd_l` (retain v1 values).

- [ ] **Step 6: Run schema test for sepiapterin entry**

```bash
pytest tests/integration/test_prodrug_v3_registry_schema.py -v -k sepiapterin
```

Expected: schema tests pass for sepiapterin (other drugs may still fail since their v3_metadata is not yet added — that's OK at this stage).

If failures: fix the JSON entry per the test failure message and re-run.

- [ ] **Step 7: Commit Item 1**

```bash
git add docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md data/sbi/prodrug_activation_registry.json
git commit -m "feat(prodrug-v3): Item 1 BH4 — <disposition> per spec §5.1"
```

Replace `<disposition>` with the actual disposition state (e.g., `literature_applied`, `interpretation_resolved`, or `ceiling_accepted`).

---

### Task 5: Item 2 — GS-441524 active CL/Vd (remdesivir)

**Files:**
- Modify: `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`
- Modify: `data/sbi/prodrug_activation_registry.json` (remdesivir entry)

**Spec ref:** §5.2 (GS-441524 path), §4.1 (doctrine, same-entity strict human-only).

- [ ] **Step 1: Search literature for GS-441524 active CL/Vd**

Search terms:
- `"GS-441524 human pharmacokinetics popPK"`
- `"remdesivir GS-441524 metabolite plasma kinetics IV human"`
- `"Sukeishi 2022 GS-441524"` (verify species/route — T1 cited but doctrine eligibility unverified per §5.2)
- `"GS-441524 disposition Vss central volume"`

Document for each candidate (template per §4.4 + same-entity check):
- Species (must be human per §4.1 Gap 1; non-human = reject)
- Route (IV, oral with F separable, or unsuitable)
- Compartmental model (1-comp, 2-comp Vss, etc.)

- [ ] **Step 2: Apply doctrine per §5.2**

If Sukeishi 2022 is non-human or non-eligible route: reject. Fall back to other GS-441524 popPK sources.

If eligible source found: extract CL and Vss (or 1-comp Vd as Vss-equivalent per §4.1), apply CV doctrine §4.2.

If no eligible source found: `ceiling_accepted` with rationale documenting search attempt.

- [ ] **Step 3: Append GS-441524 section to literature deliverable**

Append `## Item 2 — GS-441524 active CL/Vd (remdesivir)` section to `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md` per §4.4 template (full template structure same as Task 4 Step 4).

- [ ] **Step 4: Update remdesivir entry in registry**

Modify `data/sbi/prodrug_activation_registry.json` remdesivir entry. Add `v3_metadata` block. If `literature_applied` or `interpretation_resolved`, update `active_metabolite.cl_per_h` and `vd_l`.

- [ ] **Step 5: Run schema test for remdesivir entry**

```bash
pytest tests/integration/test_prodrug_v3_registry_schema.py -v -k remdesivir
```

Expected: schema tests pass for remdesivir.

- [ ] **Step 6: Commit Item 2**

```bash
git add docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md data/sbi/prodrug_activation_registry.json
git commit -m "feat(prodrug-v3): Item 2 GS-441524 — <disposition> per spec §5.2"
```

---

### Task 6: Item 3 — R406 active CL/Vd (fostamatinib)

**Files:**
- Modify: `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`
- Modify: `data/sbi/prodrug_activation_registry.json` (fostamatinib entry)

**Spec ref:** §5.3 (cleanest item; IV direct).

- [ ] **Step 1: Search literature for R406 active CL/Vd**

Search terms:
- `"R406 fostamatinib metabolite IV pharmacokinetics human micro-dose"`
- `"PMC9250994 R406"` (locate the T1-cited paper for full bibliographic details)
- `"tamatinib R406 popPK human disposition"`

The T1 reference (PMC9250994) is the primary candidate. Verify it is human, IV, peer-reviewed, and reports Vd/CL with BSV.

- [ ] **Step 2: Apply doctrine per §5.3**

IV direct route → central Vd directly applicable. If 2-comp model reported, take Vss per §4.1. CV from BSV per §4.2 1st priority.

Expected disposition per spec: `literature_applied`.

- [ ] **Step 3: Append R406 section to literature deliverable**

Append `## Item 3 — R406 active CL/Vd (fostamatinib)` per §4.4 template.

- [ ] **Step 4: Update fostamatinib entry in registry**

Modify `data/sbi/prodrug_activation_registry.json` fostamatinib entry. Add `v3_metadata` block. Update `active_metabolite.cl_per_h` and `vd_l` per literature.

- [ ] **Step 5: Run schema test for fostamatinib entry**

```bash
pytest tests/integration/test_prodrug_v3_registry_schema.py -v -k fostamatinib
```

Expected: pass.

- [ ] **Step 6: Commit Item 3**

```bash
git add docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md data/sbi/prodrug_activation_registry.json
git commit -m "feat(prodrug-v3): Item 3 R406 — <disposition> per spec §5.3"
```

---

### Task 7: Item 4 — tebipenem active CL/Vd (tebipenem_pivoxil)

**Files:**
- Modify: `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`
- Modify: `data/sbi/prodrug_activation_registry.json` (tebipenem_pivoxil entry)

**Spec ref:** §5.4 (T1: "mostly OK" — but full §4.3 exhaustiveness required, NO lazy ceiling).

- [ ] **Step 1: Search literature for tebipenem active CL/Vd**

Search terms:
- `"tebipenem pivoxil pharmacokinetics popPK human"`
- `"tebipenem oral bioavailability F human"`
- `"tebipenem IV human disposition"`
- `"SPR-994 tebipenem popPK"` (Spero Therapeutics product code)

T1 noted "mostly OK" — this means small adjustment expected but does NOT skip the search. Apply full §4.3 source exhaustiveness.

- [ ] **Step 2: Apply doctrine per §5.4**

Oral popPK accepted iff F separable per §4.1. If oral V/F: divide by F. CV from BSV per §4.2.

If literature confirms current values within reasonable margin: still `literature_applied` with documentation showing convergence.

If literature unavailable: `ceiling_accepted` with rationale.

- [ ] **Step 3: Append tebipenem section to literature deliverable**

Append `## Item 4 — tebipenem active CL/Vd (tebipenem_pivoxil)` per §4.4 template.

- [ ] **Step 4: Update tebipenem_pivoxil entry in registry**

Modify `data/sbi/prodrug_activation_registry.json` tebipenem_pivoxil entry. Add `v3_metadata` block. Update `active_metabolite.cl_per_h` and `vd_l` if needed.

- [ ] **Step 5: Run schema test for tebipenem_pivoxil entry**

```bash
pytest tests/integration/test_prodrug_v3_registry_schema.py -v -k tebipenem
```

Expected: pass.

- [ ] **Step 6: Commit Item 4**

```bash
git add docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md data/sbi/prodrug_activation_registry.json
git commit -m "feat(prodrug-v3): Item 4 tebipenem CL/Vd — <disposition> per spec §5.4"
```

---

### Task 8: Item 5 — SPR primary proteomic abundance

**Files:**
- Modify: `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`
- Modify: `data/physiology/reference_man.yaml` (SPR entries: liver, gut_wall, kidney)

**Spec ref:** §5.5 (SPR proteomic), §4.5 (parallel doctrine for non-popPK), unit conversion mandatory.

- [ ] **Step 1: Search literature for human SPR primary proteomic data**

Search terms:
- `"sepiapterin reductase SPR human liver proteomic abundance MS"`
- `"SPR enzyme expression kidney gut human quantitative"`
- `"Wegler enzyme atlas SPR"` (T1 reference patterns; Wegler is one common proteomic atlas)
- `"ProteomicsDB SPR human"`
- `"BH4 biosynthesis enzyme abundance pmol/mg"`

Check existing v2 enzyme abundances in `data/physiology/reference_man.yaml` for SPR (liver=1e5, gut_wall=3e3, kidney=3e4) — these are class-estimated per T1 caution.

- [ ] **Step 2: Apply parallel doctrine per §4.5**

If primary proteomic measurement found:
- Verify units (pmol/mg microsomal vs pmol/g organ) — convert if necessary using cited conversion factor
- Verify same-entity (human SPR isoform; not species extrapolation)
- Mean: from primary source
- CV: inter-individual variability if reported, else inter-study GSD (n≥3), else class default 0.5

If not found: `ceiling_accepted` with rationale.

- [ ] **Step 3: Append SPR section to literature deliverable**

Append `## Item 5 — SPR primary proteomic abundance` per §4.4 template, with extra emphasis on unit conversion documentation.

- [ ] **Step 4: Update reference_man.yaml SPR entries (if literature_applied)**

Locate SPR entries in `data/physiology/reference_man.yaml`. Update mean and CV per literature for liver, gut_wall, kidney (each may have different proteomic measurement).

Add YAML comment above each updated SPR entry:
```yaml
SPR: {mean: <new>, cv: <new>}  # v3 literature_applied: <citation>; per docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md Item 5
```

If `ceiling_accepted`: do not modify yaml; document ceiling in literature deliverable only.

- [ ] **Step 5: Verify yaml syntax**

```bash
python -c "import yaml; yaml.safe_load(open('data/physiology/reference_man.yaml'))"
```

Expected: no exception (yaml parses).

- [ ] **Step 6: Run physiology graph build smoke test**

```bash
pytest tests/integration/ -v -k "physiology or reference_man" --no-header -x
```

Expected: existing physiology tests pass (yaml structure unchanged, only SPR values updated).

- [ ] **Step 7: Commit Item 5**

```bash
git add docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md data/physiology/reference_man.yaml
git commit -m "feat(prodrug-v3): Item 5 SPR proteomic — <disposition> per spec §5.5"
```

---

### Task 9: Item 6 — CES2/tebipenem direct Vmax/Km

**Files:**
- Modify: `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`
- Modify: `data/sbi/prodrug_activation_registry.json` (tebipenem_pivoxil entry — `enzyme_affinity_for_conversion["CES2"]`)

**Spec ref:** §5.6 (CES2/tebipenem direct), §4.5 (parallel doctrine), IVIVE chain validation.

- [ ] **Step 1: Search literature for direct CES2/tebipenem in vitro kinetics**

Search terms:
- `"tebipenem pivoxil hydrolysis CES2 in vitro Vmax Km recombinant"`
- `"tebipenem pivoxil esterase kinetics human"`
- `"carboxylesterase CES2 tebipenem CLint"`
- `"SPR-994 hydrolysis CES isoform"`
- `"tebipenem activation human liver microsomes"`

Verify isoform specificity: CES2 (NOT CES1 — see §5.6). Animal studies rejected per §4.1 Gap 1.

- [ ] **Step 2: Apply parallel doctrine per §4.5 + IVIVE chain validation**

If direct in vitro data found:
- Compute CLint = Vmax/Km
- Verify units → CLint per pmol enzyme conversion (consistent with v2 `enzyme_affinity_for_conversion` units)
- IVIVE scaling factors must be cited if used
- Mean: CLint per pmol
- CV: inter-experiment GSD if multiple, else class default 0.5

If not found: `ceiling_accepted`.

- [ ] **Step 3: Append CES2/tebipenem section to literature deliverable**

Append `## Item 6 — CES2/tebipenem direct Vmax/Km` per §4.4 template, with explicit unit-chain documentation (Vmax raw → CLint per pmol → registry units).

- [ ] **Step 4: Update tebipenem_pivoxil registry entry — CES2 affinity**

Modify `data/sbi/prodrug_activation_registry.json` tebipenem_pivoxil entry. The `enzyme_affinity_for_conversion["CES2"]` field receives the new CLint mean and CV. Update `v3_metadata` (NB: tebipenem_pivoxil now has TWO v3 changes — Task 7 D1 active CL/Vd AND Task 9 CES2 affinity. The `v3_metadata` block must reflect both via doctrine_path and citation lists).

For doctrine_path field, append a second doctrine entry:
```
"doctrine_path": "§4.1 (Item 4 D1: <path>) + §4.5 + §5.6 (Item 6: CES2 affinity, <path>)"
```

If both items literature_applied with separate citations:
```
"citation": "Item 4: <citation A>; Item 6: <citation B>"
```

If Item 6 ceiling and Item 4 literature: only Item 6 contributes to ceiling_rationale field; combine with care.

- [ ] **Step 5: Run schema test**

```bash
pytest tests/integration/test_prodrug_v3_registry_schema.py -v
```

Expected: ALL 4 prodrug entries (sepiapterin, remdesivir, fostamatinib, tebipenem_pivoxil) now pass schema validation.

- [ ] **Step 6: Commit Item 6**

```bash
git add docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md data/sbi/prodrug_activation_registry.json
git commit -m "feat(prodrug-v3): Item 6 CES2/tebipenem affinity — <disposition> per spec §5.6"
```

---

## Phase C — Test Infrastructure

### Task 10: Refactor pipeline_smoke (functional-only)

**Files:**
- Modify: `tests/integration/test_prodrug_v2_pipeline_smoke.py`

**Spec ref:** §6.1 (refactor rationale — avoid v3 tautology), §10 SC #3 (must pass post-v3).

**Why TDD:** existing smoke has hardcoded Cmax expected values that will fail post-v3 (or worse: become tautological if updated). Refactor to functional-only assertions.

- [ ] **Step 1: Read current pipeline_smoke to understand structure**

```bash
cat tests/integration/test_prodrug_v2_pipeline_smoke.py
```

Identify: which tests have hardcoded numerical expected values; which are pure functional.

- [ ] **Step 2: Replace hardcoded Cmax assertions with functional assertions**

For each test that previously asserted `assert result.pk.cmax.mean == approx(<hardcoded>, rel=...)`, replace with:

```python
# Functional-only: pipeline executes, returns valid result.
# Numerical regression handled by test_prodrug_v2_snapshot.py.
assert result is not None
assert result.pk is not None
assert result.pk.cmax is not None
assert result.pk.cmax.mean > 0
import math
assert math.isfinite(result.pk.cmax.mean)
```

Add a docstring at the top of each test:
```python
"""Functional smoke test for <drug> prodrug pipeline.

Verifies: pipeline executes without crash, returns valid PredictionResult,
Cmax is positive and finite. Numerical regression assertions live in
tests/regression/test_prodrug_v2_snapshot.py per spec §6.1.
"""
```

- [ ] **Step 3: Run refactored smoke tests**

```bash
pytest tests/integration/test_prodrug_v2_pipeline_smoke.py -v
```

Expected: all 4 prodrug smoke tests PASS (functional assertions hold for current registry state).

- [ ] **Step 4: Commit smoke refactor**

```bash
git add tests/integration/test_prodrug_v2_pipeline_smoke.py
git commit -m "refactor(test): pipeline_smoke functional-only per v3 spec §6.1"
```

---

### Task 11: enzyme_leak_audit test (TDD)

**Files:**
- Create: `tests/regression/test_prodrug_v3_enzyme_leak_audit.py`

**Spec ref:** §6.2 (logic, two-dimension separation), §10 SC #6 (invariance).

- [ ] **Step 1: Write the leak audit test**

Create `tests/regression/test_prodrug_v3_enzyme_leak_audit.py`:

```python
"""v3 enzyme-leak audit per spec §6.2.

Verifies v3 registry/yaml changes affect only intended drugs. Drugs in
expected_unchanged set must produce byte-identical deterministic Cmax
compared to pre-v3 baseline (tests/regression/data/prodrug_v3_pre_baseline.json).

Two-dimension change tracking:
- CHANGED_ENZYME_ABUNDANCES: physiology yaml abundance changes (cross-drug)
- DRUG_SPECIFIC_CHANGES: drug-side registry changes (drug-isolated)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sisyphus.pipeline import predict
from sisyphus.validation.reference_loader import load_holdout_drugs

PRE_BASELINE_PATH = Path("tests/regression/data/prodrug_v3_pre_baseline.json")
REGISTRY_PATH = Path("data/sbi/prodrug_activation_registry.json")

# v3 change dimensions — implementer fills these based on which items
# closed as literature_applied vs ceiling_accepted (Tasks 4-9 outcomes).
# These constants drive expected_changed/expected_unchanged partitioning.

# Enzymes whose physiology abundance changed in v3.
# {"SPR"} if Item 5 literature_applied; {} if Item 5 ceiling_accepted.
CHANGED_ENZYME_ABUNDANCES: frozenset[str] = frozenset()  # UPDATE per Task 8 outcome

# Drugs whose drug-side registry changed in v3 (D1 active CL/Vd OR Item 6 affinity).
# Subset of {sepiapterin, remdesivir, fostamatinib, tebipenem_pivoxil}.
DRUG_SPECIFIC_CHANGES: frozenset[str] = frozenset()  # UPDATE per Tasks 4-7, 9 outcomes


@pytest.fixture(scope="module")
def pre_baseline() -> dict[str, float]:
    return json.loads(PRE_BASELINE_PATH.read_text())


def _drug_uses_enzyme(drug_graph, enzyme_tag: str) -> bool:
    """Return True if drug_graph uses enzyme_tag in elimination or activation."""
    elim = getattr(drug_graph, "enzyme_affinity", {}) or {}
    activ = getattr(drug_graph, "enzyme_affinity_for_conversion", {}) or {}
    return enzyme_tag in elim or enzyme_tag in activ


@pytest.mark.slow
def test_enzyme_leak_audit(pre_baseline: dict[str, float]) -> None:
    """Drugs not affected by v3 changes must have byte-identical deterministic Cmax."""
    drugs = load_holdout_drugs()
    expected_unchanged: list[str] = []
    expected_changed: list[str] = []

    for drug in drugs:
        # Build deterministic graph to inspect enzyme usage
        from sisyphus.predict import build_drug_on_graph, compute_profile, predict_adme

        profile = compute_profile(drug.smiles)
        adme = predict_adme(profile)
        drug_graph = build_drug_on_graph(profile, adme, drug.dose_mg, drug.route)

        uses_changed_enzyme = any(
            _drug_uses_enzyme(drug_graph, enz) for enz in CHANGED_ENZYME_ABUNDANCES
        )
        is_drug_specific = drug.name in DRUG_SPECIFIC_CHANGES

        if uses_changed_enzyme or is_drug_specific:
            expected_changed.append(drug.name)
        else:
            expected_unchanged.append(drug.name)

    # All-ceiling sanity: if no changes, all 107 should be unchanged
    if not CHANGED_ENZYME_ABUNDANCES and not DRUG_SPECIFIC_CHANGES:
        assert len(expected_unchanged) == len(drugs), (
            f"All-ceiling scenario expects 107 unchanged, got {len(expected_unchanged)}"
        )

    # Verify byte-identical AND finite for each unchanged drug
    failures = []
    for drug in drugs:
        if drug.name not in expected_unchanged:
            continue
        baseline_cmax = pre_baseline.get(drug.name)
        if baseline_cmax is None or not math.isfinite(baseline_cmax):
            # Skip drugs that had non-finite baseline (documented at Task 2)
            continue
        result = predict(
            smiles=drug.smiles,
            dose_mg=drug.dose_mg,
            route=drug.route,
            drug_name=drug.name,
            deterministic=True,
        )
        cmax_v3 = float(result.pk.cmax.mean) if result.pk and result.pk.cmax else math.nan
        if not math.isfinite(cmax_v3):
            failures.append(f"{drug.name}: v3 produced non-finite Cmax {cmax_v3}")
            continue
        if cmax_v3 != baseline_cmax:
            failures.append(
                f"{drug.name}: v3 Cmax {cmax_v3} != pre-v3 baseline {baseline_cmax}"
            )

    assert not failures, "Leak detected (expected_unchanged drug Cmax differs):\n" + "\n".join(failures)
```

- [ ] **Step 2: Run leak audit test**

```bash
pytest tests/regression/test_prodrug_v3_enzyme_leak_audit.py -v -m slow
```

Expected outcome depends on which dimensions were populated:
- If implementer has not yet updated `CHANGED_ENZYME_ABUNDANCES` / `DRUG_SPECIFIC_CHANGES` constants (still empty), test asserts all 107 byte-identical. If items 1-4 D1 changed registry values, assertion fails (correctly catching the leak as unaccounted-for change).

The test failure here is INFORMATIVE: it forces the implementer to fill the constants based on Tasks 4-9 outcomes.

- [ ] **Step 3: Update CHANGED_ENZYME_ABUNDANCES and DRUG_SPECIFIC_CHANGES constants**

Edit lines marked `# UPDATE per Task ... outcome`. Populate based on which items literature_applied vs ceiling_accepted:

Example post-implementation values (illustrative):
```python
CHANGED_ENZYME_ABUNDANCES: frozenset[str] = frozenset({"SPR"})  # Item 5 lit_applied
DRUG_SPECIFIC_CHANGES: frozenset[str] = frozenset({
    "sepiapterin",        # Item 1 BH4 CL/Vd interpretation_resolved
    "fostamatinib",       # Item 3 R406 literature_applied
    "tebipenem_pivoxil",  # Item 4 + Item 6 (multiple changes)
    # remdesivir: Item 2 ceiling_accepted → NOT in set
})
```

- [ ] **Step 4: Re-run leak audit**

```bash
pytest tests/regression/test_prodrug_v3_enzyme_leak_audit.py -v -m slow
```

Expected: PASS (now `expected_unchanged` correctly reflects v3 changes; byte-identical assertion holds for unaffected drugs).

If still fails: investigate — there may be unintended cross-leak (e.g., a non-prodrug drug uses SPR for elimination), or the change-tracking constants are misconfigured.

- [ ] **Step 5: Commit leak audit test**

```bash
git add tests/regression/test_prodrug_v3_enzyme_leak_audit.py
git commit -m "test(prodrug-v3): enzyme-leak audit per spec §6.2"
```

---

### Task 12: ddi_smoke tolerance verification

**Files:**
- Modify: `tests/integration/test_prodrug_v2_ddi_smoke.py`

**Spec ref:** §6.1 (ddi_smoke disposition: re-execute + ±5% tolerance check, widen to ±10% with rationale if needed).

- [ ] **Step 1: Run existing ddi_smoke test**

```bash
pytest tests/integration/test_prodrug_v2_ddi_smoke.py -v
```

Capture outcome:
- PASS → no action; tolerance still holds. Skip to Step 4 (no edits).
- FAIL with ratio outside ±5% → Step 2 (investigate).

- [ ] **Step 2: Investigate non-linearity (if test failed)**

Compute the new DDI ratio (with v3 active CL/Vd values):

```bash
python -c "
from sisyphus.pipeline import predict
import json

# Load remdesivir registry entry to confirm v3 values
reg = json.load(open('data/sbi/prodrug_activation_registry.json'))
print('remdesivir v3 active CL/Vd:', reg['remdesivir']['active_metabolite'])

# Run with default CES1 abundance
res_default = predict(smiles='<remdesivir SMILES>', dose_mg=200, route='IV', drug_name='remdesivir', deterministic=True)
# Run with CES1 0.5×: requires DDI scaling — see existing ddi_smoke for exact API
"
```

The exact API for DDI scaling is in the existing `test_prodrug_v2_ddi_smoke.py`. Mirror that pattern with v3 values.

Determine new ratio R3. If R3 within `[0.45, 0.55]` (i.e., still ±5% of 0.5 nominal), tolerance holds.

If R3 outside ±5% but within ±10%: justify by saturation analysis of the well-stirred extraction (E = fup·CLint / (Q + fup·CLint) approaches 1 at high CLint, breaking linear DDI scaling).

- [ ] **Step 3: Update ddi_smoke tolerance with rationale (if needed)**

If tolerance widens to ±10%, edit `tests/integration/test_prodrug_v2_ddi_smoke.py`:

```python
# v3 update: tolerance widened from ±5% to ±10% per spec §6.1.
# Rationale: with v3 GS-441524 CL/Vd updated, well-stirred extraction
# enters higher-saturation regime where E (extraction ratio) is
# non-linear in CES1 abundance. DDI ratio shift is mechanistic, not bug.
# See docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md Item 2.
EXPECTED_RATIO_LOWER = 0.45
EXPECTED_RATIO_UPPER = 0.55
TOLERANCE = 0.10  # was 0.05 in v2
```

If ratio shifted to a new central value (e.g., 0.48 vs old 0.53), update expected ratio also and document.

If tolerance shift exceeds ±10%: investigate as bug, not tolerance issue. STOP and debug.

- [ ] **Step 4: Re-run ddi_smoke**

```bash
pytest tests/integration/test_prodrug_v2_ddi_smoke.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit ddi_smoke update**

If edits made:
```bash
git add tests/integration/test_prodrug_v2_ddi_smoke.py
git commit -m "test(prodrug-v3): ddi_smoke tolerance verified per spec §6.1"
```

If no edits (test still passed at original tolerance): skip commit.

---

## Phase D — Benchmark & Validation

### Task 13: Run benchmark + regenerate predictions

**Files:**
- Regenerate: `data/training/4track_holdout_predictions.json`

**Spec ref:** §6.3 (benchmark protocol).

- [ ] **Step 1: Verify pre-conditions for benchmark**

Confirm at least one item resolved as literature_applied or interpretation_resolved (else skip per §6.3 skip condition):

```bash
jq -r '.[] | select(type=="object" and has("v3_metadata")) | .v3_metadata.disposition_state' data/sbi/prodrug_activation_registry.json | sort | uniq -c
```

Expected: at least one `literature_applied` or `interpretation_resolved`.

If all `ceiling_accepted`: jump to Phase E with all-ceiling contingency (spec §7.2).

- [ ] **Step 2: Run benchmark**

```bash
python scripts/run_engine_benchmark.py
```

Expected: completes successfully, regenerates `data/training/4track_holdout_predictions.json`.

Runtime: typically 30-60 minutes (107 drugs × 1000 MC samples × 4 tracks). Use `--background` if available to monitor without blocking.

- [ ] **Step 3: Verify benchmark output**

```bash
python -c "
import json
d = json.load(open('data/training/4track_holdout_predictions.json'))
print(f'Drugs: {len(d)}')
print(f'Sample drug keys: {list(d.keys())[:3]}')
"
```

Expected: 107 drugs in output.

- [ ] **Step 4: Compute AAFE delta vs pre-v3**

```bash
python -c "
import json, math
post = json.load(open('data/training/4track_holdout_predictions.json'))
pre = json.load(open('tests/regression/data/prodrug_v3_pre_baseline.json'))
# Use git to get pre-v3 4track predictions (reference value):
# git show HEAD~N:data/training/4track_holdout_predictions.json
# Compute AAFE for both, report delta
"
```

Compare against `git show <pre-v3 commit>:data/training/4track_holdout_predictions.json` for canonical pre-v3 AAFE.

- [ ] **Step 5: Commit regenerated predictions**

```bash
git add data/training/4track_holdout_predictions.json
git commit -m "feat(prodrug-v3): regenerate 4track holdout predictions"
```

---

### Task 14: xfail removal per spec §6.4

**Files:**
- Modify: `tests/regression/test_prodrug_v2_validation_gate.py`

**Spec ref:** §6.4 (xfail procedure with NaN/inf hard-failure).

- [ ] **Step 1: Run gate test (current state)**

```bash
pytest tests/regression/test_prodrug_v2_validation_gate.py -v --no-header
```

Capture per-drug outcome:
- XFAIL: drug still fails 3-fold (expected)
- XPASS: drug now passes (strict=True will mark as FAIL — this is a flip signal)
- FAIL: regression or NaN/inf

- [ ] **Step 2: For each prodrug drug, apply §6.4 procedure**

```
For sepiapterin, remdesivir, fostamatinib, tebipenem_pivoxil:
    Compute v3 fold-error from current benchmark predictions vs clinical reference

    IF prediction non-finite (NaN, inf):
        STOP — block PR. Debug why v3 broke this drug. (Hard failure per §6.4)

    ELIF v3 fold-error <= 3.0:
        Remove @pytest.mark.xfail decorator from this drug's parametrize entry
        Update reason comment to: "v3 <disposition> → passes 3-fold"

    ELIF v3 fold-error > 3.0 AND v3 fold-error <= v2 fold-error (improvement):
        Keep xfail decorator
        Update reason: "v3 <fold>×, improvement vs v2 <prev_fold>×, ceiling per spec §5.X"

    ELIF v3 fold-error > v2 fold-error:
        REGRESSION RED FLAG
        Investigate: intended doctrine consequence OR unknown bug?
        - Intended (e.g., new literature value moves prediction in mechanistic direction): document in test reason, in PR body, proceed
        - Unknown bug: STOP — debug
```

- [ ] **Step 3: Edit gate test per Step 2 outcomes**

For each drug now passing, find its parametrize entry and:
- Remove the `pytest.mark.xfail(strict=True, reason=...)` decorator
- Update inline comment to reflect new state

For each drug still failing but improved or same:
- Keep xfail
- Update `reason=` string with v3 fold-error number and disposition reference

- [ ] **Step 4: Re-run gate test**

```bash
pytest tests/regression/test_prodrug_v2_validation_gate.py -v --no-header
```

Expected:
- All xfail-removed drugs: PASS (regular)
- All retained xfail drugs: XFAIL (expected fail)
- 0 XPASS (else strict failure)
- 0 FAIL (else regression or NaN — investigate)

- [ ] **Step 5: Commit xfail flip**

```bash
git add tests/regression/test_prodrug_v2_validation_gate.py
git commit -m "test(prodrug-v3): xfail flip per spec §6.4"
```

---

### Task 15: Snapshot regeneration

**Files:**
- Modify: `tests/regression/test_prodrug_v2_snapshot.py`

**Spec ref:** §6.1 (snapshot disposition: regenerate per-prodrug ±5%).

- [ ] **Step 1: Locate snapshot mechanism**

```bash
grep -n "snapshot\|expected\|approx" tests/regression/test_prodrug_v2_snapshot.py | head -30
```

Identify how expected Cmax values are stored (inline literals, fixture, or external file).

- [ ] **Step 2: Update snapshot expected values**

For each prodrug, update the hardcoded Cmax value to the v3 deterministic prediction:

```python
# v3 snapshot (regenerated 2026-04-29 post-Item N <disposition>)
EXPECTED_CMAX_SEPIAPTERIN = <new value from v3 deterministic prediction>  # was <v2 value>
EXPECTED_CMAX_REMDESIVIR = <new value or unchanged if Item 2 ceiling>
# ... etc
```

Add an inline comment per drug showing v2 → v3 transition for git history clarity.

- [ ] **Step 3: Run snapshot test**

```bash
pytest tests/regression/test_prodrug_v2_snapshot.py -v
```

Expected: PASS for all 4 drugs.

- [ ] **Step 4: Commit snapshot regeneration**

```bash
git add tests/regression/test_prodrug_v2_snapshot.py
git commit -m "test(prodrug-v3): snapshot regen per spec §6.1"
```

---

### Task 16: Update CLAUDE.md headline metrics table

**Files:**
- Modify: `CLAUDE.md` (top metrics table)

**Spec ref:** §6.5 (AAFE table update synchronous with predictions JSON).

- [ ] **Step 1: Read current CLAUDE.md metrics block**

```bash
grep -n "AAFE\|Track\|Meta" CLAUDE.md | head -20
```

Locate the "## Current Performance" section's table.

- [ ] **Step 2: Compute new AAFE values from v3 predictions**

```bash
python scripts/run_engine_benchmark.py --aafe-only --print-table 2>/dev/null || \
python -c "
import json, numpy as np
d = json.load(open('data/training/4track_holdout_predictions.json'))
# Adapt to actual JSON structure of predictions file
# Compute geometric mean of fold errors per track
"
```

If the script supports `--aafe-only` mode, use it. Else compute manually from predictions JSON structure.

Capture:
- Engine track AAFE (95% CI from bootstrap if benchmark provides)
- Meta track AAFE
- ML track AAFE
- VDss track AAFE
- In-domain Meta AAFE

- [ ] **Step 3: Update CLAUDE.md metrics table**

Edit `CLAUDE.md` "## Current Performance" table. Update each row's AAFE / 95% CI / %2-fold / %3-fold / N columns with v3 values.

Add a footnote to the table:
```markdown
> Updated 2026-04-29 post v3 input-data refresh (Items <list>). v1→v2→v3 fold-error per `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`.
```

- [ ] **Step 4: Verify CLAUDE.md still parses cleanly**

```bash
head -100 CLAUDE.md
```

Spot-check formatting integrity.

- [ ] **Step 5: Commit CLAUDE.md update**

```bash
git add CLAUDE.md
git commit -m "docs(claude): top metrics table updated post v3 refresh"
```

---

## Phase E — Documentation & PR

### Task 17: Literature deliverable consolidation

**Files:**
- Modify: `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`

**Spec ref:** §9.3 (literature deliverable as v3 PR diff component), §10 SC #2.

- [ ] **Step 1: Add summary table to literature deliverable**

Append at the end of `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`:

```markdown
---

## Summary Table — Item × Disposition

| # | Item | Disposition | Citation | v1/v2 → v3 Δ |
|---|---|---|---|---|
| 1 | BH4 CL/Vd | <state> | <citation or null> | <e.g., Vd 150→835 OR no change (ceiling)> |
| 2 | GS-441524 CL/Vd | <state> | <...> | <...> |
| 3 | R406 CL/Vd | <state> | <...> | <...> |
| 4 | tebipenem CL/Vd | <state> | <...> | <...> |
| 5 | SPR proteomic | <state> | <...> | <e.g., liver SPR 1e5→8e4 OR no change> |
| 6 | CES2/tebipenem CLint | <state> | <...> | <...> |

## Per-prodrug Cmax fold-error progression

| Drug | v1 | v2 | v3 | gate (3-fold)? |
|---|---|---|---|---|
| sepiapterin | 5356× over | 4692× over | <v3>× <over/under> | <pass/fail> |
| remdesivir | 4.45× under | 4.43× under | <v3> | <pass/fail> |
| fostamatinib | 4.78× under | 4.51× under | <v3> | <pass/fail> |
| tebipenem_pivoxil | 8.63× under | 9.02× under | <v3> | <pass/fail> |

## SBI-prodrug intersection re-check

Re-confirmed at v3 implementation (Task 1 Step 3): intersection ∅. SBI staleness warning not required.
```

- [ ] **Step 2: Self-review literature deliverable**

Read through the file. Check:
- Each item has §4.4 template fields filled
- No "TBD" or placeholder text remains
- Citations are full (Author Year + DOI/URL where available)
- Ceiling rationales are non-empty per spec §5.X fallbacks

- [ ] **Step 3: Commit literature deliverable consolidation**

```bash
git add docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md
git commit -m "docs(spec): consolidate v3 literature deliverable with summary tables"
```

---

### Task 18: CHANGELOG and experiment-log entries

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/claude/experiment-log.md`

**Spec ref:** §9.3, CLAUDE.md self-maintenance §2.

- [ ] **Step 1: Add CHANGELOG entry**

Edit `CHANGELOG.md`. At the top (most recent first), add:

```markdown
## v3 — Prodrug Activation Input-Data Quality Refresh (2026-04-29)

Resolution of T1 caution flags deferred from v2. Architecture unchanged.

### Items resolved
- Item 1 (BH4 CL/Vd, sepiapterin): <disposition>
- Item 2 (GS-441524 CL/Vd, remdesivir): <disposition>
- Item 3 (R406 CL/Vd, fostamatinib): <disposition>
- Item 4 (tebipenem CL/Vd): <disposition>
- Item 5 (SPR primary proteomic): <disposition>
- Item 6 (CES2/tebipenem direct CLint): <disposition>

### v1→v2→v3 fold-error progression

| Drug | v1 | v2 | v3 | Δ v2→v3 |
|---|---|---|---|---|
| sepiapterin | 5356× over | 4692× over | <v3>× | <pct> |
| remdesivir | 4.45× under | 4.43× under | <v3>× | <pct> |
| fostamatinib | 4.78× under | 4.51× under | <v3>× | <pct> |
| tebipenem_pivoxil | 8.63× under | 9.02× under | <v3>× | <pct> |

### Headline AAFE delta
- Engine track: <v2 value> → <v3 value>
- Meta track: 2.695 → <v3 value>

Per spec §3.3 mechanistic-A promise, gate-fail with mechanistic-A-compliant
values is acceptable outcome (informative not failing). v4 candidates: see
spec §3.2 out-of-scope and §7.2 contingency.

References: `docs/superpowers/specs/2026-04-29-prodrug-activation-v3-design.md`
+ `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`.
```

- [ ] **Step 2: Add experiment-log entry**

Edit `docs/claude/experiment-log.md`. At the top, add:

```markdown
## 2026-04-29 — Prodrug Activation v3 (input-data refresh)

**Commit:** <head SHA after Task 18 commit>
**Branch:** feat/prodrug-activation-v3
**Spec:** docs/superpowers/specs/2026-04-29-prodrug-activation-v3-design.md

**Outcome:**
- Engine AAFE: <v2 value> → <v3 value> (delta <pct>)
- Meta AAFE: 2.695 → <v3 value>
- 4-drug 3-fold gate: <X> pass, <Y> ceiling-with-improvement, <Z> ceiling-no-improvement
- Items resolved: <X> literature_applied, <Y> interpretation_resolved, <Z> ceiling_accepted

**Significance:**
v3 closes T1 caution flags deferred from v2 with mechanistic-A discipline.
<Brief 1-2 sentence interpretation: did data refresh move gate? What
remains as data ceiling for v4?>
```

- [ ] **Step 3: Commit CHANGELOG + experiment-log**

```bash
git add CHANGELOG.md docs/claude/experiment-log.md
git commit -m "docs(claude): CHANGELOG + experiment-log v3 entries"
```

---

### Task 19: Open v3 PR

**Files:** none (gh action)

**Spec ref:** §9.3, §10 success criteria.

- [ ] **Step 1: Final spec compliance check**

Run all v3 tests one more time:

```bash
pytest tests/integration/test_prodrug_v3_registry_schema.py \
       tests/regression/test_prodrug_v3_enzyme_leak_audit.py \
       tests/regression/test_prodrug_v2_validation_gate.py \
       tests/regression/test_prodrug_v2_snapshot.py \
       tests/integration/test_prodrug_v2_pipeline_smoke.py \
       tests/integration/test_prodrug_v2_ddi_smoke.py \
       tests/integration/test_prodrug_v2_mass_balance.py \
       tests/regression/test_prodrug_v2_identity_blind.py \
       -v -m "not slow"
```

Expected: all v3 tests + v2 tests (excluding slow leak audit) PASS.

- [ ] **Step 2: Run leak audit (slow) once for final verification**

```bash
pytest tests/regression/test_prodrug_v3_enzyme_leak_audit.py -v -m slow
```

Expected: PASS.

- [ ] **Step 3: Push v3 branch**

```bash
git push -u origin feat/prodrug-activation-v3
```

- [ ] **Step 4: Open PR**

```bash
gh pr create --base main --head feat/prodrug-activation-v3 --title "feat(prodrug): v3 input-data quality refresh" --body "$(cat <<'EOF'
## Summary

v3 resolves T1 caution flags deferred from v2 (active species CL/Vd, SPR proteomic abundance, CES2/tebipenem direct kinetics). Each flag closes with documented disposition (`literature_applied` / `interpretation_resolved` / `ceiling_accepted`) per spec §2 acceptance gate. Architecture from v2 unchanged; pure data-quality refresh with mechanistic-A discipline preserved.

## Per-item disposition

| # | Item | Disposition | Affects |
|---|---|---|---|
| 1 | BH4 active CL/Vd | <state> | sepiapterin |
| 2 | GS-441524 active CL/Vd | <state> | remdesivir |
| 3 | R406 active CL/Vd | <state> | fostamatinib |
| 4 | tebipenem active CL/Vd | <state> | tebipenem_pivoxil |
| 5 | SPR primary proteomic | <state> | sepiapterin |
| 6 | CES2/tebipenem CLint | <state> | tebipenem_pivoxil |

## v1→v2→v3 fold-error progression

| Drug | v1 | v2 | v3 |
|---|---|---|---|
| sepiapterin | 5356× over | 4692× over | <v3>× |
| remdesivir | 4.45× under | 4.43× under | <v3>× |
| fostamatinib | 4.78× under | 4.51× under | <v3>× |
| tebipenem_pivoxil | 8.63× under | 9.02× under | <v3>× |

## Headline AAFE

- Engine: <v2 → v3>
- Meta: 2.695 → <v3>

## Test plan

- [x] Schema test (registry_schema): all 4 prodrug entries valid
- [x] Leak audit (enzyme_leak_audit, slow): expected_unchanged drugs byte-identical
- [x] Validation gate post xfail-flip: all CI green
- [x] Snapshot regeneration
- [x] Pipeline smoke (functional-only refactor)
- [x] DDI smoke (tolerance verified)
- [x] Mass balance + identity-blind invariance (v2 invariants)

## Mechanistic-A discipline

Per spec §3.3 mechanistic-A promise, **gate-fail with literature-grounded values is acceptable outcome** — values are NOT clinical-fit. <X> drugs pass 3-fold, <Y> remain xfail with documented improvement, <Z> remain xfail at ceiling.

## Out-of-scope (v4 candidates)

Per spec §3.2: SBI retraining (intersection check confirmed empty per §8.2 — not v3-blocking), CES1/ALPI abundance refresh (no T1 caution), 1-comp→2-comp active species, PI recalibration.

## References

- v3 spec: \`docs/superpowers/specs/2026-04-29-prodrug-activation-v3-design.md\`
- v3 plan: \`docs/superpowers/plans/2026-04-29-prodrug-activation-v3.md\`
- v3 literature deliverable: \`docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md\`
- v2 PR: #7 (merged <date>)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

- [ ] **Step 5: Verify PR opens cleanly**

```bash
gh pr view --json state,mergeable,statusCheckRollup
```

Expected: `state: OPEN`, `mergeable: MERGEABLE` (or `UNKNOWN` while CI runs).

- [ ] **Step 6: No commit at this step (PR is the deliverable)**

---

## Self-Review (writing-plans skill checklist)

### Spec coverage check

| Spec section | Plan task(s) | Coverage |
|---|---|---|
| §1 Goal | Tasks 4-9 (per-item resolution) | ✓ |
| §2 Success criterion (acceptance gates) | Tasks 4-9 disposition decisions; §6.2 schema test (Task 3) | ✓ |
| §3 Scope (combined v3, 6 items) | Phase B (Tasks 4-9) | ✓ |
| §3.2 Out-of-scope | Not implemented (correct — out of scope) | ✓ |
| §4.1 Mean-Value Doctrine | Tasks 4-7 doctrine application steps | ✓ |
| §4.2 CV Doctrine | Tasks 4-9 CV rule selection | ✓ |
| §4.3 Source Exhaustiveness | Tasks 4-9 search corpus + 30-cap | ✓ |
| §4.4 Documentation Template | Tasks 4-9 literature deliverable sections | ✓ |
| §4.5 Items 5-6 Parallel Doctrine | Tasks 8-9 | ✓ |
| §5.1-5.6 Per-item Acceptance Gates | Tasks 4-9 | ✓ |
| §6.1 Existing test reuse | Tasks 10 (smoke refactor), 12 (ddi), 14 (gate), 15 (snapshot) | ✓ |
| §6.2 New v3 tests | Tasks 3 (schema), 11 (leak audit) | ✓ |
| §6.3 Benchmark protocol | Task 13 | ✓ |
| §6.4 xfail removal | Task 14 | ✓ |
| §6.5 AAFE table update | Task 16 | ✓ |
| §7 Risks | Implicit in per-task fallbacks (e.g., Task 4 BH4 ceiling fallback chain) | ✓ |
| §8.1 v2-v3 sequencing | Task 1 Step 1 (verify v2 merged) | ✓ |
| §8.2 SBI intersection re-check | Task 1 Step 3 | ✓ |
| §9.3 Implementation deliverables | Tasks 4-19 produce all listed files | ✓ |
| §10 Success criteria | Task 19 final compliance check + PR diff | ✓ |

### Placeholder scan

- "TBD"/"TODO"/"implement later": none in plan body
- "Add appropriate error handling": replaced with concrete spec rules per item
- "Write tests for the above": tests are explicit per task
- "Similar to Task N": Tasks 5-7 share structure with Task 4 but each has its own search terms and full code; no abbreviation
- Steps without code: only verification steps (Step 1 of various tasks); these have explicit commands + expected output

### Type consistency check

- `v3_metadata` block name used consistently across Task 3 (schema), Tasks 4-9 (registry updates), Task 11 (leak audit references it indirectly)
- `disposition_state` enum consistent: `{literature_applied, interpretation_resolved, ceiling_accepted}` across Task 3 schema, Tasks 4-9 documentation, Task 11 logic
- `CHANGED_ENZYME_ABUNDANCES` / `DRUG_SPECIFIC_CHANGES` constant names consistent in Task 11
- Disposition state names match spec §2 exactly

### Gaps surfaced (to consider during execution)

1. **Task 13 Step 4 AAFE delta computation script structure unknown** — implementer may need to inspect `4track_holdout_predictions.json` schema to compute AAFE. Acknowledged as runtime adaptation, not a plan defect.

2. **Task 12 ddi_smoke API for DDI scaling not detailed** — depends on existing `test_prodrug_v2_ddi_smoke.py` implementation pattern. Plan delegates to "mirror that pattern". Acceptable per "Modify" file-list semantics.

3. **Task 8 Step 4 yaml SPR entries — implementer must locate exact YAML keys** — `data/physiology/reference_man.yaml` structure not enumerated. Acknowledged; implementer reads file.

These gaps are intentional: the plan provides sufficient context for an implementer to navigate, without prescribing every line of generated code where existing patterns already exist.

---

## Plan Complete

Plan saved to `docs/superpowers/plans/2026-04-29-prodrug-activation-v3.md`. 19 tasks across 5 phases (A: Setup, B: Per-Item Resolution, C: Test Infrastructure, D: Benchmark, E: Documentation/PR).

**Two execution options per writing-plans skill:**

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review (spec compliance + code quality)
2. **Inline Execution** — execute tasks in this session with batch checkpoints

User chooses approach; implementation begins thereafter (subject to spec §8.1 precondition: v2 PR #7 must be merged first).
