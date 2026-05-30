# B-14 Hepatic UGT IVIVE Differential — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a predict-side, per-substrate/per-enzyme UGT in-vitro→in-vivo scaling-factor (SF) registry that corrects hepatic UGT under-prediction, and run it as a *bounded blind decisive experiment* whose expected, first-class terminal is DE-40 (no-op).

**Architecture:** A no-op-by-default JSON registry (`ugt_ivive_sf.json`) + a loader in `non_cyp_substrates.py` + a one-line multiply in `_decompose_clint` (engine untouched, identity-blind preserved). Infrastructure ships as a bit-identical no-op (Tasks 1–4). A blind, bounded literature step (Task 5) sets the SFs; a pre-registration + go/no-go (Task 6) decides whether to apply them (Tasks 7–8) or retire to DE-40.

**Tech Stack:** Python 3.10+, pytest, RDKit (InChIKey), miniconda interpreter `/opt/miniconda3/bin/python3` (numerics-stack-consistent with the committed cache).

**Spec:** `docs/superpowers/specs/2026-05-30-hepatic-ugt-ivive-differential-design.md` (v2).

**Branch:** `b14-hepatic-ugt-ivive` (already created; spec committed at `d674a6a`).

**Numerics discipline:** All `predict()`/benchmark commands MUST use `/opt/miniconda3/bin/python3` (numpy 2.2.6) so Gate-D comparisons are same-stack. Homebrew `python3` lacks numpy and drifts.

---

## File Structure

- **Create** `data/enzymes/ugt_ivive_sf.json` — the SF registry (seeded all-1.0 no-op; Task 5 populates).
- **Modify** `src/sisyphus/predict/non_cyp_substrates.py` — add `_load_ugt_ivive_sf_index()` + `get_ugt_ivive_sf()`.
- **Modify** `src/sisyphus/predict/ivive.py` — `_decompose_clint` param + multiply; `build_drug_on_graph` wiring.
- **Create** `tests/unit/test_ugt_ivive_sf.py` — loader + `_decompose_clint` hook unit tests.
- **Create** `tests/regression/test_ugt_ivive_sf_registry_schema.py` — registry integrity test.
- **Modify (docs, Task 8)** `docs/claude/experiment-log.md`, `docs/claude/dead-ends.md`, `tests/integration/test_holdout_regression.py` docstring, and `CLAUDE.md` top table *only if* a headline moves.

---

## Task 1: SF registry scaffold + schema test (no-op)

**Files:**
- Create: `data/enzymes/ugt_ivive_sf.json`
- Test: `tests/regression/test_ugt_ivive_sf_registry_schema.py`

- [ ] **Step 1: Create the registry seeded all-1.0 (no-op).** Copy the exact `inchikey` for each of the 8 drugs from `data/enzymes/ugt2b7_substrates.json` (morphine, codeine, ketorolac, indomethacin) and `data/enzymes/ugt1a9_substrates.json` (dapagliflozin, etodolac, bexagliflozin, glasdegib) — do **not** hand-type InChIKeys. Each UGT2B7 seed gets `"ivive_sf": {"UGT2B7": 1.0}`; each UGT1A9 seed gets `"ivive_sf": {"UGT1A9": 1.0}`.

```json
{
  "version": 1,
  "description": "Per-substrate, per-enzyme UGT hepatocyte-basis in-vitro->in-vivo scaling factors (B-14). Default 1.0 = no-op. Spec: docs/superpowers/specs/2026-05-30-hepatic-ugt-ivive-differential-design.md. NOTE: this SF acts at EVERY node carrying the enzyme (today liver + gut UGT2B7, reference_man.yaml); a future kidney-UGT node REQUIRES basis re-derivation to avoid double-counting the renal fraction withheld in Phase 0.",
  "substrates": [
    {"drug": "morphine", "inchikey": "<copy from ugt2b7_substrates.json>", "ivive_sf": {"UGT2B7": 1.0}, "basis": "hepatocyte", "hepatic_fraction_of_deficit": 1.0, "renal_fraction_withheld": 0.0, "disposition": "default_1.0", "literature": []},
    {"drug": "codeine", "inchikey": "<copy>", "ivive_sf": {"UGT2B7": 1.0}, "basis": "hepatocyte", "hepatic_fraction_of_deficit": 1.0, "renal_fraction_withheld": 0.0, "disposition": "default_1.0", "literature": []},
    {"drug": "ketorolac", "inchikey": "<copy>", "ivive_sf": {"UGT2B7": 1.0}, "basis": "hepatocyte", "disposition": "default_1.0", "literature": []},
    {"drug": "indomethacin", "inchikey": "<copy>", "ivive_sf": {"UGT2B7": 1.0}, "basis": "hepatocyte", "disposition": "default_1.0", "literature": []},
    {"drug": "dapagliflozin", "inchikey": "<copy from ugt1a9_substrates.json>", "ivive_sf": {"UGT1A9": 1.0}, "basis": "hepatocyte", "disposition": "default_1.0", "literature": []},
    {"drug": "etodolac", "inchikey": "<copy>", "ivive_sf": {"UGT1A9": 1.0}, "basis": "hepatocyte", "disposition": "default_1.0", "literature": []},
    {"drug": "bexagliflozin", "inchikey": "<copy>", "ivive_sf": {"UGT1A9": 1.0}, "basis": "hepatocyte", "disposition": "default_1.0", "literature": []},
    {"drug": "glasdegib", "inchikey": "<copy>", "ivive_sf": {"UGT1A9": 1.0}, "basis": "hepatocyte", "disposition": "default_1.0", "literature": []}
  ]
}
```

(The `<copy ...>` markers are an explicit instruction to copy the real InChIKey from the named registry — anti-confabulation. They MUST be replaced with the real keys before the test in Step 3 passes.)

- [ ] **Step 2: Write the failing schema test.**

```python
"""B-14 registry integrity: enforces the anti-fudge + correct-basis invariants."""
from __future__ import annotations
import json
import pathlib

_PATH = pathlib.Path("data/enzymes/ugt_ivive_sf.json")
_VALID_BASIS = {"hepatocyte", "hepatocyte_scaled"}
_VALID_DISP = {"literature_applied", "ceiling_accepted", "not_applicable", "default_1.0"}


def test_ugt_ivive_sf_schema():
    data = json.loads(_PATH.read_text())
    subs = data["substrates"]
    assert subs, "registry must list the seed substrates"
    seen = set()
    for e in subs:
        ikey = e["inchikey"]
        assert ikey not in seen, f"duplicate inchikey {ikey}"
        seen.add(ikey)
        assert "<copy" not in ikey, f"{e['drug']}: placeholder InChIKey not replaced"
        sf = e["ivive_sf"]
        assert isinstance(sf, dict) and sf, f"{e['drug']}: ivive_sf must be a non-empty map"
        assert all(k.startswith("UGT") for k in sf), f"{e['drug']}: SF keys must be UGT tags"
        disp = e["disposition"]
        assert disp in _VALID_DISP, f"{e['drug']}: bad disposition {disp}"
        if disp in {"default_1.0", "not_applicable", "ceiling_accepted"}:
            assert all(v == 1.0 for v in sf.values()), f"{e['drug']}: {disp} entries must be exactly 1.0"
        if disp == "literature_applied":
            assert any(v != 1.0 for v in sf.values()), f"{e['drug']}: literature_applied but all 1.0"
            assert e["basis"] in _VALID_BASIS, f"{e['drug']}: basis must be in {_VALID_BASIS}"
            lits = e.get("literature", [])
            assert lits and all(l.get("verified") and l.get("pmid_or_doi") for l in lits), \
                f"{e['drug']}: literature_applied needs a verified PMID/DOI"
            if any(v > 5 for v in sf.values()):
                assert len(lits) >= 2, f"{e['drug']}: ivive_sf>5 needs a second verifying source"
        if e["drug"] in {"morphine", "codeine"}:
            assert "hepatic_fraction_of_deficit" in e, f"{e['drug']}: must record hepatic/renal partition"
```

- [ ] **Step 3: Run the test (after replacing the `<copy>` InChIKeys).**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/regression/test_ugt_ivive_sf_registry_schema.py -v`
Expected: PASS (scaffold is all `default_1.0` / `1.0`).

- [ ] **Step 4: Commit.**

```bash
git add data/enzymes/ugt_ivive_sf.json tests/regression/test_ugt_ivive_sf_registry_schema.py
git commit -m "feat(b14): ugt_ivive_sf registry scaffold (no-op) + schema test"
```

---

## Task 2: Loader `get_ugt_ivive_sf`

**Files:**
- Modify: `src/sisyphus/predict/non_cyp_substrates.py`
- Test: `tests/unit/test_ugt_ivive_sf.py`

- [ ] **Step 1: Write the failing loader tests.**

```python
"""B-14 loader unit tests."""
from __future__ import annotations
from sisyphus.predict.non_cyp_substrates import get_ugt_ivive_sf

_CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # not a UGT substrate


def test_unlisted_returns_empty():
    assert get_ugt_ivive_sf(_CAFFEINE) == {}


def test_invalid_smiles_returns_empty_no_raise():
    assert get_ugt_ivive_sf("not_a_smiles") == {}
    assert get_ugt_ivive_sf("") == {}


def test_seed_returns_ugt_map():
    # morphine SMILES — copy the exact string from data/enzymes/ugt2b7_substrates.json
    morphine = "<copy morphine smiles from ugt2b7_substrates.json>"
    sf = get_ugt_ivive_sf(morphine)
    assert "UGT2B7" in sf
    assert isinstance(sf["UGT2B7"], float)  # value set by Phase 0; structure stable
```

- [ ] **Step 2: Run to verify it fails.**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/unit/test_ugt_ivive_sf.py -v`
Expected: FAIL (`ImportError: cannot import name 'get_ugt_ivive_sf'`).

- [ ] **Step 3: Add the path constant, index loader, and lookup.** In `non_cyp_substrates.py`, add the path constant beside the existing ones (after line 26):

```python
_UGT_IVIVE_SF_PATH = _REPO_ROOT / "data" / "enzymes" / "ugt_ivive_sf.json"
```

Add the index loader beside the other `_load_*_index` functions:

```python
@lru_cache(maxsize=1)
def _load_ugt_ivive_sf_index() -> dict[str, dict]:
    """Return {inchikey: entry} for ugt_ivive_sf.json (B-14)."""
    if not _UGT_IVIVE_SF_PATH.exists():
        return {}
    data = json.loads(_UGT_IVIVE_SF_PATH.read_text())
    return {entry["inchikey"]: entry for entry in data.get("substrates", [])}
```

Add the lookup at the end of the file (after `get_non_cyp_fractions`), with a section comment marking the distinct concern:

```python
# --- IVIVE magnitude correction (B-14) ---------------------------------------
# Distinct from the fm-routing lookups above: fm decides WHICH enzyme carries the
# clearance; this SF decides HOW MUCH the in-vitro CLint under-predicts in vivo.
def get_ugt_ivive_sf(smiles: str) -> dict[str, float]:
    """Return {UGT_tag: scaling_factor} for the SMILES, or {} if unlisted/invalid.

    UNLIKE the lookup_* functions above (which return None), this returns a dict
    and NEVER raises: invalid SMILES -> {}. The {} default makes the caller's
    ``.get(enzyme, 1.0)`` a bit-identical no-op. See spec
    docs/superpowers/specs/2026-05-30-hepatic-ugt-ivive-differential-design.md.
    """
    ikey = _smiles_to_inchikey(smiles)
    if ikey is None:
        return {}
    entry = _load_ugt_ivive_sf_index().get(ikey)
    if entry is None:
        return {}
    return {k: float(v) for k, v in entry.get("ivive_sf", {}).items()}
```

- [ ] **Step 4: Run to verify it passes** (after replacing the morphine SMILES placeholder in the test).

Run: `/opt/miniconda3/bin/python3 -m pytest tests/unit/test_ugt_ivive_sf.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit.**

```bash
git add src/sisyphus/predict/non_cyp_substrates.py tests/unit/test_ugt_ivive_sf.py
git commit -m "feat(b14): get_ugt_ivive_sf loader (returns {} default, no raise)"
```

---

## Task 3: `_decompose_clint` per-enzyme SF hook

**Files:**
- Modify: `src/sisyphus/predict/ivive.py:244-308`
- Test: `tests/unit/test_ugt_ivive_sf.py` (append)

- [ ] **Step 1: Find the Distribution import.** Run `grep -n "Distribution" src/sisyphus/predict/ivive.py | head -3` and use that exact import in the test below.

- [ ] **Step 2: Append the failing hook tests** to `tests/unit/test_ugt_ivive_sf.py`:

```python
from sisyphus.predict.ivive import _decompose_clint
from <Distribution import from Step 1> import Distribution  # match ivive.py


def _aff(ugt_ivive_sf=None, fractions=None, ugt=None, ctype="base"):
    clint = Distribution(mean=50.0, cv=0.3)
    return _decompose_clint(
        clint, ctype, None,
        ugt_enzymes=ugt or {"UGT2B7"},
        non_cyp_fractions=fractions or {"UGT2B7": 0.85},
        ugt_ivive_sf=ugt_ivive_sf,
    )


def test_sf_none_and_empty_are_noop():
    base = _aff()
    assert _aff(ugt_ivive_sf=None)["UGT2B7"].mean == base["UGT2B7"].mean
    assert _aff(ugt_ivive_sf={})["UGT2B7"].mean == base["UGT2B7"].mean


def test_sf_scales_only_the_named_ugt():
    base = _aff()
    scaled = _aff(ugt_ivive_sf={"UGT2B7": 3.0})
    assert scaled["UGT2B7"].mean == base["UGT2B7"].mean * 3.0
    for tag in base:
        if tag != "UGT2B7":
            assert scaled[tag].mean == base[tag].mean, f"{tag} (non-UGT2B7) must be unchanged"


def test_multi_ugt_scales_each_tag_independently():
    fr = {"UGT2B7": 0.4, "UGT1A9": 0.4}
    base = _aff(fractions=fr, ugt={"UGT2B7", "UGT1A9"}, ctype="neutral")
    scaled = _aff(ugt_ivive_sf={"UGT2B7": 2.0, "UGT1A9": 5.0}, fractions=fr,
                  ugt={"UGT2B7", "UGT1A9"}, ctype="neutral")
    assert scaled["UGT2B7"].mean == base["UGT2B7"].mean * 2.0
    assert scaled["UGT1A9"].mean == base["UGT1A9"].mean * 5.0
```

- [ ] **Step 3: Run to verify it fails.**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/unit/test_ugt_ivive_sf.py -v`
Expected: FAIL (`_decompose_clint() got an unexpected keyword argument 'ugt_ivive_sf'`).

- [ ] **Step 4: Add the param and the multiply.** In `ivive.py`, add the parameter to the `_decompose_clint` signature (after `non_cyp_fractions` on line 252):

```python
    non_cyp_fractions: dict[str, float] | None = None,
    ugt_ivive_sf: dict[str, float] | None = None,
```

Add one line inside the loop, immediately after `scaled_affinity = max(affinity, 0.0) * metabolic_fraction` (line 306). The SF map contains only UGT keys, so `.get(enzyme, 1.0)` is a no-op for every non-UGT enzyme:

```python
        scaled_affinity = max(affinity, 0.0) * metabolic_fraction
        # B-14: hepatic UGT IVIVE differential — scales ONLY the UGT-routed affinity.
        scaled_affinity *= (ugt_ivive_sf or {}).get(enzyme, 1.0)
```

Add a one-line `Args` doc for `ugt_ivive_sf` in the docstring (after the `metabolic_fraction` arg block):

```python
        ugt_ivive_sf: Per-UGT-enzyme in-vitro->in-vivo scaling factor map
            (B-14). ``None``/``{}`` is a bit-identical no-op. ``{"UGT2B7": k}``
            multiplies the UGT2B7-routed affinity by k; non-UGT enzymes are
            untouched (the map carries only UGT keys).
```

- [ ] **Step 5: Run to verify it passes.**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/unit/test_ugt_ivive_sf.py -v`
Expected: PASS (6 tests total).

- [ ] **Step 6: Commit.**

```bash
git add src/sisyphus/predict/ivive.py tests/unit/test_ugt_ivive_sf.py
git commit -m "feat(b14): _decompose_clint per-enzyme UGT IVIVE SF hook (no-op default)"
```

---

## Task 4: Wire `build_drug_on_graph` + prove no-op invariance

**Files:**
- Modify: `src/sisyphus/predict/ivive.py:654-679`

- [ ] **Step 1: Add the SF lookup in `build_drug_on_graph`.** Immediately after the `get_non_cyp_fractions` block (after line 657, where `ugt_enzymes` is set), add:

```python
    # B-14: per-substrate UGT IVIVE scaling factor (hepatocyte-basis, hepatic-
    # fraction-only). Default {} -> bit-identical no-op. Spec 2026-05-30.
    from sisyphus.predict.non_cyp_substrates import get_ugt_ivive_sf
    ugt_ivive_sf = get_ugt_ivive_sf(profile.smiles)
```

- [ ] **Step 2: Pass it to `_decompose_clint`.** Add the kwarg to the call at line 672-679:

```python
    enzyme_affinity = _decompose_clint(
        adme.clint, profile.compound_type, profile.pka,
        enzyme_abundances=abundances,
        substrate_enzymes=substrate_enzymes,
        ugt_enzymes=ugt_enzymes,
        metabolic_fraction=metabolic_fraction,
        non_cyp_fractions=non_cyp_fractions,
        ugt_ivive_sf=ugt_ivive_sf,
    )
```

- [ ] **Step 3: Prove the no-op (Gate D1 against current cache).** With the registry all-`default_1.0`, every holdout drug must be predicted bit-identically.

```bash
cp data/training/4track_holdout_predictions.json /tmp/4track_preB14.json
/opt/miniconda3/bin/python3 -c "
import json
from sisyphus.pipeline.predict import predict
from sisyphus.validation.reference import load_reference
cache=json.load(open('/tmp/4track_preB14.json'))
by={d['name'].lower():d for d in cache['drugs']}
refs={r.name.lower():r for r in load_reference() if r.in_holdout}
bad=[]
for name,d in by.items():
    r=refs.get(name)
    if not r: continue
    fresh=predict(r.smiles,r.dose_mg,r.route).pk.cmax.mean
    if fresh != d['meta']:
        bad.append((name, d['meta'], fresh))
print('NON-BIT-IDENTICAL:', len(bad))
for b in bad[:10]: print(' ', b)
assert not bad, 'B-14 infra is NOT a no-op with all-1.0 registry'
print('OK: 107/107 bit-identical no-op')
"
```
Expected: `OK: 107/107 bit-identical no-op`.

- [ ] **Step 4: Confirm the unit + schema suite still green.**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/unit/test_ugt_ivive_sf.py tests/regression/test_ugt_ivive_sf_registry_schema.py tests/integration/test_ugt_path_mechanism.py -q`
Expected: all pass.

Also run the **engine identity-blind invariance** guard (locate via `grep -rl "identity\|random.*rename\|rename.*node" tests/`) — it must still pass, since `engine/` is unchanged (0 lines). This is the spec's "engine untouched" assertion.

- [ ] **Step 5: Commit (infrastructure ships as a no-op).**

```bash
git add src/sisyphus/predict/ivive.py
git commit -m "feat(b14): wire get_ugt_ivive_sf into build_drug_on_graph (no-op, D1 107/107 bit-identical)"
```

---

## Task 5: Phase 0 — blind, bounded SF derivation (the decisive experiment)

**Files:**
- Modify: `data/enzymes/ugt_ivive_sf.json` (populate verified SFs / confirm 1.0)

This is a **literature-curation task**, not code. Follow the spec §"Phase 0" exactly. It produces the SF values that determine go/no-go.

- [ ] **Step 1: For each of the 8 seeds, run a bounded, blind verification.** Hard limits per substrate: **≤2 sequential agents, ≤12 WebFetch calls, single session.** The verifier is told the drug + its UGT tag and asked for the **hepatocyte-basis** (or HLM explicitly scaled to hepatocyte-equivalent, scaling documented) **in-vivo/in-vitro glucuronidation CLint ratio**, tied to **one named primary-source quantity**. The verifier is **NOT** given any holdout Cmax/fold. Do not run all 8 in one parallel workflow (the prior 8-parallel run stalled).

- [ ] **Step 2: Partition hepatic vs renal for morphine and codeine.** Using Knights 2016 (PMID 26808419) / fraction-excreted-as-glucuronide data, apply only the **hepatic-attributable fraction** of the deficit to the hepatic SF. Record `hepatic_fraction_of_deficit` and `renal_fraction_withheld` in the entry.

- [ ] **Step 3: Adversarial confirmation on morphine + codeine.** One independent skeptic per drug confirms (a) the PMID resolves and the number is in the source (DE-39 anti-confabulation), and (b) the fold is hepatic-only, not renal/albumin-HLM-contaminated. Refute-by-default.

- [ ] **Step 4: Ratio-of-ratios guard (spec risk #4).** An IVIVE ratio is only valid relative to the in-vitro CLint it was paired with; the engine's baseline is the *ML-predicted* CLint (R²≈0.24), not that in-vitro value. For each `literature_applied` candidate, record the ratio of the engine's baseline UGT-routed CLint to the literature in-vitro CLint the SF was derived from. If they diverge `>~2×`, do NOT multiply blindly — re-anchor by targeting the absolute in-vivo glucuronidation CLint and backing out the implied SF for *this* engine's baseline, or set the SF to 1.0 (`ceiling_accepted`). Note the divergence in the entry's `literature` block.

- [ ] **Step 5: Apply the default rule.** Any substrate whose hepatocyte-basis hepatic-fraction SF is not pinned to a single verified primary-source number within budget → keep `ivive_sf` at 1.0, `disposition: "ceiling_accepted"` (or `not_applicable` if UGT is not rate-limiting, e.g. glasdegib ~7% UGT / CYP3A4-dominated). **Never invent, never take a range midpoint.** Set `disposition: "literature_applied"`, `basis`, and `literature: [{citation, pmid_or_doi, reported_value, basis, verified: true}]` only for pinned values. For `indomethacin` specifically (fm[UGT2B7]=0.15, in-AD): verify its *own* hepatocyte UGT SF and apply it if real (mechanism-honest even though it worsens the fold), or record it as a counted in-domain casualty — do not rely on "low fm" without the arithmetic.

- [ ] **Step 6: Re-run the schema test on the populated registry.**

Run: `/opt/miniconda3/bin/python3 -m pytest tests/regression/test_ugt_ivive_sf_registry_schema.py -v`
Expected: PASS (now validates the verified entries: basis, PMID, >5 second-source, morphine/codeine partition).

- [ ] **Step 7: Commit the populated registry.**

```bash
git add data/enzymes/ugt_ivive_sf.json
git commit -m "data(b14): Phase 0 blind-verified hepatocyte-basis UGT IVIVE SFs"
```

---

## Task 6: Pre-register predictions + GO/NO-GO

**Files:**
- Modify: `docs/claude/experiment-log.md` (append the pre-registration block at top)

- [ ] **Step 1: Compute and freeze the predicted per-drug effect.** With the populated registry, predict each seed and compare to the pre-B-14 baseline; compute the full-107 Meta Δ.

```bash
/opt/miniconda3/bin/python3 scripts/run_engine_benchmark.py --save-json /tmp/4track_B14_predicted.json 2>/dev/null | tail -4
/opt/miniconda3/bin/python3 -c "
import json
base=json.load(open('/tmp/4track_preB14.json')); new=json.load(open('/tmp/4track_B14_predicted.json'))
print('Meta', base['overall']['meta']['aafe'], '->', new['overall']['meta']['aafe'],
      'delta', new['overall']['meta']['aafe']-base['overall']['meta']['aafe'])
bd={d['name'].lower():d for d in base['drugs']}; nd={d['name'].lower():d for d in new['drugs']}
for n in ['morphine','codeine','indomethacin','ketorolac','dapagliflozin','etodolac','bexagliflozin','glasdegib']:
    if n in nd and n in bd:
        print(f\"{n:14s} meta_fold {bd[n].get('meta_fold')} -> {nd[n].get('meta_fold')}\")
"
```

- [ ] **Step 2: Write the pre-registration** to `experiment-log.md` (top): the frozen SF values (from the committed registry SHA), the predicted per-drug meta_fold shifts, and the predicted Meta Δ. This is the record the D3 honesty gate checks against.

- [ ] **Step 3: GO/NO-GO decision (pre-committed).**
  - **NO-GO** if predicted Meta improvement `|Δ| < 0.02` (a full morphine 3.38→2.0 + codeine 1.78→1.3 fix is only ≈ −0.021, so any realistic partial fix is sub-threshold) → **skip to Task 8a (DE-40)**. The infra ships as the audited no-op it already is.
  - **GO** if predicted Meta improvement `≥ 0.02` and morphine/codeine do not flip under 1.0 → **proceed to Task 7**.

- [ ] **Step 4: Commit the pre-registration.**

```bash
git add docs/claude/experiment-log.md
git commit -m "docs(b14): Phase 0 pre-registration + go/no-go decision"
```

---

## Task 7: Apply + acceptance gates (GO path only)

**Files:**
- Modify: `data/training/4track_holdout_predictions.json` (regenerated cache)

- [ ] **Step 1: Regenerate the canonical cache** (registry already populated in Task 5).

```bash
/opt/miniconda3/bin/python3 scripts/run_engine_benchmark.py --save-json data/training/4track_holdout_predictions.json 2>/dev/null | tail -4
```

- [ ] **Step 2: Gate D1 — non-seeded bit-identity.** Every drug NOT in the registry-with-SF≠1.0 set must equal `/tmp/4track_preB14.json` exactly.

```bash
/opt/miniconda3/bin/python3 -c "
import json
base=json.load(open('/tmp/4track_preB14.json')); new=json.load(open('data/training/4track_holdout_predictions.json'))
sf=json.load(open('data/enzymes/ugt_ivive_sf.json'))
seeded={s['drug'].lower() for s in sf['substrates'] if any(v!=1.0 for v in s['ivive_sf'].values())}
bd={d['name'].lower():d for d in base['drugs']}; nd={d['name'].lower():d for d in new['drugs']}
bad=[n for n in nd if n not in seeded and n in bd and nd[n]['meta']!=bd[n]['meta']]
print('seeded(SF!=1):', sorted(seeded)); print('D1 violations (non-seeded shifted):', bad)
assert not bad, 'D1 FAIL: a non-seeded drug shifted'
print('D1 OK')
"
```
Expected: `D1 OK`.

- [ ] **Step 3: Gate D2a/D2b — direction + flip guard.**

```bash
/opt/miniconda3/bin/python3 -c "
import json
base=json.load(open('/tmp/4track_preB14.json')); new=json.load(open('data/training/4track_holdout_predictions.json'))
sf=json.load(open('data/enzymes/ugt_ivive_sf.json'))
seeded={s['drug'].lower() for s in sf['substrates'] if any(v!=1.0 for v in s['ivive_sf'].values())}
bd={d['name'].lower():d for d in base['drugs']}; nd={d['name'].lower():d for d in new['drugs']}
flips=[]
for n in seeded:
    if n not in nd: continue
    assert nd[n]['eng']<=bd[n]['eng'], f'D2a FAIL {n}: engine Cmax not down'
    if (bd[n]['meta_fold']-1.0)*(nd[n]['meta_fold']-1.0) < 0: flips.append(n)
    print(f\"{n:14s} eng {bd[n]['eng']:.4g}->{nd[n]['eng']:.4g}  meta_fold {bd[n]['meta_fold']:.3g}->{nd[n]['meta_fold']:.3g}\")
assert not flips, f'D2b FAIL: meta_fold crossed 1.0 (over->under) for {flips} — SF over-sized'
print('D2a/D2b OK')
"
```
Expected: `D2a/D2b OK` (if a seed flips, the SF is over-sized — return to Task 5 to re-verify that drug's hepatic fraction; do NOT shrink the SF to hit a target).

- [ ] **Step 4: Gate D3 — prediction-match.** Confirm the realized seed meta_folds equal the Task 6 pre-registered predictions (same SFs, frozen registry SHA — a mismatch means the registry was edited after pre-registration). Compare `data/training/4track_holdout_predictions.json` seed folds against the experiment-log pre-registration block; they must match to full precision.

- [ ] **Step 5: Commit the cache.**

```bash
git add data/training/4track_holdout_predictions.json
git commit -m "data(b14): regen 4-track cache under verified UGT IVIVE SFs (D1/D2/D3 pass)"
```

---

## Task 8: Docs + ship

### Task 8a — NO-GO / DE-40 path

- [ ] **Step 1: Add DE-40 to `docs/claude/dead-ends.md`** (after DE-39): the hepatic UGT IVIVE differential, done honestly (hepatocyte-basis, hepatic-fraction-only, blind), produces SFs too small to move morphine/codeine beyond noise; record the verified per-drug SFs, the predicted Meta Δ, and that this closes DE-39's "only remaining lever." Note the registry ships as an audited all-1.0 (or sub-noise) artifact (B-11/DE-37 precedent).

- [ ] **Step 2: Append the outcome to `docs/claude/experiment-log.md`** (2026-05-30 B-14 entry: classification = mechanism-correctness no-op; the infra is a permanent, audited, bit-identical no-op).

- [ ] **Step 3: Commit, then go to "Ship".**

```bash
git add docs/claude/dead-ends.md docs/claude/experiment-log.md
git commit -m "docs(b14): DE-40 — hepatic UGT IVIVE differential too small (no-op ships)"
```

### Task 8b — GO / success path

- [ ] **Step 1: Append the B-14 success entry to `docs/claude/experiment-log.md`** with the Gate D1/D2/D3 results and per-drug shifts.

- [ ] **Step 2: Update `tests/integration/test_holdout_regression.py` docstring** to record the B-14 shift (and update the `2.698` pin assertion value ONLY if the regenerated Meta moved outside `±0.020` of `2.698`; if so reconcile against the new cache).

- [ ] **Step 3: Update `CLAUDE.md` top metrics table ONLY if a new 3-sig-fig headline is promoted**, reconciled against `data/training/4track_holdout_predictions.json` (per CLAUDE.md self-maintenance: never edit the metrics block from session context alone).

- [ ] **Step 4: Add a brief dead-ends.md note** if morphine remains materially over (pointing to the DE-38-complete UGT+CYP successor cycle).

- [ ] **Step 5: Commit.**

```bash
git add docs/claude/experiment-log.md tests/integration/test_holdout_regression.py CLAUDE.md docs/claude/dead-ends.md
git commit -m "docs(b14): UGT IVIVE differential outcome + metrics reconciliation"
```

### Ship (both paths)

- [ ] **Step 1: Run the fast suite.**

Run: `/opt/miniconda3/bin/python3 -m pytest -q -m "not slow"`
Expected: all pass (new tests included).

- [ ] **Step 2: Push + open PR** (CI-gated, mirroring B-13 PR #49). CI runs the leak-audit/ECM tests that skip locally.

```bash
git push -u origin b14-hepatic-ugt-ivive
gh pr create --base main --head b14-hepatic-ugt-ivive --title "B-14: hepatic UGT IVIVE differential (<DE-40 no-op | accuracy>)" --body "<summary: Phase 0 outcome, gates, DE-40-or-ship>"
```

- [ ] **Step 3: Watch CI; squash-merge after green** (`gh pr checks <n> --watch`, then `gh pr merge <n> --squash --delete-branch`).

---

## Notes for the executor

- **Anti-confabulation is the load-bearing discipline** (DE-39 was one day before this work). No SF enters the registry as `literature_applied` without a resolving PMID/DOI and the number visible in the source.
- **Do not tune to the holdout.** If a gate fails, the fix is re-verifying the *literature* (hepatic fraction, basis), never sliding the SF to hit a fold. A seed flipping under 1.0 (D2b) means the SF is too large → re-derive, don't shrink-to-target.
- **DE-40 is success, not failure.** The expected outcome is a no-op that definitively closes DE-39's named lever with evidence.
