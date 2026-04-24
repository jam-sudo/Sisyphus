# Reproducibility — Install & Lockfile

Sisyphus pins its full dependency graph via `requirements-lock.txt` for reproducible installs. CI (`.github/workflows/ci.yml`) installs from this lockfile so local and CI environments match.

## Quick install (use the lockfile)

```bash
pip install -r requirements-lock.txt
pip install -e .
```

This pins every transitive dep to the exact version tested.

## Fresh install from source (unpinned)

```bash
pip install -e '.[ml,chem,dev]'
```

Transitive versions float within `pyproject.toml` constraints. Useful for local experimentation; not recommended for reproducing benchmark numbers.

### Optional: SBI (Simulation-Based Inference)

SBI training + TDM dispatching requires torch + sbi + nflows. These are NOT in `requirements-lock.txt` (would add ~1 GB to CI install). Install the `[sbi]` extra separately if you need to run SBI features:

```bash
pip install -e '.[sbi]'
```

Without the `[sbi]` extra, `pytest.importorskip("torch")` cleanly skips SBI test modules. Non-SBI code paths (engine, ML, pipeline, TDM non-SBI methods) work without torch.

## Regenerating `requirements-lock.txt`

Run from repository root:

```bash
python3 -m venv /tmp/sis_lock_env
source /tmp/sis_lock_env/bin/activate
pip install --upgrade pip
pip install -e '.[ml,chem,dev]'
pip freeze --exclude-editable > requirements-lock.txt
deactivate
rm -rf /tmp/sis_lock_env
```

**Always regenerate from a fresh venv**, not from your daily dev env. A typical dev env carries unrelated packages (chemprop, descriptastorus, mordred, `rdkit-pypi` etc.) that will pollute the lockfile and bloat CI install time.

Regenerate when:
- `pyproject.toml` dependencies change
- An upstream CVE requires a version bump
- A transitive version drift causes reproducibility loss

## RDKit

Project uses the PyPI-maintained `rdkit` package (2023.9+). Do NOT use the older community fork `rdkit-pypi` — it is deprecated and incompatible with newer numpy.

If `pip install rdkit` fails on a non-standard platform:
- Try conda: `conda install -c conda-forge rdkit`
- Document the platform-specific workaround in this file
- CI runs Ubuntu latest and resolves `rdkit` from PyPI without issue

## CI install path

`.github/workflows/ci.yml` mirrors the Quick Install above. It is not a secret third path.

## Numpy 2.x

Lockfile pins `numpy==2.2.6`. Sisyphus is compatible with numpy 2.x (no deprecated `np.bool`/`np.int`/`np.float` usage). If future code needs numpy 1.x behavior, pin `numpy<2` in `pyproject.toml` and regenerate the lockfile.

## Platform markers

`requirements-lock.txt` is generated on Linux. Some entries are Linux-only:

- `nvidia-nccl-cu12` (xgboost GPU dep): marked `; platform_system == "Linux"`. Skipped on macOS/Windows installs.

If you regenerate the lockfile on a different platform, verify markers are preserved by the pip version you use. Plain `pip freeze` drops markers; if you need cross-platform guarantees use `pip-compile` (pip-tools) or `uv pip compile` instead.
