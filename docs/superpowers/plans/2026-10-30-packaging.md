# Packaging Hygiene Plan (Optional)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the convenient `utils` import name, avoid import-system churn, and make the local environment reproducible by pinning dependency floors and verifying the built package.

**Architecture:** No package rename and no lazy-import facade. The repository stays a flat, eagerly imported `utils` package. This plan only touches dependency metadata, optional extras, requirements generation, and package-build verification.

**Tech Stack:** Python 3.14, uv, hatchling, pytest.

**Spec:** `docs/superpowers/plans/2026-10-30-refactor-roadmap.md`, `pyproject.toml`, and the current `utils/` layout.

## Why this plan is deliberately small

For this repo’s actual use case — importing functions in daily notebooks and local scripts — the following are **not worth the churn**:

- Renaming `utils` to `ds_utils`: it would touch every notebook, test, and README snippet for no daily benefit. The generic name is only a problem if this package is installed globally alongside another unrelated `utils` package, which is not the current workflow.
- Lazy imports: `import utils` takes a few seconds once per notebook session. A lazy `__getattr__` facade adds maintenance cost, weakens IDE discoverability, and can surprise users of `from utils import *`. Optimize later only if startup time becomes a measured problem.
- Publishing metadata polish: keywords, classifiers, and license metadata only matter if the package is published or shared. Keep the description decent, but do not invent a license choice.

What is still worth doing:

- Pin minimum versions for APIs the code actually relies on.
- Keep optional extras complete and consistent.
- Avoid `requirements.txt` drifting from `pyproject.toml`.
- Verify that `uv sync` and `uv build` still work.

## Global Constraints

- Keep Python **3.14**.
- Keep the import package name **`utils`** and the distribution name **`data-science-utils`**.
- Do not add a lazy `__getattr__` facade.
- Do not rewrite notebooks or rename imports.
- Do not change library behavior.
- Do not add a license unless the repository owner explicitly chooses one.
- Run `uv run pytest -q` before each commit.

## Review Focus

Likely failures:

- An unpinned environment resolves an old scikit-learn where `TargetEncoder(cv=...)` fails.
- A newer pandas drops an API used by preprocessing.
- `requirements.txt` installs a different set than `uv sync`.
- The wheel omits `utils/` or includes stale files.
- Optional extra names drift from the imports in text/translation modules.

---

### Task 1: Pin dependency floors

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` via `uv lock`
- Create: `tests/test_packaging_metadata.py`

**Interfaces:**
- Consumes: current installed versions and APIs used by `utils/`.
- Produces: explicit lower bounds for core and optional dependencies.

- [ ] **Step 1: Write the metadata test**

Create `tests/test_packaging_metadata.py` using `tomllib`:

```python
def test_core_dependency_floors():
    dependencies = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
    expected = {
        "scikit-learn>=1.6",
        "pandas>=2.2",
        "seaborn>=0.13",
        "statsmodels>=0.14",
        "numpy>=1.26",
        "scipy>=1.11",
    }
    assert expected <= set(dependencies)
```

Add assertions that `requires-python` is still `>=3.14` and the project name is still `data-science-utils`.

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/test_packaging_metadata.py -q`

Expected: current dependencies have no floors, so the test fails.

- [ ] **Step 3: Add floors**

Use at least:

```toml
dependencies = [
    "category-encoders>=2.6",
    "ipykernel",
    "matplotlib>=3.8",
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.6",
    "scipy>=1.11",
    "seaborn>=0.13",
    "statsmodels>=0.14",
    "tqdm>=4.66",
    "xgboost>=2.0",
]
```

Add reasonable floors to the existing optional extras:

```toml
text = [
    "beautifulsoup4>=4.12",
    "contractions>=0.1",
    "nltk>=3.9",
    "Pillow>=10",
    "textblob>=0.18",
    "wordcloud>=1.9",
]
translate = [
    "deep-translator>=1.11",
    "langid>=1.1.6",
    "tenacity>=8.2",
]
```

Do not add upper bounds.

- [ ] **Step 4: Lock and test**

```bash
uv lock
uv sync
uv run pytest -q
```

Expected: the metadata test passes and the full suite remains green.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/test_packaging_metadata.py
git commit -m "chore: pin dependency floors"
```

---

### Task 2: Make requirements.txt a generated artifact

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`
- Modify: `tests/test_packaging_metadata.py`

**Interfaces:**
- Produces: one authoritative dependency source (`pyproject.toml`) and a generated pip-compatible file.

- [ ] **Step 1: Write the failing test**

Assert the first line of `requirements.txt` contains `generated` and `pyproject.toml` and that `requirements.txt` includes `scikit-learn>=1.6`.

Run: `uv run pytest tests/test_packaging_metadata.py -q`

Expected: current requirements file is hand-written and has no generated marker/floors.

- [ ] **Step 2: Regenerate**

```bash
uv export --format requirements-txt --no-hashes --no-dev --output-file requirements.txt
```

Add a header if `uv export` does not:

```text
# Generated from pyproject.toml by uv. Do not edit manually.
```

- [ ] **Step 3: Update README**

Clarify that `uv sync` is the primary workflow and `requirements.txt` is generated for pip users.

- [ ] **Step 4: Verify**

```bash
uv run pytest tests/test_packaging_metadata.py -q
pip install --dry-run -r requirements.txt
```

If `pip` is unavailable in the active environment, use `uv pip install --dry-run -r requirements.txt`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt README.md tests/test_packaging_metadata.py
git commit -m "chore: generate requirements from pyproject"
```

---

### Task 3: Clean up project metadata without inventing details

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Produces: a non-placeholder description and consistent project naming; no unverifiable license/URL metadata.

- [ ] **Step 1: Replace the placeholder description**

Use:

```toml
description = "Reusable utilities for rapid EDA, preprocessing, modeling, and reporting"
```

Do not add authors, URLs, keywords, or classifiers unless they are already known. Do not add a license without an explicit decision from the repository owner.

- [ ] **Step 2: Check README consistency**

Ensure the README says the distribution is `data-science-utils` and the import package is `utils`. Remove any stale rename language if it exists.

- [ ] **Step 3: Verify**

```bash
uv run pytest -q
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml README.md
git commit -m "chore: clean up package description"
```

---

### Task 4: Verify the wheel contains the flat utils package

**Files:**
- Create: `tests/test_wheel_contents.py`

**Interfaces:**
- Produces: proof that `uv build` packages `utils/` and the public modules.

- [ ] **Step 1: Write the test**

Use `subprocess` to run `uv build --wheel`, find the newest wheel, and inspect it with `zipfile.ZipFile.namelist()`:

- at least one member starts with `utils/`
- `utils/__init__.py` is present
- `utils/preprocessing.py` and `utils/classification_evals_and_tuning.py` are present
- no `tests/` or `notebooks/` members are included

Skip the test if `uv` is not available, but keep it enabled in CI.

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_wheel_contents.py -q
```

Expected: PASS.

- [ ] **Step 3: Optional clean import check**

If a throwaway environment is easy to create:

```bash
uv venv .tmp-wheel-check
uv pip install --python .tmp-wheel-check/Scripts/python.exe dist/*.whl
.tmp-wheel-check/Scripts/python.exe -c "import utils; print(utils.__all__[:3])"
```

Remove `.tmp-wheel-check` afterward and ensure it is git-ignored.

- [ ] **Step 4: Commit**

```bash
git add tests/test_wheel_contents.py
git commit -m "test: verify wheel contents"
```

---

## Self-Review

- No rename, lazy-import facade, notebook rewrite, or license decision.
- The plan focuses on the actual failure modes: unpinned APIs, requirements drift, and broken builds.
- Every task is independently useful; the whole plan can be skipped if the environment remains `uv`-managed and reproducible.
- The flat `utils` import workflow and notebook ergonomics are preserved.
