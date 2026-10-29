# Packaging Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the import package from `utils` to `ds_utils`, make bare imports lightweight, pin dependencies, and produce a clean distributable wheel.

**Architecture:** Keep the PyPI distribution name `data-science-utils`, but change the installed import package to `ds_utils`. Convert `ds_utils/__init__.py` into a lazy re-export facade so importing the package does not load xgboost, statsmodels, matplotlib, or optional text/translation modules. Update every in-repo import and notebook reference; do not provide a permanent `utils` shim because it would preserve the collision this plan is removing.

**Tech Stack:** Python 3.14, uv 0.12+, hatchling, pytest.

**Spec:** `docs/superpowers/plans/2026-10-30-refactor-roadmap.md`, current `pyproject.toml`, `README.md`, and `utils/__init__.py`.

## Global Constraints

- Keep Python **3.14** exactly as requested.
- Keep the distribution name `data-science-utils`.
- No behavior changes to library functions; this is a rename/import-boundary/packaging pass.
- Do not keep a long-term top-level `utils` package; update all in-repo consumers instead.
- Bare `import ds_utils` must not import optional text/translation packages.
- All tests use the new package name; no `sys.path` hacks.
- Run `uv run pytest -q` before each commit.

## Review Focus

Likely failures:

- A notebook or test still imports `utils` and fails after the rename.
- Lazy `__getattr__` misses an alias such as `tune_single_model` or `compare_cv_metrics`.
- `from ds_utils import *` stops working.
- `import ds_utils` still pulls in matplotlib/xgboost/statsmodels.
- The wheel contains the old package or omits the new one.
- `pip install -r requirements.txt` drifts from `pyproject.toml`.
- Dependency floors are missing, so old scikit-learn installs fail on `TargetEncoder(cv=...)`.

---

### Task 1: Rename the package and update all imports

**Files:**
- Rename: `utils/` → `ds_utils/`
- Modify: every Python file importing `utils`
- Modify: `pyproject.toml` wheel packages
- Modify: `README.md`, docs, notebooks
- Create: `tests/test_package_name.py`

**Interfaces:**
- Produces: `import ds_utils`; `from ds_utils import ...`; all existing public names.

- [ ] **Step 1: Write the failing package-name test**

Create `tests/test_package_name.py`:

```python
def test_new_package_imports():
    import ds_utils
    assert callable(ds_utils.check_data_information)


def test_old_package_is_not_installed():
    with pytest.raises(ImportError):
        import utils  # noqa: F401
```

Run: `uv run pytest tests/test_package_name.py -q`

Expected: `ModuleNotFoundError: ds_utils` (the old package still resolves as `utils`).

- [ ] **Step 2: Rename the directory**

```bash
git mv utils ds_utils
```

Update `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["ds_utils"]
```

- [ ] **Step 3: Update Python imports and notebook code cells**

Replace `from utils` / `import utils` with `from ds_utils` / `import ds_utils` in:

- `tests/`
- `README.md`
- `docs/` (new plans may retain historical `utils` references only when describing the old state)
- `notebooks/*.ipynb` code cells
- any `data/` or other scripts

Use a notebook-safe replacement script that parses JSON and only edits code cells; do not blindly replace prose or output text.

- [ ] **Step 4: Sync and run the full suite**

```bash
uv sync
uv run pytest -q
```

Expected: all tests pass under `ds_utils`; `import utils` raises `ModuleNotFoundError`.

- [ ] **Step 5: Commit**

```bash
git add ds_utils pyproject.toml tests README.md docs notebooks
git commit -m "refactor: rename import package to ds_utils"
```

---

### Task 2: Add a lazy top-level facade

**Files:**
- Modify: `ds_utils/__init__.py`
- Create: `tests/test_lazy_imports.py`
- Modify: `tests/test_public_api.py`

**Interfaces:**
- Produces: bare `import ds_utils` without heavy dependencies; all `ds_utils.__all__` names resolve on first access.

- [ ] **Step 1: Write failing tests**

`tests/test_lazy_imports.py` must use a subprocess so earlier test imports do not pollute `sys.modules`:

```python
import subprocess, sys

def test_bare_import_is_lightweight():
    code = (
        "import sys, ds_utils; "
        "print(int('xgboost' in sys.modules), "
        "int('statsmodels' in sys.modules), "
        "int('matplotlib' in sys.modules))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "0 0 0"
```

Also assert:

- `from ds_utils import check_data_information` works after bare import.
- `from ds_utils import *` does not raise.
- Every name in `ds_utils.__all__` resolves with `getattr`.
- `compare_cv_metrics`, `tune_single_model`, and `plot_dynamic_stripplot_swarmplot` still exist.

Run: `uv run pytest tests/test_lazy_imports.py tests/test_public_api.py -q`

Expected: bare import loads xgboost/statsmodels/matplotlib (or the module fails while the facade is being written).

- [ ] **Step 2: Implement the lazy facade**

Replace eager imports with a mapping:

```python
_LAZY = {
    "check_data_information": ".preprocessing",
    # ... every public name
}

__all__ = list(_LAZY)

def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
```

Include all aliases and classes (`TextPreprocessor`, `TextVisualizer`) in the map. Add `__version__` from `importlib.metadata.version("data-science-utils")` with a fallback for source checkouts.

- [ ] **Step 3: Verify import cost**

```bash
uv run python -X importtime -c "import ds_utils" 2>&1 | Select-Object -Last 1
```

Expected: import time well below the previous ~4.2 seconds; no xgboost/statsmodels/matplotlib in the trace.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_lazy_imports.py tests/test_public_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ds_utils/__init__.py tests
git commit -m "refactor: lazily load the public package facade"
```

---

### Task 3: Pin dependencies and clean metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` via `uv lock`
- Modify: `requirements.txt`
- Modify: `README.md`

**Interfaces:**
- Produces: minimum versions aligned with the APIs actually used; complete project metadata; generated requirements file.

- [ ] **Step 1: Add exact dependency floors**

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

Keep the optional extras and add reasonable floors (`nltk>=3.9`, `beautifulsoup4>=4.12`, `Pillow>=10`, `deep-translator>=1.11`, `langid>=1.1.6`, `tenacity>=8.2`, etc.).

- [ ] **Step 2: Improve project metadata**

Add/update:

```toml
description = "Reusable utilities for rapid EDA, preprocessing, modeling, and reporting"
license = { text = "MIT" }
keywords = ["data science", "eda", "machine learning", "pandas", "scikit-learn"]
classifiers = [
    "Programming Language :: Python :: 3.14",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
```

Add `[project.urls]` only if the repository URL is known. Do not invent a URL.

- [ ] **Step 3: Regenerate requirements and lock**

```bash
uv lock
uv export --format requirements-txt --no-hashes --no-dev --output-file requirements.txt
```

Add a header comment to `requirements.txt` stating it is generated and `pyproject.toml` is authoritative.

- [ ] **Step 4: Verify**

```bash
uv sync
uv run pytest -q
```

Expected: PASS with the new floors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock requirements.txt README.md
git commit -m "chore: pin dependency floors and package metadata"
```

---

### Task 4: Build and inspect the wheel

**Files:**
- Create: `tests/test_wheel_contents.py`
- Maybe modify: `pyproject.toml` if hatch excludes files.

**Interfaces:**
- Produces: a wheel containing `ds_utils/` and not the old `utils/` package.

- [ ] **Step 1: Write the wheel test**

Use `subprocess` to run `uv build --wheel`, then inspect the newest wheel with `zipfile.ZipFile.namelist()`:

- at least one member starts with `ds_utils/`
- no member starts with `utils/`
- `ds_utils/__init__.py` is present

Skip the test if `uv` is unavailable, but keep it enabled in CI.

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_wheel_contents.py -q
```

Expected: PASS; inspect the printed wheel path if it fails.

- [ ] **Step 3: Commit**

```bash
git add tests/test_wheel_contents.py
git commit -m "test: verify wheel contains the renamed package"
```

---

### Task 5: Update migration documentation

**Files:**
- Modify: `README.md`
- Maybe create: `docs/migration-utils-to-ds-utils.md`

- [ ] **Step 1: Document the rename**

Add a short section:

```markdown
## Import name migration

The distribution is still `data-science-utils`, but the import package is now
`ds_utils`:

```python
from ds_utils import check_data_information
```

The old `utils` import name is no longer provided because it collides with
unrelated packages.
```

- [ ] **Step 2: Run the full suite**

```bash
uv run pytest -q
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs
git commit -m "docs: document the ds_utils import rename"
```

---

## Self-Review

- The rename is mechanical and covered by a package-name test.
- Lazy imports are verified in a subprocess, not polluted by test imports.
- Dependency floors match APIs used by the code (`TargetEncoder(cv=...)`, modern pandas/seaborn).
- The wheel test prevents shipping the old package by accident.
- No behavior changes are mixed into this plan.
