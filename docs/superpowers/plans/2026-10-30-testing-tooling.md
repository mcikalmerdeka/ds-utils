# Testing and Tooling Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add coverage reporting, linting, CI, optional-extra test markers, and final package-build checks so the refactored library is verifiable on every push.

**Architecture:** Keep tests black-box and behavior-focused. Use pytest fixtures for small messy dataframes, `Agg` for plots, `monkeypatch` for optional services, and subprocesses for import/packaging checks. CI runs on Python 3.14 and uses `uv` for deterministic installs.

**Tech Stack:** Python 3.14, pytest, pytest-cov, ruff, uv, GitHub Actions.

**Spec:** `docs/superpowers/plans/2026-10-30-refactor-roadmap.md`, and the packaged `utils` layout.

> This plan assumes the packaging hygiene plan has run; the import package remains `utils`.

## Global Constraints

- Keep Python 3.14.
- Default tests must not require NLTK, langid, deep-translator, wordcloud, or textblob.
- No network calls in tests; monkeypatch translators and language detectors.
- No interactive matplotlib windows; use `Agg`.
- CI must pass on a clean checkout with `uv sync`.
- Coverage must not be inflated with trivial assertion-free tests.
- Run `uv run pytest -q` before each commit.

## Review Focus

Likely failures:

- Optional-extra tests failing on machines without extras.
- Coverage thresholds set above the achievable baseline.
- Ruff auto-fixes changing behavior or creating a huge unrelated diff.
- CI not installing the dev dependency group.
- Wheel install succeeding in the repo but failing in a clean environment.
- Tests depending on the developer’s global Python or existing `.venv`.

---

### Task 1: Add test/lint tooling and coverage configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `tests/test_tooling_config.py`

**Interfaces:**
- Produces: `pytest-cov`, `ruff`, coverage config, ruff config, and a test that the config is present.

- [ ] **Step 1: Add dev dependencies**

```bash
uv add --dev pytest-cov ruff
```

- [ ] **Step 2: Add configuration**

In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-ra"

[tool.coverage.run]
source = ["utils"]
branch = true
omit = ["*/__init__.py"]

[tool.coverage.report]
show_missing = true
skip_covered = true

[tool.ruff]
target-version = "py314"
line-length = 100
extend-exclude = ["notebooks", "data"]

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]
```

- [ ] **Step 3: Measure the baseline before enforcing coverage**

```bash
uv run pytest --cov=utils --cov-report=term-missing -q
```

Record the baseline in `.superpowers/sdd/2026-10-30-testing-tooling/progress.md`. Do not set `fail_under` yet; Task 4 sets it to 70 after the coverage tests are added. Never lower the threshold to hide missing coverage.

- [ ] **Step 4: Write a config smoke test**

`tests/test_tooling_config.py` must assert:

- `pytest_cov` is importable.
- `shutil.which("ruff")` is not `None`.
- `pyproject.toml` contains `[tool.coverage.run]` and `[tool.ruff]`.
- `utils` import works.

- [ ] **Step 5: Run**

```bash
uv run pytest tests/test_tooling_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/test_tooling_config.py
git commit -m "test: add coverage and lint tooling"
```

---

### Task 2: Add text and translation tests with optional-dependency guards

**Files:**
- Create: `tests/test_text_preprocessing.py`
- Create: `tests/test_translation_pipeline.py`
- Modify: `tests/conftest.py` if shared fixtures are needed.

**Interfaces:**
- Consumes: `TextPreprocessor`, `quick_clean`, `extract_ngrams`, `get_text_stats`, `translate_series`, `translate_dataframe`.
- Produces: behavior tests that skip cleanly when optional packages are absent.

- [ ] **Step 1: Write tests**

Cover:

- `TextPreprocessor.remove_html`, `remove_urls`, `remove_emails`, `remove_accents`, `remove_special_chars`.
- `quick_clean` on a string and a Series.
- `extract_ngrams` for bigrams/trigrams and too-short input.
- `get_text_stats` for normal, empty, and missing-column inputs.
- `translate_series` with `deep_translator` monkeypatched: unique values are translated once, nulls preserved, failed batches keep originals.
- `translate_dataframe` with `inplace=False` does not mutate the source.
- Optional-extra tests use `pytest.importorskip` or monkeypatched availability flags.

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_text_preprocessing.py tests/test_translation_pipeline.py -q
```

Expected: PASS or clean skips when extras are absent; no network calls.

- [ ] **Step 3: Commit**

```bash
git add tests/test_text_preprocessing.py tests/test_translation_pipeline.py tests/conftest.py
git commit -m "test: cover text and translation utilities"
```

---

### Task 3: Add time-series, feature-selection, and EDA coverage gaps

**Files:**
- Modify or create: `tests/test_time_series.py`
- Modify or create: `tests/test_feature_selection.py`
- Modify: `tests/test_eda_report.py`
- Modify: `tests/test_statistics.py`

**Interfaces:**
- Consumes: public functions from P1.
- Produces: regression coverage for the remaining analysis modules.

- [ ] **Step 1: Write tests**

Time series:

- `split_time_series` float/int boundaries and invalid sizes.
- `time_series_cv_splits` expanding windows and invalid `n_splits`.
- `create_lag_features`, `create_rolling_features`, `create_date_features` column names/values.
- `check_stationarity` returns ADF and KPSS rows.
- `interpolate_series` and `resample_series` behavior.

Feature selection:

- Correlation `feature_target` and `both` return shapes.
- Categorical relationship effect-size interpretation.
- Feature importance classification and regression methods.
- VIF low/moderate/high interpretation.

EDA/statistics:

- All-NaT date summary.
- Empty categorical and all-null numeric frames.
- HTML title escaping and missing-value plot omission when no missing values.

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_time_series.py tests/test_feature_selection.py tests/test_eda_report.py tests/test_statistics.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests
git commit -m "test: cover time series, feature selection, and EDA gaps"
```

---

### Task 4: Add characterization tests for evaluation and tuning

**Files:**
- Modify: `tests/test_eval_classification.py`
- Modify: `tests/test_eval_regression.py`
- Create: `tests/test_tuning.py`

**Interfaces:**
- Consumes: `eval_classification`, `eval_regression`, `tune_single_model`, `tune_pipelines`, `compare_cv_metrics`.
- Produces: known-score assertions instead of key-only smoke tests.

- [ ] **Step 1: Write tests**

- Train a deterministic classifier/regressor on a small synthetic dataset and assert metric values are within a tolerance of a direct sklearn computation.
- Assert `compare_cv_metrics(as_numeric=True)` returns floats; default remains formatted strings.
- Tune one model with a one-value grid and `n_splits=2`, `n_repeats=1`, `plot_learning_curves=False`; assert `best_params_` and timing return shape.
- Assert tuning with `display=False` does not print but still returns the fitted search.
- Assert `eval_classification` rejects multiclass and no-`predict_proba` models (already present) and that the error is raised before any CV fit.

- [ ] **Step 2: Run and measure coverage**

```bash
uv run pytest tests/test_eval_classification.py tests/test_eval_regression.py tests/test_tuning.py -q
uv run pytest --cov=utils --cov-report=term-missing -q
```

Expected: PASS. Add `fail_under = 70` to `[tool.coverage.report]` only if the measured coverage is at least 70%; otherwise add focused tests until it reaches 70%. Do not lower the threshold to hide gaps.

- [ ] **Step 3: Commit**

```bash
git add tests
git commit -m "test: characterize evaluation and tuning behavior"
```

---

### Task 5: Add CI and pre-commit tooling

**Files:**
- Create: `.github/workflows/tests.yml`
- Create: `.pre-commit-config.yaml`
- Modify: `README.md`

**Interfaces:**
- Produces: CI on push/pull request and an optional local pre-commit hook.

- [ ] **Step 1: Add the GitHub Actions workflow**

Use `astral-sh/setup-uv@v6` (or the current stable action) with Python 3.14, then:

```yaml
- run: uv sync --group dev
- run: uv run ruff check .
- run: uv run pytest --cov=utils --cov-report=term-missing
- run: uv build
```

Run on push and pull request to the main branch. Do not install optional extras in the default job.

- [ ] **Step 2: Add pre-commit config**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.0
    hooks:
      - id: ruff
```

Pin the revision to the installed ruff release; verify with `uv run ruff --version`. Keep formatting as a separate optional cleanup so this plan does not create a repository-wide style diff.

- [ ] **Step 3: Document the commands**

Add to README:

```bash
uv run pytest --cov=utils --cov-report=term-missing
uv run ruff check .
uv build
```

- [ ] **Step 4: Verify locally**

```bash
uv run ruff check .
uv run pytest --cov=utils --cov-report=term-missing -q
uv build
```

Expected: lint clean, coverage threshold met, wheel built.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/tests.yml .pre-commit-config.yaml README.md
git commit -m "ci: add test, lint, coverage, and build checks"
```

---

### Task 6: Final clean-environment package check

**Files:**
- Create: `tests/test_wheel_install.py`
- Modify: `pyproject.toml` only if package data/exclusions are wrong.

**Interfaces:**
- Produces: proof that a built wheel installs and imports outside the repository.

- [ ] **Step 1: Write the test**

In a subprocess:

1. Build the wheel with `uv build --wheel`.
2. Create a temporary virtual environment with `venv`.
3. Install the wheel with `pip`.
4. Run `python -c "import utils; print(utils.__version__)"`.
5. Assert the import does not require optional extras.

Skip the test if the platform cannot create a venv; keep it enabled in CI.

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_wheel_install.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_wheel_install.py
git commit -m "test: verify wheel install in a clean environment"
```

---

## Self-Review

- Optional packages are never required by default tests.
- Coverage configuration starts from a measured baseline and is not lowered silently.
- CI runs lint, tests with coverage, and a wheel build.
- Notebooks are excluded from ruff; their code is still exercised indirectly through package imports.
- Packaging is validated both by wheel contents and a clean install.
