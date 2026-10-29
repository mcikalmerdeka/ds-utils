# P0 Correctness and Test Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real pytest harness and fix the verified P0 correctness/API-surface defects in `utils/` without changing the package layout or Python version.

**Architecture:** Keep the existing public modules and names. Add focused tests first, then minimal fixes. Extract only small private helpers needed to remove duplicated buggy logic. Preserve compatibility aliases for documented names.

**Tech Stack:** Python 3.14, pandas 2.3.3, scikit-learn 1.7.2, matplotlib, pytest, uv.

**Spec:** `README.md`, `pyproject.toml`, and the review findings recorded in this session.

## Global Constraints

- Keep `requires-python = ">=3.14"` and `.python-version = 3.14` unchanged.
- Do not rename the `utils` import package in this pass.
- Do not remove or rename existing public functions; add aliases where the README uses unsuffixed names.
- Matplotlib tests must use the non-interactive `Agg` backend.
- Tests must not require optional text/translation dependencies; use skips or monkeypatching.
- Existing user changes in the worktree must be preserved.
- Every production fix must have a test that failed before the fix.

## Review Focus

These are the inputs most likely to break a real user and are explicitly tested in this plan:

- One-column plot grids and empty column lists.
- `float32` numeric columns and object-typed date strings.
- Unlisted object columns passed through `feature_encoding`.
- Missing values and zero-valued outliers in `filter_outliers`.
- `target_langs=None` and short non-English ASCII text.
- Multiclass classification passed to `eval_classification`.
- Documented names imported directly from `utils`.
- Categorical median imputation, negative log scaling, and empty DataFrames.
- URL removal on decimals, money, IP addresses, and emails.
- EDA reports with a single numeric feature.

---

### Task 1: Test harness and characterization tests

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`
- Modify: `tests/test_eval_classification.py`
- Modify: `tests/test_eval_regression.py`
- Modify: `tests/test_ab_testing_manual.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: existing `utils` modules.
- Produces: `pytest` runnable from the repository root; `Agg` backend; small assertion-based smoke tests.

- [ ] **Step 1: Add pytest to the project**

Run:

```bash
uv add --dev pytest
```

Expected: `pyproject.toml` and `uv.lock` update with a dev dependency group; `.python-version` remains `3.14`.

- [ ] **Step 2: Add pytest configuration**

In `pyproject.toml`, add:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-ra"
```

- [ ] **Step 3: Add test fixtures and Agg backend**

Create `tests/conftest.py`:

```python
import matplotlib
matplotlib.use("Agg")
```

Add a fixture that returns a small mixed dataframe with numeric, categorical, missing, and text columns.

- [ ] **Step 4: Replace the three smoke scripts with assertions**

`tests/test_eval_classification.py` must build a tiny binary dataset, call `eval_classification` with every plot flag disabled, `n_splits=2`, `n_repeats=1`, and assert:

- `"cv"` and `"accuracy"` are in the result
- `0.0 <= result["accuracy"]["test"] <= 1.0`
- `{"accuracy", "precision", "recall", "f1", "roc_auc"} == set(result["cv"])`

`tests/test_eval_regression.py` must do the same with `eval_regression`, all plot flags disabled, and assert `mae`, `mse`, `rmse`, `r2` are present and non-negative (except R² may be negative).

`tests/test_ab_testing_manual.py` must assert:

- `calculate_sample_size(0.10, 0.02) > 0`
- `perform_t_test(group_a, group_b)["mean_b"] > perform_t_test(group_a, group_b)["mean_a"]`
- `perform_chi_squared_test(100, 1000, 130, 1000)["significant"] is True`
- `analyze_ab_test_results` returns `sample_size_a`, `sample_size_b`, and `p_value`

- [ ] **Step 5: Add smoke import test**

Create `tests/test_smoke.py` asserting `import utils` succeeds and `callable(utils.check_data_information)`.

- [ ] **Step 6: Run the suite**

Run: `uv run pytest -q`

Expected: all tests pass. If any fail, fix the test or the existing behavior before moving on.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tests
git commit -m "test: add pytest harness and baseline utility tests"
```

---

### Task 2: Fix single-column visualization grids

**Files:**
- Modify: `utils/visualization.py`
- Create: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `plot_dynamic_hisplots_kdeplots`, `plot_dynamic_boxplots_violinplots`, `plot_dynamic_countplot`, `plot_dynamic_stripplot_swarmplot`.
- Produces: a private `_flatten_axes(ax)` helper; every grid function accepts one column with the default `ncols`/`n_cols`.

- [ ] **Step 1: Write failing tests**

Parameterize the four plot functions with one column and assert they do not raise. Also assert an empty `col_series` raises `ValueError`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_visualization.py -q`

Expected: the single-column cases fail with `AttributeError: 'numpy.ndarray' object has no attribute 'xaxis'`.

- [ ] **Step 3: Implement the shared axis helper**

In `utils/visualization.py` add:

```python
def _flatten_axes(ax):
    if isinstance(ax, np.ndarray):
        return ax.ravel()
    return np.array([ax])
```

Use it in all four grid functions instead of the current `np.array([ax])` / `ax.flatten()` branch. Add a non-empty `col_series` validation at the start of each function.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_visualization.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/visualization.py tests/test_visualization.py
git commit -m "fix: support single-column plot grids"
```

---

### Task 3: Make numeric, date, categorical, and EDA summaries dtype-safe

**Files:**
- Modify: `utils/statistics.py`
- Modify: `utils/eda_report.py`
- Create: `tests/test_statistics.py`
- Create: `tests/test_eda_report.py`

**Interfaces:**
- Consumes: `describe_numerical_combined`, `describe_categorical_combined`, `describe_date_columns`, `generate_eda_report`.
- Produces: numeric summaries for any numeric dtype; date parsing for object strings; safe categorical statistics for empty hue groups; EDA report support for `float32` and one numeric feature.

- [ ] **Step 1: Write failing tests**

Tests must cover:

- `describe_numerical_combined` on a `float32` frame returns one row with finite skewness/kurtosis.
- `describe_date_columns` on `["2024-01-01", "2024-02-01"]` returns a date range in days.
- `describe_categorical_combined` with a hue group that has no non-null category returns a DataFrame instead of raising.
- `generate_eda_report` on two `float32` columns returns HTML containing `"Numerical Summary"`.
- `generate_eda_report` with one numeric feature and a numeric target includes the feature-target correlation table, not the “No target column provided” message.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_statistics.py tests/test_eda_report.py -q`

Expected: failures for the `float32` length mismatch, the object-date `KeyError: 'max'`, the empty hue `idxmin`, and the EDA report dtype failure.

- [ ] **Step 3: Fix numeric summaries**

Compute skewness and kurtosis per requested column directly:

```python
values = pd.to_numeric(data[col], errors="coerce").dropna()
skew(values) if len(values) > 1 else np.nan
kurtosis(values) if len(values) > 1 else np.nan
```

Apply the same helper to hue/class summaries.

- [ ] **Step 4: Fix date and categorical summaries**

In `describe_date_columns`, parse each requested column with `pd.to_datetime` before `describe()`; raise `ValueError` naming the column if parsing fails. In `describe_categorical_combined`, use a helper that handles an empty `value_counts()` and skip NaN hue classes.

- [ ] **Step 5: Fix EDA target correlation and float dtype support**

Move the feature-target correlation block outside the `len(numeric_features) >= 2` condition, while keeping the heatmap conditional on at least two numeric features. The float32 failure should disappear once the statistics helper is fixed.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_statistics.py tests/test_eda_report.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add utils/statistics.py utils/eda_report.py tests/test_statistics.py tests/test_eda_report.py
git commit -m "fix: make summaries and EDA reports dtype-safe"
```

---

### Task 4: Protect feature encoding from silent data loss

**Files:**
- Modify: `utils/preprocessing.py`
- Create: `tests/test_preprocessing_encoding.py`

**Interfaces:**
- Consumes: `feature_encoding`.
- Produces: unlisted object columns preserved; unknown binary categories raise by default; `handle_unknown="ignore"` explicitly maps unknown binaries to 0; unsupported one-hot `use_encoded_value` raises a clear error.

- [ ] **Step 1: Write failing tests**

Tests must cover:

- A dataframe with `cat` (nominal) and `free_text` (unlisted object); output `free_text` equals the input values.
- Train on `["yes", "no"]`, transform `["maybe"]` with default settings; expect `ValueError` mentioning the unknown value.
- The same transform with `handle_unknown="ignore"` maps the unknown value to 0.
- `nominal_columns=["cat"]` with `handle_unknown="use_encoded_value"` raises `ValueError` explaining that one-hot does not support that policy.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_preprocessing_encoding.py -q`

Expected: the free-text column becomes `NaN`; unknown binary silently becomes 0; the one-hot policy either fails later or behaves inconsistently.

- [ ] **Step 3: Preserve passthrough object columns**

Only run `pd.to_numeric` on columns produced by ordinal/one-hot encoding. Do not convert `passthrough_cols` or unrelated object columns.

- [ ] **Step 4: Validate unknown-category behavior**

At transform time, detect values outside the fitted binary mapping. If `handle_unknown == "error"`, raise `ValueError` listing the values. If `"ignore"`, map them to 0 and emit a warning. At fit time, reject `handle_unknown="use_encoded_value"` when `nominal_columns` is non-empty; the one-hot encoder cannot represent that policy.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_preprocessing_encoding.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/preprocessing.py tests/test_preprocessing_encoding.py
git commit -m "fix: preserve encoding passthrough columns and validate unknowns"
```

---

### Task 5: Fix outlier, missing-value, scaling, and empty-frame guards

**Files:**
- Modify: `utils/preprocessing.py`
- Create: `tests/test_preprocessing_guards.py`

**Interfaces:**
- Consumes: `filter_outliers`, `handle_missing_values`, `feature_scaling`, `check_data_information`.
- Produces: NaN rows are not outliers; detection reports a truthful `Outlier Exist`; categorical median/mean has an actionable error; `ffill`/`bfill` use modern pandas; negative log transforms are rejected; empty dataframes return an empty info table.

- [ ] **Step 1: Write failing tests**

Tests must cover:

- `filter_outliers` with `[1, 2, 3, NaN]`, z-score, filtering mode: all four rows remain.
- The same frame in detection mode: 0% outliers and `Outlier Exist is False`.
- `[10, 10, 10, 10, 0]` in detection mode: 20% outliers and `Outlier Exist is True`.
- `handle_missing_values(..., strategy="median")` on a string column raises `ValueError` mentioning `mode`.
- `handle_missing_values(..., strategy="ffill")` fills the missing value without a `FutureWarning`.
- `feature_scaling(..., apply_log=True)` with a negative value raises `ValueError`.
- `check_data_information` on an empty frame returns a DataFrame with the documented columns.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_preprocessing_guards.py -q`

Expected: NaN rows are filtered as outliers, `Outlier Exist` is false for the zero outlier, categorical median raises `TypeError`, ffill emits a deprecation warning, negative log raises a sklearn error, and empty info raises `ZeroDivisionError`.

- [ ] **Step 3: Fix outlier masks**

Treat missing values as non-outliers:

```python
filter_outlier = series.isna() | bounds_check
```

For z-scores, compute z only on non-null values. Change the detection flag to:

```python
bool(((~filter_outlier) & series.notna()).any())
```

- [ ] **Step 4: Fix missing values and scaling**

Validate strategies up front. For `median`/`mean`, raise `ValueError` for non-numeric selected columns. Replace `fillna(method=strategy)` with `.ffill()`/`.bfill()`. Add a shared `_apply_log1p` helper that raises `ValueError` for negative values and use it in both scaling modes. Add the empty-frame return to `check_data_information`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_preprocessing_guards.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/preprocessing.py tests/test_preprocessing_guards.py
git commit -m "fix: guard outlier filtering, imputation, scaling, and empty frames"
```

---

### Task 6: Fix text URL removal and language detection

**Files:**
- Modify: `utils/text_preprocessing.py`
- Modify: `utils/translation_utils.py`
- Create: `tests/test_text_language.py`

**Interfaces:**
- Consumes: `TextPreprocessor.remove_urls`, `process_text`, `_is_target_language`, `detect_language_rows`.
- Produces: URL removal that preserves decimals/money/IPs; email removal that does not leave half an address; `target_langs=None` detecting non-English; short non-English ASCII text detected when a target language is requested.

- [ ] **Step 1: Write failing tests**

Tests must cover:

- `remove_urls("Version 3.14 costs $2.50")` preserves both numbers.
- `remove_urls("Server 192.168.1.1 is down")` preserves the IP.
- `remove_urls("Visit https://example.com and www.example.org")` removes both URLs.
- `process_text("Contact test@example.com", options={"remove_stopwords": False, "lemmatize": False})` does not leave `test@`.
- `_is_target_language("Hola amigo", {"es"}, dummy)` returns `True`.
- `_is_target_language("Hello world", set(), dummy_en)` returns `False`.
- `detect_language_rows` with `target_langs=None` and a monkeypatched identifier splits Spanish from English.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_text_language.py -q`

Expected: decimals/money/IPs are removed, `test@` remains, short Spanish is treated as English, and `target_langs=None` detects nothing.

- [ ] **Step 3: Fix URL removal**

Replace the current overly broad regex with a scheme/`www.` alternative plus a domain alternative requiring a letter-based TLD and not preceded by `@` or a word character. Keep real URLs working. Remove emails before URLs in `process_text` so an email is never partially consumed.

- [ ] **Step 4: Fix language detection**

Remove the “fewer than five ASCII words means English” shortcut. In `_is_target_language`, use the English fast path only when the caller has not supplied explicit target languages. For an empty target set, return `lang != "en"`; for a non-empty set, return `lang in target_langs`. Validate confidence for both branches.

- [ ] **Step 5: Fix the empty-text split**

When `total_unique == 0`, return `df.iloc[0:0]` as the target frame and the original `df` as the other frame.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_text_language.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add utils/text_preprocessing.py utils/translation_utils.py tests/test_text_language.py
git commit -m "fix: correct URL cleaning and non-English detection"
```

---

### Task 7: Export documented names and guard multiclass evaluation

**Files:**
- Modify: `utils/__init__.py`
- Modify: `utils/classification_evals_and_tuning.py`
- Create: `tests/test_public_api.py`
- Create: `tests/test_classification_guard.py`

**Interfaces:**
- Consumes: `utils` package exports; `eval_classification`.
- Produces: unsuffixed regression tuning aliases and strip/swarm plot export; binary-only guard and `predict_proba` guard with actionable messages.

- [ ] **Step 1: Write failing tests**

`tests/test_public_api.py` must assert these are importable from `utils`:

`tune_pipelines`, `tune_single_model`, `tune_all_models`, `compare_cv_metrics`, `get_model_pipeline`, `get_hyperparameters`, `plot_dynamic_stripplot_swarmplot`.

Also assert every name in `utils.__all__` resolves with `getattr`.

`tests/test_classification_guard.py` must build a three-class dataset and assert `eval_classification` raises `ValueError` matching `"binary classification"` before fitting the CV. It must also assert a model without `predict_proba` raises a clear `ValueError`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_public_api.py tests/test_classification_guard.py -q`

Expected: the imports fail; multiclass raises the sklearn `average='binary'` error; a no-proba model raises `AttributeError`.

- [ ] **Step 3: Add exports and aliases**

Import `plot_dynamic_stripplot_swarmplot` and add the unsuffixed regression aliases:

```python
compare_cv_metrics = compare_cv_metrics_regression
tune_pipelines = tune_pipelines_regression
tune_single_model = tune_single_model_regression
tune_all_models = tune_all_models_regression
get_model_pipeline = get_model_pipeline_regression
get_hyperparameters = get_hyperparameters_regression
```

Add all aliases to `__all__`. Keep the classification-specific suffixed names unchanged.

- [ ] **Step 4: Add the classification guards**

After converting `y_train`/`y_test` to arrays, reject more than two classes with a message naming `eval_classification` and `binary classification`. Immediately check `hasattr(model, "predict_proba")` and raise a clear `ValueError` if absent. Do this before the first `predict_proba` call.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_public_api.py tests/test_classification_guard.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/__init__.py utils/classification_evals_and_tuning.py tests/test_public_api.py tests/test_classification_guard.py
git commit -m "fix: align public exports and guard classification evaluation"
```

---

### Task 8: Full-suite verification and documentation note

**Files:**
- Modify: `README.md` (only if the API aliases need clarification)
- Modify: `docs/superpowers/plans/2026-10-29-p0-correctness.md` (checklist only)

**Interfaces:**
- Consumes: all prior tasks.
- Produces: green full suite and a reviewable branch.

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`

Expected: PASS, no warnings promoted to errors.

- [ ] **Step 2: Run a clean import and a minimal end-to-end check**

Run:

```bash
uv run python -c "import utils; print(len(utils.__all__)); print(utils.tune_single_model.__name__)"
uv run python -c "from utils import generate_eda_report; import pandas as pd; print('EDA ok' if generate_eda_report(pd.DataFrame({'a':[1.0,2.0],'b':[2.0,3.0]})) else 'fail')"
```

Expected: imports succeed; the EDA report returns an HTML string.

- [ ] **Step 3: Update the plan checkboxes**

Mark all completed steps and record any rulings in the ledger at `.superpowers/sdd/2026-10-29-p0-correctness/progress.md`.

- [ ] **Step 4: Commit final docs/tests**

```bash
git add README.md docs tests
git commit -m "docs: document P0 correctness fixes"
```

---

## Self-Review

- Spec coverage: every P0 item from the review is assigned to Task 2–7.
- Type consistency: `feature_encoding` keeps its existing signature; `__init__` adds aliases only.
- Review focus: each of the ten likely failure modes has a named test in Tasks 2–7.
- Scope: this plan intentionally excludes package renaming, ML-layer extraction, and the full P1 list; those should be a separate plan.
