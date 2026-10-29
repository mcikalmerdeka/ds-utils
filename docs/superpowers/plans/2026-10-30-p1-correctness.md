# P1 Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the remaining P1 correctness issues in feature selection, A/B testing, time series, preprocessing/EDA edge cases, and optional-dependency boundaries, without renaming the package or extracting the ML core.

**Architecture:** Preserve the current flat modules and public signatures. Add parameters only at the end with backward-compatible defaults. Add focused helpers for validation, confidence intervals, frequency aliases, and encoder completeness. Keep presentation concerns for the architecture plan.

**Tech Stack:** Python 3.14, pandas 2.3.3, scikit-learn 1.7.2, statsmodels 0.14.5, scipy 1.16.3, pytest.

**Spec:** `docs/superpowers/plans/2026-10-30-refactor-roadmap.md`, the P0 plan, `README.md`, and the P0 review findings.

## Global Constraints

- Keep Python 3.14 and `uv`.
- Do not rename `utils` or change `pyproject.toml` package metadata in this pass.
- Do not extract `_ml_common.py` or change classification/regression evaluation architecture in this pass.
- Preserve existing return keys and function names. New parameters are optional and added at the end.
- All matplotlib tests use `Agg`.
- Optional text/translation tests must monkeypatch dependency flags; do not install extra packages.
- Every production change starts with a failing test.
- Run `uv run pytest -q` before each commit.

## Review Focus

The inputs most likely to break a reasonable user:

- Duplicate correlation pairs with equal absolute values.
- Feature importance `k` and reproducibility.
- VIF on object/NaN/constant columns.
- A/B tests with empty groups, non-0/1 “binary” metrics, tiny samples, and impossible rates.
- Modern pandas frequency aliases (`h`, `ME`, `QE`, `W`).
- Time-series masks with NaN and zero actuals in MAPE.
- Numeric/all-NaT date columns.
- Empty frames in outlier detection/scaling.
- Missing fitted encoders during transform.
- Optional dependency checks that demand unrelated packages.

---

### Task 1: Feature-selection correctness

**Files:**
- Modify: `utils/feature_selection.py`
- Create: `tests/test_feature_selection.py`

**Interfaces:**
- Consumes: `calculate_correlation_tabular`, `calculate_feature_importance`, `calculate_vif`, `analyze_categorical_relationships`.
- Produces: unordered-pair deduplication; top-k importance; reproducible mutual information; validated VIF; safe single-category chi-square output.

- [ ] **Step 1: Write failing tests**

Add tests:

- `test_correlation_deduplicates_unordered_pairs`: four features where `x1 == x2` and `x3 == x4`, plus a random target. Assert the result contains both `(x1, x2)` and `(x3, x4)` and no reversed duplicate.
- `test_feature_importance_k_returns_top_k`: `k=2` returns exactly two rows, sorted descending.
- `test_feature_importance_mutual_info_is_reproducible`: two calls with `random_state=7` return equal scores.
- `test_vif_rejects_non_numeric`: an object feature raises `ValueError` naming the column.
- `test_vif_returns_infinite_for_constant_feature`: a constant numeric feature yields `np.inf` and `High multicollinearity`.
- `test_categorical_single_category_returns_nan_effect_size`: a feature with one category returns one row and `np.isnan(Cramér's V)` without raising.

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_feature_selection.py -q`

Expected: correlation dedupe loses a valid pair; `k` is ignored; mutual-info calls differ; VIF returns `None`; single-category chi-square produces a warning/nan without a guard.

- [ ] **Step 3: Implement the fixes**

- Deduplicate feature pairs with an unordered key: `tuple(sorted((a, b)))`.
- Apply `k` after sorting: `feature_scores_df.head(k)` when `k != "all"`; reject `k <= 0`.
- Add `random_state: int = 42` to `calculate_feature_importance` and pass it to `mutual_info_classif`/`mutual_info_regression`.
- In `calculate_vif`, validate numeric-only features, explicitly handle NaN (drop rows or raise with the affected columns), and do not catch-all return `None`; raise a clear `ValueError`.
- In `analyze_categorical_relationships`, detect `min(contingency.shape) < 2` and return `np.nan` Cramér's V while still returning a row; skip the bar plot if the effect size is NaN.
- Add `random_state` to the docstring and maintain the existing default behavior for `k="all"`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_feature_selection.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/feature_selection.py tests/test_feature_selection.py
git commit -m "fix: correct feature-selection ranking and validation"
```

---

### Task 2: A/B test validation and inferential output

**Files:**
- Modify: `utils/ab_testing.py`
- Create: `tests/test_ab_testing.py`

**Interfaces:**
- Consumes: existing A/B functions.
- Produces: `alpha` parameters, binary/empty validation, sample-size validation, effect sizes and confidence intervals.

- [ ] **Step 1: Write failing tests**

Tests:

- `test_sample_size_rejects_impossible_rate`: `calculate_sample_size(0.9, 0.2)` raises `ValueError`.
- `test_sample_size_continuous_with_std`: with `std_dev=10`, result equals `TTestIndPower().solve_power(effect_size=0.5, ...)` for `MDE=5` (use `ceil`).
- `test_t_test_alpha_is_respected`: a p≈0.03 difference is non-significant at `alpha=0.001`.
- `test_binary_metric_must_be_zero_or_one`: strings raise `ValueError`.
- `test_empty_group_raises`: a group with no rows raises `ValueError`.
- `test_ab_result_reports_lift_and_ci`: result contains `absolute_lift`, `relative_lift`, `ci_lower`, `ci_upper`; the CI contains the observed difference.

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_ab_testing.py -q`

Expected: impossible rates return a result; continuous `std_dev` is unsupported; alpha is hard-coded; binary strings crash with a `TypeError`; tiny/empty groups produce NaN; no CI keys.

- [ ] **Step 3: Implement the fixes**

- Add `alpha: float = 0.05` to `check_normality`, `check_variance_homogeneity`, `perform_t_test`, `perform_mann_whitney_u_test`, `perform_chi_squared_test`, and `perform_z_test_proportions`.
- Add `std_dev: Optional[float] = None` to `calculate_sample_size`. For `metric_type="continuous"`, use `effect_size = minimum_detectable_effect / std_dev` when `std_dev` is provided; otherwise keep the existing Cohen's-d interpretation and document it.
- Validate proportion rates: `0 < baseline_rate < 1`, `minimum_detectable_effect > 0`, `baseline_rate + minimum_detectable_effect < 1`, `0 < alpha < 1`, `0 < power < 1`.
- Add `alpha` to `analyze_ab_test_results`; validate `metric_type`; reject empty groups; for `"binary"`, require values in `{0, 1}` (or booleans).
- Add effect-size and CI output. For proportions use `statsmodels.stats.proportion.confint_proportions_2indep`; for continuous use a two-sample mean-difference interval (Welch or pooled according to the selected test). Keep all existing result keys.
- Include `absolute_lift = rate_b - rate_a` and `relative_lift = absolute_lift / rate_a` for binary metrics; use mean difference/relative lift for continuous metrics.
- Do not change the default test-selection behavior beyond respecting `alpha` and validation.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ab_testing.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/ab_testing.py tests/test_ab_testing.py
git commit -m "fix: validate A/B inputs and report effect sizes"
```

---

### Task 3: Time-series aliases, masks, and forecast edge cases

**Files:**
- Modify: `utils/time_series.py`
- Create: `tests/test_time_series.py`

**Interfaces:**
- Consumes: `_infer_period`, `decompose_series`, `detect_outliers_iqr`, `evaluate_forecast`, `resample_series`.
- Produces: modern-frequency inference; index-preserving outlier masks; `period >= 2` validation; documented MAPE behavior; whitelisted resampling aggregations.

- [ ] **Step 1: Write failing tests**

Tests:

- `test_infer_period_handles_modern_aliases`: hourly `"h"` → 24; monthly `"ME"` → 12; quarterly `"QE"` → 4; daily `"D"` → 7; weekly `"W"` → 52.
- `test_outlier_mask_preserves_input_index`: a 4-row series with one NaN returns a 4-row mask with the original index.
- `test_decompose_rejects_period_one`: a two-row series raises `ValueError` mentioning period.
- `test_mape_is_nan_with_zero_actual`: `evaluate_forecast([0, 1], [0, 2])["mape"]` is `np.nan`.
- `test_resample_rejects_non_aggregation`: `aggregation="plot"` raises `ValueError`.

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_time_series.py -q`

Expected: hourly/monthly/quarterly periods are wrong; outlier masks are short; period 1 is accepted; MAPE is `inf`; arbitrary attributes are accepted.

- [ ] **Step 3: Implement the fixes**

- Normalize frequency aliases before matching: strip leading counts, uppercase, map `H/BH` → 24, `D/B` → 7, `W*` → 52, `M/ME/BM/BME` → 12, `Q/QE/BQ/BQE` → 4.
- Make `detect_outliers_iqr` return `pd.Series(False, index=series.index)` and assign the computed mask only on non-null positions.
- In `decompose_series`, raise `ValueError` if the resolved period is less than 2.
- In `evaluate_forecast`, return `mape=np.nan` when any actual value is zero and document the choice in the docstring; keep MAE/MSE/RMSE/sMAPE/MPE unchanged.
- In `resample_series`, whitelist common aggregation names (`mean`, `sum`, `min`, `max`, `median`, `first`, `last`, `count`, `std`, `var`) and raise for anything else.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_time_series.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/time_series.py tests/test_time_series.py
git commit -m "fix: align time-series helpers with modern pandas"
```

---

### Task 4: Preprocessing and EDA edge cases

**Files:**
- Modify: `utils/statistics.py`
- Modify: `utils/preprocessing.py`
- Modify: `utils/eda_report.py`
- Create: `tests/test_preprocessing_edges.py`
- Modify: `tests/test_statistics.py`
- Modify: `tests/test_eda_report.py`

**Interfaces:**
- Consumes: `describe_date_columns`, `filter_outliers`, `feature_scaling`, `feature_encoding`, `generate_eda_report`.
- Produces: safe numeric/all-NaT dates; empty-frame behavior; missing-encoder validation; escaped report title.

- [ ] **Step 1: Write failing tests**

Tests:

- `test_date_summary_rejects_numeric_dates`: `describe_date_columns(pd.DataFrame({"d": [20240101]}), ["d"])` raises `ValueError` mentioning datetime.
- `test_date_summary_handles_all_nat`: an all-`NaT` datetime column returns a row with `most_common_month`/`most_common_weekday` as NaN.
- `test_filter_outliers_empty_frame_detection`: an empty column returns an empty summary with the documented columns.
- `test_feature_scaling_empty_frame_raises_clear_error`: empty input raises `ValueError` matching `"empty"`.
- `test_feature_encoding_transform_requires_fitted_encoder`: passing an empty `encoders={}` with ordinal columns raises `ValueError` naming the missing encoder.
- `test_eda_report_escapes_title`: title `<script>alert(1)</script>` appears escaped and not as a raw tag.

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_preprocessing_edges.py tests/test_statistics.py tests/test_eda_report.py -q`

Expected: numeric dates become 1970; all-NaT raises on `.mode()`; empty outlier detection divides by zero; empty scaling raises sklearn’s message; missing encoders coerce strings to NaN; title is injected raw.

- [ ] **Step 3: Implement the fixes**

- In `describe_date_columns`, reject numeric dtypes unless explicitly converted by the caller; accept datetime/object/string. Guard `mode()` for empty series and return `np.nan` for month/weekday.
- In `filter_outliers`, when `len(data) == 0` and `detect_only=True`, return an empty summary with the standard columns; avoid percentage division.
- In `feature_scaling`, detect empty input and raise `ValueError("Cannot scale an empty DataFrame")`.
- In `feature_encoding` transform, validate that every requested ordinal/one-hot/frequency column has a fitted encoder; raise `ValueError` listing missing keys instead of coercing.
- In `generate_eda_report`, escape `title` with `html.escape` before interpolating it into HTML.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_preprocessing_edges.py tests/test_statistics.py tests/test_eda_report.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/statistics.py utils/preprocessing.py utils/eda_report.py tests
git commit -m "fix: cover preprocessing and EDA edge cases"
```

---

### Task 5: Optional-dependency boundaries

**Files:**
- Modify: `utils/translation_utils.py`
- Modify: `utils/text_preprocessing.py`
- Create: `tests/test_optional_deps.py`

**Interfaces:**
- Consumes: translation entry points and text preprocessing.
- Produces: detection requires only `langid`; translation requires only `deep-translator`; NLTK failures mention the text extra.

- [ ] **Step 1: Write failing tests**

Use `monkeypatch` on module-level availability flags and assert messages:

- `test_detection_requires_only_langid`: monkeypatch `_require_translate_deps` to raise and `_get_identifier` to a dummy identifier, then run a non-empty detection call. It must not raise a deep-translator error.
- `test_translation_requires_only_deep_translator`: monkeypatch `_translate_batch_with_retry` to identity and `DEEP_TRANSLATOR_AVAILABLE=True`; set `LANGID_AVAILABLE=False`; `translate_series` must not raise a langid error.
- `test_nltk_missing_message_mentions_extra`: with `NLTK_AVAILABLE=False`, `TextPreprocessor().process_text(...)` raises `ImportError` mentioning `text` extra or `nltk`.

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_optional_deps.py -q`

Expected: detection/translation require unrelated extras; NLTK message only suggests `pip install nltk`.

- [ ] **Step 3: Implement the fixes**

- Split `_require_translate_deps()` into `_require_detection_deps()` (langid) and `_require_translation_deps()` (deep-translator; tenacity is optional because the module has a fallback).
- Use the detection check in `detect_language_rows`/`run_translation_pipeline` detection step and the translation check in `translate_series`/`translate_dataframe`/`translate_all_text`.
- Update the NLTK error message to mention `uv sync --extra text` and `nltk.download(...)` for the required corpora.
- Do not install optional packages in this plan; tests use monkeypatching.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_optional_deps.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/translation_utils.py utils/text_preprocessing.py tests/test_optional_deps.py
git commit -m "fix: scope optional dependencies to their features"
```

---

### Task 6: Full suite and roadmap update

**Files:**
- Modify: `docs/superpowers/plans/2026-10-30-refactor-roadmap.md`

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`

Expected: all tests pass; only the known seaborn warning and intentional warning remain.

- [ ] **Step 2: Run a real-data smoke check**

Run:

```bash
uv run python -c "import pandas as pd; from utils import check_data_information, describe_numerical_combined; df=pd.read_csv('data/classification_data.csv'); print(check_data_information(df, df.columns.tolist()).shape); print(describe_numerical_combined(df, df.select_dtypes('number').columns.tolist()).shape)"
```

Expected: both frames print without exception.

- [ ] **Step 3: Record completion**

Add a line to `.superpowers/sdd/2026-10-30-p1-correctness/progress.md` with the commit range and test result. Do not edit the P0 plan.

- [ ] **Step 4: Commit**

```bash
git add docs tests
git commit -m "docs: mark P1 correctness complete"
```

---

## Self-Review

- Every P1 correctness item from the review has a task and a named test.
- Public API changes are additive; defaults preserve existing behavior except deliberate correctness fixes (MAPE zero behavior, numeric dates rejected, empty frames).
- ML architecture, package rename, and CI are explicitly out of scope.
- The plan can be split at task boundaries if a session runs short; each task ends with a green suite and a commit.
