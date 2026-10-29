# ML Architecture Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the near-duplicate classification/regression ML logic, make evaluation single-pass, return figures instead of always showing them, and preserve every existing public result key.

**Architecture:** Keep `classification_evals_and_tuning.py` and `regression_evals_and_tuning.py` as public adapters. Move shared CV collection, metric aggregation, tuning, and curve plotting into `utils/_ml_common.py` and `utils/_plotting.py`. Existing function names and return dictionaries remain stable; new parameters are additive.

**Tech Stack:** Python 3.14, scikit-learn 1.7.2, pandas 2.3.3, matplotlib 3.10, pytest.

**Spec:** `docs/superpowers/plans/2026-10-30-refactor-roadmap.md`, the P0/P1 plans, and the current classification/regression modules.

## Global Constraints

- Keep Python 3.14.
- Do not rename the package in this plan.
- Do not change the existing metric dictionary schema or the public function names.
- Preserve `eval_classification` binary-only behavior and its `predict_proba` guard.
- All new plotting helpers must work with the `Agg` backend and return objects.
- Tests must not depend on optional text/translation extras.
- Run `uv run pytest -q` before each commit.

## Review Focus

Likely user-visible failures:

- A classification evaluation that silently fits the model 100+ times when curves are enabled.
- CV metrics that no longer match the previous schema after refactoring.
- Tuning scores computed with the wrong scorer or with a private `_score_func`.
- Learning curves disappearing when `display=False`.
- Validation curves using the test set for model selection.
- `compare_cv_metrics` becoming unusable for sorting/plotting.
- Figure-returning changes breaking notebook display.

---

### Task 1: Add fold-result collection and a counting-estimator test

**Files:**
- Create: `utils/_ml_common.py`
- Create: `tests/test_ml_common.py`

**Interfaces:**
- Produces:
  - `@dataclass FoldResult`: `y_true`, `y_pred`, `y_proba`, `y_train_true`, `y_train_pred`, `train_index`, `val_index`
  - `collect_fold_results(estimator, X, y, cv, *, predict_proba=True) -> list[FoldResult]`
  - `aggregate_fold_metrics(results, metric_funcs, *, probability_metrics=...) -> dict[str, dict[str, float]]`

- [ ] **Step 1: Write failing tests**

- `test_collect_fold_results_fits_once_per_fold`: a class-attribute `CountingClassifier` records fits; with `n_splits=3`, `n_repeats=1`, assert `fit_count == 3`.
- `test_collect_fold_results_returns_expected_shapes`: each result has validation predictions/probabilities and training predictions.
- `test_aggregate_fold_metrics_matches_manual_sklearn`: compare accuracy/F1/ROC-AUC aggregation against a manual loop on a tiny dataset.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ml_common.py -q`

Expected: `_ml_common` does not exist.

- [ ] **Step 3: Implement the shared collector**

Fit one clone per fold, collect validation predictions/probabilities and training predictions, and aggregate metrics with mean/std. Keep the API task-agnostic: metric functions receive `(y_true, y_pred)` or `(y_true, y_proba)` as configured by the caller.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ml_common.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/_ml_common.py tests/test_ml_common.py
git commit -m "refactor: add shared fold-result collection"
```

---

### Task 2: Make classification evaluation single-pass

**Files:**
- Modify: `utils/classification_evals_and_tuning.py`
- Modify: `utils/_ml_common.py`
- Modify: `tests/test_eval_classification.py`
- Create: `tests/test_classification_single_pass.py`

**Interfaces:**
- Consumes: `collect_fold_results`, `aggregate_fold_metrics`.
- Produces: the same `eval_classification` signature/result keys, but ROC/PR/calibration/threshold plots reuse one set of fold results.

- [ ] **Step 1: Write failing tests**

- `test_classification_eval_fits_once_per_fold`: a counting estimator with all plot flags enabled, `n_splits=2`, `n_repeats=1`, must record exactly `2` fits after the caller’s initial fit.
- `test_classification_metrics_schema_unchanged`: assert the existing `cv`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, and `confusion_matrix` keys still exist.
- `test_all_curve_flags_return_figures_with_show_false`: with a new `show_plots=False` parameter, ROC/PR/calibration/threshold helpers return figures and do not call `plt.show`.
- `test_threshold_analysis_uses_cached_fold_probabilities`: monkeypatch the fold collector and assert it is called once.

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_classification_single_pass.py tests/test_eval_classification.py -q`

Expected: the fit count is 100+; `show_plots` is unsupported; multiple CV loops occur.

- [ ] **Step 3: Refactor evaluation**

Replace `cross_validate` plus the four separate `cv.split` loops with one `collect_fold_results` call. Build the existing `metrics['cv']` dictionary from the fold results. Keep the single-run test/train metrics from the supplied fitted model. Feed the cached fold probabilities to ROC, PR, calibration, and threshold plotting.

Add `show_plots: bool = True` at the end of `eval_classification`; when false, do not call `plt.show`. Keep the existing return dictionary and add figure objects only if a new `return_figures` flag is explicitly requested; do not put figures in the default metrics dict.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_classification_single_pass.py tests/test_eval_classification.py -q`

Expected: PASS; fit count is exactly one per fold.

- [ ] **Step 5: Commit**

```bash
git add utils/classification_evals_and_tuning.py utils/_ml_common.py tests
git commit -m "refactor: evaluate classification curves from one CV pass"
```

---

### Task 3: Share tuning and curve logic

**Files:**
- Modify: `utils/_ml_common.py`
- Modify: `utils/classification_evals_and_tuning.py`
- Modify: `utils/regression_evals_and_tuning.py`
- Create: `tests/test_tuning_common.py`

**Interfaces:**
- Produces:
  - `tune_models(pipelines, param_grid, X_train, y_train, X_test, y_test, *, task, scoring, search_method, n_iter, n_splits, n_repeats, random_state, display, plot_validation_curves, plot_learning_curves, progress_bar, show=True) -> tuple[dict, list[float]]`
  - `plot_learning_curves(estimator, X_train, y_train, *, scoring, name, cv, show=True) -> Figure`
  - `plot_validation_curves(estimator, X_train, y_train, *, param_name, param_values, scoring, cv, show=True) -> Figure`

- [ ] **Step 1: Write failing tests**

- `test_tuning_uses_public_scorer`: tuning with `scoring="f1_macro"` on multiclass data produces the same score as `get_scorer("f1_macro")`, not `f1_score` defaults.
- `test_learning_curves_plot_when_display_false`: `tune_models(..., display=False, plot_learning_curves=True, show=False)` returns a learning-curve figure.
- `test_validation_curves_use_training_data`: monkeypatch the test-set scorer and assert it is not called during validation-curve generation.
- `test_single_model_wrapper_preserves_return_shape`: `tune_single_model` still returns `(fitted_search, fit_time)`.

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_tuning_common.py -q`

Expected: classification test scoring uses `_score_func`; learning curves are inside `if display`; validation curves score `X_test`.

- [ ] **Step 3: Implement the shared tuning layer**

Move `GridSearchCV`/`RandomizedSearchCV` construction, timing, result printing, learning curves, and validation curves into `_ml_common.py`. Classification and regression pass a task config (CV factory and default scorer). Use `get_scorer(scoring)(estimator, X, y)` for holdout test scores. Generate validation curves with `sklearn.model_selection.validation_curve` on `X_train` and its CV; do not use `X_test` for selecting hyperparameters. Decouple learning-curve plotting from `display`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tuning_common.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/_ml_common.py utils/classification_evals_and_tuning.py utils/regression_evals_and_tuning.py tests
git commit -m "refactor: share tuning and curve generation"
```

---

### Task 4: Numeric comparison output and figure contracts

**Files:**
- Modify: `utils/classification_evals_and_tuning.py`
- Modify: `utils/regression_evals_and_tuning.py`
- Create: `utils/_plotting.py`
- Create: `tests/test_plot_contracts.py`

**Interfaces:**
- Produces:
  - `compare_cv_metrics(metrics_dict, *, as_numeric: bool = False)` preserving the existing formatted default.
  - `finalize_figure(fig, *, show: bool) -> Figure`
  - `save_figure(fig, path)` helper.

- [ ] **Step 1: Write failing tests**

- `test_compare_cv_metrics_numeric_mode`: `as_numeric=True` yields float columns, not strings.
- `test_compare_cv_metrics_default_is_backward_compatible`: default still yields `"xx.xx% ± yy.yy%"` strings.
- `test_plot_helper_does_not_show_when_false`: monkeypatch `plt.show`; assert not called.
- `test_plot_helper_returns_figure`: `isinstance(fig, matplotlib.figure.Figure)`.

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_plot_contracts.py -q`

Expected: numeric mode doesn’t exist; helpers don’t exist.

- [ ] **Step 3: Implement the output/plot layer**

Add `as_numeric` to both comparison functions. Keep the formatted default for notebook compatibility. Add `_plotting.py` with figure finalization/saving helpers. Update the evaluation and tuning functions to call the helper with a `show` flag. Do not change the default display behavior of existing public functions in this task.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_plot_contracts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/_plotting.py utils/classification_evals_and_tuning.py utils/regression_evals_and_tuning.py tests
git commit -m "refactor: add numeric metrics and figure contracts"
```

---

### Task 5: Replace ML print output with logging

**Files:**
- Modify: `utils/_ml_common.py`
- Modify: `utils/classification_evals_and_tuning.py`
- Modify: `utils/regression_evals_and_tuning.py`
- Create: `tests/test_ml_logging.py`

**Interfaces:**
- Produces: module logger `logging.getLogger("utils.ml")`; `display` controls logger verbosity rather than swallowing plots.

- [ ] **Step 1: Write failing tests**

- `test_eval_does_not_print_when_display_false`: `capsys.readouterr().out == ""`; use `display=False` if added or assert logger output only.
- `test_eval_logs_metric_summary_at_info`: use `caplog.at_level(logging.INFO)` and assert a metric name appears.
- `test_tuning_errors_are_logged_and_raised`: invalid grid raises and logs the pipeline name.

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest tests/test_ml_logging.py -q`

Expected: functions print unconditionally.

- [ ] **Step 3: Implement logging**

Add a module logger; replace `print` in `_ml_common.py` and the evaluation/tuning paths with `logger.info`/`logger.warning`. Keep public behavior: when `display=True` (default), configure a `StreamHandler` or log at INFO; when false, suppress. Do not change result dictionaries.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ml_logging.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils tests
git commit -m "refactor: use logging for ML diagnostics"
```

---

### Task 6: Full verification

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`

Expected: all tests pass, including P1 tests; warnings only the known intentional/upstream ones.

- [ ] **Step 2: Check duplication**

Run:

```bash
uv run python -c "from pathlib import Path; import re; files=['utils/classification_evals_and_tuning.py','utils/regression_evals_and_tuning.py']; print({f: len(Path(f).read_text().splitlines()) for f in files})"
```

Expected: both adapter files are substantially smaller than the pre-refactor versions; shared logic lives in `_ml_common.py`.

- [ ] **Step 3: Smoke a notebook-style flow**

Run a tiny binary classification evaluation with `show_plots=False` and a tiny regression evaluation with plots disabled; print metric keys.

- [ ] **Step 4: Commit**

```bash
git add utils tests
git commit -m "refactor: complete shared ML core"
```

---

## Self-Review

- The plan preserves the metric dict schema and public function names.
- Single-pass evaluation is the central correctness/performance change.
- Tuning and plotting are shared without moving package files.
- Logging is deliberately last so behavior changes are easy to isolate.
- Packaging and CI are not included.
