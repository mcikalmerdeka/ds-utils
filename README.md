# Data Science Utils

Reusable Python utilities for rapid exploratory data analysis, preprocessing, modeling, and reporting on tabular data.

The distribution is `data-science-utils`; the import package is `utils`. Requires **Python 3.14+**.

## Highlights

- Tabular, notebook-friendly API built around pandas and scikit-learn.
- One import surface: `from utils import ...` or `from utils import *`.
- Consistent helpers for data quality, cleaning, feature engineering, feature selection, statistics, and visualization.
- Classification and regression evaluation with repeated cross-validation, diagnostic plots, and optional headless operation.
- Grid and random-search tuning with shared behavior across both task types.
- Optional NLP, language-detection, and translation extras.
- Self-contained HTML EDA reports.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Feature overview](#feature-overview)
- [Machine learning](#machine-learning)
- [Project layout](#project-layout)
- [Development](#development)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Installation

Python **3.14** or newer is required.

### uv (recommended)

```bash
# Core dependencies and the package itself
uv sync

# Optional extras
uv sync --extra text
uv sync --extra translate
uv sync --extra all
```

`uv sync` is the primary workflow. `pyproject.toml` is the authoritative dependency source, and `uv.lock` records the tested dependency set.

### pip

```bash
# Dependencies only
pip install -r requirements.txt

# Install the package and its dependencies
pip install -e .

# With optional extras
pip install -e ".[text]"
pip install -e ".[translate]"
pip install -e ".[all]"
```

`requirements.txt` mirrors the direct core dependencies in `pyproject.toml`; keep both in sync if you edit dependency metadata.

### Optional extras

| Extra | Adds | Enables |
|---|---|---|
| `text` | `beautifulsoup4`, `contractions`, `nltk`, `Pillow`, `textblob`, `wordcloud` | `TextPreprocessor`, `TextVisualizer`, and text helpers |
| `translate` | `deep-translator`, `langid`, `tenacity` | Language detection and translation utilities |
| `all` | Both of the above | All optional text and translation features |

Text features based on NLTK may also need corpora downloaded locally:

```python
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
```

## Quick start

```python
import pandas as pd

from utils import (
    check_data_information,
    feature_scaling,
    generate_eda_report,
    handle_missing_values,
)

df = pd.read_csv("your_dataset.csv")

# Inspect data types, missing values, duplicates, and cardinality
info = check_data_information(df, df.columns.tolist())

# Impute missing values and keep the fitted imputer
df_clean, imputer = handle_missing_values(
    df,
    columns=["age", "salary"],
    strategy="median",
)

# Scale features and keep the fitted scaler
df_scaled, scaler = feature_scaling(
    df_clean,
    columns=["salary"],
    method="robust",
)

# Write a self-contained HTML report (the HTML is also returned)
report_html = generate_eda_report(
    df,
    target="target_column",
    output_path="eda_report.html",
)
```

## Feature overview

### Data inspection and cleaning

- **`check_data_information`** — data types, null values, duplicates, unique samples, zero and negative values, empty strings, and cardinality.
- **`handle_missing_values`** — mean, median, mode, forward fill, backward fill, and KNN imputation. Returns the cleaned frame and fitted imputer.
- **`filter_outliers`** — IQR or Z-score filtering, with detection-only and mask-return modes.
- **`drop_columns`** — safely remove one or more columns.
- **`change_binary_dtype`** — toggle binary columns between numeric `0/1` and categorical `No/Yes` representations.

### Feature engineering

- **`feature_scaling`** — standard, min-max, robust, power transform (Yeo-Johnson or Box-Cox), and quantile scaling. Returns the transformed frame and fitted scaler.
- **`feature_encoding`** — ordinal, one-hot, mean, binary, and frequency encoding with fitted encoders for train/test consistency and configurable unknown-value handling.

### Feature selection and statistics

- **`calculate_correlation_tabular`** — feature-feature or feature-target correlations in tabular form.
- **`analyze_categorical_relationships`** — chi-square tests and Cramér's V for categorical associations.
- **`calculate_vif`** — variance inflation factor for multicollinearity detection.
- **`calculate_feature_importance`** — univariate feature importance for classification and regression.
- **`describe_numerical_combined`** — descriptive statistics including skewness, kurtosis, IQR, and coefficient of variation, with optional grouping.
- **`describe_categorical_combined`** — frequency analysis, top and bottom categories, and percentages.
- **`describe_date_columns`** — temporal patterns, seasonality, gaps, and ranges.
- **`identify_distribution_types`** — classify numerical features by skewness and kurtosis.

### Visualization

- **`plot_dynamic_hisplots_kdeplots`** — histogram and KDE grids with hue support.
- **`plot_dynamic_boxplots_violinplots`** — box and violin grids for distribution and outlier inspection.
- **`plot_dynamic_countplot`** — count plots for categorical variables.
- **`plot_dynamic_stripplot_swarmplot`** — strip and swarm plots for categorical versus numerical relationships.
- **`plot_correlation_heatmap`** — Pearson, Spearman, or Kendall correlation heatmaps.

### A/B testing

- **`calculate_sample_size`** — required sample size for proportion or continuous metrics.
- **`check_normality`** — Shapiro-Wilk normality test.
- **`check_variance_homogeneity`** — Levene's test for equal variances.
- **`perform_t_test`** and **`perform_mann_whitney_u_test`** — parametric and non-parametric two-group tests.
- **`perform_chi_squared_test`** and **`perform_z_test_proportions`** — tests for conversion and proportion metrics.
- **`analyze_ab_test_results`** — high-level analysis that selects appropriate tests and reports effect sizes.

### Text processing and translation

Requires the `text` or `translate` extras; see [Installation](#installation).

- **`TextPreprocessor`** — HTML, URL, and email removal; normalization; tokenization; lemmatization; and stemming.
- **`TextVisualizer`** — word clouds and word-frequency visualizations.
- **`quick_clean`**, **`extract_ngrams`**, **`get_text_stats`** — convenience text utilities.
- **`detect_language_rows`** — language detection with target-language filtering.
- **`translate_series`**, **`translate_dataframe`**, **`run_translation_pipeline`**, **`translate_all_text`** — translation helpers for series and DataFrames.

### Automated reporting

- **`generate_eda_report`** — self-contained HTML report with data quality, summaries, correlations, visualizations, and optional target analysis.

### Time series

- **`decompose_series`** — trend, seasonal, and residual decomposition.
- **`adf_test`**, **`kpss_test`**, **`check_stationarity`** — stationarity tests.
- **`make_stationary`** — automatic differencing until stationarity.
- **`create_lag_features`**, **`create_rolling_features`**, **`create_date_features`** — time-series feature engineering.
- **`split_time_series`**, **`time_series_cv_splits`** — chronological train/test splits and walk-forward cross-validation.
- **`evaluate_forecast`** — forecast accuracy metrics.
- **`plot_acf_pacf`**, **`get_acf_pacf_values`** — ACF and PACF analysis.
- **`detect_outliers_iqr`**, **`interpolate_series`**, **`resample_series`** — outlier detection, interpolation, and resampling.

## Machine learning

Classification and regression helpers share the same repeated cross-validation evaluation and tuning core.

### Evaluation

- **`eval_classification`** — test-set metrics, repeated cross-validated metrics, confusion matrix, ROC, precision-recall, calibration, and threshold analysis. Requires a model with `predict_proba`.
- **`eval_regression`** — test-set metrics, repeated cross-validated metrics, predicted-versus-actual plots, and residual diagnostics.
- **`compare_cv_metrics`** — compare several models in one table. Results are formatted strings by default; pass `as_numeric=True` for numeric values.
- **`return_figures=True`** on `eval_classification` adds the generated figures to the returned dictionary.
- **`show_plots=False`** prevents `plt.show`; **`display=False`** suppresses the logged metric summary.

```python
from sklearn.model_selection import train_test_split

from utils import eval_classification, get_model_pipeline

# X and y are your feature matrix and target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = get_model_pipeline("randomforest")
model.fit(X_train, y_train)

metrics = eval_classification(
    model=model,
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,
    y_train=y_train,
    show_plots=False,
    display=False,
)
```

### Tuning

- **`tune_pipelines`** — tune a dictionary of pipelines against parameter grids.
- **`tune_single_model`** — build and tune one of the supported pipelines by name.
- **`tune_all_models`** — tune every supported model for the task.
- **`get_model_pipeline`** and **`get_hyperparameters`** — build pipelines and retrieve default search spaces.

Tuning helpers return `(fitted_models, fit_times)` and support `display=False` and `show=False` for headless use.

```python
from utils import tune_single_model

models, fit_times = tune_single_model(
    "randomforest",
    X_train,
    y_train,
    X_test,
    y_test,
    search_method="random",
    n_iter=20,
    display=False,
    show=False,
)
```

### Supported models

| Task | Model keys |
|---|---|
| Classification | `logisticregression`, `knn`, `decisiontree`, `randomforest`, `gb`, `xgboost` |
| Regression | `linearregression`, `ridge`, `lasso`, `elasticnet`, `knn`, `decisiontree`, `randomforest`, `gb`, `xgboost` |

## Project layout

```text
utils/                         Flat import package
  preprocessing.py
  statistics.py
  visualization.py
  feature_selection.py
  ab_testing.py
  classification_evals_and_tuning.py
  regression_evals_and_tuning.py
  text_preprocessing.py
  translation_utils.py
  eda_report.py
  time_series.py
  _ml_common.py                Shared evaluation/tuning core (internal)
  _plotting.py                 Shared figure finalization (internal)
tests/                         pytest suite
docs/                          Plans and guides
notebooks/                     Example notebooks
```

## Development

```bash
uv sync
uv run pytest
```

The test suite uses pytest and the non-interactive matplotlib `Agg` backend, so it does not open windows. Optional text and translation tests skip or monkeypatch missing dependencies.

## Roadmap

Core EDA, preprocessing, model evaluation and tuning, A/B testing, time series, text and translation, and automated reporting are implemented. Planned additions:

- Model persistence for preprocessing and model artifacts.
- Imbalanced-classification helpers and resampling wrappers.
- A unified pipeline orchestrator for end-to-end workflows.

## Contributing

Fork the repository, create a feature branch, run the full test suite, and open a pull request with a clear description of the change.
