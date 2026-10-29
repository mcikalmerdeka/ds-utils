# Rapid EDA and Preprocessing Utility Scripts for Data Science Projects

This repository contains a collection of utility scripts and functions designed to accelerate Exploratory Data Analysis (EDA), preprocessing, modeling, and reporting tasks in Data Science projects. These scripts are optimized for quick usage in environments like Kaggle kernels or local Jupyter notebooks. Initially this repository was created for personal use, but it can be useful for others as well.

## 🚀 Features

The scripts can be found in the `utils` folder. The available scripts are:

- `preprocessing.py`
- `statistics.py`
- `visualization.py`
- `feature_selection.py`
- `ab_testing.py`
- `classification_evals_and_tuning.py`
- `regression_evals_and_tuning.py`
- `text_preprocessing.py`
- `translation_utils.py`
- `eda_report.py`
- `time_series.py`

### 1. Data Inspection & Cleaning

- **`check_data_information`**: Comprehensive summary of data types, null values, duplicates, unique samples, zeros, negatives, and cardinality.
- **`handle_missing_values`**: Flexible imputation strategies including:
  - Simple: Mean, Median, Mode
  - Time-series: Forward fill, Backward fill
  - Advanced: KNN Imputation
- **`filter_outliers`**: Detect and filter outliers using IQR or Z-score methods, with detection-only mode.
- **`drop_columns`**: Safely remove unwanted columns.
- **`change_binary_dtype`**: Toggle binary columns between numerical (0/1) and categorical (No/Yes) formats.

### 2. Feature Engineering

- **`feature_scaling`**: Unified interface for multiple scaling methods:
  - Standard Scaler
  - MinMax Scaler
  - Robust Scaler
  - Power Transformer (Yeo-Johnson/Box-Cox)
  - Quantile Transformer
- **`feature_encoding`**: Wrapper for ordinal, one-hot, mean, binary, and frequency encoding with train/test-safe fitted encoders.

### 3. Feature Selection

- **`calculate_correlation_tabular`**: Feature-feature and feature-target correlations in tabular form.
- **`analyze_categorical_relationships`**: Chi-square test and Cramér's V for categorical associations.
- **`calculate_vif`**: Variance Inflation Factor for multicollinearity detection.
- **`calculate_feature_importance`**: Univariate feature importance for classification and regression.

### 4. Statistical Analysis

- **`describe_numerical_combined`**: Detailed stats including skewness, kurtosis, IQR, and CV, with optional grouping (hue).
- **`describe_categorical_combined`**: Frequency analysis, top/bottom categories, and percentages.
- **`describe_date_columns`**: Extract temporal patterns, seasonality, gaps, and ranges from datetime columns.
- **`identify_distribution_types`**: Automatically categorize numerical features by skewness and kurtosis.

### 5. Visualization

- **`plot_dynamic_hisplots_kdeplots`**: Grid of histograms/KDEs with hue support.
- **`plot_dynamic_boxplots_violinplots`**: Grid of box/violin plots for outlier detection.
- **`plot_dynamic_countplot`**: Grid of count plots for categorical variables.
- **`plot_correlation_heatmap`**: Pearson, Spearman, or Kendall correlation heatmaps.
- **`plot_dynamic_stripplot_swarmplot`**: Strip/swarm plots for categorical vs numerical relationships.

### 6. Machine Learning Evaluation & Tuning

- **`eval_classification`** / **`eval_regression`**: Comprehensive model evaluation with CV metrics and diagnostic plots.
- **`compare_cv_metrics`**: Compare CV metrics across multiple models in a single table.
- **`tune_pipelines`** / **`tune_single_model`** / **`tune_all_models`**: Hyperparameter tuning with grid/random search, validation curves, and learning curves.
- **`get_model_pipeline`** / **`get_hyperparameters`**: Pre-built pipelines and hyperparameter grids for common models.

Supported classification models: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
Supported regression models: Linear Regression, Ridge, Lasso, ElasticNet, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost.

### 7. A/B Testing

- **`calculate_sample_size`**: Required sample size for proportion or continuous metrics.
- **`check_normality`**: Shapiro-Wilk normality test.
- **`check_variance_homogeneity`**: Levene's test for equal variances.
- **`perform_t_test`** / **`perform_mann_whitney_u_test`**: Parametric and non-parametric two-group tests.
- **`perform_chi_squared_test`** / **`perform_z_test_proportions`**: Tests for conversion/proportion metrics.
- **`analyze_ab_test_results`**: High-level A/B test analysis that auto-selects appropriate tests.

### 8. Text Processing & Translation

- **`TextPreprocessor`**: Comprehensive NLP preprocessing class with HTML/URL/email removal, normalization, tokenization, lemmatization, and stemming.
- **`TextVisualizer`**: Word clouds and word-frequency visualizations.
- **`quick_clean`**, **`extract_ngrams`**, **`get_text_stats`**: Convenience text utilities.
- **`detect_language_rows`**, **`translate_series`**, **`translate_dataframe`**, **`run_translation_pipeline`**, **`translate_all_text`**: Language detection and translation utilities.

> Text and translation features require optional dependencies. See the Installation section below.

### 9. Automated Reporting

- **`generate_eda_report`**: Generate a self-contained HTML EDA report with data quality, summaries, correlations, visualizations, and target analysis.

### 10. Time Series Analysis

- **`decompose_series`**: Trend/seasonal/residual decomposition.
- **`adf_test`** / **`kpss_test`** / **`check_stationarity`**: Stationarity tests.
- **`make_stationary`**: Auto-difference a series until stationary.
- **`create_lag_features`** / **`create_rolling_features`** / **`create_date_features`**: Time-series feature engineering.
- **`split_time_series`** / **`time_series_cv_splits`**: Chronological train/test split and walk-forward CV.
- **`evaluate_forecast`**: Forecast accuracy metrics.
- **`plot_acf_pacf`** / **`get_acf_pacf_values`**: ACF/PACF analysis.
- **`detect_outliers_iqr`**, **`interpolate_series`**, **`resample_series`**: Outlier detection, interpolation, and resampling.

## 📦 Installation

The distribution is `data-science-utils`, and the import package is `utils`. This project uses `uv` for dependency management.

```bash
# Install core dependencies
uv sync

# Install with text processing extras
uv sync --extra text

# Install with translation extras
uv sync --extra translate

# Install everything
uv sync --extra all
```

`uv sync` is the primary installation workflow, and `pyproject.toml` is the authoritative dependency source. `requirements.txt` is generated from it with `uv export` for pip users; do not edit it manually.

To regenerate it from the repository root:

```bash
uv export --format requirements-txt --no-hashes --no-dev --output-file requirements.txt
```

The generated file includes `-e .`, so run the pip commands below from the repository root; `pip install -r requirements.txt` also installs the local package. `uv.lock` records the tested dependency set, while the floors in `pyproject.toml` state the minimum APIs the code relies on.

```bash
pip install -r requirements.txt

# Optional extras with pip
pip install -e ".[text]"
```

## 🧪 Development & Tests

The test suite uses `pytest` and a non-interactive matplotlib backend.

```bash
uv run pytest
```

Optional text/translation tests are skipped or monkeypatched unless the relevant extras are installed.

## 🛠️ Usage

You can import individual functions or use `from utils import *`:

```python
import pandas as pd
from utils import (
    check_data_information,
    handle_missing_values,
    feature_scaling,
    generate_eda_report,
)

# Load your data
df = pd.read_csv("your_dataset.csv")

# 1. Check Data Info
info = check_data_information(df, df.columns.tolist())
print(info)

# 2. Handle Missing Values
df_clean, _ = handle_missing_values(df, columns=["age", "salary"], strategy="median")

# 3. Scale Features
df_scaled, scaler = feature_scaling(df_clean, columns=["salary"], method="robust")

# 4. Generate an EDA report
html_report = generate_eda_report(df, target="target_column", output_path="eda_report.html")
```

## 🔮 Future Roadmap

This project is actively evolving. Planned additions and improvements include:

- [x] **Advanced Feature Selection**: Correlation, VIF, univariate importance.
- [x] **Model Evaluation Utils**: Classification/regression evaluation, CV comparison, ROC/PR curves, confusion matrices.
- [x] **Time Series Analysis**: Decomposition, ADF/KPSS stationarity tests, lag/rolling features, ACF/PACF.
- [x] **Text Processing**: Text cleaning, tokenization, lemmatization, stemming, word clouds, n-grams.
- [x] **Automated Reporting**: HTML EDA report generator.
- [ ] **Model Persistence**: Utilities to save/load preprocessing + model artifacts.
- [ ] **Imbalanced Classification Helpers**: Class-balance checks and resampling wrappers.
- [ ] **Pipeline Orchestrator**: Unified AutoML-style entry point for end-to-end workflows.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests if you have useful scripts to share!
