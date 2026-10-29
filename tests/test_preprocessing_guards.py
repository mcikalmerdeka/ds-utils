import warnings

import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import (
    check_data_information,
    feature_scaling,
    filter_outliers,
    handle_missing_values,
)


def test_filter_outliers_keeps_nan_rows():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})

    result = filter_outliers(df, ["a"], method="zscore", threshold=3.0)

    assert len(result) == len(df)
    assert result["a"].isna().sum() == 1


def test_filter_outliers_detection_ignores_nan_as_outlier():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})

    summary = filter_outliers(
        df, ["a"], method="zscore", threshold=3.0, detect_only=True, verbose=False
    )

    assert summary.loc[0, "Outlier Data"] == 0
    assert summary.loc[0, "Outlier Percentage (%)"] == 0.0
    assert bool(summary.loc[0, "Outlier Exist"]) is False


def test_filter_outliers_detection_reports_zero_valued_outlier():
    df = pd.DataFrame({"a": [10.0, 10.0, 10.0, 10.0, 0.0]})

    summary = filter_outliers(df, ["a"], detect_only=True, verbose=False)

    assert summary.loc[0, "Outlier Data"] == 1
    assert summary.loc[0, "Outlier Percentage (%)"] == 20.0
    assert bool(summary.loc[0, "Outlier Exist"]) is True


def test_filter_outliers_zscore_keeps_constant_column_rows():
    df = pd.DataFrame({"a": [10.0, 10.0, 10.0]})

    result = filter_outliers(df, ["a"], method="zscore", threshold=3.0)
    summary = filter_outliers(
        df, ["a"], method="zscore", threshold=3.0, detect_only=True, verbose=False
    )

    assert len(result) == 3
    assert summary.loc[0, "Outlier Data"] == 0
    assert bool(summary.loc[0, "Outlier Exist"]) is False


def test_filter_outliers_zscore_keeps_single_row():
    df = pd.DataFrame({"a": [5.0]})

    result = filter_outliers(df, ["a"], method="zscore", threshold=3.0)

    assert len(result) == 1


def test_handle_missing_values_median_rejects_strings_with_actionable_error():
    df = pd.DataFrame({"a": ["x", None, "y"]})

    with pytest.raises(ValueError, match="mode"):
        handle_missing_values(df, ["a"], strategy="median")


def test_handle_missing_values_ffill_has_no_deprecation_warning():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, _ = handle_missing_values(df, ["a"], strategy="ffill")

    assert result.loc[1, "a"] == 1.0
    assert not any(issubclass(w.category, FutureWarning) for w in caught)


def test_feature_scaling_rejects_log_of_negative_values():
    df = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0]})

    with pytest.raises(ValueError, match="non-negative"):
        feature_scaling(df, columns=["x"], method="standard", apply_log=True)


def test_check_data_information_empty_frame_returns_columns():
    result = check_data_information(pd.DataFrame({"a": []}), ["a"])

    assert result.empty
    assert "Feature" in result.columns
    assert "Null Percentage" in result.columns
