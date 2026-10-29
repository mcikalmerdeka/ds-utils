import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import (
    feature_encoding,
    feature_scaling,
    filter_outliers,
)

OUTLIER_SUMMARY_COLUMNS = [
    "Column Name",
    "Outlier Exist",
    "Lower Limit",
    "Upper Limit",
    "Outlier Data",
    "Non-Outlier Data",
    "Outlier Percentage (%)",
]


def test_filter_outliers_empty_frame_detection():
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})

    summary = filter_outliers(df, ["a"], detect_only=True, verbose=False)

    assert summary.empty
    assert list(summary.columns) == OUTLIER_SUMMARY_COLUMNS


def test_feature_scaling_empty_frame_raises_clear_error():
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})

    with pytest.raises(ValueError, match="empty"):
        feature_scaling(df, columns=["a"])


def test_feature_encoding_transform_requires_fitted_encoder():
    train = pd.DataFrame({"size": ["small", "medium", "small"]})
    categories = {"size": ["small", "medium", "large"]}
    _, encoders = feature_encoding(
        train,
        ordinal_columns=["size"],
        ordinal_categories=categories,
    )
    assert "ordinal_size" in encoders

    with pytest.raises(ValueError, match="ordinal_size"):
        feature_encoding(
            pd.DataFrame({"size": ["small"]}),
            ordinal_columns=["size"],
            ordinal_categories=categories,
            encoders={},
        )
