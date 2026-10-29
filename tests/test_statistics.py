import numpy as np
import pandas as pd
import pytest

from utils.statistics import (
    describe_categorical_combined,
    describe_date_columns,
    describe_numerical_combined,
)


def test_numerical_summary_supports_float32():
    df = pd.DataFrame({"a": np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")})

    result = describe_numerical_combined(df, ["a"])

    assert len(result) == 1
    assert np.isfinite(result.loc[0, "overall_skewness"])
    assert np.isfinite(result.loc[0, "overall_kurtosis"])


def test_date_summary_parses_object_strings():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-02-01", "2024-03-01"]})

    result = describe_date_columns(df, ["d"])

    assert result.loc["d", "date_range_days"] == pd.Timedelta(days=60)
    assert result.loc["d", "unique_months"] == 3


def test_date_summary_rejects_unparseable_values():
    df = pd.DataFrame({"d": ["not-a-date"]})

    with pytest.raises(ValueError, match="could not be parsed"):
        describe_date_columns(df, ["d"])


def test_categorical_summary_handles_empty_hue_group():
    df = pd.DataFrame(
        {
            "cat": ["a", "b", None],
            "group": [1, 1, 2],
        }
    )

    result = describe_categorical_combined(df, ["cat"], hue="group")

    assert len(result) == 1
    assert "overall_n_categories" in result.columns
