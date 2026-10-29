import numpy as np
import pandas as pd
import pytest

from utils.time_series import (
    _infer_period,
    decompose_series,
    detect_outliers_iqr,
    evaluate_forecast,
    resample_series,
)


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("h", 24),
        ("ME", 12),
        ("QE-DEC", 4),
        ("D", 7),
        ("W-SUN", 52),
    ],
)
def test_infer_period_handles_modern_aliases(freq, expected):
    start = "2024-01-31" if freq in {"ME", "QE-DEC"} else "2024-01-01"
    series = pd.Series(
        np.arange(12, dtype=float),
        index=pd.date_range(start, periods=12, freq=freq),
    )

    assert _infer_period(series) == expected


def test_outlier_mask_preserves_input_index():
    index = pd.Index([10, 11, 12, 13])
    series = pd.Series([1.0, 2.0, np.nan, 100.0], index=index)

    mask = detect_outliers_iqr(series)

    assert len(mask) == len(series)
    assert mask.index.equals(series.index)
    assert mask.dtype == bool
    assert not mask.loc[12]


def test_outlier_mask_flags_outlier_with_nan():
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, np.nan, 100.0],
        index=range(11),
    )

    mask = detect_outliers_iqr(series)

    assert len(mask) == len(series)
    assert bool(mask.loc[10]) is True
    assert int(mask.sum()) == 1


def test_decompose_rejects_period_one():
    series = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError, match="period"):
        decompose_series(series, plot=False)


def test_mape_is_nan_with_zero_actual():
    metrics = evaluate_forecast([0, 1], [0, 2])

    assert np.isnan(metrics["mape"])


def test_resample_rejects_non_aggregation():
    series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    # 'plot' is not an aggregation at all; 'ohlc' is a real resampler method
    # but not an approved scalar aggregation.
    for aggregation in ("plot", "ohlc"):
        with pytest.raises(ValueError, match="aggregation"):
            resample_series(series, rule="2D", aggregation=aggregation)
