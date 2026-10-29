import numpy as np
import pandas as pd

from utils.ab_testing import (
    analyze_ab_test_results,
    calculate_sample_size,
    perform_chi_squared_test,
    perform_t_test,
)


def test_calculate_sample_size_returns_positive_count():
    n = calculate_sample_size(0.10, 0.02, metric_type="proportion")
    assert isinstance(n, int)
    assert n > 0


def test_t_test_detects_higher_treatment_mean():
    rng = np.random.default_rng(42)
    group_a = rng.normal(100, 10, 100)
    group_b = rng.normal(105, 10, 100)

    result = perform_t_test(group_a, group_b)

    assert result["mean_b"] > result["mean_a"]
    assert 0.0 <= result["p_value"] <= 1.0


def test_chi_squared_detects_large_conversion_difference():
    result = perform_chi_squared_test(100, 1000, 150, 1000)
    assert bool(result["significant"]) is True
    assert result["rate_b"] > result["rate_a"]


def test_high_level_analysis_returns_expected_keys():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "group": ["A"] * 100 + ["B"] * 100,
            "revenue": np.concatenate(
                [rng.normal(100, 10, 100), rng.normal(105, 10, 100)]
            ),
        }
    )

    result = analyze_ab_test_results(df, "revenue", "group", metric_type="continuous")

    assert result["sample_size_a"] == 100
    assert result["sample_size_b"] == 100
    assert "p_value" in result
