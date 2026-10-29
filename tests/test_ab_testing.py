from math import ceil

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.power import TTestIndPower

from utils.ab_testing import (
    analyze_ab_test_results,
    calculate_sample_size,
    perform_t_test,
)


def test_sample_size_rejects_impossible_rate():
    with pytest.raises(ValueError):
        calculate_sample_size(0.9, 0.2)


@pytest.mark.parametrize(
    "baseline, mde, alpha, power",
    [
        (0.0, 0.02, 0.05, 0.8),
        (1.0, 0.02, 0.05, 0.8),
        (0.1, 0.0, 0.05, 0.8),
        (0.1, 0.02, 0.0, 0.8),
        (0.1, 0.02, 1.0, 0.8),
        (0.1, 0.02, 0.05, 0.0),
        (0.1, 0.02, 0.05, 1.0),
    ],
)
def test_sample_size_rejects_invalid_bounds(baseline, mde, alpha, power):
    with pytest.raises(ValueError):
        calculate_sample_size(
            baseline, mde, alpha=alpha, power=power, metric_type="proportion"
        )


def test_sample_size_continuous_with_std():
    expected = ceil(
        TTestIndPower().solve_power(
            effect_size=5 / 10, alpha=0.05, power=0.8, ratio=1.0
        )
    )

    result = calculate_sample_size(
        100.0, 5.0, metric_type="continuous", std_dev=10.0
    )

    assert result == expected


def test_t_test_alpha_is_respected():
    rng = np.random.default_rng(9)
    group_a = rng.normal(0, 1, 20)
    group_b = rng.normal(1.1, 1, 20)

    default_result = perform_t_test(group_a, group_b)
    strict_result = perform_t_test(group_a, group_b, alpha=0.001)

    assert default_result["p_value"] < 0.05
    assert bool(default_result["significant"]) is True
    assert bool(strict_result["significant"]) is False


def test_binary_metric_must_be_zero_or_one():
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "metric": ["yes", "no", "yes", "yes"],
        }
    )

    with pytest.raises(ValueError, match="0/1|binary"):
        analyze_ab_test_results(data, "metric", "group", metric_type="binary")


def test_empty_group_raises():
    data = pd.DataFrame(
        {
            "group": ["A"] * 10,
            "metric": np.arange(10, dtype=float),
        }
    )

    with pytest.raises(ValueError, match="empty|no rows"):
        analyze_ab_test_results(data, "metric", "group", metric_type="continuous")


def test_ab_result_reports_lift_and_ci():
    data = pd.DataFrame(
        {
            "group": ["A"] * 1000 + ["B"] * 1000,
            "converted": [0] * 900 + [1] * 100 + [0] * 850 + [1] * 150,
        }
    )

    result = analyze_ab_test_results(
        data, "converted", "group", metric_type="binary"
    )

    assert result["absolute_lift"] == pytest.approx(0.05)
    assert result["relative_lift"] == pytest.approx(0.5)
    assert result["ci_lower"] <= result["absolute_lift"] <= result["ci_upper"]


def test_continuous_result_reports_lift_and_ci():
    rng = np.random.default_rng(9)
    group_a = rng.normal(0, 1, 20)
    group_b = rng.normal(1.1, 1, 20)
    data = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "metric": np.concatenate([group_a, group_b]),
        }
    )

    result = analyze_ab_test_results(
        data, "metric", "group", metric_type="continuous"
    )

    expected_lift = group_b.mean() - group_a.mean()
    assert result["absolute_lift"] == pytest.approx(expected_lift)
    assert result["relative_lift"] == pytest.approx(expected_lift / group_a.mean())
    assert result["ci_lower"] <= result["absolute_lift"] <= result["ci_upper"]
