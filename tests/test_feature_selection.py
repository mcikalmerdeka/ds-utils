import warnings

import numpy as np
import pandas as pd
import pytest

from utils.feature_selection import (
    analyze_categorical_relationships,
    calculate_correlation_tabular,
    calculate_feature_importance,
    calculate_vif,
)


def _pair_set(result: pd.DataFrame) -> set[tuple[str, str]]:
    return {tuple(sorted((row["A"], row["B"]))) for _, row in result.iterrows()}


def test_correlation_deduplicates_unordered_pairs():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=30),
            "x2": rng.normal(size=30),
            "x3": rng.normal(size=30),
            "x4": rng.normal(size=30),
            "target": rng.normal(size=30),
        }
    )
    # Perfectly duplicate two distinct feature pairs. The old dedupe key
    # (absolute value + sign) kept only one of these equally strong pairs.
    df["x2"] = df["x1"]
    df["x4"] = df["x3"]

    result = calculate_correlation_tabular(
        df, target_col="target", corr_type="feature_feature"
    )

    pairs = _pair_set(result)
    directional_pairs = set(zip(result["A"], result["B"]))
    assert ("x1", "x2") in pairs
    assert ("x3", "x4") in pairs
    assert ("x2", "x1") not in directional_pairs
    assert ("x4", "x3") not in directional_pairs


def test_feature_importance_k_returns_top_k():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] + rng.normal(scale=0.1, size=50) > 0).astype(int))

    result = calculate_feature_importance(
        X, y, task="classification", method="mutual_info", k=2, random_state=7
    )

    assert len(result) == 2
    scores = result["Score"].to_list()
    assert scores == sorted(scores, reverse=True)


def test_feature_importance_mutual_info_is_reproducible():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.integers(0, 2, size=40))

    first = calculate_feature_importance(
        X, y, task="classification", method="mutual_info", random_state=7
    )
    second = calculate_feature_importance(
        X, y, task="classification", method="mutual_info", random_state=7
    )

    assert first["Score"].tolist() == second["Score"].tolist()


def test_feature_importance_rejects_non_positive_k():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    y = pd.Series([0, 1, 0, 1])

    with pytest.raises(ValueError, match="k"):
        calculate_feature_importance(X, y, k=0)


def test_vif_rejects_non_numeric():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "label": ["a", "b", "c", "d"],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="label"):
        calculate_vif(df, target_col="target")


def test_vif_returns_infinite_for_constant_feature():
    df = pd.DataFrame(
        {
            "varying": [1.0, 2.0, 3.0, 4.0, 5.0],
            "constant": [7.0, 7.0, 7.0, 7.0, 7.0],
            "target": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )

    result = calculate_vif(df, target_col="target")
    row = result.loc[result["Feature"] == "constant"].iloc[0]

    assert np.isinf(row["VIF"])
    assert row["Interpretation"] == "High multicollinearity"


def test_vif_drops_nan_rows():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, np.nan],
            "b": [2.0, 4.0, 5.0, 8.0, 1.0],
            "target": [0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )

    result = calculate_vif(df, target_col="target")

    assert set(result["Feature"]) == {"a", "b"}
    assert np.isfinite(result["VIF"]).all()


def test_categorical_single_category_returns_nan_effect_size():
    df = pd.DataFrame(
        {
            "feature": ["only", "only", "only", "only"],
            "target": ["yes", "no", "yes", "no"],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = analyze_categorical_relationships(df, ["feature"], "target")

    assert len(result) == 1
    assert np.isnan(result.loc[0, "Cramér's V"])
