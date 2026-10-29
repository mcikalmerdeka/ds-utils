import numpy as np
import pandas as pd

from utils.eda_report import generate_eda_report


def test_eda_report_supports_float32_columns():
    df = pd.DataFrame(
        {
            "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32"),
            "b": np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype="float32"),
        }
    )

    html = generate_eda_report(df, verbose=False)

    assert "<!DOCTYPE html>" in html
    assert "Numerical Summary" in html


def test_eda_report_single_numeric_feature_includes_target_correlation():
    df = pd.DataFrame(
        {
            "x": np.arange(10, dtype=float),
            "target": np.arange(10, dtype=float),
        }
    )

    html = generate_eda_report(df, target="target", verbose=False)

    assert "No target column provided or no numeric features." not in html
    assert "Feature-Target Correlations" in html
