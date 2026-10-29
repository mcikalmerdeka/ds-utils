import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """Small messy dataframe used by preprocessing/statistics tests."""
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0, 5.0],
            "float32": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32"),
            "cat": ["a", "b", None, "a", "b"],
            "text": ["hello", "world", None, "foo", "bar"],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        }
    )
