import matplotlib.pyplot as plt
import pandas as pd
import pytest

from utils.visualization import (
    plot_dynamic_boxplots_violinplots,
    plot_dynamic_countplot,
    plot_dynamic_hisplots_kdeplots,
    plot_dynamic_stripplot_swarmplot,
)


@pytest.mark.parametrize(
    "plot",
    [
        pytest.param(
            lambda df: plot_dynamic_hisplots_kdeplots(df, ["num"]), id="hist"
        ),
        pytest.param(
            lambda df: plot_dynamic_boxplots_violinplots(df, ["num"]), id="box"
        ),
        pytest.param(
            lambda df: plot_dynamic_countplot(df, ["cat"]), id="count"
        ),
        pytest.param(
            lambda df: plot_dynamic_stripplot_swarmplot(df, "cat", ["num"]),
            id="strip",
        ),
    ],
)
def test_single_column_grid_does_not_raise(plot, mixed_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plot(mixed_df)


@pytest.mark.parametrize(
    "plot",
    [
        pytest.param(
            lambda df: plot_dynamic_hisplots_kdeplots(df, []), id="hist"
        ),
        pytest.param(
            lambda df: plot_dynamic_boxplots_violinplots(df, []), id="box"
        ),
        pytest.param(
            lambda df: plot_dynamic_countplot(df, []), id="count"
        ),
    ],
)
def test_empty_column_list_raises(plot, mixed_df):
    with pytest.raises(ValueError, match="at least one"):
        plot(mixed_df)
