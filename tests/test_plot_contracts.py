import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from utils._plotting import finalize_figure, save_figure
from utils.classification_evals_and_tuning import (
    compare_cv_metrics as compare_classification_metrics,
)
from utils.regression_evals_and_tuning import (
    compare_cv_metrics as compare_regression_metrics,
)


def _classification_metrics():
    return {
        "Model A": {
            "cv": {
                "accuracy": {"test_mean": 0.90, "test_std": 0.02},
                "precision": {"test_mean": 0.80, "test_std": 0.03},
                "recall": {"test_mean": 0.70, "test_std": 0.04},
                "f1": {"test_mean": 0.75, "test_std": 0.05},
                "roc_auc": {"test_mean": 0.95, "test_std": 0.01},
            }
        }
    }


def _regression_metrics():
    return {
        "Model A": {
            "cv": {
                "mae": {"test_mean": 1.5, "test_std": 0.1},
                "mse": {"test_mean": 4.0, "test_std": 0.2},
                "rmse": {"test_mean": 2.0, "test_std": 0.3},
                "mape": {"test_mean": 0.10, "test_std": 0.01},
                "r2": {"test_mean": 0.90, "test_std": 0.02},
            }
        }
    }


def test_compare_cv_metrics_numeric_mode():
    classification = compare_classification_metrics(
        _classification_metrics(), as_numeric=True
    )
    regression = compare_regression_metrics(_regression_metrics(), as_numeric=True)

    for column in ["ACCURACY", "PRECISION", "RECALL", "F1", "ROC_AUC"]:
        assert classification[column].dtype == float
    for column in ["MAE", "MSE", "RMSE", "MAPE", "R2"]:
        assert regression[column].dtype == float

    assert classification.loc[0, "ACCURACY"] == pytest.approx(0.90)
    assert regression.loc[0, "MAPE"] == pytest.approx(0.10)
    assert regression.loc[0, "R2"] == pytest.approx(0.90)


def test_compare_cv_metrics_default_is_backward_compatible():
    classification = compare_classification_metrics(_classification_metrics())
    regression = compare_regression_metrics(_regression_metrics())

    assert classification.loc[0, "ACCURACY"] == "90.00% ± 2.00%"
    assert regression.loc[0, "MAPE"] == "10.00% ± 1.00%"
    assert regression.loc[0, "R2"] == "0.9000 ± 0.0200"


def test_plot_helper_does_not_show_when_false(monkeypatch):
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    figure = plt.figure()
    returned = finalize_figure(figure, show=False)

    assert returned is figure
    assert show_calls == []


def test_plot_helper_returns_figure():
    figure = plt.figure()
    assert isinstance(finalize_figure(figure, show=False), Figure)


def test_plot_helper_shows_when_true(monkeypatch):
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    finalize_figure(plt.figure(), show=True)

    assert show_calls == [True]


def test_save_figure_writes_file(tmp_path):
    figure = plt.figure()
    figure.add_subplot().plot([0, 1], [0, 1])
    path = tmp_path / "figure.png"

    returned = save_figure(figure, path)

    assert returned == path
    assert path.exists()
    assert path.stat().st_size > 0
