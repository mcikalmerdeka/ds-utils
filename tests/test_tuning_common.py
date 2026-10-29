import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

import utils._ml_common as ml_common
from utils._ml_common import (
    CLASSIFICATION_TASK,
    plot_learning_curves,
    tune_models,
)
from utils.regression_evals_and_tuning import tune_single_model


def _classification_split(n_classes=2, n_samples=240):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42,
    )
    return train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )


def _pipeline():
    return Pipeline([("clf", LogisticRegression(max_iter=1000, random_state=42))])


def test_tuning_uses_public_scorer():
    X_train, X_test, y_train, y_test = _classification_split(n_classes=3)

    fitted, _ = tune_models(
        {"clf": _pipeline()},
        {"clf": {"clf__C": [0.1, 1.0]}},
        X_train,
        y_train,
        X_test,
        y_test,
        task=CLASSIFICATION_TASK,
        scoring="f1_macro",
        n_splits=2,
        n_repeats=1,
        display=False,
        plot_validation_curves=False,
        plot_learning_curves=False,
        progress_bar=False,
    )

    search = fitted["clf"]
    expected = get_scorer("f1_macro")(search.best_estimator_, X_test, y_test)
    assert search.test_score_ == pytest.approx(expected)


def test_learning_curves_plot_when_display_false(monkeypatch):
    X_train, X_test, y_train, y_test = _classification_split()
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
    estimator = _pipeline().fit(X_train, y_train)

    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    figure = plot_learning_curves(
        estimator,
        X_train,
        y_train,
        scoring="accuracy",
        name="clf",
        cv=cv,
        show=False,
    )

    assert isinstance(figure, Figure)

    draw_calls = []

    def fake_draw(*args, **kwargs):
        draw_calls.append((args, kwargs))
        return figure

    monkeypatch.setattr(ml_common, "_draw_learning_curves", fake_draw)

    tune_models(
        {"clf": _pipeline()},
        {"clf": {"clf__C": [0.1, 1.0]}},
        X_train,
        y_train,
        X_test,
        y_test,
        task=CLASSIFICATION_TASK,
        scoring="accuracy",
        n_splits=2,
        n_repeats=1,
        display=False,
        plot_validation_curves=False,
        plot_learning_curves=True,
        progress_bar=False,
        show=False,
    )

    assert draw_calls
    assert show_calls == []


def test_validation_curves_use_training_data(monkeypatch):
    X_train, X_test, y_train, y_test = _classification_split()

    calls = []

    def fake_validation_curve(estimator, X, y, **kwargs):
        calls.append({"X": X, "y": y, "kwargs": kwargs})
        return np.full((2, 2), 0.8), np.full((2, 2), 0.7)

    monkeypatch.setattr(ml_common, "validation_curve", fake_validation_curve)

    tune_models(
        {"clf": _pipeline()},
        {"clf": {"clf__C": [0.1, 1.0]}},
        X_train,
        y_train,
        X_test,
        y_test,
        task=CLASSIFICATION_TASK,
        scoring="accuracy",
        n_splits=2,
        n_repeats=1,
        display=False,
        plot_validation_curves=True,
        plot_learning_curves=False,
        progress_bar=False,
        show=False,
    )

    assert calls
    assert all(call["X"] is X_train for call in calls)
    assert all(call["y"] is y_train for call in calls)
    assert all(call["X"] is not X_test for call in calls)


def test_single_model_wrapper_preserves_return_shape():
    X, y = make_regression(n_samples=80, n_features=4, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    search, fit_time = tune_single_model(
        "linearregression",
        X_train,
        y_train,
        X_test,
        y_test,
        scoring="r2",
        n_splits=2,
        n_repeats=1,
        display=False,
        plot_validation_curves=False,
        plot_learning_curves=False,
        progress_bar=False,
    )

    assert hasattr(search, "best_estimator_")
    assert isinstance(fit_time, float)
