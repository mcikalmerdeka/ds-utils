import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

import utils.classification_evals_and_tuning as classification_module
from utils._ml_common import collect_fold_results
from utils.classification_evals_and_tuning import (
    _plot_calibration_curve,
    _plot_pr_curve,
    _plot_roc_curve,
    _plot_threshold_analysis,
    eval_classification,
)


class CountingClassifier(LogisticRegression):
    """Logistic regression that counts every fit, including fold fits."""

    fit_count = 0

    def fit(self, X, y):
        type(self).fit_count += 1
        return super().fit(X, y)


def _split_dataset():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.25, random_state=42)


def test_classification_eval_fits_once_per_fold():
    X_train, X_test, y_train, y_test = _split_dataset()
    model = CountingClassifier(max_iter=1000).fit(X_train, y_train)
    CountingClassifier.fit_count = 0

    eval_classification(
        model=model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        plot_confusion_matrix=True,
        plot_roc_curve=True,
        plot_pr_curve=True,
        plot_calibration_curve=True,
        plot_threshold_analysis=True,
        n_splits=2,
        n_repeats=1,
        show_plots=False,
    )

    assert CountingClassifier.fit_count == 2


def test_all_curve_flags_return_figures_with_show_false(monkeypatch):
    X_train, X_test, y_train, y_test = _split_dataset()
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
    fold_results = collect_fold_results(model, X_train, y_train, cv)

    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    figures = [
        _plot_roc_curve(
            fold_results,
            model_name="LogisticRegression",
            n_splits=2,
            n_repeats=1,
            show=False,
        ),
        _plot_pr_curve(
            fold_results,
            model_name="LogisticRegression",
            n_splits=2,
            n_repeats=1,
            show=False,
        ),
        _plot_calibration_curve(
            fold_results,
            model_name="LogisticRegression",
            n_splits=2,
            n_repeats=1,
            show=False,
        ),
        _plot_threshold_analysis(
            fold_results,
            model_name="LogisticRegression",
            n_splits=2,
            n_repeats=1,
            show=False,
        ),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)
    assert show_calls == []


def test_threshold_analysis_uses_cached_fold_probabilities(monkeypatch):
    X_train, X_test, y_train, y_test = _split_dataset()
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    real_collector = classification_module.collect_fold_results
    calls = []

    def counting_collector(*args, **kwargs):
        calls.append((args, kwargs))
        return real_collector(*args, **kwargs)

    monkeypatch.setattr(
        classification_module, "collect_fold_results", counting_collector
    )

    eval_classification(
        model=model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        plot_confusion_matrix=False,
        plot_roc_curve=False,
        plot_pr_curve=False,
        plot_calibration_curve=False,
        plot_threshold_analysis=True,
        n_splits=2,
        n_repeats=1,
        show_plots=False,
    )

    assert len(calls) == 1
