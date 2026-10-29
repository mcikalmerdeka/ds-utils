"""Shared cross-validation collection and aggregation helpers for ML utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone


@dataclass
class FoldResult:
    """Predictions collected from a single cross-validation fold."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    y_train_true: np.ndarray
    y_train_pred: np.ndarray
    train_index: np.ndarray
    val_index: np.ndarray
    y_train_proba: np.ndarray | None = None


def _to_numpy(values):
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.to_numpy()
    return np.asarray(values)


def _positive_class_proba(y_proba: np.ndarray) -> np.ndarray:
    """Return the positive-class probability vector for binary predictions."""
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2:
        return y_proba[:, 1]
    return y_proba


def collect_fold_results(
    estimator,
    X,
    y,
    cv,
    *,
    predict_proba: bool = True,
) -> list[FoldResult]:
    """Fit one clone per fold and collect train/validation predictions.

    Fitting happens exactly once per fold, so callers can reuse the same
    predictions for metrics and every downstream diagnostic plot.
    """
    X = _to_numpy(X)
    y = _to_numpy(y)
    results: list[FoldResult] = []

    for train_index, val_index in cv.split(X, y):
        model = clone(estimator)
        model.fit(X[train_index], y[train_index])

        y_pred = model.predict(X[val_index])
        y_train_pred = model.predict(X[train_index])

        y_proba = None
        y_train_proba = None
        if predict_proba:
            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    "predict_proba=True requires an estimator with predict_proba."
                )
            y_proba = model.predict_proba(X[val_index])
            y_train_proba = model.predict_proba(X[train_index])

        results.append(
            FoldResult(
                y_true=y[val_index],
                y_pred=y_pred,
                y_proba=y_proba,
                y_train_true=y[train_index],
                y_train_pred=y_train_pred,
                train_index=train_index,
                val_index=val_index,
                y_train_proba=y_train_proba,
            )
        )

    return results


def _summarize(test_scores: Sequence[float], train_scores: Sequence[float]) -> dict:
    test_scores = np.asarray(test_scores, dtype=float)
    train_scores = np.asarray(train_scores, dtype=float)
    return {
        "test_mean": float(np.mean(test_scores)),
        "test_std": float(np.std(test_scores)),
        "train_mean": float(np.mean(train_scores)),
        "train_std": float(np.std(train_scores)),
    }


def aggregate_fold_metrics(
    results: Sequence[FoldResult],
    metric_funcs: Mapping[str, Callable],
    *,
    probability_metrics: Mapping[str, Callable] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate per-fold metrics into the public mean/std schema.

    ``metric_funcs`` receive ``(y_true, y_pred)``; ``probability_metrics``
    receive ``(y_true, y_proba)`` where ``y_proba`` is the positive-class
    probability vector.
    """
    metrics: dict[str, dict[str, float]] = {}

    for name, metric_func in metric_funcs.items():
        test_scores = [
            metric_func(result.y_true, result.y_pred) for result in results
        ]
        train_scores = [
            metric_func(result.y_train_true, result.y_train_pred)
            for result in results
        ]
        metrics[name] = _summarize(test_scores, train_scores)

    for name, metric_func in (probability_metrics or {}).items():
        test_scores = []
        train_scores = []
        for result in results:
            if result.y_proba is None or result.y_train_proba is None:
                raise ValueError(
                    f"Metric '{name}' needs probabilities; collect folds with "
                    "predict_proba=True and an estimator that supports it."
                )
            test_scores.append(
                metric_func(result.y_true, _positive_class_proba(result.y_proba))
            )
            train_scores.append(
                metric_func(
                    result.y_train_true,
                    _positive_class_proba(result.y_train_proba),
                )
            )
        metrics[name] = _summarize(test_scores, train_scores)

    return metrics
