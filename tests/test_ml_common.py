import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from utils._ml_common import (
    FoldResult,
    aggregate_fold_metrics,
    collect_fold_results,
)


class CountingClassifier(BaseEstimator, ClassifierMixin):
    """Small classifier that records how many times it is fitted."""

    fit_count = 0

    def __init__(self, offset=0.0):
        self.offset = offset

    def fit(self, X, y):
        type(self).fit_count += 1
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        positive = np.clip(np.asarray(X, dtype=float).mean(axis=1) + self.offset, 0, 1)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _dataset():
    return make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )


def test_collect_fold_results_fits_once_per_fold():
    X, y = _dataset()
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=0)
    CountingClassifier.fit_count = 0

    results = collect_fold_results(CountingClassifier(), X, y, cv)

    assert len(results) == 3
    assert CountingClassifier.fit_count == 3


def test_collect_fold_results_returns_expected_shapes():
    X, y = _dataset()
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=0)

    results = collect_fold_results(LogisticRegression(max_iter=1000), X, y, cv)

    assert len(results) == 2
    for result in results:
        assert isinstance(result, FoldResult)
        assert len(result.y_true) == len(result.val_index)
        assert result.y_pred.shape == result.y_true.shape
        assert result.y_proba.shape == (len(result.y_true), 2)
        assert len(result.y_train_true) == len(result.train_index)
        assert result.y_train_pred.shape == result.y_train_true.shape
        assert np.array_equal(result.y_true, y[result.val_index])
        assert np.array_equal(result.y_train_true, y[result.train_index])


def test_collect_fold_results_supports_sparse_features():
    from scipy.sparse import csr_matrix

    X, y = _dataset()
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=0)

    results = collect_fold_results(
        LogisticRegression(max_iter=1000), csr_matrix(X), y, cv
    )

    assert len(results) == 2
    for result in results:
        assert result.y_proba.shape == (len(result.y_true), 2)


def test_aggregate_fold_metrics_matches_manual_sklearn():
    X, y = _dataset()
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=7)
    estimator = LogisticRegression(max_iter=1000)

    results = collect_fold_results(estimator, X, y, cv)

    test_accuracy = []
    train_accuracy = []
    test_f1 = []
    train_f1 = []
    test_roc_auc = []
    train_roc_auc = []

    for train_index, val_index in cv.split(X, y):
        fold_model = clone(estimator).fit(X[train_index], y[train_index])
        train_pred = fold_model.predict(X[train_index])
        val_pred = fold_model.predict(X[val_index])
        train_proba = fold_model.predict_proba(X[train_index])[:, 1]
        val_proba = fold_model.predict_proba(X[val_index])[:, 1]

        test_accuracy.append(accuracy_score(y[val_index], val_pred))
        train_accuracy.append(accuracy_score(y[train_index], train_pred))
        test_f1.append(f1_score(y[val_index], val_pred))
        train_f1.append(f1_score(y[train_index], train_pred))
        test_roc_auc.append(roc_auc_score(y[val_index], val_proba))
        train_roc_auc.append(roc_auc_score(y[train_index], train_proba))

    metrics = aggregate_fold_metrics(
        results,
        {"accuracy": accuracy_score, "f1": f1_score},
        probability_metrics={"roc_auc": roc_auc_score},
    )

    assert metrics["accuracy"]["test_mean"] == pytest.approx(np.mean(test_accuracy))
    assert metrics["accuracy"]["test_std"] == pytest.approx(np.std(test_accuracy))
    assert metrics["accuracy"]["train_mean"] == pytest.approx(np.mean(train_accuracy))
    assert metrics["accuracy"]["train_std"] == pytest.approx(np.std(train_accuracy))
    assert metrics["f1"]["test_mean"] == pytest.approx(np.mean(test_f1))
    assert metrics["f1"]["train_mean"] == pytest.approx(np.mean(train_f1))
    assert metrics["roc_auc"]["test_mean"] == pytest.approx(np.mean(test_roc_auc))
    assert metrics["roc_auc"]["test_std"] == pytest.approx(np.std(test_roc_auc))
    assert metrics["roc_auc"]["train_mean"] == pytest.approx(np.mean(train_roc_auc))
    assert metrics["roc_auc"]["train_std"] == pytest.approx(np.std(train_roc_auc))
