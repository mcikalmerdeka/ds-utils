import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.classification_evals_and_tuning import eval_classification


def test_eval_classification_binary_returns_expected_metrics():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LogisticRegression(random_state=42).fit(X_train, y_train)

    metrics = eval_classification(
        model=model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        plot_confusion_matrix=False,
        plot_roc_curve=False,
        plot_pr_curve=False,
        plot_calibration_curve=False,
        plot_threshold_analysis=False,
        n_splits=2,
        n_repeats=1,
    )

    assert "cv" in metrics
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} == set(metrics["cv"])
    assert 0.0 <= metrics["accuracy"]["test"] <= 1.0
    assert 0.0 <= metrics["roc_auc"]["test"] <= 1.0
    assert np.isfinite(metrics["cv"]["f1"]["test_mean"])


def test_eval_classification_metrics_schema_is_unchanged():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LogisticRegression(random_state=42).fit(X_train, y_train)

    metrics = eval_classification(
        model=model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        plot_confusion_matrix=False,
        plot_roc_curve=False,
        plot_pr_curve=False,
        plot_calibration_curve=False,
        plot_threshold_analysis=False,
        n_splits=2,
        n_repeats=1,
    )

    expected_keys = {
        "cv",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
    }
    assert expected_keys.issubset(metrics)
    assert set(metrics["cv"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    for metric in metrics["cv"].values():
        assert {"test_mean", "test_std", "train_mean", "train_std"} == set(metric)
