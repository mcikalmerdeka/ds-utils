import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from utils.classification_evals_and_tuning import eval_classification


def _split_dataset(n_classes):
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.25, random_state=42)


def test_eval_classification_rejects_multiclass_with_clear_message():
    X_train, X_test, y_train, y_test = _split_dataset(3)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    with pytest.raises(ValueError, match="binary classification"):
        eval_classification(
            model,
            X_test,
            y_test,
            X_train,
            y_train,
            plot_confusion_matrix=False,
            plot_roc_curve=False,
            plot_pr_curve=False,
            plot_calibration_curve=False,
            plot_threshold_analysis=False,
            n_splits=2,
            n_repeats=1,
        )


def test_eval_classification_rejects_models_without_predict_proba():
    X_train, X_test, y_train, y_test = _split_dataset(2)
    model = LinearSVC().fit(X_train, y_train)

    with pytest.raises(ValueError, match="predict_proba"):
        eval_classification(
            model,
            X_test,
            y_test,
            X_train,
            y_train,
            plot_confusion_matrix=False,
            plot_roc_curve=False,
            plot_pr_curve=False,
            plot_calibration_curve=False,
            plot_threshold_analysis=False,
            n_splits=2,
            n_repeats=1,
        )
