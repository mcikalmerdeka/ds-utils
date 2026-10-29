import logging

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils._ml_common import CLASSIFICATION_TASK, tune_models
from utils.classification_evals_and_tuning import eval_classification


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


def test_eval_does_not_print_when_display_false(capsys):
    X_train, X_test, y_train, y_test = _split_dataset()
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

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
        plot_threshold_analysis=False,
        n_splits=2,
        n_repeats=1,
        display=False,
    )

    assert capsys.readouterr().out == ""


def test_eval_logs_metric_summary_at_info(caplog):
    X_train, X_test, y_train, y_test = _split_dataset()
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    with caplog.at_level(logging.INFO):
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
            plot_threshold_analysis=False,
            n_splits=2,
            n_repeats=1,
        )

    assert "accuracy" in caplog.text.lower()


def test_eval_logs_single_run_roc_auc(caplog):
    X_train, X_test, y_train, y_test = _split_dataset()
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    with caplog.at_level(logging.INFO):
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
            plot_threshold_analysis=False,
            n_splits=2,
            n_repeats=1,
        )

    messages = [record.getMessage() for record in caplog.records]
    single_run_start = next(
        index
        for index, message in enumerate(messages)
        if "Single Run Metrics" in message
    )
    single_run_output = "\n".join(messages[single_run_start:])

    assert "ROC_AUC" in single_run_output


def test_tuning_errors_are_logged_and_raised(caplog):
    X_train, X_test, y_train, y_test = _split_dataset()
    pipeline = Pipeline([("clf", LogisticRegression(max_iter=1000))])

    with pytest.raises(ValueError):
        with caplog.at_level(logging.ERROR):
            tune_models(
                {"bad_pipeline": pipeline},
                {"bad_pipeline": {"clf__not_a_parameter": [1]}},
                X_train,
                y_train,
                X_test,
                y_test,
                task=CLASSIFICATION_TASK,
                scoring="accuracy",
                n_splits=2,
                n_repeats=1,
                display=True,
                plot_validation_curves=False,
                plot_learning_curves=False,
                progress_bar=False,
            )

    assert "bad_pipeline" in caplog.text
