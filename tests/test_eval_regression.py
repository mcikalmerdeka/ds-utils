import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from utils.regression_evals_and_tuning import eval_regression


def test_eval_regression_returns_expected_metrics():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)

    metrics = eval_regression(
        model=model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        plot_predicted_vs_actual=False,
        plot_residuals=False,
        plot_residuals_dist=False,
        n_splits=2,
        n_repeats=1,
    )

    assert {"mae", "mse", "rmse", "mape", "r2"} == set(metrics["cv"])
    for metric in ("mae", "mse", "rmse", "mape"):
        assert metrics[metric]["test"] >= 0
        assert np.isfinite(metrics["cv"][metric]["test_mean"])
    assert np.isfinite(metrics["r2"]["test"])
