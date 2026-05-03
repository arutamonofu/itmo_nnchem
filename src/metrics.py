from __future__ import annotations

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # Older scikit-learn versions.
    root_mean_squared_error = None


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    if root_mean_squared_error is None:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    else:
        rmse = root_mean_squared_error(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }
