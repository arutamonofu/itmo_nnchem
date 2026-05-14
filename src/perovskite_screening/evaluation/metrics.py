from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if root_mean_squared_error is None:
        rmse = mean_squared_error(y_true_array, y_pred_array, squared=False)
    else:
        rmse = root_mean_squared_error(y_true_array, y_pred_array)
    nonzero = y_true_array != 0
    if not nonzero.any():
        mape = float("nan")
    else:
        percentage_errors = np.abs((y_true_array[nonzero] - y_pred_array[nonzero]) / y_true_array[nonzero])
        mape = float(np.mean(percentage_errors) * 100)
    return {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true_array, y_pred_array)),
        "mape": mape,
    }
