from __future__ import annotations

from perovskite_screening.evaluation.metrics import compute_regression_metrics


def test_compute_regression_metrics() -> None:
    metrics = compute_regression_metrics([1.0, 2.0], [1.0, 3.0])
    assert metrics["mae"] == 0.5
    assert metrics["rmse"] > 0
    assert "r2" in metrics
    assert metrics["mape"] == 25.0
