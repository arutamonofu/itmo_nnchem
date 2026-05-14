from __future__ import annotations

import pandas as pd
import pytest

from perovskite_screening.evaluation import tail_metrics


def test_compute_tail_metrics_for_predictions_counts_overall_and_bins() -> None:
    predictions = pd.DataFrame(
        {
            "sample_id": [10, 20, 30, 40, 50, 60],
            "y_true": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y_pred": [0.0, 2.0, 2.0, 2.0, 6.0, 5.0],
        }
    )
    target_bins = pd.DataFrame(
        {
            "sample_id": [10, 20, 30, 40, 50, 60],
            "target_bin_10": ["low_10", "low_10", "middle_80", "middle_80", "high_10", "high_10"],
        }
    )

    result = tail_metrics.compute_tail_metrics_for_predictions(
        predictions,
        target_bins,
        schemes=("target_bin_10",),
    )

    by_bin = result.set_index("target_bin")
    assert by_bin.loc["overall", "n_samples"] == 6
    assert by_bin.loc["overall", "mae"] == pytest.approx(4 / 6)
    assert by_bin.loc["low_10", "n_samples"] == 2
    assert by_bin.loc["middle_80", "n_samples"] == 2
    assert by_bin.loc["high_10", "n_samples"] == 2


def test_compute_tail_metrics_for_predictions_keeps_empty_bin() -> None:
    predictions = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "y_true": [1.0, 2.0],
            "y_pred": [1.0, 2.5],
        }
    )
    target_bins = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "target_bin_10": ["low_10", "middle_80"],
        }
    )

    result = tail_metrics.compute_tail_metrics_for_predictions(
        predictions,
        target_bins,
        schemes=("target_bin_10",),
    )

    high = result.loc[result["target_bin"] == "high_10"].iloc[0]
    assert high["n_samples"] == 0
    assert pd.isna(high["mae"])


def test_compute_tail_metrics_for_predictions_joins_by_sample_id_not_row_order() -> None:
    predictions = pd.DataFrame(
        {
            "sample_id": [3, 1, 2],
            "y_true": [30.0, 10.0, 20.0],
            "y_pred": [33.0, 11.0, 22.0],
        }
    )
    target_bins = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "target_bin_10": ["low_10", "middle_80", "high_10"],
        }
    )

    result = tail_metrics.compute_tail_metrics_for_predictions(
        predictions,
        target_bins,
        schemes=("target_bin_10",),
    ).set_index("target_bin")

    assert result.loc["low_10", "mae"] == pytest.approx(1.0)
    assert result.loc["middle_80", "mae"] == pytest.approx(2.0)
    assert result.loc["high_10", "mae"] == pytest.approx(3.0)


def test_compute_tail_metrics_for_predictions_uses_shared_regression_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_compute_regression_metrics(y_true, y_pred):
        calls.append((list(y_true), list(y_pred)))
        return {"mae": 1.0, "rmse": 2.0, "r2": 3.0, "mape": 4.0}

    monkeypatch.setattr(tail_metrics, "compute_regression_metrics", fake_compute_regression_metrics)
    predictions = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.5, 2.5, 3.5],
        }
    )
    target_bins = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "target_bin_10": ["low_10", "middle_80", "high_10"],
        }
    )

    result = tail_metrics.compute_tail_metrics_for_predictions(
        predictions,
        target_bins,
        schemes=("target_bin_10",),
    )

    assert len(calls) == 4
    assert set(result["mae"]) == {1.0}


def test_compute_tail_metrics_from_result_table_skips_missing_prediction_file(tmp_path) -> None:
    result_table = pd.DataFrame(
        [
            {
                "model_name": "example",
                "model_family": "dummy",
                "budget_name": "B500",
                "model_seed": 42,
                "split_id": "random_iid_seed_42_80_10_10",
                "predictions_path": "missing.csv",
            }
        ]
    )
    target_bins = pd.DataFrame(
        {
            "sample_id": [1],
            "target_bin_10": ["low_10"],
            "target_bin_5": ["low_5"],
        }
    )

    with pytest.warns(UserWarning, match="prediction file does not exist"):
        result = tail_metrics.compute_tail_metrics_from_result_table(
            result_table,
            target_bins,
            project_root=tmp_path,
        )

    assert result.empty
    assert result.attrs["tail_metrics_summary"]["skipped"] == 1
