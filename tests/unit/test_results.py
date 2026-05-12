from __future__ import annotations

import pandas as pd

from perovskite_screening.io.results import REQUIRED_RESULT_COLUMNS, ordered_result_frame


def test_ordered_result_frame_accepts_required_columns() -> None:
    row = {
        "model_name": "mean_baseline",
        "model_family": "dummy",
        "budget_name": "B500",
        "train_budget_samples": 500,
        "train_fraction_actual": 0.1,
        "full_train_size": 5000,
        "budget_strategy": "absolute_doubling_budgets",
        "split_seed": 42,
        "model_seed": 42,
        "split_id": "random_iid_seed_42_80_10_10",
        "mae": 1.0,
        "rmse": 1.2,
        "r2": 0.0,
        "n_train": 500,
        "n_val": 100,
        "n_test": 100,
        "target_unit": "eV/unit cell",
        "predictions_path": "",
        "notes": "",
    }
    assert list(ordered_result_frame(pd.DataFrame([row])).columns) == REQUIRED_RESULT_COLUMNS
