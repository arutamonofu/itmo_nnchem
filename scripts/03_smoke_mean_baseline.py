from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_io import load_config, project_path
from src.metrics import compute_regression_metrics
from src.project_data import make_train_val_test_dataframes_for_budget
from src.result_schema import make_result_row, ordered_result_frame
from src.training_cli import add_budget_arguments, requested_budget_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mean baseline smoke test.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-strategy", default=None)
    add_budget_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    seed = int(config["random_seed"] if args.seed is None else args.seed)
    split_strategy = str(args.split_strategy or config.get("default_split_strategy", "random_iid"))

    rows = []
    args.split_strategy = split_strategy
    for budget_name in requested_budget_names(args):
        train_df, val_df, test_df = make_train_val_test_dataframes_for_budget(
            budget_name=budget_name,
            split_strategy=split_strategy,
        )
        train_mean = float(train_df["target"].mean())
        y_test = test_df["target"].to_numpy()
        y_pred = np.full(shape=len(test_df), fill_value=train_mean)
        metrics = compute_regression_metrics(y_test, y_pred)

        prediction_rel_path = (
            f"results/predictions/{split_strategy}_mean_baseline_{budget_name}_seed{seed}.csv"
        )
        prediction_path = project_path(*prediction_rel_path.split("/"))
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "sample_id": test_df["sample_id"].astype(int).to_numpy(),
                "split": "test",
                "y_true": y_test,
                "y_pred": y_pred,
            }
        ).to_csv(prediction_path, index=False)

        rows.append(
            make_result_row(
                model_name="mean_baseline",
                model_family="dummy",
                budget_name=budget_name,
                model_seed=seed,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
                predictions_path=prediction_rel_path,
                notes="Constant prediction equal to the selected train subset target mean.",
                split_strategy=split_strategy,
            )
        )

    result_df = ordered_result_frame(pd.DataFrame(rows))
    result_path = project_path("results", "raw", "mean_baseline.csv")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(result_path, index=False)
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
