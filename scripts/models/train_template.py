from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_io import project_path
from src.metrics import compute_regression_metrics
from src.project_data import make_train_val_test_dataframes_for_budget
from src.result_schema import make_result_row, ordered_result_frame
from src.training_cli import add_budget_arguments, requested_budget_names


ALLOWED_SPLIT_STRATEGIES = ["random_iid", "element_set"]
MODEL_NAME = "template_mean"
MODEL_FAMILY = "dummy"
RESULT_FILE = "template.csv"
PREDICTION_PREFIX = "template"
NOTES = "Template script: mean target baseline. Replace training block with a real model."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal template for project training scripts.")
    add_budget_arguments(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-strategy", choices=ALLOWED_SPLIT_STRATEGIES, default="random_iid")
    return parser.parse_args()


def load_data(budget_name: str, split_strategy: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        return make_train_val_test_dataframes_for_budget(
            budget_name=budget_name,
            split_strategy=split_strategy,
        )
    except FileNotFoundError as exc:
        message = str(exc)
        if "dataset.pkl" in message:
            raise FileNotFoundError(
                "data/processed/dataset.pkl not found. Run python scripts/01_load_dataset.py first."
            ) from exc
        raise FileNotFoundError("Split files not found. Run python scripts/02_make_splits.py first.") from exc


def save_predictions(
    *,
    sample_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    budget_name: str,
    seed: int,
    split_strategy: str,
) -> str:
    prediction_rel_path = (
        f"results/predictions/{split_strategy}_{PREDICTION_PREFIX}_{budget_name}_seed{seed}.csv"
    )
    prediction_path = project_path(*prediction_rel_path.split("/"))
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": sample_ids.astype(int).to_numpy(),
            "split": "test",
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(prediction_path, index=False)
    return prediction_rel_path


def save_result_row(row: dict[str, object], result_file: str) -> Path:
    result_path = project_path("results", "raw", result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([row])
    if result_path.exists():
        old_df = pd.read_csv(result_path)
        duplicate = (
            (old_df["model_name"] == row["model_name"])
            & (old_df["budget_name"] == row["budget_name"])
            & (old_df["model_seed"].astype(int) == int(row["model_seed"]))
            & (old_df["split_id"] == row["split_id"])
        )
        new_df = pd.concat([old_df.loc[~duplicate], new_df], ignore_index=True)

    ordered_result_frame(new_df).to_csv(result_path, index=False)
    return result_path


def print_summary(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    budget_name: str,
    model: str,
    metrics: dict[str, float],
    predictions_path: str,
    result_path: Path,
) -> None:
    print(f"Loaded dataset: {len(train_df) + len(val_df) + len(test_df)} samples")
    print(f"Budget: {budget_name}")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Model: {model}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved result row to: {result_path}")


def run_one_budget(args: argparse.Namespace, budget_name: str) -> None:
    train_df, val_df, test_df = load_data(budget_name, args.split_strategy)

    # TODO: Replace this block with real model training.
    train_mean = float(train_df["target"].mean())
    _ = val_df

    # TODO: Replace this block with real model inference.
    y_true = test_df["target"].to_numpy()
    y_pred = np.full(shape=len(test_df), fill_value=train_mean)

    # TODO: Update model name and model_family.
    metrics = compute_regression_metrics(y_true, y_pred)
    predictions_path = save_predictions(
        sample_ids=test_df["sample_id"],
        y_true=y_true,
        y_pred=y_pred,
        budget_name=budget_name,
        seed=args.seed,
        split_strategy=args.split_strategy,
    )
    result_row = make_result_row(
        model_name=MODEL_NAME,
        model_family=MODEL_FAMILY,
        budget_name=budget_name,
        model_seed=args.seed,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        predictions_path=predictions_path,
        notes=NOTES,
        split_strategy=args.split_strategy,
    )
    result_path = save_result_row(result_row, RESULT_FILE)
    print_summary(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        budget_name=budget_name,
        model=MODEL_NAME,
        metrics=metrics,
        predictions_path=predictions_path,
        result_path=result_path,
    )


def main() -> None:
    args = parse_args()
    for budget_name in requested_budget_names(args):
        run_one_budget(args, budget_name)


if __name__ == "__main__":
    main()
