from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_io import project_path
from src.metrics import compute_regression_metrics
from src.project_data import make_train_val_test_dataframes
from src.result_schema import make_result_row, ordered_result_frame

try:
    import matgl
except ImportError:
    matgl = None


ALLOWED_TRAIN_FRACTIONS = [0.025, 0.25, 1.0]
MODEL_NAME = "matgl_scaffold"
MODEL_FAMILY = "matgl"
RESULT_FILE = "matgl.csv"
NOTES = "MatGL scaffold. Replace fallback baseline with pretrained MatGL inference/fine-tuning."


def fraction_to_name(train_fraction: float) -> str:
    return str(float(train_fraction)).replace(".", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MatGL starter scaffold.")
    parser.add_argument("--train-fraction", type=float, required=True, choices=ALLOWED_TRAIN_FRACTIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained-model", default="M3GNet-MP-2021.2.8-PES")
    return parser.parse_args()


def load_data(train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        return make_train_val_test_dataframes(train_fraction=train_fraction)
    except FileNotFoundError as exc:
        message = str(exc)
        if "dataset.pkl" in message:
            raise FileNotFoundError(
                "data/processed/dataset.pkl not found. Run python scripts/01_load_dataset.py first."
            ) from exc
        raise FileNotFoundError("Split files not found. Run python scripts/02_make_splits.py first.") from exc


def mean_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    train_mean = float(train_df["target"].mean())
    val_pred = np.full(shape=len(val_df), fill_value=train_mean)
    test_pred = np.full(shape=len(test_df), fill_value=train_mean)
    val_metrics = compute_regression_metrics(val_df["target"].to_numpy(), val_pred)
    return test_pred, val_metrics


def save_predictions(
    *,
    sample_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_fraction: float,
    seed: int,
) -> str:
    prediction_rel_path = f"results/predictions/{MODEL_NAME}_{fraction_to_name(train_fraction)}_seed{seed}.csv"
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


def save_result_row(row: dict[str, object]) -> Path:
    result_path = project_path("results", "raw", RESULT_FILE)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([row])
    if result_path.exists():
        old_df = pd.read_csv(result_path)
        duplicate = (
            (old_df["model"] == row["model"])
            & (old_df["train_fraction"].astype(float) == float(row["train_fraction"]))
            & (old_df["seed"].astype(int) == int(row["seed"]))
            & (old_df["split_id"] == row["split_id"])
        )
        new_df = pd.concat([old_df.loc[~duplicate], new_df], ignore_index=True)

    ordered_result_frame(new_df).to_csv(result_path, index=False)
    return result_path


def main() -> None:
    args = parse_args()
    train_df, val_df, test_df = load_data(args.train_fraction)

    # TODO for MatGL:
    # 1. Load pretrained MatGL / M3GNet / MEGNet model.
    # 2. Convert pymatgen Structure objects into MatGL dataset format.
    # 3. Run inference or fine-tuning.
    # 4. Use validation set for model selection.
    # 5. Keep final metrics on the fixed test set.
    if matgl is None:
        print("MatGL is not installed. Running dummy mean baseline inside MatGL scaffold.")
    else:
        print(f"MatGL is available. Pretrained model placeholder: {args.pretrained_model}")
        print("Full MatGL inference/fine-tuning is intentionally left as a TODO in this starter script.")

    y_pred, val_metrics = mean_baseline(train_df, val_df, test_df)
    y_true = test_df["target"].to_numpy()
    metrics = compute_regression_metrics(y_true, y_pred)
    predictions_path = save_predictions(
        sample_ids=test_df["sample_id"],
        y_true=y_true,
        y_pred=y_pred,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    result_row = make_result_row(
        model=MODEL_NAME,
        model_family=MODEL_FAMILY,
        train_fraction=args.train_fraction,
        seed=args.seed,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        predictions_path=predictions_path,
        notes=NOTES,
    )
    result_path = save_result_row(result_row)

    print(f"Loaded dataset: {len(train_df) + len(val_df) + len(test_df)} samples")
    print(f"Train fraction: {args.train_fraction}")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Validation MAE: {val_metrics['mae']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved result row to: {result_path}")


if __name__ == "__main__":
    main()
