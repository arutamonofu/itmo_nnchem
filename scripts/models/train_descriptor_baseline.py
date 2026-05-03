from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_io import project_path
from src.metrics import compute_regression_metrics
from src.project_data import make_train_val_test_dataframes
from src.result_schema import make_result_row, ordered_result_frame


ALLOWED_TRAIN_FRACTIONS = [0.025, 0.25, 1.0]
MODEL_NAME = "descriptor_rf"
MODEL_FAMILY = "descriptor_baseline"
RESULT_FILE = "descriptor_baseline.csv"
NOTES = "RandomForest on simple structure/composition features."


def fraction_to_name(train_fraction: float) -> str:
    return str(float(train_fraction)).replace(".", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descriptor RandomForest baseline.")
    parser.add_argument("--train-fraction", type=float, required=True, choices=ALLOWED_TRAIN_FRACTIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["rf"], default="rf")
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


def structure_features(structure) -> dict[str, float]:
    composition = structure.composition
    atomic_numbers = np.array([element.Z for element in composition.elements], dtype=float)
    return {
        "n_sites": float(structure.num_sites),
        "volume": float(structure.volume),
        "density": float(structure.density),
        "num_unique_elements": float(len(composition.elements)),
        "mean_atomic_number": float(atomic_numbers.mean()),
        "std_atomic_number": float(atomic_numbers.std()),
    }


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([structure_features(structure) for structure in df["structure"]], index=df.index)


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


def print_summary(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_fraction: float,
    metrics: dict[str, float],
    val_metrics: dict[str, float],
    predictions_path: str,
    result_path: Path,
) -> None:
    print(f"Loaded dataset: {len(train_df) + len(val_df) + len(test_df)} samples")
    print(f"Train fraction: {train_fraction}")
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


def main() -> None:
    args = parse_args()
    train_df, val_df, test_df = load_data(args.train_fraction)

    x_train = featurize(train_df)
    x_val = featurize(val_df)
    x_test = featurize(test_df)
    y_train = train_df["target"].to_numpy()
    y_val = val_df["target"].to_numpy()
    y_true = test_df["target"].to_numpy()

    model = RandomForestRegressor(n_estimators=100, random_state=args.seed, n_jobs=-1)
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    y_pred = model.predict(x_test)
    val_metrics = compute_regression_metrics(y_val, val_pred)
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
    print_summary(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_fraction=args.train_fraction,
        metrics=metrics,
        val_metrics=val_metrics,
        predictions_path=predictions_path,
        result_path=result_path,
    )


if __name__ == "__main__":
    main()
