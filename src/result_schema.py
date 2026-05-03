from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.project_data import split_metadata


REQUIRED_RESULT_COLUMNS = [
    "model",
    "model_family",
    "train_fraction",
    "seed",
    "split_id",
    "mae",
    "rmse",
    "r2",
    "n_train",
    "n_val",
    "n_test",
    "target_unit",
    "predictions_path",
    "notes",
]

ALLOWED_MODEL_FAMILIES = {"descriptor_baseline", "cgcnn", "matgl", "dummy"}
ALLOWED_TRAIN_FRACTIONS = {0.025, 0.25, 1.0}
USER_RESULT_COLUMNS = [
    "model",
    "model_family",
    "train_fraction",
    "seed",
    "mae",
    "rmse",
    "r2",
    "predictions_path",
    "notes",
]


def missing_result_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_RESULT_COLUMNS if col not in df.columns]


def validate_result_rows(df: pd.DataFrame, source: str | Path = "result rows") -> None:
    missing = missing_result_columns(df)
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")

    invalid_families = set(df["model_family"]) - ALLOWED_MODEL_FAMILIES
    if invalid_families:
        raise ValueError(f"{source} contains invalid model_family values: {sorted(invalid_families)}")

    invalid_fractions = set(df["train_fraction"].astype(float)) - ALLOWED_TRAIN_FRACTIONS
    if invalid_fractions:
        raise ValueError(f"{source} contains invalid train_fraction values: {sorted(invalid_fractions)}")


def ordered_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_result_rows(df)
    return df[REQUIRED_RESULT_COLUMNS]


def make_result_row(
    *,
    model: str,
    model_family: str,
    train_fraction: float,
    seed: int,
    mae: float,
    rmse: float,
    r2: float,
    predictions_path: str,
    notes: str = "",
) -> dict[str, object]:
    row = {
        "model": model,
        "model_family": model_family,
        "train_fraction": float(train_fraction),
        "seed": int(seed),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "predictions_path": predictions_path,
        "notes": notes,
    }
    row.update(split_metadata(train_fraction))
    return row
