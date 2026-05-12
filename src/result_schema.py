from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.project_data import budget_metadata


REQUIRED_RESULT_COLUMNS = [
    "model_name",
    "model_family",
    "budget_name",
    "train_budget_samples",
    "train_fraction_actual",
    "full_train_size",
    "budget_strategy",
    "split_seed",
    "model_seed",
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


def missing_result_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_RESULT_COLUMNS if col not in df.columns]


def validate_result_rows(df: pd.DataFrame, source: str | Path = "result rows") -> None:
    missing = missing_result_columns(df)
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")

    invalid_families = set(df["model_family"]) - ALLOWED_MODEL_FAMILIES
    if invalid_families:
        raise ValueError(f"{source} contains invalid model_family values: {sorted(invalid_families)}")

    if (df["train_budget_samples"].astype(int) <= 0).any():
        raise ValueError(f"{source} contains non-positive train_budget_samples")

    if (df["full_train_size"].astype(int) <= 0).any():
        raise ValueError(f"{source} contains non-positive full_train_size")

    if (df["train_fraction_actual"].astype(float) <= 0).any():
        raise ValueError(f"{source} contains non-positive train_fraction_actual")

    if df["budget_name"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{source} contains empty budget_name values")


def ordered_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_result_rows(df)
    return df[REQUIRED_RESULT_COLUMNS]


def make_result_row(
    *,
    model_name: str,
    model_family: str,
    budget_name: str,
    model_seed: int,
    mae: float,
    rmse: float,
    r2: float,
    predictions_path: str,
    notes: str = "",
    split_strategy: str | None = None,
) -> dict[str, object]:
    row = {
        "model_name": model_name,
        "model_family": model_family,
        "model_seed": int(model_seed),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "predictions_path": predictions_path,
        "notes": notes,
    }
    row.update(budget_metadata(budget_name, split_strategy=split_strategy))
    return row
