from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_io import project_path
from src.result_schema import ordered_result_frame, validate_result_rows


REQUIRED_PREDICTION_COLUMNS = ["sample_id", "split", "y_true", "y_pred"]


def is_empty_prediction_path(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def validate_prediction_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"predictions_path does not exist: {path}")

    df = pd.read_csv(path, nrows=5)
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {missing}")


def validate_result_file(path: Path) -> int:
    df = pd.read_csv(path)
    validate_result_rows(df, source=path)
    ordered = ordered_result_frame(df)

    invalid_budget_names = ordered["budget_name"].isna() | (ordered["budget_name"].astype(str).str.strip() == "")
    if invalid_budget_names.any():
        raise ValueError(f"{path} contains empty budget_name values")

    invalid_budget_sizes = ordered["train_budget_samples"].astype(int) <= 0
    if invalid_budget_sizes.any():
        raise ValueError(f"{path} contains non-positive train_budget_samples")

    for rel_path in ordered["predictions_path"]:
        if is_empty_prediction_path(rel_path):
            continue
        validate_prediction_file(project_path(*str(rel_path).split("/")))

    return len(ordered)


def main() -> None:
    raw_dir = project_path("results", "raw")
    result_files = sorted(raw_dir.glob("*.csv"))
    if not result_files:
        raise FileNotFoundError("No result CSV files found in results/raw/")

    total_rows = 0
    for path in result_files:
        try:
            total_rows += validate_result_file(path)
        except Exception as exc:
            raise ValueError(f"Invalid result file {path}: {exc}") from exc

    print(f"Validated {len(result_files)} result file(s), {total_rows} row(s)")


if __name__ == "__main__":
    main()
