from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_io import load_config, project_path
from src.result_schema import ALLOWED_TRAIN_FRACTIONS, ordered_result_frame, validate_result_rows


REQUIRED_PREDICTION_COLUMNS = ["sample_id", "split", "y_true", "y_pred"]


def validate_prediction_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"predictions_path does not exist: {path}")

    df = pd.read_csv(path, nrows=5)
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {missing}")


def validate_result_file(path: Path, expected_split_id: str) -> int:
    df = pd.read_csv(path)
    validate_result_rows(df, source=path)
    ordered = ordered_result_frame(df)

    invalid_split_ids = set(ordered["split_id"]) - {expected_split_id}
    if invalid_split_ids:
        raise ValueError(f"{path} contains invalid split_id values: {sorted(invalid_split_ids)}")

    invalid_fractions = set(ordered["train_fraction"].astype(float)) - ALLOWED_TRAIN_FRACTIONS
    if invalid_fractions:
        raise ValueError(f"{path} contains invalid train_fraction values: {sorted(invalid_fractions)}")

    for rel_path in ordered["predictions_path"]:
        validate_prediction_file(project_path(*str(rel_path).split("/")))

    return len(ordered)


def main() -> None:
    config = load_config()
    raw_dir = project_path("results", "raw")
    result_files = sorted(raw_dir.glob("*.csv"))
    if not result_files:
        raise FileNotFoundError("No result CSV files found in results/raw/")

    total_rows = 0
    for path in result_files:
        try:
            total_rows += validate_result_file(path, expected_split_id=config["split_id"])
        except Exception as exc:
            raise ValueError(f"Invalid result file {path}: {exc}") from exc

    print(f"Validated {len(result_files)} result file(s), {total_rows} row(s)")


if __name__ == "__main__":
    main()
