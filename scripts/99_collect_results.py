from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_io import project_path
from src.result_schema import ordered_result_frame, validate_result_rows


def main() -> None:
    raw_dir = project_path("results", "raw")
    result_files = sorted(raw_dir.glob("*.csv"))
    if not result_files:
        raise FileNotFoundError("No result CSV files found in results/raw/")

    frames = []
    for path in result_files:
        df = pd.read_csv(path)
        try:
            validate_result_rows(df, source=path)
        except ValueError as exc:
            raise ValueError(f"Invalid result file {path}: {exc}") from exc
        frames.append(ordered_result_frame(df))

    summary = pd.concat(frames, ignore_index=True)
    summary = summary.sort_values(
        ["model_family", "model_name", "train_budget_samples", "budget_name", "model_seed"]
    ).reset_index(drop=True)

    output_path = project_path("results", "summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(summary)} rows")


if __name__ == "__main__":
    main()
