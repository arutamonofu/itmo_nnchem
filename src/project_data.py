from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_io import load_config, load_indices, project_path


FRACTION_TO_SPLIT_FILE = {
    0.025: "train_2_5_indices.csv",
    0.25: "train_25_indices.csv",
    1.0: "train_100_indices.csv",
}


def _normalize_train_fraction(train_fraction: float) -> float:
    fraction = float(train_fraction)
    for allowed in FRACTION_TO_SPLIT_FILE:
        if abs(fraction - allowed) < 1e-12:
            return allowed
    allowed_values = sorted(FRACTION_TO_SPLIT_FILE)
    raise ValueError(f"train_fraction must be one of {allowed_values}; got {train_fraction}")


def load_dataset(path: Path | None = None, *, index_by_sample_id: bool = True) -> pd.DataFrame:
    dataset_path = path or project_path("data", "processed", "dataset.pkl")
    if not dataset_path.exists():
        raise FileNotFoundError("Missing data/processed/dataset.pkl. Run scripts/01_load_dataset.py first.")

    df = pd.read_pickle(dataset_path)
    required_columns = {"sample_id", "structure", "target"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_path} is missing required columns: {sorted(missing)}")

    if index_by_sample_id:
        return df.set_index("sample_id", drop=False)
    return df


def load_split_ids(train_fraction: float) -> tuple[list[int], list[int], list[int]]:
    fraction = _normalize_train_fraction(train_fraction)
    split_dir = project_path("data", "splits")
    train_ids = load_indices(split_dir / FRACTION_TO_SPLIT_FILE[fraction])
    val_ids = load_indices(split_dir / "val_indices.csv")
    test_ids = load_indices(split_dir / "test_indices.csv")
    return train_ids, val_ids, test_ids


def make_train_val_test_dataframes(train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_dataset(index_by_sample_id=True)
    train_ids, val_ids, test_ids = load_split_ids(train_fraction)
    return df.loc[train_ids].copy(), df.loc[val_ids].copy(), df.loc[test_ids].copy()


def split_metadata(train_fraction: float) -> dict[str, int | str]:
    train_df, val_df, test_df = make_train_val_test_dataframes(train_fraction)
    config = load_config()
    return {
        "split_id": config["split_id"],
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "target_unit": config["target_unit"],
    }
