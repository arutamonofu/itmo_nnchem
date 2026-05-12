from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_io import load_config, project_path, read_json
from src.budgets import (
    build_budgets_metadata,
    budget_record,
    load_budgets_metadata,
    parse_training_budget_config,
    write_budgets_metadata,
)
from src.splits import SPLIT_NAMES


DEFAULT_SPLIT_STRATEGY = "random_iid"


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


def _split_dir(split_strategy: str) -> Path:
    return project_path("data", "splits", split_strategy)


def _budgets_path(split_strategy: str) -> Path:
    return _split_dir(split_strategy) / "budgets.json"


def resolve_split_strategy(split_strategy: str | None = None) -> str:
    if split_strategy is not None:
        return split_strategy
    config = load_config()
    return str(config.get("default_split_strategy", DEFAULT_SPLIT_STRATEGY))


def load_split_assignment(split_strategy: str | None = None) -> pd.DataFrame:
    split_strategy = resolve_split_strategy(split_strategy)
    frames = []
    split_dir = _split_dir(split_strategy)
    for split in SPLIT_NAMES:
        path = split_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        frame = pd.read_csv(path)
        expected_columns = ["sample_id", "split", "split_strategy"]
        if list(frame.columns) != expected_columns:
            raise ValueError(f"{path} must contain columns: {expected_columns}")
        invalid_splits = set(frame["split"]) - {split}
        if invalid_splits:
            raise ValueError(f"{path} contains rows for other splits: {sorted(invalid_splits)}")
        invalid_strategies = set(frame["split_strategy"]) - {split_strategy}
        if invalid_strategies:
            raise ValueError(f"{path} contains invalid split_strategy values: {sorted(invalid_strategies)}")
        frames.append(frame)

    assignment = pd.concat(frames, ignore_index=True)
    if assignment["sample_id"].duplicated().any():
        duplicated = sorted(assignment.loc[assignment["sample_id"].duplicated(), "sample_id"].unique())[:10]
        raise ValueError(f"{split_strategy} contains duplicated sample_id values: {duplicated}")

    return assignment.astype({"sample_id": int})


def load_split_ids(split_strategy: str | None = None) -> tuple[list[int], list[int], list[int]]:
    split_strategy = resolve_split_strategy(split_strategy)
    assignment = load_split_assignment(split_strategy=split_strategy)
    return tuple(
        assignment.loc[assignment["split"] == split, "sample_id"].astype(int).tolist()
        for split in SPLIT_NAMES
    )


def ensure_budgets_metadata(split_strategy: str | None = None) -> dict[str, object]:
    split_strategy = resolve_split_strategy(split_strategy)
    path = _budgets_path(split_strategy)
    if path.exists():
        return load_budgets_metadata(path)

    config = load_config()
    train_ids, _, _ = load_split_ids(split_strategy=split_strategy)
    budget_config = parse_training_budget_config(config)
    metadata = build_budgets_metadata(
        train_ids=train_ids,
        split_seed=int(config["random_seed"]),
        base_budget=int(budget_config["base_budget"]),
        growth_factor=int(budget_config["growth_factor"]),
        min_final_growth_ratio=float(budget_config["min_final_growth_ratio"]),
        sampling_strategy=str(budget_config["sampling_strategy"]),
    )
    write_budgets_metadata(metadata, path)
    return metadata


def load_budget_ids(
    budget_name: str,
    *,
    split_strategy: str | None = None,
) -> list[int]:
    metadata = ensure_budgets_metadata(split_strategy=split_strategy)
    record = budget_record(metadata, budget_name)
    indices = record.get("indices")
    if not isinstance(indices, list):
        raise ValueError(f"Budget {budget_name!r} is missing indices")
    return [int(sample_id) for sample_id in indices]


def make_train_val_test_dataframes_for_budget(
    budget_name: str,
    *,
    split_strategy: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_strategy = resolve_split_strategy(split_strategy)
    df = load_dataset(index_by_sample_id=True)
    _, val_ids, test_ids = load_split_ids(split_strategy=split_strategy)
    train_ids = load_budget_ids(budget_name, split_strategy=split_strategy)
    return df.loc[train_ids].copy(), df.loc[val_ids].copy(), df.loc[test_ids].copy()


def budget_metadata(
    budget_name: str,
    *,
    split_strategy: str | None = None,
) -> dict[str, int | float | str]:
    split_strategy = resolve_split_strategy(split_strategy)
    metadata = ensure_budgets_metadata(split_strategy=split_strategy)
    record = budget_record(metadata, budget_name)
    config = load_config()
    metadata_path = _split_dir(split_strategy) / "split_metadata.json"
    split_file_metadata = read_json(metadata_path) if metadata_path.exists() else {}
    split_id = split_file_metadata.get(
        "split_id",
        f"{split_strategy}_seed_{int(config['random_seed'])}_80_10_10",
    )
    train_budget_samples = int(record["n_samples"])
    full_train_size = int(metadata["full_train_size"])
    return {
        "split_id": str(split_id),
        "n_train": train_budget_samples,
        "n_val": int(split_file_metadata.get("val_size", 0)) or len(load_split_ids(split_strategy)[1]),
        "n_test": int(split_file_metadata.get("test_size", 0)) or len(load_split_ids(split_strategy)[2]),
        "target_unit": str(config["target_unit"]),
        "budget_name": budget_name,
        "train_budget_samples": train_budget_samples,
        "train_fraction_actual": float(train_budget_samples / full_train_size),
        "full_train_size": full_train_size,
        "budget_strategy": str(metadata["budget_strategy"]),
        "split_seed": int(metadata["split_seed"]),
    }
