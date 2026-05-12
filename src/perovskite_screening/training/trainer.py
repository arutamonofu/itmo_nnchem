from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from perovskite_screening.config import ProjectConfig
from perovskite_screening.data.budgets import load_budget_metadata, resolve_budget
from perovskite_screening.data.dataset import load_dataset
from perovskite_screening.data.splits import SPLIT_NAMES, load_split
from perovskite_screening.io.paths import project_path


@dataclass(frozen=True)
class ExperimentData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_experiment_data(
    *,
    budget_name: str,
    split_strategy: str,
    config: ProjectConfig,
) -> ExperimentData:
    data_config = config.raw.get("data", {})
    dataset_path = project_path(data_config.get("processed_path", "data/processed/dataset.pkl"))
    df = load_dataset(dataset_path, index_by_sample_id=True)
    assignment = load_split(split_strategy)
    ids_by_split = {
        split: assignment.loc[assignment["split"] == split, "sample_id"].astype(int).tolist()
        for split in SPLIT_NAMES
    }
    metadata = load_budget_metadata(project_path("data", "splits", split_strategy, "budgets.json"))
    train_ids = [int(sample_id) for sample_id in resolve_budget(metadata, budget_name)["indices"]]
    return ExperimentData(
        train=df.loc[train_ids].copy(),
        val=df.loc[ids_by_split["val"]].copy(),
        test=df.loc[ids_by_split["test"]].copy(),
    )
