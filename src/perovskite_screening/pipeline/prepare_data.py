from __future__ import annotations

from perovskite_screening.config import ProjectConfig
from perovskite_screening.data.dataset import load_matbench_perovskites, write_dataset_artifacts


def prepare_data(config: ProjectConfig) -> dict[str, object]:
    dataset_name = str(config.raw.get("data", {}).get("dataset_name", "matbench_perovskites"))
    df = load_matbench_perovskites(dataset_name)
    paths = write_dataset_artifacts(df, config)
    return {"n_samples": int(len(df)), "paths": paths}
