from __future__ import annotations

from pathlib import Path

from perovskite_screening.io.paths import project_path


def run_artifact_stem(
    *,
    split_strategy: str,
    model_name: str,
    budget_name: str,
    seed: int,
) -> str:
    return f"{split_strategy}_{model_name}_{budget_name}_seed{seed}"


def prediction_rel_path(stem: str) -> str:
    return f"outputs/runs/predictions/{stem}.csv"


def history_file_rel_path(stem: str) -> str:
    return f"outputs/runs/history/{stem}.csv"


def history_dir_rel_path(stem: str) -> str:
    return f"outputs/runs/history/{stem}"


def model_file_rel_path(*, model_family: str, stem: str, suffix: str) -> str:
    return f"outputs/models/{model_family}/{stem}{suffix}"


def model_dir_rel_path(*, model_family: str, stem: str) -> str:
    return f"outputs/models/{model_family}/{stem}"


def cache_dir_rel_path(*, cache_family: str, stem: str, partition: str | None = None) -> str:
    rel_path = f"outputs/cache/{cache_family}/{stem}"
    if partition is not None:
        rel_path = f"{rel_path}/{partition}"
    return rel_path


def project_rel_path(rel_path: str) -> Path:
    return project_path(*rel_path.split("/"))
