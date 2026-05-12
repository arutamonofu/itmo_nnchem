from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from perovskite_screening.config import ProjectConfig
from perovskite_screening.io.artifacts import save_json, write_structures_jsonl_gz
from perovskite_screening.io.paths import project_path


REQUIRED_DATASET_COLUMNS = {"sample_id", "structure", "target"}


def validate_dataset(df: pd.DataFrame, source: str | Path = "dataset") -> None:
    missing = REQUIRED_DATASET_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{source} contains duplicated sample_id values")


def load_dataset(
    path: str | Path | None = None,
    *,
    index_by_sample_id: bool = True,
) -> pd.DataFrame:
    dataset_path = Path(path) if path is not None else project_path("data", "processed", "dataset.pkl")
    if not dataset_path.is_absolute():
        dataset_path = project_path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing canonical dataset: {dataset_path}. "
            "Run `perovskite-screening prepare-data` first."
        )
    df = pd.read_pickle(dataset_path)
    validate_dataset(df, dataset_path)
    if index_by_sample_id:
        return df.set_index("sample_id", drop=False)
    return df


def infer_target_column(df: pd.DataFrame) -> str:
    for candidate in ("target", "e_form", "formation_energy"):
        if candidate in df.columns:
            return candidate
    non_structure_cols = [col for col in df.columns if col != "structure"]
    numeric_cols = [col for col in non_structure_cols if pd.api.types.is_numeric_dtype(df[col])]
    if numeric_cols:
        return numeric_cols[-1]
    if non_structure_cols:
        return non_structure_cols[-1]
    raise ValueError("Could not infer target column from loaded dataframe")


def raw_dataset_path(dataset_name: str) -> Path:
    return project_path("data", "raw", f"{dataset_name}.pkl")


def normalize_matbench_dataframe(df: pd.DataFrame, source: str | Path) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    if REQUIRED_DATASET_COLUMNS.issubset(df.columns):
        normalized = df[["sample_id", "structure", "target"]].copy()
    else:
        structure_col = "structure" if "structure" in df.columns else df.columns[0]
        target_col = infer_target_column(df)
        normalized = pd.DataFrame(
            {
                "sample_id": range(len(df)),
                "structure": df[structure_col],
                "target": df[target_col],
            }
        )
    validate_dataset(normalized, source)
    return normalized


def load_matbench_perovskites(dataset_name: str, *, prefer_cache: bool = True) -> pd.DataFrame:
    raw_path = raw_dataset_path(dataset_name)
    if prefer_cache and raw_path.exists():
        return normalize_matbench_dataframe(pd.read_pickle(raw_path), raw_path)

    try:
        from matminer.datasets import load_dataset as load_matminer_dataset
    except ImportError as exc:
        if raw_path.exists():
            return normalize_matbench_dataframe(pd.read_pickle(raw_path), raw_path)
        raise ImportError(
            "Install data dependencies first with `pip install -e .` or provide "
            f"a local raw cache at {raw_path}."
        ) from exc

    try:
        df = load_matminer_dataset(dataset_name)
    except Exception:
        if raw_path.exists():
            return normalize_matbench_dataframe(pd.read_pickle(raw_path), raw_path)
        raise
    return normalize_matbench_dataframe(df, dataset_name)


def write_dataset_artifacts(df: pd.DataFrame, config: ProjectConfig) -> dict[str, Path]:
    data_config = config.raw.get("data", {})
    dataset_name = str(data_config.get("dataset_name", "matbench_perovskites"))
    target_unit = str(data_config.get("target_unit", "eV/unit cell"))
    processed_path = project_path(data_config.get("processed_path", "data/processed/dataset.pkl"))
    raw_path = raw_dataset_path(dataset_name)
    targets_path = processed_path.parent / "targets.csv"
    structures_path = processed_path.parent / "structures.json.gz"
    info_path = processed_path.parent / "dataset_info.json"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(raw_path)
    if raw_path != processed_path:
        shutil.copyfile(raw_path, processed_path)
    df[["sample_id", "target"]].to_csv(targets_path, index=False)
    write_structures_jsonl_gz(df, structures_path)
    save_json(
        {
            "dataset_name": dataset_name,
            "n_samples": int(len(df)),
            "columns": ["sample_id", "structure", "target"],
            "target_name": "target",
            "target_unit": target_unit,
        },
        info_path,
    )
    return {
        "raw": raw_path,
        "processed": processed_path,
        "targets": targets_path,
        "structures": structures_path,
        "info": info_path,
    }
