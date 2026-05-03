from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_io import load_config, project_path, save_json, write_structures_jsonl_gz


def infer_target_column(task, df: pd.DataFrame) -> str:
    metadata = getattr(task, "metadata", {})
    candidates = []
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("target"),
                metadata.get("target_name"),
                metadata.get("target_col"),
                metadata.get("output"),
            ]
        )
    else:
        candidates.extend(
            [
                getattr(metadata, "target", None),
                getattr(metadata, "target_name", None),
                getattr(metadata, "target_col", None),
                getattr(metadata, "output", None),
            ]
        )

    candidates.extend(["target", "e_form", "formation_energy"])
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    non_structure_cols = [col for col in df.columns if col != "structure"]
    numeric_cols = [col for col in non_structure_cols if pd.api.types.is_numeric_dtype(df[col])]
    if numeric_cols:
        return numeric_cols[-1]
    if non_structure_cols:
        return non_structure_cols[-1]
    raise ValueError("Could not infer target column from loaded matbench dataframe")


def load_matbench_perovskites(dataset_name: str) -> pd.DataFrame:
    try:
        from matminer.datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install dependencies first: pip install -r requirements.txt") from exc

    df = load_dataset(dataset_name).reset_index(drop=True)
    structure_col = "structure" if "structure" in df.columns else df.columns[0]
    target_col = infer_target_column(None, df)

    normalized = pd.DataFrame(
        {
            "sample_id": range(len(df)),
            "structure": df[structure_col],
            "target": df[target_col],
        }
    )
    return normalized


def main() -> None:
    config = load_config()
    dataset_name = config["dataset_name"]

    raw_path = project_path("data", "raw", f"{dataset_name}.pkl")
    dataset_path = project_path("data", "processed", "dataset.pkl")
    targets_path = project_path("data", "processed", "targets.csv")
    structures_path = project_path("data", "processed", "structures.json.gz")
    info_path = project_path("data", "processed", "dataset_info.json")

    df = load_matbench_perovskites(dataset_name)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_pickle(raw_path)
    if raw_path != dataset_path:
        shutil.copyfile(raw_path, dataset_path)

    df[["sample_id", "target"]].to_csv(targets_path, index=False)
    write_structures_jsonl_gz(df, structures_path)

    save_json(
        {
            "dataset_name": dataset_name,
            "n_samples": int(len(df)),
            "columns": ["sample_id", "structure", "target"],
            "target_name": "target",
            "target_unit": config["target_unit"],
        },
        info_path,
    )

    print(f"Loaded {dataset_name}: {len(df)} samples")
    print(f"Wrote {dataset_path}")


if __name__ == "__main__":
    main()
