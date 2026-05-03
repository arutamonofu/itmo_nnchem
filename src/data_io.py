from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.json"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_indices(sample_ids: list[int], path: Path) -> None:
    ensure_parent(path)
    pd.DataFrame({"sample_id": sample_ids}).to_csv(path, index=False)


def load_indices(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != ["sample_id"]:
        raise ValueError(f"{path} must contain exactly one column: sample_id")
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated sample_id values")
    return df["sample_id"].astype(int).tolist()


def write_structures_jsonl_gz(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            record = {
                "sample_id": int(row.sample_id),
                "structure": row.structure.as_dict(),
                "target": float(row.target),
            }
            f.write(json.dumps(record) + "\n")
