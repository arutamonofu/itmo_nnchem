from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from perovskite_screening.io.paths import ensure_parent


def save_json(data: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)
