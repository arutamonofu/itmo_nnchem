from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from perovskite_screening.config import ProjectConfig
from perovskite_screening.data.budgets import load_budget_metadata, resolve_budget
from perovskite_screening.data.splits import SPLIT_NAMES, load_split, load_split_metadata
from perovskite_screening.io.paths import ensure_parent, project_path


REQUIRED_RESULT_COLUMNS = [
    "model_name",
    "model_family",
    "budget_name",
    "train_budget_samples",
    "train_fraction_actual",
    "full_train_size",
    "budget_strategy",
    "split_seed",
    "model_seed",
    "split_id",
    "mae",
    "rmse",
    "r2",
    "mape",
    "n_train",
    "n_val",
    "n_test",
    "target_unit",
    "predictions_path",
    "model_path",
    "history_path",
    "model_params_json",
    "notes",
]
REQUIRED_PREDICTION_COLUMNS = ["sample_id", "split", "y_true", "y_pred"]
ALLOWED_MODEL_FAMILIES = {"descriptor_baseline", "cgcnn", "matgl", "dummy"}


def model_params_to_json(model_params: dict[str, object] | None) -> str:
    return json.dumps(model_params or {}, sort_keys=True, separators=(",", ":"))


def normalize_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def validate_result_rows(df: pd.DataFrame, source: str | Path = "result rows") -> None:
    df = normalize_result_frame(df)
    missing = [col for col in REQUIRED_RESULT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    invalid_families = set(df["model_family"]) - ALLOWED_MODEL_FAMILIES
    if invalid_families:
        raise ValueError(f"{source} contains invalid model_family values: {sorted(invalid_families)}")
    if (df["train_budget_samples"].astype(int) <= 0).any():
        raise ValueError(f"{source} contains non-positive train_budget_samples")
    if (df["full_train_size"].astype(int) <= 0).any():
        raise ValueError(f"{source} contains non-positive full_train_size")
    if (df["train_fraction_actual"].astype(float) <= 0).any():
        raise ValueError(f"{source} contains non-positive train_fraction_actual")
    if df["budget_name"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{source} contains empty budget_name values")
    for value in df["model_params_json"].fillna("{}"):
        try:
            decoded = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} contains invalid model_params_json: {value!r}") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"{source} contains non-object model_params_json: {value!r}")


def ordered_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_result_frame(df)
    validate_result_rows(normalized)
    return normalized[REQUIRED_RESULT_COLUMNS]


def budget_result_metadata(
    budget_name: str,
    *,
    split_strategy: str,
    config: ProjectConfig,
) -> dict[str, int | float | str]:
    split_dir = project_path("data", "splits")
    budget_metadata = load_budget_metadata(split_dir / split_strategy / "budgets.json")
    record = resolve_budget(budget_metadata, budget_name)
    split_file_metadata = load_split_metadata(split_strategy, split_dir=split_dir)
    assignment = load_split(split_strategy, split_dir=split_dir)
    counts = assignment["split"].value_counts()
    train_budget_samples = int(record["n_samples"])
    full_train_size = int(budget_metadata["full_train_size"])
    return {
        "split_id": str(split_file_metadata.get("split_id", f"{split_strategy}_seed_{config.random_seed}_80_10_10")),
        "n_train": train_budget_samples,
        "n_val": int(split_file_metadata.get("val_size", 0)) or int(counts.get("val", 0)),
        "n_test": int(split_file_metadata.get("test_size", 0)) or int(counts.get("test", 0)),
        "target_unit": config.target_unit,
        "budget_name": budget_name,
        "train_budget_samples": train_budget_samples,
        "train_fraction_actual": float(train_budget_samples / full_train_size),
        "full_train_size": full_train_size,
        "budget_strategy": str(budget_metadata["budget_strategy"]),
        "split_seed": int(budget_metadata["split_seed"]),
    }


def make_result_row(
    *,
    model_name: str,
    model_family: str,
    budget_name: str,
    model_seed: int,
    mae: float,
    rmse: float,
    r2: float,
    mape: float,
    predictions_path: str,
    split_strategy: str,
    config: ProjectConfig,
    model_path: str = "",
    history_path: str = "",
    model_params: dict[str, object] | None = None,
    notes: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_name": model_name,
        "model_family": model_family,
        "model_seed": int(model_seed),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "predictions_path": predictions_path,
        "model_path": model_path,
        "history_path": history_path,
        "model_params_json": model_params_to_json(model_params),
        "notes": notes,
    }
    row.update(budget_result_metadata(budget_name, split_strategy=split_strategy, config=config))
    return row


def upsert_result_row(row: dict[str, object], path: Path) -> Path:
    ensure_parent(path)
    new_df = pd.DataFrame([row])
    if path.exists():
        old_df = pd.read_csv(path)
        duplicate = (
            (old_df["model_name"] == row["model_name"])
            & (old_df["budget_name"] == row["budget_name"])
            & (old_df["model_seed"].astype(int) == int(row["model_seed"]))
            & (old_df["split_id"] == row["split_id"])
        )
        new_df = pd.concat([old_df.loc[~duplicate], new_df], ignore_index=True)
    ordered_result_frame(new_df).to_csv(path, index=False)
    return path


def validate_prediction_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"predictions_path does not exist: {path}")
    df = pd.read_csv(path, nrows=5)
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {missing}")


def validate_results(runs_dir: Path) -> int:
    result_files = sorted(runs_dir.glob("*.csv"))
    if not result_files:
        raise FileNotFoundError(f"No result CSV files found in {runs_dir}/")
    total_rows = 0
    for path in result_files:
        df = pd.read_csv(path)
        validate_result_rows(df, source=path)
        ordered = ordered_result_frame(df)
        for rel_path in ordered["predictions_path"]:
            if pd.isna(rel_path) or str(rel_path).strip() == "":
                continue
            validate_prediction_file(project_path(*str(rel_path).split("/")))
        for rel_path in ordered["model_path"]:
            if pd.isna(rel_path) or str(rel_path).strip() == "":
                continue
            artifact_path = project_path(*str(rel_path).split("/"))
            if not artifact_path.exists():
                raise FileNotFoundError(f"model_path does not exist: {artifact_path}")
        for rel_path in ordered["history_path"]:
            if pd.isna(rel_path) or str(rel_path).strip() == "":
                continue
            history_path = project_path(*str(rel_path).split("/"))
            if not history_path.exists():
                raise FileNotFoundError(f"history_path does not exist: {history_path}")
        total_rows += len(ordered)
    return total_rows


def collect_results(runs_dir: Path, output_path: Path) -> pd.DataFrame:
    result_files = sorted(runs_dir.glob("*.csv"))
    if not result_files:
        raise FileNotFoundError(f"No result CSV files found in {runs_dir}/")
    frames = []
    for path in result_files:
        df = pd.read_csv(path)
        validate_result_rows(df, source=path)
        frames.append(ordered_result_frame(df))
    summary = pd.concat(frames, ignore_index=True)
    summary = summary.sort_values(
        ["model_family", "model_name", "train_budget_samples", "budget_name", "model_seed"]
    ).reset_index(drop=True)
    ensure_parent(output_path)
    summary.to_csv(output_path, index=False)
    return summary
