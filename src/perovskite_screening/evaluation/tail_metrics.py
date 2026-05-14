from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from perovskite_screening.evaluation.metrics import compute_regression_metrics
from perovskite_screening.io.paths import ensure_parent, project_path


REQUIRED_TARGET_BIN_COLUMNS = ["sample_id", "target_bin_10", "target_bin_5"]
REQUIRED_PREDICTION_COLUMNS = ["sample_id", "y_true", "y_pred"]
TAIL_BIN_SCHEMES = {
    "target_bin_10": ["low_10", "middle_80", "high_10"],
    "target_bin_5": ["low_5", "middle_90", "high_5"],
}
TAIL_METRIC_COLUMNS = [
    "model_name",
    "model_family",
    "split_strategy",
    "budget_name",
    "model_seed",
    "target_bin_scheme",
    "target_bin",
    "n_samples",
    "mae",
    "rmse",
    "r2",
    "mape",
    "predictions_path",
    "source_result_path",
]


def load_target_bins(path: Path | None = None) -> pd.DataFrame:
    target_bins_path = path or project_path("data", "splits", "target_tails", "target_bins.csv")
    if not target_bins_path.exists():
        raise FileNotFoundError(f"Missing target-bin file: {target_bins_path}. Run make-splits first.")
    target_bins = pd.read_csv(target_bins_path)
    missing = [col for col in REQUIRED_TARGET_BIN_COLUMNS if col not in target_bins.columns]
    if missing:
        raise ValueError(f"{target_bins_path} is missing target-bin columns: {missing}")
    if target_bins["sample_id"].duplicated().any():
        duplicated = sorted(target_bins.loc[target_bins["sample_id"].duplicated(), "sample_id"].tolist())[:10]
        raise ValueError(f"{target_bins_path} contains duplicated sample_id values: {duplicated}")
    return target_bins


def _resolve_project_path(path_value: object, project_root: Path | None = None) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (project_root or project_path()) / path


def _infer_split_strategy(row: pd.Series | dict[str, object]) -> str:
    if "split_strategy" in row and not pd.isna(row["split_strategy"]):
        return str(row["split_strategy"])
    split_id = str(row.get("split_id", ""))
    for strategy in ("random_iid", "element_set"):
        if split_id.startswith(f"{strategy}_"):
            return strategy
    return ""


def _empty_metric_values() -> dict[str, float]:
    return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "mape": np.nan}


def _compute_metric_values(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return _empty_metric_values()
    return compute_regression_metrics(frame["y_true"], frame["y_pred"])


def _base_metadata(
    *,
    result_row: pd.Series | dict[str, object] | None = None,
    predictions_path: str = "",
    source_result_path: str = "",
) -> dict[str, object]:
    result_row = {} if result_row is None else result_row
    return {
        "model_name": result_row.get("model_name", ""),
        "model_family": result_row.get("model_family", ""),
        "split_strategy": _infer_split_strategy(result_row),
        "budget_name": result_row.get("budget_name", ""),
        "model_seed": result_row.get("model_seed", ""),
        "predictions_path": predictions_path,
        "source_result_path": source_result_path,
    }


def compute_tail_metrics_for_predictions(
    predictions: pd.DataFrame,
    target_bins: pd.DataFrame,
    *,
    result_row: pd.Series | dict[str, object] | None = None,
    predictions_path: str = "",
    source_result_path: str = "",
    schemes: tuple[str, ...] = ("target_bin_10", "target_bin_5"),
    include_empty_slices: bool = True,
) -> pd.DataFrame:
    missing_predictions = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in predictions.columns]
    if missing_predictions:
        raise ValueError(f"predictions are missing required columns: {missing_predictions}")
    missing_bins = [col for col in ["sample_id", *schemes] if col not in target_bins.columns]
    if missing_bins:
        raise ValueError(f"target bins are missing required columns: {missing_bins}")

    merged = predictions.merge(
        target_bins[["sample_id", *schemes]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    missing_joined_bins = merged[list(schemes)].isna().any(axis=1)
    if missing_joined_bins.any():
        warnings.warn(
            f"{int(missing_joined_bins.sum())} prediction row(s) have no target-bin assignment",
            stacklevel=2,
        )

    base = _base_metadata(
        result_row=result_row,
        predictions_path=predictions_path,
        source_result_path=source_result_path,
    )
    rows: list[dict[str, object]] = []
    for scheme in schemes:
        for target_bin, frame in [("overall", merged), *(
            (bin_name, merged.loc[merged[scheme] == bin_name])
            for bin_name in TAIL_BIN_SCHEMES.get(scheme, [])
        )]:
            if frame.empty and not include_empty_slices:
                continue
            metrics = _compute_metric_values(frame)
            rows.append(
                {
                    **base,
                    "target_bin_scheme": scheme,
                    "target_bin": target_bin,
                    "n_samples": int(len(frame)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows, columns=TAIL_METRIC_COLUMNS)


def compute_tail_metrics_from_result_table(
    result_table: pd.DataFrame,
    target_bins: pd.DataFrame | None = None,
    *,
    project_root: Path | None = None,
    source_result_path: str = "",
    schemes: tuple[str, ...] = ("target_bin_10", "target_bin_5"),
    include_empty_slices: bool = True,
) -> pd.DataFrame:
    target_bins = load_target_bins() if target_bins is None else target_bins
    frames: list[pd.DataFrame] = []
    rows_processed = 0
    prediction_files_found = 0
    skipped = 0

    for _, result_row in result_table.iterrows():
        rows_processed += 1
        predictions_path_value = result_row.get("predictions_path", "")
        if pd.isna(predictions_path_value) or str(predictions_path_value).strip() == "":
            warnings.warn("result row has an empty predictions_path; skipping", stacklevel=2)
            skipped += 1
            continue

        predictions_path = _resolve_project_path(predictions_path_value, project_root=project_root)
        if not predictions_path.exists():
            warnings.warn(f"prediction file does not exist: {predictions_path}; skipping", stacklevel=2)
            skipped += 1
            continue

        prediction_files_found += 1
        predictions = pd.read_csv(predictions_path)
        try:
            frames.append(
                compute_tail_metrics_for_predictions(
                    predictions,
                    target_bins,
                    result_row=result_row,
                    predictions_path=str(predictions_path_value),
                    source_result_path=source_result_path,
                    schemes=schemes,
                    include_empty_slices=include_empty_slices,
                )
            )
        except ValueError as exc:
            warnings.warn(f"{predictions_path}: {exc}; skipping", stacklevel=2)
            skipped += 1

    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(columns=TAIL_METRIC_COLUMNS)
    result.attrs["tail_metrics_summary"] = {
        "result_rows_processed": rows_processed,
        "prediction_files_found": prediction_files_found,
        "skipped": skipped,
    }
    return result


def compute_tail_metrics_from_results_file(
    results_path: Path,
    *,
    target_bins_path: Path | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    result_table = pd.read_csv(results_path)
    target_bins = load_target_bins(target_bins_path)
    return compute_tail_metrics_from_result_table(
        result_table,
        target_bins,
        project_root=project_root,
        source_result_path=str(results_path),
    )


def compute_tail_metrics_from_results_source(
    results_path: Path,
    *,
    target_bins_path: Path | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    target_bins = load_target_bins(target_bins_path)
    if results_path.is_dir():
        result_files = sorted(results_path.glob("*.csv"))
        if not result_files:
            raise FileNotFoundError(f"No result CSV files found in {results_path}/")
    else:
        result_files = [results_path]

    frames: list[pd.DataFrame] = []
    summary = {"result_rows_processed": 0, "prediction_files_found": 0, "skipped": 0}
    for result_file in result_files:
        result_table = pd.read_csv(result_file)
        frame = compute_tail_metrics_from_result_table(
            result_table,
            target_bins,
            project_root=project_root,
            source_result_path=str(result_file),
        )
        frames.append(frame)
        file_summary = frame.attrs.get("tail_metrics_summary", {})
        for key in summary:
            summary[key] += int(file_summary.get(key, 0))

    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(columns=TAIL_METRIC_COLUMNS)
    result.attrs["tail_metrics_summary"] = summary
    return result


def write_tail_metrics(
    results_path: Path,
    output_path: Path,
    *,
    target_bins_path: Path | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    tail_metrics = compute_tail_metrics_from_results_source(
        results_path,
        target_bins_path=target_bins_path,
        project_root=project_root,
    )
    ensure_parent(output_path)
    tail_metrics.to_csv(output_path, index=False)
    return tail_metrics
