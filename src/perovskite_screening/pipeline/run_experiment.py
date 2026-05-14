from __future__ import annotations

import numpy as np
import pandas as pd

from perovskite_screening.config import ProjectConfig
from perovskite_screening.data.budgets import load_budget_metadata, resolve_budgets
from perovskite_screening.evaluation.metrics import compute_regression_metrics
from perovskite_screening.io.paths import ensure_parent, project_path
from perovskite_screening.io.results import make_result_row, upsert_result_row
from perovskite_screening.io.run_artifacts import prediction_rel_path, project_rel_path, run_artifact_stem
from perovskite_screening.training.cgcnn_trainer import run_cgcnn_experiment
from perovskite_screening.training.descriptor_trainer import (
    descriptor_model_params,
    run_descriptor_experiment,
)
from perovskite_screening.training.matgl_trainer import run_matgl_experiment
from perovskite_screening.training.trainer import load_experiment_data


def _experiment_kind(config: ProjectConfig) -> str:
    return str(config.raw.get("model", {}).get("family", "descriptor_baseline"))


def run_mean_baseline(
    *,
    config: ProjectConfig,
    split_strategy: str,
    budget_name: str,
    seed: int,
) -> dict[str, object]:
    data = load_experiment_data(budget_name=budget_name, split_strategy=split_strategy, config=config)
    train_mean = float(data.train["target"].mean())
    y_test = data.test["target"].to_numpy()
    y_pred = np.full(shape=len(data.test), fill_value=train_mean)
    metrics = compute_regression_metrics(y_test, y_pred)
    artifact_stem = run_artifact_stem(
        split_strategy=split_strategy,
        model_name="mean_baseline",
        budget_name=budget_name,
        seed=seed,
    )
    rel_path = prediction_rel_path(artifact_stem)
    prediction_path = project_rel_path(rel_path)
    ensure_parent(prediction_path)
    pd.DataFrame(
        {
            "sample_id": data.test["sample_id"].astype(int).to_numpy(),
            "split": "test",
            "y_true": y_test,
            "y_pred": y_pred,
        }
    ).to_csv(prediction_path, index=False)
    row = make_result_row(
        model_name="mean_baseline",
        model_family="dummy",
        budget_name=budget_name,
        model_seed=seed,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        mape=metrics["mape"],
        predictions_path=rel_path,
        split_strategy=split_strategy,
        config=config,
        notes="Constant prediction equal to selected train subset target mean.",
    )
    result_path = project_path("outputs", "runs", "mean_baseline.csv")
    upsert_result_row(row, result_path)
    row["result_path"] = str(result_path)
    return row


def run_experiment(
    *,
    config: ProjectConfig,
    split_strategy: str,
    budget_name: str,
    seed: int,
) -> dict[str, object]:
    family = _experiment_kind(config)
    if family == "dummy":
        return run_mean_baseline(
            config=config,
            split_strategy=split_strategy,
            budget_name=budget_name,
            seed=seed,
        )
    if family == "descriptor_baseline":
        model_config = config.raw.get("model", {})
        return run_descriptor_experiment(
            config=config,
            split_strategy=split_strategy,
            budget_name=budget_name,
            seed=seed,
            model_kind=str(model_config.get("kind", "rf")),
            feature_set=str(model_config.get("feature_set", "starter")),
            model_params=descriptor_model_params(config),
        )
    if family == "cgcnn":
        return run_cgcnn_experiment(
            config=config,
            split_strategy=split_strategy,
            budget_name=budget_name,
            seed=seed,
        )
    if family == "matgl":
        return run_matgl_experiment(
            config=config,
            split_strategy=split_strategy,
            budget_name=budget_name,
            seed=seed,
        )
    raise NotImplementedError(f"Experiment family {family!r} is not implemented in the base package")


def available_budgets_for_split(split_strategy: str) -> list[str]:
    metadata = load_budget_metadata(project_path("data", "splits", split_strategy, "budgets.json"))
    return resolve_budgets(metadata, "all")
