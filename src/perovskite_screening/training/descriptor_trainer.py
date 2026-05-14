from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any

import pandas as pd

from perovskite_screening.config import ProjectConfig
from perovskite_screening.evaluation.metrics import compute_regression_metrics
from perovskite_screening.features.descriptors import featurize
from perovskite_screening.io.paths import ensure_parent, project_path
from perovskite_screening.io.results import make_result_row, upsert_result_row
from perovskite_screening.io.run_artifacts import (
    history_file_rel_path,
    model_file_rel_path,
    prediction_rel_path,
    project_rel_path,
    run_artifact_stem,
)
from perovskite_screening.models.descriptor import build_descriptor_model, descriptor_effective_params
from perovskite_screening.training.trainer import load_experiment_data


MODEL_FAMILY = "descriptor_baseline"


def descriptor_model_name(model_kind: str, feature_set: str) -> str:
    return f"descriptor_{model_kind}_{feature_set}"


def descriptor_model_params(config: ProjectConfig) -> dict[str, Any]:
    params = config.raw.get("model", {}).get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError("model.params must be a mapping when provided")
    return dict(params)


def _save_predictions(
    *,
    artifact_stem: str,
    sample_ids: pd.Series,
    y_true,
    y_pred,
) -> str:
    rel_path = prediction_rel_path(artifact_stem)
    path = project_rel_path(rel_path)
    ensure_parent(path)
    pd.DataFrame(
        {
            "sample_id": sample_ids.astype(int).to_numpy(),
            "split": "test",
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(path, index=False)
    return rel_path


def _save_model(
    *,
    model,
    artifact_stem: str,
) -> str:
    rel_path = model_file_rel_path(model_family=MODEL_FAMILY, stem=artifact_stem, suffix=".pkl")
    path = project_rel_path(rel_path)
    ensure_parent(path)
    with path.open("wb") as f:
        pickle.dump(model, f)
    return rel_path


def _save_training_history(
    *,
    artifact_stem: str,
    model_kind: str,
    feature_set: str,
    n_features: int,
    train_metrics: dict[str, float],
) -> str:
    rel_path = history_file_rel_path(artifact_stem)
    path = project_rel_path(rel_path)
    ensure_parent(path)
    pd.DataFrame(
        [
            {
                "stage": "fit",
                "model_kind": model_kind,
                "feature_set": feature_set,
                "n_features": int(n_features),
                "train_mae": float(train_metrics["mae"]),
                "train_rmse": float(train_metrics["rmse"]),
                "train_r2": float(train_metrics["r2"]),
            }
        ]
    ).to_csv(path, index=False)
    return rel_path


def run_descriptor_experiment(
    *,
    config: ProjectConfig,
    split_strategy: str,
    budget_name: str,
    seed: int,
    model_kind: str,
    feature_set: str,
    model_params: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    data = load_experiment_data(budget_name=budget_name, split_strategy=split_strategy, config=config)
    x_train = featurize(data.train, feature_set)
    x_test = featurize(data.test, feature_set)
    y_train = data.train["target"].to_numpy()
    y_test = data.test["target"].to_numpy()
    effective_params = descriptor_effective_params(model_kind, seed, model_params)
    model = build_descriptor_model(model_kind, seed, params=model_params)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_metrics = compute_regression_metrics(y_train, train_pred)
    metrics = compute_regression_metrics(y_test, test_pred)
    run_model_name = descriptor_model_name(model_kind, feature_set)
    artifact_stem = run_artifact_stem(
        split_strategy=split_strategy,
        model_name=run_model_name,
        budget_name=budget_name,
        seed=seed,
    )
    model_path = _save_model(
        model=model,
        artifact_stem=artifact_stem,
    )
    predictions_path = _save_predictions(
        artifact_stem=artifact_stem,
        sample_ids=data.test["sample_id"],
        y_true=y_test,
        y_pred=test_pred,
    )
    history_path = _save_training_history(
        artifact_stem=artifact_stem,
        model_kind=model_kind,
        feature_set=feature_set,
        n_features=x_train.shape[1],
        train_metrics=train_metrics,
    )
    row = make_result_row(
        model_name=run_model_name,
        model_family=MODEL_FAMILY,
        budget_name=budget_name,
        model_seed=seed,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        mape=metrics["mape"],
        predictions_path=predictions_path,
        model_path=model_path,
        history_path=history_path,
        split_strategy=split_strategy,
        config=config,
        model_params=effective_params,
        notes=(
            f"{model_kind.upper()} descriptor baseline with feature_set={feature_set}; "
            f"n_features={x_train.shape[1]}; model_path={model_path}; history_path={history_path}."
        ),
    )
    result_path = project_path("outputs", "runs", "descriptor_baseline.csv")
    upsert_result_row(row, result_path)
    row["result_path"] = str(result_path)
    return row
