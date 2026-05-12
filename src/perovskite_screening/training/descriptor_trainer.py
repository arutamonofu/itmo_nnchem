from __future__ import annotations

from pathlib import Path

import pandas as pd

from perovskite_screening.config import ProjectConfig
from perovskite_screening.evaluation.metrics import compute_regression_metrics
from perovskite_screening.features.descriptors import featurize
from perovskite_screening.io.paths import ensure_parent, project_path
from perovskite_screening.io.results import make_result_row, upsert_result_row
from perovskite_screening.models.descriptor import build_descriptor_model
from perovskite_screening.training.trainer import load_experiment_data


MODEL_FAMILY = "descriptor_baseline"


def descriptor_model_name(model_kind: str, feature_set: str) -> str:
    return f"descriptor_{model_kind}_{feature_set}"


def _save_predictions(
    *,
    model_name: str,
    sample_ids: pd.Series,
    y_true,
    y_pred,
    budget_name: str,
    seed: int,
    split_strategy: str,
) -> str:
    rel_path = f"outputs/runs/predictions/{split_strategy}_{model_name}_{budget_name}_seed{seed}.csv"
    path = project_path(*rel_path.split("/"))
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


def run_descriptor_experiment(
    *,
    config: ProjectConfig,
    split_strategy: str,
    budget_name: str,
    seed: int,
    model_kind: str,
    feature_set: str,
) -> dict[str, object]:
    data = load_experiment_data(budget_name=budget_name, split_strategy=split_strategy, config=config)
    x_train = featurize(data.train, feature_set)
    x_test = featurize(data.test, feature_set)
    y_train = data.train["target"].to_numpy()
    y_test = data.test["target"].to_numpy()
    model = build_descriptor_model(model_kind, seed)
    model.fit(x_train, y_train)
    test_pred = model.predict(x_test)
    metrics = compute_regression_metrics(y_test, test_pred)
    run_model_name = descriptor_model_name(model_kind, feature_set)
    predictions_path = _save_predictions(
        model_name=run_model_name,
        sample_ids=data.test["sample_id"],
        y_true=y_test,
        y_pred=test_pred,
        budget_name=budget_name,
        seed=seed,
        split_strategy=split_strategy,
    )
    row = make_result_row(
        model_name=run_model_name,
        model_family=MODEL_FAMILY,
        budget_name=budget_name,
        model_seed=seed,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        predictions_path=predictions_path,
        split_strategy=split_strategy,
        config=config,
        notes=f"{model_kind.upper()} descriptor baseline with feature_set={feature_set}; n_features={x_train.shape[1]}.",
    )
    result_path = project_path("outputs", "runs", "descriptor_baseline.csv")
    upsert_result_row(row, result_path)
    row["result_path"] = str(result_path)
    return row
