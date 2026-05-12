from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from perovskite_screening.config import ProjectConfig
from perovskite_screening.evaluation.metrics import compute_regression_metrics
from perovskite_screening.io.paths import ensure_parent, project_path
from perovskite_screening.io.results import make_result_row, upsert_result_row
from perovskite_screening.models.matgl import (
    PRETRAINED_MODEL_NAME,
    build_matgl_model,
    configure_matgl_optimizer,
    make_transformed_target_model,
    prepare_matgl_datasets,
    require_matgl_dependencies,
)
from perovskite_screening.training.trainer import load_experiment_data


LOGGER = logging.getLogger(__name__)
MODEL_FAMILY = "matgl"
BASE_MODEL_NAME = "matgl_megnet"
NOTES = "Fine-tuning of pre-trained MEGNet with Huber loss and early stopping."


def _model_config(config: ProjectConfig) -> dict[str, object]:
    return dict(config.raw.get("model", {}).get("params", {}))


def _save_predictions(
    *,
    sample_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
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


def train_matgl(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
    budget_name: str,
    split_strategy: str,
    seed: int,
):
    deps = require_matgl_dependencies()
    torch = deps["torch"]
    L = deps["L"]
    MGLDataLoader = deps["MGLDataLoader"]
    collate_fn_graph = deps["collate_fn_graph"]
    ModelLightningModule = deps["ModelLightningModule"]
    EarlyStopping = deps["EarlyStopping"]
    CSVLogger = deps["CSVLogger"]

    params = _model_config(config)
    strategy = str(params.get("strategy", "full"))
    batch_size = int(params.get("batch_size", 16))
    epochs = int(params.get("epochs", 500))
    cutoff = float(params.get("cutoff", 4.0))
    pretrained_model_name = str(params.get("pretrained_model_name", PRETRAINED_MODEL_NAME))
    num_workers = int(params.get("num_workers", 0))
    progress_bar = bool(params.get("progress_bar", True))

    L.seed_everything(seed, workers=True)
    train_dataset, val_dataset, test_dataset = prepare_matgl_datasets(
        train_df,
        val_df,
        test_df,
        cutoff=cutoff,
    )
    train_loader, val_loader, _ = MGLDataLoader(
        train_data=train_dataset,
        val_data=val_dataset,
        test_data=test_dataset,
        collate_fn=collate_fn_graph,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    loaded_model = build_matgl_model(pretrained_model_name=pretrained_model_name)
    megnet_model = loaded_model.model
    y_train = train_dataset.labels["labels"]
    data_mean = float(np.mean(y_train))
    data_std = float(np.std(y_train)) or 1.0
    optimizer, scheduler, base_lr, patience = configure_matgl_optimizer(
        megnet_model,
        strategy=strategy,
        epochs=epochs,
    )
    lightning_model = ModelLightningModule(
        model=megnet_model,
        data_mean=data_mean,
        data_std=data_std,
        loss="huber_loss",
        optimizer=optimizer,
        scheduler=scheduler,
        lr=base_lr,
    )

    logger_name = f"MEGNet_{strategy}_{split_strategy}_{budget_name}_seed{seed}"
    trainer = L.Trainer(
        max_epochs=epochs,
        log_every_n_steps=5,
        accelerator="auto",
        devices="auto",
        logger=CSVLogger(project_path("logs"), name=logger_name),
        callbacks=[EarlyStopping(monitor="val_MAE", patience=patience, mode="min")],
        deterministic=True,
        enable_progress_bar=progress_bar,
    )
    LOGGER.info("Starting MatGL fine-tuning: strategy=%s, budget=%s", strategy, budget_name)
    trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_model = make_transformed_target_model(
        loaded_model=loaded_model,
        megnet_model=megnet_model,
        y_train=y_train,
    )
    model_path = project_path("outputs", "models", "finetuned", logger_name)
    model_path.mkdir(parents=True, exist_ok=True)
    final_model.save(model_path)
    return final_model, str(model_path), strategy


def evaluate_matgl(model, df: pd.DataFrame, *, device_name: str) -> np.ndarray:
    deps = require_matgl_dependencies()
    torch = deps["torch"]
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    return df["structure"].apply(lambda structure: float(model.predict_structure(structure).detach().cpu())).to_numpy()


def run_matgl_experiment(
    *,
    config: ProjectConfig,
    split_strategy: str,
    budget_name: str,
    seed: int,
) -> dict[str, object]:
    require_matgl_dependencies()
    data = load_experiment_data(budget_name=budget_name, split_strategy=split_strategy, config=config)
    params = _model_config(config)
    device_name = str(params.get("device", "cuda"))
    final_model, model_path, strategy = train_matgl(
        train_df=data.train,
        val_df=data.val,
        test_df=data.test,
        config=config,
        budget_name=budget_name,
        split_strategy=split_strategy,
        seed=seed,
    )

    val_pred = evaluate_matgl(final_model, data.val, device_name=device_name)
    test_pred = evaluate_matgl(final_model, data.test, device_name=device_name)
    val_metrics = compute_regression_metrics(data.val["target"].to_numpy(), val_pred)
    test_metrics = compute_regression_metrics(data.test["target"].to_numpy(), test_pred)
    model_name = f"{BASE_MODEL_NAME}_{strategy}"
    predictions_path = _save_predictions(
        sample_ids=data.test["sample_id"],
        y_true=data.test["target"].to_numpy(),
        y_pred=test_pred,
        model_name=model_name,
        budget_name=budget_name,
        seed=seed,
        split_strategy=split_strategy,
    )
    row = make_result_row(
        model_name=model_name,
        model_family=MODEL_FAMILY,
        budget_name=budget_name,
        model_seed=seed,
        mae=test_metrics["mae"],
        rmse=test_metrics["rmse"],
        r2=test_metrics["r2"],
        predictions_path=predictions_path,
        split_strategy=split_strategy,
        config=config,
        notes=f"{NOTES} Strategy={strategy}; model_path={model_path}; validation MAE={val_metrics['mae']:.6f}.",
    )
    result_path = project_path("outputs", "runs", "matgl.csv")
    upsert_result_row(row, result_path)
    row["result_path"] = str(result_path)
    return row
