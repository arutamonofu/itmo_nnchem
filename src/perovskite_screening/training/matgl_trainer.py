from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from perovskite_screening.config import ProjectConfig
from perovskite_screening.evaluation.metrics import compute_regression_metrics
from perovskite_screening.io.paths import ensure_parent, project_path
from perovskite_screening.io.results import make_result_row, upsert_result_row
from perovskite_screening.io.run_artifacts import (
    cache_dir_rel_path,
    history_dir_rel_path,
    model_dir_rel_path,
    prediction_rel_path,
    project_rel_path,
    run_artifact_stem,
)
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
    params = config.raw.get("model", {}).get("params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError("model.params must be a mapping when provided")
    return dict(params)


def matgl_effective_params(config: ProjectConfig, *, strategy_patience: int | None = None) -> dict[str, object]:
    params = _model_config(config)
    effective: dict[str, object] = {
        "strategy": str(params.get("strategy", "full")),
        "pretrained_model_name": str(params.get("pretrained_model_name", PRETRAINED_MODEL_NAME)),
        "epochs": int(params.get("epochs", 500)),
        "batch_size": int(params.get("batch_size", 16)),
        "cutoff": float(params.get("cutoff", 4.0)),
        "device": str(params.get("device", "cuda")),
        "num_workers": int(params.get("num_workers", 0)),
        "progress_bar": bool(params.get("progress_bar", True)),
        "force_reload_cache": bool(params.get("force_reload_cache", False)),
    }
    configured_patience = params.get("early_stopping_patience")
    if configured_patience is not None:
        effective["early_stopping_patience"] = int(configured_patience)
    elif strategy_patience is not None:
        effective["early_stopping_patience"] = int(strategy_patience)
    return effective


def _resolve_device_name(torch, requested_device: str) -> str:
    device_name = requested_device.lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("MatGL requested device='cuda', but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    if device_name not in {"cuda", "cpu", "auto"}:
        raise ValueError(f"Unsupported MatGL device: {requested_device!r}. Use 'cuda', 'cpu', or 'auto'.")
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def _lightning_accelerator(device_name: str) -> str:
    return "gpu" if device_name == "cuda" else "cpu"


def _save_predictions(
    *,
    artifact_stem: str,
    sample_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
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


def _matgl_graph_cache_stem(*, artifact_stem: str, cutoff: float) -> str:
    cutoff_label = f"{cutoff:g}".replace(".", "p")
    return f"{artifact_stem}_cutoff{cutoff_label}"


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

    params = matgl_effective_params(config)
    strategy = str(params.get("strategy", "full"))
    batch_size = int(params.get("batch_size", 16))
    epochs = int(params.get("epochs", 500))
    cutoff = float(params.get("cutoff", 4.0))
    pretrained_model_name = str(params.get("pretrained_model_name", PRETRAINED_MODEL_NAME))
    num_workers = int(params.get("num_workers", 0))
    progress_bar = bool(params.get("progress_bar", True))
    force_reload_cache = bool(params.get("force_reload_cache", False))
    device_name = _resolve_device_name(torch, str(params.get("device", "cuda")))
    model_name = f"{BASE_MODEL_NAME}_{strategy}"
    artifact_stem = run_artifact_stem(
        split_strategy=split_strategy,
        model_name=model_name,
        budget_name=budget_name,
        seed=seed,
    )
    graph_cache_stem = _matgl_graph_cache_stem(artifact_stem=artifact_stem, cutoff=cutoff)
    graph_cache_path = cache_dir_rel_path(cache_family=MODEL_FAMILY, stem=graph_cache_stem)

    L.seed_everything(seed, workers=True)
    train_dataset, val_dataset, test_dataset = prepare_matgl_datasets(
        train_df,
        val_df,
        test_df,
        cutoff=cutoff,
        cache_stem=graph_cache_stem,
        force_reload_cache=force_reload_cache,
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
    optimizer, scheduler, base_lr, strategy_patience = configure_matgl_optimizer(
        megnet_model,
        strategy=strategy,
        epochs=epochs,
    )
    params = matgl_effective_params(config, strategy_patience=strategy_patience)
    patience = int(params["early_stopping_patience"])
    lightning_model = ModelLightningModule(
        model=megnet_model,
        data_mean=data_mean,
        data_std=data_std,
        loss="huber_loss",
        optimizer=optimizer,
        scheduler=scheduler,
        lr=base_lr,
    )

    rel_history_dir = history_dir_rel_path(artifact_stem)
    history_logger = CSVLogger(
        project_rel_path(rel_history_dir).parent,
        name=Path(rel_history_dir).name,
        version="",
    )
    trainer = L.Trainer(
        max_epochs=epochs,
        log_every_n_steps=5,
        accelerator=_lightning_accelerator(device_name),
        devices=1,
        logger=history_logger,
        callbacks=[EarlyStopping(monitor="val_MAE", patience=patience, mode="min")],
        deterministic=True,
        enable_progress_bar=progress_bar,
    )
    LOGGER.info(
        "Starting MatGL fine-tuning: strategy=%s, patience=%s, budget=%s, device=%s",
        strategy,
        patience,
        budget_name,
        device_name,
    )
    trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    history_path = (Path(history_logger.log_dir) / "metrics.csv").relative_to(project_path()).as_posix()

    final_model = make_transformed_target_model(
        loaded_model=loaded_model,
        megnet_model=megnet_model,
        y_train=y_train,
    )
    rel_model_path = model_dir_rel_path(model_family=MODEL_FAMILY, stem=artifact_stem)
    model_path = project_rel_path(rel_model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    final_model.save(model_path)
    return final_model, rel_model_path, history_path, graph_cache_path, strategy, device_name, params


def evaluate_matgl(model, df: pd.DataFrame, *, device_name: str) -> np.ndarray:
    deps = require_matgl_dependencies()
    torch = deps["torch"]
    device = torch.device(_resolve_device_name(torch, device_name))
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
    final_model, model_path, history_path, graph_cache_path, strategy, device_name, params = train_matgl(
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
    artifact_stem = run_artifact_stem(
        split_strategy=split_strategy,
        model_name=model_name,
        budget_name=budget_name,
        seed=seed,
    )
    predictions_path = _save_predictions(
        artifact_stem=artifact_stem,
        sample_ids=data.test["sample_id"],
        y_true=data.test["target"].to_numpy(),
        y_pred=test_pred,
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
        model_path=model_path,
        history_path=history_path,
        split_strategy=split_strategy,
        config=config,
        model_params=params,
        notes=(
            f"{NOTES} Strategy={strategy}; early_stopping_patience={params['early_stopping_patience']}; "
            f"model_path={model_path}; history_path={history_path}; graph_cache_path={graph_cache_path}; "
            f"validation MAE={val_metrics['mae']:.6f}."
        ),
    )
    result_path = project_path("outputs", "runs", "matgl.csv")
    upsert_result_row(row, result_path)
    row["result_path"] = str(result_path)
    return row
