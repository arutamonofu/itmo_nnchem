from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from perovskite_screening.config import ProjectConfig
from perovskite_screening.evaluation.metrics import compute_regression_metrics
from perovskite_screening.io.paths import ensure_parent, project_path
from perovskite_screening.io.results import make_result_row, upsert_result_row
from perovskite_screening.models.cgcnn import build_cgcnn_model, dataframe_to_cgcnn_graphs
from perovskite_screening.training.trainer import load_experiment_data


LOGGER = logging.getLogger(__name__)
MODEL_FAMILY = "cgcnn"
MODEL_NAME = "cgcnn_optimized"
NOTES = "CGCNN with RBF edge expansion, BatchNorm, residual CGConv blocks, and target standardization."
CGCNN_DEFAULT_PARAMS: dict[str, object] = {
    "epochs": 40,
    "batch_size": 128,
    "lr": 5e-5,
    "weight_decay": 0.0,
    "device": "cuda",
    "cutoff": 8.0,
    "hidden_dim": 64,
    "edge_dim": 64,
    "early_stopping_patience": None,
    "scheduler_patience": 5,
    "monitor": "val_mae",
}


def _require_training_dependencies():
    try:
        import torch
        from torch import nn
        from torch_geometric.loader import DataLoader as GeoDataLoader
    except ImportError as exc:
        raise ImportError(
            "CGCNN training requires optional dependencies. "
            "Install them with `pip install -r requirements/cgcnn.txt`."
        ) from exc
    return torch, nn, GeoDataLoader


def _resolve_device(torch, requested: str):
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def cgcnn_model_params(config: ProjectConfig) -> dict[str, object]:
    params = config.raw.get("model", {}).get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise TypeError("model.params must be a mapping when provided")
    merged = {**CGCNN_DEFAULT_PARAMS, **dict(params)}
    return {
        "epochs": int(merged["epochs"]),
        "batch_size": int(merged["batch_size"]),
        "lr": float(merged["lr"]),
        "weight_decay": float(merged["weight_decay"]),
        "device": str(merged["device"]),
        "cutoff": float(merged["cutoff"]),
        "hidden_dim": int(merged["hidden_dim"]),
        "edge_dim": int(merged["edge_dim"]),
        "early_stopping_patience": (
            None if merged["early_stopping_patience"] is None else int(merged["early_stopping_patience"])
        ),
        "scheduler_patience": int(merged["scheduler_patience"]),
        "monitor": str(merged["monitor"]),
    }


def _format_cgcnn_params(params: Mapping[str, object]) -> str:
    return json.dumps(dict(params), sort_keys=True, separators=(",", ":"))


def update_early_stopping_state(
    *,
    val_mae: float,
    best_val_mae: float,
    bad_epochs: int,
    patience: int | None,
) -> tuple[bool, float, int, bool]:
    improved = val_mae < best_val_mae
    if improved:
        best_val_mae = val_mae
        bad_epochs = 0
    else:
        bad_epochs += 1
    should_stop = patience is not None and bad_epochs >= patience
    return improved, best_val_mae, bad_epochs, should_stop


def _predict_loader(model, loader, *, device, target_mean: float, target_std: float) -> np.ndarray:
    pred_values: list[float] = []
    for batch in loader:
        pred = model(batch.to(device)) * target_std + target_mean
        pred_values.extend(pred.cpu().numpy())
    return np.asarray(pred_values)


def train_cgcnn(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch, nn, GeoDataLoader = _require_training_dependencies()
    params = cgcnn_model_params(config)
    epochs = int(params["epochs"])
    batch_size = int(params["batch_size"])
    lr = float(params["lr"])
    weight_decay = float(params["weight_decay"])
    scheduler_patience = int(params["scheduler_patience"])
    early_stopping_patience = params["early_stopping_patience"]
    if params["monitor"] != "val_mae":
        raise ValueError("CGCNN currently supports only monitor='val_mae'")
    requested_device = str(params["device"])
    cutoff = float(params["cutoff"])
    hidden_dim = int(params["hidden_dim"])
    edge_dim = int(params["edge_dim"])

    torch.manual_seed(seed)
    device = _resolve_device(torch, requested_device)
    target_mean = float(train_df["target"].mean())
    target_std = float(train_df["target"].std())
    if target_std == 0:
        target_std = 1.0

    train_loader = GeoDataLoader(
        dataframe_to_cgcnn_graphs(train_df, cutoff=cutoff),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = GeoDataLoader(dataframe_to_cgcnn_graphs(val_df, cutoff=cutoff), batch_size=batch_size)
    test_loader = GeoDataLoader(dataframe_to_cgcnn_graphs(test_df, cutoff=cutoff), batch_size=batch_size)

    model = build_cgcnn_model(hidden_dim=hidden_dim, edge_dim=edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_patience,
    )
    loss_fn = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            y_pred = model(batch)
            y_true_scaled = (batch.y - target_mean) / target_std
            loss = loss_fn(y_pred, y_true_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        avg_loss = epoch_loss / max(len(train_loader), 1)
        model.eval()
        with torch.no_grad():
            val_pred = _predict_loader(model, val_loader, device=device, target_mean=target_mean, target_std=target_std)
        val_metrics = compute_regression_metrics(val_df["target"].to_numpy(), val_pred)
        val_mae = float(val_metrics["mae"])
        scheduler.step(val_mae)
        improved, best_val_mae, bad_epochs, should_stop = update_early_stopping_state(
            val_mae=val_mae,
            best_val_mae=best_val_mae,
            bad_epochs=bad_epochs,
            patience=early_stopping_patience,
        )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
        LOGGER.info(
            "CGCNN epoch %s/%s | train_loss=%.6f | val_mae=%.6f | best_val_mae=%.6f | lr=%.2e",
            epoch + 1,
            epochs,
            avg_loss,
            val_mae,
            best_val_mae,
            optimizer.param_groups[0]["lr"],
        )
        if should_stop:
            LOGGER.info(
                "Stopping CGCNN early at epoch %s/%s after %s epochs without validation MAE improvement.",
                epoch + 1,
                epochs,
                bad_epochs,
            )
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = _predict_loader(model, val_loader, device=device, target_mean=target_mean, target_std=target_std)
        test_pred = _predict_loader(model, test_loader, device=device, target_mean=target_mean, target_std=target_std)

    val_metrics = compute_regression_metrics(val_df["target"].to_numpy(), val_pred)
    return test_pred, val_metrics


def _save_predictions(
    *,
    sample_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    budget_name: str,
    seed: int,
    split_strategy: str,
) -> str:
    rel_path = f"outputs/runs/predictions/{split_strategy}_{MODEL_NAME}_{budget_name}_seed{seed}.csv"
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


def run_cgcnn_experiment(
    *,
    config: ProjectConfig,
    split_strategy: str,
    budget_name: str,
    seed: int,
) -> dict[str, object]:
    _require_training_dependencies()
    params = cgcnn_model_params(config)
    data = load_experiment_data(budget_name=budget_name, split_strategy=split_strategy, config=config)
    test_pred, val_metrics = train_cgcnn(
        train_df=data.train,
        val_df=data.val,
        test_df=data.test,
        config=config,
        seed=seed,
    )
    y_test = data.test["target"].to_numpy()
    metrics = compute_regression_metrics(y_test, test_pred)
    predictions_path = _save_predictions(
        sample_ids=data.test["sample_id"],
        y_true=y_test,
        y_pred=test_pred,
        budget_name=budget_name,
        seed=seed,
        split_strategy=split_strategy,
    )
    row = make_result_row(
        model_name=MODEL_NAME,
        model_family=MODEL_FAMILY,
        budget_name=budget_name,
        model_seed=seed,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        predictions_path=predictions_path,
        split_strategy=split_strategy,
        config=config,
        model_params=params,
        notes=f"{NOTES} Params={_format_cgcnn_params(params)}; validation MAE={val_metrics['mae']:.6f}.",
    )
    result_path = project_path("outputs", "runs", "cgcnn.csv")
    upsert_result_row(row, result_path)
    row["result_path"] = str(result_path)
    return row
