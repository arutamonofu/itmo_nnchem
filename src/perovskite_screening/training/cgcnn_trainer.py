from __future__ import annotations

import logging

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


def _model_config(config: ProjectConfig) -> dict[str, object]:
    return dict(config.raw.get("model", {}).get("params", {}))


def train_cgcnn(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch, nn, GeoDataLoader = _require_training_dependencies()
    params = _model_config(config)
    epochs = int(params.get("epochs", 40))
    batch_size = int(params.get("batch_size", 128))
    lr = float(params.get("lr", 5e-5))
    requested_device = str(params.get("device", "cuda"))
    cutoff = float(params.get("cutoff", 8.0))
    hidden_dim = int(params.get("hidden_dim", 64))
    edge_dim = int(params.get("edge_dim", 64))

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

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
        scheduler.step(avg_loss)
        LOGGER.info(
            "CGCNN epoch %s/%s | loss=%.6f | lr=%.2e",
            epoch + 1,
            epochs,
            avg_loss,
            optimizer.param_groups[0]["lr"],
        )

    model.eval()
    val_pred: list[float] = []
    test_pred: list[float] = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch.to(device)) * target_std + target_mean
            val_pred.extend(pred.cpu().numpy())
        for batch in test_loader:
            pred = model(batch.to(device)) * target_std + target_mean
            test_pred.extend(pred.cpu().numpy())

    val_metrics = compute_regression_metrics(val_df["target"].to_numpy(), np.asarray(val_pred))
    return np.asarray(test_pred), val_metrics


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
        notes=f"{NOTES} Validation MAE={val_metrics['mae']:.6f}.",
    )
    result_path = project_path("outputs", "runs", "cgcnn.csv")
    upsert_result_row(row, result_path)
    row["result_path"] = str(result_path)
    return row
