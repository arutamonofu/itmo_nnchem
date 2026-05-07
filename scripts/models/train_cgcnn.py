from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.nn import Linear, BatchNorm1d
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.data import Data
    from torch_geometric.nn import CGConv, global_mean_pool
except ImportError as exc:
    print(
        "CGCNN dependencies are required. Install them with: "
        "pip install -r requirements/cgcnn.txt",
        file=sys.stderr,
    )
    print(f"Missing dependency: {exc.name}", file=sys.stderr)
    sys.exit(1)

# Добавляем пути к проекту
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_io import project_path
from src.metrics import compute_regression_metrics
from src.project_data import make_train_val_test_dataframes
from src.result_schema import make_result_row, ordered_result_frame

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Глобальные константы (Твои настройки) ---
ALLOWED_TRAIN_FRACTIONS = [0.025, 0.25, 1.0]
MODEL_NAME = "cgcnn_optimized"
MODEL_FAMILY = "cgcnn"
RESULT_FILE = "cgcnn.csv"
NOTES = "CGCNN with RBF, BatchNorm and Residual connections."

# --- Вспомогательные функции ---
def fraction_to_name(train_fraction: float) -> str:
    return str(float(train_fraction)).replace(".", "_")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized CGCNN training.")
    parser.add_argument("--train-fraction", type=float, required=True, choices=ALLOWED_TRAIN_FRACTIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()

def load_data(train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        return make_train_val_test_dataframes(train_fraction=train_fraction)
    except Exception as exc:
        logger.error(f"Error loading data: {exc}")
        raise

# --- Архитектура модели ---
class RBFExpansion(nn.Module):
    def __init__(self, dmin=0, dmax=8, steps=64):
        super().__init__()
        self.register_buffer("centers", torch.linspace(dmin, dmax, steps))
        self.gamma = 1.0 / (steps / (dmax - dmin))**2

    def forward(self, edge_attr):
        return torch.exp(-self.gamma * (edge_attr - self.centers)**2)

class CGCNN(nn.Module):
    def __init__(self, node_dim=100, hidden_dim=64, edge_dim=64):
        super().__init__()
        self.embedding = Linear(node_dim, hidden_dim)
        self.rbf = RBFExpansion(steps=edge_dim)
        
        self.conv1 = CGConv(hidden_dim, dim=edge_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.conv2 = CGConv(hidden_dim, dim=edge_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.conv3 = CGConv(hidden_dim, dim=edge_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        
        self.fc = nn.Sequential(
            Linear(hidden_dim, 32),
            nn.ReLU(),
            Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = self.rbf(edge_attr)
        
        x = F.relu(self.embedding(x))
        x = x + F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = x + F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = x + F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        
        x = global_mean_pool(x, batch)
        return self.fc(x).view(-1)

# --- Основной цикл обучения ---
def run_cgcnn_training(
    *, train_df, val_df, test_df, seed, epochs, batch_size, lr, device,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    
    resolved_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    if "mps" in device and torch.backends.mps.is_available():
        resolved_device = torch.device("mps")

    t_mean = float(train_df["target"].mean())
    t_std = float(train_df["target"].std())

    def create_graphs(df):
        graphs = []
        for _, row in df.iterrows():
            struct, target = row["structure"], row["target"]
            z = torch.tensor([site.specie.Z for site in struct], dtype=torch.long)
            x = F.one_hot(z, num_classes=100).to(torch.float)
            neigh = struct.get_neighbor_list(r=8.0)
            # Исправляем Warning через numpy
            edge_index = torch.from_numpy(np.array([neigh[0], neigh[1]])).long()
            edge_attr = torch.from_numpy(np.array(neigh[3])).float().unsqueeze(1)
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                               y=torch.tensor([target], dtype=torch.float)))
        return graphs

    train_loader = GeoDataLoader(create_graphs(train_df), batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(create_graphs(val_df), batch_size=batch_size)
    test_loader = GeoDataLoader(create_graphs(test_df), batch_size=batch_size)

    model = CGCNN().to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(resolved_device)
            optimizer.zero_grad()
            y_pred = model(batch)
            y_true_scaled = (batch.y - t_mean) / t_std
            loss = loss_fn(y_pred, y_true_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    model.eval()
    val_pred, test_pred = [], []
    with torch.no_grad():
        for b in val_loader:
            p = model(b.to(resolved_device)) * t_std + t_mean
            val_pred.extend(p.cpu().numpy())
        for b in test_loader:
            p = model(b.to(resolved_device)) * t_std + t_mean
            test_pred.extend(p.cpu().numpy())
            
    val_metrics = compute_regression_metrics(val_df["target"].to_numpy(), np.array(val_pred))
    return np.array(test_pred), val_metrics

# --- Сохранение результатов (Твои функции) ---
def save_predictions(*, sample_ids, y_true, y_pred, train_fraction, seed) -> str:
    rel_path = f"results/predictions/{MODEL_NAME}_{fraction_to_name(train_fraction)}_seed{seed}.csv"
    path = project_path(*rel_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": sample_ids.astype(int).to_numpy(), "split": "test", 
                  "y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)
    return rel_path

def save_result_row(row: dict[str, object]) -> Path:
    result_path = project_path("results", "raw", RESULT_FILE)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if result_path.exists():
        old_df = pd.read_csv(result_path)
        duplicate = ((old_df["model"] == row["model"]) & 
                     (old_df["train_fraction"].astype(float) == float(row["train_fraction"])) & 
                     (old_df["seed"].astype(int) == int(row["seed"])))
        new_df = pd.concat([old_df.loc[~duplicate], new_df], ignore_index=True)
    ordered_result_frame(new_df).to_csv(result_path, index=False)
    return result_path

def main() -> None:
    args = parse_args()
    train_df, val_df, test_df = load_data(args.train_fraction)

    logger.info(f"Starting {MODEL_NAME} training on {args.device}...")
    y_pred, val_metrics = run_cgcnn_training(
        train_df=train_df, val_df=val_df, test_df=test_df,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, 
        lr=args.lr, device=args.device
    )

    y_true = test_df["target"].to_numpy()
    metrics = compute_regression_metrics(y_true, y_pred)
    
    predictions_path = save_predictions(
        sample_ids=test_df["sample_id"], y_true=y_true, y_pred=y_pred,
        train_fraction=args.train_fraction, seed=args.seed
    )
    
    result_row = make_result_row(
        model=MODEL_NAME, model_family=MODEL_FAMILY,
        train_fraction=args.train_fraction, seed=args.seed,
        mae=metrics["mae"], rmse=metrics["rmse"], r2=metrics["r2"],
        predictions_path=predictions_path, notes=NOTES
    )
    result_path = save_result_row(result_row)

    print(f"\nLoaded dataset: {len(train_df) + len(val_df) + len(test_df)} samples")
    print(f"Train fraction: {args.train_fraction}")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Validation MAE: {val_metrics['mae']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved result row to: {result_path}")

if __name__ == "__main__":
    main()
