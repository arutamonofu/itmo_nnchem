from __future__ import annotations

from typing import Any

import numpy as np


def _require_cgcnn_dependencies() -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
        from torch.nn import BatchNorm1d, Linear
        from torch_geometric.data import Data
        from torch_geometric.nn import CGConv, global_mean_pool
    except ImportError as exc:
        raise ImportError(
            "CGCNN requires optional PyTorch Geometric dependencies. "
            "Install them with `pip install -r requirements/cgcnn.txt`."
        ) from exc
    return {
        "torch": torch,
        "nn": nn,
        "F": F,
        "BatchNorm1d": BatchNorm1d,
        "Linear": Linear,
        "Data": Data,
        "CGConv": CGConv,
        "global_mean_pool": global_mean_pool,
    }


def _make_cgcnn_class():
    deps = _require_cgcnn_dependencies()
    torch = deps["torch"]
    nn = deps["nn"]
    F = deps["F"]
    BatchNorm1d = deps["BatchNorm1d"]
    Linear = deps["Linear"]
    CGConv = deps["CGConv"]
    global_mean_pool = deps["global_mean_pool"]

    class RBFExpansion(nn.Module):
        def __init__(self, dmin: float = 0.0, dmax: float = 8.0, steps: int = 64) -> None:
            super().__init__()
            self.register_buffer("centers", torch.linspace(dmin, dmax, steps))
            self.gamma = 1.0 / (steps / (dmax - dmin)) ** 2

        def forward(self, edge_attr):
            return torch.exp(-self.gamma * (edge_attr - self.centers) ** 2)

    class CGCNN(nn.Module):
        """CGCNN regressor restored from the pre-refactor training script."""

        def __init__(self, node_dim: int = 119, hidden_dim: int = 64, edge_dim: int = 64) -> None:
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
                nn.Dropout(p=0.1),
                Linear(32, 1),
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

    return CGCNN


def structure_to_cgcnn_graph(structure, target: float, *, cutoff: float = 8.0):
    deps = _require_cgcnn_dependencies()
    torch = deps["torch"]
    F = deps["F"]
    Data = deps["Data"]
    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    x = F.one_hot(z, num_classes=119).to(torch.float)
    neighbors = structure.get_neighbor_list(r=cutoff)
    edge_index = torch.from_numpy(np.array([neighbors[0], neighbors[1]])).long()
    edge_attr = torch.from_numpy(np.array(neighbors[3])).float().unsqueeze(1)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([target], dtype=torch.float),
    )


def dataframe_to_cgcnn_graphs(df, *, cutoff: float = 8.0) -> list[Any]:
    return [
        structure_to_cgcnn_graph(row["structure"], float(row["target"]), cutoff=cutoff)
        for _, row in df.iterrows()
    ]


def build_cgcnn_model(
    *,
    node_dim: int = 119,
    hidden_dim: int = 64,
    edge_dim: int = 64,
) -> Any:
    model_class = _make_cgcnn_class()
    return model_class(node_dim=node_dim, hidden_dim=hidden_dim, edge_dim=edge_dim)
