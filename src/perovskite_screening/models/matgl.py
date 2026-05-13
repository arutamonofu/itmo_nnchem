from __future__ import annotations

from typing import Any

import numpy as np


PRETRAINED_MODEL_NAME = "MEGNet-Eform-MP-2018.6.1"
ALLOWED_MATGL_STRATEGIES = {"frozen", "differential", "full"}


def require_matgl_dependencies() -> dict[str, Any]:
    try:
        import torch
        import lightning as L
        import matgl
        from lightning.pytorch.callbacks import EarlyStopping
        from lightning.pytorch.loggers import CSVLogger
        from matgl.config import DEFAULT_ELEMENTS
        from matgl.ext.pymatgen import Structure2Graph
        from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
        from matgl.models import TransformedTargetModel
        from matgl.utils.training import ModelLightningModule
    except ImportError as exc:
        raise ImportError(
            "MatGL training requires optional dependencies. "
            "Install them with `pip install -r requirements/matgl.txt`."
        ) from exc
    return {
        "torch": torch,
        "L": L,
        "matgl": matgl,
        "EarlyStopping": EarlyStopping,
        "CSVLogger": CSVLogger,
        "DEFAULT_ELEMENTS": DEFAULT_ELEMENTS,
        "Structure2Graph": Structure2Graph,
        "MGLDataset": MGLDataset,
        "MGLDataLoader": MGLDataLoader,
        "collate_fn_graph": collate_fn_graph,
        "TransformedTargetModel": TransformedTargetModel,
        "ModelLightningModule": ModelLightningModule,
    }


def prepare_matgl_datasets(
    train_df,
    val_df,
    test_df,
    *,
    cutoff: float = 4.0,
):
    deps = require_matgl_dependencies()
    converter = deps["Structure2Graph"](element_types=deps["DEFAULT_ELEMENTS"], cutoff=cutoff)
    dataset_class = deps["MGLDataset"]
    return (
        dataset_class(
            structures=train_df["structure"].tolist(),
            converter=converter,
            labels={"labels": train_df["target"].to_numpy()},
        ),
        dataset_class(
            structures=val_df["structure"].tolist(),
            converter=converter,
            labels={"labels": val_df["target"].to_numpy()},
        ),
        dataset_class(
            structures=test_df["structure"].tolist(),
            converter=converter,
            labels={"labels": test_df["target"].to_numpy()},
        ),
    )


def build_matgl_model(*, pretrained_model_name: str = PRETRAINED_MODEL_NAME):
    deps = require_matgl_dependencies()
    return deps["matgl"].load_model(pretrained_model_name)


def configure_matgl_optimizer(megnet_model, *, strategy: str, epochs: int):
    deps = require_matgl_dependencies()
    torch = deps["torch"]
    if strategy not in ALLOWED_MATGL_STRATEGIES:
        raise ValueError(f"Unsupported MatGL fine-tuning strategy: {strategy!r}")

    if strategy == "frozen":
        for param in megnet_model.parameters():
            param.requires_grad = False
        for param in megnet_model.output_proj.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, megnet_model.parameters()),
            lr=1e-3,
            weight_decay=1e-5,
        )
        return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), 1e-3, 30

    if strategy == "differential":
        for param in megnet_model.parameters():
            param.requires_grad = True
        param_groups = [
            {"params": megnet_model.embedding.parameters(), "lr": 1e-5},
            {"params": megnet_model.edge_encoder.parameters(), "lr": 1e-5},
            {"params": megnet_model.node_encoder.parameters(), "lr": 1e-5},
            {"params": megnet_model.state_encoder.parameters(), "lr": 1e-5},
            {"params": megnet_model.blocks.parameters(), "lr": 1e-5},
            {"params": megnet_model.edge_s2s.parameters(), "lr": 1e-4},
            {"params": megnet_model.node_s2s.parameters(), "lr": 1e-4},
            {"params": megnet_model.output_proj.parameters(), "lr": 1e-3},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), 1e-3, 30

    for param in megnet_model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(megnet_model.parameters(), lr=1e-4, weight_decay=1e-4)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), 1e-4, 20


def make_transformed_target_model(*, loaded_model, megnet_model, y_train: np.ndarray):
    deps = require_matgl_dependencies()
    normalizer_class = type(loaded_model.transformer)
    normalizer = normalizer_class(mean=float(np.mean(y_train)), std=float(np.std(y_train)))
    return deps["TransformedTargetModel"](model=megnet_model, target_transformer=normalizer)
