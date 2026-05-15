from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)
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


def _format_cache_float(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


def matgl_dataset_cache_name(
    *,
    split_strategy: str,
    model_name: str,
    budget_name: str,
    seed: int,
    cutoff: float,
    partition: str,
) -> str:
    raw_name = (
        f"{split_strategy}_{model_name}_{budget_name}_seed{seed}_"
        f"cutoff{_format_cache_float(cutoff)}_{partition}"
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("_")


def matgl_dataset_cache_root(cache_name: str) -> str:
    return (Path("MGLDataset") / cache_name).as_posix()


def _constructor_supports_kwarg(constructor, kwarg: str) -> bool:
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return kwarg in signature.parameters


def _mgl_dataset_kwargs(
    dataset_class,
    *,
    name: str,
    force_reload_cache: bool,
) -> dict[str, object]:
    supported: dict[str, object] = {}
    if _constructor_supports_kwarg(dataset_class, "name"):
        supported["name"] = name
    elif _constructor_supports_kwarg(dataset_class, "root"):
        supported["root"] = matgl_dataset_cache_root(name)
    else:
        raise RuntimeError(
            "MGLDataset exposes neither 'name' nor 'root'; unique MatGL cache namespaces cannot be enforced. "
            "Use a MatGL version that supports MGLDataset(name=...) or MGLDataset(root=...)."
        )

    if _constructor_supports_kwarg(dataset_class, "force_reload"):
        supported["force_reload"] = force_reload_cache
    elif _constructor_supports_kwarg(dataset_class, "clear_processed"):
        supported["clear_processed"] = force_reload_cache
    elif force_reload_cache:
        LOGGER.warning("MGLDataset does not expose 'force_reload'; ignoring force_reload_cache=true.")
    return supported


def validate_matgl_dataset_size(*, partition: str, expected_size: int, dataset) -> None:
    actual_size = len(dataset)
    if actual_size != expected_size:
        raise ValueError(
            "MatGL dataset cache mismatch: "
            f"partition={partition}, expected {expected_size} structures, got {actual_size}. "
            "This likely indicates stale/shared MGLDataset cache. "
            "Clear MGLDataset/ or use unique cache names."
        )


def prepare_matgl_datasets(
    train_df,
    val_df,
    test_df,
    *,
    cutoff: float = 4.0,
    split_strategy: str,
    model_name: str,
    budget_name: str,
    seed: int,
    force_reload_cache: bool = False,
):
    deps = require_matgl_dependencies()
    converter = deps["Structure2Graph"](element_types=deps["DEFAULT_ELEMENTS"], cutoff=cutoff)
    dataset_class = deps["MGLDataset"]
    cache_names = {
        partition: matgl_dataset_cache_name(
            split_strategy=split_strategy,
            model_name=model_name,
            budget_name=budget_name,
            seed=seed,
            cutoff=cutoff,
            partition=partition,
        )
        for partition in ("train", "val", "test")
    }
    train_dataset = dataset_class(
        structures=train_df["structure"].tolist(),
        converter=converter,
        labels={"labels": train_df["target"].to_numpy()},
        **_mgl_dataset_kwargs(
            dataset_class,
            name=cache_names["train"],
            force_reload_cache=force_reload_cache,
        ),
    )
    val_dataset = dataset_class(
        structures=val_df["structure"].tolist(),
        converter=converter,
        labels={"labels": val_df["target"].to_numpy()},
        **_mgl_dataset_kwargs(
            dataset_class,
            name=cache_names["val"],
            force_reload_cache=force_reload_cache,
        ),
    )
    test_dataset = dataset_class(
        structures=test_df["structure"].tolist(),
        converter=converter,
        labels={"labels": test_df["target"].to_numpy()},
        **_mgl_dataset_kwargs(
            dataset_class,
            name=cache_names["test"],
            force_reload_cache=force_reload_cache,
        ),
    )
    validate_matgl_dataset_size(partition="train", expected_size=len(train_df), dataset=train_dataset)
    validate_matgl_dataset_size(partition="val", expected_size=len(val_df), dataset=val_dataset)
    validate_matgl_dataset_size(partition="test", expected_size=len(test_df), dataset=test_dataset)
    return train_dataset, val_dataset, test_dataset


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
