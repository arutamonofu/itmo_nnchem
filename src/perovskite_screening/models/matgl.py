from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from perovskite_screening.io.run_artifacts import cache_dir_rel_path, project_rel_path


PRETRAINED_MODEL_NAME = "MEGNet-Eform-MP-2018.6.1"
ALLOWED_MATGL_STRATEGIES = {"frozen", "differential", "full"}
MATGL_CACHE_FAMILY = "matgl"


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


def _accepts_parameter(callable_obj, parameter_name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return parameter_name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def _matgl_dataset_kwargs(
    dataset_class,
    *,
    structures: list,
    converter,
    labels: np.ndarray,
    cutoff: float,
    cache_stem: str | None,
    partition: str,
    force_reload: bool = False,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "structures": structures,
        "converter": converter,
        "labels": {"labels": labels},
    }
    if _accepts_parameter(dataset_class, "threebody_cutoff"):
        kwargs["threebody_cutoff"] = cutoff
    if cache_stem is None:
        return kwargs

    cache_base = project_rel_path(cache_dir_rel_path(cache_family=MATGL_CACHE_FAMILY, stem=cache_stem))
    cache_base.mkdir(parents=True, exist_ok=True)

    accepts_raw_dir = _accepts_parameter(dataset_class, "raw_dir")
    accepts_save_dir = _accepts_parameter(dataset_class, "save_dir")
    if _accepts_parameter(dataset_class, "name"):
        kwargs["name"] = partition if accepts_raw_dir or accepts_save_dir else str(cache_base / partition)
    if accepts_raw_dir:
        kwargs["raw_dir"] = str(cache_base)
    if accepts_save_dir:
        kwargs["save_dir"] = str(cache_base)
    if _accepts_parameter(dataset_class, "force_reload"):
        kwargs["force_reload"] = bool(force_reload)
    if _accepts_parameter(dataset_class, "verbose"):
        kwargs["verbose"] = False
    return kwargs


def prepare_matgl_datasets(
    train_df,
    val_df,
    test_df,
    *,
    cutoff: float = 4.0,
    cache_stem: str | None = None,
    force_reload_cache: bool = False,
):
    deps = require_matgl_dependencies()
    converter = deps["Structure2Graph"](element_types=deps["DEFAULT_ELEMENTS"], cutoff=cutoff)
    dataset_class = deps["MGLDataset"]
    return (
        dataset_class(
            **_matgl_dataset_kwargs(
                dataset_class,
                structures=train_df["structure"].tolist(),
                converter=converter,
                labels=train_df["target"].to_numpy(),
                cutoff=cutoff,
                cache_stem=cache_stem,
                partition="train",
                force_reload=force_reload_cache,
            )
        ),
        dataset_class(
            **_matgl_dataset_kwargs(
                dataset_class,
                structures=val_df["structure"].tolist(),
                converter=converter,
                labels=val_df["target"].to_numpy(),
                cutoff=cutoff,
                cache_stem=cache_stem,
                partition="val",
                force_reload=force_reload_cache,
            )
        ),
        dataset_class(
            **_matgl_dataset_kwargs(
                dataset_class,
                structures=test_df["structure"].tolist(),
                converter=converter,
                labels=test_df["target"].to_numpy(),
                cutoff=cutoff,
                cache_stem=cache_stem,
                partition="test",
                force_reload=force_reload_cache,
            )
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
