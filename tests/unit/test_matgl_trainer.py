from __future__ import annotations

import logging

import pandas as pd
import pytest

from perovskite_screening.config import ProjectConfig
from perovskite_screening.models import matgl as matgl_models
from perovskite_screening.training import matgl_trainer


class _CudaUnavailable:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    cuda = _CudaUnavailable()

    class optim:
        class AdamW:
            def __init__(self, params, *, lr=None, weight_decay=None) -> None:
                self.params = params
                self.lr = lr
                self.weight_decay = weight_decay

        class lr_scheduler:
            class CosineAnnealingLR:
                def __init__(self, optimizer, *, T_max) -> None:
                    self.optimizer = optimizer
                    self.T_max = T_max


class _FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeModule:
    def __init__(self) -> None:
        self._params = [_FakeParameter()]

    def parameters(self):
        return self._params


class _FakeMegNet:
    def __init__(self) -> None:
        self.root_parameters = [_FakeParameter(), _FakeParameter()]
        self.embedding = _FakeModule()
        self.edge_encoder = _FakeModule()
        self.node_encoder = _FakeModule()
        self.state_encoder = _FakeModule()
        self.blocks = _FakeModule()
        self.edge_s2s = _FakeModule()
        self.node_s2s = _FakeModule()
        self.output_proj = _FakeModule()

    def parameters(self):
        return [
            *self.root_parameters,
            *self.embedding.parameters(),
            *self.edge_encoder.parameters(),
            *self.node_encoder.parameters(),
            *self.state_encoder.parameters(),
            *self.blocks.parameters(),
            *self.edge_s2s.parameters(),
            *self.node_s2s.parameters(),
            *self.output_proj.parameters(),
        ]


class _FakeStructure2Graph:
    def __init__(self, *, element_types, cutoff: float) -> None:
        self.element_types = element_types
        self.cutoff = cutoff


class _FakeMGLDataset:
    calls: list[dict[str, object]] = []

    def __init__(
        self,
        *,
        structures,
        converter,
        labels,
        name: str,
        force_reload: bool,
        verbose: bool = False,
        threebody_cutoff: float | None = None,
    ) -> None:
        self.structures = structures
        self.converter = converter
        self.labels = labels
        self.name = name
        self.force_reload = force_reload
        self.verbose = verbose
        self.threebody_cutoff = threebody_cutoff
        self.calls.append(
            {
                "name": name,
                "force_reload": force_reload,
                "cutoff": converter.cutoff,
                "n_structures": len(structures),
            }
        )

    def __len__(self) -> int:
        return len(self.structures)


class _WrongSizeDataset:
    def __len__(self) -> int:
        return 500


class _NoNameMGLDataset:
    def __init__(self, *, structures, converter, labels) -> None:
        self.structures = structures
        self.converter = converter
        self.labels = labels


class _RootMGLDataset:
    calls: list[dict[str, object]] = []

    def __init__(
        self,
        *,
        structures,
        converter,
        labels,
        root: str,
        clear_processed: bool = False,
        save_cache: bool = True,
    ) -> None:
        self.structures = structures
        self.converter = converter
        self.labels = labels
        self.root = root
        self.clear_processed = clear_processed
        self.save_cache = save_cache
        self.calls.append(
            {
                "root": root,
                "clear_processed": clear_processed,
                "cutoff": converter.cutoff,
                "n_structures": len(structures),
            }
        )

    def __len__(self) -> int:
        return len(self.structures)


def test_matgl_cuda_request_falls_back_to_cpu_with_message(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)

    device_name = matgl_trainer._resolve_device_name(_FakeTorch, "cuda")

    assert device_name == "cpu"
    assert "CUDA is unavailable" in caplog.text


def test_matgl_frozen_strategy_trains_only_output_projection(monkeypatch) -> None:
    monkeypatch.setattr(matgl_models, "require_matgl_dependencies", lambda: {"torch": _FakeTorch})
    model = _FakeMegNet()

    matgl_models.configure_matgl_optimizer(model, strategy="frozen", epochs=100)

    frozen_params = [
        *model.root_parameters,
        *model.embedding.parameters(),
        *model.edge_encoder.parameters(),
        *model.node_encoder.parameters(),
        *model.state_encoder.parameters(),
        *model.blocks.parameters(),
        *model.edge_s2s.parameters(),
        *model.node_s2s.parameters(),
    ]
    assert all(not param.requires_grad for param in frozen_params)
    assert all(param.requires_grad for param in model.output_proj.parameters())


def test_matgl_config_patience_overrides_strategy_default(monkeypatch) -> None:
    monkeypatch.setattr(matgl_models, "require_matgl_dependencies", lambda: {"torch": _FakeTorch})
    _, _, _, strategy_patience = matgl_models.configure_matgl_optimizer(
        _FakeMegNet(),
        strategy="frozen",
        epochs=100,
    )

    params = matgl_trainer.matgl_effective_params(
        ProjectConfig.from_file("configs/experiments/matgl.yaml"),
        strategy_patience=strategy_patience,
    )

    assert strategy_patience == 30
    assert params["strategy"] == "frozen"
    assert params["early_stopping_patience"] == 10


def test_matgl_cache_name_includes_run_context_and_partition() -> None:
    train_name = matgl_models.matgl_dataset_cache_name(
        split_strategy="random_iid",
        model_name="matgl_megnet_differential",
        budget_name="B500",
        seed=42,
        cutoff=4.0,
        partition="train",
    )
    val_name = matgl_models.matgl_dataset_cache_name(
        split_strategy="random_iid",
        model_name="matgl_megnet_differential",
        budget_name="B500",
        seed=42,
        cutoff=4.0,
        partition="val",
    )
    other_budget_name = matgl_models.matgl_dataset_cache_name(
        split_strategy="random_iid",
        model_name="matgl_megnet_differential",
        budget_name="B2000",
        seed=42,
        cutoff=4.0,
        partition="train",
    )

    assert train_name == "random_iid_matgl_megnet_differential_B500_seed42_cutoff4p0_train"
    assert "random_iid" in train_name
    assert "matgl_megnet_differential" in train_name
    assert "B500" in train_name
    assert "seed42" in train_name
    assert "cutoff4p0" in train_name
    assert train_name != val_name
    assert train_name != other_budget_name
    assert (
        matgl_models.matgl_dataset_cache_root(train_name)
        == "MGLDataset/random_iid_matgl_megnet_differential_B500_seed42_cutoff4p0_train"
    )


def test_prepare_matgl_datasets_sets_unique_cache_names_and_force_reload(monkeypatch) -> None:
    _FakeMGLDataset.calls = []
    monkeypatch.setattr(
        matgl_models,
        "require_matgl_dependencies",
        lambda: {
            "Structure2Graph": _FakeStructure2Graph,
            "DEFAULT_ELEMENTS": ["H", "O"],
            "MGLDataset": _FakeMGLDataset,
        },
    )
    train_df = pd.DataFrame({"structure": ["s1", "s2"], "target": [1.0, 2.0]})
    val_df = pd.DataFrame({"structure": ["s3"], "target": [3.0]})
    test_df = pd.DataFrame({"structure": ["s4"], "target": [4.0]})

    datasets = matgl_models.prepare_matgl_datasets(
        train_df,
        val_df,
        test_df,
        cutoff=4.0,
        split_strategy="random_iid",
        model_name="matgl_megnet_differential",
        budget_name="B500",
        seed=42,
        force_reload_cache=True,
    )

    names = [call["name"] for call in _FakeMGLDataset.calls]
    assert len(datasets) == 3
    assert len(set(names)) == 3
    assert all(name != "MGLDataset" for name in names)
    assert names == [
        "random_iid_matgl_megnet_differential_B500_seed42_cutoff4p0_train",
        "random_iid_matgl_megnet_differential_B500_seed42_cutoff4p0_val",
        "random_iid_matgl_megnet_differential_B500_seed42_cutoff4p0_test",
    ]
    assert all(call["force_reload"] is True for call in _FakeMGLDataset.calls)
    assert all(call["cutoff"] == 4.0 for call in _FakeMGLDataset.calls)


def test_prepare_matgl_datasets_uses_unique_cache_roots_when_name_is_unsupported(monkeypatch) -> None:
    _RootMGLDataset.calls = []
    monkeypatch.setattr(
        matgl_models,
        "require_matgl_dependencies",
        lambda: {
            "Structure2Graph": _FakeStructure2Graph,
            "DEFAULT_ELEMENTS": ["H", "O"],
            "MGLDataset": _RootMGLDataset,
        },
    )
    train_df = pd.DataFrame({"structure": ["s1", "s2"], "target": [1.0, 2.0]})
    val_df = pd.DataFrame({"structure": ["s3"], "target": [3.0]})
    test_df = pd.DataFrame({"structure": ["s4"], "target": [4.0]})

    matgl_models.prepare_matgl_datasets(
        train_df,
        val_df,
        test_df,
        cutoff=4.0,
        split_strategy="random_iid",
        model_name="matgl_megnet_differential",
        budget_name="B500",
        seed=42,
        force_reload_cache=True,
    )

    roots = [call["root"] for call in _RootMGLDataset.calls]
    assert len(set(roots)) == 3
    assert roots == [
        "MGLDataset/random_iid_matgl_megnet_differential_B500_seed42_cutoff4p0_train",
        "MGLDataset/random_iid_matgl_megnet_differential_B500_seed42_cutoff4p0_val",
        "MGLDataset/random_iid_matgl_megnet_differential_B500_seed42_cutoff4p0_test",
    ]
    assert all(call["clear_processed"] is True for call in _RootMGLDataset.calls)
    assert all(call["cutoff"] == 4.0 for call in _RootMGLDataset.calls)


def test_prepare_matgl_datasets_requires_supported_cache_namespace(monkeypatch) -> None:
    monkeypatch.setattr(
        matgl_models,
        "require_matgl_dependencies",
        lambda: {
            "Structure2Graph": _FakeStructure2Graph,
            "DEFAULT_ELEMENTS": ["H", "O"],
            "MGLDataset": _NoNameMGLDataset,
        },
    )
    df = pd.DataFrame({"structure": ["s1"], "target": [1.0]})

    with pytest.raises(RuntimeError, match="unique MatGL cache namespaces cannot be enforced"):
        matgl_models.prepare_matgl_datasets(
            df,
            df,
            df,
            cutoff=4.0,
            split_strategy="random_iid",
            model_name="matgl_megnet_differential",
            budget_name="B500",
            seed=42,
        )


def test_validate_matgl_dataset_size_reports_cache_mismatch() -> None:
    with pytest.raises(ValueError, match="MatGL dataset cache mismatch: partition=train"):
        matgl_models.validate_matgl_dataset_size(
            partition="train",
            expected_size=2000,
            dataset=_WrongSizeDataset(),
        )


def test_matgl_effective_params_includes_force_reload_cache() -> None:
    params = matgl_trainer.matgl_effective_params(
        ProjectConfig(
            {
                "model": {
                    "params": {
                        "force_reload_cache": True,
                    }
                }
            }
        )
    )

    assert params["force_reload_cache"] is True
