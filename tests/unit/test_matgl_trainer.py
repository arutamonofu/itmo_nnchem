from __future__ import annotations

import logging

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
