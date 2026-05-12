from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from perovskite_screening.features.descriptors import expanded_features, starter_features
from perovskite_screening.config import ProjectConfig
from perovskite_screening.models.descriptor import build_descriptor_model
from perovskite_screening.models import matgl as matgl_models
from perovskite_screening.training.descriptor_trainer import descriptor_model_params


def test_xgb_missing_dependency_message_mentions_project_extra(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("missing xgboost")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"\.\[xgb\]"):
        build_descriptor_model("xgb", seed=42)


def test_descriptor_random_forest_matches_pre_refactor_defaults() -> None:
    model = build_descriptor_model("rf", seed=42)

    imputer = model.named_steps["simpleimputer"]
    regressor = model.named_steps["randomforestregressor"]

    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == "median"
    assert isinstance(regressor, RandomForestRegressor)
    assert regressor.n_estimators == 300
    assert regressor.random_state == 42
    assert regressor.n_jobs == -1
    assert regressor.min_samples_leaf == 1


def test_descriptor_xgb_uses_config_params_and_seed_overrides_random_state(monkeypatch) -> None:
    class FakeXGBRegressor:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=FakeXGBRegressor))
    config = ProjectConfig.from_file("configs/experiments/descriptor_xgb.yaml")

    params = descriptor_model_params(config)
    params["random_state"] = 999
    model = build_descriptor_model("xgb", seed=123, params=params)
    regressor = model.steps[-1][1]

    assert regressor.kwargs["objective"] == "reg:squarederror"
    assert regressor.kwargs["n_estimators"] == 800
    assert regressor.kwargs["learning_rate"] == 0.03
    assert regressor.kwargs["max_depth"] == 4
    assert regressor.kwargs["min_child_weight"] == 2
    assert regressor.kwargs["subsample"] == 0.85
    assert regressor.kwargs["colsample_bytree"] == 0.85
    assert regressor.kwargs["reg_lambda"] == 2.0
    assert regressor.kwargs["reg_alpha"] == 0.0
    assert regressor.kwargs["tree_method"] == "hist"
    assert regressor.kwargs["n_jobs"] == -1
    assert regressor.kwargs["random_state"] == 123


def test_descriptor_rejects_unsupported_params() -> None:
    with pytest.raises(ValueError, match="Unsupported rf model.params"):
        build_descriptor_model("rf", seed=42, params={"not_a_rf_param": 1})


class _Element:
    def __init__(
        self,
        *,
        symbol: str,
        z: int,
        atomic_mass: float,
        electronegativity: float,
        atomic_radius: float,
        mendeleev_no: float,
        row: int,
        group: int,
    ) -> None:
        self.symbol = symbol
        self.Z = z
        self.atomic_mass = atomic_mass
        self.X = electronegativity
        self.atomic_radius = atomic_radius
        self.mendeleev_no = mendeleev_no
        self.row = row
        self.group = group


class _Composition:
    def __init__(self, amounts: dict[_Element, float]) -> None:
        self._amounts = amounts
        self.elements = list(amounts)

    def __getitem__(self, element: _Element) -> float:
        return self._amounts[element]


def test_descriptor_features_match_pre_refactor_formulas() -> None:
    a = _Element(
        symbol="A",
        z=10,
        atomic_mass=20.0,
        electronegativity=1.0,
        atomic_radius=2.0,
        mendeleev_no=3.0,
        row=2,
        group=4,
    )
    b = _Element(
        symbol="B",
        z=20,
        atomic_mass=40.0,
        electronegativity=3.0,
        atomic_radius=4.0,
        mendeleev_no=5.0,
        row=3,
        group=6,
    )
    structure = SimpleNamespace(
        num_sites=4,
        volume=80.0,
        density=5.0,
        composition=_Composition({a: 1.0, b: 3.0}),
        lattice=SimpleNamespace(a=1.0, b=2.0, c=3.0, alpha=90.0, beta=91.0, gamma=92.0),
    )

    starter = starter_features(structure)
    expanded = expanded_features(structure)

    assert starter == {
        "n_sites": 4.0,
        "volume": 80.0,
        "density": 5.0,
        "num_unique_elements": 2.0,
        "mean_atomic_number": 15.0,
        "std_atomic_number": 5.0,
    }
    assert expanded["volume_per_site"] == 20.0
    assert expanded["max_element_fraction"] == 0.75
    assert expanded["min_element_fraction"] == 0.25
    assert expanded["atomic_number_mean"] == 17.5
    assert expanded["atomic_number_min"] == 10.0
    assert expanded["atomic_number_max"] == 20.0
    assert expanded["atomic_number_range"] == 10.0
    assert np.isclose(expanded["atomic_number_std"], np.sqrt(18.75))


class _FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeModule:
    def __init__(self, count: int = 1) -> None:
        self._params = [_FakeParameter() for _ in range(count)]

    def parameters(self):
        return self._params


class _FakeOptimizer:
    def __init__(self, params, *, lr=None, weight_decay=None) -> None:
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay


class _FakeScheduler:
    def __init__(self, optimizer, *, T_max) -> None:
        self.optimizer = optimizer
        self.T_max = T_max


class _FakeTorch:
    class optim:
        AdamW = _FakeOptimizer

        class lr_scheduler:
            CosineAnnealingLR = _FakeScheduler


class _FakeMegNet:
    def __init__(self) -> None:
        self.root_parameters = [_FakeParameter() for _ in range(2)]
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


@pytest.mark.parametrize(
    ("strategy", "expected_lr", "expected_weight_decay", "expected_patience"),
    [
        ("frozen", 1e-3, 1e-5, 30),
        ("differential", None, 1e-4, 30),
        ("full", 1e-4, 1e-4, 20),
    ],
)
def test_matgl_optimizer_strategies_match_pre_refactor_defaults(
    monkeypatch,
    strategy: str,
    expected_lr: float | None,
    expected_weight_decay: float,
    expected_patience: int,
) -> None:
    monkeypatch.setattr(matgl_models, "require_matgl_dependencies", lambda: {"torch": _FakeTorch})

    optimizer, scheduler, base_lr, patience = matgl_models.configure_matgl_optimizer(
        _FakeMegNet(),
        strategy=strategy,
        epochs=500,
    )

    assert optimizer.lr == expected_lr
    assert optimizer.weight_decay == expected_weight_decay
    assert scheduler.T_max == 500
    assert base_lr == (expected_lr or 1e-3)
    assert patience == expected_patience
