from __future__ import annotations

import importlib

import pytest

from perovskite_screening.config import ProjectConfig
from perovskite_screening.pipeline.run_experiment import run_experiment


@pytest.mark.parametrize(
    ("family", "runner_name", "message"),
    [
        ("cgcnn", "run_cgcnn_experiment", "CGCNN training requires optional dependencies"),
        ("matgl", "run_matgl_experiment", "MatGL training requires optional dependencies"),
    ],
)
def test_optional_trainer_families_route_to_dependency_messages(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    runner_name: str,
    message: str,
) -> None:
    run_experiment_module = importlib.import_module("perovskite_screening.pipeline.run_experiment")

    def _raise_missing_dependency(**kwargs):
        raise ImportError(message)

    monkeypatch.setattr(run_experiment_module, runner_name, _raise_missing_dependency)
    config = ProjectConfig({"model": {"family": family}})

    with pytest.raises(ImportError, match=message):
        run_experiment(config=config, split_strategy="random_iid", budget_name="B500", seed=42)
