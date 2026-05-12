from __future__ import annotations

import pytest

from perovskite_screening.config import ProjectConfig
from perovskite_screening.pipeline.run_experiment import run_experiment


@pytest.mark.parametrize(
    ("family", "message"),
    [
        ("cgcnn", "CGCNN training requires optional dependencies"),
        ("matgl", "MatGL training requires optional dependencies"),
    ],
)
def test_optional_trainer_families_route_to_dependency_messages(family: str, message: str) -> None:
    config = ProjectConfig({"model": {"family": family}})

    with pytest.raises(ImportError, match=message):
        run_experiment(config=config, split_strategy="random_iid", budget_name="B500", seed=42)
