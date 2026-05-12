from __future__ import annotations

from perovskite_screening.config import ProjectConfig


def test_default_config_loads() -> None:
    config = ProjectConfig.from_file("configs/default.yaml")
    assert config.raw["project"]["name"] == "perovskite_screening"
    assert config.default_split_strategy == "random_iid"
    assert config.random_seed == 42
