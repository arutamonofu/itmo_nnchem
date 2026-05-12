from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perovskite_screening.io.paths import project_path


DEFAULT_CONFIG_PATH = project_path("configs", "default.yaml")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "None"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid YAML line: {raw_line!r}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_path(config_path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError:
        return _minimal_yaml_load(text)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a mapping")
    return data


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "ProjectConfig":
        return cls(load_config(path))

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("splits", {}).get("random_seed", 42))

    @property
    def default_split_strategy(self) -> str:
        return str(self.raw.get("splits", {}).get("default_strategy", "random_iid"))

    @property
    def target_unit(self) -> str:
        return str(self.raw.get("data", {}).get("target_unit", "eV/unit cell"))
