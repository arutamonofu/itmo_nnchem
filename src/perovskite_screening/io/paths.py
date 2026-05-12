from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*map(str, parts))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
