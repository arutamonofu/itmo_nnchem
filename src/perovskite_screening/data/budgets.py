from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from perovskite_screening.io.artifacts import read_json, save_json


BUDGET_STRATEGY = "absolute_doubling_budgets"
DEFAULT_BASE_BUDGET = 500
DEFAULT_GROWTH_FACTOR = 2
DEFAULT_MIN_FINAL_GROWTH_RATIO = 1.25
DEFAULT_SAMPLING_STRATEGY = "deterministic_random"


def budget_name(n_samples: int) -> str:
    return f"B{int(n_samples)}"


def generate_training_budgets(
    full_train_size: int,
    base_budget: int = DEFAULT_BASE_BUDGET,
    growth_factor: int = DEFAULT_GROWTH_FACTOR,
    min_final_growth_ratio: float = DEFAULT_MIN_FINAL_GROWTH_RATIO,
) -> list[dict[str, object]]:
    full_train_size = int(full_train_size)
    base_budget = int(base_budget)
    growth_factor = int(growth_factor)
    min_final_growth_ratio = float(min_final_growth_ratio)
    if full_train_size <= 0:
        raise ValueError(f"full_train_size must be positive; got {full_train_size}")
    if base_budget <= 0:
        raise ValueError(f"base_budget must be positive; got {base_budget}")
    if growth_factor <= 1:
        raise ValueError(f"growth_factor must be greater than 1; got {growth_factor}")
    if min_final_growth_ratio <= 1.0:
        raise ValueError("min_final_growth_ratio must be greater than 1.0")

    budgets: list[dict[str, object]] = []
    candidate = base_budget
    while full_train_size > base_budget and full_train_size >= candidate * min_final_growth_ratio:
        budgets.append({"name": budget_name(candidate), "n_samples": int(candidate), "is_full": False})
        candidate *= growth_factor
    budgets.append({"name": "Bfull", "n_samples": full_train_size, "is_full": True})
    return budgets


def parse_training_budget_config(config: dict[str, Any]) -> dict[str, int | float | str]:
    budget_config = dict(config.get("budgets", config.get("training_budgets", {})))
    return {
        "base_budget": int(budget_config.get("base_budget", DEFAULT_BASE_BUDGET)),
        "growth_factor": int(budget_config.get("growth_factor", DEFAULT_GROWTH_FACTOR)),
        "min_final_growth_ratio": float(
            budget_config.get("min_final_growth_ratio", DEFAULT_MIN_FINAL_GROWTH_RATIO)
        ),
        "sampling_strategy": str(budget_config.get("sampling_strategy", DEFAULT_SAMPLING_STRATEGY)),
    }


def deterministic_train_ordering(train_ids: list[int], split_seed: int) -> list[int]:
    rng = np.random.default_rng(int(split_seed))
    shuffled = rng.permutation(np.array(train_ids, dtype=int))
    return shuffled.astype(int).tolist()


def build_budget_metadata(
    *,
    train_ids: list[int],
    split_seed: int,
    base_budget: int = DEFAULT_BASE_BUDGET,
    growth_factor: int = DEFAULT_GROWTH_FACTOR,
    min_final_growth_ratio: float = DEFAULT_MIN_FINAL_GROWTH_RATIO,
    sampling_strategy: str = DEFAULT_SAMPLING_STRATEGY,
) -> dict[str, object]:
    if sampling_strategy != DEFAULT_SAMPLING_STRATEGY:
        raise ValueError(f"Unsupported budget sampling strategy: {sampling_strategy!r}")
    train_ids = [int(sample_id) for sample_id in train_ids]
    full_train_size = len(train_ids)
    ordering = deterministic_train_ordering(train_ids, split_seed=split_seed)
    budgets: dict[str, dict[str, object]] = {}
    for entry in generate_training_budgets(
        full_train_size=full_train_size,
        base_budget=base_budget,
        growth_factor=growth_factor,
        min_final_growth_ratio=min_final_growth_ratio,
    ):
        name = str(entry["name"])
        n_samples = int(entry["n_samples"])
        is_full = bool(entry["is_full"])
        indices = train_ids if is_full else ordering[:n_samples]
        budgets[name] = {"n_samples": n_samples, "is_full": is_full, "indices": [int(i) for i in indices]}
    metadata = {
        "full_train_size": full_train_size,
        "budget_strategy": BUDGET_STRATEGY,
        "base_budget": int(base_budget),
        "growth_factor": int(growth_factor),
        "min_final_growth_ratio": float(min_final_growth_ratio),
        "sampling_strategy": sampling_strategy,
        "split_seed": int(split_seed),
        "budgets": budgets,
    }
    validate_nested_budgets(metadata)
    return metadata


build_budgets = build_budget_metadata


def save_budget_metadata(metadata: dict[str, object], path: Path) -> None:
    save_json(metadata, path)


def load_budget_metadata(path: Path) -> dict[str, object]:
    metadata = read_json(path)
    if "budgets" not in metadata or not isinstance(metadata["budgets"], dict):
        raise ValueError(f"{path} is missing a budgets object")
    validate_nested_budgets(metadata)
    return metadata


load_budgets_metadata = load_budget_metadata


def available_budget_names(metadata: dict[str, object]) -> list[str]:
    budgets = metadata["budgets"]
    if not isinstance(budgets, dict):
        raise ValueError("Budget metadata has invalid budgets field")
    return list(budgets.keys())


def resolve_budget(metadata: dict[str, object], name: str) -> dict[str, object]:
    budgets = metadata["budgets"]
    if not isinstance(budgets, dict) or name not in budgets:
        raise ValueError(f"Unknown budget {name!r}. Available budgets: {available_budget_names(metadata)}")
    record = budgets[name]
    if not isinstance(record, dict):
        raise ValueError(f"Budget {name!r} has invalid metadata")
    return record


def resolve_budgets(metadata: dict[str, object], requested: str | None) -> list[str]:
    available = available_budget_names(metadata)
    if requested in (None, ""):
        return ["Bfull"]
    if requested == "all":
        return available
    names = [item.strip() for item in requested.split(",") if item.strip()]
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Unknown budget(s): {missing}. Available budgets: {available}")
    return names


def validate_nested_budgets(metadata: dict[str, object]) -> None:
    budgets = metadata.get("budgets")
    if not isinstance(budgets, dict):
        raise ValueError("Budget metadata has invalid budgets field")
    previous: set[int] = set()
    for name, record in budgets.items():
        if not isinstance(record, dict):
            raise ValueError(f"Budget {name!r} has invalid metadata")
        indices = record.get("indices")
        if not isinstance(indices, list):
            raise ValueError(f"Budget {name!r} is missing indices")
        current = {int(sample_id) for sample_id in indices}
        if len(current) != len(indices):
            raise ValueError(f"Budget {name!r} contains duplicated sample IDs")
        if name != "Bfull" and not previous.issubset(current):
            raise ValueError(f"Budget {name!r} is not nested in previous budget")
        previous = current
