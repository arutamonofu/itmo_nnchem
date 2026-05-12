from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "dataset.pkl"
SPLIT_ROOT = ROOT / "data" / "splits"
SPLIT_STRATEGIES = ("random_iid", "element_set")
SPLIT_NAMES = ("train", "val", "test")
REQUIRED_BUDGETS = ("B500", "B2000", "B8000", "Bfull")
REQUIRED_CONFIGS = (
    ROOT / "configs" / "experiments" / "descriptor_xgb.yaml",
    ROOT / "configs" / "experiments" / "cgcnn.yaml",
    ROOT / "configs" / "experiments" / "matgl.yaml",
)
MAKE_SPLITS_COMMAND = "perovskite-screening make-splits --config configs/default.yaml"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path, errors: list[str]) -> dict[str, object] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        errors.append(
            f"{rel(path)} is not valid JSON: {exc}. Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
        return None
    if not isinstance(data, dict):
        errors.append(
            f"{rel(path)} must contain a JSON object. Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
        return None
    return data


def load_split_ids(path: Path, strategy: str, split_name: str, errors: list[str]) -> set[int]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive CLI reporting
        errors.append(
            f"Could not read {rel(path)}: {exc}. Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
        return set()

    expected_columns = ["sample_id", "split", "split_strategy"]
    if list(frame.columns) != expected_columns:
        errors.append(
            f"{rel(path)} must contain columns {expected_columns}. "
            f"Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
        return set()

    invalid_splits = sorted(set(frame["split"].astype(str)) - {split_name})
    if invalid_splits:
        errors.append(
            f"{rel(path)} contains split labels {invalid_splits}, expected only {split_name!r}. "
            f"Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
    invalid_strategies = sorted(set(frame["split_strategy"].astype(str)) - {strategy})
    if invalid_strategies:
        errors.append(
            f"{rel(path)} contains split_strategy values {invalid_strategies}, expected only {strategy!r}. "
            f"Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )

    sample_ids = frame["sample_id"].astype(int)
    duplicated = sorted(sample_ids[sample_ids.duplicated()].unique().tolist())
    if duplicated:
        errors.append(
            f"{rel(path)} contains duplicated sample_id values, for example {duplicated[:10]}. "
            f"Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
    return set(sample_ids.tolist())


def validate_split_files(strategy: str, errors: list[str]) -> None:
    split_ids: dict[str, set[int]] = {}
    for split_name in SPLIT_NAMES:
        path = SPLIT_ROOT / strategy / f"{split_name}.csv"
        if not path.exists():
            errors.append(
                f"Missing split file: {rel(path)}. Run `{MAKE_SPLITS_COMMAND}` before the final suite."
            )
            continue
        split_ids[split_name] = load_split_ids(path, strategy, split_name, errors)

    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            overlap = split_ids.get(left_name, set()) & split_ids.get(right_name, set())
            if overlap:
                errors.append(
                    f"{strategy} has overlapping sample_id values in {left_name}/{right_name}, "
                    f"for example {sorted(overlap)[:10]}. Recreate splits with `{MAKE_SPLITS_COMMAND}`."
                )


def as_index_set(record: object, budget_name: str, path: Path, errors: list[str]) -> set[int]:
    if not isinstance(record, dict):
        errors.append(
            f"{rel(path)} budget {budget_name} must be an object. "
            f"Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
        return set()
    indices = record.get("indices")
    if not isinstance(indices, list):
        errors.append(
            f"{rel(path)} budget {budget_name} is missing an indices list. "
            f"Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
        return set()
    current = {int(sample_id) for sample_id in indices}
    if len(current) != len(indices):
        errors.append(
            f"{rel(path)} budget {budget_name} contains duplicated sample_id values. "
            f"Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
    return current


def validate_budgets(strategy: str, errors: list[str]) -> None:
    path = SPLIT_ROOT / strategy / "budgets.json"
    if not path.exists():
        errors.append(
            f"Missing budget metadata: {rel(path)}. Run `{MAKE_SPLITS_COMMAND}` before the final suite."
        )
        return

    metadata = load_json(path, errors)
    if metadata is None:
        return

    budgets = metadata.get("budgets")
    if not isinstance(budgets, dict):
        errors.append(
            f"{rel(path)} is missing a budgets object. Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
        return

    missing = [name for name in REQUIRED_BUDGETS if name not in budgets]
    if missing:
        available = ", ".join(str(name) for name in budgets)
        errors.append(
            f"{rel(path)} is missing required budget(s): {', '.join(missing)}. "
            f"Available budgets: {available or 'none'}. Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )
        return

    bfull = budgets["Bfull"]
    if not isinstance(bfull, dict) or bfull.get("is_full") is not True:
        errors.append(
            f"{rel(path)} budget Bfull must have is_full = true. "
            f"Recreate splits with `{MAKE_SPLITS_COMMAND}`."
        )

    previous_name = REQUIRED_BUDGETS[0]
    previous = as_index_set(budgets[previous_name], previous_name, path, errors)
    for budget_name in REQUIRED_BUDGETS[1:]:
        current = as_index_set(budgets[budget_name], budget_name, path, errors)
        if previous and current and not previous.issubset(current):
            errors.append(
                f"{rel(path)} budgets are not nested: {previous_name} is not a subset of {budget_name}. "
                f"Recreate splits with `{MAKE_SPLITS_COMMAND}`."
            )
        previous_name = budget_name
        previous = current


def main() -> int:
    errors: list[str] = []

    if not DATASET_PATH.exists():
        errors.append(
            f"Missing processed dataset: {rel(DATASET_PATH)}. Run "
            "`perovskite-screening prepare-data --config configs/data/matbench_perovskites.yaml`."
        )

    for config_path in REQUIRED_CONFIGS:
        if not config_path.exists():
            errors.append(f"Missing experiment config: {rel(config_path)}. Restore the config file.")

    for strategy in SPLIT_STRATEGIES:
        validate_split_files(strategy, errors)
        validate_budgets(strategy, errors)

    if errors:
        print("Preflight validation failed for the reduced final protocol:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Reduced final protocol preflight passed.")
    for strategy in SPLIT_STRATEGIES:
        print(
            f"- {strategy}: split files present, train/val/test disjoint, "
            f"budgets present ({', '.join(REQUIRED_BUDGETS)})"
        )
    print("- configs: descriptor_xgb.yaml, cgcnn.yaml, matgl.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
