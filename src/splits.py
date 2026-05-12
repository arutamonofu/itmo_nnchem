from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_io import ensure_parent, save_json


SPLIT_NAMES = ("train", "val", "test")
MAIN_SPLIT_STRATEGIES = ("random_iid", "element_set")
TARGET_QUANTILES = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)


@dataclass(frozen=True)
class SplitConfig:
    random_seed: int
    train_size: float
    val_size: float
    test_size: float

    @classmethod
    def from_project_config(cls, config: dict[str, object]) -> "SplitConfig":
        val_size = float(config["val_size"])
        test_size = float(config["test_size"])
        train_size = 1.0 - val_size - test_size
        if train_size <= 0:
            raise ValueError("train/val/test fractions must sum to less than or equal to 1")
        return cls(
            random_seed=int(config["random_seed"]),
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )


def add_chemistry_keys(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"sample_id", "structure", "target"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    enriched = df.copy()
    enriched["formula"] = [
        structure.composition.reduced_formula for structure in enriched["structure"]
    ]
    enriched["element_set_key"] = [
        "-".join(sorted(element.symbol for element in structure.composition.elements))
        for structure in enriched["structure"]
    ]
    return enriched


def build_random_iid_split(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    sample_ids = df["sample_id"].astype(int).to_numpy()
    train_val_ids, test_ids = train_test_split(
        sample_ids,
        test_size=config.test_size,
        random_state=config.random_seed,
        shuffle=True,
    )
    val_fraction_of_train_val = config.val_size / (config.train_size + config.val_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_fraction_of_train_val,
        random_state=config.random_seed,
        shuffle=True,
    )
    return assignments_from_ids(
        split_strategy="random_iid",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )


def build_element_set_split(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    group_sizes = (
        df.groupby("element_set_key", observed=True)["sample_id"]
        .size()
        .reset_index(name="n_samples")
    )
    rng = np.random.default_rng(config.random_seed)
    group_sizes["tie_breaker"] = rng.permutation(len(group_sizes))
    group_sizes = group_sizes.sort_values(
        ["n_samples", "tie_breaker"], ascending=[False, True]
    ).reset_index(drop=True)

    n_total = int(group_sizes["n_samples"].sum())
    target_sizes = {
        "test": config.test_size * n_total,
        "val": config.val_size * n_total,
        "train": config.train_size * n_total,
    }
    assigned_sizes = {split: 0 for split in SPLIT_NAMES}
    key_to_split: dict[str, str] = {}

    for row in group_sizes.itertuples(index=False):
        group_size = int(row.n_samples)
        deficits = {
            split: target_sizes[split] - assigned_sizes[split]
            for split in SPLIT_NAMES
        }
        split = max(SPLIT_NAMES, key=lambda name: (deficits[name], -assigned_sizes[name]))
        key_to_split[str(row.element_set_key)] = split
        assigned_sizes[split] += group_size

    assignment = df[["sample_id", "element_set_key"]].copy()
    assignment["split"] = assignment["element_set_key"].map(key_to_split)
    assignment["split_strategy"] = "element_set"
    return assignment[["sample_id", "split", "split_strategy"]].astype(
        {"sample_id": int}
    )


def assignments_from_ids(
    *,
    split_strategy: str,
    train_ids: Iterable[int],
    val_ids: Iterable[int],
    test_ids: Iterable[int],
) -> pd.DataFrame:
    rows = []
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        rows.extend(
            {
                "sample_id": int(sample_id),
                "split": split,
                "split_strategy": split_strategy,
            }
            for sample_id in ids
        )
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def validate_assignment(
    *,
    assignment: pd.DataFrame,
    all_sample_ids: Iterable[int],
    split_strategy: str,
) -> None:
    required_columns = {"sample_id", "split", "split_strategy"}
    missing = required_columns - set(assignment.columns)
    if missing:
        raise ValueError(f"{split_strategy} assignment is missing columns: {sorted(missing)}")

    invalid_splits = set(assignment["split"]) - set(SPLIT_NAMES)
    if invalid_splits:
        raise ValueError(f"{split_strategy} has invalid split labels: {sorted(invalid_splits)}")

    invalid_strategies = set(assignment["split_strategy"]) - {split_strategy}
    if invalid_strategies:
        raise ValueError(
            f"{split_strategy} has invalid split_strategy values: {sorted(invalid_strategies)}"
        )

    ids = assignment["sample_id"].astype(int)
    if ids.duplicated().any():
        duplicated = sorted(ids[ids.duplicated()].unique())[:10]
        raise ValueError(f"{split_strategy} has duplicated sample_id values: {duplicated}")

    expected_ids = set(map(int, all_sample_ids))
    actual_ids = set(ids)
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)[:10]
        extra_ids = sorted(actual_ids - expected_ids)[:10]
        raise ValueError(
            f"{split_strategy} must cover every sample exactly once; "
            f"missing={missing_ids}, extra={extra_ids}"
        )


def validate_element_set_assignment(df: pd.DataFrame, assignment: pd.DataFrame) -> None:
    merged = assignment.merge(df[["sample_id", "element_set_key"]], on="sample_id", how="left")
    split_sets = {
        split: set(merged.loc[merged["split"] == split, "element_set_key"])
        for split in SPLIT_NAMES
    }
    overlaps = {
        f"{left}/{right}": sorted(split_sets[left] & split_sets[right])[:10]
        for left, right in combinations(SPLIT_NAMES, 2)
        if split_sets[left] & split_sets[right]
    }
    if overlaps:
        raise ValueError(f"element_set split leaks element_set_key across splits: {overlaps}")


def make_target_bins(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    target = df["target"].astype(float)
    thresholds = {
        "q05": float(target.quantile(0.05)),
        "q10": float(target.quantile(0.10)),
        "q90": float(target.quantile(0.90)),
        "q95": float(target.quantile(0.95)),
    }

    bins = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(int),
            "target": target,
            "target_bin_10": np.select(
                [target <= thresholds["q10"], target >= thresholds["q90"]],
                ["low_10", "high_10"],
                default="middle_80",
            ),
            "target_bin_5": np.select(
                [target <= thresholds["q05"], target >= thresholds["q95"]],
                ["low_5", "high_5"],
                default="middle_90",
            ),
        }
    )
    return bins.sort_values("sample_id").reset_index(drop=True), thresholds


def write_split_files(
    *,
    assignment: pd.DataFrame,
    split_strategy: str,
    output_dir: Path,
    metadata: dict[str, object],
) -> None:
    strategy_dir = output_dir / split_strategy
    for split in SPLIT_NAMES:
        split_df = assignment.loc[assignment["split"] == split].sort_values("sample_id")
        path = strategy_dir / f"{split}.csv"
        ensure_parent(path)
        split_df[["sample_id", "split", "split_strategy"]].to_csv(path, index=False)
    save_json(metadata, strategy_dir / "split_metadata.json")


def target_summary(values: pd.Series) -> dict[str, float]:
    quantiles = values.quantile(TARGET_QUANTILES)
    summary = {
        "target_mean": float(values.mean()),
        "target_std": float(values.std(ddof=1)),
        "target_min": float(values.min()),
        "target_max": float(values.max()),
    }
    for quantile, value in quantiles.items():
        summary[f"target_q{int(quantile * 100):02d}"] = float(value)
    return summary


def build_split_diagnostics(
    *,
    df: pd.DataFrame,
    assignments: dict[str, pd.DataFrame],
    target_bins: pd.DataFrame,
) -> pd.DataFrame:
    diagnostic_rows: list[dict[str, object]] = []
    target_bin_lookup = target_bins.set_index("sample_id")

    for split_strategy, assignment in assignments.items():
        enriched = assignment.merge(
            df[["sample_id", "target", "formula", "element_set_key"]],
            on="sample_id",
            how="left",
        ).merge(
            target_bin_lookup[["target_bin_10"]],
            left_on="sample_id",
            right_index=True,
            how="left",
        )
        split_frames = {
            split: enriched.loc[enriched["split"] == split].copy()
            for split in SPLIT_NAMES
        }
        formula_sets = {
            split: set(split_frames[split]["formula"])
            for split in SPLIT_NAMES
        }
        element_set_sets = {
            split: set(split_frames[split]["element_set_key"])
            for split in SPLIT_NAMES
        }
        train_element_keys = element_set_sets["train"]

        for split, split_df in split_frames.items():
            row: dict[str, object] = {
                "split_strategy": split_strategy,
                "split": split,
                "n_samples": int(len(split_df)),
                "n_unique_formulas": int(split_df["formula"].nunique()),
                "n_unique_element_sets": int(split_df["element_set_key"].nunique()),
                "n_low_10": int((split_df["target_bin_10"] == "low_10").sum()),
                "n_middle_80": int((split_df["target_bin_10"] == "middle_80").sum()),
                "n_high_10": int((split_df["target_bin_10"] == "high_10").sum()),
            }
            row.update(target_summary(split_df["target"]))

            if split == "test":
                row["share_test_element_sets_seen_in_train"] = float(
                    split_df["element_set_key"].isin(train_element_keys).mean()
                )
                row["n_test_elements_absent_from_train"] = int(
                    len(elements_in_keys(element_set_sets["test"]) - elements_in_keys(train_element_keys))
                )
            else:
                row["share_test_element_sets_seen_in_train"] = np.nan
                row["n_test_elements_absent_from_train"] = np.nan

            for left, right in combinations(SPLIT_NAMES, 2):
                row[f"formula_overlap_{left}_{right}"] = int(
                    len(formula_sets[left] & formula_sets[right])
                )
                row[f"element_set_overlap_{left}_{right}"] = int(
                    len(element_set_sets[left] & element_set_sets[right])
                )

            diagnostic_rows.append(row)

    return pd.DataFrame(diagnostic_rows)


def elements_in_keys(keys: set[str]) -> set[str]:
    return {element for key in keys for element in key.split("-") if element}

