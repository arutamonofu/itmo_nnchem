from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_io import load_config, project_path, save_json
from src.budgets import (
    build_budgets_metadata,
    parse_training_budget_config,
    write_budgets_metadata,
)
from src.splits import (
    MAIN_SPLIT_STRATEGIES,
    SplitConfig,
    add_chemistry_keys,
    build_element_set_split,
    build_random_iid_split,
    build_split_diagnostics,
    make_target_bins,
    validate_assignment,
    validate_element_set_assignment,
    write_split_files,
)


def split_metadata(
    *,
    assignment: pd.DataFrame,
    split_strategy: str,
    split_config: SplitConfig,
    notes: str,
) -> dict[str, object]:
    counts = assignment["split"].value_counts()
    return {
        "split_strategy": split_strategy,
        "split_id": f"{split_strategy}_seed_{split_config.random_seed}_80_10_10",
        "random_seed": split_config.random_seed,
        "train_size": int(counts.get("train", 0)),
        "val_size": int(counts.get("val", 0)),
        "test_size": int(counts.get("test", 0)),
        "notes": notes,
    }


def main() -> None:
    config = load_config()
    split_config = SplitConfig.from_project_config(config)

    dataset_path = project_path("data", "processed", "dataset.pkl")
    if not dataset_path.exists():
        raise FileNotFoundError("Missing data/processed/dataset.pkl. Run scripts/01_load_dataset.py first.")

    df = add_chemistry_keys(pd.read_pickle(dataset_path))
    all_sample_ids = df["sample_id"].astype(int).tolist()
    split_dir = project_path("data", "splits")

    assignments = {
        "random_iid": build_random_iid_split(df, split_config),
        "element_set": build_element_set_split(df, split_config),
    }

    for split_strategy, assignment in assignments.items():
        validate_assignment(
            assignment=assignment,
            all_sample_ids=all_sample_ids,
            split_strategy=split_strategy,
        )
        if split_strategy == "element_set":
            validate_element_set_assignment(df, assignment)

    target_bins, thresholds = make_target_bins(df)
    target_dir = split_dir / "target_tails"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_bins.to_csv(target_dir / "target_bins.csv", index=False)
    save_json(thresholds, target_dir / "tail_thresholds.json")

    notes = {
        "random_iid": "Independent random split of individual samples.",
        "element_set": (
            "Split by unique set of chemical elements. Test contains unseen element "
            "combinations, not necessarily unseen individual elements."
        ),
    }
    budget_names_by_strategy: dict[str, list[str]] = {}
    for split_strategy in MAIN_SPLIT_STRATEGIES:
        assignment = assignments[split_strategy]
        metadata = split_metadata(
            assignment=assignment,
            split_strategy=split_strategy,
            split_config=split_config,
            notes=notes[split_strategy],
        )
        write_split_files(
            assignment=assignment,
            split_strategy=split_strategy,
            output_dir=split_dir,
            metadata=metadata,
        )
        train_ids = (
            assignment.loc[assignment["split"] == "train", "sample_id"]
            .astype(int)
            .sort_values()
            .tolist()
        )
        budget_config = parse_training_budget_config(config)
        budgets_metadata = build_budgets_metadata(
            train_ids=train_ids,
            split_seed=split_config.random_seed,
            base_budget=int(budget_config["base_budget"]),
            growth_factor=int(budget_config["growth_factor"]),
            min_final_growth_ratio=float(budget_config["min_final_growth_ratio"]),
            sampling_strategy=str(budget_config["sampling_strategy"]),
        )
        write_budgets_metadata(
            budgets_metadata,
            split_dir / split_strategy / "budgets.json",
        )
        budget_names_by_strategy[split_strategy] = list(budgets_metadata["budgets"].keys())

    diagnostics = build_split_diagnostics(
        df=df,
        assignments=assignments,
        target_bins=target_bins,
    )
    diagnostics.to_csv(split_dir / "split_diagnostics.csv", index=False)

    for split_strategy in MAIN_SPLIT_STRATEGIES:
        counts = assignments[split_strategy]["split"].value_counts()
        print(
            f"Created {split_strategy}: "
            f"train={int(counts.get('train', 0))}, "
            f"val={int(counts.get('val', 0))}, "
            f"test={int(counts.get('test', 0))}"
        )
        print(f"Created {split_strategy} budgets: {', '.join(budget_names_by_strategy[split_strategy])}")
    print(f"Created target tail labels for {len(target_bins)} samples")


if __name__ == "__main__":
    main()
