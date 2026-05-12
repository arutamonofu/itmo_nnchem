from __future__ import annotations

from perovskite_screening.config import ProjectConfig
from perovskite_screening.data.budgets import build_budget_metadata, parse_training_budget_config, save_budget_metadata
from perovskite_screening.data.dataset import load_dataset
from perovskite_screening.data.splits import (
    MAIN_SPLIT_STRATEGIES,
    SplitConfig,
    add_chemistry_keys,
    build_split,
    build_split_diagnostics,
    make_target_bins,
    save_split,
    split_metadata,
    validate_element_set_no_leakage,
    validate_split_assignment,
)
from perovskite_screening.io.artifacts import save_json
from perovskite_screening.io.paths import project_path


def make_splits(config: ProjectConfig) -> dict[str, list[str]]:
    df = add_chemistry_keys(load_dataset(index_by_sample_id=False))
    all_sample_ids = df["sample_id"].astype(int).tolist()
    split_config = SplitConfig.from_config(config.raw)
    split_dir = project_path("data", "splits")
    notes = {
        "random_iid": "Independent random split of individual samples.",
        "element_set": (
            "Split by unique set of chemical elements. Test contains unseen element "
            "combinations, not necessarily unseen individual elements."
        ),
    }
    assignments = {
        strategy: build_split(df, strategy=strategy, config=split_config)
        for strategy in MAIN_SPLIT_STRATEGIES
    }
    for strategy, assignment in assignments.items():
        validate_split_assignment(
            assignment=assignment,
            all_sample_ids=all_sample_ids,
            split_strategy=strategy,
        )
        if strategy == "element_set":
            validate_element_set_no_leakage(df, assignment)

    target_bins, thresholds = make_target_bins(df)
    target_dir = split_dir / "target_tails"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_bins.to_csv(target_dir / "target_bins.csv", index=False)
    save_json(thresholds, target_dir / "tail_thresholds.json")

    budget_config = parse_training_budget_config(config.raw)
    budget_names_by_strategy: dict[str, list[str]] = {}
    for strategy in MAIN_SPLIT_STRATEGIES:
        assignment = assignments[strategy]
        save_split(
            assignment=assignment,
            split_strategy=strategy,
            output_dir=split_dir,
            metadata=split_metadata(
                assignment=assignment,
                split_strategy=strategy,
                split_config=split_config,
                notes=notes[strategy],
            ),
        )
        train_ids = (
            assignment.loc[assignment["split"] == "train", "sample_id"]
            .astype(int)
            .sort_values()
            .tolist()
        )
        budget_metadata = build_budget_metadata(
            train_ids=train_ids,
            split_seed=split_config.random_seed,
            base_budget=int(budget_config["base_budget"]),
            growth_factor=int(budget_config["growth_factor"]),
            min_final_growth_ratio=float(budget_config["min_final_growth_ratio"]),
            sampling_strategy=str(budget_config["sampling_strategy"]),
        )
        save_budget_metadata(budget_metadata, split_dir / strategy / "budgets.json")
        budget_names_by_strategy[strategy] = list(budget_metadata["budgets"].keys())

    diagnostics = build_split_diagnostics(df=df, assignments=assignments, target_bins=target_bins)
    diagnostics.to_csv(split_dir / "split_diagnostics.csv", index=False)
    return budget_names_by_strategy
