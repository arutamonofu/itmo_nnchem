from __future__ import annotations

from perovskite_screening.config import ProjectConfig
from perovskite_screening.data.budgets import load_budget_metadata, resolve_budgets
from perovskite_screening.io.paths import project_path
from perovskite_screening.pipeline.run_experiment import run_experiment


def run_suite(
    *,
    config: ProjectConfig,
    split_strategy: str,
    budgets: str,
    seeds: list[int],
) -> list[dict[str, object]]:
    metadata = load_budget_metadata(project_path("data", "splits", split_strategy, "budgets.json"))
    budget_names = resolve_budgets(metadata, budgets)
    rows = []
    for seed in seeds:
        for budget_name in budget_names:
            rows.append(
                run_experiment(
                    config=config,
                    split_strategy=split_strategy,
                    budget_name=budget_name,
                    seed=seed,
                )
            )
    return rows
