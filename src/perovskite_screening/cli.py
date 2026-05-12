from __future__ import annotations

import argparse
from pathlib import Path

from perovskite_screening.config import ProjectConfig
from perovskite_screening.data.budgets import load_budget_metadata, resolve_budgets
from perovskite_screening.io.paths import project_path
from perovskite_screening.io.results import collect_results, validate_results
from perovskite_screening.pipeline.make_splits import make_splits
from perovskite_screening.pipeline.prepare_data import prepare_data
from perovskite_screening.pipeline.run_experiment import run_experiment
from perovskite_screening.pipeline.run_suite import run_suite


def _config(path: str) -> ProjectConfig:
    return ProjectConfig.from_file(path)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perovskite-screening")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Load and normalize the canonical dataset.")
    prepare.add_argument("--config", default="configs/data/matbench_perovskites.yaml")

    splits = subparsers.add_parser("make-splits", help="Build fixed split and budget artifacts.")
    splits.add_argument("--config", default="configs/default.yaml")

    run = subparsers.add_parser("run", help="Run one experiment budget.")
    run.add_argument("--config", default="configs/experiments/descriptor_rf.yaml")
    run.add_argument("--split-strategy", default=None, choices=["random_iid", "element_set"])
    run.add_argument("--budget", default="Bfull")
    run.add_argument("--seed", type=int, default=None)

    suite = subparsers.add_parser("run-suite", help="Run an experiment over budgets and seeds.")
    suite.add_argument("--config", default="configs/experiments/descriptor_rf.yaml")
    suite.add_argument("--split-strategy", default=None, choices=["random_iid", "element_set"])
    suite.add_argument("--budgets", default="all")
    suite.add_argument("--seeds", default="42")

    collect = subparsers.add_parser("collect", help="Collect result CSV files into one summary.")
    collect.add_argument("--runs-dir", default="outputs/runs")
    collect.add_argument("--out", default="outputs/summary/results.csv")

    validate = subparsers.add_parser("validate-results", help="Validate result and prediction files.")
    validate.add_argument("--runs-dir", default="outputs/runs")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-data":
        result = prepare_data(_config(args.config))
        print(f"Loaded dataset with {result['n_samples']} samples")
        print(f"Wrote {result['paths']['processed']}")
        return

    if args.command == "make-splits":
        budget_names = make_splits(_config(args.config))
        for split_strategy, names in budget_names.items():
            print(f"Created {split_strategy} budgets: {', '.join(names)}")
        return

    if args.command == "run":
        config = _config(args.config)
        split_strategy = args.split_strategy or config.default_split_strategy
        seed = config.random_seed if args.seed is None else int(args.seed)
        metadata = load_budget_metadata(project_path("data", "splits", split_strategy, "budgets.json"))
        budget_name = resolve_budgets(metadata, args.budget)[0]
        row = run_experiment(
            config=config,
            split_strategy=split_strategy,
            budget_name=budget_name,
            seed=seed,
        )
        print(f"Wrote {row['result_path']}")
        return

    if args.command == "run-suite":
        config = _config(args.config)
        split_strategy = args.split_strategy or config.default_split_strategy
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
        rows = run_suite(
            config=config,
            split_strategy=split_strategy,
            budgets=args.budgets,
            seeds=seeds,
        )
        print(f"Completed {len(rows)} run(s)")
        return

    if args.command == "collect":
        summary = collect_results(_resolve_path(args.runs_dir), _resolve_path(args.out))
        print(f"Wrote {_resolve_path(args.out)} with {len(summary)} rows")
        return

    if args.command == "validate-results":
        total_rows = validate_results(_resolve_path(args.runs_dir))
        print(f"Validated {total_rows} result row(s)")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
