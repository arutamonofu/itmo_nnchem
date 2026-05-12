from __future__ import annotations

import argparse

from src.budgets import resolve_requested_budgets
from src.project_data import ensure_budgets_metadata


def add_budget_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--budget", default=None, help="Training budget name, e.g. B500 or Bfull.")
    group.add_argument(
        "--budgets",
        default=None,
        help="Comma-separated budget names, or 'all'. Defaults to Bfull when omitted.",
    )


def requested_budget_names(args: argparse.Namespace) -> list[str]:
    metadata = ensure_budgets_metadata(split_strategy=args.split_strategy)
    return resolve_requested_budgets(
        metadata=metadata,
        budget=args.budget,
        budgets=args.budgets,
    )
