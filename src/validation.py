from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.splits import validate_assignment, validate_element_set_assignment


def validate_split_assignment(
    *,
    assignment: pd.DataFrame,
    all_sample_ids: Iterable[int],
    split_strategy: str,
) -> None:
    validate_assignment(
        assignment=assignment,
        all_sample_ids=all_sample_ids,
        split_strategy=split_strategy,
    )


def validate_grouped_element_set_split(df: pd.DataFrame, assignment: pd.DataFrame) -> None:
    validate_element_set_assignment(df, assignment)
