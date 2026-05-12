from __future__ import annotations

import pandas as pd

from perovskite_screening.data.splits import (
    SplitConfig,
    build_random_iid_split,
    validate_split_assignment,
)


def test_random_iid_split_assigns_each_sample_once() -> None:
    df = pd.DataFrame({"sample_id": range(20), "structure": [None] * 20, "target": range(20)})
    assignment = build_random_iid_split(
        df,
        SplitConfig(random_seed=42, train_size=0.8, val_size=0.1, test_size=0.1),
    )
    validate_split_assignment(
        assignment=assignment,
        all_sample_ids=df["sample_id"],
        split_strategy="random_iid",
    )
    assert set(assignment["split"]) == {"train", "val", "test"}
