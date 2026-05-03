from __future__ import annotations

from collections.abc import Sequence


def _ensure_unique(name: str, values: Sequence[int]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicated sample_id values")


def _ensure_subset(child_name: str, child: set[int], parent_name: str, parent: set[int]) -> None:
    missing = child - parent
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(f"{child_name} must be a subset of {parent_name}; offending ids: {preview}")


def validate_splits(
    *,
    all_sample_ids: Sequence[int],
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
    train_2_5_ids: Sequence[int],
    train_25_ids: Sequence[int],
    train_100_ids: Sequence[int],
) -> None:
    named_lists = {
        "all_sample_ids": all_sample_ids,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "train_2_5": train_2_5_ids,
        "train_25": train_25_ids,
        "train_100": train_100_ids,
    }
    for name, values in named_lists.items():
        _ensure_unique(name, values)

    all_ids = set(all_sample_ids)
    train = set(train_ids)
    val = set(val_ids)
    test = set(test_ids)
    train_2_5 = set(train_2_5_ids)
    train_25 = set(train_25_ids)
    train_100 = set(train_100_ids)

    overlaps = {
        "train/val": train & val,
        "train/test": train & test,
        "val/test": val & test,
    }
    bad_overlaps = {name: sorted(ids)[:10] for name, ids in overlaps.items() if ids}
    if bad_overlaps:
        raise ValueError(f"Main splits overlap: {bad_overlaps}")

    assigned = train | val | test
    if assigned != all_ids:
        missing = sorted(all_ids - assigned)[:10]
        extra = sorted(assigned - all_ids)[:10]
        raise ValueError(f"Every sample must belong to exactly one main split; missing={missing}, extra={extra}")

    if len(train_ids) + len(val_ids) + len(test_ids) != len(all_sample_ids):
        raise ValueError("Dataset size in split files does not equal dataset size")

    _ensure_subset("train_2_5", train_2_5, "train", train)
    _ensure_subset("train_25", train_25, "train", train)

    if train_100 != train:
        raise ValueError("train_100 must equal the full train split")

    _ensure_subset("train_2_5", train_2_5, "train_25", train_25)
    _ensure_subset("train_25", train_25, "train_100", train_100)
