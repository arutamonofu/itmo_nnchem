from __future__ import annotations

from perovskite_screening.data.budgets import build_budget_metadata, generate_training_budgets


def names(full_train_size: int) -> list[str]:
    return [str(item["name"]) for item in generate_training_budgets(full_train_size)]


def test_generate_training_budgets_expected_sequences() -> None:
    assert names(15142) == ["B500", "B1000", "B2000", "B4000", "B8000", "Bfull"]
    assert names(9200) == ["B500", "B1000", "B2000", "B4000", "Bfull"]
    assert names(3600) == ["B500", "B1000", "B2000", "Bfull"]
    assert names(500) == ["Bfull"]
    assert names(499) == ["Bfull"]


def test_budget_metadata_subsets_are_nested_and_deterministic() -> None:
    train_ids = list(range(15142))
    metadata_a = build_budget_metadata(train_ids=train_ids, split_seed=42)
    metadata_b = build_budget_metadata(train_ids=train_ids, split_seed=42)

    assert metadata_a == metadata_b
    assert metadata_a["full_train_size"] == len(train_ids)

    budgets = metadata_a["budgets"]
    previous: set[int] = set()
    for name in ["B500", "B1000", "B2000", "B4000", "B8000"]:
        record = budgets[name]
        indices = [int(sample_id) for sample_id in record["indices"]]
        assert len(indices) == int(record["n_samples"])
        assert len(indices) == len(set(indices))
        assert previous.issubset(set(indices))
        previous = set(indices)

    full_indices = [int(sample_id) for sample_id in budgets["Bfull"]["indices"]]
    assert set(full_indices) == set(train_ids)
    assert previous.issubset(set(full_indices))
