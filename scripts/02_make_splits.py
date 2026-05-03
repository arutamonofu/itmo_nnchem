from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_io import load_config, project_path, save_indices, save_json
from src.validation import validate_splits


def subset_name(fraction: float) -> str:
    return str(fraction).replace(".", "_").rstrip("0").rstrip("_")


def main() -> None:
    config = load_config()
    seed = int(config["random_seed"])
    test_size = float(config["test_size"])
    val_size = float(config["val_size"])

    dataset_path = project_path("data", "processed", "dataset.pkl")
    if not dataset_path.exists():
        raise FileNotFoundError("Missing data/processed/dataset.pkl. Run scripts/01_load_dataset.py first.")

    df = pd.read_pickle(dataset_path)
    all_ids = df["sample_id"].astype(int).tolist()
    n_total = len(all_ids)

    train_val_ids, test_ids = train_test_split(all_ids, test_size=test_size, random_state=seed, shuffle=True)
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_fraction_of_train_val,
        random_state=seed,
        shuffle=True,
    )

    rng = np.random.default_rng(seed)
    shuffled_train = list(rng.permutation(np.array(train_ids, dtype=int)))
    train_2_5_ids = shuffled_train[: round(0.025 * n_total)]
    train_25_ids = shuffled_train[: round(0.25 * n_total)]
    train_100_ids = shuffled_train

    validate_splits(
        all_sample_ids=all_ids,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        train_2_5_ids=train_2_5_ids,
        train_25_ids=train_25_ids,
        train_100_ids=train_100_ids,
    )

    split_dir = project_path("data", "splits")
    save_indices(train_ids, split_dir / "train_indices.csv")
    save_indices(val_ids, split_dir / "val_indices.csv")
    save_indices(test_ids, split_dir / "test_indices.csv")
    save_indices(train_2_5_ids, split_dir / "train_2_5_indices.csv")
    save_indices(train_25_ids, split_dir / "train_25_indices.csv")
    save_indices(train_100_ids, split_dir / "train_100_indices.csv")

    split_by_id = {sample_id: "train" for sample_id in train_ids}
    split_by_id.update({sample_id: "val" for sample_id in val_ids})
    split_by_id.update({sample_id: "test" for sample_id in test_ids})
    train_2_5 = set(train_2_5_ids)
    train_25 = set(train_25_ids)
    train_100 = set(train_100_ids)
    assignments = pd.DataFrame(
        {
            "sample_id": all_ids,
            "split": [split_by_id[sample_id] for sample_id in all_ids],
            "in_train_2_5": [sample_id in train_2_5 for sample_id in all_ids],
            "in_train_25": [sample_id in train_25 for sample_id in all_ids],
            "in_train_100": [sample_id in train_100 for sample_id in all_ids],
        }
    )
    assignments.to_csv(split_dir / "split_assignments.csv", index=False)

    save_json(
        {
            "split_id": config["split_id"],
            "random_seed": seed,
            "n_total": int(n_total),
            "n_train": int(len(train_ids)),
            "n_val": int(len(val_ids)),
            "n_test": int(len(test_ids)),
            "n_train_2_5": int(len(train_2_5_ids)),
            "n_train_25": int(len(train_25_ids)),
            "n_train_100": int(len(train_100_ids)),
        },
        split_dir / "split_info.json",
    )

    print(f"Created split {config['split_id']}: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")


if __name__ == "__main__":
    main()
