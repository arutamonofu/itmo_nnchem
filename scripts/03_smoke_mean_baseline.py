from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_io import load_config, load_indices, project_path
from src.metrics import compute_regression_metrics
from src.result_schema import ordered_result_frame


FRACTION_TO_SPLIT_FILE = {
    0.025: "train_2_5_indices.csv",
    0.25: "train_25_indices.csv",
    1.0: "train_100_indices.csv",
}

FRACTION_TO_FILE_TOKEN = {
    0.025: "0_025",
    0.25: "0_25",
    1.0: "1_0",
}


def main() -> None:
    config = load_config()
    seed = int(config["random_seed"])

    dataset_path = project_path("data", "processed", "dataset.pkl")
    if not dataset_path.exists():
        raise FileNotFoundError("Missing data/processed/dataset.pkl. Run scripts/01_load_dataset.py first.")

    df = pd.read_pickle(dataset_path).set_index("sample_id")
    split_dir = project_path("data", "splits")
    test_ids = load_indices(split_dir / "test_indices.csv")
    val_ids = load_indices(split_dir / "val_indices.csv")
    y_test = df.loc[test_ids, "target"].to_numpy()

    rows = []
    for fraction in config["train_fractions"]:
        fraction = float(fraction)
        train_ids = load_indices(split_dir / FRACTION_TO_SPLIT_FILE[fraction])
        train_mean = float(df.loc[train_ids, "target"].mean())
        y_pred = np.full(shape=len(test_ids), fill_value=train_mean)
        metrics = compute_regression_metrics(y_test, y_pred)

        prediction_rel_path = f"results/predictions/mean_baseline_{FRACTION_TO_FILE_TOKEN[fraction]}_seed{seed}.csv"
        prediction_path = project_path(*prediction_rel_path.split("/"))
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "sample_id": test_ids,
                "split": "test",
                "y_true": y_test,
                "y_pred": y_pred,
            }
        ).to_csv(prediction_path, index=False)

        rows.append(
            {
                "model": "mean_baseline",
                "model_family": "dummy",
                "train_fraction": fraction,
                "seed": seed,
                "split_id": config["split_id"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "n_test": len(test_ids),
                "target_unit": config["target_unit"],
                "predictions_path": prediction_rel_path,
                "notes": "Constant prediction equal to the selected train subset target mean.",
            }
        )

    result_df = ordered_result_frame(pd.DataFrame(rows))
    result_path = project_path("results", "raw", "mean_baseline.csv")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(result_path, index=False)
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
