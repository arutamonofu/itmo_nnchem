from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_io import project_path
from src.metrics import compute_regression_metrics
from src.project_data import make_train_val_test_dataframes
from src.result_schema import make_result_row, ordered_result_frame


ALLOWED_TRAIN_FRACTIONS = [0.025, 0.25, 1.0]
ALLOWED_MODELS = ["rf", "xgb"]
ALLOWED_FEATURE_SETS = ["starter", "expanded"]

MODEL_FAMILY = "descriptor_baseline"
RESULT_FILE = "descriptor_baseline.csv"
DIAGNOSTICS_FILE = "descriptor_diagnostics.csv"


def fraction_to_name(train_fraction: float) -> str:
    return str(float(train_fraction)).replace(".", "_")


def model_name(model_kind: str, feature_set: str) -> str:
    return f"descriptor_{model_kind}_{feature_set}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descriptor baseline: RandomForest / XGBoost on structure-composition descriptors"
    )
    parser.add_argument("--train-fraction", type=float, required=True, choices=ALLOWED_TRAIN_FRACTIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=ALLOWED_MODELS, default="rf")
    parser.add_argument("--feature-set", choices=ALLOWED_FEATURE_SETS, default="starter")
    return parser.parse_args()


def load_data(train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        return make_train_val_test_dataframes(train_fraction=train_fraction)
    except FileNotFoundError as exc:
        message = str(exc)
        if "dataset.pkl" in message:
            raise FileNotFoundError(
                "data/processed/dataset.pkl not found. Run python scripts/01_load_dataset.py first."
            ) from exc
        raise FileNotFoundError("Split files not found. Run python scripts/02_make_splits.py first.") from exc


def _as_float(value) -> float:
    """Convert pymatgen values to float; return nan if a descriptor is unavailable"""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _weighted_stats(values: np.ndarray, weights: np.ndarray, prefix: str) -> dict[str, float]:
    """Weighted mean/std/min/max/range for elemental properties"""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_range": float("nan"),
        }

    values = values[valid]
    weights = weights[valid]
    weights = weights / weights.sum()

    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": float(np.sqrt(max(var, 0.0))),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
    }


def _composition_entropy(amounts: np.ndarray) -> float:
    amounts = np.asarray(amounts, dtype=float)
    amounts = amounts[amounts > 0]
    if len(amounts) == 0:
        return float("nan")
    p = amounts / amounts.sum()
    return float(-np.sum(p * np.log(p)))


def starter_features(structure) -> dict[str, float]:
    """The six descriptors from the starter baseline"""
    composition = structure.composition
    atomic_numbers = np.array([element.Z for element in composition.elements], dtype=float)

    return {
        "n_sites": float(structure.num_sites),
        "volume": float(structure.volume),
        "density": float(structure.density),
        "num_unique_elements": float(len(composition.elements)),
        "mean_atomic_number": float(atomic_numbers.mean()),
        "std_atomic_number": float(atomic_numbers.std()),
    }


def expanded_features(structure) -> dict[str, float]:
    """
    Expanded descriptor set

    It keeps the original starter descriptors and adds:
    - lattice/volume descriptors;
    - composition entropy;
    - weighted elemental statistics for several periodic-table properties
    """
    composition = structure.composition
    elements = list(composition.elements)
    amounts = np.array([composition[element] for element in elements], dtype=float)

    features = dict(starter_features(structure))

    lattice = structure.lattice
    features.update(
        {
            "volume_per_site": float(structure.volume / structure.num_sites),
            "lattice_a": float(lattice.a),
            "lattice_b": float(lattice.b),
            "lattice_c": float(lattice.c),
            "lattice_alpha": float(lattice.alpha),
            "lattice_beta": float(lattice.beta),
            "lattice_gamma": float(lattice.gamma),
            "composition_entropy": _composition_entropy(amounts),
            "max_element_fraction": float((amounts / amounts.sum()).max()),
            "min_element_fraction": float((amounts / amounts.sum()).min()),
        }
    )

    elemental_properties = {
        "atomic_number": [element.Z for element in elements],
        "atomic_mass": [_as_float(element.atomic_mass) for element in elements],
        "electronegativity": [_as_float(element.X) for element in elements],
        "atomic_radius": [_as_float(element.atomic_radius) for element in elements],
        "mendeleev_no": [_as_float(element.mendeleev_no) for element in elements],
        "periodic_row": [_as_float(element.row) for element in elements],
        "periodic_group": [_as_float(element.group) for element in elements],
    }

    for prop_name, values in elemental_properties.items():
        features.update(_weighted_stats(np.array(values, dtype=float), amounts, prop_name))

    return features


def featurize(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "starter":
        rows = [starter_features(structure) for structure in df["structure"]]
    elif feature_set == "expanded":
        rows = [expanded_features(structure) for structure in df["structure"]]
    else:
        raise ValueError(f"Unknown feature_set={feature_set!r}")

    return pd.DataFrame(rows, index=df.index)


def build_model(model_kind: str, seed: int):
    if model_kind == "rf":
        regressor = RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        )
    elif model_kind == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install it inside the active conda environment:\n"
                "pip install xgboost"
            ) from exc

        regressor = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        raise ValueError(f"Unknown model={model_kind!r}")

    return make_pipeline(SimpleImputer(strategy="median"), regressor)


def save_predictions(
    *,
    run_model_name: str,
    sample_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_fraction: float,
    seed: int,
) -> str:
    prediction_rel_path = (
        f"results/predictions/{run_model_name}_{fraction_to_name(train_fraction)}_seed{seed}.csv"
    )
    prediction_path = project_path(*prediction_rel_path.split("/"))
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "sample_id": sample_ids.astype(int).to_numpy(),
            "split": "test",
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(prediction_path, index=False)

    return prediction_rel_path


def save_result_row(row: dict[str, object]) -> Path:
    result_path = project_path("results", "raw", RESULT_FILE)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([row])
    if result_path.exists():
        old_df = pd.read_csv(result_path)
        duplicate = (
            (old_df["model"] == row["model"])
            & (old_df["train_fraction"].astype(float) == float(row["train_fraction"]))
            & (old_df["seed"].astype(int) == int(row["seed"]))
            & (old_df["split_id"] == row["split_id"])
        )
        new_df = pd.concat([old_df.loc[~duplicate], new_df], ignore_index=True)

    ordered_result_frame(new_df).to_csv(result_path, index=False)
    return result_path


def save_diagnostics_row(row: dict[str, object]) -> Path:
    diagnostics_path = project_path("results", "analysis", DIAGNOSTICS_FILE)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([row])
    if diagnostics_path.exists():
        old_df = pd.read_csv(diagnostics_path)
        duplicate = (
            (old_df["model"] == row["model"])
            & (old_df["feature_set"] == row["feature_set"])
            & (old_df["train_fraction"].astype(float) == float(row["train_fraction"]))
            & (old_df["seed"].astype(int) == int(row["seed"]))
        )
        new_df = pd.concat([old_df.loc[~duplicate], new_df], ignore_index=True)

    new_df.to_csv(diagnostics_path, index=False)
    return diagnostics_path


def make_notes(model_kind: str, feature_set: str, n_features: int) -> str:
    return (
        f"{model_kind.upper()} descriptor baseline with feature_set={feature_set}; "
        f"n_features={n_features}. Test metrics are reported in required result columns."
    )


def print_metrics_table(
    *,
    run_model_name: str,
    feature_set: str,
    train_fraction: float,
    n_features: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    predictions_path: str,
    result_path: Path,
    diagnostics_path: Path,
) -> None:
    print(f"Model: {run_model_name}")
    print(f"Feature set: {feature_set}")
    print(f"Train fraction: {train_fraction}")
    print(f"Number of descriptors: {n_features}")
    print()
    print("Metrics:")
    print(
        pd.DataFrame(
            [
                {"split": "train", **train_metrics},
                {"split": "val", **val_metrics},
                {"split": "test", **test_metrics},
            ]
        ).to_string(index=False)
    )
    print()
    print(f"Saved test predictions to: {predictions_path}")
    print(f"Saved required result row to: {result_path}")
    print(f"Saved train/val/test diagnostics to: {diagnostics_path}")


def main() -> None:
    args = parse_args()

    run_model_name = model_name(args.model, args.feature_set)

    train_df, val_df, test_df = load_data(args.train_fraction)

    x_train = featurize(train_df, args.feature_set)
    x_val = featurize(val_df, args.feature_set)
    x_test = featurize(test_df, args.feature_set)

    y_train = train_df["target"].to_numpy()
    y_val = val_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    model = build_model(args.model, args.seed)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    train_metrics = compute_regression_metrics(y_train, train_pred)
    val_metrics = compute_regression_metrics(y_val, val_pred)
    test_metrics = compute_regression_metrics(y_test, test_pred)

    predictions_path = save_predictions(
        run_model_name=run_model_name,
        sample_ids=test_df["sample_id"],
        y_true=y_test,
        y_pred=test_pred,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    result_row = make_result_row(
        model=run_model_name,
        model_family=MODEL_FAMILY,
        train_fraction=args.train_fraction,
        seed=args.seed,
        mae=test_metrics["mae"],
        rmse=test_metrics["rmse"],
        r2=test_metrics["r2"],
        predictions_path=predictions_path,
        notes=make_notes(args.model, args.feature_set, n_features=x_train.shape[1]),
    )
    result_path = save_result_row(result_row)

    diagnostics_row = {
        "model": run_model_name,
        "model_kind": args.model,
        "feature_set": args.feature_set,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "n_features": x_train.shape[1],
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "train_r2": train_metrics["r2"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "predictions_path": predictions_path,
    }
    diagnostics_path = save_diagnostics_row(diagnostics_row)

    print_metrics_table(
        run_model_name=run_model_name,
        feature_set=args.feature_set,
        train_fraction=args.train_fraction,
        n_features=x_train.shape[1],
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        predictions_path=predictions_path,
        result_path=result_path,
        diagnostics_path=diagnostics_path,
    )


if __name__ == "__main__":
    main()
