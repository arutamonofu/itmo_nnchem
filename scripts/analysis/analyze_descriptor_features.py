from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_io import project_path
from src.metrics import compute_regression_metrics
from src.project_data import make_train_val_test_dataframes

from scripts.models.train_descriptor_baseline import build_model, featurize


TRAIN_FRACTIONS = [0.025, 0.25, 1.0]
MODELS = ["rf", "xgb"]
FEATURE_SET = "expanded"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descriptor diagnostics: feature importance, correlations, and correlation filtering."
    )
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Permutation importance repeats. Keep small on Windows to avoid large temp files.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=1000,
        help="Subsample validation set for permutation importance. Use 0 for the full validation set.",
    )
    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Skip permutation importance and only run correlation/filtering analysis.",
    )
    return parser.parse_args()


def find_correlated_pairs(x_train: pd.DataFrame, threshold: float) -> pd.DataFrame:
    corr = x_train.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    rows = []
    for feature_1 in upper.index:
        for feature_2 in upper.columns:
            value = upper.loc[feature_1, feature_2]
            if pd.notna(value) and value >= threshold:
                rows.append(
                    {
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "abs_corr": float(value),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["feature_1", "feature_2", "abs_corr"])

    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def choose_corr_filtered_features(x_train: pd.DataFrame, threshold: float) -> list[str]:
    """
    Greedy correlation filter fitted only on train features.

    For each highly correlated pair, remove the feature with larger average
    absolute correlation to the rest of the feature set.
    """
    corr = x_train.corr(numeric_only=True).abs()
    remaining = list(corr.columns)

    while True:
        corr_remaining = corr.loc[remaining, remaining]
        upper = corr_remaining.where(np.triu(np.ones(corr_remaining.shape), k=1).astype(bool))
        max_corr = upper.max().max()

        if pd.isna(max_corr) or max_corr < threshold:
            break

        pair_location = np.where(upper == max_corr)
        feature_a = upper.index[pair_location[0][0]]
        feature_b = upper.columns[pair_location[1][0]]

        mean_corr_a = corr_remaining[feature_a].drop(feature_a).mean()
        mean_corr_b = corr_remaining[feature_b].drop(feature_b).mean()

        feature_to_drop = feature_a if mean_corr_a >= mean_corr_b else feature_b
        remaining.remove(feature_to_drop)

    return remaining


def get_tree_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    regressor = model[-1]
    importances = getattr(regressor, "feature_importances_", None)

    if importances is None:
        return pd.DataFrame(columns=["feature", "tree_importance"])

    return (
        pd.DataFrame({"feature": feature_names, "tree_importance": importances})
        .sort_values("tree_importance", ascending=False)
        .reset_index(drop=True)
    )


def maybe_subsample_validation(
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    max_val_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if max_val_samples <= 0 or len(x_val) <= max_val_samples:
        return x_val, y_val

    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(x_val), size=max_val_samples, replace=False))
    return x_val.iloc[positions], y_val[positions]


def run_feature_diagnostics(args: argparse.Namespace) -> None:
    train_df, val_df, _test_df = make_train_val_test_dataframes(train_fraction=1.0)

    x_train = featurize(train_df, FEATURE_SET)
    x_val = featurize(val_df, FEATURE_SET)
    y_train = train_df["target"].to_numpy()
    y_val = val_df["target"].to_numpy()

    x_perm, y_perm = maybe_subsample_validation(x_val, y_val, args.max_val_samples, SEED)

    out_dir = project_path("results", "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    correlated_pairs = find_correlated_pairs(x_train, args.corr_threshold)
    correlated_pairs.to_csv(out_dir / "descriptor_correlated_pairs.csv", index=False)

    kept_features = choose_corr_filtered_features(x_train, args.corr_threshold)
    removed_features = sorted(set(x_train.columns) - set(kept_features))

    pd.DataFrame({"kept_feature": kept_features}).to_csv(
        out_dir / "descriptor_corr095_kept_features.csv", index=False
    )
    pd.DataFrame({"removed_feature": removed_features}).to_csv(
        out_dir / "descriptor_corr095_removed_features.csv", index=False
    )

    print(f"Expanded features: {x_train.shape[1]}")
    print(f"Correlated pairs with abs(corr) >= {args.corr_threshold}: {len(correlated_pairs)}")
    print(f"Features kept after corr filter: {len(kept_features)}")
    print(f"Features removed after corr filter: {len(removed_features)}")
    print()

    if len(correlated_pairs) > 0:
        print("Top correlated pairs:")
        print(correlated_pairs.head(20).to_string(index=False))
        print()

    for model_kind in MODELS:
        model = build_model(model_kind, SEED)
        model.fit(x_train, y_train)

        tree_imp = get_tree_feature_importance(model, list(x_train.columns))
        tree_imp.to_csv(out_dir / f"descriptor_{model_kind}_expanded_tree_importance.csv", index=False)

        print(f"Top 15 tree importances for {model_kind.upper()} + expanded:")
        print(tree_imp.head(15).to_string(index=False))
        print()

        if args.skip_permutation:
            continue

        perm = permutation_importance(
            model,
            x_perm,
            y_perm,
            n_repeats=args.n_repeats,
            random_state=SEED,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )

        perm_imp = (
            pd.DataFrame(
                {
                    "feature": x_train.columns,
                    "permutation_importance_mae_mean": perm.importances_mean,
                    "permutation_importance_mae_std": perm.importances_std,
                }
            )
            .sort_values("permutation_importance_mae_mean", ascending=False)
            .reset_index(drop=True)
        )
        perm_imp.to_csv(out_dir / f"descriptor_{model_kind}_expanded_permutation_importance_val.csv", index=False)

        print(f"Top 15 validation permutation importances for {model_kind.upper()} + expanded:")
        print(perm_imp.head(15).to_string(index=False))
        print()


def compare_corr_filtered_features(args: argparse.Namespace) -> None:
    rows = []

    for train_fraction in TRAIN_FRACTIONS:
        train_df, val_df, test_df = make_train_val_test_dataframes(train_fraction=train_fraction)

        x_train_full = featurize(train_df, FEATURE_SET)
        x_val_full = featurize(val_df, FEATURE_SET)
        x_test_full = featurize(test_df, FEATURE_SET)

        kept_features = choose_corr_filtered_features(x_train_full, args.corr_threshold)

        feature_variants = {
            "expanded": list(x_train_full.columns),
            "expanded_corr095": kept_features,
        }

        y_train = train_df["target"].to_numpy()
        y_val = val_df["target"].to_numpy()
        y_test = test_df["target"].to_numpy()

        for feature_variant, features in feature_variants.items():
            x_train = x_train_full[features]
            x_val = x_val_full[features]
            x_test = x_test_full[features]

            for model_kind in MODELS:
                model = build_model(model_kind, SEED)
                model.fit(x_train, y_train)

                train_metrics = compute_regression_metrics(y_train, model.predict(x_train))
                val_metrics = compute_regression_metrics(y_val, model.predict(x_val))
                test_metrics = compute_regression_metrics(y_test, model.predict(x_test))

                rows.append(
                    {
                        "model_kind": model_kind,
                        "feature_variant": feature_variant,
                        "train_fraction": train_fraction,
                        "n_features": len(features),
                        "train_mae": train_metrics["mae"],
                        "val_mae": val_metrics["mae"],
                        "test_mae": test_metrics["mae"],
                        "train_r2": train_metrics["r2"],
                        "val_r2": val_metrics["r2"],
                        "test_r2": test_metrics["r2"],
                    }
                )

    comparison = pd.DataFrame(rows).sort_values(["train_fraction", "test_mae"])
    out_path = project_path("results", "analysis", "descriptor_corr_filter_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_path, index=False)

    print("Comparison: expanded vs expanded_corr095")
    print(comparison.to_string(index=False))
    print()
    print(f"Saved to: {out_path}")


def main() -> None:
    args = parse_args()
    run_feature_diagnostics(args)
    compare_corr_filtered_features(args)


if __name__ == "__main__":
    main()
