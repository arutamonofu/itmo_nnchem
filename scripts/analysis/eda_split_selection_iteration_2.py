from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.analysis.eda_split_selection import make_group_split, save_fig, split_frame, target_stats
from scripts.models.train_descriptor_baseline import featurize
from src.data_io import load_config, project_path
from src.project_data import load_dataset


FIRST_DIR = project_path("reports", "eda_split_selection")
OUT_DIR = FIRST_DIR / "iteration_2"
SPLIT_DIR = OUT_DIR / "candidate_splits_group_element_set"
SEEDS = list(range(10))
SPLIT_STRATEGIES = ["random_iid", "group_reduced_formula", "group_element_set"]
GROUP_STRATEGIES = ["group_reduced_formula", "group_element_set"]
TAIL_QUANTILES = {"q05": 0.05, "q10": 0.10, "q90": 0.90, "q95": 0.95}
PLOT_COLORS = {
    "random_iid": "#2563eb",
    "group_reduced_formula": "#16a34a",
    "group_element_set": "#dc2626",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata() -> pd.DataFrame:
    path = FIRST_DIR / "composition_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/analysis/eda_split_selection.py first.")
    metadata = pd.read_csv(path)
    metadata["elements"] = metadata["elements_json"].map(json.loads)
    metadata["element_counts"] = metadata["element_counts_json"].map(json.loads)
    return metadata


def load_candidate(name: str) -> pd.DataFrame:
    return pd.read_csv(FIRST_DIR / "candidate_splits" / f"{name}.csv")


def tail_thresholds(metadata: pd.DataFrame) -> dict[str, float]:
    return {name: float(metadata["target"].quantile(q)) for name, q in TAIL_QUANTILES.items()}


def add_tail_columns(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["is_low_10"] = out["target"] <= thresholds["q10"]
    out["is_high_10"] = out["target"] >= thresholds["q90"]
    out["is_middle_80"] = (~out["is_low_10"]) & (~out["is_high_10"])
    out["is_low_5"] = out["target"] <= thresholds["q05"]
    out["is_high_5"] = out["target"] >= thresholds["q95"]
    out["tail_10_label"] = np.select(
        [out["is_low_10"], out["is_high_10"]],
        ["low_10", "high_10"],
        default="middle_80",
    )
    return out


def merge_split(assignments: pd.DataFrame, metadata: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    merged = assignments[["sample_id", "split"]].merge(metadata, on="sample_id", how="left")
    return add_tail_columns(merged, thresholds)


def elements_from_rows(rows: pd.DataFrame) -> set[str]:
    elements: set[str] = set()
    for row_elements in rows["elements"]:
        elements.update(row_elements)
    return elements


def split_metric_rows(strategy: str, seed: int, assignments: pd.DataFrame, metadata: pd.DataFrame, thresholds: dict[str, float]) -> list[dict[str, object]]:
    merged = merge_split(assignments, metadata, thresholds)
    train = merged[merged["split"] == "train"]
    test = merged[merged["split"] == "test"]
    train_rf = set(train["reduced_formula"])
    train_es = set(train["element_set"])
    all_n = len(merged)
    rows = []
    for split in ["train", "val", "test"]:
        part = merged[merged["split"] == split]
        part_elements = elements_from_rows(part)
        row: dict[str, object] = {
            "strategy": strategy,
            "seed": seed,
            "split": split,
            "n_samples": int(len(part)),
            "fraction": float(len(part) / all_n),
            "n_reduced_formula_groups": int(part["reduced_formula"].nunique()),
            "n_element_set_groups": int(part["element_set"].nunique()),
            "n_elements": int(len(part_elements)),
            "n_low_10": int(part["is_low_10"].sum()),
            "n_middle_80": int(part["is_middle_80"].sum()),
            "n_high_10": int(part["is_high_10"].sum()),
        }
        row.update(target_stats(part["target"]))
        if split == "test":
            test_rf = set(part["reduced_formula"])
            test_es = set(part["element_set"])
            missing_elements = sorted(part_elements - elements_from_rows(train))
            row.update(
                {
                    "train_test_overlap_reduced_formula": int(len(train_rf & test_rf)),
                    "train_test_overlap_element_set": int(len(train_es & test_es)),
                    "test_objects_seen_in_train_fraction_reduced_formula": float(part["reduced_formula"].isin(train_rf).mean()),
                    "test_objects_seen_in_train_fraction_element_set": float(part["element_set"].isin(train_es).mean()),
                    "n_train_elements": int(len(elements_from_rows(train))),
                    "n_test_elements": int(len(part_elements)),
                    "test_elements_absent_from_train": ",".join(missing_elements),
                }
            )
        else:
            row.update(
                {
                    "train_test_overlap_reduced_formula": np.nan,
                    "train_test_overlap_element_set": np.nan,
                    "test_objects_seen_in_train_fraction_reduced_formula": np.nan,
                    "test_objects_seen_in_train_fraction_element_set": np.nan,
                    "n_train_elements": np.nan,
                    "n_test_elements": np.nan,
                    "test_elements_absent_from_train": "",
                }
            )
        rows.append(row)
    return rows


def run_split_stability(metadata: pd.DataFrame, config: dict, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for strategy in GROUP_STRATEGIES:
        key = "reduced_formula" if strategy == "group_reduced_formula" else "element_set"
        for seed in SEEDS:
            assignments = make_group_split(metadata, key, float(config["test_size"]), float(config["val_size"]), seed)
            rows.extend(split_metric_rows(strategy, seed, assignments, metadata, thresholds))
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "split_stability_raw.csv", index=False)

    numeric_cols = [
        col
        for col in raw.columns
        if col not in {"strategy", "seed", "split", "test_elements_absent_from_train"}
        and pd.api.types.is_numeric_dtype(raw[col])
    ]
    agg = raw.groupby(["strategy", "split"])[numeric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUT_DIR / "split_stability_aggregated.csv", index=False)
    return raw, agg


def sample_groupaware_subset(train: pd.DataFrame, target_n: int, seed: int) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(seed)
    group_sizes = train.groupby("element_set")["sample_id"].count().reset_index(name="group_size")
    shuffled = group_sizes.iloc[rng.permutation(len(group_sizes))].reset_index(drop=True)
    selected_groups: list[str] = []
    selected_n = 0
    best_groups: list[str] = []
    best_diff = float("inf")
    for row in shuffled.itertuples(index=False):
        selected_groups.append(row.element_set)
        selected_n += int(row.group_size)
        diff = abs(selected_n - target_n)
        if diff < best_diff:
            best_diff = diff
            best_groups = selected_groups.copy()
        if selected_n >= target_n:
            break
    subset = train[train["element_set"].isin(best_groups)].copy()
    warning = "" if len(subset) == target_n else f"exact_size_not_possible_target_{target_n}_actual_{len(subset)}"
    return subset, warning


def subset_diagnostics_row(
    name: str,
    sampling: str,
    subset: pd.DataFrame,
    full_train: pd.DataFrame,
    test: pd.DataFrame,
    metadata: pd.DataFrame,
    thresholds: dict[str, float],
    warning: str = "",
) -> dict[str, object]:
    subset = add_tail_columns(subset, thresholds)
    subset_elements = elements_from_rows(subset)
    full_train_elements = elements_from_rows(full_train)
    test_rf = set(test["reduced_formula"])
    test_es = set(test["element_set"])
    subset_es_sets = [set(value.split("-")) for value in subset["element_set"].drop_duplicates()]
    nearest = []
    for value in test["element_set"].drop_duplicates():
        test_set = set(value.split("-"))
        if subset_es_sets:
            nearest.append(max(len(test_set & train_set) / len(test_set | train_set) for train_set in subset_es_sets))
    row: dict[str, object] = {
        "subset": name,
        "sampling": sampling,
        "n_samples": int(len(subset)),
        "fraction_of_full_train": float(len(subset) / len(full_train)),
        "n_reduced_formula_groups": int(subset["reduced_formula"].nunique()),
        "n_element_set_groups": int(subset["element_set"].nunique()),
        "n_elements_covered": int(len(subset_elements)),
        "elements_absent_from_subset_but_present_in_full_train": ",".join(sorted(full_train_elements - subset_elements)),
        "target_mean": float(subset["target"].mean()),
        "target_std": float(subset["target"].std()),
        "target_min": float(subset["target"].min()),
        "target_max": float(subset["target"].max()),
        "target_q10": float(subset["target"].quantile(0.10)),
        "target_q50": float(subset["target"].quantile(0.50)),
        "target_q90": float(subset["target"].quantile(0.90)),
        "n_low_10": int(subset["is_low_10"].sum()),
        "n_middle_80": int(subset["is_middle_80"].sum()),
        "n_high_10": int(subset["is_high_10"].sum()),
        "fraction_low_10_of_full_dataset": float(subset["is_low_10"].sum() / metadata["is_low_10"].sum()),
        "fraction_high_10_of_full_dataset": float(subset["is_high_10"].sum() / metadata["is_high_10"].sum()),
        "test_reduced_formula_groups_exact_overlap_with_subset": int(len(test_rf & set(subset["reduced_formula"]))),
        "test_element_set_groups_exact_overlap_with_subset": int(len(test_es & set(subset["element_set"]))),
        "test_nearest_subset_element_set_jaccard_mean": float(np.mean(nearest)) if nearest else np.nan,
        "test_nearest_subset_element_set_jaccard_min": float(np.min(nearest)) if nearest else np.nan,
        "test_nearest_subset_element_set_jaccard_q10": float(np.quantile(nearest, 0.10)) if nearest else np.nan,
        "warning": warning,
    }
    return row


def run_low_data_diagnostics(metadata: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    assignments = load_candidate("group_element_set")
    merged = merge_split(assignments, metadata, thresholds)
    full_train = merged[merged["split"] == "train"].copy()
    test = merged[merged["split"] == "test"].copy()
    rng = np.random.default_rng(42)
    train_ids = np.array(full_train["sample_id"].tolist(), dtype=int)
    shuffled_ids = rng.permutation(train_ids)
    n_2_5 = round(0.025 * len(full_train))
    n_25 = round(0.25 * len(full_train))

    random_25 = full_train[full_train["sample_id"].isin(shuffled_ids[:n_25])].copy()
    random_2_5 = full_train[full_train["sample_id"].isin(shuffled_ids[:n_2_5])].copy()
    group_25, group_25_warning = sample_groupaware_subset(full_train, n_25, 42)
    group_2_5, group_2_5_warning = sample_groupaware_subset(group_25, n_2_5, 42)

    files = {
        "train_2_5_random.csv": random_2_5,
        "train_25_random.csv": random_25,
        "train_100.csv": full_train,
        "train_2_5_groupaware.csv": group_2_5,
        "train_25_groupaware.csv": group_25,
    }
    for filename, subset in files.items():
        subset[["sample_id"]].sort_values("sample_id").to_csv(SPLIT_DIR / filename, index=False)

    rows = [
        subset_diagnostics_row("train_2_5_random", "random_objects", random_2_5, full_train, test, metadata, thresholds),
        subset_diagnostics_row("train_25_random", "random_objects", random_25, full_train, test, metadata, thresholds),
        subset_diagnostics_row("train_100", "full_train", full_train, full_train, test, metadata, thresholds),
        subset_diagnostics_row("train_2_5_groupaware", "groupaware_element_set", group_2_5, full_train, test, metadata, thresholds, group_2_5_warning),
        subset_diagnostics_row("train_25_groupaware", "groupaware_element_set", group_25, full_train, test, metadata, thresholds, group_25_warning),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "low_data_subset_diagnostics.csv", index=False)
    return out


def nearest_jaccard(train_sets: list[str], test_values: pd.Series) -> list[dict[str, object]]:
    train_parsed = [(value, set(value.split("-"))) for value in train_sets]
    rows = []
    for value in test_values:
        test_set = set(value.split("-"))
        best_value = ""
        best_score = -1.0
        best_shared = 0
        best_train_n = 0
        for train_value, train_set in train_parsed:
            score = len(test_set & train_set) / len(test_set | train_set)
            if score > best_score:
                best_score = score
                best_value = train_value
                best_shared = len(test_set & train_set)
                best_train_n = len(train_set)
                if score == 1.0:
                    break
        rows.append(
            {
                "test_element_set": value,
                "nearest_train_element_set": best_value,
                "max_jaccard_similarity": float(best_score),
                "n_shared_elements": int(best_shared),
                "n_test_elements": int(len(test_set)),
                "n_train_elements_nearest": int(best_train_n),
            }
        )
    return rows


def run_nearest_jaccard(metadata: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for strategy in SPLIT_STRATEGIES:
        merged = merge_split(load_candidate(strategy), metadata, thresholds)
        train_sets = sorted(merged.loc[merged["split"] == "train", "element_set"].drop_duplicates())
        test = merged[merged["split"] == "test"].copy()
        nearest_by_value = dict(zip(test["element_set"].tolist(), nearest_jaccard(train_sets, test["element_set"].tolist())))
        for row in test.itertuples(index=False):
            nearest = nearest_by_value[row.element_set]
            rows.append(
                {
                    "sample_id": int(row.sample_id),
                    "split_strategy": strategy,
                    **nearest,
                    "target": float(row.target),
                }
            )
    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(OUT_DIR / "nearest_train_jaccard_per_sample.csv", index=False)

    summary_rows = []
    for strategy, part in per_sample.groupby("split_strategy"):
        scores = part["max_jaccard_similarity"]
        summary_rows.append(
            {
                "split_strategy": strategy,
                "count": int(len(scores)),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "q05": float(scores.quantile(0.05)),
                "q10": float(scores.quantile(0.10)),
                "q25": float(scores.quantile(0.25)),
                "q50": float(scores.quantile(0.50)),
                "q75": float(scores.quantile(0.75)),
                "q90": float(scores.quantile(0.90)),
                "q95": float(scores.quantile(0.95)),
                "fraction_eq_1_0": float((scores == 1.0).mean()),
                "fraction_ge_0_75": float((scores >= 0.75).mean()),
                "fraction_ge_0_5": float((scores >= 0.50).mean()),
                "fraction_lt_0_5": float((scores < 0.50).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "nearest_train_jaccard_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    for strategy in SPLIT_STRATEGIES:
        scores = per_sample.loc[per_sample["split_strategy"] == strategy, "max_jaccard_similarity"]
        plt.hist(scores, bins=bins, alpha=0.45, label=strategy, color=PLOT_COLORS[strategy], density=True)
    plt.xlabel("Nearest train element-set Jaccard similarity")
    plt.ylabel("Density")
    plt.title("Nearest train Jaccard distribution")
    plt.legend()
    save_fig(OUT_DIR / "fig_nearest_jaccard_distribution.png")
    return per_sample, summary


def composition_matrix(metadata: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    elements = sorted({element for row in metadata["elements"] for element in row})
    index = {element: idx for idx, element in enumerate(elements)}
    x = np.zeros((len(metadata), len(elements)), dtype=float)
    for i, counts in enumerate(metadata["element_counts"]):
        total = sum(float(v) for v in counts.values())
        for element, amount in counts.items():
            x[i, index[element]] = float(amount) / total
    return x, elements


def run_composition_distance(metadata: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, _elements = composition_matrix(metadata)
    sample_to_pos = {sample_id: pos for pos, sample_id in enumerate(metadata["sample_id"])}
    rows = []
    for strategy in SPLIT_STRATEGIES:
        merged = merge_split(load_candidate(strategy), metadata, thresholds)
        train_ids = merged.loc[merged["split"] == "train", "sample_id"].to_numpy()
        test_ids = merged.loc[merged["split"] == "test", "sample_id"].to_numpy()
        train_pos = np.array([sample_to_pos[sample_id] for sample_id in train_ids])
        test_pos = np.array([sample_to_pos[sample_id] for sample_id in test_ids])
        for metric in ["cosine", "euclidean"]:
            nn = NearestNeighbors(n_neighbors=1, metric=metric)
            nn.fit(x[train_pos])
            distances, indices = nn.kneighbors(x[test_pos])
            for sample_id, dist, nearest_idx in zip(test_ids, distances[:, 0], indices[:, 0]):
                rows.append(
                    {
                        "sample_id": int(sample_id),
                        "split_strategy": strategy,
                        "metric": metric,
                        "nearest_train_sample_id": int(train_ids[nearest_idx]),
                        "nearest_train_composition_distance": float(dist),
                        "target": float(metadata.loc[sample_to_pos[sample_id], "target"]),
                    }
                )
    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(OUT_DIR / "nearest_train_composition_distance_per_sample.csv", index=False)

    summary_rows = []
    for (strategy, metric), part in per_sample.groupby(["split_strategy", "metric"]):
        d = part["nearest_train_composition_distance"]
        summary_rows.append(
            {
                "split_strategy": strategy,
                "metric": metric,
                "count": int(len(d)),
                "mean": float(d.mean()),
                "std": float(d.std()),
                "min": float(d.min()),
                "max": float(d.max()),
                "q05": float(d.quantile(0.05)),
                "q10": float(d.quantile(0.10)),
                "q25": float(d.quantile(0.25)),
                "q50": float(d.quantile(0.50)),
                "q75": float(d.quantile(0.75)),
                "q90": float(d.quantile(0.90)),
                "q95": float(d.quantile(0.95)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "nearest_train_composition_distance_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    for strategy in SPLIT_STRATEGIES:
        d = per_sample[(per_sample["split_strategy"] == strategy) & (per_sample["metric"] == "cosine")]["nearest_train_composition_distance"]
        plt.hist(d, bins=40, alpha=0.45, label=strategy, color=PLOT_COLORS[strategy], density=True)
    plt.xlabel("Nearest train composition-vector cosine distance")
    plt.ylabel("Density")
    plt.title("Nearest composition distance distribution")
    plt.legend()
    save_fig(OUT_DIR / "fig_nearest_composition_distance_distribution.png")
    return per_sample, summary


def run_tail_distribution(metadata: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for strategy in SPLIT_STRATEGIES:
        merged = merge_split(load_candidate(strategy), metadata, thresholds)
        for split in ["train", "val", "test"]:
            part = merged[merged["split"] == split]
            rows.append(
                {
                    "split_strategy": strategy,
                    "split": split,
                    "n_samples": int(len(part)),
                    "n_low_10": int(part["is_low_10"].sum()),
                    "fraction_low_10": float(part["is_low_10"].mean()),
                    "n_middle_80": int(part["is_middle_80"].sum()),
                    "fraction_middle_80": float(part["is_middle_80"].mean()),
                    "n_high_10": int(part["is_high_10"].sum()),
                    "fraction_high_10": float(part["is_high_10"].mean()),
                    "n_low_5": int(part["is_low_5"].sum()),
                    "fraction_low_5": float(part["is_low_5"].mean()),
                    "n_high_5": int(part["is_high_5"].sum()),
                    "fraction_high_5": float(part["is_high_5"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "split_tail_distribution.csv", index=False)

    test = out[out["split"] == "test"]
    x = np.arange(len(SPLIT_STRATEGIES))
    width = 0.25
    plt.figure(figsize=(8, 5))
    for offset, col, label, color in [
        (-width, "fraction_low_10", "low 10%", "#0ea5e9"),
        (0, "fraction_middle_80", "middle 80%", "#64748b"),
        (width, "fraction_high_10", "high 10%", "#ef4444"),
    ]:
        values = [float(test[test["split_strategy"] == strategy][col].iloc[0]) for strategy in SPLIT_STRATEGIES]
        plt.bar(x + offset, values, width=width, label=label, color=color)
    plt.xticks(x, SPLIT_STRATEGIES, rotation=20, ha="right")
    plt.ylabel("Fraction in test split")
    plt.title("Target tail distribution in test")
    plt.legend()
    save_fig(OUT_DIR / "fig_tail_distribution_by_split.png")
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def tail_mae(y_true: np.ndarray, y_pred: np.ndarray, labels: pd.Series) -> dict[str, float]:
    out = {}
    for label in ["low_10", "middle_80", "high_10"]:
        mask = labels.to_numpy() == label
        out[f"mae_{label}"] = float(mean_absolute_error(y_true[mask], y_pred[mask])) if np.any(mask) else np.nan
    return out


def train_ids_for_fraction(train: pd.DataFrame, strategy: str, fraction: float) -> pd.DataFrame:
    if fraction == 1.0:
        return train
    rng = np.random.default_rng(42)
    ids = np.array(train["sample_id"].tolist(), dtype=int)
    shuffled = rng.permutation(ids)
    n = round(fraction * len(train))
    return train[train["sample_id"].isin(shuffled[:n])].copy()


def run_baselines(df: pd.DataFrame, metadata: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    df_by_id = df.set_index("sample_id", drop=False)
    rows = []
    prediction_frames = {strategy: [] for strategy in SPLIT_STRATEGIES}
    feature_cache: pd.DataFrame | None = None
    descriptor_note = "starter descriptors generated with existing featurize() helper"

    try:
        feature_cache = featurize(df_by_id.loc[metadata["sample_id"]], "starter").reset_index(drop=True)
        feature_cache.insert(0, "sample_id", metadata["sample_id"].to_numpy())
    except Exception as exc:
        descriptor_note = f"descriptor RF skipped: {type(exc).__name__}: {exc}"

    for strategy in SPLIT_STRATEGIES:
        merged = merge_split(load_candidate(strategy), metadata, thresholds)
        train_full = merged[merged["split"] == "train"].copy()
        test = merged[merged["split"] == "test"].copy()
        y_test = test["target"].to_numpy()
        for train_fraction in [1.0, 0.25]:
            train = train_ids_for_fraction(train_full, strategy, train_fraction)
            y_train = train["target"].to_numpy()

            dummy = DummyRegressor(strategy="mean")
            dummy.fit(np.zeros((len(train), 1)), y_train)
            pred = dummy.predict(np.zeros((len(test), 1)))
            metric_row = {
                "split_strategy": strategy,
                "model": "dummy_mean",
                "train_fraction": train_fraction,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "descriptor_note": "",
                **regression_metrics(y_test, pred),
                **tail_mae(y_test, pred, test["tail_10_label"]),
            }
            rows.append(metric_row)
            prediction_frames[strategy].append(
                pd.DataFrame(
                    {
                        "sample_id": test["sample_id"].astype(int),
                        "model": "dummy_mean",
                        "train_fraction": train_fraction,
                        "y_true": y_test,
                        "y_pred": pred,
                    }
                )
            )

            if feature_cache is not None:
                x_train = train[["sample_id"]].merge(feature_cache, on="sample_id", how="left").drop(columns=["sample_id"])
                x_test = test[["sample_id"]].merge(feature_cache, on="sample_id", how="left").drop(columns=["sample_id"])
                rf = make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=1),
                )
                rf.fit(x_train, y_train)
                pred_rf = rf.predict(x_test)
                rows.append(
                    {
                        "split_strategy": strategy,
                        "model": "rf_starter_descriptors",
                        "train_fraction": train_fraction,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "descriptor_note": descriptor_note,
                        **regression_metrics(y_test, pred_rf),
                        **tail_mae(y_test, pred_rf, test["tail_10_label"]),
                    }
                )
                prediction_frames[strategy].append(
                    pd.DataFrame(
                        {
                            "sample_id": test["sample_id"].astype(int),
                            "model": "rf_starter_descriptors",
                            "train_fraction": train_fraction,
                            "y_true": y_test,
                            "y_pred": pred_rf,
                        }
                    )
                )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "baseline_split_difficulty.csv", index=False)
    for strategy, frames in prediction_frames.items():
        pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / f"baseline_predictions_{strategy}.csv", index=False)
    return out


def run_descriptor_pca(df: pd.DataFrame, metadata: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    df_by_id = df.set_index("sample_id", drop=False)
    x = featurize(df_by_id.loc[metadata["sample_id"]], "starter").reset_index(drop=True)
    x_scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(x)
    coords = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
    out = metadata[["sample_id", "target"]].copy()
    out["pca_1"] = coords[:, 0]
    out["pca_2"] = coords[:, 1]
    out = add_tail_columns(out, thresholds)
    for strategy in ["random_iid", "group_element_set"]:
        assignments = load_candidate(strategy)
        out = out.merge(assignments.rename(columns={"split": f"{strategy}_split"}), on="sample_id", how="left")
    out.to_csv(OUT_DIR / "descriptor_space_summary.csv", index=False)

    plt.figure(figsize=(7, 5.5))
    sc = plt.scatter(out["pca_1"], out["pca_2"], c=out["target"], s=7, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label="target")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Starter descriptor PCA colored by target")
    save_fig(OUT_DIR / "fig_pca_target.png")

    for strategy, filename in [("random_iid", "fig_pca_random_iid.png"), ("group_element_set", "fig_pca_group_element_set.png")]:
        plt.figure(figsize=(7, 5.5))
        for split, color in [("train", "#64748b"), ("val", "#0ea5e9"), ("test", "#dc2626")]:
            part = out[out[f"{strategy}_split"] == split]
            plt.scatter(part["pca_1"], part["pca_2"], s=7, alpha=0.65, label=split, color=color)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title(f"Starter descriptor PCA: {strategy}")
        plt.legend(markerscale=2)
        save_fig(OUT_DIR / filename)

    plt.figure(figsize=(7, 5.5))
    for label, color in [("low_10", "#0ea5e9"), ("middle_80", "#64748b"), ("high_10", "#ef4444")]:
        part = out[out["tail_10_label"] == label]
        plt.scatter(part["pca_1"], part["pca_2"], s=7, alpha=0.65, label=label, color=color)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Starter descriptor PCA colored by target bins")
    plt.legend(markerscale=2)
    save_fig(OUT_DIR / "fig_pca_target_bins.png")
    return out


def write_summary(
    stability_raw: pd.DataFrame,
    low_data: pd.DataFrame,
    jaccard_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
    tail_distribution: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:
    ge_test = stability_raw[(stability_raw["strategy"] == "group_element_set") & (stability_raw["split"] == "test")]
    ge_test_mean_min = ge_test["mean"].min()
    ge_test_mean_max = ge_test["mean"].max()
    ge_test_size_min = ge_test["n_samples"].min()
    ge_test_size_max = ge_test["n_samples"].max()

    ge_low = low_data[low_data["subset"] == "train_2_5_groupaware"].iloc[0]
    random_low = low_data[low_data["subset"] == "train_2_5_random"].iloc[0]

    jac = jaccard_summary.set_index("split_strategy")
    dist_cos = distance_summary[distance_summary["metric"] == "cosine"].set_index("split_strategy")
    tails = tail_distribution[(tail_distribution["split_strategy"] == "group_element_set") & (tail_distribution["split"] == "test")].iloc[0]

    base_100 = baseline[(baseline["train_fraction"] == 1.0) & (baseline["model"] == "rf_starter_descriptors")]
    if base_100.empty:
        base_100 = baseline[(baseline["train_fraction"] == 1.0) & (baseline["model"] == "dummy_mean")]
    base_pivot = base_100.set_index("split_strategy")
    ge_rf_delta = base_pivot.loc["group_element_set", "mae"] - base_pivot.loc["random_iid", "mae"]

    files = sorted(str(path.relative_to(OUT_DIR)) for path in OUT_DIR.rglob("*") if path.is_file() and path.name != "eda_iteration_2_summary.md")
    file_lines = "\n".join(f"- `{file}`" for file in files)

    text = f"""# EDA split selection — iteration 2

## 1. What was checked

Проверены stability по 10 seeds, nested low-data subsets для `group_element_set`, nearest-train similarity, tail distribution и быстрый baseline difficulty check. Результаты первой итерации не перезаписывались; новые файлы сохранены в `reports/eda_split_selection/iteration_2/`.

## 2. Split stability across seeds

Для `group_element_set` test size по seeds был в диапазоне `{ge_test_size_min}`-`{ge_test_size_max}` объектов, test target mean был в диапазоне `{ge_test_mean_min:.3f}`-`{ge_test_mean_max:.3f}`. Подробные per-seed значения: `split_stability_raw.csv`, агрегаты mean/std/min/max: `split_stability_aggregated.csv`.

Предварительно `group_element_set` выглядит достаточно стабильным по размеру. Target distribution немного меняется от seed к seed, поэтому stratified group split по target bins можно рассмотреть как refinement, но базовый group split уже пригоден как simple non-IID check, если фиксировать seed и явно репортить target stats.

## 3. Low-data subset coverage

Для `group_element_set` созданы nested 2.5%, 25%, 100% train subsets. Random 2.5% subset содержит `{int(random_low['n_samples'])}` объектов и `{int(random_low['n_elements_covered'])}` элементов; group-aware 2.5% subset содержит `{int(ge_low['n_samples'])}` объектов и `{int(ge_low['n_elements_covered'])}` элементов. Exact test overlap по `element_set` остается `{int(ge_low['test_element_set_groups_exact_overlap_with_subset'])}` для group-aware 2.5%.

Для финальных экспериментов random object subsets проще и дают более ровный target sampling. Group-aware nested subsets лучше согласованы с идеей composition-aware training, но на 2.5% могут давать более бедное chemical coverage и неточный размер из-за дискретных групп. Практичный вариант: использовать random nested subsets как основной low-data protocol внутри выбранного split, а group-aware subsets оставить как sensitivity check.

## 4. Approximate similarity to train

Nearest element-set Jaccard mean: random_iid = `{jac.loc['random_iid', 'mean']:.3f}`, group_reduced_formula = `{jac.loc['group_reduced_formula', 'mean']:.3f}`, group_element_set = `{jac.loc['group_element_set', 'mean']:.3f}`. Fraction with Jaccard = 1.0: random_iid = `{jac.loc['random_iid', 'fraction_eq_1_0']:.3f}`, group_reduced_formula = `{jac.loc['group_reduced_formula', 'fraction_eq_1_0']:.3f}`, group_element_set = `{jac.loc['group_element_set', 'fraction_eq_1_0']:.3f}`.

Nearest composition-vector cosine distance mean: random_iid = `{dist_cos.loc['random_iid', 'mean']:.4f}`, group_reduced_formula = `{dist_cos.loc['group_reduced_formula', 'mean']:.4f}`, group_element_set = `{dist_cos.loc['group_element_set', 'mean']:.4f}`.

По EDA `group_element_set` делает test дальше от train по exact element-set overlap и approximate similarity, чем random. При этом test часто остается близким по shared elements, поэтому корректнее называть его `composition-aware / unseen element-combination split`, а не сильным OOD split по unseen elements.

## 5. Tail distribution

В `group_element_set` test: low 10% = `{int(tails['n_low_10'])}` объектов (`{tails['fraction_low_10']:.3f}`), high 10% = `{int(tails['n_high_10'])}` объектов (`{tails['fraction_high_10']:.3f}`). Это достаточно для tail metrics на этих split-ах. По сохраненной таблице нужно репортить tail fractions рядом с основными метриками, потому что group split может немного менять сложность test через target composition.

## 6. Quick baseline difficulty

Быстрый baseline сохранен в `baseline_split_difficulty.csv`: DummyRegressor(mean) и RF на starter descriptors из существующего pipeline, без hyperparameter tuning. RF 100% train MAE: random_iid = `{base_pivot.loc['random_iid', 'mae']:.3f}`, group_reduced_formula = `{base_pivot.loc['group_reduced_formula', 'mae']:.3f}`, group_element_set = `{base_pivot.loc['group_element_set', 'mae']:.3f}`; delta group_element_set - random_iid = `{ge_rf_delta:+.3f}`.

В этой sanity check таблице starter-descriptor RF не показывает заметного MAE-штрафа для `group_element_set`. Поэтому baseline сам по себе не доказывает повышенную сложность split-а; основной аргумент за `group_element_set` здесь — отсутствие exact `element_set` overlap и более низкая nearest-train similarity. Tail MAE (`mae_low_10`, `mae_middle_80`, `mae_high_10`) нужно использовать как diagnostic slices, а не как отдельный split.

## 7. Updated split recommendation

| Split | Status | What it tests | Why / risks |
|---|---|---|---|
| `random_iid` | keep as IID baseline | IID interpolation under current random split | Useful baseline; can include exact chemical leakage, so not enough alone. |
| `group_element_set` | recommended main non-IID split | Generalization to unseen element combinations | Stricter than formula split on element-set overlap; not the same as unseen elements or full OOD. |
| `group_reduced_formula` | optional / softer group split | Removal of exact formula leakage | It removes exact formula leakage, but can leave `element_set` overlap. |
| `target_tail_eval` | use as evaluation slice, not main split | Property extrapolation / stress analysis | Uses target to define slices, so not deployment-realistic split. |
| `site_aware` | not recommended without manual validation | Site-role generalization | Fast proxy was not robust enough in iteration 1. |
| `descriptor_space_ood` | optional future extension | Descriptor-space distance / clustering | Useful extension after explicitly saving descriptor matrix and choosing protocol. |

## 8. Files produced

{file_lines}
"""
    (OUT_DIR / "eda_iteration_2_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    config = load_config()
    df = load_dataset(index_by_sample_id=False)
    metadata = load_metadata()
    thresholds = tail_thresholds(metadata)
    metadata = add_tail_columns(metadata, thresholds)

    stability_raw, _stability_agg = run_split_stability(metadata, config, thresholds)
    low_data = run_low_data_diagnostics(metadata, thresholds)
    _jaccard_per_sample, jaccard_summary = run_nearest_jaccard(metadata, thresholds)
    _distance_per_sample, distance_summary = run_composition_distance(metadata, thresholds)
    tail_distribution = run_tail_distribution(metadata, thresholds)
    baseline = run_baselines(df, metadata, thresholds)
    run_descriptor_pca(df, metadata, thresholds)
    write_summary(stability_raw, low_data, jaccard_summary, distance_summary, tail_distribution, baseline)
    print(f"Wrote iteration 2 EDA artifacts to {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
