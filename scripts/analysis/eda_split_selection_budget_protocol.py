from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.models.train_descriptor_baseline import featurize
from src.data_io import project_path
from src.project_data import load_dataset


FIRST_DIR = project_path("reports", "eda_split_selection")
OUT_DIR = FIRST_DIR / "iteration_3_budget_protocol"
CANDIDATE_DIR = OUT_DIR / "candidate_splits"
PRED_DIR = OUT_DIR / "baseline_predictions"
BUDGETS: list[int | str] = [500, 1000, 2000, 4000, 8000, "full"]
FIXED_BUDGETS = [500, 1000, 2000, 4000, 8000]
SPLIT_STRATEGIES = ["random_iid", "group_element_set", "group_reduced_formula"]
MAIN_STABILITY_STRATEGIES = ["random_iid", "group_element_set"]
SEEDS = list(range(10))
PLOT_COLORS = {
    "random_iid": "#2563eb",
    "group_element_set": "#dc2626",
    "group_reduced_formula": "#16a34a",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def load_metadata() -> pd.DataFrame:
    path = FIRST_DIR / "composition_metadata.csv"
    metadata = pd.read_csv(path)
    metadata["elements"] = metadata["elements_json"].map(json.loads)
    metadata["element_counts"] = metadata["element_counts_json"].map(json.loads)
    return metadata


def load_candidate(strategy: str) -> pd.DataFrame:
    return pd.read_csv(FIRST_DIR / "candidate_splits" / f"{strategy}.csv")


def tail_thresholds(metadata: pd.DataFrame) -> dict[str, float]:
    target = metadata["target"]
    return {
        "q05": float(target.quantile(0.05)),
        "q10": float(target.quantile(0.10)),
        "q90": float(target.quantile(0.90)),
        "q95": float(target.quantile(0.95)),
    }


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
    return add_tail_columns(assignments[["sample_id", "split"]].merge(metadata, on="sample_id", how="left"), thresholds)


def elements_from_rows(rows: pd.DataFrame) -> set[str]:
    elements: set[str] = set()
    for row_elements in rows["elements"]:
        elements.update(row_elements)
    return elements


def nested_budget_subsets(train: pd.DataFrame, budgets: list[int], seed: int) -> dict[int | str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.array(train["sample_id"], dtype=int))
    out: dict[int | str, pd.DataFrame] = {}
    for budget in budgets:
        if budget <= len(train):
            ids = set(shuffled[:budget])
            out[budget] = train[train["sample_id"].isin(ids)].copy()
    out["full"] = train.copy()
    return out


def budget_label(budget: int | str) -> str:
    return "full" if budget == "full" else str(budget)


def strategy_split_parts(strategy: str, metadata: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = merge_split(load_candidate(strategy), metadata, thresholds)
    return (
        merged[merged["split"] == "train"].copy(),
        merged[merged["split"] == "val"].copy(),
        merged[merged["split"] == "test"].copy(),
    )


def write_budget_definitions(metadata: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for strategy in SPLIT_STRATEGIES:
        train, _val, _test = strategy_split_parts(strategy, metadata, thresholds)
        full_n = len(train)
        for budget in BUDGETS:
            requested = full_n if budget == "full" else int(budget)
            warning = "" if requested <= full_n else "requested_budget_exceeds_full_train_size"
            rows.append(
                {
                    "split_strategy": strategy,
                    "budget_label": budget_label(budget),
                    "requested_budget": requested,
                    "actual_n_train": full_n if budget == "full" else (requested if requested <= full_n else np.nan),
                    "full_train_size": full_n,
                    "is_full_train": budget == "full",
                    "warning": warning,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "budget_definitions.csv", index=False)
    return out


def create_candidate_files(metadata: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for strategy in SPLIT_STRATEGIES:
        train, _val, _test = strategy_split_parts(strategy, metadata, thresholds)
        strategy_dir = CANDIDATE_DIR / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        subsets = nested_budget_subsets(train, FIXED_BUDGETS, seed=42)
        for budget, subset in subsets.items():
            filename = f"train_budget_{budget}.csv" if budget != "full" else "train_full.csv"
            path = strategy_dir / filename
            subset[["sample_id"]].sort_values("sample_id").to_csv(path, index=False)
            rows.append(
                {
                    "split_strategy": strategy,
                    "budget_label": budget_label(budget),
                    "file": str(path.relative_to(OUT_DIR)),
                    "n_rows": int(len(subset)),
                    "n_unique_sample_id": int(subset["sample_id"].nunique()),
                    "warning": "",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "budget_subset_files_inventory.csv", index=False)
    return out


def encode_element_sets(metadata: pd.DataFrame) -> dict[str, set[str]]:
    return {value: set(value.split("-")) for value in metadata["element_set"].drop_duplicates()}


def nearest_jaccard_stats(train_element_sets: pd.Series, test_element_sets: pd.Series, parsed: dict[str, set[str]]) -> dict[str, float]:
    train_values = sorted(set(train_element_sets))
    test_values = list(test_element_sets)
    scores = []
    for value in test_values:
        test_set = parsed[value]
        best = 0.0
        for train_value in train_values:
            train_set = parsed[train_value]
            score = len(test_set & train_set) / len(test_set | train_set)
            if score > best:
                best = score
                if best == 1.0:
                    break
        scores.append(best)
    arr = np.array(scores, dtype=float)
    return {
        "nearest_jaccard_mean": float(arr.mean()),
        "nearest_jaccard_q10": float(np.quantile(arr, 0.10)),
        "nearest_jaccard_q50": float(np.quantile(arr, 0.50)),
        "nearest_jaccard_q90": float(np.quantile(arr, 0.90)),
        "nearest_jaccard_fraction_eq_1_0": float((arr == 1.0).mean()),
        "nearest_jaccard_fraction_ge_0_75": float((arr >= 0.75).mean()),
        "nearest_jaccard_fraction_lt_0_5": float((arr < 0.5).mean()),
    }


def subset_diagnostics_row(
    strategy: str,
    budget: int | str,
    subset: pd.DataFrame,
    full_train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    parsed_sets: dict[str, set[str]],
) -> dict[str, object]:
    subset_elements = elements_from_rows(subset)
    full_elements = elements_from_rows(full_train)
    train_rf = set(subset["reduced_formula"])
    train_es = set(subset["element_set"])
    test_rf = set(test["reduced_formula"])
    test_es = set(test["element_set"])
    jaccard = nearest_jaccard_stats(subset["element_set"], test["element_set"], parsed_sets)
    row: dict[str, object] = {
        "split_strategy": strategy,
        "budget_label": budget_label(budget),
        "n_train": int(len(subset)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "is_full_train": budget == "full",
        "n_reduced_formula_groups": int(subset["reduced_formula"].nunique()),
        "n_element_set_groups": int(subset["element_set"].nunique()),
        "n_elements_covered": int(len(subset_elements)),
        "elements_absent_from_budget_but_present_in_full_train": ",".join(sorted(full_elements - subset_elements)),
        "fraction_of_full_train_reduced_formula_groups_covered": float(subset["reduced_formula"].nunique() / full_train["reduced_formula"].nunique()),
        "fraction_of_full_train_element_set_groups_covered": float(subset["element_set"].nunique() / full_train["element_set"].nunique()),
        "target_mean": float(subset["target"].mean()),
        "target_std": float(subset["target"].std()),
        "target_min": float(subset["target"].min()),
        "target_max": float(subset["target"].max()),
        "target_q05": float(subset["target"].quantile(0.05)),
        "target_q10": float(subset["target"].quantile(0.10)),
        "target_q25": float(subset["target"].quantile(0.25)),
        "target_q50": float(subset["target"].quantile(0.50)),
        "target_q75": float(subset["target"].quantile(0.75)),
        "target_q90": float(subset["target"].quantile(0.90)),
        "target_q95": float(subset["target"].quantile(0.95)),
        "n_low_10": int(subset["is_low_10"].sum()),
        "fraction_low_10": float(subset["is_low_10"].mean()),
        "n_middle_80": int(subset["is_middle_80"].sum()),
        "fraction_middle_80": float(subset["is_middle_80"].mean()),
        "n_high_10": int(subset["is_high_10"].sum()),
        "fraction_high_10": float(subset["is_high_10"].mean()),
        "n_low_5": int(subset["is_low_5"].sum()),
        "n_high_5": int(subset["is_high_5"].sum()),
        "train_test_overlap_reduced_formula": int(len(train_rf & test_rf)),
        "train_test_overlap_element_set": int(len(train_es & test_es)),
        "test_objects_seen_in_train_fraction_reduced_formula": float(test["reduced_formula"].isin(train_rf).mean()),
        "test_objects_seen_in_train_fraction_element_set": float(test["element_set"].isin(train_es).mean()),
    }
    row.update(jaccard)
    return row


def run_budget_diagnostics(metadata: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    parsed_sets = encode_element_sets(metadata)
    rows = []
    for strategy in SPLIT_STRATEGIES:
        train, val, test = strategy_split_parts(strategy, metadata, thresholds)
        subsets = nested_budget_subsets(train, FIXED_BUDGETS, seed=42)
        for budget in BUDGETS:
            if budget not in subsets:
                continue
            rows.append(subset_diagnostics_row(strategy, budget, subsets[budget], train, val, test, parsed_sets))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "budget_subset_diagnostics.csv", index=False)
    return out


def run_budget_stability(metadata: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parsed_sets = encode_element_sets(metadata)
    rows = []
    for strategy in MAIN_STABILITY_STRATEGIES:
        train, _val, test = strategy_split_parts(strategy, metadata, thresholds)
        for seed in SEEDS:
            subsets = nested_budget_subsets(train, FIXED_BUDGETS, seed=seed)
            for budget in FIXED_BUDGETS:
                subset = subsets[budget]
                jaccard = nearest_jaccard_stats(subset["element_set"], test["element_set"], parsed_sets)
                rows.append(
                    {
                        "split_strategy": strategy,
                        "seed": seed,
                        "budget_label": str(budget),
                        "n_train": int(len(subset)),
                        "target_mean": float(subset["target"].mean()),
                        "target_std": float(subset["target"].std()),
                        "target_q10": float(subset["target"].quantile(0.10)),
                        "target_q50": float(subset["target"].quantile(0.50)),
                        "target_q90": float(subset["target"].quantile(0.90)),
                        "n_elements_covered": int(len(elements_from_rows(subset))),
                        "n_reduced_formula_groups": int(subset["reduced_formula"].nunique()),
                        "n_element_set_groups": int(subset["element_set"].nunique()),
                        "n_low_10": int(subset["is_low_10"].sum()),
                        "n_high_10": int(subset["is_high_10"].sum()),
                        "nearest_jaccard_mean": jaccard["nearest_jaccard_mean"],
                        "nearest_jaccard_q10": jaccard["nearest_jaccard_q10"],
                        "nearest_jaccard_q50": jaccard["nearest_jaccard_q50"],
                        "nearest_jaccard_q90": jaccard["nearest_jaccard_q90"],
                    }
                )
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "budget_stability_raw.csv", index=False)
    numeric_cols = [
        col
        for col in raw.columns
        if col not in {"split_strategy", "seed", "budget_label"}
        and pd.api.types.is_numeric_dtype(raw[col])
    ]
    agg = raw.groupby(["split_strategy", "budget_label"])[numeric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUT_DIR / "budget_stability_aggregated.csv", index=False)
    return raw, agg


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


def run_budget_baseline(df: pd.DataFrame, metadata: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    df_by_id = df.set_index("sample_id", drop=False)
    feature_cache = featurize(df_by_id.loc[metadata["sample_id"]], "starter").reset_index(drop=True)
    feature_cache.insert(0, "sample_id", metadata["sample_id"].to_numpy())
    rows = []
    for strategy in SPLIT_STRATEGIES:
        train, _val, test = strategy_split_parts(strategy, metadata, thresholds)
        subsets = nested_budget_subsets(train, FIXED_BUDGETS, seed=42)
        y_test = test["target"].to_numpy()
        x_test = test[["sample_id"]].merge(feature_cache, on="sample_id", how="left").drop(columns=["sample_id"])
        for budget in BUDGETS:
            if budget not in subsets:
                continue
            subset = subsets[budget]
            y_train = subset["target"].to_numpy()
            for model_name in ["dummy_mean", "rf_starter_descriptors"]:
                if model_name == "dummy_mean":
                    model = DummyRegressor(strategy="mean")
                    x_train = np.zeros((len(subset), 1))
                    x_eval = np.zeros((len(test), 1))
                else:
                    model = make_pipeline(
                        SimpleImputer(strategy="median"),
                        RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, min_samples_leaf=1),
                    )
                    x_train = subset[["sample_id"]].merge(feature_cache, on="sample_id", how="left").drop(columns=["sample_id"])
                    x_eval = x_test
                model.fit(x_train, y_train)
                pred = model.predict(x_eval)
                rows.append(
                    {
                        "split_strategy": strategy,
                        "budget_label": budget_label(budget),
                        "model": model_name,
                        "n_train": int(len(subset)),
                        "n_test": int(len(test)),
                        **regression_metrics(y_test, pred),
                        **tail_mae(y_test, pred, test["tail_10_label"]),
                    }
                )
                pred_path = PRED_DIR / f"{strategy}_{model_name}_budget_{budget_label(budget)}.csv"
                pd.DataFrame(
                    {
                        "sample_id": test["sample_id"].astype(int),
                        "split_strategy": strategy,
                        "budget_label": budget_label(budget),
                        "model": model_name,
                        "y_true": y_test,
                        "y_pred": pred,
                    }
                ).to_csv(pred_path, index=False)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "budget_baseline_difficulty.csv", index=False)
    return out


def write_learning_curve_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in diagnostics.itertuples(index=False):
        for family in ["descriptor", "cgcnn", "matgl"]:
            rows.append(
                {
                    "split_strategy": row.split_strategy,
                    "budget_label": row.budget_label,
                    "n_train": int(row.n_train),
                    "n_val": int(row.n_val),
                    "n_test": int(row.n_test),
                    "budget_type": "full_train" if row.is_full_train else "fixed_budget",
                    "model_family_placeholder": family,
                    "metric_placeholder": "",
                    "notes": "nested fixed annotation budget; target not used for sampling",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "learning_curve_protocol_table.csv", index=False)
    return out


def budget_sort_key(label: str) -> int:
    return 10**9 if label == "full" else int(label)


def make_plots(diagnostics: pd.DataFrame, stability_agg: pd.DataFrame, baseline: pd.DataFrame) -> None:
    ge = diagnostics[diagnostics["split_strategy"] == "group_element_set"].copy()
    ge["budget_order"] = ge["budget_label"].map(budget_sort_key)
    ge = ge.sort_values("budget_order")

    plt.figure(figsize=(8, 5))
    labels = ge["budget_label"].tolist()
    values = [pd.read_csv(CANDIDATE_DIR / "group_element_set" / (f"train_budget_{label}.csv" if label != "full" else "train_full.csv")) for label in labels]
    target_by_id = diagnostics
    _ = target_by_id
    metadata = load_metadata()
    data = []
    for label, ids_df in zip(labels, values):
        vals = ids_df.merge(metadata[["sample_id", "target"]], on="sample_id", how="left")["target"]
        data.append(vals)
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xlabel("Budget")
    plt.ylabel("Formation energy, eV/unit cell")
    plt.title("Target distribution by budget: group_element_set")
    save_fig(OUT_DIR / "fig_budget_target_distribution.png")

    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.plot(x, ge["n_element_set_groups"], marker="o", label="element_set groups", color="#dc2626")
    plt.plot(x, ge["n_reduced_formula_groups"], marker="o", label="reduced_formula groups", color="#16a34a")
    plt.plot(x, ge["n_elements_covered"], marker="o", label="elements covered", color="#2563eb")
    plt.xticks(x, labels)
    plt.xlabel("Budget")
    plt.ylabel("Count")
    plt.title("Budget coverage: group_element_set")
    plt.legend()
    save_fig(OUT_DIR / "fig_budget_coverage.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, ge["nearest_jaccard_mean"], marker="o", label="mean", color="#dc2626")
    plt.plot(x, ge["nearest_jaccard_q10"], marker="o", label="q10", color="#2563eb")
    plt.plot(x, ge["nearest_jaccard_q90"], marker="o", label="q90", color="#16a34a")
    plt.xticks(x, labels)
    plt.xlabel("Budget")
    plt.ylabel("Nearest test-to-train Jaccard")
    plt.title("Nearest Jaccard by budget: group_element_set")
    plt.legend()
    save_fig(OUT_DIR / "fig_budget_nearest_jaccard.png")

    rf = baseline[baseline["model"] == "rf_starter_descriptors"].copy()
    rf["budget_order"] = rf["budget_label"].map(budget_sort_key)
    plt.figure(figsize=(8, 5))
    for strategy in ["random_iid", "group_element_set"]:
        part = rf[rf["split_strategy"] == strategy].sort_values("budget_order")
        plt.plot(part["budget_label"], part["mae"], marker="o", label=strategy, color=PLOT_COLORS[strategy])
    plt.xlabel("Budget")
    plt.ylabel("Test MAE")
    plt.title("Starter RF MAE by fixed budget")
    plt.legend()
    save_fig(OUT_DIR / "fig_budget_baseline_mae.png")


def write_summary(
    definitions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    stability_agg: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:
    ge_diag = diagnostics[diagnostics["split_strategy"] == "group_element_set"].set_index("budget_label")
    ri_diag = diagnostics[diagnostics["split_strategy"] == "random_iid"].set_index("budget_label")
    ge_stab_500 = stability_agg[
        (stability_agg["split_strategy"] == "group_element_set") & (stability_agg["budget_label"] == "500")
    ].iloc[0]
    ri_stab_500 = stability_agg[
        (stability_agg["split_strategy"] == "random_iid") & (stability_agg["budget_label"] == "500")
    ].iloc[0]
    rf = baseline[(baseline["model"] == "rf_starter_descriptors") & (baseline["budget_label"].isin(["500", "full"]))].copy()
    rf_pivot = rf.pivot_table(index=["split_strategy"], columns="budget_label", values="mae", aggfunc="first")
    files = sorted(str(path.relative_to(OUT_DIR)) for path in OUT_DIR.rglob("*") if path.is_file() and path.name != "budget_protocol_summary.md")
    file_lines = "\n".join(f"- `{file}`" for file in files)

    text = f"""# Budget-based learning curve protocol

## 1. Motivation

Мы используем absolute annotation budgets, потому что practical materials screening обычно ограничен числом структур, для которых можно получить DFT labels, а не долей от benchmark dataset. Бюджеты 500 -> 1000 -> 2000 -> 4000 -> 8000 -> full train задают логарифмическую learning-curve сетку.

## 2. What changed compared with previous EDA

В iteration 2 low-data subsets были выражены как 2.5% / 25% от train split. Теперь основной experimental protocol заменен на fixed budgets. Sampling остается nested random-object внутри train части соответствующего split-а, seed = 42, target не используется для sampling.

## 3. Budget subset diagnostics

Все fixed budgets помещаются в full train для выбранных split strategies; детали в `budget_definitions.csv`. Для `group_element_set` full train size = `{int(ge_diag.loc['full', 'n_train'])}`, budget 500 покрывает `{int(ge_diag.loc['500', 'n_elements_covered'])}` элементов, `{int(ge_diag.loc['500', 'n_element_set_groups'])}` element-set groups и `{int(ge_diag.loc['500', 'n_reduced_formula_groups'])}` reduced-formula groups. Budget 8000 покрывает `{int(ge_diag.loc['8000', 'n_element_set_groups'])}` element-set groups.

Для `group_element_set` exact train/test overlap по `element_set` равен `{int(ge_diag['train_test_overlap_element_set'].max())}` для всех budgets, как и ожидается. В budget 500 low/high tail counts для `group_element_set`: low 10% = `{int(ge_diag.loc['500', 'n_low_10'])}`, high 10% = `{int(ge_diag.loc['500', 'n_high_10'])}`; в full train: low 10% = `{int(ge_diag.loc['full', 'n_low_10'])}`, high 10% = `{int(ge_diag.loc['full', 'n_high_10'])}`.

## 4. Stability across seeds

Для budget 500 разброс target mean по seeds: random_iid std = `{ri_stab_500['target_mean_std']:.4f}`, group_element_set std = `{ge_stab_500['target_mean_std']:.4f}`. Разброс nearest Jaccard mean по seeds: random_iid std = `{ri_stab_500['nearest_jaccard_mean_std']:.4f}`, group_element_set std = `{ge_stab_500['nearest_jaccard_mean_std']:.4f}`.

Budget 500 сильнее зависит от случайного выбора subset-а, чем крупные budgets; для финальных моделей желательно запускать budget subsets по нескольким seeds, особенно для 500 и 1000 labels. Если compute budget ограничен, seed 42 можно оставить как основной candidate protocol, но uncertainty по subset seed нужно явно отметить.

## 5. Baseline difficulty under fixed budgets

Baseline sanity check сохранен в `budget_baseline_difficulty.csv`: DummyRegressor(mean) и RF на starter descriptors, без tuning. RF MAE для budget 500: random_iid = `{rf_pivot.loc['random_iid', '500']:.3f}`, group_element_set = `{rf_pivot.loc['group_element_set', '500']:.3f}`. RF MAE для full train: random_iid = `{rf_pivot.loc['random_iid', 'full']:.3f}`, group_element_set = `{rf_pivot.loc['group_element_set', 'full']:.3f}`.

Это sanity check, не финальный benchmark. Он нужен, чтобы увидеть, что fixed budgets дают ожидаемую learning-curve форму и что split protocol технически готов к моделям.

## 6. Final recommendation

- Использовать absolute train budgets: 500, 1000, 2000, 4000, 8000, full train.
- Для каждого budget использовать nested subsets, чтобы learning curve была монотонной по данным.
- Основной split protocol: `random_iid` как IID baseline и `group_element_set` как main composition-aware non-IID split.
- `group_reduced_formula` оставить optional: он убирает exact formula leakage, но допускает overlap по `element_set`.
- Tail metrics считать как evaluation slices.
- Не называть `group_element_set` unseen-elements split: это unseen element-combinations split.

## 7. Files produced

{file_lines}
"""
    (OUT_DIR / "budget_protocol_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_dataset(index_by_sample_id=False)
    metadata = load_metadata()
    thresholds = tail_thresholds(metadata)
    metadata = add_tail_columns(metadata, thresholds)
    definitions = write_budget_definitions(metadata, thresholds)
    create_candidate_files(metadata, thresholds)
    diagnostics = run_budget_diagnostics(metadata, thresholds)
    _stability_raw, stability_agg = run_budget_stability(metadata, thresholds)
    baseline = run_budget_baseline(df, metadata, thresholds)
    write_learning_curve_table(diagnostics)
    make_plots(diagnostics, stability_agg, baseline)
    write_summary(definitions, diagnostics, stability_agg, baseline)
    print(f"Wrote budget protocol artifacts to {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
