from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_io import load_config, project_path, read_json
from src.project_data import FRACTION_TO_SPLIT_FILE, load_dataset


SEED = 42
OUT_DIR = project_path("reports", "eda_split_selection")
CANDIDATE_SPLIT_DIR = OUT_DIR / "candidate_splits"
QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def split_file_inventory() -> pd.DataFrame:
    split_dir = project_path("data", "splits")
    rows = []
    for path in sorted(split_dir.glob("*.csv")):
        df = pd.read_csv(path)
        rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "n_rows": int(len(df)),
                "columns": ",".join(df.columns),
                "n_unique_sample_id": int(df["sample_id"].nunique()) if "sample_id" in df.columns else None,
                "n_duplicate_sample_id": int(df["sample_id"].duplicated().sum()) if "sample_id" in df.columns else None,
            }
        )

    info_path = split_dir / "split_info.json"
    if info_path.exists():
        info = read_json(info_path)
        for key, value in info.items():
            rows.append(
                {
                    "file": str(info_path.relative_to(PROJECT_ROOT)),
                    "n_rows": None,
                    "columns": key,
                    "n_unique_sample_id": value,
                    "n_duplicate_sample_id": None,
                }
            )
    return pd.DataFrame(rows)


def load_current_assignments(sample_ids: Iterable[int]) -> pd.DataFrame:
    split_dir = project_path("data", "splits")
    assignments_path = split_dir / "split_assignments.csv"
    if assignments_path.exists():
        return pd.read_csv(assignments_path)

    assignments = pd.DataFrame({"sample_id": list(sample_ids), "split": "unassigned"})
    split_files = {
        "train": split_dir / "train_indices.csv",
        "val": split_dir / "val_indices.csv",
        "test": split_dir / "test_indices.csv",
    }
    for split, path in split_files.items():
        if path.exists():
            ids = set(pd.read_csv(path)["sample_id"].astype(int))
            assignments.loc[assignments["sample_id"].isin(ids), "split"] = split
    return assignments


def get_composition_metadata(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    structure_hashes = []
    for row in df.itertuples(index=False):
        structure = row.structure
        composition = structure.composition
        element_counts = {str(element): float(amount) for element, amount in composition.get_el_amt_dict().items()}
        elements = sorted(element_counts)
        rows.append(
            {
                "sample_id": int(row.sample_id),
                "formula": composition.formula,
                "reduced_formula": composition.reduced_formula,
                "anonymous_formula": composition.anonymized_formula,
                "element_set": "-".join(elements),
                "n_unique_elements": len(elements),
                "elements_json": json.dumps(elements),
                "element_counts_json": json.dumps(element_counts, sort_keys=True),
                "n_sites": int(len(structure)),
                "target": float(row.target),
            }
        )
        structure_hashes.append(json.dumps(structure.as_dict(), sort_keys=True))

    metadata = pd.DataFrame(rows)
    metadata["structure_hash"] = pd.util.hash_pandas_object(pd.Series(structure_hashes), index=False).astype(str)
    return metadata


def target_stats(series: pd.Series) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "skewness": float(series.skew()),
    }
    for q in QUANTILES:
        out[f"q{int(q * 100):02d}"] = float(series.quantile(q))
    return out


def save_target_eda(df: pd.DataFrame, current_assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = df["target"]
    summary = pd.DataFrame([target_stats(target)])
    summary.to_csv(OUT_DIR / "target_summary.csv", index=False)

    quantiles = pd.DataFrame({"quantile": QUANTILES, "target": [float(target.quantile(q)) for q in QUANTILES]})
    quantiles.to_csv(OUT_DIR / "target_quantiles.csv", index=False)

    q05, q10, q90, q95 = target.quantile([0.05, 0.10, 0.90, 0.95])
    bins = pd.DataFrame(
        [
            {"bin": "low_5", "criterion": f"target <= {q05:.12g}", "n": int((target <= q05).sum())},
            {"bin": "low_10", "criterion": f"target <= {q10:.12g}", "n": int((target <= q10).sum())},
            {"bin": "middle_80", "criterion": f"{q10:.12g} < target < {q90:.12g}", "n": int(((target > q10) & (target < q90)).sum())},
            {"bin": "high_10", "criterion": f"target >= {q90:.12g}", "n": int((target >= q90).sum())},
            {"bin": "high_5", "criterion": f"target >= {q95:.12g}", "n": int((target >= q95).sum())},
        ]
    )
    bins["fraction"] = bins["n"] / len(target)
    bins.to_csv(OUT_DIR / "target_bins_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(target, bins=60, color="#3b82f6", edgecolor="white")
    plt.xlabel("Formation energy, eV/unit cell")
    plt.ylabel("Count")
    plt.title("Formation energy histogram")
    save_fig(OUT_DIR / "fig_target_histogram.png")

    plt.figure(figsize=(8, 3.8))
    plt.boxplot(target, vert=False, showfliers=True)
    plt.xlabel("Formation energy, eV/unit cell")
    plt.yticks([1], ["all"])
    plt.title("Formation energy boxplot")
    save_fig(OUT_DIR / "fig_target_boxplot.png")

    split_target = df[["sample_id", "target"]].merge(current_assignments[["sample_id", "split"]], on="sample_id", how="left")
    order = [split for split in ["train", "val", "test"] if split in set(split_target["split"])]
    if order:
        plt.figure(figsize=(8, 4.8))
        data = [split_target.loc[split_target["split"] == split, "target"] for split in order]
        plt.boxplot(data, tick_labels=order, showfliers=True)
        plt.ylabel("Formation energy, eV/unit cell")
        plt.title("Formation energy by current split")
        save_fig(OUT_DIR / "fig_target_by_current_split.png")

    return summary, quantiles, bins


def group_stats(metadata: pd.DataFrame, key: str) -> pd.DataFrame:
    grouped = metadata.groupby(key, dropna=False)
    stats = grouped["target"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    stats = stats.rename(columns={"count": "group_size", "mean": "target_mean", "std": "target_std", "min": "target_min", "max": "target_max"})
    stats["n_unique_elements_mean"] = grouped["n_unique_elements"].mean().values
    return stats.sort_values(["group_size", key], ascending=[False, True]).reset_index(drop=True)


def save_composition_eda(metadata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metadata.drop(columns=["structure_hash"]).to_csv(OUT_DIR / "composition_metadata.csv", index=False)

    outputs = {}
    for key in ["reduced_formula", "element_set", "anonymous_formula"]:
        stats = group_stats(metadata, key)
        stats.to_csv(OUT_DIR / f"group_stats_{key}.csv", index=False)
        outputs[key] = stats

    outputs["reduced_formula"].head(20).to_csv(OUT_DIR / "top_groups_reduced_formula.csv", index=False)
    outputs["element_set"].head(20).to_csv(OUT_DIR / "top_groups_element_set.csv", index=False)

    for key, filename in [
        ("reduced_formula", "fig_group_size_distribution_reduced_formula.png"),
        ("element_set", "fig_group_size_distribution_element_set.png"),
    ]:
        sizes = outputs[key]["group_size"]
        plt.figure(figsize=(8, 5))
        plt.hist(sizes, bins=np.arange(1, int(sizes.max()) + 2), color="#14b8a6", edgecolor="white")
        plt.yscale("log")
        plt.xlabel("Group size")
        plt.ylabel("Number of groups, log scale")
        plt.title(f"Group size distribution: {key}")
        save_fig(OUT_DIR / filename)

    return outputs


def group_size_bucket_counts(stats: pd.DataFrame, key: str) -> pd.DataFrame:
    sizes = stats["group_size"]
    return pd.DataFrame(
        [
            {"group_key": key, "bucket": "1", "n_groups": int((sizes == 1).sum())},
            {"group_key": key, "bucket": "2-5", "n_groups": int(((sizes >= 2) & (sizes <= 5)).sum())},
            {"group_key": key, "bucket": "6-20", "n_groups": int(((sizes >= 6) & (sizes <= 20)).sum())},
            {"group_key": key, "bucket": "21-100", "n_groups": int(((sizes >= 21) & (sizes <= 100)).sum())},
            {"group_key": key, "bucket": ">100", "n_groups": int((sizes > 100).sum())},
        ]
    )


def overlap_summary(assignments: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    data = assignments[["sample_id", "split"]].merge(metadata, on="sample_id", how="left")
    rows = []
    for key in ["reduced_formula", "element_set", "anonymous_formula"]:
        train_values = set(data.loc[data["split"] == "train", key])
        val_values = set(data.loc[data["split"] == "val", key])
        test_values = set(data.loc[data["split"] == "test", key])
        test_rows = data[data["split"] == "test"]
        rows.append(
            {
                "group_key": key,
                "n_train_groups": len(train_values),
                "n_val_groups": len(val_values),
                "n_test_groups": len(test_values),
                "n_groups_train_test_overlap": len(train_values & test_values),
                "n_groups_train_val_overlap": len(train_values & val_values),
                "n_groups_val_test_overlap": len(val_values & test_values),
                "test_objects_seen_in_train_count": int(test_rows[key].isin(train_values).sum()),
                "test_objects_seen_in_train_fraction": float(test_rows[key].isin(train_values).mean()) if len(test_rows) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_random_split(sample_ids: list[int], test_size: float, val_size: float, seed: int) -> pd.DataFrame:
    train_val, test = train_test_split(sample_ids, test_size=test_size, random_state=seed, shuffle=True)
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_fraction, random_state=seed, shuffle=True)
    return split_frame(train, val, test)


def make_group_split(metadata: pd.DataFrame, key: str, test_size: float, val_size: float, seed: int) -> pd.DataFrame:
    ids = metadata["sample_id"].to_numpy()
    groups = metadata[key].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(ids, groups=groups))

    train_val_ids = ids[train_val_idx]
    train_val_groups = groups[train_val_idx]
    val_fraction = val_size / (1.0 - test_size)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx_rel, val_idx_rel = next(splitter_val.split(train_val_ids, groups=train_val_groups))
    return split_frame(train_val_ids[train_idx_rel], train_val_ids[val_idx_rel], ids[test_idx])


def split_frame(train: Iterable[int], val: Iterable[int], test: Iterable[int]) -> pd.DataFrame:
    rows = []
    for split, ids in [("train", train), ("val", val), ("test", test)]:
        rows.extend({"sample_id": int(sample_id), "split": split} for sample_id in ids)
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def target_tail_labels(df: pd.DataFrame) -> pd.DataFrame:
    q05, q10, q90, q95 = df["target"].quantile([0.05, 0.10, 0.90, 0.95])
    out = df[["sample_id", "target"]].copy()
    out["tail_10_label"] = np.select(
        [out["target"] <= q10, out["target"] >= q90],
        ["low_10", "high_10"],
        default="middle_80",
    )
    out["tail_5_label"] = np.select(
        [out["target"] <= q05, out["target"] >= q95],
        ["low_5", "high_5"],
        default="middle_90",
    )
    return out


def candidate_diagnostics(candidates: dict[str, pd.DataFrame], metadata: pd.DataFrame, config: dict) -> None:
    size_rows = []
    target_rows = []
    overlap_rows = []
    coverage_rows = []

    all_elements = sorted({element for elements in metadata["elements_json"].map(json.loads) for element in elements})

    for name, assignments in candidates.items():
        merged = assignments.merge(metadata, on="sample_id", how="left")
        assignments.to_csv(CANDIDATE_SPLIT_DIR / f"{name}.csv", index=False)

        desired_test_fraction = float(config.get("test_size", 0.10))
        test_fraction = float((merged["split"] == "test").mean()) if "test" in set(merged["split"]) else np.nan
        warning = ""
        if pd.notna(test_fraction) and abs(test_fraction - desired_test_fraction) > 0.03:
            warning = "test_size_differs_from_target_by_more_than_3pp"

        for split in ["train", "val", "test"]:
            part = merged[merged["split"] == split]
            if part.empty:
                continue
            size_rows.append(
                {
                    "candidate": name,
                    "split": split,
                    "n_samples": int(len(part)),
                    "fraction": float(len(part) / len(metadata)),
                    "n_reduced_formula_groups": int(part["reduced_formula"].nunique()),
                    "n_element_set_groups": int(part["element_set"].nunique()),
                    "warning": warning if split == "test" else "",
                }
            )
            row = {"candidate": name, "split": split}
            row.update(target_stats(part["target"]))
            target_rows.append(row)

        train = merged[merged["split"] == "train"]
        test = merged[merged["split"] == "test"]
        for key in ["reduced_formula", "element_set"]:
            train_groups = set(train[key])
            test_groups = set(test[key])
            overlap_rows.append(
                {
                    "candidate": name,
                    "group_key": key,
                    "n_train_groups": len(train_groups),
                    "n_test_groups": len(test_groups),
                    "n_overlap_train_test": len(train_groups & test_groups),
                    "test_objects_seen_in_train_count": int(test[key].isin(train_groups).sum()) if len(test) else 0,
                    "test_objects_seen_in_train_fraction": float(test[key].isin(train_groups).mean()) if len(test) else np.nan,
                    "largest_test_groups": "; ".join(
                        f"{idx}:{value}" for idx, value in test[key].value_counts().head(10).items()
                    ),
                }
            )

        train_elements = {element for elements in train["elements_json"].map(json.loads) for element in elements}
        test_elements = {element for elements in test["elements_json"].map(json.loads) for element in elements}
        missing = sorted(test_elements - train_elements)
        coverage_rows.append(
            {
                "candidate": name,
                "n_all_elements": len(all_elements),
                "n_train_elements": len(train_elements),
                "n_test_elements": len(test_elements),
                "n_test_elements_absent_from_train": len(missing),
                "test_elements_absent_from_train": ",".join(missing),
            }
        )

    pd.DataFrame(size_rows).to_csv(OUT_DIR / "candidate_split_diagnostics.csv", index=False)
    pd.DataFrame(target_rows).to_csv(OUT_DIR / "candidate_split_target_stats.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT_DIR / "candidate_split_group_overlap.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT_DIR / "candidate_split_element_coverage.csv", index=False)


def write_inventory(df: pd.DataFrame, metadata: pd.DataFrame, config: dict) -> pd.DataFrame:
    dataset_info_path = project_path("data", "processed", "dataset_info.json")
    dataset_info = read_json(dataset_info_path) if dataset_info_path.exists() else {}
    rows = [
        {"metric": "dataset_name", "value": config.get("dataset_name", "unknown")},
        {"metric": "n_objects", "value": len(df)},
        {"metric": "columns", "value": ",".join(df.columns)},
        {"metric": "structure_format", "value": type(df.iloc[0]["structure"]).__name__ if len(df) else "unknown"},
        {"metric": "target_column", "value": "target"},
        {"metric": "target_unit", "value": config.get("target_unit", dataset_info.get("target_unit", ""))},
        {"metric": "missing_sample_id", "value": int(df["sample_id"].isna().sum())},
        {"metric": "missing_structure", "value": int(df["structure"].isna().sum())},
        {"metric": "missing_target", "value": int(df["target"].isna().sum())},
        {"metric": "duplicated_sample_id", "value": int(df["sample_id"].duplicated().sum())},
        {"metric": "duplicated_reduced_formula_rows", "value": int(metadata["reduced_formula"].duplicated().sum())},
        {"metric": "duplicated_formula_rows", "value": int(metadata["formula"].duplicated().sum())},
        {"metric": "duplicated_structure_hash_rows", "value": int(metadata["structure_hash"].duplicated().sum())},
        {"metric": "n_unique_reduced_formula", "value": int(metadata["reduced_formula"].nunique())},
        {"metric": "n_unique_element_set", "value": int(metadata["element_set"].nunique())},
        {"metric": "n_unique_anonymous_formula", "value": int(metadata["anonymous_formula"].nunique())},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "dataset_inventory.csv", index=False)
    return out


def site_proxy(metadata: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in df.itertuples(index=False):
        sites = [(site.specie.symbol, tuple(np.round(site.frac_coords % 1, 6))) for site in row.structure.sites]
        by_coord = {coords: symbol for symbol, coords in sites}
        a_site = by_coord.get((0.5, 0.5, 0.5))
        b_site = by_coord.get((0.0, 0.0, 0.0))
        x_sites = [symbol for symbol, coords in sites if coords in {(0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5)}]
        rows.append(
            {
                "sample_id": int(row.sample_id),
                "a_site_proxy": a_site,
                "b_site_proxy": b_site,
                "x_site_proxy": "-".join(sorted(set(x_sites))) if x_sites else None,
                "n_x_proxy_sites_found": len(x_sites),
                "site_proxy_complete": bool(a_site and b_site and len(x_sites) == 3 and len(set(x_sites)) == 1),
            }
        )
    proxy = pd.DataFrame(rows)
    complete_fraction = proxy["site_proxy_complete"].mean()
    if complete_fraction < 0.98:
        proxy.to_csv(OUT_DIR / "site_proxy_metadata.csv", index=False)
        return proxy, pd.DataFrame()

    proxy.to_csv(OUT_DIR / "site_proxy_metadata.csv", index=False)
    long_rows = []
    for site_col in ["a_site_proxy", "b_site_proxy", "x_site_proxy"]:
        counts = proxy[site_col].value_counts(dropna=False)
        for element, count in counts.items():
            long_rows.append({"site_group": site_col, "group": element, "group_size": int(count)})
    for key, series in {
        "a_b_pair": proxy["a_site_proxy"].astype(str) + "-" + proxy["b_site_proxy"].astype(str),
        "b_x_pair": proxy["b_site_proxy"].astype(str) + "-" + proxy["x_site_proxy"].astype(str),
        "a_b_x_combo": proxy["a_site_proxy"].astype(str) + "-" + proxy["b_site_proxy"].astype(str) + "-" + proxy["x_site_proxy"].astype(str),
    }.items():
        for group, count in series.value_counts().items():
            long_rows.append({"site_group": key, "group": group, "group_size": int(count)})
    stats = pd.DataFrame(long_rows).sort_values(["site_group", "group_size"], ascending=[True, False])
    stats.to_csv(OUT_DIR / "site_group_stats.csv", index=False)

    top = stats[stats["site_group"].isin(["a_site_proxy", "b_site_proxy", "x_site_proxy"])].groupby("site_group").head(20)
    labels = top["site_group"] + ":" + top["group"].astype(str)
    plt.figure(figsize=(10, 5.5))
    plt.bar(labels, top["group_size"], color="#f97316")
    plt.xticks(rotation=80, ha="right")
    plt.ylabel("Count")
    plt.title("Site proxy element frequencies")
    save_fig(OUT_DIR / "fig_site_element_frequencies.png")
    return proxy, stats


def write_summary(
    inventory: pd.DataFrame,
    split_inventory: pd.DataFrame,
    target_summary: pd.DataFrame,
    target_bins: pd.DataFrame,
    composition_stats: dict[str, pd.DataFrame],
    group_buckets: pd.DataFrame,
    leakage: pd.DataFrame,
    site_proxy_df: pd.DataFrame,
    site_stats: pd.DataFrame,
) -> None:
    n = int(inventory.loc[inventory["metric"] == "n_objects", "value"].iloc[0])
    unique_rf = int(inventory.loc[inventory["metric"] == "n_unique_reduced_formula", "value"].iloc[0])
    unique_es = int(inventory.loc[inventory["metric"] == "n_unique_element_set", "value"].iloc[0])
    skew = float(target_summary["skewness"].iloc[0])
    low10 = int(target_bins.loc[target_bins["bin"] == "low_10", "n"].iloc[0])
    high10 = int(target_bins.loc[target_bins["bin"] == "high_10", "n"].iloc[0])
    rf_singletons = int(group_buckets[(group_buckets["group_key"] == "reduced_formula") & (group_buckets["bucket"] == "1")]["n_groups"].iloc[0])
    es_singletons = int(group_buckets[(group_buckets["group_key"] == "element_set") & (group_buckets["bucket"] == "1")]["n_groups"].iloc[0])
    rf_leak = leakage[leakage["group_key"] == "reduced_formula"].iloc[0]
    es_leak = leakage[leakage["group_key"] == "element_set"].iloc[0]
    site_complete_fraction = float(site_proxy_df["site_proxy_complete"].mean()) if len(site_proxy_df) else 0.0

    text = f"""# EDA for split selection

## A. Factual statistics

- Dataset: `{n}` objects from `matbench_perovskites`; structure column type is `pymatgen` `Structure`; target column is `target`.
- Existing split files and low-data subsets are summarized in `current_split_inventory.csv`.
- Missing values and duplicate checks are in `dataset_inventory.csv`.
- Target summary is in `target_summary.csv`; target skewness is `{skew:.6g}`.
- Target-tail counts: low 10% = `{low10}`, high 10% = `{high10}`. Full tail table is in `target_bins_summary.csv`.
- Composition metadata is in `composition_metadata.csv`.
- Unique reduced formulas: `{unique_rf}`; unique element sets: `{unique_es}`.
- Singleton groups: reduced_formula = `{rf_singletons}`, element_set = `{es_singletons}`.
- Current random split overlap with train/test: reduced_formula groups = `{int(rf_leak['n_groups_train_test_overlap'])}`, element_set groups = `{int(es_leak['n_groups_train_test_overlap'])}`.
- Test objects whose group already appears in train: reduced_formula = `{float(rf_leak['test_objects_seen_in_train_fraction']):.3f}`, element_set = `{float(es_leak['test_objects_seen_in_train_fraction']):.3f}`.
- Site proxy complete fraction from ideal cubic perovskite coordinates = `{site_complete_fraction:.3f}`; details are in `site_proxy_metadata.csv` and `site_group_stats.csv` if generated.
- Candidate split diagnostics are saved in `candidate_split_diagnostics.csv`, `candidate_split_target_stats.csv`, `candidate_split_group_overlap.csv`, and `candidate_split_element_coverage.csv`.

## B. Preliminary conclusion

По EDA видно, что current random split remains useful as an IID baseline, but it may overestimate generalization to chemically new compositions: часть формульных и element-set групп одновременно встречается в train и test.

`reduced_formula` выглядит слишком строгим ключом для первого composition-aware split, если доля singleton-групп высока: такой split может стать близким к разбиению почти по отдельным объектам и давать нестабильные размеры групп. `element_set` предварительно выглядит более реалистичным первым group key, потому что он напрямую проверяет перенос на новые химические комбинации элементов и обычно формирует более крупные группы. Это требует проверки по сохраненным candidate diagnostics, особенно по размеру test и target distribution.

Target-tail разметка полезна как stress test / property extrapolation test, а не как deployment-realistic split: она использует target для формирования eval-подмножеств. Предварительно ее лучше использовать как дополнительную диагностику качества на low/high formation-energy tails.

Site-aware split требует осторожности. Простой координатный proxy для идеальной кубической ABX3 ячейки сработал только для малой доли объектов, поэтому robust site assignment быстро не восстановлен. Это эвристика, и ее не стоит использовать как основной split без ручной проверки site assignment и химической интерпретации ролей A/B/X. Fallback для основной non-IID проверки: composition-aware group split по `element_set` или, вторым вариантом, по `reduced_formula`.

Descriptor-space OOD split не был выбран как обязательный candidate: в репозитории есть descriptor-model code и сохраненные aggregate diagnostics, но нет готовой per-sample descriptor table. Чтобы не усложнять EDA и не ломать pipeline, этот split лучше делать отдельным шагом после явного сохранения descriptor matrix.
"""
    (OUT_DIR / "eda_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    config = load_config()
    seed = int(config.get("random_seed", SEED))
    df = load_dataset(index_by_sample_id=False)
    metadata = get_composition_metadata(df)
    current_assignments = load_current_assignments(df["sample_id"].astype(int).tolist())

    inventory = write_inventory(df, metadata, config)
    split_inventory = split_file_inventory()
    split_inventory.to_csv(OUT_DIR / "current_split_inventory.csv", index=False)

    target_summary, _target_quantiles, target_bins = save_target_eda(df, current_assignments)
    composition_stats = save_composition_eda(metadata)
    group_buckets = pd.concat(
        [
            group_size_bucket_counts(composition_stats["reduced_formula"], "reduced_formula"),
            group_size_bucket_counts(composition_stats["element_set"], "element_set"),
            group_size_bucket_counts(composition_stats["anonymous_formula"], "anonymous_formula"),
        ],
        ignore_index=True,
    )
    group_buckets.to_csv(OUT_DIR / "group_size_bucket_counts.csv", index=False)

    leakage = overlap_summary(current_assignments, metadata)
    leakage.to_csv(OUT_DIR / "current_split_leakage_summary.csv", index=False)

    site_proxy_df, site_stats = site_proxy(metadata, df)

    sample_ids = df["sample_id"].astype(int).tolist()
    candidates = {
        "random_iid": current_assignments[["sample_id", "split"]].copy()
        if set(current_assignments["split"]) >= {"train", "val", "test"}
        else make_random_split(sample_ids, float(config.get("test_size", 0.10)), float(config.get("val_size", 0.10)), seed),
        "group_reduced_formula": make_group_split(metadata, "reduced_formula", float(config.get("test_size", 0.10)), float(config.get("val_size", 0.10)), seed),
        "group_element_set": make_group_split(metadata, "element_set", float(config.get("test_size", 0.10)), float(config.get("val_size", 0.10)), seed),
    }
    if not site_stats.empty:
        site_assignable = site_proxy_df[site_proxy_df["site_proxy_complete"]].merge(metadata[["sample_id"]], on="sample_id")
        site_groups = site_proxy_df.assign(site_group=site_proxy_df["a_site_proxy"].astype(str) + "-" + site_proxy_df["b_site_proxy"].astype(str) + "-" + site_proxy_df["x_site_proxy"].astype(str))
        candidates["site_aware_group_proxy"] = make_group_split(
            metadata.merge(site_groups[["sample_id", "site_group"]], on="sample_id"),
            "site_group",
            float(config.get("test_size", 0.10)),
            float(config.get("val_size", 0.10)),
            seed,
        )
        _ = site_assignable

    tail_labels = target_tail_labels(df)
    tail_labels.to_csv(CANDIDATE_SPLIT_DIR / "target_tail_eval.csv", index=False)

    candidate_diagnostics(candidates, metadata, config)
    write_summary(
        inventory,
        split_inventory,
        target_summary,
        target_bins,
        composition_stats,
        group_buckets,
        leakage,
        site_proxy_df,
        site_stats,
    )
    print(f"Wrote EDA artifacts to {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
