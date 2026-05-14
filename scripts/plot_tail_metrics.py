from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BUDGET_ORDER = ["B500", "B2000", "B8000", "Bfull"]
SPLIT_TITLES = {
    "random_iid": "Random split",
    "element_set": "Chemical split",
}
MODEL_LABELS = {
    "cgcnn_optimized": "CGCNN",
    "matgl_megnet_frozen": "MatGL / MEGNet",
    "descriptor_xgb_expanded": "XGBoost descriptors",
}
MODEL_ORDER = ["cgcnn_optimized", "matgl_megnet_frozen", "descriptor_xgb_expanded"]
COLORS = {
    "cgcnn_optimized": "#1f77b4",
    "matgl_megnet_frozen": "#ff7f0e",
    "descriptor_xgb_expanded": "#2ca02c",
}
TARGET_BIN_LABELS = {
    "overall": "Overall",
    "low_10": "Low 10%",
    "middle_80": "Middle 80%",
    "high_10": "High 10%",
    "low_5": "Low 5%",
    "middle_90": "Middle 90%",
    "high_5": "High 5%",
}
METRIC_LABELS = {
    "mae": "MAE, eV / unit cell",
    "mape": "MAPE, %",
    "r2": "R2",
}


def style_axes(ax: plt.Axes, metric: str) -> None:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Training budget")
    ax.set_ylabel(METRIC_LABELS[metric])


def aggregate_metric(df: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    grouped = (
        df.groupby(
            ["model_name", "split_strategy", "budget_name", "target_bin_scheme", "target_bin"],
            observed=True,
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["budget_name"] = pd.Categorical(grouped["budget_name"], categories=BUDGET_ORDER, ordered=True)
    return grouped.sort_values(["split_strategy", "model_name", "budget_name"])


def plot_bin_metric(
    agg: pd.DataFrame,
    target_bin_scheme: str,
    target_bin: str,
    output_path: Path,
    *,
    metric: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.36, 5.4), sharey=True)
    for ax, split_strategy in zip(axes, ["random_iid", "element_set"], strict=True):
        split_df = agg[
            (agg["target_bin_scheme"] == target_bin_scheme)
            & (agg["target_bin"] == target_bin)
            & (agg["split_strategy"] == split_strategy)
        ]
        for model_name in MODEL_ORDER:
            model_df = split_df[split_df["model_name"] == model_name].sort_values("budget_name")
            if model_df.empty:
                continue
            x = list(range(len(model_df)))
            ax.errorbar(
                x,
                model_df["mean"],
                yerr=model_df["std"],
                marker="o",
                linewidth=2,
                capsize=3,
                color=COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )
        ax.set_title(SPLIT_TITLES[split_strategy])
        ax.set_xticks(range(len(BUDGET_ORDER)), BUDGET_ORDER)
        style_axes(ax, metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_tail_penalty(agg: pd.DataFrame, output_path: Path, *, metric: str) -> None:
    subset = agg[
        (agg["target_bin_scheme"] == "target_bin_10")
        & (agg["target_bin"].isin(["middle_80", "high_10"]))
    ]
    pivot = subset.pivot_table(
        index=["model_name", "split_strategy", "budget_name"],
        columns="target_bin",
        values="mean",
        observed=True,
    ).reset_index()
    pivot["tail_penalty"] = pivot["high_10"] - pivot["middle_80"]
    pivot["budget_name"] = pd.Categorical(pivot["budget_name"], categories=BUDGET_ORDER, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.36, 5.4), sharey=True)
    for ax, split_strategy in zip(axes, ["random_iid", "element_set"], strict=True):
        split_df = pivot[pivot["split_strategy"] == split_strategy]
        for model_name in MODEL_ORDER:
            model_df = split_df[split_df["model_name"] == model_name].sort_values("budget_name")
            if model_df.empty:
                continue
            ax.plot(
                range(len(model_df)),
                model_df["tail_penalty"],
                marker="o",
                linewidth=2,
                color=COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )
        ax.set_title(SPLIT_TITLES[split_strategy])
        ax.set_xticks(range(len(BUDGET_ORDER)), BUDGET_ORDER)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Training budget")
        ax.set_ylabel(f"High 10% {metric.upper()} - Middle 80% {metric.upper()}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_side_by_side_bins(
    agg: pd.DataFrame,
    budget_name: str,
    output_path: Path,
    *,
    metric: str,
) -> None:
    target_bins = ["low_10", "middle_80", "high_10"]
    subset = agg[
        (agg["target_bin_scheme"] == "target_bin_10")
        & (agg["target_bin"].isin(target_bins))
        & (agg["budget_name"] == budget_name)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.36, 5.4), sharey=True)
    width = 0.22
    offsets = {
        "cgcnn_optimized": -width,
        "matgl_megnet_frozen": 0.0,
        "descriptor_xgb_expanded": width,
    }
    x_base = list(range(len(target_bins)))

    for ax, split_strategy in zip(axes, ["random_iid", "element_set"], strict=True):
        split_df = subset[subset["split_strategy"] == split_strategy]
        for model_name in MODEL_ORDER:
            model_df = (
                split_df[split_df["model_name"] == model_name]
                .set_index("target_bin")
                .reindex(target_bins)
            )
            x = [value + offsets[model_name] for value in x_base]
            ax.bar(
                x,
                model_df["mean"],
                yerr=model_df["std"],
                width=width,
                capsize=3,
                color=COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )
        ax.set_title(SPLIT_TITLES[split_strategy])
        ax.set_xticks(x_base, [TARGET_BIN_LABELS[name] for name in target_bins])
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlabel("Target bin")
        ax.set_ylabel(METRIC_LABELS[metric])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot target-tail evaluation metrics.")
    parser.add_argument("--tail-metrics", default="outputs/summary/tail_metrics.csv")
    parser.add_argument("--out-dir", default="outputs/figures/target_tail")
    args = parser.parse_args()

    df = pd.read_csv(args.tail_metrics)
    out_dir = Path(args.out_dir)
    for metric in ["mape", "r2"]:
        agg = aggregate_metric(df, metric=metric)
        for target_bin in ["overall", "low_10", "middle_80", "high_10"]:
            plot_bin_metric(
                agg,
                "target_bin_10",
                target_bin,
                out_dir / f"tail_{metric}_{target_bin}.png",
                metric=metric,
            )
        for target_bin in ["low_5", "middle_90", "high_5"]:
            plot_bin_metric(
                agg,
                "target_bin_5",
                target_bin,
                out_dir / f"tail_{metric}_{target_bin}.png",
                metric=metric,
            )
        plot_side_by_side_bins(
            agg,
            "Bfull",
            out_dir / f"tail_bins_side_by_side_Bfull_{metric}.png",
            metric=metric,
        )
        if metric == "mape":
            plot_side_by_side_bins(
                agg,
                "Bfull",
                out_dir / "tail_bins_side_by_side_Bfull.png",
                metric=metric,
            )

    mae_agg = aggregate_metric(df, metric="mae")
    plot_tail_penalty(mae_agg, out_dir / "tail_penalty_high10_vs_middle80.png", metric="mae")
    print(f"Wrote target-tail plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
