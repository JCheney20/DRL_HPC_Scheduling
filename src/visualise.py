"""
visualise.py — Pipeline result visualiser

Generates plots and tables from aggregate/stats outputs.

Table coverage (one CSV per statistical pipeline stage, paper-ready):
  1. Shapiro-Wilk      -> shapiro_summary.csv
  2. Friedman + Kendall W -> friedman_summary.csv
  3. Nemenyi post-hoc  -> pairwise_nemenyi.csv      (pass-through)
  4. Wilcoxon NP CI    -> confidence_intervals.csv  (pass-through)
  5. Confidence curves -> plotted only (continuous curve, not a summary table)
  6. Page trend        -> plotted only (rank staircase vs declared order)
  7. CD diagram inputs -> plotted only (Critical Difference diagram)
  8. Descriptive stats -> descriptive_stats.csv     (mean/std/median/min/max per treatment)
"""

import argparse
import ast
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import studentized_range

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

KEY_METRICS = ["avg_waiting", "avg_slowdown", "avg_turnaround", "cpu_utilization", "max_waiting"]
KEY_METRICS_LABELS = {
    "avg_waiting": "Avg Waiting Time (s)",
    "avg_slowdown": "Avg Slowdown",
    "avg_turnaround": "Avg Turnaround Time (s)",
    "cpu_utilization": "Avg CPU Utilization",
    "max_waiting": "Max Waiting Time (s)",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise pipeline results.")

    parser.add_argument(
        "--mode",
        choices=("results",),
        default="results",
        help="Only 'results' mode is supported.",
    )

    parser.add_argument(
        "--trace-name", required=True, help="Trace name (e.g., 'physical_job')."
    )
    parser.add_argument("--stats-dir", default="result/{trace_name}/stats")
    parser.add_argument("--aggregate-dir", default="result/{trace_name}/aggregate")
    parser.add_argument("--output-dir", default="result/{trace_name}")

    # Toggle standard outputs (default True, false if --skip flag)
    parser.add_argument(
        "--skip-tables", action="store_true", help="Skip metric table generation."
    )
    parser.add_argument("--skip-cd", action="store_true", help="Skip CD diagrams and Page trend plots.")
    parser.add_argument(
        "--skip-confidence", action="store_true", help="Skip confidence curves."
    )
    parser.add_argument(
        "--skip-bar-graphs", action="store_true", help="Skip bar graph generation."
    )
    parser.add_argument(
        "--skip-comparison-csv", action="store_true", help="Skip comparison CSV."
    )
    parser.add_argument(
        "--skip-stats-tables", action="store_true",
        help="Skip tabulating Shapiro/Friedman/Nemenyi/Wilcoxon/Page-trend stages.",
    )

    parser.add_argument("--no-show", action="store_true", default=False)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pipeline_data(stats_dir: Path, aggregate_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all CSVs needed for visualisation and tabulation."""
    return {
        "cd_input": pd.read_csv(stats_dir / "cd_diagram_input.csv"),
        "confidence_curves": pd.read_csv(stats_dir / "confidence_curves.csv"),
        "pairwise_nemenyi": pd.read_csv(stats_dir / "pairwise_nemenyi.csv"),
        "confidence_intervals": pd.read_csv(stats_dir / "confidence_intervals.csv"),
        "page_trend": pd.read_csv(stats_dir / "page_trend.csv"),
        "seed_summary": pd.read_csv(aggregate_dir / "seed_summary.csv"),
        "algorithm_summary": pd.read_csv(aggregate_dir / "algorithm_summary.csv"),
        "eval_wide": pd.read_csv(aggregate_dir / "eval_wide.csv"),
    }


def load_stats_summary(stats_dir: Path) -> dict:
    with open(stats_dir / "stats_summary.json") as f:
        return json.load(f)


def save_figure(fig: plt.Figure, plots_dir: Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = plots_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Tables — per-seed descriptive + comparison (existing)
# ---------------------------------------------------------------------------

def write_metric_tables(seed_summary: pd.DataFrame, tables_dir: Path) -> None:
    """One CSV per metric: treatment_id, mean, std (averaged across seeds)."""
    for metric in KEY_METRICS:
        mean_col, std_col = metric + "_mean", metric + "_std"
        df = (
            seed_summary.groupby("treatment_id")[[mean_col, std_col]]
            .mean()
            .reset_index()
        )
        df.columns = ["treatment_id", "mean", "std"]
        df.to_csv(tables_dir / f"{metric}.csv", sep=",", index=False)


def write_comparison_csv(alg_summary: pd.DataFrame, tables_dir: Path) -> None:
    """Write master comparison CSV with all treatments and primary/secondary metrics."""
    cols = ["treatment_id", "avg_waiting_mean_mean", "avg_slowdown_mean_mean",
            "avg_turnaround_mean_mean", "cpu_utilization_mean_mean", "max_waiting_mean_mean"]

    df = alg_summary[cols].copy()
    df.columns = ["treatment_id"] + KEY_METRICS
    df.to_csv(tables_dir / "comparison.csv", sep=",", index=False)


# ---------------------------------------------------------------------------
# Tables — full statistical pipeline coverage (new)
# ---------------------------------------------------------------------------
# Stages 3, 4, 6, 7 already exist as clean per-metric CSVs from
# statistical_test.py's write_stats_outputs(); they just need copying into
# tables_dir under a stable name so every stage lives in one place for
# the paper. Stages 1, 2, 8 are nested inside stats_summary.json and need
# flattening into tidy tables.

def write_passthrough_stats_tables(data: dict[str, pd.DataFrame], tables_dir: Path) -> None:
    """Stages 3, 4 — already tidy CSVs; copy as-is into tables_dir.

    Stages 6 (Page trend) and 7 (CD diagram inputs) are intentionally excluded
    here — they are plot-only (see draw_page_trend, draw_cd_diagrams) and
    already render meaningfully as figures rather than standalone tables.
    """
    passthroughs = {
        "pairwise_nemenyi": data["pairwise_nemenyi"],          # stage 3 (Nemenyi post-hoc)
        "confidence_intervals": data["confidence_intervals"],  # stage 4 (Wilcoxon NP CI)
    }
    for name, df in passthroughs.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False)


def write_shapiro_summary(stats_summary: dict, tables_dir: Path) -> None:
    """Stage 1 — Shapiro-Wilk, flattened across all metrics."""
    rows = []
    for metric_result in stats_summary.get("metrics", []):
        metric_name = metric_result.get("metric_name")
        shapiro = metric_result.get("shapiro", {})
        for entry in shapiro.get("per_treatment", []):
            rows.append({"metric_name": metric_name, **entry})
    if rows:
        pd.DataFrame(rows).to_csv(tables_dir / "shapiro_summary.csv", index=False)


def write_friedman_summary(stats_summary: dict, tables_dir: Path) -> None:
    """Stage 2 — Friedman omnibus test + Kendall's W effect size, one row per metric."""
    rows = []
    for metric_result in stats_summary.get("metrics", []):
        metric_name = metric_result.get("metric_name")
        friedman = metric_result.get("friedman", {})
        kendall_w = metric_result.get("kendall_w", {})
        rows.append({
            "metric_name": metric_name,
            "chi2": friedman.get("chi2"),
            "df": friedman.get("df"),
            "p_value": friedman.get("p_value"),
            "significant": friedman.get("significant"),
            "n_blocks_common": friedman.get("n_blocks_common"),
            "k_treatments": friedman.get("k_treatments"),
            "kendall_w": kendall_w.get("value"),
            "kendall_w_interpretation": kendall_w.get("interpretation"),
        })
    if rows:
        pd.DataFrame(rows).to_csv(tables_dir / "friedman_summary.csv", index=False)


def write_descriptive_stats(stats_summary: dict, tables_dir: Path) -> None:
    """Stage 8 — per-treatment descriptive stats (mean/median/std/min/max/rank), per metric."""
    rows = []
    for metric_result in stats_summary.get("metrics", []):
        metric_name = metric_result.get("metric_name")
        descriptive = metric_result.get("descriptive", {})
        for entry in descriptive.get("per_treatment", []):
            rows.append({"metric_name": metric_name, **entry})
    if rows:
        pd.DataFrame(rows).to_csv(tables_dir / "descriptive_stats.csv", index=False)


def write_all_stats_tables(stats_dir: Path, data: dict[str, pd.DataFrame], tables_dir: Path) -> None:
    """Write a CSV table for every stage of the 8-step statistical pipeline."""
    write_passthrough_stats_tables(data, tables_dir)          # stages 3, 4, 6, 7
    stats_summary = load_stats_summary(stats_dir)
    write_shapiro_summary(stats_summary, tables_dir)          # stage 1
    write_friedman_summary(stats_summary, tables_dir)         # stage 2
    write_descriptive_stats(stats_summary, tables_dir)        # stage 8


# ---------------------------------------------------------------------------
# Plots/Graphs
# ---------------------------------------------------------------------------

def draw_cd_diagrams(cd_df: pd.DataFrame, seed_summary: pd.DataFrame, plots_dir: Path, alpha: float = 0.05) -> None:
    """
    cd_df: cd_diagram_input.csv columns: metric_name, treatment_id, algorithm, use_masking, avg_rank
    seed_summary: seed_summary.csv (used to infer n = number of seeds per treatment)
    """
    n = seed_summary.groupby("treatment_id").size().iloc[0]

    for metric in cd_df["metric_name"].unique():
        metric_df = cd_df[cd_df["metric_name"] == metric].copy()
        k = len(metric_df)
        y = range(k)

        # Critical difference
        q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

        # Sort by rank for clean plot
        metric_df = metric_df.sort_values(by="avg_rank")
        fig, ax = plt.subplots(figsize=(10, max(6, k * 0.6)))

        ax.scatter(metric_df["avg_rank"], y, zorder=3, s=100, color="steelblue")
        # Draw ±CD error bars
        for i, (_, row) in enumerate(metric_df.iterrows()):
            ax.errorbar(
                row["avg_rank"], i, xerr=cd,
                fmt='none', capsize=5, color='gray', alpha=0.5
            )

        ax.set_yticks(y)
        ax.set_yticklabels(metric_df["treatment_id"])
        ax.set_xlabel("Average Rank")
        ax.set_title(f"CD Diagram — {metric} (α={alpha}, CD≈{cd:.2f})")
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()  # Best rank at top

        save_figure(fig, plots_dir, f"cd_diagram_{metric}")
        plt.close(fig)


def _parse_treatment_order(raw: str) -> list[str]:
    """
    page_trend.csv stores `treatment_order` as whatever string the source
    list serialised to when written via pandas (statistical_test.py's
    run_page_trend_test() produces a Python list, which pandas writes as
    its repr, e.g. "['a', 'b']"). Try the list-repr form first via
    ast.literal_eval, falling back to a plain comma-split for safety.
    """
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except (ValueError, SyntaxError):
        pass
    return [t.strip() for t in raw.split(",")]


def draw_page_trend(page_trend_df: pd.DataFrame, cd_df: pd.DataFrame, plots_dir: Path, alpha: float = 0.05) -> None:
    """
    Plot the Page trend test as a rank staircase: average rank per treatment,
    ordered along the x-axis by the pre-declared `treatment_order` (not by
    rank). A monotone staircase visually confirms the trend the test detects.

    page_trend_df: page_trend.csv columns: metric_name, performed, statistic,
        p_value, significant, treatment_order (see _parse_treatment_order
        for why this needs defensive parsing).
    cd_df: cd_diagram_input.csv, supplies avg_rank per (metric_name, treatment_id).
    """
    if page_trend_df.empty:
        print("No Page trend results to plot.")
        return

    for _, row in page_trend_df.iterrows():
        metric = row["metric_name"]
        order = _parse_treatment_order(row["treatment_order"])

        ranks = cd_df[cd_df["metric_name"] == metric].set_index("treatment_id")["avg_rank"]
        ordered_ranks = ranks.reindex(order)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(order))
        ax.plot(x, ordered_ranks.values, marker="o", color="steelblue", linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylabel("Average Rank")
        ax.set_xlabel("Declared Treatment Order")
        sig_label = "significant" if row["significant"] else "not significant"
        ax.set_title(
            f"Page Trend — {metric} "
            f"(statistic={row['statistic']:.2f}, p={row['p_value']:.4f}, {sig_label})"
        )
        ax.grid(alpha=0.3)

        save_figure(fig, plots_dir, f"page_trend_{metric}")
        plt.close(fig)


def draw_confidence_curves(curves_df: pd.DataFrame, plots_dir: Path, alpha: float = 0.05) -> None:
    """
    curves_df: confidence_curves.csv with columns:
        metric, treatment_a, treatment_b, delta, p_value
    """
    if curves_df.empty:
        print("No confidence curves to plot.")
        return

    grouped = curves_df.groupby(["metric", "treatment_a", "treatment_b"])

    for (metric, a, b), group in grouped:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(group["delta"], group["p_value"], color="steelblue", linewidth=2)

        ax.axhline(alpha, color="red", linestyle="--", label=f"α={alpha}")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("Effect Size Δ")
        ax.set_ylabel("p-value")
        ax.set_title(f"Confidence Curve — {metric}: {a} vs {b}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()

        save_figure(fig, plots_dir, f"confidence_curve_{metric}_{a}_vs_{b}")
        plt.close(fig)


def is_baseline(tid: str) -> bool:
    baselines = ["fcfs", "lcfs", "sjf", "wfp3", "unicep", "f_1", "f_2"]
    return any(b in tid.lower() for b in baselines)


def draw_bar_graphs(seed_summary: pd.DataFrame, plots_dir: Path) -> None:
    """Draw bar graph per metric: mean ± std per treatment, DRL vs baseline colors."""

    for metric in KEY_METRICS:
        mean_col, std_col = metric + "_mean", metric + "_std"
        df = (
            seed_summary.groupby("treatment_id")[[mean_col, std_col]]
            .mean()
            .reset_index()
        )
        df.columns = ["treatment_id", "mean", "std"]

        colors = ["coral" if is_baseline(tid) else "steelblue" for tid in df["treatment_id"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df["treatment_id"], df["mean"], yerr=df["std"],
               capsize=5, color=colors, edgecolor="black", alpha=0.8)

        ax.set_title(f"{metric}")
        ax.set_xlabel("Treatment ID")
        ax.set_ylabel(f"Mean {metric}")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis="y", alpha=0.3)

        legend_elements = [
            Patch(facecolor="steelblue", label="DRL"),
            Patch(facecolor="coral", label="Baseline"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        save_figure(fig, plots_dir, f"bar_graph_{metric}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_results_mode(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir.format(trace_name=args.trace_name))
    stats_dir = Path(args.stats_dir.format(trace_name=args.trace_name))
    aggregate_dir = Path(args.aggregate_dir.format(trace_name=args.trace_name))

    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = load_pipeline_data(stats_dir, aggregate_dir)

    if not args.skip_tables:
        write_metric_tables(data["seed_summary"], tables_dir)

    if not args.skip_cd:
        draw_cd_diagrams(data["cd_input"], data["seed_summary"], plots_dir)
        draw_page_trend(data["page_trend"], data["cd_input"], plots_dir)

    if not args.skip_confidence:
        draw_confidence_curves(data["confidence_curves"], plots_dir)

    if not args.skip_bar_graphs:
        draw_bar_graphs(data["seed_summary"], plots_dir)

    if not args.skip_comparison_csv:
        write_comparison_csv(data["algorithm_summary"], tables_dir)

    if not args.skip_stats_tables:
        write_all_stats_tables(stats_dir, data, tables_dir)


def main() -> None:
    args = parse_args()
    run_results_mode(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
