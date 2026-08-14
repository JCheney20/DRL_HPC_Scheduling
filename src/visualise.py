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
import scikit_posthocs as sp
from src.utils import RANDOM_ALGORITHM, TRAD_ALGORITHMS

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

# Metrics whose bar graphs use a log y-axis. These span several orders of
# magnitude across treatments (a2c's avg_waiting is ~10^5 s while the maskable
# variants sit at ~10^2), so on a linear axis every well-performing treatment
# collapses onto the zero line. Utilisation is a 0-1 ratio with a narrow spread
# and stays linear.
LOG_SCALE_METRICS = {"avg_waiting", "avg_slowdown", "avg_turnaround", "max_waiting"}
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

    parser.add_argument(
        "--baseline-summary",
        default="result/{trace_name}/baseline/baseline_summary.csv",
        help="baseline_summary.csv to merge into the bar graphs (reviewer item "
             "N3 — without it the bars show DRL treatments only). Missing file "
             "warns and degrades to DRL-only rather than failing.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_baseline_summary(path: Path) -> pd.DataFrame | None:
    """baseline_summary.csv for the bar graphs, or None with a loud warning.

    The Snakefile always supplies this, so in the pipeline it is effectively
    required. Degrading instead of failing keeps ad-hoc DRL-only invocations
    usable — but the warning is deliberately noisy, because the bug this fixes
    (N3) was invisible precisely because nothing said the baselines were absent.
    """
    if not path.exists():
        print(
            f"[WARN] baseline_summary.csv not found at {path} — bar graphs will "
            f"show DRL treatments only, and the baseline/control legend entries "
            f"will be omitted. Pass --baseline-summary to include them.",
            flush=True,
        )
        return None
    return pd.read_csv(path)


def load_pipeline_data(stats_dir: Path, aggregate_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all CSVs needed for visualisation and tabulation."""
    
    def safe_read(directory: Path, filename: str, expected_cols: list[str]) -> pd.DataFrame:
            path = directory / filename
            try:
                return pd.read_csv(path)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                return pd.DataFrame(columns=expected_cols)

    return {
        "cd_input": safe_read(stats_dir, "cd_diagram_input.csv", ["metric_name", "treatment_id", "avg_rank"]),
        "confidence_curves": safe_read(stats_dir, "confidence_curves.csv", ["metric", "treatment_a", "treatment_b", "delta", "p_value"]),
        "pairwise_nemenyi": safe_read(stats_dir, "pairwise_nemenyi.csv", ["metric_name", "treatment_a", "treatment_b", "p_value"]),
        "confidence_intervals": safe_read(stats_dir, "confidence_intervals.csv", ["metric_name", "treatment_a", "treatment_b", "ci_low", "ci_high"]),
        "page_trend": safe_read(stats_dir, "page_trend.csv", ["metric_name", "treatment_order", "trend_direction", "statistic", "p_value", "significant"]),
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

def per_treatment_mean_std(seed_summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and across-seed spread of one metric, one row per treatment.

    The spread is recomputed here rather than read out of seed_summary's
    <metric>_std column, because that column is always NaN:
    aggregate_seed_summary groups on GROUP_KEYS (utils.py:130), which includes
    seed, while eval_wide holds exactly one row per run (uniqueness enforced at
    aggregate_results.py:229) -- so every group is a single element and its std
    is undefined. What the tables and error bars want is the spread of the
    per-seed means across seeds, which only exists one level up.
    """
    df = (
        seed_summary.groupby("treatment_id")[metric + "_mean"]
        .agg(["mean", "std"])
        .reset_index()
    )
    df.columns = ["treatment_id", "mean", "std"]
    return df


def write_metric_tables(seed_summary: pd.DataFrame, tables_dir: Path) -> None:
    """One CSV per metric: treatment_id, mean, std (across seeds)."""
    for metric in KEY_METRICS:
        df = per_treatment_mean_std(seed_summary, metric)
        df.to_csv(tables_dir / f"{metric}.csv", sep=",", index=False)


def write_comparison_csv(alg_summary: pd.DataFrame, tables_dir: Path) -> None:
    """Write master comparison CSV with all treatments and primary/secondary metrics."""
    cols = ["treatment_id", "avg_waiting_mean_mean", "avg_slowdown_mean_mean",
            "avg_turnaround_mean_mean", "cpu_utilization_mean_mean", "max_waiting_mean_mean"]

    df = alg_summary[cols].copy()
    df.columns = ["treatment_id"] + KEY_METRICS
    df.to_csv(tables_dir / "comparison.csv", sep=",", index=False)


# ---------------------------------------------------------------------------
# Tables — full statistical pipeline coverage
# ---------------------------------------------------------------------------

def write_passthrough_stats_tables(data: dict[str, pd.DataFrame], tables_dir: Path) -> None:
    """Stages 3, 4 — already tidy CSVs; copy as-is into tables_dir."""
    passthroughs = {
        "pairwise_nemenyi": data["pairwise_nemenyi"],          
        "confidence_intervals": data["confidence_intervals"], 
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
    """Write a CSV table for every TABLE-producing stage of the 8-step pipeline (stages 1, 2, 3, 4, 8)."""
    write_passthrough_stats_tables(data, tables_dir)          # stages 3, 4
    stats_summary = load_stats_summary(stats_dir)
    write_shapiro_summary(stats_summary, tables_dir)          # stage 1
    write_friedman_summary(stats_summary, tables_dir)         # stage 2
    write_descriptive_stats(stats_summary, tables_dir)        # stage 8


# ---------------------------------------------------------------------------
# Plots/Graphs
# ---------------------------------------------------------------------------

def _build_sig_matrix(nemenyi_metric_df: pd.DataFrame, treatments: list[str]) -> pd.DataFrame:
    treatment_set = set(treatments)
    sig_matrix = pd.DataFrame(
        np.ones((len(treatments), len(treatments))), index=treatments, columns=treatments
    )
    for _, row in nemenyi_metric_df.iterrows():
        a, b = row["treatment_a"], row["treatment_b"]
        if a not in treatment_set or b not in treatment_set:
            continue
        sig_matrix.loc[a, b] = row["p_value"]
        sig_matrix.loc[b, a] = row["p_value"]
    return sig_matrix


def draw_cd_diagrams(cd_df: pd.DataFrame, nemenyi_df: pd.DataFrame, plots_dir: Path, alpha: float = 0.05) -> None:
    for metric in cd_df["metric_name"].unique():
        metric_cd = cd_df[cd_df["metric_name"] == metric]
        metric_nemenyi = nemenyi_df[nemenyi_df["metric_name"] == metric] if not nemenyi_df.empty else pd.DataFrame()

        if metric_nemenyi.empty:
            print(f"  No Nemenyi results for {metric} (Friedman not significant) -- skipping CD diagram.")
            continue

        ranks = metric_cd.set_index("treatment_id")["avg_rank"]
        sig_matrix = _build_sig_matrix(metric_nemenyi, ranks.index.tolist())

        fig, ax = plt.subplots(figsize=(10, max(3, len(ranks) * 0.5)))
        # ponytail: installed scikit-posthocs predates the `alpha` kwarg; it thresholds
        # the p-value sig_matrix at a hardcoded 0.05 internally (matches alpha default).
        sp.critical_difference_diagram(ranks, sig_matrix, ax=ax)
        ax.set_title(f"CD Diagram — {metric} (α={alpha})")

        save_figure(fig, plots_dir, f"cd_diagram_{metric}")
        plt.close(fig)


def _parse_treatment_order(raw: str) -> list[str]:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except (ValueError, SyntaxError):
        pass
    return [t.strip() for t in raw.split(",")]


def draw_page_trend(page_trend_df: pd.DataFrame, cd_df: pd.DataFrame, plots_dir: Path, alpha: float = 0.05) -> None:
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
        ax.set_xlabel(f"Declared Treatment Order (predicted: {row['trend_direction']})")
        sig_label = "significant" if row["significant"] else "not significant"
        ax.set_title(
            f"Page Trend — {metric} "
            f"(statistic={row['statistic']:.2f}, p={row['p_value']:.4f}, {sig_label})"
        )
        ax.grid(alpha=alpha)

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


def baseline_bar_rows(baseline_summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Baseline rows in per_treatment_mean_std's (treatment_id, mean, std) shape.

    baseline_summary.csv uses algorithm_summary's "{metric}_mean_mean" /
    "{metric}_mean_std" naming, so it needs translating before it can sit
    alongside the DRL frame. std is NaN for the deterministic heuristics --
    correct, and matplotlib simply draws no whisker -- and real for the N27
    random control, which is the one stochastic baseline.
    """
    mean_col, std_col = f"{metric}_mean_mean", f"{metric}_mean_std"
    if mean_col not in baseline_summary.columns:
        return pd.DataFrame(columns=["treatment_id", "mean", "std"])
    out = pd.DataFrame({
        "treatment_id": baseline_summary["treatment_id"],
        "mean": baseline_summary[mean_col],
        # Pre-N27 baseline_summary.csv files have no std column at all.
        "std": (baseline_summary[std_col] if std_col in baseline_summary.columns
                else np.nan),
    })
    return out.reset_index(drop=True)


def classify_treatment(treatment_id: str) -> str:
    """Bucket a treatment_id for colouring: 'random', 'heuristic' or 'drl'.

    Matches on the algorithm segment before "__" rather than doing a substring
    test over the whole id. The old `any(b in tid.lower() for b in
    TRAD_ALGORITHMS)` test was a latent bug: TRAD_ALGORITHMS contains "f_1" and
    "f_2", which are substrings of plenty of unrelated ids.
    """
    algo = treatment_id.split("__")[0].lower()
    if algo == RANDOM_ALGORITHM:
        return "random"
    if algo in TRAD_ALGORITHMS:
        return "heuristic"
    return "drl"


# Bar colour and legend label per category. The N27 random control gets its own
# colour deliberately: it is a chance floor rather than a scheduling heuristic,
# and it is the only non-DRL row carrying a real error bar, so colouring it as a
# heuristic would misrepresent what it is and what its whisker means.
BAR_STYLE = {
    "drl":       ("steelblue", "DRL"),
    "heuristic": ("coral",     "Heuristic"),
    "random":    ("#8c8c8c",   "Random (chance floor)"),
}


def draw_bar_graphs(
    seed_summary: pd.DataFrame,
    plots_dir: Path,
    baseline_summary: pd.DataFrame | None = None,
) -> None:
    """Bar graph per metric: mean ± std per treatment, DRL vs baselines.

    Reviewer item N3: this used to receive only seed_summary (DRL rows), while
    the baselines live in baseline_summary.csv and were never merged in -- so
    every bar rendered steelblue beneath a hardcoded "Baseline" legend entry
    with nothing behind it. The baselines are now concatenated in, and the
    legend is built from the categories actually plotted rather than asserted.
    """
    for metric in KEY_METRICS:
        df = per_treatment_mean_std(seed_summary, metric)

        if baseline_summary is not None:
            extra = baseline_bar_rows(baseline_summary, metric)
            if not extra.empty:
                df = pd.concat([df, extra], ignore_index=True)

        categories = [classify_treatment(tid) for tid in df["treatment_id"]]
        colors = [BAR_STYLE[c][0] for c in categories]

        # Deterministic heuristics have no across-seed spread, so their std is
        # NaN. A NaN in yerr propagates through the clip/vstack below and
        # matplotlib then drops the whole error bar collection, taking the DRL
        # whiskers with it. Zero is the honest value for a single deterministic
        # run: no whisker is drawn, and the bars that do have spread keep theirs.
        spread = df["std"].fillna(0.0)

        # A log axis has no zero to grow bars from, so anchor them half a decade
        # below the smallest mean and draw each bar as the span floor -> mean.
        # Only usable when every mean is strictly positive.
        use_log = metric in LOG_SCALE_METRICS and bool((df["mean"] > 0).all())
        if use_log:
            floor = 10.0 ** np.floor(np.log10(df["mean"].min()) - 0.5)
            heights = df["mean"] - floor
            # The across-seed std often exceeds the mean (a2c/dqn are that
            # unstable), which would put the lower whisker at or below zero.
            # Clip it to the floor so the bar keeps its upper whisker instead of
            # being dropped by the log transform.
            lower = np.clip(df["mean"] - spread, floor, None)
            yerr = np.vstack([df["mean"] - lower, spread])
        else:
            floor, heights, yerr = 0.0, df["mean"], spread

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df["treatment_id"], heights, bottom=floor, yerr=yerr,
               capsize=5, color=colors, edgecolor="black", alpha=0.8)

        if use_log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=floor)

        ax.set_title(f"{metric}")
        ax.set_xlabel("Treatment ID")
        ax.set_ylabel(f"Mean {metric}" + (" (log scale)" if use_log else ""))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis="y", alpha=0.3, which="both" if use_log else "major")

        # Built from the categories actually plotted. The previous hardcoded
        # DRL/Baseline pair is what let the missing-baselines bug (N3) hide: the
        # legend asserted a coral series that was never drawn.
        legend_elements = [
            Patch(facecolor=BAR_STYLE[c][0], label=BAR_STYLE[c][1])
            for c in ("drl", "heuristic", "random") if c in set(categories)
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        save_figure(fig, plots_dir, f"bar_graph_{metric}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
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
        draw_cd_diagrams(data["cd_input"], data["pairwise_nemenyi"], plots_dir)
        draw_page_trend(data["page_trend"], data["cd_input"], plots_dir)

    if not args.skip_confidence:
        draw_confidence_curves(data["confidence_curves"], plots_dir)

    if not args.skip_bar_graphs:
        draw_bar_graphs(
            data["seed_summary"],
            plots_dir,
            baseline_summary=load_baseline_summary(
                Path(args.baseline_summary.format(trace_name=args.trace_name))
            ),
        )

    if not args.skip_comparison_csv:
        write_comparison_csv(data["algorithm_summary"], tables_dir)

    if not args.skip_stats_tables:
        write_all_stats_tables(stats_dir, data, tables_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
