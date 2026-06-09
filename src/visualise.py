"""
visualise.py — HeraSched result visualiser

Usage
-----
# Auto-detect mode (single CSV in result/ → single-run; multiple → multi-run):
    python visualise.py

# Explicit single file:
    python visualise.py --file result/fcfs+best_fit.csv

# Custom result directory:
    python visualise.py --result-dir /path/to/result

# Multi-run comparison across a custom directory:
    python visualise.py --result-dir /path/to/result --mode multi

# Skip interactive window (headless / batch):
    python visualise.py --no-show

Modes
-----
single  — line plot (utilisation over time + rolling mean) + histograms per metric
multi   — grouped bar chart (mean utilisation per algorithm × metric) + stdout table

Scalar metrics (max/avg wait, max/avg slowdown, avg turnaround) are read from a sidecar
JSON file written by test_scheduler.py alongside each result CSV:
    result/<selector>+<allocator>_metrics.json
If the sidecar is absent the scalar section is omitted from the output.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

ROLLING_WINDOW = 50
SELECTORS = ["f_1", "f_2", "fcfs", "lcfs", "sjf", "unicep", "wfp3"]
ALLOCATORS = ["first_available", "best_fit", "topology_aware"]

LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]

METRIC_COLS = [
    "node_utilization",
    "cpu_utilization",
    "gpu_utilization",
    "mem_utilization",
]

METRIC_LABELS = {
    "node_utilization": "Node Utilisation",
    "cpu_utilization": "CPU Utilisation",
    "gpu_utilization": "GPU Utilisation",
    "mem_utilization": "Memory Utilisation",
}

METRIC_COLOURS = {
    "node_utilization": "#2196F3",
    "cpu_utilization": "#4CAF50",
    "gpu_utilization": "#FF5722",
    "mem_utilization": "#9C27B0",
}

METRIC_FMT = {
    "max_waiting": lambda v: f"{int(v)}s",
    "avg_waiting": lambda v: f"{int(v)}s",
    "max_slowdown": lambda v: f"{v:.2f}",
    "avg_slowdown": lambda v: f"{v:.2f}",
    "avg_turnaround": lambda v: f"{int(v)}s",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_name(csv_path: Path) -> tuple[str, str]:
    """Return (selector, allocator) parsed from '<selector>+<allocator>.csv'."""
    stem = csv_path.stem
    if "+" in stem:
        selector, allocator = stem.split("+", 1)
    else:
        selector, allocator = stem, "unknown"
    return selector, allocator


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in METRIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def load_metrics_json(csv_path: Path) -> dict | None:
    json_path = csv_path.with_name(csv_path.stem + "_metrics.json")
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


def active_metrics(df: pd.DataFrame) -> list[str]:
    """Return metric columns that have at least one non-zero value."""
    return [c for c in METRIC_COLS if df[c].max() > 0]


def save_figure(fig: plt.Figure, plots_dir: Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = plots_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")


def print_scalar_summary(metrics: dict | None, selector: str, allocator: str) -> None:
    print()
    print(f"  Selector: {selector}  |  Allocator: {allocator}")
    if metrics is None:
        print("  Scalar metrics: N/A — run test_scheduler.py to generate sidecar JSON")
        return
    print(
        f"  Max waiting:       {metrics['max_waiting']:>12,.0f}s  |  "
        f"Avg waiting:       {metrics['avg_waiting']:>10.2f}s"
    )
    print(
        f"  Max slowdown:      {metrics['max_slowdown']:>12.4f}   |  "
        f"Avg slowdown:      {metrics['avg_slowdown']:>10.4f}"
    )
    print(f"  Avg turnaround:    {metrics['avg_turnaround']:>12.2f}s")


# ---------------------------------------------------------------------------
# Single-run mode
# ---------------------------------------------------------------------------


def run_single(csv_path: Path, plots_dir: Path, show: bool) -> None:
    print(f"\n[single-run] {csv_path}")
    df = load_csv(csv_path)
    metrics = load_metrics_json(csv_path)
    selector, allocator = parse_name(csv_path)
    active = active_metrics(df)

    if not active:
        print("  No non-zero utilisation metrics found — nothing to plot.")
        return

    label = f"{selector}+{allocator}"

    # ------------------------------------------------------------------ #
    # 1. Time-series line plot                                             #
    # ------------------------------------------------------------------ #
    fig_ts, ax_ts = plt.subplots(figsize=(12, 5))
    for col in active:
        colour = METRIC_COLOURS[col]
        ax_ts.plot(
            df["time"],
            df[col],
            alpha=0.25,
            color=colour,
            linewidth=0.6,
            label=f"{METRIC_LABELS[col]} (raw)",
        )
        rolled = df[col].rolling(ROLLING_WINDOW, min_periods=1).mean()
        ax_ts.plot(
            df["time"],
            rolled,
            color=colour,
            linewidth=1.6,
            label=f"{METRIC_LABELS[col]} ({ROLLING_WINDOW}-event mean)",
        )

    suppressed = [c for c in METRIC_COLS if c not in active]
    note = ""
    if suppressed:
        note = "  (GPU utilisation suppressed — all-zero for this trace)"

    ax_ts.set_xlabel("Simulation Time (s)")
    ax_ts.set_ylabel("Utilisation")
    ax_ts.set_title(f"Cluster Utilisation over Time — {label}{note}")
    ax_ts.legend(loc="upper right", ncol=2, framealpha=0.8)
    ax_ts.set_ylim(bottom=0)
    ax_ts.grid(True, linewidth=0.4, alpha=0.5)
    fig_ts.tight_layout()
    save_figure(fig_ts, plots_dir, f"{label}_timeseries")
    if show:
        plt.show()
    plt.close(fig_ts)

    # ------------------------------------------------------------------ #
    # 2. Histograms                                                        #
    # ------------------------------------------------------------------ #
    n = len(active)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig_hist, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for i, col in enumerate(active):
        ax = axes_flat[i]
        values = df[col].dropna()
        ax.hist(
            values, bins=60, color=METRIC_COLOURS[col], edgecolor="none", alpha=0.85
        )
        ax.axvline(
            values.mean(),
            color="black",
            linewidth=1.2,
            linestyle="--",
            label=f"mean={values.mean():.3f}",
        )
        ax.set_xlabel("Utilisation")
        ax.set_ylabel("Frequency")
        ax.set_title(METRIC_LABELS[col])
        ax.legend(framealpha=0.8)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    # Hide unused subplots
    for j in range(len(active), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig_hist.suptitle(f"Utilisation Distributions — {label}", fontsize=12, y=1.01)
    fig_hist.tight_layout()
    save_figure(fig_hist, plots_dir, f"{label}_hist")
    if show:
        plt.show()
    plt.close(fig_hist)

    # ------------------------------------------------------------------ #
    # 3. Scalar summary                                                    #
    # ------------------------------------------------------------------ #
    print_scalar_summary(metrics, selector, allocator)


# ---------------------------------------------------------------------------
# Multi-run mode
# ---------------------------------------------------------------------------


def run_multi(csv_paths: list[Path], plots_dir: Path, show: bool) -> None:
    print(f"\n[multi-run] comparing {len(csv_paths)} result(s)")

    records = []
    for csv_path in sorted(csv_paths):
        df = load_csv(csv_path)
        metrics = load_metrics_json(csv_path)
        selector, allocator = parse_name(csv_path)
        row = {
            "label": f"{selector}+{allocator}",
            "selector": selector,
            "allocator": allocator,
        }
        for col in METRIC_COLS:
            row[f"mean_{col}"] = df[col].mean()
        if metrics:
            for key in ("avg_waiting", "avg_slowdown", "avg_turnaround"):
                row[key] = metrics.get(key)
        records.append(row)

    partition = csv_paths[0].parent.name

    summary = pd.DataFrame(records).set_index("label")

    # ------------------------------------------------------------------ #
    # 1. Grouped bar chart — mean utilisation per algorithm per metric     #
    # ------------------------------------------------------------------ #
    mean_cols = [f"mean_{c}" for c in METRIC_COLS]
    # Only include metrics that have at least one non-zero mean across all runs
    active_mean_cols = [c for c in mean_cols if summary[c].max() > 0]
    active_labels = [METRIC_LABELS[c.replace("mean_", "")] for c in active_mean_cols]

    labels = summary.index.tolist()
    n_algos = len(labels)
    n_metrics = len(active_mean_cols)
    x = np.arange(n_metrics)
    bar_width = 0.8 / n_algos

    fig_bar, ax_bar = plt.subplots(figsize=(max(8, 3 * n_metrics), 5))
    for i, lbl in enumerate(labels):
        vals = [summary.loc[lbl, c] for c in active_mean_cols]
        offset = (i - n_algos / 2 + 0.5) * bar_width
        ax_bar.bar(x + offset, vals, bar_width, label=lbl, alpha=0.85)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(active_labels)
    ax_bar.set_ylabel("Mean Utilisation")
    ax_bar.set_title("Mean Cluster Utilisation by Algorithm")
    ax_bar.legend(loc="upper right", framealpha=0.8)
    ax_bar.set_ylim(bottom=0)
    ax_bar.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig_bar.tight_layout()
    save_figure(fig_bar, plots_dir / partition / "comparison", "comparison_utilisation")
    if show:
        plt.show()
    plt.close(fig_bar)

    # ------------------------------------------------------------------ #
    # 2. Stdout summary table                                              #
    # ------------------------------------------------------------------ #
    print()
    print("  Utilisation summary (mean per algorithm):")
    col_w = 28
    header = f"  {'Algorithm':<{col_w}}" + "".join(
        f"  {lbl:>18}" for lbl in active_labels
    )
    print(header)
    print("  " + "-" * (col_w + 20 * len(active_labels)))
    for lbl in labels:
        row_str = f"  {lbl:<{col_w}}"
        for c in active_mean_cols:
            row_str += f"  {summary.loc[lbl, c]:>18.4f}"
        print(row_str)

    # Scalar metrics table (if any JSON sidecars present)
    scalar_cols = [
        c
        for c in ("avg_waiting", "avg_slowdown", "avg_turnaround")
        if c in summary.columns
    ]
    if scalar_cols and summary[scalar_cols].notna().any(axis=None):
        print()
        print("  Scalar metrics summary:")
        scalar_labels = {
            "avg_waiting": "Avg Wait (s)",
            "avg_slowdown": "Avg Slowdown",
            "avg_turnaround": "Avg Turnaround (s)",
        }
        header2 = f"  {'Algorithm':<{col_w}}" + "".join(
            f"  {scalar_labels[c]:>22}" for c in scalar_cols
        )
        print(header2)
        print("  " + "-" * (col_w + 24 * len(scalar_cols)))
        for lbl in labels:
            row_str = f"  {lbl:<{col_w}}"
            for c in scalar_cols:
                val = summary.loc[lbl, c]
                row_str += (
                    f"  {float(val):>22.2f}" if pd.notna(val) else f"  {'N/A':>22}"
                )
            print(row_str)

    # ------------------------------------------------------------------ #
    # 3. Scalar bar charts (if data available)                           #
    # ------------------------------------------------------------------ #
    scalar_plot_cols = [c for c in scalar_cols if summary[c].notna().any()]
    if scalar_plot_cols:
        n_sc = len(scalar_plot_cols)
        fig_sc, axes_sc = plt.subplots(1, n_sc, figsize=(6 * n_sc, 5), squeeze=False)
        for i, c in enumerate(scalar_plot_cols):
            ax = axes_sc[0][i]
            vals = [summary.loc[lbl, c] for lbl in labels]
            colours = [f"C{j}" for j in range(len(labels))]
            ax.bar(labels, vals, color=colours, alpha=0.85)
            ax.set_ylabel(scalar_labels[c])
            ax.set_title(scalar_labels[c])
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        fig_sc.suptitle("Scalar Performance Metrics by Algorithm", fontsize=12)
        fig_sc.tight_layout()
        save_figure(fig_sc, plots_dir / partition / "comparison", "comparison_scalar")
        if show:
            plt.show()
        plt.close(fig_sc)


# --------------------------------------------------------------------------
# Additional Visualisation Tools
# --------------------------------------------------------------------------
def multi_line_plot(csv_paths: list, plots_dir: Path, show: bool) -> None:
    partition = csv_paths[0].parent.name
    all_data = []
    for csv_path in csv_paths:
        selector, allocator = parse_name(csv_path)
        df = load_csv(csv_path)
        all_data.append((selector, allocator, df))

    active = [
        col for col in METRIC_COLS if any(df[col].max() > 0 for _, _, df in all_data)
    ]

    if not active:
        print("  No non-zero utilisation metrics found — nothing to plot.")
        return

    groups: dict[str, list] = defaultdict(list)
    for selector, allocator, df in all_data:
        groups[allocator].append((selector, allocator, df))

    selector_colour = {s: f"C{i}" for i, s in enumerate(SELECTORS)}
    selector_style = {s: LINESTYLES[i] for i, s in enumerate(SELECTORS)}

    for allocator_name, group in groups.items():
        fig, axes = plt.subplots(
            nrows=len(active),
            figsize=(16, 8 * len(active)),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]

        for j, col in enumerate(active):
            ax = axes[j]

            for selector, allocator, df in sorted(group, key=lambda x: x[0]):
                colour = selector_colour[selector]
                linestyle = selector_style[selector]

                rolled = df[col].rolling(ROLLING_WINDOW, min_periods=1).mean()
                ax.plot(
                    df["time"],
                    rolled,
                    color=colour,
                    linewidth=2.0,
                    linestyle=linestyle,
                    label=selector,
                )

            ax.set_ylabel(METRIC_LABELS[col])
            # ax.set_ylim(bottom=None)
            ax.set_yscale("log")
            ax.grid(True, linewidth=0.4, alpha=0.5)
            ax.legend(loc="upper right", ncol=2, framealpha=0.8)

        axes[-1].set_xlabel("Simulation Time (s)")
        fig.suptitle(
            f"Cluster Utilisation - {partition.capitalize()} / {allocator_name}",
            fontsize=12,
        )
        fig.tight_layout()
        save_figure(
            fig,
            plots_dir / partition / "multi",
            f"multi_line_{partition}_{allocator_name}",
        )
        if show:
            plt.show()
        plt.close(fig)


def heatmap_plot(csv_paths: list[Path], plots_dir: Path, show: bool) -> None:
    METRICS = ["avg_waiting", "avg_slowdown", "avg_turnaround"]
    HEATMAP_LABELS = {
        "avg_waiting": "Avg Waiting Time (s)",
        "avg_slowdown": "Avg Slowdown",
        "avg_turnaround": "Avg Turnaround Time (s)",
    }

    all_data = []
    for csv_path in csv_paths:
        selector, allocator = parse_name(csv_path)
        m = load_metrics_json(csv_path)
        if m is None:
            continue
        all_data.append((selector, allocator, m))

    groups: dict[str, list[dict]] = defaultdict(list)
    for selector, allocator, m in all_data:
        groups[m["partition"]].append(m)

    for partition, records in groups.items():
        fig, axes = plt.subplots(3, 1, figsize=(10, 14))
        df = pd.DataFrame(records)
        matrices = {
            metric: df.pivot(
                index="allocator", columns="selector", values=metric
            ).reindex(index=ALLOCATORS, columns=SELECTORS)
            for metric in METRICS
        }

        for i, (metric, matrix) in enumerate(matrices.items()):
            ax = axes[i]
            norm = (matrix - matrix.min().min()) / (
                matrix.max().max() - matrix.min().min()
            )
            im = ax.imshow(norm.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="equal")
            for row_idx, selector in enumerate(matrix.index):
                for col_idx, allocator in enumerate(matrix.columns):
                    val = matrix.loc[selector, allocator]
                    label = METRIC_FMT[metric](val)
                    ax.text(
                        col_idx, row_idx, label, ha="center", va="center", fontsize=8
                    )
            ax.set_xticks(range(len(SELECTORS)))
            ax.set_xticklabels(SELECTORS, rotation=30, ha="right", fontsize=9)
            ax.set_yticks(range(len(ALLOCATORS)))
            ax.set_yticklabels(matrix.index, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            ax.set_title(HEATMAP_LABELS[metric])
        fig.suptitle(f"Scheduler Performance — {partition.capitalize()}", fontsize=13)
        fig.tight_layout()
        save_figure(fig, plots_dir / partition / "heatmap", f"heatmap_{partition}")
        if show:
            plt.show()
        plt.close(fig)


def radar_plot(csv_paths: list[Path], plots_dir: Path, show: bool) -> None:
    RADAR_METRICS = [
        "max_waiting",
        "avg_waiting",
        "max_slowdown",
        "avg_slowdown",
        "avg_turnaround",
    ]
    RADAR_LABELS = {
        "max_waiting": "Max Waiting Time (s)",
        "avg_waiting": "Avg Waiting Time (s)",
        "max_slowdown": "Max Slowdown",
        "avg_slowdown": "Avg Slowdown",
        "avg_turnaround": "Avg Turnaround Time (s)",
    }

    all_data = []
    for csv_path in csv_paths:
        selector, allocator = parse_name(csv_path)
        m = load_metrics_json(csv_path)
        if m is None:
            continue
        all_data.append((selector, allocator, m))

    partition = all_data[0][2]["partition"]

    groups: dict[str, list[dict]] = defaultdict(list)
    for selector, allocator, m in all_data:
        groups[m["selector"]].append(m)

    metric_arr = np.array(
        [[m[metric] for metric in RADAR_METRICS] for _, _, m in all_data]
    )
    col_min = metric_arr.min(axis=0)
    col_max = metric_arr.max(axis=0)
    norm = 1 - (metric_arr - col_min) / (col_max - col_min)

    for i, (selector, allocator, m) in enumerate(all_data):
        m["normalised"] = norm[i]

    angles = np.linspace(0, 2 * np.pi, len(RADAR_METRICS), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    for selector_name, records in groups.items():
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

        for i, m in enumerate(records):
            values = m["normalised"]
            values = np.concatenate([values, values[:1]])
            ax.plot(angles, values, color=f"C{i}", linewidth=1.5, label=m["allocator"])
            ax.fill(angles, values, color=f"C{i}", alpha=0.2)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels=list(RADAR_LABELS.values()))
        ax.set_ylim(0, 1)
        ax.set_rlabel_position(30)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.set_title(f"{selector_name} - {partition.capitalize()}", pad=15)

        fig.tight_layout()
        save_figure(
            fig, plots_dir / partition / "radar", f"radar_{partition}_{selector_name}"
        )
        if show:
            plt.show()
        plt.close(fig)


def run_visualise(csv_paths: list[Path], plots_dir: Path, show: bool) -> None:
    physical_csvs = [p for p in csv_paths if p.parent.name == "physical"]
    deeplearn_csvs = [p for p in csv_paths if p.parent.name == "deeplearn"]
    if physical_csvs:
        multi_line_plot(physical_csvs, plots_dir, show)
        radar_plot(physical_csvs, plots_dir, show)
    if deeplearn_csvs:
        multi_line_plot(deeplearn_csvs, plots_dir, show)
        radar_plot(deeplearn_csvs, plots_dir, show)
    heatmap_plot(csv_paths, plots_dir, show)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise HeraSched result CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--file",
        default=None,
        metavar="CSV",
        help="Path to a specific result CSV (forces single-run mode).",
    )
    parser.add_argument(
        "--result-dir",
        default="result",
        metavar="DIR",
        help="Directory to scan for result CSVs. Default: result/",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "single", "multi", "visualise"),
        default="auto",
        help="Operating mode. Default: auto (inferred from number of CSVs found).",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        metavar="DIR",
        help="Directory to write output figures. Default: plots/",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Skip plt.show() — useful for headless/batch runs.",
    )
    return parser.parse_args()


def resolve_csv_paths(args: argparse.Namespace) -> tuple[list[Path], str]:
    """Return (list_of_csv_paths, resolved_mode)."""
    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"ERROR: --file '{args.file}' does not exist.", file=sys.stderr)
            sys.exit(1)
        return [p], "single"

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(
            f"ERROR: result directory '{result_dir}' does not exist.", file=sys.stderr
        )
        sys.exit(1)

    if args.mode == "auto":
        csvs = sorted(result_dir.glob("*.csv"))
        if not csvs:
            print(f"ERROR: no CSV files found in '{result_dir}'.", file=sys.stderr)
            sys.exit(1)
        mode = "single" if len(csvs) == 1 else "multi"
    elif args.mode == "single":
        csvs = sorted(result_dir.glob("*.csv"))
        if len(csvs) > 1:
            print(
                f"  Warning: {len(csvs)} CSVs found but --mode=single; using first: {csvs[0]}",
                file=sys.stderr,
            )
        csvs = [csvs[0]]
        mode = "single"
    elif args.mode == "multi":
        csvs = sorted(result_dir.glob("*.csv"))
        if not csvs:
            print(f"ERROR: no CSV files found in '{result_dir}'.", file=sys.stderr)
            sys.exit(1)
        mode = "multi"
    elif args.mode == "visualise":
        csvs = sorted(result_dir.glob("**/*.csv"))
        if not csvs:
            print(f"ERROR: no CSV files found in '{result_dir}'.", file=sys.stderr)
            sys.exit(1)
        mode = "visualise"

    return csvs, mode


def main() -> None:
    args = parse_args()
    csv_paths, mode = resolve_csv_paths(args)
    plots_dir = Path(args.plots_dir)
    show = not args.no_show

    if mode == "single":
        run_single(csv_paths[0], plots_dir, show)
    elif mode == "visualise":
        run_visualise(csv_paths, plots_dir, show)
    else:
        run_multi(csv_paths, plots_dir, show)

    print("\nDone.")


if __name__ == "__main__":
    main()
