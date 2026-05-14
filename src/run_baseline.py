import argparse, json, os, subprocess, sys, time
import multiprocessing
from pathlib import Path
from itertools import product

SELECTORS  = ["fcfs", "lcfs", "sjf", "wfp3", "unicep", "f_1", "f_2"]
ALLOCATORS = ["best_fit", "first_available", "topology_aware"]
TRAD_ALGORITHMS = ["fcfs", "lcfs", "sjf"]

PARTITION_CONFIGS = {
    "physical": {
        "trace":    "data/physical_job.csv",
        "topology": "data/topology/physical_topology.txt",
        "nodes":    "data/topology/nodes.csv",
    },
    "deeplearn": {
        "trace":    "data/deeplearn_job.csv",
        "topology": "data/topology/deeplearn_topology.txt",
        "nodes":    "data/topology/nodes.csv",
    },
}

def run_one(cfg: dict) -> dict:
    """
    Top-level worker (must be module-level for pickle).
    Imports HPCsim locally to avoid parent-process side-effects.
    cfg keys: selector, allocator, partition, trace, topology, nodes, result_dir
    Returns: metrics dict with identifiers.
    """
    # Import inside worker
    from HPCsim.HPCsim import HPCsim
    selector   = cfg["selector"]
    allocator  = cfg["allocator"]
    partition  = cfg["partition"]
    result_dir = Path(cfg["result_dir"]) / partition
    result_dir.mkdir(parents=True, exist_ok=True)
    out_csv     = result_dir / f"{selector}+{allocator}.csv"
    out_metrics = result_dir / f"{selector}+{allocator}_metrics.json"
    # TODO: Ensure baseline outputs match eval_wide schema (run_id, algorithm, use_masking, seed, split_id, metrics)
    # TODO: so baselines can be included in stats by default.
    # TODO: Map selector+allocator to algorithm name for treatment_id ordering in stats.
    # Ref: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
    # Skip if already done (unless --force)
    if out_metrics.exists() and not cfg.get("force", False):
        print(f"[{partition}] {selector}+{allocator} — SKIP (already exists)")
        with open(out_metrics) as f:
            return json.load(f)
    t0 = time.time()
    env = HPCsim(
        scheduler=selector,
        allocator=allocator,
        backfill_enable=True,
        topology_file=cfg["topology"],
        node_file=cfg["nodes"],
        trace_file=cfg["trace"],
        random_job=False,
    )
    env.run()  # writes result/<selector>+<allocator>.csv
    # Move the CSV to the partition subdir (atomic rename)
    os.replace(f"result/{selector}+{allocator}.csv", out_csv)
    max_w, avg_w = env.evaluator.waiting_time()
    max_s, avg_s = env.evaluator.bounded_slowdown()
    avg_t        = env.evaluator.average_turnaround()
    elapsed      = time.time() - t0
    metrics = {
        "partition":     partition,
        "selector":      selector,
        "allocator":     allocator,
        "max_waiting":   float(max_w),
        "avg_waiting":   float(avg_w),
        "max_slowdown":  float(max_s),
        "avg_slowdown":  float(avg_s),
        "avg_turnaround": float(avg_t),
        "wall_time_s":   round(elapsed, 2),
    }
    # TODO: Add split_id and use_masking=false to align with stats treatment_id (algorithm__mask_false).
    # TODO: Add run_id and algorithm fields so aggregation can merge baselines with DRL outputs.
    # Ref: https://docs.python.org/3/library/json.html
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"[{partition}] {selector}+{allocator} — done "
        f"(avg_wait={avg_w:.1f}s  avg_slowdown={avg_s:.4f}  "
        f"wall={elapsed:.0f}s)"
    )
    return metrics

def build_configs(partitions: list[str], result_dir: str, force: bool) -> list[dict]:
    configs = []
    for partition in partitions:
        pc = PARTITION_CONFIGS[partition]
        for selector, allocator in product(SELECTORS, ALLOCATORS):
            configs.append({
                "selector":  selector,
                "allocator": allocator,
                "partition": partition,
                "trace":     pc["trace"],
                "topology":  pc["topology"],
                "nodes":     pc["nodes"],
                "result_dir": result_dir,
                "force":     force,
            })
    # TODO: Consider filtering to TRAD_ALGORITHMS for baseline comparison to reduce runtime.
    # TODO: Expose selector list via CLI to control baseline scope.
    # Ref: https://docs.python.org/3/library/argparse.html
    return configs

def print_summary_table(results: list[dict]) -> None:
    """Print a formatted summary table grouped by partition."""
    col_w = 12
    scalar_cols = ["avg_waiting", "avg_slowdown", "avg_turnaround", "wall_time_s"]
    scalar_labels = {
        "avg_waiting":    "Avg Wait (s)",
        "avg_slowdown":   "Avg Slowdown",
        "avg_turnaround": "Avg Turnaround (s)",
        "wall_time_s":    "Wall Time (s)",
    }

    # Group by partition
    partitions_seen = []
    by_partition: dict[str, list[dict]] = {}
    for r in results:
        p = r.get("partition", "unknown")
        if p not in by_partition:
            by_partition[p] = []
            partitions_seen.append(p)
        by_partition[p].append(r)

    for partition in partitions_seen:
        rows = by_partition[partition]
        print(f"\n{'='*72}")
        print(f"  Partition: {partition}  ({len(rows)} runs)")
        print(f"{'='*72}")
        header = (
            f"  {'Selector':<{col_w}}  {'Allocator':<18}"
            + "".join(f"  {scalar_labels[c]:>20}" for c in scalar_cols)
        )
        print(header)
        print("  " + "-" * (col_w + 20 + 22 * len(scalar_cols)))
        for r in sorted(rows, key=lambda x: (x["selector"], x["allocator"])):
            row_str = f"  {r['selector']:<{col_w}}  {r['allocator']:<18}"
            for c in scalar_cols:
                val = r.get(c)
                if val is None:
                    row_str += f"  {'N/A':>20}"
                elif c == "avg_slowdown":
                    row_str += f"  {float(val):>20.4f}"
                else:
                    row_str += f"  {float(val):>20.2f}"
            print(row_str)
    print()


def run_visualise(result_dir: str, plots_dir: str, partitions: list[str], show: bool) -> None:
    for partition in partitions:
        subdir      = str(Path(result_dir) / partition)
        plots_subdir = str(Path(plots_dir) / partition)
        cmd = [
            sys.executable, "baseline_visualise.py",
            "--result-dir", subdir,
            "--plots-dir",  plots_subdir,
            "--mode", "multi",
        ]
        if not show:
            cmd.append("--no-show")
        print(f"\n[visualise] {partition} → {plots_subdir}")
        subprocess.run(cmd, check=True)
    # TODO: Provide headless-friendly outputs for Snakemake (plots as files only).
    # Ref: https://matplotlib.org/stable/users/explain/figure/backends.html

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all heuristic baseline combinations across partitions."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel worker processes. Default: min(cpu_count, 4).",
    )
    parser.add_argument(
        "--partition",
        choices=("physical", "deeplearn", "both"),
        default="both",
        help="Which partition(s) to run. Default: both.",
    )
    parser.add_argument(
        "--result-dir",
        default="result",
        metavar="DIR",
        help="Root directory for result CSVs and metrics JSON. Default: result/",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        metavar="DIR",
        help="Root directory for output plots. Default: plots/",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Skip plt.show() in baseline_visualise.py (headless/batch runs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run combinations even if output already exists.",
    )
    # TODO: Add --selectors and --allocators args so Snakemake can control baseline scope via config.yaml.
    # TODO: Add --output-format to emit eval_wide-compatible CSV for aggregation.
    # Ref: https://docs.python.org/3/library/argparse.html
    return parser.parse_args()


def main():
    args    = parse_args()
    partitions = (
        ["physical", "deeplearn"] if args.partition == "both"
        else [args.partition]
    )
    configs = build_configs(partitions, args.result_dir, args.force)
    workers = args.workers or min(os.cpu_count() or 4, 4)
    print(f"Running {len(configs)} combinations across {workers} workers...")
    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(run_one, configs)
    print_summary_table(results)
    run_visualise(args.result_dir, args.plots_dir, partitions, show=not args.no_show)

if __name__ == "__main__":
    multiprocessing.freeze_support()   # needed on Windows; harmless on Linux
    main()
