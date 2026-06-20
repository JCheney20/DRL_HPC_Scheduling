import argparse 
import time
import os
import pandas as pd
from pathlib import Path
from HPCsim.HPCsim import HPCsim
from utils import PARTITION_CONFIGS, write_csv, write_json, write_manifest_entry

def run_one(row: dict, run_id: str, partition: str, result_dir: str) -> None:
    algorithm = str(row["algorithm"])
    trace_file = str(row["trace_file"])
    allocator  = "best_fit"
    split_id = row["split_id"] 
    use_masking = "false"
    treatment_id = f"{algorithm}__mask_{use_masking}" 
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    out_csv     = Path(result_dir) / f"{run_id}_metrics.csv"
    out_metrics     = Path(result_dir) / f"{run_id}_metrics.json"

    t0 = time.time()
    env = HPCsim(
        scheduler=algorithm,
        allocator=allocator,
        backfill_enable=True,
        topology_file=str(row["topology_file"]),
        node_file=str(row["node_file"]),
        trace_file=trace_file,
        partition=partition,
        random_job=False,
    )

    env.run()  
    os.replace(f"result/{algorithm}+{allocator}.csv", str(out_csv))

    

    max_w, avg_w = env.evaluator.waiting_time()
    max_s, avg_s = env.evaluator.bounded_slowdown()
    avg_t        = env.evaluator.average_turnaround()
    cpu_utilization, gpu_utilization = env.utilization()
    elapsed      = time.time() - t0

    metrics = {
        "run_id": row["run_id"],
        "treatment_id": treatment_id,
        "algorithm": algorithm,
        "use_masking": False,
        "window_size": 0,
        "tail_size": 0,
        "model_path": "",
        "trace_file": str(row["trace_file"]),
        "topology_file": str(row["topology_file"]),
        "node_file": str(row["node_file"]),
        "split_id": split_id,
        "episode_reward": 0.0,
        "decision_count": 0,
        "decision_latency_mean_ms": 0.0,
        "eval_wall_s": round(elapsed, 2),
        "max_waiting":   float(max_w),
        "avg_waiting":   float(avg_w),
        "max_slowdown":  float(max_s),
        "avg_slowdown":  float(avg_s),
        "avg_turnaround": float(avg_t),
        "cpu_utilization": cpu_utilization,
        "gpu_utilization": gpu_utilization
    }

    write_csv(pd.DataFrame([metrics]), out_csv)
    write_json(metrics, out_metrics)

    print(
        f"[{partition}] {algorithm} — done "
        f"(avg_wait={avg_w:.1f}s  avg_slowdown={avg_s:.4f}  "
        f"wall={elapsed:.0f}s)"
    )

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all heuristic baseline combinations across partitions."
    )
    parser.add_argument(
        "--algorithm",
        "--algo",
        dest="algorithm",
        metavar="ALGORITHM",
        help="Name of algorithm.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--split_id",
        default=None,
        dest="split_id",
        metavar="SPLIT_ID",
        help="Split ID to use (e.g., 'physical_job_r70'). Optional; auto-detects if only one split exists.",
        type=str,
    )
    parser.add_argument(
        "--partition",
        choices=("physical", "deeplearn"),
        default="physical",
        help="Which partition(s) to run. Default: physical.",
    )
    parser.add_argument(
        "--result-dir",
        default="result",
        metavar="DIR",
        help="Root directory for result CSVs and metrics JSON. Default: result/",
    )
    parser.add_argument(
        "--manifest-path",
        default="logs/baseline_run_log.csv",
        metavar="PATH",
        help="Path to traditional algorithm manifest Default: logs/baseline_run_log.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run combinations even if output already exists.",
    )
    return parser.parse_args()


def main():
    args    = parse_args()
    use_masking = False
    treatment_id = f"{args.algorithm}__mask_false"
    trace = f"data/splits/{args.split_id}.tsv"
    manifest_path = Path(args.manifest_path)
    result_path = Path(args.result_dir) / args.partition

    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        already_run = existing[
            (existing["algorithm"] == args.algorithm) &
            (existing["split_id"] == args.split_id) 
        ]
        if not already_run.empty and not args.force:
            print(f"[SKIP] {args.algorithm} already in manifest")
            return

    run_id = write_manifest_entry(
        treatment_id=treatment_id,
        algorithm=args.algorithm,
        use_masking=use_masking,
        seed=None, 
        window_size=0,
        tail_size=0, 
        split_id=args.split_id, 
        model_path="", 
        trace_file=trace, 
        topology_file=PARTITION_CONFIGS[args.partition]["topology"], 
        node_file=PARTITION_CONFIGS[args.partition]["nodes"], 
        manifest_path=manifest_path
    )

    manifest = pd.read_csv(manifest_path)
    row = manifest.loc[manifest["run_id"] == run_id].iloc[0].to_dict()
    run_one(row, run_id, args.partition, result_path)

if __name__ == "__main__":
    main()
