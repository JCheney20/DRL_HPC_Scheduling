"""run_baseline.py

Run traditional (non-DRL) scheduling heuristics as deterministic baselines,
one algorithm x one trace per invocation -- no seeds, since these algorithms
have no stochastic component and repeating them would waste compute for zero
statistical benefit.

Output contract (kept separate from, but schema-compatible with, the DRL
eval/aggregate pipeline -- see TODO.md Phase 3 "Baselines: separate stats,
combined visuals" and methodology_protocol.md's documented decision that
baselines are reported descriptively and excluded from the DRL-only
Friedman/Nemenyi/Wilcoxon/Page-trend hypothesis tests):

  result/{partition}/{run_id}_metrics.csv   : one eval_wide-compatible row
  result/{partition}/{run_id}_metrics.json  : same row, as JSON sidecar
  logs/baseline_run_log.csv                 : manifest entry (seed="" )

Use baseline_aggregate.py to fold these into baseline_summary.csv (per-trace,
no seed averaging needed -- a deterministic algorithm has exactly one value),
and baseline_compare.py to test a specific best-DRL-vs-best-baseline pair via
one-sample Wilcoxon (NOT Friedman -- there are no repeated seeds to block on
for the baseline side, so it cannot sit inside the seed-matched DRL matrix).
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from HPCsim.HPCsim import HPCsim
from utils import PARTITION_CONFIGS, write_csv, write_json, write_manifest_entry


def run_one(row: dict, run_id: str, partition: str, result_dir: Path) -> None:
    algorithm = str(row["algorithm"])
    trace_file = str(row["trace_file"])
    allocator = "best_fit"
    split_id = row["split_id"]
    treatment_id = f"{algorithm}__mask_false"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_csv = result_dir / f"{run_id}_metrics.csv"
    out_metrics = result_dir / f"{run_id}_metrics.json"

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
    # HPCsim.run() always writes to a fixed "result/{algo}+{allocator}.csv"
    # regardless of the caller's result_dir -- move it into place immediately
    # after the run completes (not deferred), since two concurrent baseline
    # runs for the SAME algorithm on DIFFERENT traces would otherwise race
    # on this fixed path if ever parallelised. (See HPCsim.run(): the source
    # path is hardcoded inside HPCsim itself, not something run_baseline.py
    # can pass in -- this rename is the only mitigation available here.)
    fixed_source_path = Path(f"result/{algorithm}+{allocator}.csv")
    fixed_source_path.replace(out_csv.with_suffix(".raw.csv"))

    max_w, avg_w = env.evaluator.waiting_time()
    max_s, avg_s = env.evaluator.bounded_slowdown()
    avg_t = env.evaluator.average_turnaround()
    cpu_utilization, gpu_utilization = env.utilization()
    elapsed = time.time() - t0

    # eval_wide-compatible row -- see utils.py's EVAL_REQUIRED for the full
    # required column set. "seed" is included (empty string, not omitted)
    # so this row has the same SCHEMA as a DRL eval row even though baselines
    # have no seed dimension; aggregate_results.py's validate_finite_numeric
    # only checks CORE_METRICS, which does not include "seed", so an empty
    # seed here does not trip any existing validation.
    metrics = {
        "run_id": row["run_id"],
        "treatment_id": treatment_id,
        "algorithm": algorithm,
        "use_masking": False,
        "window_size": 0,
        "tail_size": 0,
        "seed": "",
        "split_id": split_id,
        "model_path": "",
        "trace_file": trace_file,
        "topology_file": str(row["topology_file"]),
        "node_file": str(row["node_file"]),
        "episode_reward": 0.0,
        "decision_count": 0,
        "decision_latency_mean_ms": 0.0,
        "eval_wall_s": round(elapsed, 2),
        "max_waiting": float(max_w),
        "avg_waiting": float(avg_w),
        "max_slowdown": float(max_s),
        "avg_slowdown": float(avg_s),
        "avg_turnaround": float(avg_t),
        "cpu_utilization": float(cpu_utilization),
        "gpu_utilization": float(gpu_utilization),
    }

    write_csv(pd.DataFrame([metrics]), out_csv)
    write_json(metrics, out_metrics)

    print(
        f"[{partition}] {algorithm} — done "
        f"(avg_wait={avg_w:.1f}s  avg_slowdown={avg_s:.4f}  "
        f"wall={elapsed:.0f}s)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one traditional scheduling baseline (algorithm x trace, no seeds)."
    )
    parser.add_argument(
        "--algorithm", "--algo", dest="algorithm", metavar="ALGORITHM",
        help="Name of algorithm (must be in TRAD_ALGORITHMS).", required=True, type=str,
    )
    parser.add_argument(
        "--split_id", default=None, dest="split_id", metavar="SPLIT_ID",
        help="Split ID to use (e.g., 'physical_job_r70').", required=True, type=str,
    )
    parser.add_argument(
        "--partition", choices=("physical", "deeplearn"), default="physical",
        help="Which partition to run. Default: physical.",
    )
    parser.add_argument(
        "--result-dir", default="result", metavar="DIR",
        help="Root directory for result CSVs and metrics JSON. Default: result/",
    )
    parser.add_argument(
        "--manifest-path", default="logs/baseline_run_log.csv", metavar="PATH",
        help="Path to traditional algorithm manifest. Default: logs/baseline_run_log.csv",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Re-run even if this algorithm/split_id is already in the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    treatment_id = f"{args.algorithm}__mask_false"
    trace = f"data/splits/{args.split_id}.tsv"
    manifest_path = Path(args.manifest_path)
    result_path = Path(args.result_dir) / args.partition

    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        already_run = existing[
            (existing["algorithm"] == args.algorithm) & (existing["split_id"] == args.split_id)
        ]
        if not already_run.empty and not args.force:
            print(f"[SKIP] {args.algorithm} already in manifest for split_id={args.split_id}")
            return

    run_id = write_manifest_entry(
        treatment_id=treatment_id,
        algorithm=args.algorithm,
        use_masking=False,
        seed=None,
        window_size=0,
        tail_size=0,
        split_id=args.split_id,
        model_path="",
        trace_file=trace,
        topology_file=PARTITION_CONFIGS[args.partition]["topology"],
        node_file=PARTITION_CONFIGS[args.partition]["nodes"],
        manifest_path=manifest_path,
    )

    manifest = pd.read_csv(manifest_path)
    row = manifest.loc[manifest["run_id"] == run_id].iloc[0].to_dict()
    run_one(row, run_id, args.partition, result_path)


if __name__ == "__main__":
    main()
