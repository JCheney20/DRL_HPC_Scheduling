"""run_baseline.py

Run traditional (non-DRL) scheduling heuristics as deterministic baselines,
one algorithm x one trace per invocation -- no seeds, since these algorithms
have no stochastic component and repeating them would waste compute for zero
statistical benefit.

Also runs the ONE non-deterministic member of the baseline set: the
uniform-random-over-valid-actions control, ``--algorithm random`` (reviewer
item N27). It is invoked from here so it lands in the same manifest, the same
result directory and the same summary table as the heuristics it is reported
beside -- but it does NOT use the heuristic code path below. It runs the RL MDP
so that it is a control for policy quality and nothing else; see
src/random_control.py for that argument in full. Being stochastic it requires
``--seed`` (the heuristics forbid it), and it is the only baseline whose
summary row carries a standard deviation.

Output contract (kept separate from, but schema-compatible with, the DRL
eval/aggregate pipeline -- see TODO.md Phase 3 "Baselines: separate stats,
combined visuals" and methodology_protocol.md's documented decision that
baselines are reported descriptively and excluded from the DRL-only
Friedman/Nemenyi/Wilcoxon/Page-trend hypothesis tests):

  result/{partition}/{stem}_metrics.csv     : one eval_wide-compatible row
  result/{partition}/{stem}_metrics.json    : same row, as JSON sidecar
  logs/baseline_run_log.csv                 : manifest entry (seed="" )

where {stem} is "{treatment_id}__{split_id}" for the heuristics and
"{treatment_id}__{split_id}__seed{seed}" for the random control -- a
deterministic name, so a --force re-run replaces its predecessor instead of
leaving a second file for baseline_aggregate's glob to trip over (see
metrics_stem).

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

from src.HPCsim.HPCsim import HPCsim
from src.naming import metrics_stem
from src.utils import (
    PARTITION_CONFIGS,
    RANDOM_ALGORITHM,
    RANDOM_TREATMENT_ID,
    RunSpec,
    write_csv,
    write_json,
    write_manifest_entry,
)

# src.random_control is imported lazily inside run_random(), not here:
# importing it at module scope would drag stable-baselines3 and sb3-contrib
# into every heuristic invocation, which needs neither.


def run_one(row: dict, run_id: str, partition: str, result_dir: Path) -> None:
    algorithm = str(row["algorithm"])
    trace_file = str(row["trace_file"])
    allocator = "best_fit"
    split_id = row["split_id"]
    treatment_id = f"{algorithm}__mask_false"
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = metrics_stem(treatment_id, str(split_id))
    out_csv = result_dir / f"{stem}_metrics.csv"
    out_metrics = result_dir / f"{stem}_metrics.json"

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
        "run_id": run_id,
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


def run_random(
    row: dict,
    run_id: str,
    seed: int,
    result_dir: Path,
    window_size: int,
    tail_size: int,
    max_steps: int | None,
) -> None:
    """Run the uniform-random-over-valid-actions control (N27).

    Deliberately does NOT call run_one(): that path is HPCsim.run(), the
    heuristic loop with backfill, which is a different simulator from the MDP
    the DRL treatments are evaluated on. Rolling the control through the
    heuristic loop would make it a fourth heuristic instead of a control for
    policy quality. See src/random_control.py for the full argument.
    """
    # Local import: keeps SB3 out of the heuristic path (see note at top).
    from src.random_control import rollout_random

    result_dir.mkdir(parents=True, exist_ok=True)

    # evaluate_agents.build_env prefixes "data/topology/" itself, matching the
    # DRL run manifest, which stores bare filenames. The baseline manifest
    # stores the full PARTITION_CONFIGS path, so strip the directory here
    # rather than diverging the two manifests' schemas over one algorithm.
    spec = RunSpec(
        run_id=run_id,
        treatment_id=str(row["treatment_id"]),
        algorithm=RANDOM_ALGORITHM,
        use_masking=True,
        window_size=window_size,
        tail_size=tail_size,
        seed=seed,
        split_id=str(row["split_id"]),
        model_path="",
        trace_file=str(row["trace_file"]),
        topology_file=Path(str(row["topology_file"])).name,
        node_file=Path(str(row["node_file"])).name,
    )

    metrics = rollout_random(spec, seed=seed, max_steps=max_steps)

    stem = metrics_stem(str(row["treatment_id"]), str(row["split_id"]), seed)
    write_csv(pd.DataFrame([metrics]), result_dir / f"{stem}_metrics.csv")
    write_json(metrics, result_dir / f"{stem}_metrics.json")

    print(
        f"[random seed={seed}] done "
        f"(avg_wait={metrics['avg_waiting']:.1f}s  "
        f"avg_slowdown={metrics['avg_slowdown']:.4f}  "
        f"steps={metrics['decision_count']}  "
        f"wall={metrics['eval_wall_s']:.0f}s)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one baseline: a deterministic scheduling heuristic "
                    "(algorithm x trace, no seeds), or the stochastic "
                    "uniform-random-over-valid-actions control (--algorithm "
                    "random, one seed per invocation)."
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
    parser.add_argument(
        "--seed", default=None, type=int, metavar="SEED",
        help="Seed for the stochastic 'random' control. Required for "
             "--algorithm random; rejected for the deterministic heuristics, "
             "where it would imply a variability they do not have.",
    )
    parser.add_argument(
        "--window-size", default=512, type=int,
        help="MDP queue window. Only used by --algorithm random, which must "
             "match the DRL treatments' action space to be a valid control. "
             "Default: 512 (config.yaml window_size).",
    )
    parser.add_argument(
        "--tail-size", default=64, type=int,
        help="MDP queue tail. Only used by --algorithm random; see "
             "--window-size. Default: 64 (config.yaml tail_size).",
    )
    parser.add_argument(
        "--max-steps", default=None, type=int,
        help="Optional decision-step cap for --algorithm random (smoke tests). "
             "Default: unset, i.e. roll the full trace like the DRL evals.",
    )
    args = parser.parse_args()

    if args.algorithm == RANDOM_ALGORITHM and args.seed is None:
        parser.error("--algorithm random is stochastic and requires --seed")
    if args.algorithm != RANDOM_ALGORITHM and args.seed is not None:
        parser.error(
            f"--seed is not valid for '{args.algorithm}': the heuristics are "
            f"deterministic, so a seeded rerun would produce an identical row "
            f"and baseline_aggregate would reject it as a duplicate"
        )
    return args


def main() -> None:
    args = parse_args()
    is_random = args.algorithm == RANDOM_ALGORITHM
    treatment_id = RANDOM_TREATMENT_ID if is_random else f"{args.algorithm}__mask_false"
    trace = f"data/splits/{args.split_id}.tsv"
    manifest_path = Path(args.manifest_path)
    result_path = Path(args.result_dir)

    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        already_run = existing[
            (existing["algorithm"] == args.algorithm) & (existing["split_id"] == args.split_id)
        ]
        # For the random control the unit of work is (algorithm, split, seed),
        # not (algorithm, split): 10 seeds are 10 legitimately distinct rows,
        # so matching on algorithm alone would skip nine of them.
        if is_random:
            already_run = already_run[
                pd.to_numeric(already_run["seed"], errors="coerce") == args.seed
            ]
        if not already_run.empty and not args.force:
            seed_note = f" seed={args.seed}" if is_random else ""
            print(
                f"[SKIP] {args.algorithm}{seed_note} already in manifest "
                f"for split_id={args.split_id}"
            )
            return

    run_id = write_manifest_entry(
        treatment_id=treatment_id,
        algorithm=args.algorithm,
        # The control runs the masked MDP, so use_masking is genuinely True and
        # the window/tail must equal the DRL treatments' for the action space to
        # match. The heuristics keep 0/0/False: they have no action space.
        use_masking=is_random,
        seed=args.seed,
        window_size=args.window_size if is_random else 0,
        tail_size=args.tail_size if is_random else 0,
        split_id=args.split_id,
        model_path="",
        trace_file=trace,
        topology_file=PARTITION_CONFIGS[args.partition]["topology"],
        node_file=PARTITION_CONFIGS[args.partition]["nodes"],
        manifest_path=manifest_path,
    )

    manifest = pd.read_csv(manifest_path)
    row = manifest.loc[manifest["run_id"] == run_id].iloc[0].to_dict()
    if is_random:
        run_random(
            row,
            run_id,
            seed=args.seed,
            result_dir=result_path,
            window_size=args.window_size,
            tail_size=args.tail_size,
            max_steps=args.max_steps,
        )
    else:
        run_one(row, run_id, args.partition, result_path)


if __name__ == "__main__":
    main()
