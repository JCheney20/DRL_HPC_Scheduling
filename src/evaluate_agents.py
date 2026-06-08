"""evaluate_agents.py

Deterministic evaluation runner for trained RL agents.

Outputs per-run metrics and metadata sidecars for reproducibility.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sb3_contrib.common.maskable.utils import get_action_masks

from src.HPCsim.HPCsim import HPCsim
from src.utils import (
    ALGORITHMS,
    EvalResult,
    RunSpec,
    add_standard_debug_args,
    build_eval_metadata,
    load_run_manifest,
    safe_metric_access,
    validate_loaded_manifest,
    validate_not_holdout,
    write_dict_outputs,
    write_json,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Deterministic RL evaluation runner")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output-dir", default="result/eval_runs", type=str)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--filter-seed",
        type=int,
        default=None,
        help="Only evaluate runs matching this seed.",
    )
    add_standard_debug_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest loading and validation
# ---------------------------------------------------------------------------


def load_manifest_specs(manifest_path: Path) -> list[RunSpec]:
    df = load_run_manifest(manifest_path)
    validate_loaded_manifest(df, context="run_manifest")
    specs: list[RunSpec] = []
    for row in df.itertuples(index=False):
        specs.append(
            RunSpec(
                run_id=row.run_id,  # type: ignore
                treatment_id=row.treatment_id,  # type: ignore
                algorithm=row.algorithm.strip().lower(),  # type: ignore
                use_masking=bool(row.use_masking),  # type: ignore
                window_size=int(row.window_size),  # type: ignore
                tail_size=int(row.tail_size),  # type: ignore
                seed=None if pd.isna(row.seed) else int(row.seed),  # type: ignore
                split_id=row.split_id,  # type: ignore
                model_path=row.model_path,  # type: ignore
                trace_file=row.trace_file,  # type: ignore
                topology_file=row.topology_file,  # type: ignore
                node_file=row.node_file,  # type: ignore
            )
        )
    return specs


def validate_run_spec(spec: RunSpec) -> None:
    if spec.algorithm not in ALGORITHMS:
        raise ValueError(f"[{spec.run_id}] Unknown algorithm: {spec.algorithm}")

    validate_not_holdout(spec.trace_file, context=spec.run_id, raise_error=True)

    if not Path(spec.model_path).exists():
        raise FileNotFoundError(
            f"[{spec.run_id}] model_path not found: {spec.model_path}"
        )


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def build_env(spec: RunSpec, seed: int | None) -> HPCsim:
    return HPCsim(
        topology_file=f"data/topology/{spec.topology_file}",
        allocator="best_fit",
        node_file=f"data/topology/{spec.node_file}",
        trace_file=f"data/{spec.trace_file}",
        random_job=False,
        seed=seed,
        window_size=spec.window_size,
        tail_size=spec.tail_size,
    )


def load_model(spec: RunSpec, env: HPCsim):
    cls = ALGORITHMS[spec.algorithm]
    return cls.load(spec.model_path, env=env)


def evaluate_one_run(
    spec: RunSpec,
    deterministic: bool,
    seed_override: int | None = None,
    max_steps: int | None = None,
) -> EvalResult:
    seed = spec.seed if seed_override is None else seed_override

    validate_run_spec(spec)
    env = build_env(spec, seed=seed)
    model = load_model(spec, env)
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    episode_reward = 0.0
    n_steps = 0
    decision_latencies: list[float] = []
    t_start = time.perf_counter()

    while not (done or truncated):
        if max_steps is not None and n_steps >= max_steps:
            break

        t_dec0 = time.perf_counter()

        if spec.use_masking:
            action_masks = get_action_masks(env)
            action, _ = model.predict(
                obs, deterministic=deterministic, action_masks=action_masks
            )
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        decision_latencies.append(time.perf_counter() - t_dec0)
        obs, reward, done, truncated, _ = env.step(action)
        if not np.isfinite(reward):
            raise ValueError(f"[{spec.run_id}] Non-finite reward encountered: {reward}")
        episode_reward += float(reward)
        n_steps += 1

    eval_wall_s = time.perf_counter() - t_start
    max_w, avg_w = safe_metric_access(
        env.evaluator.waiting_time, (0.0, 0.0), "waiting_time"
    )
    max_s, avg_s = safe_metric_access(
        env.evaluator.bounded_slowdown, (0.0, 0.0), "bounded_slowdown"
    )
    avg_t = safe_metric_access(
        env.evaluator.average_turnaround, 0.0, "average_turnaround"
    )
    cpu_util, gpu_util = safe_metric_access(env.utilization, (0.0, 0.0), "utilization")

    return EvalResult(
        run_id=spec.run_id,
        treatment_id=spec.treatment_id,
        algorithm=spec.algorithm,
        use_masking=spec.use_masking,
        window_size=spec.window_size,
        tail_size=spec.tail_size,
        seed=seed,
        split_id=spec.split_id,
        model_path=spec.model_path,
        trace_file=spec.trace_file,
        topology_file=spec.topology_file,
        node_file=spec.node_file,
        episode_reward=episode_reward,
        decision_count=n_steps,
        decision_latency_mean_ms=float(np.mean(decision_latencies) * 1000.0)
        if decision_latencies
        else 0.0,
        eval_wall_s=eval_wall_s,
        max_waiting=float(max_w),
        avg_waiting=float(avg_w),
        max_slowdown=float(max_s),
        avg_slowdown=float(avg_s),
        avg_turnaround=float(avg_t),
        cpu_utilization=float(cpu_util),
        gpu_utilization=float(gpu_util),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_eval_outputs(
    output_dir: Path, result: EvalResult, metadata: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = result.run_id
    write_dict_outputs(
        result.__dict__, f"{run_id}_metrics", output_dir, as_json=True, as_csv=True
    )
    write_json(metadata, output_dir / f"{run_id}_metadata.json")


def write_summary(
    output_root: Path, results: list[EvalResult], failures: list[dict[str, str]]
) -> None:
    summary = {
        "total_runs": len(results) + len(failures),
        "success_runs": len(results),
        "failed_runs": len(failures),
        "failures": failures,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(summary, output_root / "eval_summary.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_root = Path(args.output_dir)
    runs_dir = output_root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    specs = load_manifest_specs(manifest_path)

    if args.filter_seed is not None:
        specs = [s for s in specs if s.seed == args.filter_seed]
        if not specs:
            print(f"[WARN] No runs found for seed {args.filter_seed} in manifest")
            sys.exit(0)

    if args.limit_runs is not None:
        specs = specs[-args.limit_runs :]

    results: list[EvalResult] = []
    failures: list[dict[str, str]] = []

    for spec in specs:
        try:
            result = evaluate_one_run(
                spec,
                deterministic=args.deterministic,
                seed_override=args.seed_override,
                max_steps=args.max_steps,
            )
            metadata = build_eval_metadata(
                command_args=sys.argv,
                split_id=spec.split_id,
                manifest_path=manifest_path,
                eval_stats={
                    "run_id": result.run_id,
                    "algorithm": result.algorithm,
                    "use_masking": result.use_masking,
                    "seed": result.seed,
                },
            )
            write_eval_outputs(runs_dir, result, metadata)
            results.append(result)
            print(f"[OK] {spec.run_id}")
        except Exception as e:
            failures.append({"run_id": spec.run_id, "error": str(e)})
            print(f"[FAIL] {spec.run_id} :: {e}")
            if args.fail_fast:
                break

    write_summary(output_root, results, failures)
    print(f"Done: success={len(results)} fail={len(failures)}")


if __name__ == "__main__":
    main()
