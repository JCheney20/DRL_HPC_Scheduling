"""baseline_aggregate.py

Aggregate traditional-scheduler baseline outputs (from run_baseline.py) into
baseline_summary.csv -- analogous to aggregate_results.py's
algorithm_summary.csv, but WITHOUT any seed-averaging, since each baseline
algorithm produces exactly one deterministic value per trace. There is
nothing to average across.

This is intentionally a separate, much smaller script from
aggregate_results.py rather than a mode flag on it: the DRL aggregation
pipeline's entire structure (seed_summary -> mean/std per treatment ->
algorithm_summary) assumes repeated stochastic measurements per treatment,
which baselines do not have. Folding a "skip the seed step" branch into
aggregate_results.py would complicate a script that's already correct and
tested for its actual job; a small, single-purpose script is clearer than a
conditional branch through code that doesn't apply to this case.

Output schema matches the *_mean column naming convention of
algorithm_summary.csv (e.g. "avg_waiting_mean_mean") purely so
visualise.py's existing column-selection code (write_comparison_csv,
draw_bar_graphs, etc.) can read baseline_summary.csv with zero changes if a
combined-visual comparison is wanted -- per TODO.md Phase 3, "keep baseline
stats separate but allow combined visual comparison."

Usage:
    python src/baseline_aggregate.py \\
        --result-dir result/physical \\
        --output baseline_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.utils import EVAL_REQUIRED, CORE_METRICS, validate_finite_numeric, validate_required_columns, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate traditional baseline eval outputs.")
    parser.add_argument(
        "--result-dir", required=True, type=str,
        help="Directory containing baseline *_metrics.csv files (e.g. result/physical).",
    )
    parser.add_argument(
        "--output", default="baseline_summary.csv", type=str,
        help="Output path for the aggregated summary CSV.",
    )
    parser.add_argument(
        "--strict", action=argparse.BooleanOptionalAction, default=True,
        help="Fail if any discovered file does not match the eval_wide schema.",
    )
    return parser.parse_args()


def discover_baseline_metrics(result_dir: Path) -> list[Path]:
    return sorted(result_dir.glob("*_metrics.csv"))


def load_and_validate(path: Path, strict: bool) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    try:
        validate_required_columns(df, EVAL_REQUIRED, context=f"baseline[{path.name}]")
        validate_finite_numeric(df, CORE_METRICS, context=f"baseline[{path.name}]")
    except ValueError as e:
        if strict:
            raise
        print(f"[WARNING] {e}, skipping {path.name}")
        return None
    return df


def build_baseline_summary(eval_wide: pd.DataFrame) -> pd.DataFrame:
    """
    One row per treatment_id. Column names mirror algorithm_summary.csv's
    "{metric}_mean_mean" / "{metric}_mean_std" convention so downstream
    plotting and table code that already expects that column shape can read
    this file unmodified.

    Most baselines are deterministic heuristics with a single seedless row, so
    grouping is a no-op for them: the mean of one value is that value, and its
    std is NaN (ddof=1 on n=1) -- which is the correct rendering, since
    build_results_data's fmt_mean_std omits the +/- term on NaN. The grouping
    exists for the one baseline that is stochastic: the N27
    uniform-random-over-valid-actions control, which contributes one row per
    seed and whose spread across seeds is the whole point of running it. A
    control reported without its variance could not answer the question N27
    asks (is a masked-but-unlearned policy distinguishable from MaskablePPO?),
    so the std column is load-bearing rather than decorative.
    """
    group_keys = ["treatment_id", "algorithm", "use_masking", "split_id"]
    grouped = eval_wide.groupby(group_keys, dropna=False)[CORE_METRICS]
    means = grouped.mean().rename(columns={m: f"{m}_mean_mean" for m in CORE_METRICS})
    stds = grouped.std(ddof=1).rename(columns={m: f"{m}_mean_std" for m in CORE_METRICS})
    # n_seeds distinguishes "deterministic, one run" from "10 stochastic runs"
    # in the file itself, so a reader does not have to infer it from a NaN std.
    n_seeds = grouped.size().rename("n_seeds")

    summary = pd.concat([means, stds, n_seeds], axis=1).reset_index()
    # Interleave so each metric's mean and std sit together, then the ids.
    ordered = group_keys + ["n_seeds"]
    for metric in CORE_METRICS:
        ordered += [f"{metric}_mean_mean", f"{metric}_mean_std"]
    return summary[ordered]


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_path = Path(args.output)

    metric_files = discover_baseline_metrics(result_dir)
    if not metric_files:
        print(f"[ERROR] No *_metrics.csv files found in {result_dir}")
        sys.exit(1)

    frames = []
    for path in metric_files:
        df = load_and_validate(path, strict=args.strict)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No valid baseline metrics to aggregate. Exiting.")
        sys.exit(1)

    eval_wide = pd.concat(frames, ignore_index=True)
    # Seed is part of the key. It is empty for the deterministic heuristics --
    # so (treatment_id, split_id, "") still catches a genuine duplicate run of
    # one of those -- and distinct for each seed of the N27 random control,
    # whose 10 rows are the intended output rather than a collision.
    dup_keys = ["treatment_id", "split_id", "seed"]
    duplicates = eval_wide[eval_wide.duplicated(subset=dup_keys, keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Duplicate (treatment_id, split_id, seed) rows found:\n{duplicates}")

    summary = build_baseline_summary(eval_wide)
    write_csv(eval_wide, output_path.parent / "baseline_eval_wide.csv")
    write_csv(summary, output_path)
    seeded = summary[summary["n_seeds"] > 1]
    print(
        f"[OK] {len(summary)} baseline algorithm(s) from {len(eval_wide)} run(s) "
        f"-> wrote {output_path}"
        + (f" ({len(seeded)} stochastic, seed-averaged)" if not seeded.empty else "")
    )


if __name__ == "__main__":
    main()
