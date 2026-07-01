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
    One row per treatment_id (== one row per algorithm, since baselines have
    no masking variants and no seeds). Column names mirror
    algorithm_summary.csv's "{metric}_mean_mean" convention -- not because
    any averaging happens here, but so downstream plotting code that already
    expects that column shape can read this file unmodified.
    """
    keep_cols = ["treatment_id", "algorithm", "use_masking", "split_id"] + CORE_METRICS
    summary = eval_wide[keep_cols].copy()
    rename_map = {metric: f"{metric}_mean_mean" for metric in CORE_METRICS}
    summary = summary.rename(columns=rename_map)
    return summary


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
    duplicates = eval_wide[eval_wide.duplicated(subset=["treatment_id", "split_id"], keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Duplicate (treatment_id, split_id) rows found:\n{duplicates}")

    summary = build_baseline_summary(eval_wide)
    write_csv(eval_wide, output_path.parent / "baseline_eval_wide.csv")
    write_csv(summary, output_path)
    print(f"[OK] {len(summary)} baseline algorithm(s) -> wrote {output_path}")


if __name__ == "__main__":
    main()
