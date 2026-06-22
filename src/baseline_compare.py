"""baseline_compare.py

Statistically compare the selected best DRL algorithm (from select_best.py's
best_algorithm.json) against each traditional baseline's deterministic
result, via a ONE-SAMPLE Wilcoxon signed-rank test.

Why one-sample, not the two-sample/paired Wilcoxon already in
statistical_test.py: that test compares two SETS of seed-matched
observations (DRL algo A's 5 seeds vs DRL algo B's 5 seeds, paired by seed).
A baseline has no seeds -- it is a single fixed number. There is nothing to
pair it against. The correct non-parametric test for "does this distribution
of values differ from one known fixed reference point" is the one-sample
Wilcoxon signed-rank test: pass (drl_seed_values - baseline_value) to
scipy.stats.wilcoxon with no second argument, which tests whether that
difference is symmetric around zero.

This deliberately does NOT feed into Friedman/Nemenyi/CD-diagram machinery --
per methodology_protocol.md and TODO.md's documented decision, baselines are
descriptive and excluded from the DRL-only hypothesis-testing matrix. This
script produces a separate, small comparison artefact for the
results/discussion section, not a replacement for or addition to
stats_summary.json.

Usage:
    python src/baseline_compare.py \\
        --best-algorithm result/physical_job/best_algorithm.json \\
        --seed-summary result/physical_job/aggregate/seed_summary.csv \\
        --baseline-summary result/physical_job/baseline/baseline_summary.csv \\
        --output result/physical_job/baseline/baseline_comparison.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

from utils import METRIC_DIRECTION, write_csv

PRIMARY_METRICS = ["avg_waiting", "avg_slowdown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-sample Wilcoxon comparison: best DRL algorithm vs each baseline."
    )
    parser.add_argument("--best-algorithm", required=True, type=str, help="Path to best_algorithm.json")
    parser.add_argument("--seed-summary", required=True, type=str, help="Path to seed_summary.csv")
    parser.add_argument("--baseline-summary", required=True, type=str, help="Path to baseline_summary.csv")
    parser.add_argument("--metrics", default=PRIMARY_METRICS, nargs="+", type=str)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def one_sample_comparison(
    drl_values: pd.Series, baseline_value: float, metric: str, alpha: float
) -> dict:
    direction = METRIC_DIRECTION.get(metric, "lower_is_better")
    diffs = drl_values.to_numpy() - baseline_value
    n = len(diffs)

    if (diffs == 0).all():
        # scipy.stats.wilcoxon raises on an all-zero difference vector
        # (no information to rank) -- report this explicitly rather than
        # letting the exception propagate.
        return {
            "metric": metric, "n_drl_seeds": n, "baseline_value": float(baseline_value),
            "drl_mean": float(drl_values.mean()), "drl_median": float(drl_values.median()),
            "statistic": None, "p_value": None, "significant": False,
            "drl_better": None, "note": "DRL values identical to baseline; no variation to test",
        }

    result = stats.wilcoxon(diffs)
    drl_median = float(drl_values.median())
    if direction == "lower_is_better":
        drl_better = drl_median < baseline_value
    else:
        drl_better = drl_median > baseline_value

    return {
        "metric": metric,
        "n_drl_seeds": n,
        "baseline_value": float(baseline_value),
        "drl_mean": float(drl_values.mean()),
        "drl_median": drl_median,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "drl_better": bool(drl_better),
        "note": "",
    }


def main() -> None:
    args = parse_args()

    with open(args.best_algorithm) as f:
        best = json.load(f)
    best_treatment_id = best["treatment_id"]

    seed_summary = pd.read_csv(args.seed_summary)
    baseline_summary = pd.read_csv(args.baseline_summary)

    drl_rows = seed_summary[seed_summary["treatment_id"] == best_treatment_id]
    if drl_rows.empty:
        print(f"[ERROR] No seed_summary rows found for best treatment_id={best_treatment_id}", file=sys.stderr)
        sys.exit(1)

    results = []
    for _, baseline_row in baseline_summary.iterrows():
        for metric in args.metrics:
            mean_col = f"{metric}_mean" if f"{metric}_mean" in drl_rows.columns else metric
            if mean_col not in drl_rows.columns:
                continue
            baseline_col = f"{metric}_mean_mean"
            if baseline_col not in baseline_row:
                continue

            comparison = one_sample_comparison(
                drl_rows[mean_col], baseline_row[baseline_col], metric, args.alpha
            )
            comparison["drl_treatment_id"] = best_treatment_id
            comparison["baseline_treatment_id"] = baseline_row["treatment_id"]
            results.append(comparison)

    if not results:
        print("[ERROR] No comparable metrics found between DRL and baseline summaries.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(results)
    write_csv(out_df, Path(args.output))
    print(f"[OK] {len(out_df)} comparison(s) -> wrote {args.output}")


if __name__ == "__main__":
    main()
