"""
statistical_tests.py

Runs the full non-parametric repeated-measures statistical pipeline
on the aggregated per-seed evaluation table produced by aggregate_results.py.

Outputs:
  - stats_summary.json     : full structured result payload (see schema in docs)
  - pairwise_nemenyi.csv   : wide-format pairwise p-values per metric
  - confidence_intervals.csv : Wilcoxon-based NP CI per pair
  - cd_diagram_input.csv   : average ranks per treatment per metric
  - page_trend.csv         : convergence trend test outputs
  - stats_meta.json        : reproducibility sidecar

Pipeline per metric:
  1. Data sufficiency check  (min treatments, min common blocks)
  2. Shapiro-Wilk            (normality diagnostics, report-only — non-blocking)
  3. Friedman test           (omnibus non-parametric repeated-measures)
  4. Kendall W               (effect size for Friedman result)
  5. Nemenyi post-hoc         (only if Friedman is significant)
  6. Wilcoxon NP CIs          (pairwise, always run if data sufficient)
  7. Confidence curves        (for Nemenyi-significant pairs)
  8. CD diagram input         (average ranks for visualisation tooling)
  9. Page trend test          (convergence; optional input-driven)
 10. Compile Metric results

References:
  - scipy.stats.friedmanchisquare: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html
  - scipy.stats.shapiro:           https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
  - scikit-posthocs conover:       https://scikit-posthocs.readthedocs.io/en/latest/tutorial/
  - numpy random Generator:        https://numpy.org/doc/stable/reference/random/generator.html
  - pandas pivot_table:            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html
  - argparse:                      https://docs.python.org/3/library/argparse.html
  - json:                          https://docs.python.org/3/library/json.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

from utils import (
    ALGORITHMS,
    build_run_metadata,
    interpret_stat,
    load_seed_summary,
    parse_bool,
    validate_finite_numeric,
    validate_no_duplicates,
    validate_required_columns,
    write_csv,
    write_json,
)


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SEED_SUMMARY_REQUIRED_IDS = [
    "split_id", "seed", "algorithm", "use_masking", "treatment_id",
]

PRIMARY_METRICS = [
    "avg_waiting",
    "avg_slowdown",
]

ALL_METRICS = [
    "avg_waiting", "avg_slowdown",
    "max_waiting", "max_slowdown",
    "avg_turnaround",
    "cpu_utilization", "gpu_utilization",
    "episode_reward",
    "decision_latency_mean_ms",
    "eval_wall_s",
]

# TODO: Define traditional baselines if included (e.g., fcfs, sjf, etc.).
# Ref: https://snakemake.readthedocs.io/en/stable/
TRAD_ALGORITHMS: list[str] = []

METRIC_DIRECTION: dict[str, str] = {
    "avg_waiting":              "lower_is_better",
    "avg_slowdown":             "lower_is_better",
    "max_waiting":              "lower_is_better",
    "max_slowdown":             "lower_is_better",
    "avg_turnaround":           "lower_is_better",
    "cpu_utilization":          "higher_is_better",
    "gpu_utilization":          "higher_is_better",
    "episode_reward":           "higher_is_better",
    "decision_latency_mean_ms": "lower_is_better",
    "eval_wall_s":              "lower_is_better",
}

KENDALL_W_THRESHOLDS = [
    (0.0, "Slight Agreement"),
    (0.2, "Fair Agreement"),
    (0.4, "Moderate Agreement"),
    (0.6, "Substantial Agreement"),
    (0.7, "Almost Perfect Agreement"),
]

VDA_THRESHOLDS = [
    (0.56, "small"),
    (0.64, "medium"),
    (0.71, "large"),
]

MIN_ALGORITHMS = 3
MIN_BLOCKS = 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-parametric statistical tests on seed-level RL evaluation results."
    )
    parser.add_argument("--input", required=True, type=str, help="Path to seed_summary.csv from aggregate_results.py.")
    parser.add_argument("--output-dir", default="result/stats", type=str, help="Directory to write statistical outputs.")
    parser.add_argument("--metrics", default=None, nargs="+", type=str, help="Metrics to test. Defaults to ALL_METRICS.")
    parser.add_argument("--alpha", default=0.05, type=float, help="Significance level for hypothesis tests.")
    parser.add_argument("--bootstrap-reps", default=10000, type=int, help="Number of bootstrap resamples for median CIs.")
    parser.add_argument("--bootstrap-seed", default=42, type=int, help="Random seed for bootstrap reproducibility.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------

def validate_stats_input_schema(df: pd.DataFrame, metric_cols: list[str]) -> None:
    validate_required_columns(df, SEED_SUMMARY_REQUIRED_IDS, context="seed_summary")
    validate_required_columns(df, metric_cols, context="seed_summary_metrics")
    validate_finite_numeric(df, PRIMARY_METRICS, context="seed_summary_primary_metrics")
    # TODO: Baselines should remain in stats unless a test precondition fails.
    # TODO: Do not globally drop baselines; only skip specific tests when needed.

    invalid_algos = [algo for algo in df["algorithm"] if algo not in ALGORITHMS]
    if invalid_algos:
        raise ValueError(f"Invalid algorithms in seed summary {invalid_algos}")

    validate_no_duplicates(df, ["split_id", "seed", "treatment_id"], context="seed_summary")


# ---------------------------------------------------------------------------
# Data sufficiency
# ---------------------------------------------------------------------------

def check_data_sufficiency(
    df: pd.DataFrame,
    metric: str,
    min_algorithms: int = MIN_ALGORITHMS,
    min_blocks: int = MIN_BLOCKS,
) -> dict[str, Any]:
    df = df.dropna(subset=[metric], how="any")
    tbl = df.pivot_table(index="seed", columns="treatment_id", values=metric)
    tbl_complete = tbl.dropna(how="any")

    n_treatments = tbl.shape[1]
    n_blocks_total = tbl.shape[0]
    n_blocks_common = tbl_complete.shape[0]

    return {
        "n_treatments": n_treatments,
        "n_blocks_total": n_blocks_total,
        "n_blocks_common": n_blocks_common,
        "balanced_blocks": n_blocks_common == n_blocks_total,
        "min_requirements_met": n_treatments >= min_algorithms and n_blocks_common >= min_blocks,
    }


# ---------------------------------------------------------------------------
# Repeated-measures matrix
# ---------------------------------------------------------------------------

def build_repeated_measures_matrix(
    df: pd.DataFrame,
    metric: str,
    block_col: str = "seed",
    group_col: str = "treatment_id",
) -> pd.DataFrame:
    return df.pivot_table(index=block_col, columns=group_col, values=metric).dropna(how="any")


# ---------------------------------------------------------------------------
# Shapiro-Wilk diagnostics
# ---------------------------------------------------------------------------

def _shapiro_one(treatment_id: str, values: np.ndarray) -> dict[str, Any]:
    n = len(values)
    if n < 3:
        return {"treatment_id": treatment_id, "n": n, "W": None, "p_value": None, "valid": False, "note": "n < 3"}
    W, p_value = stats.shapiro(values)
    return {"treatment_id": treatment_id, "n": n, "W": float(W), "p_value": float(p_value), "valid": True, "note": ""}


def run_shapiro_diagnostics(matrix_df: pd.DataFrame) -> dict[str, Any]:
    per_treatment = []
    for col in matrix_df.columns:
        per_treatment.append(_shapiro_one(col, matrix_df[col].to_numpy()))
    return {"performed": True, "per_treatment": per_treatment, "note": None}


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------

def run_friedman_test(matrix_df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    if matrix_df.shape[1] < MIN_ALGORITHMS or matrix_df.shape[0] < MIN_BLOCKS:
        return {
            "performed": False,
            "chi2": None,
            "df": None,
            "p_value": None,
            "alpha": alpha,
            "significant": None,
            "n_blocks_common": int(matrix_df.shape[0]),
            "k_treatments": int(matrix_df.shape[1]),
        }
    stat, p = stats.friedmanchisquare(*[matrix_df[col] for col in matrix_df.columns])
    return {
        "performed": True,
        "chi2": float(stat),
        "df": int(matrix_df.shape[1] - 1),
        "p_value": float(p),
        "alpha": alpha,
        "significant": bool(p < alpha),
        "n_blocks_common": int(matrix_df.shape[0]),
        "k_treatments": int(matrix_df.shape[1]),
    }


def compute_kendall_w(chi2: float, n_blocks: int, k_groups: int) -> dict[str, Any]:
    W = float(chi2 / (n_blocks * (k_groups - 1))) if (n_blocks * (k_groups - 1)) != 0 else 0.0
    return {
        "measure": "Kendall W",
        "value": W,
        "interpretation": interpret_stat(W, KENDALL_W_THRESHOLDS),
    }


# ---------------------------------------------------------------------------
# Nemenyi post-hoc + VDA
# ---------------------------------------------------------------------------

def run_nemenyi_posthoc(
    matrix_df: pd.DataFrame,
    alpha: float,
    metric_name: str,
) -> dict[str, Any]:
    # TODO: Confirm this is the authoritative post-hoc (Nemenyi) for the paper.
    # TODO: Ensure output schema matches stats_summary.json contract.
    # Ref: https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_nemenyi_friedman.html
    # TODO: Skip Nemenyi only if a precondition fails (e.g., rank degeneracy).
    # TODO: Define degeneracy rule (e.g., <2 distinct mean ranks) and surface in result schema.
    p_matrix = sp.posthoc_nemenyi_friedman(matrix_df)
    mean_ranks = pd.Series(matrix_df.rank(axis=1).mean(axis=0))
    result_pairs = []

    pairs = [
        (a, b)
        for i, a in enumerate(matrix_df.columns)
        for j, b in enumerate(matrix_df.columns)
        if i < j
    ]

    for a, b in pairs:
        result_pairs.append(
            {
                "metric_name": metric_name,
                "treatment_a": a,
                "treatment_b": b,
                "mean_rank_a": float(mean_ranks[a]),
                "mean_rank_b": float(mean_ranks[b]),
                "rank_diff": float(mean_ranks[a] - mean_ranks[b]),
                "p_value": float(p_matrix.loc[a, b]),
                "significant": bool(p_matrix.loc[a, b] < alpha),
            }
        )

    return {
        "performed": True,
        "only_if_friedman_significant": True,
        "alpha": alpha,
        "pairs": result_pairs,
    }


# ---------------------------------------------------------------------------
# Bootstrap median CIs
# ---------------------------------------------------------------------------

def Wilcoxon_np_ci(values_a: np.ndarray, values_b: np.ndarray, alpha: float) -> dict[str, Any]:
    # TODO: Implement Wilcoxon non-parametric CI for paired differences per Carrasco (Sec 3.4).
    # TODO: Let l = number of paired observations; compute all l^2 differences.
    # TODO: Exact K: K = W_{alpha/2} - l(l+1)/2, where W_{alpha/2} is the alpha/2 percentile
    #       of the Wilcoxon two-sample statistic distribution (see Center, pp 156-162).
    # TODO: Approximate K for l > 20:
    #       K = (l^2 / 2) - z_{1-alpha/2} * sqrt(l^2(2l+1)/12), then round up.
    # TODO: CI is [Kth smallest diff, Kth largest diff] of the l^2 differences.
    # TODO: The CI uses l^2 pairwise differences; for large l this is O(l^2) memory/time.
    # TODO: Consider vectorized computation with NumPy broadcasting and/or chunking to avoid RAM spikes.
    # Ref: https://numpy.org/doc/stable/reference/generated/numpy.subtract.outer.html
    # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    # Ref: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    return {"ci_low": None, "ci_high": None, "alpha": alpha}


def confidence_curve(values_a: np.ndarray, values_b: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    # TODO: Produce per-pair confidence curves over a delta grid (Carrasco Sec 3.4.1).
    # TODO: x-axis: delta (null difference), y-axis: Wilcoxon p-value for values_a - values_b - delta.
    # TODO: Define a standard grid (e.g., linspace over percentile range of paired diffs).
    # TODO: Compute curve using vectorized differences if grid is large (avoid Python loops).
    # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    # Evaluate Wilcoxon p-values across null differences in grid
    rows = []
    for delta in grid:
        stat, p = stats.wilcoxon(values_a - values_b - delta)
        rows.append({"delta": float(delta), "p_value": float(p)})
    return pd.DataFrame(rows)



def bootstrap_median_ci(
    matrix_df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, Any]:
    # TODO: Decide whether to keep bootstrap median CIs in addition to Wilcoxon CI.
    # Ref: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
    rng = np.random.default_rng(seed)
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 - lower_pct / 100) * 100
    treatment_data = []

    for col in matrix_df.columns:
        vals = matrix_df[col].to_numpy()
        bootstrap_medians = [np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
        ci_low, ci_high = np.percentile(bootstrap_medians, [lower_pct, upper_pct])
        treatment_data.append(
            {
                "treatment_id": col,
                "n_blocks": vals.shape[0],
                "median": float(np.median(vals)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )

    return {
        "performed": True,
        "reps": n_boot,
        "ci_low": None,
        "ci_high": None,
        "method": "median_percentile",
        "per_treatment": treatment_data,
    }


# ---------------------------------------------------------------------------
# CD diagram input
# ---------------------------------------------------------------------------

def build_cd_diagram_input(matrix_df: pd.DataFrame) -> dict[str, Any]:
    # TODO: Confirm ranking direction per metric (lower_is_better vs higher_is_better).
    # Ref: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html
    avg_ranks = pd.Series(matrix_df.rank(axis=1, ascending=True).mean(axis=0))
    avg_rank_data = []
    for treatment_id in avg_ranks.index:
        algorithm, use_masking = treatment_id.split("__mask_")
        use_masking = parse_bool(use_masking)
        avg_rank_data.append(
            {
                "treatment_id": treatment_id,
                "algorithm": algorithm,
                "use_masking": use_masking,
                "avg_rank": float(avg_ranks[treatment_id]),
            }
        )

    return {"available": True, "average_ranks": avg_rank_data}


# ---------------------------------------------------------------------------
# Result compilation
# ---------------------------------------------------------------------------

def compile_metric_result(
    metric: str,
    sufficiency: dict[str, Any],
    shapiro: dict[str, Any],
    friedman: dict[str, Any],
    kendall_w: dict[str, Any],
    nemenyi: dict[str, Any],
    wilcoxon_ci: dict[str, Any],
    confidence_curves: dict[str, Any],
    page_trend: dict[str, Any],
    cd_input: dict[str, Any],
    descriptive: dict[str, Any],
    status: str = "ok",
    skip_reason: str | None = None,
) -> dict[str, Any]:
    # TODO: Align return schema with stats_summary.json contract.
    # TODO: Include explicit flags for each stage (performed/skipped) for fail-fast auditing.
    # TODO: Ensure confidence_curves and page_trend are always present (even if skipped).
    # TODO: Add "treatment_order" field so Page trend assumptions are explicit.
    # TODO: Add "metric_direction" to enforce rank direction downstream.
    # Ref: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html
    return {

        "metric_name": metric,
        "direction": METRIC_DIRECTION[metric],
        "status": status,
        "skip_reason": skip_reason,
        "data_sufficiency": sufficiency,
        "descriptive": descriptive,
        "shapiro": shapiro,
        "friedman": friedman,
        "kendall_w": kendall_w,
        "nemenyi": nemenyi,
        "wilcoxon_ci": wilcoxon_ci,
        "confidence_curves": confidence_curves,
        "cd_input": cd_input,
        "page_trend": page_trend,
    }


def _descriptive_stats(matrix_df: pd.DataFrame) -> dict[str, Any]:
    # TODO: Align descriptive stats with treatment order (ALGORITHMS + TRAD_ALGORITHMS).
    # Ref: https://pandas.pydata.org/docs/reference/api/pandas.Series.describe.html
    treatment_data = []
    mean_ranks = pd.Series(matrix_df.rank(axis=1).mean(axis=0))

    for col in matrix_df.columns:
        desc = pd.Series(matrix_df[col].describe())
        algorithm, use_masking = col.split("__mask_")
        use_masking = parse_bool(use_masking)
        treatment_data.append(
            {
                "treatment_id": col,
                "algorithm": algorithm,
                "use_masking": bool(use_masking),
                "n_blocks": int(desc["count"]),
                "mean": float(desc["mean"]),
                "median": float(matrix_df[col].median()),
                "std": float(desc["std"]),
                "min": float(desc["min"]),
                "max": float(desc["max"]),
                "rank_mean": float(mean_ranks[col]),
            }
        )

    return {"performed": True, "per_treatment": treatment_data}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_all_metrics(
    df: pd.DataFrame,
    metrics: list[str],
    alpha: float,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # TODO: Add fail-fast option to exit immediately on first metric error.
    # TODO: Use treatment ordering based on ALGORITHMS + TRAD_ALGORITHMS for Page test.
    # TODO: Compute Wilcoxon CI per pair and confidence curves per pair.
    # TODO: Compute Page trend test when ordering is available.
    # TODO: Only skip tests when their preconditions fail; keep baselines in all other tests.
    # TODO: Add CLI flag for fail_fast in stats so pipeline stops on first error.
    # TODO: Add ordering list in metadata for Page trend (ALGORITHMS + TRAD_ALGORITHMS).
    # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.page_trend_test.html
    errors: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for metric in metrics:
        stage, status, skipped_reason = "", "ok", None
        data_sufficency = check_data_sufficiency(df, metric)

        if not data_sufficency["min_requirements_met"]:
            stage, status, skipped_reason = "data_sufficency", "skipped", "data_insufficient"
            results.append(
                {
                    "metric_name": metric,
                    "stage": stage,
                    "status": status,
                    "skipped_reason": skipped_reason,
                }
            )
            continue

        try:
            stage = "build_repeated_measures_matrix"
            matrix_df = build_repeated_measures_matrix(df, metric)
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append({"metric_name": metric, "stage": stage, "status": status, "skipped_reason": skipped_reason})
            continue

        try:
            stage = "descriptive_stats"
            descriptive_stats = _descriptive_stats(matrix_df)
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append({"metric_name": metric, "stage": stage, "status": status, "skipped_reason": skipped_reason})
            continue

        try:
            stage = "shapiro"
            shapiro = run_shapiro_diagnostics(matrix_df)
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append({"metric_name": metric, "stage": stage, "status": status, "skipped_reason": skipped_reason})
            continue

        try:
            stage = "friedman"
            friedman = run_friedman_test(matrix_df, alpha)
            chi2 = friedman["chi2"]
            n_blocks = friedman["n_blocks_common"]
            k_groups = friedman["k_treatments"]
            friedman_significant = friedman["significant"]
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append({"metric_name": metric, "stage": stage, "status": status, "skipped_reason": skipped_reason})
            continue

        if friedman_significant:
            try:
                stage = "nemenyi"
                conover = run_nemenyi_posthoc(matrix_df, alpha, metric)
            except Exception as e:
                status, skipped_reason = "error", f"{e}"
                errors.append({"metric_name": metric, "stage": stage, "status": status, "skipped_reason": skipped_reason})
                continue
        else:
            skipped_reason = "Friedman not significant"
            conover = {"performed": False, "skipped_reason": skipped_reason}

        try:
            stage = "kendall_w"
            kendall_w = compute_kendall_w(chi2, n_blocks, k_groups)
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append({"metric_name": metric, "stage": stage, "status": status, "skipped_reason": skipped_reason})
            continue

        try:
            stage = "wilcoxon_ci"
            # TODO: Replace placeholder call with per-pair Wilcoxon CI calculation.
            # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
            median_ci = Wilcoxon_np_ci(matrix_df, bootstrap_reps, bootstrap_seed)
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append({"metric_name": metric, "stage": stage, "status": status, "skipped_reason": skipped_reason})
            continue

        try:
            stage = "build_cd_diagram_input"
            cd_diagram_input = build_cd_diagram_input(matrix_df)
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append({"metric_name": metric, "stage": stage, "status": status, "skipped_reason": skipped_reason})
            continue

        metric_result = compile_metric_result(
            metric,
            data_sufficency,
            shapiro,
            friedman,
            kendall_w,
            conover,
            median_ci,
            # TODO: Pass confidence_curves and page_trend once implemented.
            # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.page_trend_test.html
            cd_diagram_input,
            descriptive_stats,
        )
        results.append(metric_result)

    return results, errors


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------

def build_stats_metadata(
    args: argparse.Namespace,
    input_df: pd.DataFrame,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    # TODO: Add treatment order (ALGORITHMS + TRAD_ALGORITHMS) to metadata.
    # TODO: Capture whether Page trend test ran and why it may be skipped.
    n_treatments_input = input_df["treatment_id"].nunique()
    n_blocks_input = input_df["seed"].nunique()
    duplicate_key_rows = int(input_df.duplicated(subset=["split_id", "seed", "treatment_id"]).sum())
    non_finite_primary_metric_rows = int(
        (~np.isfinite(input_df[PRIMARY_METRICS])).any(axis=1).sum()
    )

    run_metadata = build_run_metadata(
        command_args=sys.argv,
        stage="stats",
        manifest_path=Path(args.input),
        additional_info={
            "split_ids": input_df["split_id"].unique().tolist(),
            "n_treatments": n_treatments_input,
            "n_blocks": n_blocks_input,
        },
    )

    global_checks = {
        "schema_valid": True,
        "duplicate_key_rows": duplicate_key_rows,
        "non_finite_primary_metric_rows": non_finite_primary_metric_rows,
        "error_count": len(errors),
    }

    return {"run_metadata": run_metadata, "global_checks": global_checks}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_stats_outputs(
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    errors: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_summary = {
        "stats_summary_version": "1.0.0",
        "run_metadata": metadata["run_metadata"],
        "global_checks": metadata["global_checks"],
        "metrics": results,
        "errors": errors,
    }
    write_json(stats_summary, out_dir / "stats_summary.json")

    all_pairs: list[dict[str, Any]] = []
    for result in results:
        nemenyi = result.get("nemenyi", {})
        if nemenyi.get("performed"):
            all_pairs.extend(nemenyi.get("pairs", []))
    if all_pairs:
        write_csv(pd.DataFrame(all_pairs), out_dir / "pairwise_nemenyi.csv")
    else:
        write_csv(pd.DataFrame([]), out_dir / "pairwise_nemenyi.csv")

    all_ranks: list[dict[str, Any]] = []
    for result in results:
        cd_input = result.get("cd_input", {})
        if cd_input.get("available"):
            for row in cd_input.get("average_ranks", []):
                row_with_metric = dict(row)
                row_with_metric["metric_name"] = result.get("metric_name")
                all_ranks.append(row_with_metric)
    if all_ranks:
        write_csv(pd.DataFrame(all_ranks), out_dir / "cd_diagram_input.csv")
    else:
        write_csv(pd.DataFrame([]), out_dir / "cd_diagram_input.csv")

    write_json(metadata["run_metadata"], out_dir / "stats_meta.json")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    df = load_seed_summary(input_path, required_ids=SEED_SUMMARY_REQUIRED_IDS)

    metrics = args.metrics if args.metrics is not None else ALL_METRICS
    try:
        validate_stats_input_schema(df, metric_cols=metrics)
    except ValueError as e:
        print(f"[SCHEMA ERROR] {e}")
        sys.exit(1)

    results, errors = run_all_metrics(
        df,
        metrics=metrics,
        alpha=args.alpha,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )

    metadata = build_stats_metadata(args, df, errors)
    write_stats_outputs(results, metadata, errors, out_dir)

    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = sum(1 for r in results if r.get("status") == "error")

    print(f"\nDone. Outputs written to {out_dir}/")
    print(f"  Metrics OK      : {ok}")
    print(f"  Metrics skipped : {skipped}")
    print(f"  Metrics errored : {failed}")
    print(f"  Pipeline errors : {len(errors)}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
