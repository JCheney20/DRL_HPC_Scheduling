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
  - scipy.stats.wilcoxon:          https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
  - scipy.stats.page_trend_test:   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.page_trend_test.html
  - scikit-posthocs nemenyi:       https://scikit-posthocs.readthedocs.io/en/latest/tutorial/scipy.stats.nemenyi.html
  - Hollander & Wolfe (1999), Nonparametric Statistical Methods, 2nd ed., Sec 3.3
    (classical Wilcoxon signed-rank CI procedure that Carrasco et al. 2020 builds on)
  - numpy random Generator:        https://numpy.org/doc/stable/reference/random/generator.html
  - pandas pivot_table:            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html
  - argparse:                      https://docs.python.org/3/library/argparse.html
  - json:                          https://docs.python.org/3/library/json.html
"""

from __future__ import annotations

import argparse
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

from utils import (
    ALGORITHMS,
    TRAD_ALGORITHMS,
    SEED_SUMMARY_REQUIRED_IDS,
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


PRIMARY_METRICS = [
    "avg_waiting",
    "avg_slowdown",
]

ALL_METRICS = [
    "avg_waiting",
    "avg_slowdown",
    "max_waiting",
    "max_slowdown",
    "avg_turnaround",
    "cpu_utilization",
    "gpu_utilization",
    "episode_reward",
    "decision_latency_mean_ms",
    "eval_wall_s",
]


METRIC_DIRECTION: dict[str, str] = {
    "avg_waiting": "lower_is_better",
    "avg_slowdown": "lower_is_better",
    "max_waiting": "lower_is_better",
    "max_slowdown": "lower_is_better",
    "avg_turnaround": "lower_is_better",
    "cpu_utilization": "higher_is_better",
    "gpu_utilization": "higher_is_better",
    "episode_reward": "higher_is_better",
    "decision_latency_mean_ms": "lower_is_better",
    "eval_wall_s": "lower_is_better",
}

KENDALL_W_THRESHOLDS = [
    (0.0, "Slight Agreement"),
    (0.2, "Fair Agreement"),
    (0.4, "Moderate Agreement"),
    (0.6, "Substantial Agreement"),
    (0.7, "Almost Perfect Agreement"),
]

MIN_ALGORITHMS = 3
MIN_BLOCKS = 2

DECLARED_TREATMENT_ORDER: list[str] = list(ALGORITHMS.keys()) + TRAD_ALGORITHMS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-parametric statistical tests on seed-level RL evaluation results."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to seed_summary.csv from aggregate_results.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/stats",
        type=str,
        help="Directory to write statistical outputs.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        nargs="+",
        type=str,
        help="Metrics to test. Defaults to ALL_METRICS.",
    )
    parser.add_argument(
        "--alpha",
        default=0.05,
        type=float,
        help="Significance level for hypothesis tests.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------


def validate_stats_input_schema(df: pd.DataFrame, metric_cols: list[str]) -> None:
    validate_required_columns(df, SEED_SUMMARY_REQUIRED_IDS, context="seed_summary")
    validate_required_columns(df, metric_cols, context="seed_summary_metrics")
    validate_finite_numeric(df, PRIMARY_METRICS, context="seed_summary_primary_metrics")

    invalid_algos = [
        algo for algo in df["algorithm"] if algo.split("__mask_")[0] not in (list(ALGORITHMS.keys()) + TRAD_ALGORITHMS)
    ]
    if invalid_algos:
        raise ValueError(f"Invalid algorithms in seed summary {invalid_algos}")

    validate_no_duplicates(
        df, ["split_id", "seed", "treatment_id"], context="seed_summary"
    )


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
        "min_requirements_met": n_treatments >= min_algorithms
        and n_blocks_common >= min_blocks,
    }


def build_repeated_measures_matrix(
    df: pd.DataFrame,
    metric: str,
    block_col: str = "seed",
    group_col: str = "treatment_id",
) -> pd.DataFrame:
    return df.pivot_table(index=block_col, columns=group_col, values=metric).dropna(
        how="any"
    )


# ---------------------------------------------------------------------------
# Shapiro-Wilk diagnostics
# ---------------------------------------------------------------------------


def _shapiro_one(treatment_id: str, values: np.ndarray) -> dict[str, Any]:
    n = len(values)
    if n < 3:
        return {
            "treatment_id": treatment_id,
            "n": n,
            "W": None,
            "p_value": None,
            "valid": False,
            "note": "n < 3",
        }
    W, p_value = stats.shapiro(values)
    return {
        "treatment_id": treatment_id,
        "n": n,
        "W": float(W),
        "p_value": float(p_value),
        "valid": True,
        "note": "",
    }


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
    W = (
        float(chi2 / (n_blocks * (k_groups - 1)))
        if (n_blocks * (k_groups - 1)) != 0
        else 0.0
    )
    return {
        "measure": "Kendall W",
        "value": W,
        "interpretation": interpret_stat(W, KENDALL_W_THRESHOLDS),
    }


# ---------------------------------------------------------------------------
# Nemenyi post-hoc
# ---------------------------------------------------------------------------


def run_nemenyi_posthoc(
    matrix_df: pd.DataFrame,
    alpha: float,
    metric_name: str,
) -> dict[str, Any]:
    p_matrix = sp.posthoc_nemenyi_friedman(matrix_df)
    asc = METRIC_DIRECTION[metric_name] == "lower_is_better"
    mean_ranks = pd.Series(matrix_df.rank(axis=1, ascending=asc).mean(axis=0))
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




@lru_cache(maxsize=None)
def _wilcoxon_signed_rank_cumulative_dist(n: int) -> tuple[float, ...]:
    """
    Exact null CDF of W+ (sum of positive signed ranks) for n untied, nonzero
    observations, via DP: counts[k] after rank r = counts[k] (rank r negative)
    + counts[k-r] (rank r positive). Depends only on n, cached accordingly.
    """
    max_w = n * (n + 1) // 2
    dist = np.zeros(max_w + 1)
    dist[0] = 1.0
    for r in range(1, n + 1):
        new_dist = dist.copy()
        new_dist[r:] += dist[:-r]
        dist = new_dist
    cumulative = np.cumsum(dist / (2 ** n))
    return tuple(cumulative)


def _exact_critical_value(n: int, alpha_half: float) -> int:
    """Largest integer c such that P(W+ <= c) <= alpha_half under the exact null distribution."""
    cumulative = _wilcoxon_signed_rank_cumulative_dist(n)
    c_alpha = -1
    for w, p in enumerate(cumulative):
        if p <= alpha_half:
            c_alpha = w
        else:
            break
    return c_alpha


def Wilcoxon_np_ci(
    values_a: pd.Series, values_b: pd.Series, alpha: float
) -> dict[str, Any]:
    """
    Non-parametric (Hodges-Lehmann / Walsh-average) confidence interval for
    the median of paired differences. See module-level note above for the
    bug this replaces and its empirical verification.

    # Ref: Hollander & Wolfe (1999), Nonparametric Statistical Methods, Sec 3.3
    # Ref: Carrasco et al. (2020) -- non-parametric CI procedure this extends
    # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    """
    assert len(values_a) == len(values_b), "Paired inputs must be the same length"
    d = (values_a - values_b).to_numpy()
    l = len(d)

    i, j = np.tril_indices(l)
    walsh_sorted = np.sort((d[i] + d[j]) / 2)
    n_walsh = len(walsh_sorted)

    alpha_half = alpha / 2

    if l <= 20:
        c_alpha = _exact_critical_value(l, alpha_half)
        K = c_alpha + 1
        method = "exact"
        achieved_alpha_half = (
            float(_wilcoxon_signed_rank_cumulative_dist(l)[c_alpha]) if c_alpha >= 0 else None
        )
    else:
        z_crit = stats.norm.ppf(1 - alpha_half)  
        K = math.ceil((l * (l + 1) / 4) - z_crit * math.sqrt(l * (l + 1) * (2 * l + 1) / 24))
        method = "asymptotic"
        achieved_alpha_half = alpha_half

    if K < 1 or K > n_walsh:
        return {
            "ci_low": None,
            "ci_high": None,
            "alpha": alpha,
            "achieved_alpha": None,
            "method": method,
            "n_pairs": l,
            "note": f"No valid critical value for n={l} at alpha={alpha}; CI not achievable at this sample size.",
        }

    return {
        "ci_low": float(walsh_sorted[K - 1]),
        "ci_high": float(walsh_sorted[n_walsh - K]),
        "alpha": alpha,
        "achieved_alpha": float(2 * achieved_alpha_half) if achieved_alpha_half is not None else None,
        "method": method,
        "n_pairs": l,
    }


def compute_confidence_curves(
    values_a: pd.Series, values_b: pd.Series, grid: pd.Series, metric: str
) -> pd.DataFrame:
    treatment_a = values_a.name
    treatment_b = values_b.name
    d = values_a - values_b
    l = len(d)
    method = "exact" if l <= 20 else "asymptotic"

    rows = []
    for delta in grid:
        result = stats.wilcoxon(d - delta, method=method)
        rows.append(
            {
                "metric": metric,
                "treatment_a": treatment_a,
                "treatment_b": treatment_b,
                "delta": float(delta),
                "p_value": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page trend test
# ---------------------------------------------------------------------------
#


def run_page_trend_test(
    matrix_df: pd.DataFrame,
    declared_order: list[str],
    alpha: float,
    trend_direction: str,
) -> dict[str, Any]:
    """
    :param trend_direction: derived from METRIC_DIRECTION for this metric, not
        a free choice -- see the call site in run_all_metrics(). Rule:
        "higher_is_better" metrics -> "increasing" (last algorithm in
        declared_order predicted to have the highest mean); "lower_is_better"
        metrics -> "decreasing" (last algorithm predicted to have the lowest,
        i.e. best, mean). This makes the predicted trend along
        ALGORITHMS + TRAD_ALGORITHMS consistently mean "later in the sequence
        is better," regardless of which way a given metric's raw values run.

        scipy's page_trend_test defaults to "increasing" implicitly via
        column order; this parameter makes the direction explicit via
        predicted_ranks instead, since reversing column order would also
        (confusingly) reverse the reported treatment_order.
    """
    order_index = {algo: i for i, algo in enumerate(declared_order)}

    def _algo_prefix(treatment_id: str) -> str:
        return treatment_id.split("__mask_")[0]

    ordered_treatments = sorted(
        (col for col in matrix_df.columns if _algo_prefix(col) in order_index),
        key=lambda col: order_index[_algo_prefix(col)],
    )
    if len(ordered_treatments) < 3:
        return {
            "performed": False,
            "skipped_reason": "fewer than 3 ordered treatments present",
        }
    ordered = matrix_df[ordered_treatments]
    n = len(ordered_treatments)
    predicted_ranks = (
        list(range(1, n + 1)) if trend_direction == "increasing" else list(range(n, 0, -1))
    )
    result = stats.page_trend_test(ordered.to_numpy(), predicted_ranks=predicted_ranks)
    return {
        "performed": True,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "treatment_order": ordered_treatments,
        "trend_direction": trend_direction,
    }


# ---------------------------------------------------------------------------
# CD diagram input
# ---------------------------------------------------------------------------


def build_cd_diagram_input(matrix_df: pd.DataFrame, metric: str) -> dict[str, Any]:
    asc = METRIC_DIRECTION[metric] == "lower_is_better"
    avg_ranks = pd.Series(matrix_df.rank(axis=1, ascending=asc).mean(axis=0))
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


def _descriptive_stats(matrix_df: pd.DataFrame, metric: str) -> dict[str, Any]:
    treatment_data = []
    asc = METRIC_DIRECTION[metric] == "lower_is_better"
    mean_ranks = pd.Series(matrix_df.rank(axis=1, ascending=asc).mean(axis=0))

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
    declared_order: list[str] = DECLARED_TREATMENT_ORDER,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for metric in metrics:
        stage, status, skipped_reason = "", "ok", None
        data_sufficency = check_data_sufficiency(df, metric)

        if not data_sufficency["min_requirements_met"]:
            stage, status, skipped_reason = (
                "data_sufficency",
                "skipped",
                "data_insufficient",
            )
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
            errors.append(
                {
                    "metric_name": metric,
                    "stage": stage,
                    "status": status,
                    "skipped_reason": skipped_reason,
                }
            )
            continue

        try:
            stage = "descriptive_stats"
            descriptive_stats = _descriptive_stats(matrix_df, metric)
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append(
                {
                    "metric_name": metric,
                    "stage": stage,
                    "status": status,
                    "skipped_reason": skipped_reason,
                }
            )
            continue

        try:
            stage = "shapiro"
            shapiro = run_shapiro_diagnostics(matrix_df)
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append(
                {
                    "metric_name": metric,
                    "stage": stage,
                    "status": status,
                    "skipped_reason": skipped_reason,
                }
            )
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
            errors.append(
                {
                    "metric_name": metric,
                    "stage": stage,
                    "status": status,
                    "skipped_reason": skipped_reason,
                }
            )
            continue

        if friedman_significant:
            try:
                stage = "nemenyi"
                nemenyi = run_nemenyi_posthoc(matrix_df, alpha, metric)
            except Exception as e:
                status, skipped_reason = "error", f"{e}"
                errors.append(
                    {
                        "metric_name": metric,
                        "stage": stage,
                        "status": status,
                        "skipped_reason": skipped_reason,
                    }
                )
                continue
        else:
            skipped_reason = "Friedman not significant"
            nemenyi = {"performed": False, "skipped_reason": skipped_reason}

        try:
            stage = "confidence_curve"
            confidence_curves_list = []
            if nemenyi.get("performed"):
                for pair in nemenyi.get("pairs", []):
                    if pair["significant"]:
                        a, b = pair["treatment_a"], pair["treatment_b"]
                        d = matrix_df[a].to_numpy() - matrix_df[b].to_numpy()
                        grid = np.linspace(
                            np.percentile(d, 1), np.percentile(d, 99), 200
                        )
                        curve_df = compute_confidence_curves(
                            matrix_df[a], matrix_df[b], grid, metric
                        )
                        confidence_curves_list.append(curve_df)

                if confidence_curves_list:
                    confidence_curve = {
                        "performed": True,
                        "curves": pd.concat(confidence_curves_list).to_dict("records"),
                    }
                else:
                    confidence_curve = {
                        "performed": False,
                        "skipped_reason": "no significant Nemenyi pairs",
                    }
            else:
                skipped_reason = "Nemenyi not performed"
                confidence_curve = {
                    "performed": False,
                    "skipped_reason": skipped_reason,
                }
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append(
                {
                    "metric_name": metric,
                    "stage": stage,
                    "status": status,
                    "skipped_reason": skipped_reason,
                }
            )
            continue

        try:
            stage = "kendall_w"
            kendall_w = compute_kendall_w(chi2, n_blocks, k_groups)
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append(
                {
                    "metric_name": metric,
                    "stage": stage,
                    "status": status,
                    "skipped_reason": skipped_reason,
                }
            )
            continue

        try:
            stage = "wilcoxon_ci"
            wilcoxon_pairs = []
            pairs = [
                (a, b)
                for i, a in enumerate(matrix_df.columns)
                for j, b in enumerate(matrix_df.columns)
                if i < j
            ]
            for a, b in pairs:
                result = Wilcoxon_np_ci(matrix_df[a], matrix_df[b], alpha)
                wilcoxon_pairs.append({"treatment_a": a, "treatment_b": b, **result})
            wilcoxon = {"performed": True, "pairs": wilcoxon_pairs}
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append(
                {
                    "metric_name": metric,
                    "stage": stage,
                    "status": status,
                    "skipped_reason": skipped_reason,
                }
            )
            continue

        try:
            stage = "page_trend"
            trend_direction = (
                "increasing" if METRIC_DIRECTION[metric] == "higher_is_better" else "decreasing"
            )
            page_trend = run_page_trend_test(matrix_df, declared_order, alpha, trend_direction)

        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append(
                {
                    "metric_name": metric,
                    "stage": stage,
                    "status": status,
                    "skipped_reason": skipped_reason,
                }
            )
            continue

        try:
            stage = "build_cd_diagram_input"
            cd_diagram_input = build_cd_diagram_input(matrix_df, metric)
        except Exception as e:
            status, skipped_reason = "error", f"{e}"
            errors.append(
                {
                    "metric_name": metric,
                    "stage": stage,
                    "status": status,
                    "skipped_reason": skipped_reason,
                }
            )
            continue

        metric_result = compile_metric_result(
            metric,
            data_sufficency,
            shapiro,
            friedman,
            kendall_w,
            nemenyi,
            wilcoxon,
            confidence_curve,
            page_trend,
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
    n_treatments_input = input_df["treatment_id"].nunique()
    n_blocks_input = input_df["seed"].nunique()
    duplicate_key_rows = int(
        input_df.duplicated(subset=["split_id", "seed", "treatment_id"]).sum()
    )
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
            "treatment_order": DECLARED_TREATMENT_ORDER,
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

    metadata["global_checks"]["page_trend_ran"] = any(
        r.get("page_trend", {}).get("performed") for r in results
    )

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
    write_csv(
        pd.DataFrame(all_pairs) if all_pairs else pd.DataFrame([]),
        out_dir / "pairwise_nemenyi.csv",
    )

    all_wilcoxon: list[dict[str, Any]] = []
    for result in results:
        wilcoxon_ci = result.get("wilcoxon_ci", {})
        if wilcoxon_ci.get("performed"):
            for row in wilcoxon_ci.get("pairs", []):
                all_wilcoxon.append({"metric_name": result.get("metric_name"), **row})
    write_csv(
        pd.DataFrame(all_wilcoxon) if all_wilcoxon else pd.DataFrame([]),
        out_dir / "confidence_intervals.csv",
    )

    all_ranks: list[dict[str, Any]] = []
    for result in results:
        cd_input = result.get("cd_input", {})
        if cd_input.get("available"):
            for row in cd_input.get("average_ranks", []):
                all_ranks.append(
                    {"metric_name": result.get("metric_name"), **dict(row)}
                )
    write_csv(
        pd.DataFrame(all_ranks) if all_ranks else pd.DataFrame([]),
        out_dir / "cd_diagram_input.csv",
    )

    all_page: list[dict[str, Any]] = []
    for result in results:
        pt = result.get("page_trend", {})
        if pt.get("performed"):
            all_page.append({"metric_name": result.get("metric_name"), **pt})
    write_csv(
        pd.DataFrame(all_page) if all_page else pd.DataFrame([]),
        out_dir / "page_trend.csv",
    )

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
