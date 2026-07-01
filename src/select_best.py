"""select_best.py

Select the best-performing algorithm via Pareto dominance over primary metrics,
filtered by statistical indistinguishability (Nemenyi), with pre-declared
tie-breaking using Wilcoxon non-parametric CIs.

Pipeline:
  1. Pareto front over PRIMARY_METRICS (paretoset)              -> see step [1]
  2. Statistical filter: collapse Pareto candidates that are NOT
     significantly different on ALL primaries into equivalence
     classes (networkx connected components)                    -> see step [2]
  3. If >1 candidate remains, break ties using TIE_BREAKERS in
     declared order, using Wilcoxon CIs for significance          -> see step [3]
  4. Write best_algorithm.json with full rationale.

References:
  - paretoset: https://github.com/tommyod/paretoset
    (vectorized Pareto front; sense=['min'|'max', ...] mirrors METRIC_DIRECTION)
  - networkx connected_components: https://networkx.org/documentation/stable/reference/algorithms/component.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
from paretoset import paretoset

from src.utils import METRIC_DIRECTION, write_json

PRIMARY_METRICS = ["avg_waiting", "avg_slowdown"]
SECONDARY_METRICS = ["max_waiting", "max_slowdown", "avg_turnaround", "cpu_utilization"]
RESOURCE_METRICS = ["decision_latency_mean_ms", "eval_wall_s"]
TIE_BREAKERS = ["avg_waiting", "avg_slowdown", "cpu_utilization"]  # pre-declared order
ALPHA = 0.05


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Pareto-optimal algorithm with statistical filtering",
    )
    parser.add_argument("--nemenyi", required=True, help="Path to pairwise_nemenyi.csv")
    parser.add_argument("--seed-summary", required=True, help="Path to seed_summary.csv")
    parser.add_argument("--ci", default=None, help="Path to confidence_intervals.csv (Wilcoxon CIs)")
    parser.add_argument(
        "--page-trend", default=None,
        help="Path to page_trend.csv (validity check only, NOT used for selection)",
    )
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    return parser.parse_args()


def find_pareto_front(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    sense = ["min" if METRIC_DIRECTION.get(m, "lower_is_better") == "lower_is_better" else "max"
             for m in metrics]
    mask = paretoset(df[metrics], sense=sense)
    return df.loc[mask].reset_index(drop=True)

def build_indistinguishable_edges(nemenyi_df: pd.DataFrame, metric: str, alpha: float) -> set[tuple[str, str]]:
    """Edges = pairs NOT significantly different (p_value >= alpha) on `metric`."""
    subset = nemenyi_df[nemenyi_df["metric_name"] == metric]
    tied = subset[subset["p_value"] >= alpha]
    return set(zip(tied["treatment_a"], tied["treatment_b"]))


def filter_statistical_pareto(
    pareto_df: pd.DataFrame, nemenyi_df: pd.DataFrame, primary_metrics: list[str], alpha: float
) -> tuple[list[list[str]], dict]:
    treatments = pareto_df["treatment_id"].astype(str).tolist()

    per_metric_edges = [build_indistinguishable_edges(nemenyi_df, m, alpha) for m in primary_metrics]

    intersection = set.intersection(*per_metric_edges) if per_metric_edges else set()
    intersection = {(a, b) for a, b in intersection if a in treatments and b in treatments}

    graph = nx.Graph()
    graph.add_nodes_from(treatments)
    graph.add_edges_from(intersection)
    classes = [sorted(c) for c in nx.connected_components(graph)]

    rationale = {
        "pareto_count": len(treatments),
        "indistinguishable_pairs_across_primaries": [list(e) for e in intersection],
        "equivalence_classes": classes,
    }
    return classes, rationale

def ci_lookup(ci_df: pd.DataFrame, metric: str, a: str, b: str) -> dict | None:
    if ci_df is None or ci_df.empty:
        return None
    subset = ci_df[ci_df["metric_name"] == metric]
    row = subset[(subset["treatment_a"] == a) & (subset["treatment_b"] == b)]
    if not row.empty:
        r = row.iloc[0]
        if pd.isna(r["ci_low"]) or pd.isna(r["ci_high"]):
            return None
        return {"ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])}
    row = subset[(subset["treatment_a"] == b) & (subset["treatment_b"] == a)]
    if not row.empty:
        r = row.iloc[0]
        if pd.isna(r["ci_low"]) or pd.isna(r["ci_high"]):
            return None
        return {"ci_low": -float(r["ci_high"]), "ci_high": -float(r["ci_low"])}
    return None


def is_significantly_better(ci_df: pd.DataFrame, metric: str, a: str, b: str, direction: str) -> bool | None:
    """
    True if `a` is significantly better than `b` on `metric` (CI for a-b excludes 0
    in the favourable direction). None if no CI is available (treat as inconclusive).
    """
    ci = ci_lookup(ci_df, metric, a, b)
    if ci is None:
        return None
    if direction == "lower_is_better":
        return ci["ci_high"] < 0  # a - b entirely negative => a is lower => better
    return ci["ci_low"] > 0      # a - b entirely positive => a is higher => better


def break_ties_with_cis(
    candidates: list[str], seed_summary: pd.DataFrame, ci_df: pd.DataFrame, tie_breakers: list[str]
) -> tuple[str, dict]:
    rationale = {"initial_candidates": candidates.copy(), "steps": []}
    means = seed_summary.groupby("treatment_id").mean(numeric_only=True)
    current = candidates.copy()

    for metric in tie_breakers:
        if len(current) == 1:
            break
        if metric not in means.columns:
            rationale["steps"].append({"metric": metric, "skipped": "metric_missing"})
            continue

        direction = METRIC_DIRECTION.get(metric, "lower_is_better")
        cand_means = means.loc[current, metric]
        best_val = cand_means.min() if direction == "lower_is_better" else cand_means.max()
        leaders = cand_means[cand_means == best_val].index.tolist()

        # A leader "wins" this metric outright if it's significantly better than
        # every other current candidate (per CI). Check every leader, in case of
        # an exact mean tie among more than one.
        winner = None
        evidence = []
        for leader in leaders:
            beats_all = True
            for other in current:
                if other == leader:
                    continue
                result = is_significantly_better(ci_df, metric, leader, other, direction)
                evidence.append({"leader": leader, "other": other, "significant": result})
                if not result:  # None (no CI) or False both fail to clinch it
                    beats_all = False
            if beats_all:
                winner = leader
                break

        rationale["steps"].append({"metric": metric, "leaders_by_mean": leaders, "evidence": evidence})

        if winner is not None:
            return winner, rationale

        # No decisive winner: narrow to candidates not shown inferior to any leader.
        survivors = set(current)
        for leader in leaders:
            for other in list(survivors):
                if other in leaders:
                    continue
                result = is_significantly_better(ci_df, metric, leader, other, direction)
                if result:  # leader provably beats `other` -> drop `other`
                    survivors.discard(other)
        current = sorted(survivors) if survivors else leaders

    fallback = sorted(current)[0]
    rationale["final_fallback"] = {"chosen": fallback, "remaining": current}
    return fallback, rationale


# ---------------------------------------------------------------------------
# Page trend (validity check only — not used for selection)
# ---------------------------------------------------------------------------


def load_page_trend(page_trend_path: Path) -> dict | None:
    if not page_trend_path.exists():
        return None
    
    try:
        df = pd.read_csv(page_trend_path)
    except pd.errors.EmptyDataError:
        return None  # Treat an empty file the same as a missing/empty dataframe

    if df.empty:
        return None
        
    rows = df[["metric_name", "p_value"]].assign(significant=lambda d: d["p_value"] < ALPHA)
    return {"available": True, "rows": rows.to_dict("records"), "any_significant": bool(rows["significant"].any())}


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    alpha = float(args.alpha)

    try:
        nemenyi = pd.read_csv(args.nemenyi)
    except pd.errors.EmptyDataError:
        nemenyi = pd.DataFrame(columns=["metric_name", "p_value", "treatment_a", "treatment_b"])

    ci_df = None
    if args.ci and Path(args.ci).exists():
        try:
            ci_df = pd.read_csv(args.ci)
        except pd.errors.EmptyDataError:
            ci_df = pd.DataFrame()
    elif args.ci:
        print(f"[WARN] CI path {args.ci} not found; tie-breaking will be less powerful", file=sys.stderr)

    seed_summary = pd.read_csv(args.seed_summary)

    rename_map = {col: col[:-5] for col in seed_summary.columns if col.endswith("_mean")}
    seed_summary = seed_summary.rename(columns=rename_map)

    page_trend_result = load_page_trend(Path(args.page_trend)) if args.page_trend else None

    missing = [m for m in PRIMARY_METRICS if m not in seed_summary.columns]
    if missing:
        print(f"[ERROR] seed_summary missing primary metric columns: {missing}", file=sys.stderr)
        sys.exit(1)

    grouped_means = seed_summary.groupby("treatment_id").mean(numeric_only=True)
    algo_means = grouped_means[PRIMARY_METRICS].reset_index()

    pareto_df = find_pareto_front(algo_means, PRIMARY_METRICS)              # [1]
    classes, stats_rationale = filter_statistical_pareto(                   # [2]
        pareto_df, nemenyi, PRIMARY_METRICS, alpha
    )
    final_candidates = sorted({t for cls in classes for t in cls})

    overall_rationale = {
        "pareto_front_before": pareto_df["treatment_id"].astype(str).tolist(),
        "pareto_classes_after_stat_filter": classes,
        "stats_rationale": stats_rationale,
        "page_trend": page_trend_result,
    }

    if not final_candidates:
        direction = METRIC_DIRECTION.get(PRIMARY_METRICS[0], "lower_is_better")
        col = algo_means.set_index("treatment_id")[PRIMARY_METRICS[0]]
        final_candidates = [col.idxmin() if direction == "lower_is_better" else col.idxmax()]
        print("[WARN] No Pareto candidates found; fell back to best single metric", file=sys.stderr)

    if len(final_candidates) == 1:
        winner, rationale = final_candidates[0], {"method": "single_pareto_candidate"}
    else:
        winner, tie_rationale = break_ties_with_cis(final_candidates, seed_summary, ci_df, TIE_BREAKERS)  # [3]
        rationale = {"method": "tie_breaker_with_cis", "details": tie_rationale}

    tie_metrics = {
        m: float(grouped_means.loc[winner, m]) if m in grouped_means.columns and winner in grouped_means.index else None
        for m in TIE_BREAKERS
    }

    best_algorithm = {
        "treatment_id": winner,
        "selection_rationale": {"overall": overall_rationale, "final_rationale": rationale},
        "tie_break_metrics": tie_metrics,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_algorithm.json"
    write_json(best_algorithm, out_path)
    print(f"[OK] winner={winner} -> wrote {out_path}")


if __name__ == "__main__":
    main()
