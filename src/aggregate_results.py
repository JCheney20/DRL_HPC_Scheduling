"""aggregate_results.py
Aggregates evaluation outputs from evaluate_agents.py into:
  - eval_wide.csv          : per-run wide-format table
  - seed_summary.csv       : per-seed aggregation (mean/std per metric)
  - algorithm_summary.csv  : algorithm-level aggregation across seeds
  - aggregate_metadata.json: reproducibility sidecar

References:
  - pandas groupby: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
  - pandas concat:  https://pandas.pydata.org/docs/reference/api/pandas.concat.html
  - argparse:       https://docs.python.org/3/library/argparse.html
  - pathlib:        https://docs.python.org/3/library/pathlib.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from utils import (
    ALGO_KEYS,
    CANON_KEYS,
    CORE_METRICS,
    EVAL_REQUIRED,
    GROUP_KEYS,
    build_aggregate_metadata,
    load_eval_summary,
    load_run_manifest,
    validate_finite_numeric,
    validate_loaded_manifest,
    validate_required_columns,
    write_csv,
    write_json,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate RL evaluation outputs.")
    parser.add_argument("--manifest", required=True, type=str, help="Path to run manifest CSV.")
    parser.add_argument(
        "--eval-root",
        default="result/eval_runs/runs",
        type=str,
        help="Directory containing per-run eval CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/aggregate",
        type=str,
        help="Directory to write aggregated outputs.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any expected eval file is missing.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Eval artifact discovery
# ---------------------------------------------------------------------------

def discover_eval_artifacts(
    manifest_df: pd.DataFrame,
    eval_root: Path,
    strict: bool,
) -> dict[str, Path]:
    eval_paths: dict[str, Path] = {}
    for run_id in manifest_df["run_id"]:
        path = eval_root / f"{run_id}_metrics.csv"
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"{path} not found")
            print(f"[WARNING] {path} not found, skipping")
            continue
        eval_paths[run_id] = path
    return eval_paths


# ---------------------------------------------------------------------------
# Metadata attachment
# ---------------------------------------------------------------------------

def attach_manifest_metadata(eval_df: pd.DataFrame, manifest_row: pd.Series) -> pd.DataFrame:
    return eval_df.assign(**{col: manifest_row[col] for col in CANON_KEYS})


# ---------------------------------------------------------------------------
# Wide table construction
# ---------------------------------------------------------------------------

def build_eval_wide(eval_frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(eval_frames, ignore_index=True)


def check_key_uniqueness(df: pd.DataFrame, key_cols: list[str], context: str) -> None:
    duplicates = df[df.duplicated(subset=key_cols, keep=False)]
    if not duplicates.empty:
        raise ValueError(f"[{context}] Duplicate rows found on keys {key_cols}:\n{duplicates}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_seed_summary(eval_wide: pd.DataFrame) -> pd.DataFrame:
    df = eval_wide[GROUP_KEYS + CORE_METRICS]
    df["treatment_id"] = df["algorithm"].astype(str) + "__mask_" + df["use_masking"].astype(str)
    # TODO: treatment_id must match stats expectations exactly: "{algorithm}__mask_{use_masking}".
    # TODO: If use_masking is bool, ensure lower-case "true/false" for stable ordering.
    # Ref: https://pandas.pydata.org/docs/reference/api/pandas.Series.astype.html
    df = df.groupby(GROUP_KEYS).agg(["mean", "std"])
    df.columns = ["_".join(col) for col in df.columns]
    return df.reset_index()


def aggregate_algorithm_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    mean_cols = [col for col in seed_summary.columns if col.endswith("_mean")]
    df = seed_summary[ALGO_KEYS + mean_cols]
    df = df.groupby(ALGO_KEYS).agg(["mean", "std"])
    df.columns = ["_".join(col) for col in df.columns]
    return df.reset_index()


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def compute_aggregation_qc(
    eval_wide: pd.DataFrame,
    seed_summary: pd.DataFrame,
    algo_summary: pd.DataFrame,
) -> dict[str, Any]:
    # TODO: Add average train-time per model per trace using train metadata sidecars.
    # TODO: This should summarize wall_clock_s across runs for physical vs deeplearn traces.
    return {
        "eval_wide_count": eval_wide.shape[0],
        "seed_summary_count": seed_summary.shape[0],
        "algo_summary_count": algo_summary.shape[0],
        "eval_wide_nan_count": eval_wide[CORE_METRICS].isnull().sum().to_dict(),
        "run_duplicates_count": eval_wide[eval_wide.duplicated(subset=["run_id"], keep=False)].shape[0],
        "eval_wide_stats": eval_wide[CORE_METRICS].describe().to_dict(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = load_run_manifest(manifest_path)
    validate_loaded_manifest(manifest_df, context="run_manifest")

    artifact_paths = discover_eval_artifacts(manifest_df, eval_root, strict=args.strict)

    eval_frames: list[pd.DataFrame] = []
    failures: list[str] = []

    required_cols = list(dict.fromkeys(EVAL_REQUIRED + CORE_METRICS))

    for run_id, csv_path in artifact_paths.items():
        try:
            df = load_eval_summary(csv_path)
            validate_required_columns(df, required_cols, context=f"eval[{run_id}]")
            validate_finite_numeric(df, CORE_METRICS, context=f"eval[{run_id}]")
            manifest_row = manifest_df[manifest_df["run_id"] == run_id].iloc[0]
            df = attach_manifest_metadata(df, manifest_row)
            eval_frames.append(df)
        except Exception as e:
            failures.append(f"{run_id}: {e}")
            print(f"[FAIL] {run_id} :: {e}")

    if failures and args.strict:
        print(f"\n{len(failures)} run(s) failed validation. Exiting.")
        sys.exit(1)

    if not eval_frames:
        print("No valid eval frames to aggregate. Exiting.")
        sys.exit(1)

    eval_wide = build_eval_wide(eval_frames)
    check_key_uniqueness(eval_wide, key_cols=["run_id"], context="eval_wide")

    seed_summary = aggregate_seed_summary(eval_wide)
    algorithm_summary = aggregate_algorithm_summary(seed_summary)

    qc_stats = compute_aggregation_qc(eval_wide, seed_summary, algorithm_summary)
    metadata = build_aggregate_metadata(
        command_args=sys.argv,
        manifest_path=manifest_path,
        split_ids=manifest_df["split_id"].unique().tolist(),
        qc_stats=qc_stats,
    )
    write_json(metadata, output_dir / "aggregate_metadata.json")

    write_csv(eval_wide, output_dir / "eval_wide.csv")
    write_csv(seed_summary, output_dir / "seed_summary.csv")
    write_csv(algorithm_summary, output_dir / "algorithm_summary.csv")


if __name__ == "__main__":
    main()
