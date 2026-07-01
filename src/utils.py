from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import fcntl

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib.ppo_mask import MaskablePPO

from src.a2c_mask import MaskableA2C
from src.dqn_mask import MaskableDQN


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    run_id: str
    treatment_id: str
    algorithm: str
    use_masking: bool
    window_size: int
    tail_size: int
    seed: int | None
    split_id: str
    model_path: str
    trace_file: str
    topology_file: str
    node_file: str


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    treatment_id: str
    algorithm: str
    use_masking: bool
    window_size: int
    tail_size: int
    seed: int | None
    split_id: str
    model_path: str
    trace_file: str
    topology_file: str
    node_file: str
    episode_reward: float
    decision_count: int
    decision_latency_mean_ms: float
    eval_wall_s: float
    max_waiting: float
    avg_waiting: float
    max_slowdown: float
    avg_slowdown: float
    avg_turnaround: float
    cpu_utilization: float
    gpu_utilization: float
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SEED_SUMMARY_REQUIRED_IDS = [
    "split_id", "seed", "algorithm", "use_masking", "treatment_id",
]

MANIFEST_REQUIRED = [
    "run_id", "treatment_id", "algorithm", "use_masking", "seed", "window_size", "tail_size",
    "split_id", "model_path", "trace_file", "topology_file", "node_file",
]

EVAL_REQUIRED = [
    "run_id",  "treatment_id", "algorithm", "use_masking", "window_size", "tail_size","seed", "split_id",
    "episode_reward", "decision_count",
    "max_waiting", "avg_waiting",
    "max_slowdown", "avg_slowdown",
    "avg_turnaround",
    "cpu_utilization", "gpu_utilization",
]

# Metrics that must be finite (non-NaN, non-inf) in every row
CORE_METRICS = [
    "episode_reward",
    "max_waiting", "avg_waiting",
    "max_slowdown", "avg_slowdown",
    "avg_turnaround",
    "cpu_utilization", "gpu_utilization",
    "decision_count",
    "decision_latency_mean_ms",
    "eval_wall_s",
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

ALGORITHMS = {
    "dqn": DQN,
    "a2c": A2C,
    "ppo": PPO,
    "maskable_dqn": MaskableDQN,
    "maskable_a2c": MaskableA2C,
    "maskable_ppo": MaskablePPO,
}

TRAD_ALGORITHMS  = ["fcfs", "lcfs", "sjf", "wfp3", "unicep", "f_1", "f_2"]

# Grouping keys for aggregation
CANON_KEYS = ["run_id", "treatment_id", "algorithm", "use_masking", "seed", "split_id"]
GROUP_KEYS = ["treatment_id", "algorithm", "use_masking", "seed", "split_id"]
ALGO_KEYS = ["treatment_id", "algorithm", "use_masking", "split_id"]

HOLDOUT_PATTERNS = [
    "/holdout",
    "_holdout",
    "holdout30",
    "test_set",
    "_test_",
    "final_test",
]

PARTITION_CONFIGS = {
    "physical": {
        "topology": "data/topology/physical_topology.txt",
        "nodes":    "data/topology/nodes.csv",
    },
    "deeplearn": {
        "topology": "data/topology/deeplearn_topology.txt",
        "nodes":    "data/topology/nodes.csv",
    },
}

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
# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

class ArgumentParserWithDefaults(argparse.ArgumentParser):
    def add_argument(self, *args, help: str | None = None, default: Any | None = None, **kwargs: Any) -> None:
        if help is not None:
            kwargs["help"] = help
        if default is not None and args[0] != "-h":
            kwargs["default"] = default
            if help is not None:
                kwargs["help"] += f" (Default: {default})"
        super().add_argument(*args, **kwargs)


def add_standard_debug_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit immediately on first error (don't continue processing)",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print detailed logging info",
    )
    parser.add_argument(
        "--seed-override",
        type=int,
        default=None,
        help="Override seed from manifest (for testing)",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Limit number of runs processed (for smoke testing)",
    )
    return parser


# ---------------------------------------------------------------------------
# Type helpers and validation
# ---------------------------------------------------------------------------

def parse_bool(value: str) -> bool:
    """
    Parse a string representation of a boolean.
    Ref: reused from evaluate_agents.py for consistency.
    """
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def validate_required_columns(df: pd.DataFrame, required: list[str], context: str = "") -> None:
    missing = [col for col in required if col not in df.columns.tolist()]
    if missing:
        raise ValueError(f"[{context}] Missing required columns: {missing}")


def validate_no_duplicates(df: pd.DataFrame, subset: list[str], context: str = "") -> None:
    duplicates = df.duplicated(subset=subset)
    if duplicates.any():
        dup_count = int(duplicates.sum())
        raise ValueError(f"[{context}] Found {dup_count} duplicate rows on columns {subset}")


def validate_finite_numeric(df: pd.DataFrame, cols: list[str], context: str = "") -> None:
    issues: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"{col}: non_numeric")
            continue
        nan_count = int(df[col].isna().sum())
        if nan_count:
            issues.append(f"{col}: NaN({nan_count})")
        inf_count = int((~np.isfinite(df[col])).sum())
        if inf_count:
            issues.append(f"{col}: inf({inf_count})")
    if issues:
        raise ValueError(f"[{context}] Non-finite numeric values: {', '.join(issues)}")



def load_csv_with_types(
    path: Path,
    dtype_spec: dict[str, str] | None = None,
    bool_cols: list[str] | None = None,
    int_cols: list[str] | None = None,
    float_cols: list[str] | None = None,
    context: str = "",
) -> pd.DataFrame:
    """
    Load CSV with flexible type coercion and nullable integer support.

    Ref: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    """
    if not path.exists():
        raise FileNotFoundError(f"[{context}] CSV not found: {path}")

    if dtype_spec:
        df = pd.read_csv(path, dtype=dtype_spec, skipinitialspace=True)
    else:
        df = pd.read_csv(path, dtype=str, skipinitialspace=True)

    if bool_cols:
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_bool)

    if float_cols:
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    if int_cols:
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype("Int64")

    return df


def validate_loaded_manifest(df: pd.DataFrame, context: str = "manifest") -> None:
    validate_required_columns(df, MANIFEST_REQUIRED, context=context)
    validate_no_duplicates(df, ["run_id"], context=context)


def load_run_manifest(path: Path) -> pd.DataFrame:
    dtype_spec = {col: "str" for col in MANIFEST_REQUIRED}
    df = load_csv_with_types(
        path,
        dtype_spec=dtype_spec,
        int_cols=["seed"],
        context="run_manifest",
    )
    df["use_masking"] = df["use_masking"].apply(parse_bool)
    return df


def load_eval_summary(path: Path) -> pd.DataFrame:
    dtype_map = defaultdict(lambda: "float64")
    dtype_map.update({col: "str" for col in EVAL_REQUIRED if col not in CORE_METRICS})
    df = load_csv_with_types(
        path,
        dtype_spec=dict(dtype_map),
        int_cols=["seed", "decision_count"],
        float_cols=CORE_METRICS,
        context="eval_summary",
    )
    df["use_masking"] = df["use_masking"].apply(parse_bool)
    return df


def load_seed_summary(path: Path, required_ids: list[str]) -> pd.DataFrame:
    dtype_map = defaultdict(lambda: "float64")
    dtype_map.update({col: "str" for col in required_ids})
    df = load_csv_with_types(
        path,
        dtype_spec=dict(dtype_map),
        int_cols=["seed"],
        context="seed_summary",
    )
    df["use_masking"] = df["use_masking"].apply(parse_bool)
    rename_map = {
        f"{col}_mean": col
        for col in ALL_METRICS
        if f"{col}_mean" in df.columns
    }
    df = df.rename(columns=rename_map)
    return df


def validate_not_holdout(trace_file: str, context: str = "", raise_error: bool = True) -> bool:
    trace_lower = trace_file.lower()
    for pattern in HOLDOUT_PATTERNS:
        if pattern in trace_lower:
            msg = f"[{context}] Refuse to use holdout trace: {trace_file}"
            if raise_error:
                raise ValueError(msg)
            print(f"[WARNING] {msg}")
            return False
    return True


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def git_hash() -> str | None:
    """Return the current git commit hash, or None if unavailable.

    Prefers the GIT_COMMIT env var (injected by the Snakefile from a login-node
    capture) so the hash is recorded even when git cannot run inside the
    Apptainer runtime. Falls back to an in-process git call otherwise.
    """
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit:
        return env_commit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build_run_metadata(
    command_args: list[str] | None = None,
    stage: str = "unknown",
    split_id: str | None = None,
    manifest_path: Path | None = None,
    additional_info: dict[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(command_args) if command_args else "",
        "git_commit": git_hash(),
        "stage": stage,
    }
    if split_id:
        metadata["split_id"] = split_id
    if manifest_path and manifest_path.exists():
        metadata["manifest_sha256"] = file_sha256(manifest_path)
    if note:
        metadata["note"] = note
    if additional_info:
        metadata.update(additional_info)
    return metadata


def build_eval_metadata(
    command_args: list[str],
    split_id: str,
    manifest_path: Path,
    eval_stats: dict[str, Any],
    note: str | None = None,
) -> dict[str, Any]:
    return build_run_metadata(
        command_args=command_args,
        stage="evaluate",
        split_id=split_id,
        manifest_path=manifest_path,
        additional_info={"eval_stats": eval_stats},
        note=note,
    )


def build_aggregate_metadata(
    command_args: list[str],
    manifest_path: Path,
    split_ids: list[str],
    qc_stats: dict[str, Any],
    note: str | None = None,
) -> dict[str, Any]:
    return build_run_metadata(
        command_args=command_args,
        stage="aggregate",
        manifest_path=manifest_path,
        additional_info={"split_ids": split_ids, "qc_stats": qc_stats},
        note=note,
    )


def build_train_metadata(
    command_args: list[str],
    split_id: str,
    run_id: str,
    treatment_id: str,
    algorithm: str,
    use_masking: bool,
    window_size: int,
    tail_size: int,
    seed: int | None,
    total_timesteps: int,
    save_interval: int,
    total_saving: int,
    model_dir: Path,
    selector_dir: Path,
    hyperparams: dict[str, Any],
    wall_clock_s: float,
    episodes_completed: int | None,
    note: str | None = None,
) -> dict[str, Any]:
    return build_run_metadata(
        command_args=command_args,
        stage="train",
        split_id=split_id,
        additional_info={
            "run_id": run_id,
            "treatment_id": treatment_id,
            "algorithm": algorithm,
            "use_masking": use_masking,
            "window_size": window_size,
            "tail_size": tail_size,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "save_interval": save_interval,
            "total_saving": total_saving,
            "model_dir": str(model_dir),
            "selector_dir": str(selector_dir),
            "hyperparameters": hyperparams,
            "wall_clock_s": wall_clock_s,
            "episodes_completed": episodes_completed,
        },
        note=note,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Ref: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")


def write_json(obj: dict[str, Any], path: Path) -> None:
    """
    Ref: https://docs.python.org/3/library/json.html
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)


def write_dict_outputs(
    data_dict: dict[str, Any],
    base_name: str,
    output_dir: Path,
    as_json: bool = True,
    as_csv: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if as_json:
        write_json(data_dict, output_dir / f"{base_name}.json")
    if as_csv:
        write_csv(pd.DataFrame([data_dict]), output_dir / f"{base_name}.csv")


def interpret_stat(stat: float, thresholds: list[tuple[float, str]]) -> str:
    interpretation = "negligible"
    for threshold, label in thresholds:
        if stat >= threshold:
            interpretation = label
    return interpretation


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def safe_metric_access(fn: Callable[[], Any], default: Any, context: str = "") -> Any:
    try:
        result = fn()
        return result if result is not None else default
    except Exception as e:
        print(f"[WARNING] {context}: {e}, using default")
        return default


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def resolve_algorithm_config(
    algorithm: str,
    hidden_layer: list[int],
    activation_fn: Callable[..., Any],
) -> tuple[type, dict[str, Any]]:
    algo_class = ALGORITHMS[algorithm.lower()]
    if "dqn" in algorithm.lower():
        net_arch: dict[str, Any] | list[int] = hidden_layer
    else:
        net_arch = {"pi": hidden_layer, "vf": hidden_layer}
    policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}
    return algo_class, policy_kwargs

def load_split_metadata(
    splits_log_dir: str = "data/splits/logs",
    split_id: str | None = None  # NEW PARAM
) -> dict[str, Any]:
    log_files = list(Path(splits_log_dir).glob("*.json"))
    if not log_files:
        raise FileNotFoundError(f"No split log files found in {splits_log_dir}")
    
    # If split_id provided, filter to matching file
    if split_id:
        matching_files = [f for f in log_files if split_id in f.name]
        if not matching_files:
            raise FileNotFoundError(
                f"No split log matching '{split_id}' in {splits_log_dir}. Available: {[f.name for f in log_files]}"
            )
        log_files = matching_files
    
    if len(log_files) > 1:
        raise ValueError(
            f"Multiple split logs found: {[f.name for f in log_files]}. Provide more specific split_id."
        )
    with log_files[0].open("r") as f:
        return json.load(f)

def write_manifest_entry(
    treatment_id: str,
    algorithm: str,
    use_masking: bool,
    seed: int | None,
    window_size: int,
    tail_size: int,
    split_id: str,
    model_path: str,
    trace_file: str,
    topology_file: str,
    node_file: str,
    manifest_path: Path = Path("logs/run_log.csv"),
) -> str:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if manifest_path.exists():
                df = pd.read_csv(manifest_path)
                next_index = len(df[df["algorithm"] == algorithm]) + 1
                file_exists = True
            else:
                next_index = 1
                file_exists = False

            run_id = f"{algorithm}_{next_index:03d}"
            entry = pd.DataFrame(
                [[
                    run_id,
                    treatment_id,
                    algorithm,
                    use_masking,
                    seed,
                    window_size,
                    tail_size,
                    split_id,
                    model_path,
                    trace_file,
                    topology_file,
                    node_file,
                ]],
                columns=MANIFEST_REQUIRED,
            )
            entry.to_csv(manifest_path, mode="a" if file_exists else "w", index=False, header=not file_exists)
            print(f"[LOGGED] {run_id}")
            return run_id
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
