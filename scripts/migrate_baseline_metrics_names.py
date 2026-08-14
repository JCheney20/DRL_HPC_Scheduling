"""migrate_baseline_metrics_names.py

One-time migration for baseline result directories written before
run_baseline.py switched to deterministic metrics filenames.

WHY
---
run_baseline.py used to name its outputs "{run_id}_metrics.csv". run_id is
minted fresh by write_manifest_entry on every invocation, and the Snakefile's
baseline rules pass --force, so re-running one random-control seed wrote a
SECOND file ("random_016_metrics.csv", then "random_021_metrics.csv") rather
than replacing the first. baseline_aggregate.py globs *_metrics.csv, loaded
both, and died on its own (treatment_id, split_id, seed) duplicate check.

run_baseline.py now names files by that same key (see metrics_stem), so
re-runs overwrite and the problem cannot recur. But files already on disk keep
their old names, and the glob still finds them. This script renames the
surviving measurement for each key to its deterministic name and removes the
superseded ones.

It is NOT part of the pipeline: run it once per baseline result directory,
then forget it.

WHICH ROW SURVIVES
------------------
The newest by timestamp_utc (falling back to file mtime when that column is
absent). For the random control the environment is deterministic and the seed
fixes the action draws, so repeat runs of one key should agree anyway -- but
where they do not, the most recent run is the one that matches the current
code, which is the defensible choice. Divergent duplicates are reported.

Usage:
    # look first -- prints the plan, changes nothing
    python -m scripts.migrate_baseline_metrics_names \\
        --result-dir result/physical_job/baseline_holdout

    # then commit to it
    python -m scripts.migrate_baseline_metrics_names \\
        --result-dir result/physical_job/baseline_holdout --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.naming import metrics_stem

SUFFIXES = ("_metrics.csv", "_metrics.json", "_metrics.raw.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--result-dir", required=True, nargs="+", type=Path,
        help="Baseline result directory/directories to migrate.",
    )
    parser.add_argument(
        "--apply", action="store_true", default=False,
        help="Actually rename/delete. Without it this is a dry run.",
    )
    return parser.parse_args()


def key_of(path: Path) -> tuple[str, str, int | None] | None:
    """(treatment_id, split_id, seed) read from the file's single row."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  [skip] {path.name}: unreadable ({exc})")
        return None
    if df.empty:
        print(f"  [skip] {path.name}: no rows")
        return None
    row = df.iloc[0]
    seed = pd.to_numeric(row.get("seed"), errors="coerce")
    return (
        str(row["treatment_id"]),
        str(row["split_id"]),
        None if pd.isna(seed) else int(seed),
    )


def sort_key(path: Path) -> tuple:
    """Newest-last ordering: timestamp_utc when present, else mtime."""
    try:
        df = pd.read_csv(path)
        stamp = str(df.iloc[0]["timestamp_utc"]) if "timestamp_utc" in df.columns else ""
    except Exception:
        stamp = ""
    return (stamp, path.stat().st_mtime)


def siblings(path: Path) -> list[Path]:
    """The .csv/.json/.raw.csv trio sharing one stem."""
    stem = path.name[: -len("_metrics.csv")]
    return [path.parent / f"{stem}{suffix}" for suffix in SUFFIXES]


def migrate(result_dir: Path, apply: bool) -> int:
    if not result_dir.is_dir():
        print(f"[ERROR] not a directory: {result_dir}")
        return 1

    print(f"\n=== {result_dir} ===")
    by_key: dict[tuple, list[Path]] = {}
    for path in sorted(result_dir.glob("*_metrics.csv")):
        key = key_of(path)
        if key is not None:
            by_key.setdefault(key, []).append(path)

    if not by_key:
        print("  nothing to do (no readable *_metrics.csv)")
        return 0

    renames: list[tuple[Path, Path]] = []
    removals: list[Path] = []

    for key, paths in sorted(by_key.items()):
        treatment_id, split_id, seed = key
        target_stem = metrics_stem(treatment_id, split_id, seed)
        paths.sort(key=sort_key)
        keep, superseded = paths[-1], paths[:-1]
        keep_stem = keep.name[: -len("_metrics.csv")]

        if superseded:
            values = {sort_key(p)[0]: p.name for p in paths}
            print(f"  {treatment_id} split={split_id} seed={seed}: "
                  f"{len(paths)} files -> keeping {keep.name}")
            for stamp, name in sorted(values.items()):
                print(f"      {stamp or '(no timestamp)'}  {name}")

        for path in superseded:
            removals.extend(p for p in siblings(path) if p.exists())

        if keep_stem != target_stem:
            for suffix in SUFFIXES:
                src = keep.parent / f"{keep_stem}{suffix}"
                if src.exists():
                    renames.append((src, keep.parent / f"{target_stem}{suffix}"))

    if not renames and not removals:
        print("  already migrated — nothing to change")
        return 0

    for src, dst in renames:
        print(f"  RENAME {src.name}  ->  {dst.name}")
    for path in removals:
        print(f"  DELETE {path.name}  (superseded)")

    if not apply:
        print(f"\n  dry run — {len(renames)} rename(s), {len(removals)} deletion(s). "
              f"Re-run with --apply to commit.")
        return 0

    for path in removals:
        path.unlink()
    for src, dst in renames:
        src.replace(dst)
    print(f"\n  applied: {len(renames)} renamed, {len(removals)} deleted")
    return 0


def main() -> None:
    args = parse_args()
    status = 0
    for result_dir in args.result_dir:
        status |= migrate(result_dir, args.apply)
    sys.exit(status)


if __name__ == "__main__":
    main()
