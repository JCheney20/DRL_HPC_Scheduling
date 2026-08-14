"""dump_tb_scalars.py -- print the tail of a run's TensorBoard scalars.

Reading the run-up to a training crash matters more than reading the crash.
A2C records train/entropy_loss, train/value_loss, train/policy_loss and
train/explained_variance every update (a2c_mask.py:381-386), so a run that
died of divergence leaves its whole approach in the event file -- entropy
collapsing or pinning to the ceiling, value_loss climbing, explained_variance
falling away. That evidence is already on disk after a failure, which makes it
cheaper than any re-run.

Note on reading entropy: SB3 logs entropy_LOSS, which is -mean(entropy). For a
513-action space, entropy_loss near 0 means a saturated near-deterministic
policy; near -6.24 (= -ln 513) means near-uniform. Both extremes are
informative and they mean opposite things.

Usage:
    python -m scripts.dump_tb_scalars --logdir logs/physical_job/57434/maskable_a2c
    python -m scripts.dump_tb_scalars --logdir logs/... --tail 40 --csv out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Printed in this order when present; anything else found is appended after.
PREFERRED = [
    "train/entropy_loss",
    "train/policy_loss",
    "train/value_loss",
    "train/loss",
    "train/explained_variance",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--logdir", required=True, type=Path,
                        help="Run directory (searched recursively for event files).")
    parser.add_argument("--tail", default=25, type=int,
                        help="How many of the most recent points to print. Default 25.")
    parser.add_argument("--csv", default=None, type=Path,
                        help="Also write the full series to this CSV.")
    return parser.parse_args()


def find_event_files(logdir: Path) -> list[Path]:
    return sorted(logdir.rglob("events.out.tfevents.*"))


def main() -> None:
    args = parse_args()
    events = find_event_files(args.logdir)
    if not events:
        print(f"[ERROR] no event files under {args.logdir}")
        sys.exit(1)

    print(f"Found {len(events)} event file(s) under {args.logdir}")
    for path in events:
        print(f"  {path}")

    # Load them all; SB3 restarts a run into *_1, *_2, ... subdirectories.
    series: dict[str, dict[int, float]] = {}
    for path in events:
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            bucket = series.setdefault(tag, {})
            for event in acc.Scalars(tag):
                bucket[event.step] = event.value

    if not series:
        print("[ERROR] event files contain no scalars")
        sys.exit(1)

    tags = [t for t in PREFERRED if t in series]
    tags += [t for t in sorted(series) if t not in tags]

    steps = sorted({step for bucket in series.values() for step in bucket})
    shown = steps[-args.tail:]

    print(f"\n{len(steps)} update(s) recorded; showing the last {len(shown)}.")
    print(f"\n{'step':>12}  " + "  ".join(f"{t[-18:]:>18}" for t in tags))
    for step in shown:
        cells = []
        for tag in tags:
            value = series[tag].get(step)
            cells.append("".rjust(18) if value is None else f"{value:>18.6g}")
        print(f"{step:>12}  " + "  ".join(cells))

    # Flag the things worth acting on rather than making the reader spot them.
    print("\nNotes:")
    entropy = series.get("train/entropy_loss")
    if entropy:
        last = entropy[max(entropy)]
        print(f"  entropy_loss last = {last:.6g}  (0 => saturated/deterministic; "
              f"-ln(n_actions) => uniform)")
    value = series.get("train/value_loss")
    if value:
        finite = [v for v in value.values() if v == v]
        if finite:
            print(f"  value_loss  min={min(finite):.6g}  max={max(finite):.6g}  "
                  f"last={value[max(value)]:.6g}")
    for tag, bucket in series.items():
        bad = [s for s, v in bucket.items() if v != v or v in (float("inf"), float("-inf"))]
        if bad:
            print(f"  {tag}: NON-FINITE at step(s) {bad[:5]}"
                  f"{' ...' if len(bad) > 5 else ''}  <-- first divergence lands here")

    if args.csv:
        rows = ["step," + ",".join(tags)]
        for step in steps:
            rows.append(str(step) + "," + ",".join(
                "" if series[t].get(step) is None else repr(series[t][step]) for t in tags
            ))
        args.csv.write_text("\n".join(rows) + "\n")
        print(f"\nWrote full series to {args.csv}")


if __name__ == "__main__":
    main()
