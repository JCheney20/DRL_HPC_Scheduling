"""naming.py

Output-path naming shared by the baseline runner and its maintenance scripts.

Deliberately stdlib-only. run_baseline.py cannot host this: importing it pulls
src.utils (torch, stable-baselines3, sb3-contrib) and src.HPCsim (gymnasium),
so scripts/migrate_baseline_metrics_names.py would need the full training
environment to compute a filename. It is run from the snakemake env, which has
neither -- the same constraint prune_manifest.py already respects.
"""

from __future__ import annotations


def metrics_stem(treatment_id: str, split_id: str, seed: int | None = None) -> str:
    """Deterministic basename for one baseline's metrics files.

    Keyed on (treatment_id, split_id, seed) -- the same tuple
    baseline_aggregate.py's duplicate check uses -- and deliberately NOT on
    run_id. run_id is minted fresh by write_manifest_entry on every
    invocation, so under the Snakefile's --force a re-run of one seed wrote a
    second file ("random_016_metrics.csv", then "random_021_metrics.csv")
    rather than replacing the first. baseline_aggregate globs *_metrics.csv,
    so it loaded both and died on its own duplicate check -- the check was
    right, the inputs were stale. Keying the filename on the identity of the
    measurement makes a re-run overwrite its predecessor, so the glob can
    only ever see one row per key.

    run_id still goes into the manifest and into the row itself, so which
    invocation produced a given number remains recoverable.
    """
    stem = f"{treatment_id}__{split_id}"
    if seed is not None:
        stem += f"__seed{seed}"
    return stem
