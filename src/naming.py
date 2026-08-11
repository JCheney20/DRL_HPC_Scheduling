"""naming.py

Output-path naming shared by the baseline runner and its maintenance scripts.

Deliberately stdlib-only. run_baseline.py cannot host this: importing it pulls
src.utils (torch, stable-baselines3, sb3-contrib) and src.HPCsim (gymnasium),
so scripts/migrate_baseline_metrics_names.py would need the full training
environment to compute a filename. It is run from the snakemake env, which has
neither -- the same constraint prune_manifest.py already respects.
"""

from __future__ import annotations

# Backfill is a *confound*, not a property of the heuristic: HPCsim.run() sweeps
# the whole queue for backfillable jobs and holds a reservation for the head job,
# whereas HPCsim.step() -- the MDP the DRL treatments and the N27 random control
# are evaluated on -- has no such sweep. Comparing a backfilling heuristic against
# a non-backfilling agent measures the two together. So the heuristics are run in
# BOTH configurations and reported as two bands: backfill=off is the controlled
# comparison (same mechanism on both sides of the table), backfill=on is the
# production reference (what a real scheduler would actually do).
#
# The two configurations are the same `algorithm` with different treatment_ids,
# which is what keeps them apart in the manifest, in baseline_aggregate's
# duplicate check (keyed on treatment_id) and in its per-treatment grouping.
# Anything downstream that selects a baseline by `algorithm` ALONE is ambiguous
# once both bands exist and must select on treatment_id instead.
NO_BACKFILL_SUFFIX = "__nobf"


def heuristic_treatment_id(algorithm: str, backfill: bool = True) -> str:
    """treatment_id for one deterministic heuristic in one backfill configuration.

    The backfill=on id is left EXACTLY as it was (``{algo}__mask_false``) so the
    already-completed with-backfill runs keep their identity and do not have to
    be re-run or migrated; only the new configuration takes a suffix.
    """
    base = f"{algorithm}__mask_false"
    return base if backfill else base + NO_BACKFILL_SUFFIX


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
