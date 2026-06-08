# DRLScheduler Implementation TODO

This file is the actionable checklist for pipeline completion.
It is organized by phase, followed by a file-by-file checklist, plus a paper
alignment checklist for Submission 2. Use the tags below to track status.

Status tags (prepend to tasks): [PENDING] [IN-PROGRESS] [DONE] [BLOCKED]
Priority tags: [MUST] [SHOULD] [NICE]

--------------------------------------------------------------------------------
Phase Plan
--------------------------------------------------------------------------------

Phase 0 — Define selection + reporting contract (unblocker)
[☑ ] [MUST] [PENDING] Document Pareto selection policy (primaries + secondaries) and tie-breakers.
[ ] [MUST] [PENDING] Fix output contracts for plots (PNG) and tables (CSV) used by Typst.

Phase 1 — Core pipeline alignment (schemas + safety)
[☑] [MUST] [DONE] Implement manifest locking to avoid race conditions on cluster runs.
[☑] [MUST] [DONE] Standardize treatment_id format: "{algorithm}__mask_{use_masking}" (producer: train/eval manifest; consumers: aggregate/stats).
[☑] [MUST] [DONE] Align train/eval window_size + tail_size to ensure reproducibility (persist in manifest; eval reads manifest, not CLI defaults).
[☑] [MUST] [DONE] Ensure split_id token usage (no .json) across CLI + Snakemake (producer: CLI parsing; consumers: Snakemake rules + downstream manifest filters).

Phase 2 — Stats pipeline (Carrasco-aligned + fail-fast)
[☑] [MUST] [DONE] Implement Wilcoxon non-parametric CI (exact K for l <= 20, approx K for l > 20).
[☑] [MUST] [DONE] Implement confidence curves (delta vs p-value) and export CSV.
[☑] [MUST] [DONE] Implement Page trend test with ordering = ALGORITHMS + TRAD_ALGORITHMS.
[☑] [MUST] [DONE] Add fail-fast option and deterministic outputs for stats.

Phase 3 — Baselines (separate stats, combined visuals)
[ ] [SHOULD] [PENDING] Emit baseline outputs in eval_wide-compatible schema.
[ ] [SHOULD] [PENDING] Aggregate baseline results into baseline_* summaries.
[ ] [SHOULD] [PENDING] Keep baseline stats separate but allow combined visual comparison.

Phase 4 — Best algorithm selection + holdout evaluation
[ ] [MUST] [PENDING] Implement Pareto selector and tie-breakers; write best_algorithm.json.
[ ] [MUST] [PENDING] Evaluate best algorithm on holdout across all seeds.
[ ] [MUST] [PENDING] Aggregate holdout results into holdout_summary.csv.

Phase 5 — Visualization + Typst-ready outputs
[ ] [MUST] [PENDING] Create visualization runner to generate PNG plots + CSV tables.
[ ] [MUST] [PENDING] Include Pareto plot, CD diagram, confidence curves, and DRL vs baseline plots.
[ ] [MUST] [PENDING] Ensure all outputs are deterministic and stored under result/{trace}/.

Phase 6 — Snakemake orchestration
[ ] [MUST] [PENDING] Add baseline lane (baseline_only) and best/holdout stages.
[ ] [MUST] [PENDING] Wire visualization stage and update rule all inputs.
[ ] [MUST] [PENDING] Ensure config.yaml is the only user change needed to run.

Phase 7 — Slurm + Apptainer
[ ] [SHOULD] [PENDING] Add Slurm profile and validate submission.
[ ] [SHOULD] [PENDING] Export Nix environment to Apptainer image and document usage.

--------------------------------------------------------------------------------
File-by-File Implementation Checklist
--------------------------------------------------------------------------------

AGENTS.md
[☑] [DONE] Update statistical framework text (Nemenyi + Wilcoxon CI + confidence curves + Page trend).
[ ] [PENDING] Keep paper alignment notes consistent with stats implementation.

src/utils.py
[☑] [MUST] [DONE] Acquire fcntl LOCK_EX in write_manifest_entry before read/write. (src/utils.py:581)
[☑] [MUST] [DONE] Add structured fields needed by eval/stats (window_size, tail_size, split_id). (src/utils.py:444)
[☑] [MUST] [DONE] Ensure treatment_id format is applied consistently downstream. (src/utils.py:142)

train_agents.py
[☑] [MUST] [DONE] Enforce split_id token (reject filenames like *.json). (train_agents.py:192)
[☑] [MUST] [DONE] Persist train-time window_size/tail_size into manifest metadata so eval uses trained values (pair with evaluate_agents.py:94). (train_agents.py:229)
[☑] [MUST] [DONE] Persist use_masking and/or treatment_id in manifest for downstream grouping (pair with evaluate_agents.py:115 and aggregate_results.py:116). (train_agents.py:288)
[☑] [SHOULD] [DONE] Capture per-model training wall time in metadata for average per trace. (train_agents.py:300)

evaluate_agents.py
[☑] [MUST] [DONE] Fix manifest iteration: use df.iterrows() directly (remove pd.Series wrapper). (evaluate_agents.py:56)
[☑] [MUST] [DONE] Load window_size/tail_size from manifest metadata (trained values), not CLI defaults (pair with train_agents.py:229). (evaluate_agents.py:94)
[☑] [MUST] [DONE] Use manifest-stored use_masking/treatment_id to keep grouping consistent with training (pair with train_agents.py:288). (evaluate_agents.py:115)

aggregate_results.py (herasched)
[☑] [MUST] [DONE] Confirm eval_wide.csv output and required columns are present. (aggregate_results.py:1)
[☑] [MUST] [DONE] Ensure seed_summary includes treatment_id (from manifest) and metric means/std (pair with train_agents.py:288). (aggregate_results.py:116)
[☑] [MUST] [DONE] Add strict validation of numeric fields (no NaN/inf) and fail-fast behavior. (aggregate_results.py:175)
[̌☑] [SHOULD] [DONE] Aggregate average train-time per model per trace from train metadata (pair with train_agents.py:300). (aggregate_results.py:139)

statistical_test.py
[☑] [MUST] [DONE] Implement Wilcoxon CI per Carrasco (exact/approx K) and avoid O(l^2) memory spikes. (statistical_test.py:300)
[☑] [MUST] [DONE] Implement confidence curves and export CSV format: (statistical_test.py:316)
    metric, treatment_a, treatment_b, delta, p_value.
[☑] [MUST] [DONE] Implement Page trend test using ordering = ALGORITHMS + TRAD_ALGORITHMS. (statistical_test.py:468)
[☑] [MUST] [DONE] Add fail-fast mode and include ordering metadata in stats_meta.json. (statistical_test.py:597)
[☑] [MUST] [DONE] Update outputs: pairwise_nemenyi.csv, confidence_curves.csv, page_trend.csv. (statistical_test.py:636)
[☑] [MUST] [DONE] Keep baselines in stats; only skip tests when preconditions fail (e.g., rank degeneracy for Nemenyi). (statistical_test.py:255)

run_baseline.py
[ ] [SHOULD] [PENDING] Add CLI args for selectors/allocators to be config-driven. (run_baseline.py:170)
[ ] [SHOULD] [PENDING] Emit eval_wide-compatible outputs with algorithm, use_masking=false, split_id, seed (consumer: aggregate_results.py expects treatment_id fields). (run_baseline.py:23)
[ ] [NICE] [PENDING] Optionally restrict to TRAD_ALGORITHMS for faster runs. (run_baseline.py:87)
[ ] [SHOULD] [PENDING] Ensure baseline schema supports inclusion in stats by default. (run_baseline.py:23)

select_best.py (new)
[ ] [MUST] [PENDING] Compute Pareto front over primaries + chosen secondaries. (select_best.py:TBD)
[ ] [MUST] [PENDING] Tie-break: avg_waiting -> avg_slowdown -> cpu_utilization. (select_best.py:TBD)
[ ] [MUST] [PENDING] Write best_algorithm.json with rationale and metrics. (select_best.py:TBD)

visualise_results.py (new)
[ ] [MUST] [PENDING] Generate PNG plots: Pareto, CD, confidence curves, DRL vs baseline. (visualise_results.py:TBD)
[ ] [MUST] [PENDING] Emit CSV tables for Typst inclusion. (visualise_results.py:TBD)
[ ] [MUST] [PENDING] Keep outputs deterministic and under result/{trace}/plots and /tables. (visualise_results.py:TBD)

Snakefile
[ ] [MUST] [PENDING] Fix eval_seed inputs (manifest + filter_seed) and result path duplication. (Snakefile:199)
[ ] [MUST] [PENDING] Align outputs with stats files (pairwise_nemenyi, page_trend, eval_wide). (Snakefile:272)
[ ] [MUST] [PENDING] Add baseline-only lane and best/holdout stages. (Snakefile:311)
[ ] [MUST] [PENDING] Add visualization stage and include outputs in rule all. (Snakefile:103)
[ ] [MUST] [PENDING] Ensure all toggles are driven by config.yaml. (Snakefile:41)
[ ] [MUST] [PENDING] Keep baselines included in stats by default; only skip tests on precondition failure. (Snakefile:311)

config.yaml
[ ] [MUST] [PENDING] Add baseline_only, baseline_selectors, baseline_allocators. (config.yaml:40)
[ ] [MUST] [PENDING] Add pareto_metrics, pareto_tiebreakers, pareto_direction. (config.yaml:35)
[ ] [MUST] [PENDING] Add visualization toggles and best_algo_policy. (config.yaml:40)
[ ] [MUST] [PENDING] Ensure eval_deterministic and eval_max_steps are used in Snakefile. (config.yaml:40)

Project_Github docs (workflow_hpc.md, workflow_local.md)
[ ] [SHOULD] [PENDING] Document Slurm profile usage and Apptainer execution.
[ ] [SHOULD] [PENDING] Document how to run baseline-only and holdout evaluation paths.


--------------------------------------------------------------------------------
Paper Alignment Checklist (Submmisions/4323819_Paper.typ)
--------------------------------------------------------------------------------

Keep these changes minimal; only add text if the current draft does not already
cover the point or if a clarification is needed.

[☑] [MUST] [DONE] Add a one-line note that baselines are reported descriptively
    (combined visuals) but excluded from statistical hypothesis testing.
[☑] [MUST] [DONE] Clarify that Pareto selection uses primaries + selected
    secondaries, with tie-breakers: avg_waiting -> avg_slowdown -> cpu_utilization.
[☑] [MUST] [DONE] Add a brief sentence that best-algorithm holdout evaluation is
    performed across multiple seeds (no tuning on holdout).
[☑] [SHOULD] [DONE] Mention that plots/tables are generated from pipeline outputs
    (eval_wide/seed_summary/stats) and exported as PNG/CSV for Typst inclusion.
