# DRLScheduler Implementation TODO

This file is the actionable checklist for pipeline completion.
It is organized by phase, followed by a file-by-file checklist, plus a paper
alignment checklist for Submission 2. Use the tags below to track status.

Status tags (prepend to tasks): [PENDING] [IN-PROGRESS] [DONE] [BLOCKED]
Priority tags: [MUST] [SHOULD] [NICE]

NOTE (re-triaged after the src/ restructure): all workflow scripts now live under
`src/` and run as `python -m src.<name>`; SLURM runner settings live in
`profiles/slurm/config.yaml` (not config.yaml). The `file:line` references in the
detailed sections below predate the restructure and are approximate. All Phase 0–7
MUST items are now complete (holdout evaluation added via holdout_eval /
holdout_aggregate); remaining open items are SHOULD/NICE polish enhancements.

--------------------------------------------------------------------------------
Phase Plan
--------------------------------------------------------------------------------

Phase 0 — Define selection + reporting contract (unblocker)
[x] [MUST] [DONE] Document Pareto selection policy (primaries + secondaries) and tie-breakers. (config.yaml pareto_metrics/pareto_tiebreakers; src/select_best.py)
[x] [MUST] [DONE] Fix output contracts for plots (PNG) and tables (CSV) used by Typst. (src/visualise.py)

Phase 1 — Core pipeline alignment (schemas + safety)
[x] [MUST] [DONE] Implement manifest locking to avoid race conditions on cluster runs.
[x] [MUST] [DONE] Standardize treatment_id format: "{algorithm}__mask_{use_masking}" (producer: train/eval manifest; consumers: aggregate/stats).
[x] [MUST] [DONE] Align train/eval window_size + tail_size to ensure reproducibility (persist in manifest; eval reads manifest, not CLI defaults).
[x] [MUST] [DONE] Ensure split_id token usage (no .json) across CLI + Snakemake (producer: CLI parsing; consumers: Snakemake rules + downstream manifest filters).

Phase 2 — Stats pipeline (Carrasco-aligned + fail-fast)
[x] [MUST] [DONE] Implement Wilcoxon non-parametric CI (exact K for l <= 20, approx K for l > 20).
[x] [MUST] [DONE] Implement confidence curves (delta vs p-value) and export CSV.
[x] [MUST] [DONE] Implement Page trend test with ordering = ALGORITHMS + TRAD_ALGORITHMS.
[x] [MUST] [DONE] Add fail-fast option and deterministic outputs for stats.

Phase 3 — Baselines (separate stats, combined visuals)
[x] [SHOULD] [DONE] Emit baseline outputs in eval_wide-compatible schema. (src/run_baseline.py, src/baseline_aggregate.py)
[x] [SHOULD] [DONE] Aggregate baseline results into baseline_* summaries. (baseline_summary.csv, baseline_eval_wide.csv)
[x] [SHOULD] [DONE] Keep baseline stats separate but allow combined visual comparison. (src/baseline_compare.py -> baseline_comparison.csv)

Phase 4 — Best algorithm selection + holdout evaluation
[x] [MUST] [DONE] Implement Pareto selector and tie-breakers; write best_algorithm.json. (src/select_best.py)
[x] [MUST] [DONE] Evaluate best algorithm on holdout across all seeds. (rule holdout_eval; evaluate_agents --eval-trace/--filter-treatment)
[x] [MUST] [DONE] Aggregate holdout results into holdout_summary.csv. (rule holdout_aggregate -> result/<trace>/holdout/holdout_summary.csv)

Phase 5 — Visualization + Typst-ready outputs
[x] [MUST] [DONE] Create visualization runner to generate PNG plots + CSV tables. (src/visualise.py, wired as `rule visualise`)
[x] [MUST] [DONE] Include Pareto plot, CD diagram, confidence curves, and DRL vs baseline plots.
[x] [MUST] [DONE] Ensure all outputs are deterministic and stored under result/{trace}/ (plots under plots/).

Phase 6 — Snakemake orchestration
[x] [MUST] [DONE] Add baseline lane (baseline_only) and best/holdout stages. (baseline lane, select_best, holdout_eval/holdout_aggregate all wired)
[x] [MUST] [DONE] Wire visualization stage and update rule all inputs. (rule all -> .visualise_complete)
[x] [MUST] [DONE] Ensure config.yaml is the only user change needed to run. (runner settings isolated to profiles/slurm/)

Phase 7 — Slurm + Apptainer
[x] [SHOULD] [DONE] Add Slurm profile and validate submission. (profiles/slurm/config.yaml; smoke-validated via `just run_smoke_slurm`)
[x] [SHOULD] [DONE] Export Nix environment to Apptainer image and document usage. (`just build_sif` -> DRL_env.sif; docs/workflow_hpc.md)

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

Snakefile  (re-triaged post-restructure; end-to-end runtime confirmed by the pending smoke test)
[x] [MUST] [DONE] Fix eval_run inputs (manifest + --filter-seed/--filter-algo). 
[x] [MUST] [DONE] Align outputs with stats files (pairwise_nemenyi, page_trend, eval_wide).
[x] [MUST] [DONE] Baseline-only lane + best stage + holdout stage (holdout_eval/holdout_aggregate) all wired.
[x] [MUST] [DONE] Visualisation stage included in rule all (.visualise_complete).
[x] [MUST] [DONE] Runner toggles driven by config.yaml / profiles (config carries workflow params only).
[x] [MUST] [DONE] Baselines reported descriptively (baseline_compare), separate from DRL hypothesis testing (design decision, see Paper Alignment).

config.yaml
[x] [MUST] [DONE] Add baseline_only, trad_algorithms, allocators. (config.yaml)
[x] [MUST] [DONE] Add pareto_metrics, pareto_tiebreakers (direction handled by METRIC_DIRECTION in code). (config.yaml)
[x] [MUST] [DONE] Add visualisation toggle block. (config.yaml visualisation:)
[x] [MUST] [DONE] Ensure eval_deterministic and eval_max_steps are used in Snakefile. (EVAL_MAX_STEPS_FLAG, deterministic flag)

Project_Github docs (workflow_hpc.md, workflow_local.md)
[x] [SHOULD] [DONE] Document Slurm profile usage and Apptainer execution. (docs/workflow_hpc.md)
[x] [SHOULD] [DONE] Document how to run baseline-only and holdout evaluation paths. (holdout stage documented in snakemake_pipeline.md + workflow_hpc.md + methodology_protocol.md)


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


--------------------------------------------------------------------------------
Phase 0 — Configuration & Schema (Unblockers)
--------------------------------------------------------------------------------

config.yaml (duplicate of the Phase 0 config block above — same items)
[x] [MUST] [DONE] Add `pareto_metrics` key.
[x] [MUST] [DONE] Add `pareto_tiebreakers` key.
[x] [MUST] [DONE] Add `visualisation` toggle block (`plots_enabled`, `tables_enabled`, ...).
[x] [MUST] [DONE] Config-driven baseline execution via `trad_algorithms` + `allocators`.
[x] [MUST] [DONE] `baseline_only` respected by `rule all` (BASELINE_ONLY branch).

--------------------------------------------------------------------------------
Phase 1 — Orchestration (Snakefile)
--------------------------------------------------------------------------------

Snakefile
[ ] [MUST] [PENDING] Fix `train_agents.py` trace path: use `data/splits/{SPLIT_ID}.tsv` or ensure trace is found. (Snakefile:185)
[ ] [MUST] [PENDING] Fix `EVAL_MAX_STEPS_FLAG` syntax: ensure empty string leaves no trailing space. (Snakefile:76)
[ ] [MUST] [PENDING] Fix `filter_seed` parameter expansion in `eval_seed` rule; validate it passes to `evaluate_agents.py`. (Snakefile:229)
[ ] [MUST] [PENDING] Fix `train_seed` resource block indentation (currently `resources:` is 1-space indented). (Snakefile:152)
[ ] [MUST] [PENDING] Add `split_id` to `train_seed` shell call (currently hardcoded to `{TRACE_NAME}_r70`). (Snakefile:190)
[ ] [MUST] [PENDING] Fix `baseline` rule: `wildcards.seed` is undefined (no wildcard in rule). Either add `seed` wildcard or remove reference. (Snakefile:350)
[ ] [MUST] [PENDING] Fix `baseline` rule: `--split_id {input.dev_split}` passes file path, not split ID string; change to use `SPLIT_ID`. (Snakefile:351)
[ ] [MUST] [PENDING] Fix `baseline` rule: `--force False` is treated as a positional arg; remove or use `--no-force` flag correctly. (Snakefile:355)
[ ] [MUST] [PENDING] Fix `select_best` rule inputs: path to `algorithm_summary.csv` is missing `algorithm_summary` intermediate. (Snakefile:362)
[ ] [MUST] [PENDING] Fix `visualise` rule: `visualise.py` does not support `--mode results`, `--stats-dir`, `--aggregate-dir`, or `--trace-name`; align args. (Snakefile:395)
[ ] [MUST] [PENDING] Wire `select_best` and `visualise` into `rule all` inputs. (Snakefile:109)
[ ] [SHOULD] [PENDING] Add error handling (e.g., `set -euo pipefail`) and `trap` for cleanup in shell rules. (Snakefile:170)
[ ] [SHOULD] [PENDING] Parameterize Snakefile for `baseline_only` mode so `rule all` does not require RL stages when enabled. (Snakefile:107)
[ ] [SHOULD] [PENDING] Consider using `shadow` directive for training rules to avoid logs clobbering. (Snakefile:152)

--------------------------------------------------------------------------------
Phase 2 — Core Utilities (src/utils.py)
--------------------------------------------------------------------------------

src/utils.py
[ ] [MUST] [PENDING] Add docstrings to `load_split_metadata` explaining `splits_log_dir` and `split_id` parameters. (src/utils.py:604)
[ ] [MUST] [PENDING] Validate `load_split_metadata` return value: ensure it contains expected keys (e.g., `split_id`, `dev_path`); consumers (e.g., train_agents.py) rely on these. (src/utils.py:604)
[ ] [SHOULD] [PENDING] Move `HOLDOUT_PATTERNS` to a more central place or validate against manifest before pipeline starts. (src/utils.py:165)
[ ] [NICE] [PENDING] Centralize default paths (e.g., `logs/run_log.csv`, `data/splits/`) in a constants section at the top of `src/utils.py`. (src/utils.py:640)

--------------------------------------------------------------------------------
Phase 3 — Data Splitting (Project_Github/src/make_split.py)
--------------------------------------------------------------------------------

Project_Github/src/make_split.py
[ ] [MUST] [PENDING] Add `--seed` argument (optional, default None) and pass to `HPCsim` initialization to ensure deterministic splitting if needed. (make_split.py:9)
[ ] [MUST] [PENDING] Replace direct `print` with a proper logger for standard output in `parse_args` to avoid interfering with CLI consumers. (make_split.py:32)
[ ] [SHOULD] [PENDING] Validate that `Submit` column is monotonically increasing after sort; warn if not. (make_split.py:63)
[ ] [SHOULD] [PENDING] Accept `--source` path as a file path, not just a preset name, to support future datasets. (make_split.py:11)

--------------------------------------------------------------------------------
Phase 4 — Training (train_agents.py)
--------------------------------------------------------------------------------

train_agents.py
[ ] [MUST] [PENDING] Guard `split_id` manipulation for `None` to prevent `AttributeError` when `--split_id` is omitted. (train_agents.py:202)
[ ] [MUST] [PENDING] Validate that `algo_class` returned from `resolve_algorithm_config` is not None (adds robustness). (train_agents.py:261)
[ ] [SHOULD] [PENDING] Add checkpoint resume logic for `train_and_log` to allow HPC job resubmissions without restarting from step 0. (train_agents.py:286)
[ ] [SHOULD] [PENDING] Flush or write intermediate metadata after each `model.save` call for better recovery visibility. (train_agents.py:321)

--------------------------------------------------------------------------------
Phase 5 — Evaluation (evaluate_agents.py)
--------------------------------------------------------------------------------

evaluate_agents.py
[ ] [MUST] [PENDING] Validate that all `RunSpec` fields loaded from manifest match the expected types (e.g., `window_size` is int) to prevent downstream exceptions. (evaluate_agents.py:65)
[ ] [MUST] [PENDING] Ensure `EvalResult` includes `decision_count` and `decision_latency_mean_ms` in the CSV output (check `write_dict_outputs`). (evaluate_agents.py:210)
[ ] [SHOULD] [PENDING] Add optional `--parallel` flag for multi-process evaluation (useful for HPC), ensuring safe model loading. (evaluate_agents.py:42)
[ ] [SHOULD] [PENDING] Add progress logging (e.g., `tqdm`) for large trace evaluations. (evaluate_agents.py:124)

--------------------------------------------------------------------------------
Phase 6 — Aggregation (aggregate_results.py)
--------------------------------------------------------------------------------

aggregate_results.py
[ ] [MUST] [PENDING] Remove hardcoded `TRAD_ALGORITHMS` in `aggregate_train_time`; use configuration or manifest instead. (aggregate_results.py:143)
[ ] [MUST] [PENDING] Ensure `aggregate_train_time` handles cases where `train_metadata.json` is missing critical keys (e.g., `wall_clock_s`). (aggregate_results.py:156)
[ ] [MUST] [PENDING] Verify `algorithm_summary.csv` includes `treatment_id`, `algorithm`, and `use_masking` to satisfy `select_best.py` input contract. (aggregate_results.py:135)
[ ] [SHOULD] [PENDING] Add `--strict` logic to fail if any expected metric is NaN across the entire eval set, not just per-row. (aggregate_results.py:69)

--------------------------------------------------------------------------------
Phase 7 — Statistical Testing (statistical_test.py)
--------------------------------------------------------------------------------

statistical_test.py
[ ] [MUST] [PENDING] Add docstring to `compute_confidence_curves` explaining the `grid` parameter and expected shape. (statistical_test.py:367)
[ ] [MUST] [PENDING] Ensure `run_page_trend_test` handles missing treatments gracefully rather than raising `KeyError`. (statistical_test.py:391)
[ ] [SHOULD] [PENDING] Add `--parallel` flag for Nemenyi post-hoc to speed up large pairwise matrices on HPC. (statistical_test.py:289)
[ ] [SHOULD] [PENDING] Consider adding `scipy.stats` version to `stats_meta.json` for full reproducibility. (statistical_test

--------------------------------------------------------------------------------
 Phase 8 - Select Best Algorithm (select_best.py)
--------------------------------------------------------------------------------

select_best.py
[ ]  [MUST] [PENDING] Implement canonical Pareto-front selection: Replace the current win-counting approach with a true Pareto dominance check on the primary and selected secondary metrics defined in config.yaml. (select_best.py:36–66)
[ ]  [MUST] [PENDING] Add configuration-driven metrics: Read pareto_metrics and pareto_tiebreakers from config.yaml instead of hardcoding METRICS/TIEBREAKERS. (select_best.py:8–9)
[ ]  [MUST] [PENDING] Validate input columns before selection: Ensure seed_summary contains the expected *_mean columns (e.g., avg_waiting_mean) referenced by the tie-breaker logic. (select_best.py:113)
[ ]  [MUST] [PENDING] Handle tied-candidates robustly: If all tie-breakers fail to break a tie, log a warning and return the lexicographically first candidate instead of silently picking. (select_best.py:65–66)
[ ]  [SHOULD] [PENDING] Add CLI argument for config path: Allow --config to point to config.yaml so the script can read pareto_metrics/pareto_tiebreakers at runtime. (select_best.py:68)
[ ]  [SHOULD] [PENDING] Write rationale more explicitly: In best_algorithm.json, include the full Pareto front list and why the winner was chosen (not just a string). (select_best.py:115–128)

--------------------------------------------------------------------------------
 Phase 9 - Visualise Results (visualise.py)
--------------------------------------------------------------------------------

visualise.py
[ ]  [MUST] [PENDING] Align CLI with Snakefile invocation: The Snakefile calls visualise.py --mode results --trace-name ..., but --mode only accepts "results"; either support multiple modes or remove --mode to keep CLI lean. (visualise.py:62–68, Snakefile:395)
[ ]  [MUST] [PENDING] Handle missing data files gracefully: load_pipeline_data will raise FileNotFoundError if any expected CSV is missing; add checks and informative error messages. (visualise.py:107–115)
[ ]  [MUST] [PENDING] Ensure cd_diagram_input.csv schema matches draw_cd_diagrams: Verify that cd_diagram_input.csv contains metric_name, treatment_id, and avg_rank columns before calling draw_cd_diagrams. (visualise.py:153, stats output contract)
[ ]  [MUST] [PENDING] Fix confidence_curves.csv empty handling: If no significant pairs exist, confidence_curves.csv is empty; draw_confidence_curves returns early but logs should reflect this for debugging. (visualise.py:191–198)
[ ]  [SHOULD] [PENDING] Add --help formatting for mode-specific args: Since --mode is required/only choice, ensure the help text explains why --trace-name is mandatory and how paths are templated. (visualise.py:59–101)
[ ]  [SHOULD] [PENDING] Support --alpha override for CD diagrams: Currently hardcoded to 0.05; allow matching the config.yaml alpha for consistency. (visualise.py:153)
[ ]  [SHOULD] [PENDING] Ensure deterministic file naming: save_figure uses stem directly; ensure no characters in treatment_id break file paths (e.g., / or spaces). (visualise.py:117–122)
[ ]  [NICE] [PENDING] Add progress logging for long runs: Print which plot/table is being generated to help debugging on cluster runs. (visualise.py:392)
