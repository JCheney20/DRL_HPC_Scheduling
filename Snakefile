"""
Snakefile — DRLScheduler Complete Pipeline

Purpose:
    Orchestrate end-to-end training → evaluation → aggregation → statistics
    pipeline for DRL scheduling algorithms on HPC traces.

DAG Flow:
    make_split (idempotent)
        └─→ train_seed (parallelised by seed × algorithm)
                └─→ eval_seed (parallelised by seed)
                        └─→ aggregate
                                └─→ stats
    (Optional) baseline_comparison

Usage:
    # Full production run
    snakemake --configfile config.yaml --cores all

    # Smoke test
    snakemake --configfile config.smoke.yaml --cores all

    # Dry run (validate DAG only)
    snakemake --configfile config.smoke.yaml --dry-run

    # Baseline only
    snakemake --configfile config.yaml result/{trace_name}/baseline/baseline_metadata.json

Ref:
    Snakemake 9.x docs: https://snakemake.readthedocs.io/en/stable/
"""

from pathlib import Path

# =============================================================================
# CONFIGURATION
# Load from --configfile; fall back to sensible defaults so --dry-run works
# without a configfile.
# =============================================================================

configfile: "config.yaml"

TRACE_NAME      = config["trace_name"]
SEEDS           = config["seeds"]
ALGORITHMS      = config["algorithms"]
SAVE_INTERVAL   = config["save_interval"]
TOTAL_SAVING    = config["total_saving"]
WINDOW_SIZE     = config["window_size"]
BUFFER_SIZE     = config["buffer_size"]
TAIL_SIZE       = config["tail_size"]
TOPOLOGY_FILE   = config["topology_file"]
NODE_FILE       = config["node_file"]
ALPHA           = config["alpha"]
BOOTSTRAP_REPS  = config["bootstrap_reps"]
BOOTSTRAP_SEED  = config["bootstrap_seed"]
EVAL_MAX_STEPS  = config.get("eval_max_steps", None)
EVAL_DETERMINISTIC = config.get("eval_deterministic", True)
BASELINE_ONLY = config.get("baseline_only", False)
BASELINE_SELECTORS = config.get("baseline_selectors", ["fcfs", "lcfs", "sjf"])
BASELINE_ALLOCATORS = config.get("baseline_allocators", ["best_fit"])

# Derived paths
RAW_TRACE    = f"data/{TRACE_NAME}.csv"
SPLIT_ID     = f"{TRACE_NAME}_dev70"
HOLDOUT_ID   = f"{TRACE_NAME}_holdout30"
DEV_SPLIT    = f"data/splits/{SPLIT_ID}.tsv"
HOLDOUT_SPLIT = f"data/splits/{HOLDOUT_ID}.tsv"
SPLIT_META   = f"data/splits/logs/{TRACE_NAME}_r70.json"

# Algorithms string for shell loops
ALGORITHMS_STR = " ".join(ALGORITHMS)

# Optional eval max-steps flag
EVAL_MAX_STEPS_FLAG = f"--max-steps {EVAL_MAX_STEPS}" if EVAL_MAX_STEPS else ""

# =============================================================================
# INPUT VALIDATION (checked at DAG build time)
# =============================================================================

if TRACE_NAME not in ["physical_job", "deep_learn"]:
    raise ValueError(
        f"config trace_name must be 'physical_job' or 'deep_learn', got '{TRACE_NAME}'"
    )

if not Path(RAW_TRACE).exists():
    raise FileNotFoundError(f"Raw trace not found: {RAW_TRACE}")

REQUIRED_SCRIPTS = [
    "src/train_agents.py",
    "src/evaluate_agents.py",
    "src/aggregate_results.py",
    "src/statistical_test.py",
]
for script in REQUIRED_SCRIPTS:
    if not Path(script).exists():
        raise FileNotFoundError(f"Required script not found: {script}")


# =============================================================================
# RULE all — Default target
# =============================================================================

rule all:
    """Full pipeline target: train → eval → aggregate → stats."""
    input:
        f"result/{TRACE_NAME}/stats/stats_summary.json",
        # TODO: Add baseline stage outputs here once integrated (baseline eval/aggregate/stats).
        # TODO: Baselines do not train; they must still emit eval/aggregate inputs for stats.
        # Ref: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html


# =============================================================================
# RULE make_split — Create train/holdout splits (idempotent)
# =============================================================================

rule make_split:
    """
    Create time-ordered dev (70%) and holdout (30%) splits from raw trace.
    Idempotent: Snakemake skips if outputs already exist.
    """
    input:
        trace=RAW_TRACE,
    output:
        dev=DEV_SPLIT,
        holdout=HOLDOUT_SPLIT,
        metadata=SPLIT_META,
    log:
        f"logs/snakemake/{TRACE_NAME}/make_split.log",
    shell:
        """
        python src/make_split.py \
            --src {TRACE_NAME} \
            >> {log} 2>&1
        """


# =============================================================================
# RULE train_seed — Train all algorithms for one seed
# =============================================================================

rule train_seed:
    """
    Train all algorithms sequentially for a given seed on the DEV split.
    Parallelisation: seeds run in parallel, algorithms run sequentially
    within each seed to avoid manifest write conflicts on run_log.csv.
    """
    input:
        dev_split=DEV_SPLIT,
        split_meta=SPLIT_META,
    output:
        marker=touch(f"trained_model/{TRACE_NAME}/{{seed}}/.train_complete"),
        # manifest="logs/run_log.csv",
    log:
        f"logs/snakemake/{TRACE_NAME}/train_seed_{{seed}}.log",
    params:
        algorithms=ALGORITHMS_STR,
        save_interval=SAVE_INTERVAL,
        total_saving=TOTAL_SAVING,
        window_size=WINDOW_SIZE,
        buffer_size=BUFFER_SIZE,
        tail_size=TAIL_SIZE,
        topology=TOPOLOGY_FILE,
        node=NODE_FILE,
        trace=SPLIT_ID,
        trace_name=TRACE_NAME,
    # TODO: split_id passed to train_agents.py must be a token (e.g., physical_job_r70),
    # TODO: not a filename with .json. Using a filename breaks load_split_metadata.
    # Ref: https://docs.python.org/3/library/pathlib.html
    wildcard_constraints:
        seed=r"\d+",
    shell:
        """
        mkdir -p trained_model/{params.trace_name}/{wildcards.seed}

        for algo in {params.algorithms}; do
            echo "[train_seed] seed={wildcards.seed} algo=$algo" >> {log}
            python src/train_agents.py \
                --algorithm "$algo" \
                --seed {wildcards.seed} \
                --trace splits/{SPLIT_ID}.tsv \
                --split_id {TRACE_NAME}_r70.json \
                --save_interval {params.save_interval} \
                --total_saving {params.total_saving} \
                --window {params.window_size} \
                --buffer_size {params.buffer_size} \
                --tail {params.tail_size} \
                --topology {params.topology} \
                --node {params.node} \
                --name {params.trace_name}/{wildcards.seed}/$algo \
                >> {log} 2>&1
        done
        """


# =============================================================================
# RULE eval_seed — Evaluate trained models for one seed
# =============================================================================

rule eval_seed:
    """
    Evaluate all trained models for a specific seed.
    Uses --filter-seed to evaluate only runs matching this seed from the manifest.
    Depends on train_seed marker to ensure training completed first.
    """
    input:
        marker=f"trained_model/{TRACE_NAME}/{{seed}}/.train_complete"
    output:
        marker=touch(f"result/{TRACE_NAME}/eval_runs/.seed_{{seed}}_complete"),
    log:
        f"logs/snakemake/{TRACE_NAME}/eval_seed_{{seed}}.log",
    params:
        eval_root=f"result/{TRACE_NAME}/eval_runs",
        max_steps_flag=EVAL_MAX_STEPS_FLAG,
        deterministic_flag=("--deterministic" if EVAL_DETERMINISTIC else "--no-deterministic"),
    # TODO: Define manifest input explicitly (logs/run_log.csv) to avoid undefined {input.manifest}.
    # TODO: Add filter_seed param so --filter-seed has a value; otherwise eval runs all seeds.
    # Ref: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#input-functions
    wildcard_constraints:
        seed=r"\d+",
    shell:
        """
        # TODO: eval_root already includes "result/"; avoid "result/result" duplication.
        # TODO: Use "mkdir -p {params.eval_root}/runs" instead.
        # Ref: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#shellcmd
        mkdir -p result/{params.eval_root}/runs

        python src/evaluate_agents.py \
            --manifest {input.manifest} \
            --output-dir {params.eval_root} \
            --filter-seed {params.filter_seed} \
            {params.deterministic_flag} \
            {params.max_steps_flag} \
            >> {log} 2>&1
        """


# =============================================================================
# RULE aggregate — Merge all seed eval results
# =============================================================================

rule aggregate:
    """
    Aggregate eval outputs from all seeds into summary tables.
    Waits for all eval_seed rules via expand().
    """
    input:
        eval_markers=expand(
            f"result/{TRACE_NAME}/eval_runs/.seed_{{seed}}_complete",
            seed=SEEDS,
        ),
        # manifest="logs/run_log.csv",
    output:
        eval_wide=f"result/{TRACE_NAME}/aggregate/eval_wide.csv",
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
        algorithm_summary=f"result/{TRACE_NAME}/aggregate/algorithm_summary.csv",
        aggregate_meta=f"result/{TRACE_NAME}/aggregate/aggregate_metadata.json",
    log:
        f"logs/snakemake/{TRACE_NAME}/aggregate.log",
    params:
        eval_root=f"result/{TRACE_NAME}/eval_runs/runs",
        output_dir=f"result/{TRACE_NAME}/aggregate",
    shell:
        """
        python src/aggregate_results.py \
            --manifest {input.manifest} \
            --eval-root {params.eval_root} \
            --output-dir {params.output_dir} \
            >> {log} 2>&1
        """


# =============================================================================
# RULE stats — Run statistical tests
# =============================================================================

rule stats:
    """
    Run full non-parametric statistical pipeline on seed_summary.csv.
    Tests: Shapiro-Wilk, Friedman, Conover post-hoc, Kendall W,
           VDA, bootstrap CIs, CD diagram input.
    """
    input:
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
    output:
        stats_summary=f"result/{TRACE_NAME}/stats/stats_summary.json",
        pairwise_conover=f"result/{TRACE_NAME}/stats/pairwise_conover.csv",
        cd_diagram=f"result/{TRACE_NAME}/stats/cd_diagram_input.csv",
        stats_meta=f"result/{TRACE_NAME}/stats/stats_meta.json",
    # TODO: Update output filenames to match stats pipeline: pairwise_nemenyi.csv, page_trend.csv.
    # TODO: Keep filenames consistent with AGENTS.md statistical framework.
    # Ref: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#output-files
    log:
        f"logs/snakemake/{TRACE_NAME}/stats.log",
    params:
        output_dir=f"result/{TRACE_NAME}/stats",
        alpha=ALPHA,
        bootstrap_reps=BOOTSTRAP_REPS,
        bootstrap_seed=BOOTSTRAP_SEED,
    shell:
        """
        python src/statistical_test.py \
            --input {input.seed_summary} \
            --output-dir {params.output_dir} \
            --alpha {params.alpha} \
            --bootstrap-reps {params.bootstrap_reps} \
            --bootstrap-seed {params.bootstrap_seed} \
            >> {log} 2>&1
        """


# =============================================================================
# RULE baseline_comparison — Optional traditional scheduler baseline
# =============================================================================

rule baseline_comparison:
    """
    Run traditional scheduling baseline (FCFS + best_fit) for comparison.
    Optional: only runs if explicitly requested.

    Usage:
        snakemake result/{TRACE_NAME}/baseline/baseline_metadata.json \\
            --configfile config.yaml
    """
    input:
        dev_split=DEV_SPLIT,
    output:
        baseline_meta=f"result/{TRACE_NAME}/baseline/baseline_metadata.json",
    log:
        f"logs/snakemake/{TRACE_NAME}/baseline.log",
    params:
        output_dir=f"result/{TRACE_NAME}/baseline",
        selectors=" ".join(BASELINE_SELECTORS),
        allocators=" ".join(BASELINE_ALLOCATORS),
    shell:
        """
        python src/run_baseline.py \
            --selector fcfs \
            --allocator best_fit \
            --trace {input.dev_split} \
            --output {params.output_dir} \
            >> {log} 2>&1
        """
    # TODO: Expose baseline selectors/allocators via config and pass into run_baseline.py.
    # TODO: Add baseline-only entrypoint (baseline_only=true) to skip training and run eval+aggregate+stats.
    # Ref: https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html
    # TODO: Integrate baselines into eval/aggregate/stats path with treatment_id = "{algorithm}__mask_{use_masking}".
    # TODO: Baselines should skip training but still produce eval_wide-compatible rows.
    # TODO: Include baselines in stats by default; only skip tests when preconditions fail.
    # Ref: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html
