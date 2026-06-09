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

VALID_ALGORITHMS = ["dqn", "a2c", "ppo", "maskable_dqn", "maskable_a2c","maskable_ppo"]
VALID_TRAD_ALGORITHMS = ["fcfs", "lcfs", "sjf", "wfp3", "unicep", "f_1", "f_2"]

configfile: "config.yaml"
container: config["container"]

TRACE_NAME      = config["trace_name"]
SEEDS           = config["seeds"]
ALGORITHMS      = [x for x in config["algorithms"] if x in VALID_ALGORITHMS] 
TRAD_ALGORITHMS = [x for x in config["trad_algorithms"] if x in VALID_TRAD_ALGORITHMS] 
SAVE_INTERVAL   = config["save_interval"]
TOTAL_SAVING    = config["total_saving"]
WINDOW_SIZE     = config.get("window_size", 512)
TAIL_SIZE       = config.get("tail_size", 64)
BUFFER_SIZE     = config.get("tail_size", 100_000)
TOPOLOGY_FILE   = config["topology_file"]
NODE_FILE       = config["node_file"]
ALPHA           = config["alpha"]
EVAL_MAX_STEPS  = config.get("eval_max_steps", None)
EVAL_DETERMINISTIC = config.get("eval_deterministic", True)
BASELINE_ONLY = config.get("baseline_only", False)

# Derived paths
RAW_TRACE    = f"data/{TRACE_NAME}.csv"
SPLIT_ID     = f"{TRACE_NAME}_dev70"
HOLDOUT_ID   = f"{TRACE_NAME}_holdout30"
DEV_SPLIT    = f"data/splits/{SPLIT_ID}.tsv"
HOLDOUT_SPLIT = f"data/splits/{HOLDOUT_ID}.tsv"
SPLIT_META   = f"data/splits/logs/{TRACE_NAME}_r70.json"

# Algorithms string for shell loops
ALGORITHMS_STR = " ".join(ALGORITHMS)
TRAD_ALGORITHMS_STR = " ".join(TRAD_ALGORITHMS)

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
    "train_agents.py",
    "evaluate_agents.py",
    "aggregate_results.py",
    "statistical_test.py",
]
for script in REQUIRED_SCRIPTS:
    if not Path(script).exists():
        raise FileNotFoundError(f"Required script not found: {script}")


# =============================================================================
# RULE all — Default target
# =============================================================================

localrules: make_split
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
     resources:
        mem_mb=16000,
        runtime=480,
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1"
    input:
        dev_split=DEV_SPLIT,
        split_meta=SPLIT_META,
    output:
        marker=touch(f"trained_model/{TRACE_NAME}/{{seed}}/.train_complete"),
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
    wildcard_constraints:
        seed=r"\d+",
    shell:
        """
        set -e
        mkdir -p trained_model/{params.trace_name}/{wildcards.seed}

        for algo in {params.algorithms}; do
            echo "[train_seed] seed={wildcards.seed} algo=$algo" >> {log}
            python train_agents.py \
                --algorithm "$algo" \
                --seed {wildcards.seed} \
                --trace splits/{SPLIT_ID}.tsv \
                --split_id {TRACE_NAME}_r70 \
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
    resources:
        mem_mb=8000,
        runtime=120,
        slurm_partition="cpu"
    input:
        marker=f"trained_model/{TRACE_NAME}/{{seed}}/.train_complete"
        manifest="logs/run_log.csv"
    output:
        marker=touch(f"result/{TRACE_NAME}/eval_runs/.seed_{{seed}}_complete"),
    log:
        f"logs/snakemake/{TRACE_NAME}/eval_seed_{{seed}}.log",
    params:
        eval_root=f"result/{TRACE_NAME}/eval_runs",
        max_steps_flag=EVAL_MAX_STEPS_FLAG,
        deterministic_flag=("--deterministic" if EVAL_DETERMINISTIC else "--no-deterministic"),
        filter_seed = lambda wildcards: wildcards.seed
    wildcard_constraints:
        seed=r"\d+",
    shell:
        """
        mkdir -p {params.eval_root}/runs

        python evaluate_agents.py \
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
    resources:
        mem_mb=8000,
        runtime=60,
        slurm_partition="cpu"
    input:
        eval_markers=expand(
            f"result/{TRACE_NAME}/eval_runs/.seed_{{seed}}_complete",
            seed=SEEDS,
        ),
        manifest="logs/run_log.csv"
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
        python aggregate_results.py \
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
    resources:
        mem_mb=8000,
        runtime=60,
        slurm_partition="cpu"
    input:
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
    output:
        stats_summary=f"result/{TRACE_NAME}/stats/stats_summary.json",
        pairwise_nemenyi=f"result/{TRACE_NAME}/stats/pairwise_nemenyi.csv",   # was pairwise_conover
        confidence_intervals=f"result/{TRACE_NAME}/stats/confidence_intervals.csv",  # new
        page_trend=f"result/{TRACE_NAME}/stats/page_trend.csv",               # new
        cd_diagram=f"result/{TRACE_NAME}/stats/cd_diagram_input.csv",
        stats_meta=f"result/{TRACE_NAME}/stats/stats_meta.json",
    log:
        f"logs/snakemake/{TRACE_NAME}/stats.log",
    params:
        output_dir=f"result/{TRACE_NAME}/stats",
        alpha=ALPHA,
    shell:
        """
        python statistical_test.py \
            --input {input.seed_summary} \
            --output-dir {params.output_dir} \
            --alpha {params.alpha} \
            >> {log} 2>&1
        """


# =============================================================================
# RULE baseline_comparison — Optional traditional scheduler baseline
# =============================================================================

rule baseline:
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
        algorithm=" ".join(TRAD_ALGORITHMS_STR),
        partition=TRACE_NAME,
    wildcard_constraints:
        seed=r"\d+",
    shell:
        """
        for algo in {params.algorithms}; do
          python run_baseline.py \
              --algorithm algo \
              --seed \
              --split_id {input.dev_split} \
              --partition {params.partition} \ 
              --result-dir {params.output_dir} \
              --manifest-path "logs/baseline_run_log.csv" \
              --force False \
              >> {log} 2>&1
        done
        """
    # TODO: Add baseline-only entrypoint (baseline_only=true) to skip training and run eval+aggregate+stats.
    # Ref: https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html
    # Ref: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html
