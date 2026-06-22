"""
Snakefile — DRLScheduler Complete Pipeline

Purpose:
    Orchestrate end-to-end training → evaluation → aggregation → statistics
    pipeline for DRL scheduling algorithms on HPC traces, plus a separate
    deterministic-baseline lane that is aggregated and statistically
    compared against the best DRL algorithm without ever entering the
    DRL-only Friedman/Nemenyi/Wilcoxon-CI/Page-trend test matrix (see
    methodology_protocol.md and TODO.md Phase 3: baselines are descriptive,
    not part of the repeated-measures hypothesis tests, by design).

DAG Flow:
    make_split (idempotent)
        ├─→ train_seed (parallelised by seed × algorithm)
        │       └─→ eval_seed (parallelised by seed)
        │               └─→ aggregate
        │                       └─→ stats ──────────────┐
        │                       └─→ select_best ←───────┘
        │                               └─→ visualise
        └─→ baseline (per trad-algorithm, NO seeds -- deterministic)
                └─→ baseline_aggregate
                        └─→ baseline_compare (needs select_best's output too)

Usage:
    # Full production run (train, eval, aggregate, stats, baseline, compare, visualise)
    snakemake --configfile config.yaml --cores all

    # Smoke test
    snakemake --configfile config.smoke.yaml --cores all

    # Dry run (validate DAG only)
    snakemake --configfile config.smoke.yaml --dry-run

    # Baseline only (skip all DRL training/eval)
    snakemake --configfile config.yaml --config baseline_only=True --cores all

Important:
    This Snakefile targets the DEVELOPMENT directory layout: workflow scripts
    (train_agents.py, evaluate_agents.py, aggregate_results.py,
    statistical_test.py, run_baseline.py, baseline_aggregate.py,
    baseline_compare.py, select_best.py, visualise.py, make_split.py) live at
    the REPO ROOT alongside this Snakefile and config.yaml. Only utils.py and
    HPCsim/ live under src/. Every script does `from src.utils import ...` /
    `from src.HPCsim.HPCsim import ...`, which resolves correctly when each
    script is run directly as `python <script>.py` from the repo root (the
    repo root itself lands on sys.path, making the src package importable).

    PRODUCTION LAYOUT DIFFERS: if workflow scripts are moved into src/ (as in
    a packaged/production layout), every `python <script>.py` invocation below
    must become `python -m src.<script>` instead (running scripts as direct
    file paths from inside src/ does NOT make the repo root importable the
    same way, so the `from src.utils import ...` style absolute imports break
    otherwise). REQUIRED_SCRIPTS' paths below would also need a `src/` prefix.
    No changes are needed to any of the Python scripts themselves either way
    -- only to how this Snakefile invokes them.

Ref:
    Snakemake 9.x docs: https://snakemake.readthedocs.io/en/stable/
"""

from pathlib import Path

# =============================================================================
# CONFIGURATION
# Load from --configfile; fall back to sensible defaults so --dry-run works
# without a configfile.
# =============================================================================

VALID_ALGORITHMS = ["dqn", "a2c", "ppo", "maskable_dqn", "maskable_a2c", "maskable_ppo"]
VALID_TRAD_ALGORITHMS = ["fcfs", "lcfs", "sjf", "wfp3", "unicep", "f_1", "f_2"]

# run_baseline.py's --partition CLI accepts "physical"/"deeplearn" (matching
# PARTITION_CONFIGS in src/utils.py), but TRACE_NAME is "physical_job" or
# "deep_learn" (matching the raw trace filename). These are deliberately
# different vocabularies -- TRACE_NAME names a FILE, partition names a
# CLUSTER PARTITION -- so they need an explicit map, not a string transform.
TRACE_TO_PARTITION = {
    "physical_job": "physical",
    "deep_learn": "deeplearn",
}


configfile: "config.yaml"


container: config["container"]


TRACE_NAME = config["trace_name"]
PARTITION = TRACE_TO_PARTITION[TRACE_NAME]
SEEDS = config["seeds"]
ALGORITHMS = [x for x in config["algorithms"] if x in VALID_ALGORITHMS]
TRAD_ALGORITHMS = [x for x in config["trad_algorithms"] if x in VALID_TRAD_ALGORITHMS]
SAVE_INTERVAL = config["save_interval"]
TOTAL_SAVING = config["total_saving"]
WINDOW_SIZE = config.get("window_size", 512)
TAIL_SIZE = config.get("tail_size", 64)
BUFFER_SIZE = config.get("buffer_size", 100_000)
TOPOLOGY_FILE = config["topology_file"]
NODE_FILE = config["node_file"]
ALPHA = config["alpha"]
EVAL_MAX_STEPS = config.get("eval_max_steps", None)
EVAL_DETERMINISTIC = config.get("eval_deterministic", True)
BASELINE_ONLY = config.get("baseline_only", False)

# Derived paths
RAW_TRACE = f"data/{TRACE_NAME}.csv"
SPLIT_ID = f"{TRACE_NAME}_dev70"
HOLDOUT_ID = f"{TRACE_NAME}_holdout30"
DEV_SPLIT = f"data/splits/{SPLIT_ID}.tsv"
HOLDOUT_SPLIT = f"data/splits/{HOLDOUT_ID}.tsv"
SPLIT_META = f"data/splits/logs/{TRACE_NAME}_r70.json"

# Algorithms string for shell loops
ALGORITHMS_STR = " ".join(ALGORITHMS)
TRAD_ALGORITHMS_STR = " ".join(TRAD_ALGORITHMS)

# Optional eval max-steps flag
EVAL_MAX_STEPS_FLAG = f"--max-steps {EVAL_MAX_STEPS}" if EVAL_MAX_STEPS else ""

PARETO_METRICS = config["pareto_metrics"]
PARETO_METRICS_STR = " ".join(PARETO_METRICS)
PARETO_TIEBREAKERS = config["pareto_tiebreakers"]
PARETO_TIEBREAKERS_STR = " ".join(PARETO_TIEBREAKERS)

VIS_CONFIG = config["visualisation"]

# =============================================================================
# INPUT VALIDATION (checked at DAG build time)
# =============================================================================

if TRACE_NAME not in ["physical_job", "deep_learn"]:
    raise ValueError(
        f"config trace_name must be 'physical_job' or 'deep_learn', got '{TRACE_NAME}'"
    )

if not Path(RAW_TRACE).exists():
    raise FileNotFoundError(f"Raw trace not found: {RAW_TRACE}")

# Paths are relative to the repo root, matching the DEVELOPMENT layout where
# workflow scripts live alongside this Snakefile (not under src/ -- see the
# module docstring above for the production-layout alternative).
REQUIRED_SCRIPTS = [
    "train_agents.py",
    "evaluate_agents.py",
    "aggregate_results.py",
    "statistical_test.py",
    "run_baseline.py",
    "baseline_aggregate.py",
    "baseline_compare.py",
    "select_best.py",
    "visualise.py",
]
for script in REQUIRED_SCRIPTS:
    if not Path(script).exists():
        raise FileNotFoundError(f"Required script not found: {script}")


# =============================================================================
# RULE all — Default target
# =============================================================================

localrules:
    make_split,

if BASELINE_ONLY:

    rule all:
        input:
            f"result/{TRACE_NAME}/baseline/baseline_summary.csv",

else:

    rule all:
        """Full pipeline target: train → eval → aggregate → stats → select_best
        → baseline → baseline_compare → visualise."""
        input:
            f"result/{TRACE_NAME}/stats/stats_summary.json",
            f"result/{TRACE_NAME}/.visualise_complete",
            f"result/{TRACE_NAME}/baseline/baseline_comparison.csv",

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
    params:
        trace_name=TRACE_NAME,
    shell:
        """
        python make_split.py \
            --src {params.trace_name} \
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

    Note: logs/run_log.csv is deliberately NOT declared as an output here.
    It is a single shared file that every seed's invocation appends to
    (via write_manifest_entry's fcntl-locked append in train_agents.py) --
    not a per-seed artifact. Snakemake requires every output of a rule to
    share the rule's wildcards (here, {seed}), and a shared cumulative file
    cannot honestly satisfy that for more than one wildcard value at a time.
    The per-seed `marker` output already correctly gates downstream rules:
    by the time it's touched, this seed's algorithms have all finished
    training AND had their rows appended to the manifest (the shell loop
    below does both, in order, before the rule's output is considered done),
    so eval_seed's marker-based dependency is sufficient on its own.
    """
    input:
        dev_split=DEV_SPLIT,
        split_meta=SPLIT_META,
    output:
        marker=touch(f"trained_model/{TRACE_NAME}/{{seed}}/.train_complete"),
    log:
        f"logs/snakemake/{TRACE_NAME}/train_seed_{{seed}}.log",
    wildcard_constraints:
        seed=r"\d+",
    resources:
        mem_mb=16000,
        runtime=480,
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1",
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
    shell:
        """
        set -e 

        mkdir -p trained_model/{params.trace_name}/{wildcards.seed}

        for algo in {params.algorithms}; do
            echo "[train_seed] seed={wildcards.seed} algo=$algo" >> {log}
            python train_agents.py \
                --algorithm "$algo" \
                --seed {wildcards.seed} \
                --trace data/splits/{SPLIT_ID}.tsv \
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
    input:
        marker=f"trained_model/{TRACE_NAME}/{{seed}}/.train_complete",
    output:
        marker=touch(f"result/{TRACE_NAME}/eval_runs/.seed_{{seed}}_complete"),
    log:
        f"logs/snakemake/{TRACE_NAME}/eval_seed_{{seed}}.log",
    wildcard_constraints:
        seed=r"\d+",
    resources:
        mem_mb=8000,
        runtime=120,
        slurm_partition="cpu",
    params:
        # NOT a tracked Snakemake input -- logs/run_log.csv is a single file
        # cumulatively appended to by every seed's train_seed job (see
        # train_seed's docstring). The per-seed `marker` input above already
        # guarantees this seed's rows exist in it by the time this rule runs;
        # Snakemake's file-dependency DAG isn't the right tool for tracking a
        # shared, multiply-written file, so the path is passed as a plain
        # CLI argument instead of a declared input.
        manifest="logs/run_log.csv",
        eval_root=f"result/{TRACE_NAME}/eval_runs",
        max_steps_flag=EVAL_MAX_STEPS_FLAG,
        deterministic_flag=(
            "--deterministic" if EVAL_DETERMINISTIC else "--no-deterministic"
        ),
    shell:
        """
        set -e 

        mkdir -p {params.eval_root}/runs

        python evaluate_agents.py \
            --manifest {params.manifest} \
            --output-dir {params.eval_root} \
            --filter-seed {wildcards.seed} \
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
    output:
        eval_wide=f"result/{TRACE_NAME}/aggregate/eval_wide.csv",
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
        algorithm_summary=f"result/{TRACE_NAME}/aggregate/algorithm_summary.csv",
        aggregate_meta=f"result/{TRACE_NAME}/aggregate/aggregate_metadata.json",
    log:
        f"logs/snakemake/{TRACE_NAME}/aggregate.log",
    resources:
        mem_mb=8000,
        runtime=60,
        slurm_partition="cpu",
    params:
        # Not a tracked input -- see eval_seed's params.manifest comment.
        # Waiting on every seed's eval_markers (above) already transitively
        # guarantees every seed's train_seed has appended its manifest rows.
        manifest="logs/run_log.csv",
        eval_root=f"result/{TRACE_NAME}/eval_runs/runs",
        output_dir=f"result/{TRACE_NAME}/aggregate",
    shell:
        """
        python aggregate_results.py \
            --manifest {params.manifest} \
            --eval-root {params.eval_root} \
            --output-dir {params.output_dir} \
            >> {log} 2>&1
        """


# =============================================================================
# RULE stats — Run statistical tests (DRL algorithms only -- see module docstring)
# =============================================================================


rule stats:
    """
    Run full non-parametric statistical pipeline on seed_summary.csv.
    Tests: Shapiro-Wilk, Friedman, Kendall W, Nemenyi, Wilcoxon NP CI,
           confidence curves, Page trend, CD diagram input.
    DRL algorithms only -- baselines are compared separately via
    baseline_compare (one-sample Wilcoxon), not folded into this matrix.
    """
    input:
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
    output:
        stats_summary=f"result/{TRACE_NAME}/stats/stats_summary.json",
        pairwise_nemenyi=f"result/{TRACE_NAME}/stats/pairwise_nemenyi.csv",
        confidence_intervals=f"result/{TRACE_NAME}/stats/confidence_intervals.csv",
        page_trend=f"result/{TRACE_NAME}/stats/page_trend.csv",
        cd_diagram=f"result/{TRACE_NAME}/stats/cd_diagram_input.csv",
        stats_meta=f"result/{TRACE_NAME}/stats/stats_meta.json",
    log:
        f"logs/snakemake/{TRACE_NAME}/stats.log",
    resources:
        mem_mb=8000,
        runtime=60,
        slurm_partition="cpu",
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
# RULE select_best — Pareto + statistical tie-breaking selection
# =============================================================================


rule select_best:
    """
    Select the best-performing DRL algorithm via Pareto dominance over
    pareto_metrics, filtered by Nemenyi indistinguishability, with ties
    broken by Wilcoxon-CI-backed comparison in pareto_tiebreakers order.
    """
    input:
        nemenyi=f"result/{TRACE_NAME}/stats/pairwise_nemenyi.csv",
        confidence_intervals=f"result/{TRACE_NAME}/stats/confidence_intervals.csv",
        page_trend=f"result/{TRACE_NAME}/stats/page_trend.csv",
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
    output:
        best_algo_json=f"result/{TRACE_NAME}/best/best_algorithm.json",
        marker=touch(f"result/{TRACE_NAME}/.select_best_complete"),
    log:
        f"logs/snakemake/{TRACE_NAME}/select_best.log",
    params:
        trace=TRACE_NAME,
        alpha=ALPHA,
    shell:
        """
        python select_best.py \
            --nemenyi {input.nemenyi} \
            --seed-summary {input.seed_summary} \
            --ci {input.confidence_intervals} \
            --page-trend {input.page_trend} \
            --output-dir result/{params.trace}/best \
            --alpha {params.alpha} \
            >> {log} 2>&1
        """


# =============================================================================
# RULE baseline — Run traditional scheduling baselines (no seeds: deterministic)
# =============================================================================


rule baseline:
    """
    Run each traditional scheduling heuristic once per trace (NOT per seed --
    these algorithms are deterministic, so repeating them across seeds would
    waste compute for zero statistical benefit; see run_baseline.py).
    """
    input:
        dev_split=DEV_SPLIT,
        split_meta=SPLIT_META,
    output:
        baseline_meta=f"result/{TRACE_NAME}/baseline/baseline_metadata.json",
    log:
        f"logs/snakemake/{TRACE_NAME}/baseline.log",
    params:
        output_dir=f"result/{TRACE_NAME}/baseline",
        manifest_path="logs/baseline_run_log.csv",
        algorithms=TRAD_ALGORITHMS_STR,
        split_id=SPLIT_ID,
        partition=PARTITION,
    shell:
        """
        set -e

        for algo in {params.algorithms}; do
          python run_baseline.py \
              --algorithm "$algo" \
              --split_id {params.split_id} \
              --partition {params.partition} \
              --result-dir {params.output_dir} \
              --manifest-path {params.manifest_path} \
              >> {log} 2>&1
        done
        touch {output.baseline_meta}
        """


# =============================================================================
# RULE baseline_aggregate — Fold baseline metrics into a summary table
# =============================================================================


rule baseline_aggregate:
    """
    Aggregate traditional baseline outputs into baseline_summary.csv.
    No seed-averaging (see baseline_aggregate.py) -- each baseline algorithm
    has exactly one deterministic value per trace.
    """
    input:
        baseline_meta=f"result/{TRACE_NAME}/baseline/baseline_metadata.json",
    output:
        baseline_summary=f"result/{TRACE_NAME}/baseline/baseline_summary.csv",
        baseline_eval_wide=f"result/{TRACE_NAME}/baseline/baseline_eval_wide.csv",
    log:
        f"logs/snakemake/{TRACE_NAME}/baseline_aggregate.log",
    params:
        result_dir=f"result/{TRACE_NAME}/baseline",
    shell:
        """
        python baseline_aggregate.py \
            --result-dir {params.result_dir} \
            --output {output.baseline_summary} \
            >> {log} 2>&1
        """


# =============================================================================
# RULE baseline_compare — Statistically compare best DRL vs each baseline
# =============================================================================


rule baseline_compare:
    """
    One-sample Wilcoxon comparison of the selected best DRL algorithm's seed
    distribution against each deterministic baseline's fixed value, plus a
    plain descriptive side-by-side table (all DRL algorithms + all
    baselines). Kept separate from stats/ -- see module docstrings in
    baseline_compare.py and run_baseline.py for why this is a one-sample
    test, not Friedman/Nemenyi.
    """
    input:
        best_algo_json=f"result/{TRACE_NAME}/best/best_algorithm.json",
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
        algorithm_summary=f"result/{TRACE_NAME}/aggregate/algorithm_summary.csv",
        baseline_summary=f"result/{TRACE_NAME}/baseline/baseline_summary.csv",
    output:
        comparison=f"result/{TRACE_NAME}/baseline/baseline_comparison.csv",
        descriptive=f"result/{TRACE_NAME}/baseline/descriptive_comparison_table.csv",
    log:
        f"logs/snakemake/{TRACE_NAME}/baseline_compare.log",
    params:
        alpha=ALPHA,
        metrics=PARETO_METRICS_STR,
    shell:
        """
        python baseline_compare.py \
            --best-algorithm {input.best_algo_json} \
            --seed-summary {input.seed_summary} \
            --algorithm-summary {input.algorithm_summary} \
            --baseline-summary {input.baseline_summary} \
            --metrics {params.metrics} \
            --alpha {params.alpha} \
            --output {output.comparison} \
            >> {log} 2>&1
        """


# =============================================================================
# RULE visualise — Generate all plots and tables
# =============================================================================


rule visualise:
    """
    Generate all plots and tables from aggregate/stats outputs.
    """
    input:
        marker=f"result/{TRACE_NAME}/.select_best_complete",
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
        eval_wide=f"result/{TRACE_NAME}/aggregate/eval_wide.csv",
        algo_summary=f"result/{TRACE_NAME}/aggregate/algorithm_summary.csv",
        cd_input=f"result/{TRACE_NAME}/stats/cd_diagram_input.csv",
        pairwise_nemenyi=f"result/{TRACE_NAME}/stats/pairwise_nemenyi.csv",
        confidence_intervals=f"result/{TRACE_NAME}/stats/confidence_intervals.csv",
        page_trend=f"result/{TRACE_NAME}/stats/page_trend.csv",
        stats_summary=f"result/{TRACE_NAME}/stats/stats_summary.json",
    output:
        marker=touch(f"result/{TRACE_NAME}/.visualise_complete"),
    log:
        f"logs/snakemake/{TRACE_NAME}/visualise.log",
    params:
        trace=TRACE_NAME,
    shell:
        """
        python visualise.py --mode results \
            --trace-name {params.trace} \
            --stats-dir result/{params.trace}/stats \
            --aggregate-dir result/{params.trace}/aggregate \
            --output-dir result/{params.trace} \
            --no-show \
            >> {log} 2>&1
        """
