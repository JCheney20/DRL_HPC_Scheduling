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
    All Python lives under src/: the workflow scripts (train_agents.py,
    evaluate_agents.py, aggregate_results.py, statistical_test.py,
    run_baseline.py, baseline_aggregate.py, baseline_compare.py, select_best.py,
    visualise.py, make_split.py) alongside the library modules (utils.py,
    HPCsim/, the custom maskable algorithms). Root holds only this Snakefile,
    the justfile, flake.nix, and the config files.

    Every script uses absolute imports (`from src.utils import ...`,
    `from src.HPCsim.HPCsim import ...`), so each rule invokes them as a module
    from the repo root: `python -m src.<name>`. The repo root is the working
    directory (and thus on sys.path), which makes the `src` namespace package
    importable; running the files as direct paths (`python src/<name>.py`) would
    NOT resolve the `from src.*` imports and must not be used. REQUIRED_SCRIPTS
    below therefore lists the `src/`-prefixed paths.

Ref:
    Snakemake 9.x docs: https://snakemake.readthedocs.io/en/stable/
"""

import subprocess
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================


def _capture_git_commit() -> str:
    """Capture HEAD on the orchestrator (login node, outside the container).

    The metadata writers call git inside the Apptainer runtime, where git may be
    unavailable or refuse the bind-mounted repo — yielding git_commit=null. This
    Snakefile preamble runs on the login node inside the checkout, so it resolves
    HEAD reliably and injects it via the GIT_COMMIT env var (see utils.git_hash).
    Empty string if not resolvable (falls back to the in-process git call).
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


GIT_COMMIT = _capture_git_commit()

VALID_ALGORITHMS = ["dqn", "a2c", "ppo", "maskable_dqn", "maskable_a2c", "maskable_ppo"]
VALID_TRAD_ALGORITHMS = ["fcfs", "lcfs", "sjf", "wfp3", "unicep", "f_1", "f_2"]

TRACE_TO_PARTITION = {
    "physical_job": "physical",
    "deeplearn_job": "deeplearn",
}

configfile: "config.yaml"

container: config.get("container", "DRL_env.sif")

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
N_ENVS = config.get("n_envs", 1)     # SubprocVecEnv workers for PPO/A2C
BATCH_SIZE = config.get("batch_size", 2048)  # minibatch size for PPO/DQN gradient updates
N_EPOCHS = config.get("n_epochs", 5)         # PPO optimisation epochs per rollout
LEARNING_RATE = config.get("learning_rate", "3e-4")  # constant float or "linear_<start>" schedule

RAW_TRACE = f"data/{TRACE_NAME}.csv"
SPLIT_ID = f"{TRACE_NAME}_dev70"
HOLDOUT_ID = f"{TRACE_NAME}_holdout30"
DEV_SPLIT = f"data/splits/{SPLIT_ID}.tsv"
HOLDOUT_SPLIT = f"data/splits/{HOLDOUT_ID}.tsv"
SPLIT_META = f"data/splits/logs/{TRACE_NAME}_r70.json"

TRAD_ALGORITHMS_STR = " ".join(TRAD_ALGORITHMS)
EVAL_MAX_STEPS_FLAG = f"--max-steps {EVAL_MAX_STEPS}" if EVAL_MAX_STEPS else ""

PARETO_METRICS = config["pareto_metrics"]
PARETO_METRICS_STR = " ".join(PARETO_METRICS)
PARETO_TIEBREAKERS = config["pareto_tiebreakers"]
PARETO_TIEBREAKERS_STR = " ".join(PARETO_TIEBREAKERS)

VIS_CONFIG = config.get("visualisation", {})

# =============================================================================
# INPUT VALIDATION
# =============================================================================

if TRACE_NAME not in ["physical_job", "deeplearn_job"]:
    raise ValueError(f"config trace_name must be 'physical_job' or 'deeplearn_job', got '{TRACE_NAME}'")

if not Path(RAW_TRACE).exists():
    raise FileNotFoundError(f"Raw trace not found: {RAW_TRACE}")

REQUIRED_SCRIPTS = [
    "src/train_agents.py", "src/evaluate_agents.py", "src/aggregate_results.py",
    "src/statistical_test.py", "src/run_baseline.py", "src/baseline_aggregate.py",
    "src/baseline_compare.py", "src/select_best.py", "src/visualise.py",
]
for script in REQUIRED_SCRIPTS:
    if not Path(script).exists():
        raise FileNotFoundError(f"Required script not found: {script}")

# =============================================================================
# RULE all
# =============================================================================

localrules:
    make_split,

if BASELINE_ONLY:
    rule all:
        input: f"result/{TRACE_NAME}/baseline/baseline_summary.csv"
else:
    rule all:
        input:
            f"result/{TRACE_NAME}/stats/stats_summary.json",
            f"result/{TRACE_NAME}/.visualise_complete",
            f"result/{TRACE_NAME}/baseline/baseline_comparison.csv",
            f"result/{TRACE_NAME}/holdout/holdout_summary.csv",

# =============================================================================
# RULE make_split
# =============================================================================

rule make_split:
    input: trace=RAW_TRACE
    output:
        dev=DEV_SPLIT,
        holdout=HOLDOUT_SPLIT,
        metadata=SPLIT_META,
    log: f"logs/snakemake/{TRACE_NAME}/make_split.log"
    params: trace_name=TRACE_NAME
    shell:
        """
        python -m src.make_split --src {params.trace_name} >> {log} 2>&1
        """

# =============================================================================
# RULE train_agent — Fully Parallelized (1 Job per Seed x Algorithm)
# =============================================================================

rule train_agent:
    """
    Train a single algorithm for a single seed. SLURM handles concurrency.
    """
    input:
        dev_split=DEV_SPLIT,
        split_meta=SPLIT_META,
    output:
        marker=touch(f"trained_model/{TRACE_NAME}/{{seed}}/{{algo}}/.train_complete"),
    log:
        f"logs/snakemake/{TRACE_NAME}/train_{{seed}}_{{algo}}.log",
    wildcard_constraints:
        seed=r"\d+",
        algo="|".join(ALGORITHMS),
    resources:
        # Uniform sizing: every algorithm now runs N_ENVS SubprocVecEnv workers —
        # DQN included. Single-env DQN was throughput-bound on the L4s (HPCsim's
        # per-step obs build is the wall); MaskableDQN/SB3 DQN both support
        # multi-env collection, so all algorithms share one resource profile.
        mem_mb=120000,   # 120 GB of the 128 GB nodes: DQN's 150k float32 replay (~66 GB) + 20 env workers (~40 GB) ≈ 106 GB. On-policy needs far less but shares the uniform request (1 job/node, GPU-bound). 125 GB tripped the node's configurable RAM limit, so 120.
        runtime=720,     # 12 h ceiling kept as safety; with 20-env collection + TF32 (already on) every algorithm is expected to finish < 5 h.
        slurm_partition="main",
        gres="gpu:1",
        cpus_per_task=N_ENVS + 1,   # 20 SubprocVecEnv workers + 1 main, all algorithms.
    params:
        save_interval=SAVE_INTERVAL,
        total_saving=TOTAL_SAVING,
        window_size=WINDOW_SIZE,
        buffer_size=BUFFER_SIZE,
        tail_size=TAIL_SIZE,
        topology=TOPOLOGY_FILE,
        node=NODE_FILE,
        trace=SPLIT_ID,
        trace_name=TRACE_NAME,
        n_envs=N_ENVS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
    shell:
        """
        set -e
        # Commit captured on the login node (git may be unavailable in-container).
        export GIT_COMMIT="{GIT_COMMIT}"
        # Cap numpy/BLAS threads per process so the {N_ENVS} SubprocVecEnv
        # workers don't each spawn a full thread pool and thrash the node's cores.
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        # Unbuffered so a hard crash (e.g. OOM kill) still flushes its traceback to the log.
        export PYTHONUNBUFFERED=1
        mkdir -p trained_model/{params.trace_name}/{wildcards.seed}/{wildcards.algo}
        python -m src.train_agents \
            --algorithm {wildcards.algo} \
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
            --name {params.trace_name}/{wildcards.seed}/{wildcards.algo} \
            --n-envs {params.n_envs} \
            --batch-size {params.batch_size} \
            --n-epochs {params.n_epochs} \
            --learning-rate {params.learning_rate} \
            >> {log} 2>&1
        """

# =============================================================================
# RULE eval_run — one SLURM job per (seed × algo), all run in parallel
# =============================================================================

rule eval_run:
    input:
        marker=f"trained_model/{TRACE_NAME}/{{seed}}/{{algo}}/.train_complete",
    output:
        marker=touch(f"result/{TRACE_NAME}/eval_runs/.seed_{{seed}}_{{algo}}_complete"),
    log:
        f"logs/snakemake/{TRACE_NAME}/eval_{{seed}}_{{algo}}.log",
    wildcard_constraints:
        seed=r"\d+",
        algo="|".join(ALGORITHMS),
    resources:
        mem_mb=8000,
        # Eval is a single-env, full-trace deterministic pass — the same Python
        # per-step obs-build wall as ONE training worker, no 20-env parallelism.
        # A dev70 pass is ~59k steps; maskable adds a get_action_masks per step
        # and empirically blew BOTH 60 and 240 min ceilings with an empty log.
        # The loop now prints a steps/s heartbeat every 2k steps (unbuffered
        # below), so the log reveals fps and a hang is no longer mistaken for a
        # timeout. No GPU: the env build is the wall, not the forward pass. 480 =
        # headroom for slow maskable; tune from the heartbeat / eval_wall_s.
        runtime=480,
        slurm_partition="main",
    params:
        manifest="logs/run_log.csv",
        eval_root=f"result/{TRACE_NAME}/eval_runs",
        max_steps_flag=EVAL_MAX_STEPS_FLAG,
        deterministic_flag="--deterministic" if EVAL_DETERMINISTIC else "--no-deterministic",
    shell:
        """
        set -e
        export GIT_COMMIT="{GIT_COMMIT}"
        export PYTHONUNBUFFERED=1
        mkdir -p {params.eval_root}/runs

        python -m src.evaluate_agents \
            --manifest {params.manifest} \
            --output-dir {params.eval_root} \
            --filter-seed {wildcards.seed} \
            --filter-algo {wildcards.algo} \
            {params.deterministic_flag} \
            {params.max_steps_flag} \
            >> {log} 2>&1
        """

# =============================================================================
# RULE aggregate
# =============================================================================

rule aggregate:
    input:
        eval_markers=expand(
            f"result/{TRACE_NAME}/eval_runs/.seed_{{seed}}_{{algo}}_complete",
            seed=SEEDS,
            algo=ALGORITHMS,
        ),
    output:
        eval_wide=f"result/{TRACE_NAME}/aggregate/eval_wide.csv",
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
        algorithm_summary=f"result/{TRACE_NAME}/aggregate/algorithm_summary.csv",
        aggregate_meta=f"result/{TRACE_NAME}/aggregate/aggregate_metadata.json",
    log: f"logs/snakemake/{TRACE_NAME}/aggregate.log"
    resources:
        mem_mb=8000,
        runtime=60,
        slurm_partition="main",
    params:
        manifest="logs/run_log.csv",
        eval_root=f"result/{TRACE_NAME}/eval_runs/runs",
        output_dir=f"result/{TRACE_NAME}/aggregate",
    shell:
        """
        export GIT_COMMIT="{GIT_COMMIT}"
        python -m src.aggregate_results \
            --manifest {params.manifest} \
            --eval-root {params.eval_root} \
            --output-dir {params.output_dir} \
            >> {log} 2>&1
        """

# =============================================================================
# RULE stats
# =============================================================================

rule stats:
    input:
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
    output:
        stats_summary=f"result/{TRACE_NAME}/stats/stats_summary.json",
        pairwise_nemenyi=f"result/{TRACE_NAME}/stats/pairwise_nemenyi.csv",
        confidence_intervals=f"result/{TRACE_NAME}/stats/confidence_intervals.csv",
        page_trend=f"result/{TRACE_NAME}/stats/page_trend.csv",
        cd_diagram=f"result/{TRACE_NAME}/stats/cd_diagram_input.csv",
        stats_meta=f"result/{TRACE_NAME}/stats/stats_meta.json",
    log: f"logs/snakemake/{TRACE_NAME}/stats.log"
    resources:
        mem_mb=8000,
        runtime=60,
        slurm_partition="main",
    params:
        output_dir=f"result/{TRACE_NAME}/stats",
        alpha=ALPHA,
    shell:
        """
        python -m src.statistical_test \
            --input {input.seed_summary} \
            --output-dir {params.output_dir} \
            --alpha {params.alpha} \
            >> {log} 2>&1
        """

# =============================================================================
# RULE select_best
# =============================================================================

rule select_best:
    input:
        nemenyi=f"result/{TRACE_NAME}/stats/pairwise_nemenyi.csv",
        confidence_intervals=f"result/{TRACE_NAME}/stats/confidence_intervals.csv",
        page_trend=f"result/{TRACE_NAME}/stats/page_trend.csv",
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
    output:
        best_algo_json=f"result/{TRACE_NAME}/best/best_algorithm.json",
        marker=touch(f"result/{TRACE_NAME}/.select_best_complete"),
    log: f"logs/snakemake/{TRACE_NAME}/select_best.log"
    params:
        trace=TRACE_NAME,
        alpha=ALPHA,
    shell:
        """
        python -m src.select_best \
            --nemenyi {input.nemenyi} \
            --seed-summary {input.seed_summary} \
            --ci {input.confidence_intervals} \
            --page-trend {input.page_trend} \
            --output-dir result/{params.trace}/best \
            --alpha {params.alpha} \
            >> {log} 2>&1
        """

# =============================================================================
# RULE holdout_eval — evaluate the winning algorithm on the reserved holdout
# split (the one time the holdout is used; the winner is read from
# best_algorithm.json at runtime and its models come from the run manifest).
# =============================================================================

rule holdout_eval:
    input:
        best_algo_json=f"result/{TRACE_NAME}/best/best_algorithm.json",
        holdout=HOLDOUT_SPLIT,
    output:
        marker=touch(f"result/{TRACE_NAME}/holdout/.holdout_eval_{{seed}}_complete"),
    log:
        f"logs/snakemake/{TRACE_NAME}/holdout_eval_{{seed}}.log",
    wildcard_constraints:
        seed=r"\d+",
    resources:
        mem_mb=8000,
        # One holdout30 pass (~25k steps) per seed: the 10 seeds now run as
        # parallel jobs, replacing the old single job that serialised all 10
        # and blew this ceiling. Same slow-maskable risk as eval_run, so 480
        # with the same steps/s heartbeat in the log. Winner is read per job.
        runtime=480,
        slurm_partition="main",
    params:
        manifest="logs/run_log.csv",
        holdout_root=f"result/{TRACE_NAME}/holdout",
        holdout_trace=HOLDOUT_SPLIT,
        max_steps_flag=EVAL_MAX_STEPS_FLAG,
        deterministic_flag="--deterministic" if EVAL_DETERMINISTIC else "--no-deterministic",
    shell:
        """
        set -e
        export GIT_COMMIT="{GIT_COMMIT}"
        export PYTHONUNBUFFERED=1
        mkdir -p {params.holdout_root}/runs
        WINNER=$(python -c "import json; print(json.load(open('{input.best_algo_json}'))['treatment_id'])")
        echo "Holdout eval: winner=$WINNER seed={wildcards.seed} on {params.holdout_trace}"
        python -m src.evaluate_agents \
            --manifest {params.manifest} \
            --output-dir {params.holdout_root} \
            --filter-treatment "$WINNER" \
            --filter-seed {wildcards.seed} \
            --eval-trace {params.holdout_trace} \
            {params.deterministic_flag} \
            {params.max_steps_flag} \
            >> {log} 2>&1
        """

# =============================================================================
# RULE holdout_aggregate — summarise the winner's holdout runs across seeds
# (reuses aggregate_results; strict=off skips the non-winner manifest rows).
# =============================================================================

rule holdout_aggregate:
    input:
        markers=expand(
            f"result/{TRACE_NAME}/holdout/.holdout_eval_{{seed}}_complete",
            seed=SEEDS,
        ),
    output:
        holdout_summary=f"result/{TRACE_NAME}/holdout/holdout_summary.csv",
    log:
        f"logs/snakemake/{TRACE_NAME}/holdout_aggregate.log",
    resources:
        mem_mb=8000,
        runtime=60,
        slurm_partition="main",
    params:
        manifest="logs/run_log.csv",
        eval_root=f"result/{TRACE_NAME}/holdout/runs",
        output_dir=f"result/{TRACE_NAME}/holdout",
    shell:
        """
        set -e
        export GIT_COMMIT="{GIT_COMMIT}"
        python -m src.aggregate_results \
            --manifest {params.manifest} \
            --eval-root {params.eval_root} \
            --output-dir {params.output_dir} \
            --no-strict \
            >> {log} 2>&1
        cp {params.output_dir}/algorithm_summary.csv {output.holdout_summary}
        """

# =============================================================================
# RULE baseline
# =============================================================================

rule baseline:
    input:
        dev_split=DEV_SPLIT,
        split_meta=SPLIT_META,
    output:
        baseline_meta=f"result/{TRACE_NAME}/baseline/baseline_metadata.json",
    log: f"logs/snakemake/{TRACE_NAME}/baseline.log"
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
          python -m src.run_baseline \
              --algorithm "$algo" \
              --split_id {params.split_id} \
              --partition {params.partition} \
              --result-dir {params.output_dir} \
              --manifest-path {params.manifest_path} \
              --force \
              >> {log} 2>&1 &
        done
        wait
        touch {output.baseline_meta}
        """

# =============================================================================
# RULE baseline_aggregate
# =============================================================================

rule baseline_aggregate:
    input:
        baseline_meta=f"result/{TRACE_NAME}/baseline/baseline_metadata.json",
    output:
        baseline_summary=f"result/{TRACE_NAME}/baseline/baseline_summary.csv",
        baseline_eval_wide=f"result/{TRACE_NAME}/baseline/baseline_eval_wide.csv",
    log: f"logs/snakemake/{TRACE_NAME}/baseline_aggregate.log"
    params:
        result_dir=f"result/{TRACE_NAME}/baseline",
    shell:
        """
        python -m src.baseline_aggregate \
            --result-dir {params.result_dir} \
            --output {output.baseline_summary} \
            >> {log} 2>&1
        """

# =============================================================================
# RULE baseline_compare
# =============================================================================

rule baseline_compare:
    input:
        best_algo_json=f"result/{TRACE_NAME}/best/best_algorithm.json",
        seed_summary=f"result/{TRACE_NAME}/aggregate/seed_summary.csv",
        algorithm_summary=f"result/{TRACE_NAME}/aggregate/algorithm_summary.csv",
        baseline_summary=f"result/{TRACE_NAME}/baseline/baseline_summary.csv",
    output:
        comparison=f"result/{TRACE_NAME}/baseline/baseline_comparison.csv",
        descriptive=f"result/{TRACE_NAME}/baseline/descriptive_comparison_table.csv",
    log: f"logs/snakemake/{TRACE_NAME}/baseline_compare.log"
    params:
        alpha=ALPHA,
        metrics=PARETO_METRICS_STR,
    shell:
        """
        python -m src.baseline_compare \
            --best-algorithm {input.best_algo_json} \
            --seed-summary {input.seed_summary} \
            --algorithm-summary {input.algorithm_summary} \
            --baseline-summary {input.baseline_summary} \
            --metrics {params.metrics} \
            --alpha {params.alpha} \
            --output {output.comparison} \
            --descriptive-output {output.descriptive} \
            >> {log} 2>&1
        """

# =============================================================================
# RULE visualise
# =============================================================================

rule visualise:
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
    log: f"logs/snakemake/{TRACE_NAME}/visualise.log"
    params:
        trace=TRACE_NAME,
    shell:
        """
        python -m src.visualise --mode results \
            --trace-name {params.trace} \
            --stats-dir result/{params.trace}/stats \
            --aggregate-dir result/{params.trace}/aggregate \
            --output-dir result/{params.trace} \
            --no-show \
            >> {log} 2>&1
        """
