# Methodology Protocol (Template)

Use this file as the canonical methodology specification for Submission 2.

## 1. Study Scope

- Thesis title:
- Submission milestone:
- Date updated:
- Author:

## 2. Research Questions and Hypotheses

### Research Questions

- RQ1:
- RQ2:
- RQ3:

### Hypotheses

- H1:
- H2:
- H3:
- H4:

## 3. Algorithm Set

| Algorithm | Family | Masking | Implementation package | Notes |
|---|---|---|---|---|
| MaskablePPO | Policy gradient | Yes | sb3-contrib | |
| MaskableA2C | Actor-critic | Yes | local (`src/`) | custom implementation |
| MaskableDQN | Value-based | Yes | local (`src/`) | custom implementation |
| PPO | Policy gradient | No | stable-baselines3 | |
| A2C | Actor-critic | No | stable-baselines3 | |
| DQN | Value-based | No | stable-baselines3 | |

## 4. Environment and Data

- Environment implementation path:
- Primary traces:
- Topologies:
- Allocator policy:

### Data Governance

- Development/train split definition: first 70% of trace rows after stable time sort on `Submit` (`*_dev70.tsv`)
- Final holdout definition: last 30% of trace rows after stable time sort on `Submit` (`*_holdout30.tsv`)
- Optional blocked CV configuration: allowed on development/train partition only
- Leakage prevention controls: script-level holdout guard in `train_agents.py` rejects holdout-like trace paths

## 5. Training Protocol

- Timesteps per run: total steps = `save_interval × total_saving`. Production: `100000 × 30 = 3M`. Smoke: `100 × 2 = 200`.
- Seed set: fixed seed for smoke reproducibility (e.g. `123456`); production uses 5 seeds (`config.yaml`).
- Hyperparameter source: `config.yaml` (`batch_size`, `n_epochs`, `learning_rate`, `n_envs`, `window_size`, `tail_size`, `buffer_size`).
- Checkpoint cadence: every `save_interval` steps to `trained_model/<trace>/<seed>/<algo>/selector/<step>.zip`.
- Logging: manifest row per run in `logs/run_log.csv`; per-rule logs under `logs/snakemake/<trace>/`.

### Command Template

```bash
python -m src.train_agents --algorithm <algo> --name <run_name> \
  --trace data/splits/<trace>_dev70.tsv --seed <seed> \
  --save_interval <n> --total_saving <k>
```

Note for DQN smoke on high-dimensional dict observations:

- use reduced replay buffer to avoid memory exhaustion, e.g. `--buffer-size 2000`.

## 6. Evaluation Protocol

- Deterministic/stochastic policy mode: `eval_deterministic` (config; default deterministic).
- Development evaluation trace: the `*_dev70.tsv` split each model trained on (from the manifest).
- Holdout evaluation: after `select_best`, the winning algorithm is re-evaluated across all seeds on the reserved `*_holdout30.tsv` split (`holdout_eval` → `holdout_aggregate` → `result/<trace>/holdout/holdout_summary.csv`). This is the only use of the holdout; no tuning is performed on it.
- Evaluation outputs: per-run metrics CSV/JSON under `result/<trace>/eval_runs/runs/`; holdout under `result/<trace>/holdout/`.
- Resource profiling: per-decision latency (`decision_latency_mean_ms`) and eval wall time captured per run.

### Command Template

```bash
python -m src.evaluate_agents --manifest logs/run_log.csv \
  --output-dir result/<trace>/eval_runs \
  --filter-seed <seed> --filter-algo <algo> --deterministic
```

## 7. Metrics

### Primary Metrics

- average waiting time
- average slowdown

### Secondary Metrics

- max waiting time
- max slowdown
- average turnaround
- CPU utilization
- node utilization

### Resource Metrics

- training wall-clock
- inference decision latency
- peak memory footprint

## 8. Statistical Workflow

Sequence (as implemented in `src/statistical_test.py`):

1. Shapiro-Wilk (normality diagnostic only; report-only, non-blocking)
2. Friedman (omnibus non-parametric repeated-measures)
3. Kendall's W (effect size for the Friedman result)
4. Nemenyi post-hoc (only if Friedman is significant) → `pairwise_nemenyi.csv`
5. Wilcoxon signed-rank non-parametric CIs (pairwise) → `confidence_intervals.csv`
6. Page trend test (convergence/ordering) → `page_trend.csv`
7. CD diagram inputs → `cd_diagram_input.csv`
8. Pareto analysis (in `select_best`)

Notes:

- Friedman is sometimes critiqued for low power, but can be adequate when the number of groups is five or more (Mangiafico). Alternative higher-power options (Quade, ART ANOVA) are considered optional extensions rather than core pipeline steps.

### Command Template

```bash
python -m src.statistical_test --seed-summary result/<trace>/aggregate/seed_summary.csv \
  --output-dir result/<trace>/stats --alpha 0.05
```

## 9. Output Contracts

- Training outputs:
- Evaluation metrics files:
- Aggregated summary tables:
- Statistical results files:
- Plot outputs:

## 10. Threats to Validity

- Internal validity:
- External validity:
- Construct validity:
- Mitigation actions:

## 11. Reproducibility Metadata

- Git commit hash:
- Nix environment version:
- Seeds used:
- Split ID:
- Command logs location:
