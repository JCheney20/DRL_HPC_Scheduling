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
<<<<<<< HEAD
| MaskableA2C | Actor-critic | Yes | local (`github_repos/herasched/src/`) | custom implementation |
| MaskableDQN | Value-based | Yes | local (`github_repos/herasched/src/`) | custom implementation |
=======
  | MaskableA2C | Actor-critic | Yes | local (`src/`) | custom implementation |
  | MaskableDQN | Value-based | Yes | local (`src/`) | custom implementation |
>>>>>>> e7ed95b (update:ver1.1)
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

- Timesteps per run: smoke default `--save_interval 1000 --total_saving 1` (1k steps)
- Seed set: fixed seed for smoke reproducibility (for example `123456`), multi-seed set for full comparison runs
- Hyperparameter source:
- Checkpoint cadence: every `save_interval` steps to `trained_model/<name>/selector/`
- Logging path conventions:

### Command Template

```bash
<<<<<<< HEAD
python train_agents.py --algo <algo> --trace splits/<trace>_dev70.tsv --seed <seed> --save_interval <n> --total_saving <k>
=======
python src/train_agents.py --algo <algo> --trace splits/<trace>_dev70.tsv --seed <seed> --save_interval <n> --total_saving <k>
>>>>>>> e7ed95b (update:ver1.1)
```

Note for DQN smoke on high-dimensional dict observations:

- use reduced replay buffer to avoid memory exhaustion, e.g. `--buffer-size 2000`.

## 6. Evaluation Protocol

- Deterministic/stochastic policy mode:
- Evaluation trace/split:
- Evaluation outputs:
- Resource profiling method:

### Command Template

```bash
<<<<<<< HEAD
python evaluate_agents.py --models-dir <dir> --split <split_id> --output <dir>
=======
python src/evaluate_agents.py --models-dir <dir> --split <split_id> --output <dir>
>>>>>>> e7ed95b (update:ver1.1)
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

Sequence:

<<<<<<< HEAD
1. Shapiro-Wilk
2. Friedman
3. Nemenyi
4. epsilon2 effect size
5. bootstrap 95% CI
6. CD diagram inputs
7. Pareto analysis
=======
1. Shapiro-Wilk (diagnostic only; low power at small n)
2. Friedman (omnibus)
3. Conover (post-hoc)
4. Kendall's W (effect size)
5. Vargha-Delaney A (pairwise effect size)
6. bootstrap 95% CI
7. CD diagram inputs
8. Pareto analysis

Notes:

- Friedman is sometimes critiqued for low power, but can be adequate when the number of groups is five or more (Mangiafico). Alternative higher-power options (Quade, ART ANOVA) are considered optional extensions rather than core pipeline steps.
>>>>>>> e7ed95b (update:ver1.1)

### Command Template

```bash
<<<<<<< HEAD
python statistical_tests.py --input <aggregate_csv> --output <analysis_dir>
=======
python src/statistical_test.py --input <aggregate_csv> --output <analysis_dir>
>>>>>>> e7ed95b (update:ver1.1)
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
