# Local Workflow Runbook (Template)

This file documents the local execution workflow from smoke checks to analysis.

## 1. Purpose

- validate the full pipeline locally before HPC deployment;
- catch schema and integration failures early;
- produce reproducible smoke artefacts for Submission 2.

## 2. Prerequisites

- Nix environment available
- Required scripts present
- Data split prepared (`*_dev70.tsv`, `*_holdout30.tsv`)

## 3. Environment Setup

```bash
<<<<<<< HEAD
cd github_repos/herasched
=======
cd Project_Github
>>>>>>> e7ed95b (update:ver1.1)
nix develop
```

## 4. Smoke Workflow

### Step 0: Generate Time-Aware Split

```bash
python scripts/make_split.py --source physical_job --ratio 0.7 --out-dir data/splits/
```

Expected outputs:

- `data/splits/physical_job_dev70.tsv`
- `data/splits/physical_job_holdout30.tsv`
- `data/splits/logs/<split_id>.json`

### Step 1: Guard Test (Holdout Rejection)

```bash
<<<<<<< HEAD
python train_agents.py \
=======
python src/train_agents.py \
>>>>>>> e7ed95b (update:ver1.1)
  --algo maskable_dqn \
  --name smoke_guard_should_fail \
  --trace splits/physical_job_holdout30.tsv \
  --save_interval 1000 \
  --total_saving 1 \
  --seed 123456
```

Expected behavior: fail-fast before environment setup with holdout guard error.

### Step 2: Smoke Train (Interim Maskable Gate, 1k steps)

```bash
<<<<<<< HEAD
python train_agents.py \
=======
python src/train_agents.py \
>>>>>>> e7ed95b (update:ver1.1)
  --algo maskable_a2c \
  --name smoke_a2c_mask_on \
  --trace splits/physical_job_dev70.tsv \
  --use-masking \
  --save_interval 1000 \
  --total_saving 1 \
  --seed 123456

<<<<<<< HEAD
python train_agents.py \
=======
python src/train_agents.py \
>>>>>>> e7ed95b (update:ver1.1)
  --algo maskable_dqn \
  --name smoke_dqn_mask_on \
  --trace splits/physical_job_dev70.tsv \
  --use-masking \
  --save_interval 1000 \
  --total_saving 1 \
  --seed 123456 \
  --buffer-size 2000

<<<<<<< HEAD
python train_agents.py \
=======
python src/train_agents.py \
>>>>>>> e7ed95b (update:ver1.1)
  --algo maskable_dqn \
  --name smoke_dqn_mask_off \
  --trace splits/physical_job_dev70.tsv \
  --no-use-masking \
  --save_interval 1000 \
  --total_saving 1 \
  --seed 123456 \
  --buffer-size 2000
```

### Step 3: Smoke Evaluate

```bash
<<<<<<< HEAD
python evaluate_agents.py --models-dir <dir> --split <split_id> --output <dir>
=======
python src/evaluate_agents.py --models-dir <dir> --split <split_id> --output <dir>
>>>>>>> e7ed95b (update:ver1.1)
```

### Step 4: Aggregate

```bash
<<<<<<< HEAD
python aggregate_results.py --input <metrics_dir> --output <summary_dir>
=======
python src/aggregate_results.py --input <metrics_dir> --output <summary_dir>
>>>>>>> e7ed95b (update:ver1.1)
```

### Step 5: Stats Sanity

```bash
<<<<<<< HEAD
python statistical_tests.py --input <summary_csv> --output <analysis_dir>
=======
python src/statistical_test.py --input <summary_csv> --output <analysis_dir>
>>>>>>> e7ed95b (update:ver1.1)
```

## 5. Smoke Gate Checklist

- [ ] time-aware split artefacts generated and metadata logged
- [ ] holdout guard test fails fast on holdout trace
- [ ] interim maskable smoke matrix completes (A2C mask-on, DQN mask-on, DQN mask-off)
- [ ] all 6 algorithms complete 1k-step smoke run (full gate)
- [ ] rewards are finite
- [ ] masks are valid for maskable algorithms
- [ ] evaluation outputs are written
- [ ] aggregation succeeds
- [ ] statistics script runs without schema errors

## 6. Expected Output Layout

- `models/`
- `logs/`
- `result/drl/`
- `analysis/`
- `plots/`

## 7. Common Failure Modes

- holdout trace accidentally used for training (guard should block):
- DQN replay OOM for dict observations (use smaller smoke `--buffer-size`):
- mask shape mismatch:
- missing metrics field:
- unstable run naming:
- split ID not propagated:

## 8. Local Run Manifest

| run_id | algorithm | seed | split_id | command | output_path | status |
|---|---|---|---|---|---|---|
| | | | | | | |
