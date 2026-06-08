# DRL Scheduler Statistical Testbed

This repository is a reproducible statistical testing environment for deep reinforcement learning (DRL) job schedulers in heterogeneous HPC settings. It provides an end-to-end pipeline to train, evaluate, aggregate, and statistically compare six DRL algorithms (PPO, A2C, DQN) and their maskable variants, with time-aware data splits and a non-parametric analysis suite.

The implementation is Nix-first. For cluster execution, the same environment is intended to be containerized with Apptainer (see `docs/workflow_hpc.md`).

## Highlights

- single repo for training, evaluation, aggregation, and statistics
- time-aware split policy with holdout guardrails
- Snakemake orchestration with smoke and full runs
- reproducibility via Nix (pins Python + dependencies)
- structured outputs with metadata sidecars for audit trails

## Quickstart (Nix)

```bash
cd Project_Github
nix develop
just dry_run_smoke
```

## Key Commands

Snakemake targets (via just):

```bash
just dry_run_smoke
just run_smoke
just run_full
just run_full TRACE=deep_learn
```

Direct script entrypoints:

```bash
python src/make_split.py --source physical_job --out-dir data/splits/
python src/train_agents.py --algo maskable_a2c --trace data/splits/physical_job_dev70.tsv --seed 123456 --save_interval 1000 --total_saving 1
python src/evaluate_agents.py --manifest logs/run_log.csv --output-dir result/physical_job/eval_runs
python src/aggregate_results.py --manifest logs/run_log.csv --eval-root result/physical_job/eval_runs/runs --output-dir result/physical_job/aggregate
python src/statistical_test.py --input result/physical_job/aggregate/seed_summary.csv --output-dir result/physical_job/stats
```

## Project Layout

```
Project_Github/
├── src/                 # pipeline scripts + HPCsim + custom maskable algorithms
├── docs/                # methodology, workflow, and reproducibility docs
├── data/                # traces, topologies, splits (not committed)
└── presentations/       # presentation artefacts
```

## Results (Coming Soon)

- primary metrics summary table (avg_waiting, avg_slowdown)
- secondary metrics table (turnaround, utilization)
- CD diagram inputs and plots
- seed-level summary and statistical outputs

When results land, this section will link to the generated artefacts and the evidence map in `docs/submission2_evidence_map.md`.

## What This Repo Is (and Is Not)

- This repo is a reproducible statistical testing environment for DRL job schedulers.
- It is not a production scheduler or a benchmark leaderboard.
- It focuses on transparent pipelines, fixed splits, and auditable statistics.

## Data Access

The HPCSim environment and Slurm traces originate from Wang et al. The canonical source is the HPCSim repository and dataset release:

- https://gitlab.unimelb.edu.au/lingfeiw/herasched

Use this link for trace acquisition details and upstream documentation.

## Citations

If you use this repository, cite:

- Wang et al. for the Slurm traces and HPCSim environment (bib key: `Wang2025_1`).
- Stable-Baselines3 for PPO/A2C/DQN implementations (bib key: `stable-baselines3`).

Citation entries live in `Submmisions/bibliography.bib`.

## Documentation Index

- `docs/methodology_protocol.md`
- `docs/data_split_policy.md`
- `docs/workflow_local.md`
- `docs/workflow_hpc.md`
- `docs/snakemake_pipeline.md`
- `docs/reproducibility_checklist.md`

## Contact

Justin M. Cheney — 4323819@myuwc.ac.za
