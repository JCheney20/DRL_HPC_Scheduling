# HPC Workflow Runbook (Template)

This file defines the migration path from local validated workflow to cluster execution.

## 1. HPC Objective

- execute full training/evaluation/statistics workflow at scale;
- preserve local output contracts and reproducibility rules;
- avoid redesign between local and HPC environments.

## 2. Entry Conditions (Must Be True)

- [ ] local smoke gate passed
- [ ] pipeline DAG runs locally end-to-end
- [ ] split policy locked
- [ ] run manifest prepared
- [ ] resume/skip-completed behavior tested

## 3. Environment Parity

- Nix environment used on cluster
- package versions pinned
- command-line interface consistent with local workflow
<<<<<<< HEAD
=======
- Apptainer image built from Nix flake (container path recorded in run metadata)
>>>>>>> e7ed95b (update:ver1.1)

## 4. Job Plan

| job_group | algorithm | seeds | timesteps | split_id | expected_runtime | resources |
|---|---|---|---|---|---|---|
| | | | | | | |

## 5. Execution Strategy

- scheduler type:
- parallelization approach:
- checkpoint frequency:
- failure recovery strategy:

## 6. Commands (Template)

```bash
snakemake --profile <hpc_profile> --cores <n>
```

<<<<<<< HEAD
=======
Suggested profile layout:

- `profiles/slurm/config.yaml`
- `profiles/slurm/slurm-submit.py`
- `profiles/slurm/cluster-config.yaml`

>>>>>>> e7ed95b (update:ver1.1)
or, if running script-by-script:

```bash
python train_batch.py --split <split_id> --timesteps <steps> --output <dir>
<<<<<<< HEAD
python evaluate_agents.py --split <split_id> --output <dir>
python aggregate_results.py --input <dir> --output <dir>
python statistical_tests.py --input <summary_csv> --output <analysis_dir>
=======
python src/evaluate_agents.py --split <split_id> --output <dir>
python src/aggregate_results.py --input <dir> --output <dir>
python src/statistical_test.py --input <summary_csv> --output <analysis_dir>
>>>>>>> e7ed95b (update:ver1.1)
```

## 7. Monitoring and Logging

- stdout/stderr log location:
- checkpoint location:
- health checks:
- progress report cadence:

## 8. HPC Output Validation

- [ ] all planned models generated
- [ ] evaluation files complete
- [ ] aggregate outputs complete
- [ ] stats outputs complete
- [ ] plots generated

## 9. Post-Run Archive

- archive path:
- metadata captured:
- export list for Submission 2:
