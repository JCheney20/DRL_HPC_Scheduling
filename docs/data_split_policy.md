# Data Split Policy (Template)

Use this file to formalize split rules and leakage controls.

## 1. Policy Statement

- Final holdout is reserved for final reporting only.
- Tuning/model selection is restricted to development/train data.
- Optional blocked CV is allowed only within development/train partition.
- No random shuffle of time-ordered traces.

## 2. Data Sources

- Trace files: `data/physical_job.csv`, `data/deeplearn_job.csv` (selected via `--src`)
- Topology files: `data/topology/physical_topology.txt`, `data/topology/nodes.csv`
- Split script: `src/make_split.py`, invoked as `python -m src.make_split`

## 3. Split Definition

- Split basis (timestamp column): `Submit`
- Development/train proportion: 70% earliest rows after stable sort (`mergesort`)
- Final holdout proportion: 30% latest rows after stable sort
- Split ID naming convention: `<source>_r<ratio_percent>_<timestamp>`

Example:

- `physical_job_r70_20260422T101530`

## 4. Optional Blocked CV (Development Only)

- Enabled: yes/no
- Number of folds:
- Fold construction rule:
- Validation fold rotation strategy:

## 5. Leakage Prevention Controls

- Holdout access controls: final holdout is generated as separate `*_holdout30.tsv` artefact and excluded from tuning runs
- Script-level guardrails: `src/train_agents.py` rejects holdout-like training traces before environment setup
- Config flags that prevent accidental tuning on holdout: training commands must target `data/splits/*_dev70.tsv`; holdout training is fail-fast by design

Current holdout guard rule in training entrypoint:

- reject trace paths that contain `holdout` (case-insensitive), with explicit error.

## 6. Audit Trail Requirements

For each run, record:

- split ID;
- algorithm and seed;
- command;
- commit hash;
- outputs generated;
- timestamp.

## 7. Approval and Change Management

- Policy owner:
- Last updated: 2026-04-22
- Change log:
  - 2026-04-22: standardized split artefacts (`*_dev70.tsv`, `*_holdout30.tsv`) and metadata JSON logging under `data/splits/logs/`; documented training holdout guard.
