# HPCSim: Simulation Environment

HPCSim is a lightweight, trace-driven simulation framework for exploring job scheduling strategies in HPC environments. It provides a Gymnasium-compatible environment where the observation space comprises the cluster state and a job queue window, and the action space represents job selection decisions.

This project uses HPCSim as its training and evaluation environment. The original implementation is from Wang et al. (2025); see the [citation](#citation) below.

## Features

- Flexible scheduling: supports heuristic, rule-based, and RL-based job selectors and resource allocators
- Detailed metrics: waiting time, bounded slowdown, turnaround time, CPU/GPU utilisation
- Trace-driven: uses real Slurm workload traces for realistic simulation
- Modular design: clean separation of scheduler, cluster, queue, and evaluator components

## Job Trace Data Format

HPCSim uses tab-separated trace files derived from Slurm accounting logs. Key fields:

| Field | Description |
|---|---|
| `JobID` | Unique job identifier |
| `UID`, `GID` | User and group IDs |
| `Account` | Project account |
| `AllocCPUS` | CPU cores allocated |
| `AllocNodes` | Compute nodes allocated |
| `Allgpu` | GPUs allocated |
| `Allmem` | Total memory allocated |
| `ReqCPUS` / `ReqNodes` / `Reqgpu` / `ReqMem` | Requested resources |
| `TimelimitRaw` | Job time limit (seconds) |
| `ElapsedRaw` | Actual wall-clock duration |
| `Submit` | Submission timestamp |
| `Start` / `End` | Start and end times |
| `State` | Final job state (`COMPLETED`, `FAILED`, `TIMEOUT`, …) |
| `Partition` | Slurm partition |

For unlisted fields see the [Slurm accounting documentation](https://slurm.schedmd.com/accounting.html).

## Node Configuration Format

CSV file describing available compute nodes:

| Field | Description |
|---|---|
| `Features` | CPU features (e.g. `avx512`) |
| `core` | CPU cores per node |
| `memory` | Total memory (MB) |
| `node_type` | Node identifier |
| `gpu` | GPU model, or `(null)` |
| `gpu_number` | Number of GPUs |
| `partition` | Partition membership |

## Topology Data Format

Plain-text file defining the cluster's switch hierarchy, used for topology-aware allocation:

```
SwitchName=sp-266-p16-1 Level=1 LinkSpeed=1 Switches=le-266-q11-1-res,...
SwitchName=le-266-q11-1-res Level=0 LinkSpeed=1 Nodes=spartan-bm[053-066]
```

| Field | Description |
|---|---|
| `SwitchName` | Switch identifier |
| `Level` | Topology level (0 = edge, 1 = core) |
| `LinkSpeed` | Link speed for simulation scaling |
| `Switches` | Connected child switches (higher-level only) |
| `Nodes` | Node hostnames connected to this switch |

## Citation

```bibtex
@article{Wang2025_1,
  author    = {Lingfei Wang and Maria A. Rodriguez and Nir Lipovetzky},
  title     = {Optimizing {HPC} scheduling: a hierarchical reinforcement learning
               approach for intelligent job selection and allocation},
  journal   = {Journal of Supercomputing},
  year      = {2025},
  volume    = {81},
  number    = {8},
  month     = {June},
  publisher = {Springer},
  doi       = {10.1007/s11227-025-07396-3}
}
```
