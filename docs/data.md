# Dataset Documentation

## Overview

This project uses **real Slurm trace logs** from production HPC clusters to ensure realistic evaluation.

**Note:** Raw trace files are **not included** in this repository due to:
- File size (~100MB+)
- Potential privacy/licensing constraints
- Reproducibility via documented sourcing

---

## Primary Datasets

### 1. `physical_job.csv` (CPU Cluster)

**Source:** Production Slurm logs from heterogeneous CPU cluster  
**Period:** [To be filled]  
**Jobs:** ~84,000  
**Nodes:** 87 heterogeneous nodes  
**Characteristics:**
- Mixed CPU architectures
- Varying memory capacities (32GB - 256GB)
- Heterogeneous core counts (8 - 64 cores/node)

**Schema:**
```
job_id, submit_time, wait_time, run_time, num_nodes, num_cores, memory_req, user_id, partition
```

**Preprocessing:**
- Removed failed/cancelled jobs
- Anonymized user IDs
- Filtered jobs with missing resource specifications

---

### 2. `deeplearn_job.csv` (GPU Cluster)

**Source:** Production Slurm logs from GPU cluster  
**Period:** [To be filled]  
**Jobs:** ~28,000  
**Nodes:** 28 heterogeneous GPU nodes  
**Characteristics:**
- Mixed GPU types (V100, A100)
- Varying VRAM (16GB - 80GB)
- CPU+GPU heterogeneity

**Schema:**
```
job_id, submit_time, wait_time, run_time, num_gpus, gpu_type, cpu_cores, memory_req, user_id, partition
```

**Preprocessing:**
- Same as `physical_job.csv`
- GPU type encoding (0=V100, 1=A100, etc.)

---

## Cluster Topologies

### `physical_topology.txt`

Adjacency list representation of the physical cluster network topology.

**Format:**
```
node_id neighbor_id_1 neighbor_id_2 ...
```

**Example:**
```
0 1 2
1 0 3 4
2 0 5
...
```

Used for **topology-aware allocation** in classical baselines.

---

### `deeplearn_topology.txt`

GPU cluster topology (similar format).

---

## Obtaining the Datasets

### Option 1: Official HPCSim Release (Recommended)

The canonical release for the HPCSim environment and Slurm traces is the Wang et al. repository:

- https://gitlab.unimelb.edu.au/lingfeiw/herasched

### Option 2: Use Publicly Available Traces

Alternative datasets with similar characteristics:

1. **Parallel Workloads Archive**  
   URL: [https://www.cs.huji.ac.il/labs/parallel/workload/](https://www.cs.huji.ac.il/labs/parallel/workload/)  
   Suggested traces: ANL-Intrepid, SDSC-SP2, CEA-Curie

2. **Grid Workloads Archive**  
   URL: [http://gwa.ewi.tudelft.nl/](http://gwa.ewi.tudelft.nl/)

### Option 3: Generate Synthetic Traces

Use the provided synthetic workload generator (future addition):
```bash
python data/generate_synthetic.py \
  --type balanced \
  --num_jobs 10000 \
  --output data/traces/synthetic_balanced.csv
```

---

## Data Statistics

### Physical Cluster (`physical_job.csv`)

| Metric | Value |
|--------|-------|
| Total jobs | 84,127 |
| Avg wait time | 2,098s |
| Avg slowdown | 15.21 |
| Max wait time | 81,105s |
| Avg job duration | 3,245s |
| Node utilization | 68.4% |
| CPU utilization | 72.1% |

### GPU Cluster (`deeplearn_job.csv`)

| Metric | Value |
|--------|-------|
| Total jobs | 28,543 |
| Avg wait time | [TBD] |
| Avg slowdown | [TBD] |
| GPU utilization | [TBD] |

---

## Data Validation

Before training, validate dataset integrity:

```bash
python data/validate_traces.py \
  --trace data/traces/physical_job.csv \
  --topology data/topologies/physical_topology.txt
```

Expected output:
```
✅ Schema validation passed
✅ No missing values
✅ Temporal ordering verified
✅ Resource constraints valid
```

---

## Privacy & Ethics

All datasets used in this project:
- Have been **anonymized** (user IDs hashed)
- Contain **no personally identifiable information**
- Are used strictly for **academic research purposes**
- Comply with institutional data policies

---

## Citation

If using the datasets and HPCSim environment, please cite:

```bibtex
@article{Wang2025_1,
  author = {Lingfei Wang and Maria A. Rodriguez and Nir Lipovetzky},
  title = {Optimizing {HPC} scheduling: a hierarchical reinforcement learning approach for intelligent job selection and allocation},
  journal = {Journal of Supercomputing},
  year = {2025},
  volume = {81},
  number = {8},
  month = {June},
  publisher = {Springer},
  doi = {10.1007/s11227-025-07396-3}
}
```

```bibtex
@misc{Cheney2026hpc_data,
  title={HPC Job Scheduling Traces for DRL Evaluation},
  author={Cheney, Justin M.},
  year={2026},
  howpublished={\url{https://github.com/[username]/HPC-DRL-Scheduler}}
}
```
