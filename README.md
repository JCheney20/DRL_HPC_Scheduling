# HPC-DRL-Scheduler

**A Statistical Evaluation of Deep Reinforcement Learning Approaches for HPC Job Scheduling**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Honours Research Project | University of the Western Cape | 2026

---

## 📋 Overview

This repository contains the complete implementation, evaluation framework, and statistical analysis code for comparing Deep Reinforcement Learning algorithm families for HPC job scheduling on heterogeneous clusters.

**Primary Research Question:** Is the field's preference for masked PPO variants empirically justified compared to other DRL algorithm families?

### Key Contributions

- **Statistical Rigor:** First comprehensive comparison using Friedman tests and Nemenyi post-hoc analysis (Demšar 2006)
- **Real Workloads:** Production Slurm traces (~84k CPU jobs, ~28k GPU jobs) from heterogeneous clusters
- **Algorithm Coverage:** 4 DRL algorithms spanning 3 major families (Policy Gradient, Value-Based, Actor-Critic)
- **Reproducibility:** Multi-seed evaluation protocol following Henderson et al. 2018 best practices

---

## 🏗️ Repository Structure

```
HPC-DRL-Scheduler/
+-- presentations/           # Conference/submission presentations (Markdown format)
|   +-- submission1/        # Honours Submission 1 (4-min presentation)
|   +-- submission2/        # Honours Submission 2 (progress update)
|   +-- symposium/          # Honours Symposium (10-min presentation)
|
+-- training/               # DRL training infrastructure
|   +-- configs/           # Hyperparameter configurations (YAML/JSON)
|   +-- scripts/           # Training entry points (train_agent.py, etc.)
|   +-- logs/              # Training logs and checkpoints
|
+-- evaluation/            # Evaluation framework
|   +-- baselines/         # Classical scheduler results (42 combinations)
|   +-- drl_results/       # DRL algorithm evaluation outputs
|   +-- metrics/           # Performance metric computation
|
+-- statistical_analysis/  # Statistical testing framework
|   +-- scripts/           # Friedman, Nemenyi, CD diagram generation
|   +-- results/           # Test outputs (p-values, effect sizes)
|   +-- figures/           # Critical Difference diagrams, plots
|
+-- data/                  # Datasets (not included - see docs/data.md)
|   +-- traces/            # Slurm trace CSVs (physical_job.csv, deeplearn_job.csv)
|   +-- topologies/        # Cluster topology definitions
|
+-- docs/                  # Documentation
|   +-- setup.md           # Environment setup (Nix/pip)
|   +-- data.md            # Dataset descriptions and sourcing
|   +-- algorithms.md      # DRL algorithm details
|   +-- reproduction.md    # Step-by-step reproduction guide
|
+-- tests/                 # Test suite (smoke tests, integration tests)
|
+-- .github/
    +-- workflows/         # CI/CD (future: automated testing)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Dependencies:** PyTorch, Stable-Baselines3, Gymnasium, NumPy, SciPy, Matplotlib

### Installation

**Option 1: Nix (Recommended - Reproducible Environment)**
```bash
nix develop
```

**Option 2: pip (Fallback)**
```bash
pip install -r requirements.txt
```

### Running Baseline Evaluation

```bash
python evaluation/baselines/run_classical.py \
  --trace data/traces/physical_job.csv \
  --topology data/topologies/physical_topology.txt \
  --output evaluation/baselines/results/
```

### Training DRL Agents

```bash
python training/scripts/train_agent.py \
  --algorithm MaskablePPO \
  --config training/configs/maskable_ppo.yaml \
  --trace data/traces/physical_job.csv \
  --seed 42
```

### Statistical Analysis

```bash
python statistical_analysis/scripts/run_friedman.py \
  --results evaluation/drl_results/ \
  --metric avg_waiting_time \
  --output statistical_analysis/results/
```

---

## 📊 Algorithms Compared

| Algorithm | Family | Action Masking | Rationale |
|-----------|--------|----------------|-----------|
| **MaskablePPO** | Policy Gradient | Yes | Field's dominant choice (46% of papers) |
| **MaskableDQN** | Value-Based | Yes | Fair comparison (both masked) |
| **Vanilla PPO** | Policy Gradient | No | Ablation: isolate masking benefit |
| **A2C** | Actor-Critic | No | Represents A3C family |

---

## 📈 Evaluation Metrics

### Scheduling Performance
- **Average Waiting Time** (minimize) ⬇️
- **Average Slowdown** (fairness) ⬇️
- **Max Waiting Time** ⬇️
- **CPU/Node Utilization** (maximize) ⬆️

### Resource Costs (DRL only)
- **Peak Memory** (`tracemalloc`)
- **Mean Decision Latency**
- **Total Training Time**

---

## 🧪 Statistical Framework

Following **Demšar (2006)** methodology for algorithm comparison:

1. **Shapiro-Wilk Test** → Validate non-normality
2. **Friedman Test** → Omnibus significance (related samples)
3. **Nemenyi Post-hoc** → Pairwise comparisons
4. **Epsilon² (ε²)** → Effect size (practical significance)
5. **Critical Difference (CD) Diagrams** → Visualization

**Why Friedman?** Same workloads run through all algorithms → **related samples**, not independent.

---

## 📂 Datasets

### Primary Traces (Real Slurm Logs)
- **`physical_job.csv`**: ~84,000 CPU jobs, 87 heterogeneous nodes
- **`deeplearn_job.csv`**: ~28,000 GPU jobs, 28 heterogeneous nodes

**Note:** Datasets are not included in this repository due to size/privacy. See [`docs/data.md`](docs/data.md) for sourcing instructions.

---

## 🎓 Citation

If you use this code or methodology, please cite:

```bibtex
@misc{Cheney2026,
  title={Intelligent Job Scheduling for HPC Systems: A Statistical Evaluation of Deep Reinforcement Learning Approaches},
  author={Cheney, Justin M.},
  year={2026},
  school={University of the Western Cape},
  type={Honours Thesis}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Supervisor:** [To be filled]
- **Cluster Access:** UWC eResearch 
- **Frameworks:** Stable-Baselines3, Gymnasium, SciPy

---

## 📧 Contact

**Justin M. Cheney**  
University of the Western Cape  
Email: [Email TBD]  
Website: [Project Website TBD]

---

## 🛠️ Development Status

- [x] Gymnasium simulation environment (HPCsim)
- [x] Classical baseline evaluation (42 combinations)
- [x] Statistical analysis framework design
- [ ] DRL training infrastructure (In Progress)
- [ ] Multi-seed training runs (Planned: Weeks 3-5)
- [ ] Statistical analysis execution (Planned: Week 6)
- [ ] Results + Discussion writing (Planned: Weeks 7-12)

**Current Phase:** Infrastructure & Training Setup  
**Last Updated:** April 2026
