# HPCsim: A Simulation Framework for HPC Scheduling Research

**HPCsim** is a lightweight, trace-driven simulation framework designed for exploring job scheduling strategies in high-performance computing (HPC) environments. It enables rapid evaluation of selector–allocator combinations, supports both heuristic and learning-based policies, and captures detailed system metrics such as job waiting time, bounded slowdown, turnaround time, and multi-level resource utilization (CPU, GPU, memory).

---

## Features

- 🧠 **Flexible Scheduling**: Supports heuristic, rule-based, and RL-based job selectors and resource allocators.
- 📊 **Detailed Metrics**: Tracks system-level and user-centric metrics (e.g., waiting time, slowdown, utilization).
- 🧪 **Trace-Driven**: Uses real HPC workload traces (e.g., Slurm) for realistic simulation behavior.
- ⚙️ **Modular Design**: Clean separation of components (scheduler, cluster, queue, evaluator).
- 📉 **Visualization**: Outputs data suitable for heatmaps, trend plots, and comparative analysis.

---

## Example Use

```python
from hpcsim import HPCsim

env = HPCsim(
    scheduler='UNICEP',
    allocator='topology_aware',
    backfill_enable=True,
    topology_file='topology/deeplearn_topology.txt',
    node_file='topology/nodes.csv',
    trace_file='deeplearn_job.csv',
    random_job=False,
    window_size=100,
    tail_size=10
)

env.run()
print("Average Waiting Time:", env.evaluator.waiting_time())
print("Bounded Slowdown:", env.evaluator.bounded_slowdown())
print("Utilization:", env.utilization())
```

## Job Trace Data Format

HPCsim uses job trace files derived from Slurm accounting logs to simulate real workload behavior. Each row in the trace file represents a job and includes detailed fields about resource requests, allocation, timing, and job state. The format is tab-separated. A brief description of key fields is provided below:

| **Field**         | **Description** |
|-------------------|-----------------|
| `JobID`           | Unique identifier for the job. |
| `UID`, `GID`      | User and group IDs associated with the job. |
| `Account`         | Account name or project under which the job was submitted. |
| `AllocCPUS`       | Number of CPU cores allocated to the job. |
| `AllocNodes`      | Number of compute nodes allocated. |
| `Allgpu`          | Number of GPUs allocated. |
| `Allmem`          | Total memory allocated (e.g., `100G`). |
| `ReqCPUS`         | Number of CPU cores requested. |
| `ReqNodes`        | Number of nodes requested. |
| `Reqgpu`          | Number of GPUs requested (if specified). |
| `ReqMem`          | Memory requested by the job. |
| `TimelimitRaw`    | Job time limit in seconds. |
| `AdminComment`    | Optional administrator comments, such as system hints. |
| `Constraints`     | Hardware or system constraints specified (e.g., GPU model). |
| `CPUTimeRAW`      | Total raw CPU time consumed by the job. |
| `ElapsedRaw`      | Actual wall-clock duration of the job. |
| `Eligible`        | Timestamp when the job became eligible to run. |
| `Submit`          | Job submission timestamp. |
| `Start`           | Job start time. |
| `End`             | Job end time. |
| `State`           | Final job state (e.g., `COMPLETED`, `FAILED`, `TIMEOUT`). |
| `NodeList`        | Names of nodes used for job execution. |
| `Partition`       | Slurm partition the job was submitted to. |
| `Reserved`        | Time reserved but not used for execution. |
| `QOS`, `QOSRAW`   | Quality-of-service class and raw value. |
| `Reason`          | Scheduler reason for delay or hold (e.g., `Resources`). |
| `difference`      | Custom field (e.g., delay or scheduling discrepancy tracking). |

> ℹ️ For any unclear fields, refer to the [Slurm accounting documentation](https://slurm.schedmd.com/accounting.html). HPCsim may also preprocess or extend fields (e.g., normalize memory or timestamps) to suit simulation needs.

## Node Configuration Format

HPCsim uses a node configuration file to describe the compute nodes available in each partition. This file is typically a CSV where each row corresponds to a unique node or node type. It defines the node's hardware resources and attributes relevant to scheduling and allocation decisions.

| **Field**     | **Description** |
|---------------|-----------------|
| `Features`    | CPU features or labels (e.g., architecture extensions such as `avx512`). |
| `core`        | Number of CPU cores available on the node. |
| `memory`      | Total memory (in MB) available on the node. |
| `node_type`   | Unique node identifier (e.g., hostname or alias). |
| `config`      | Optional CPU configuration descriptor (e.g., socket/core/thread like `4:18:1`). |
| `gpu`         | GPU model or specification if available (e.g., `v100`). `(null)` if no GPU. |
| `partition`   | Partition to which the node belongs (e.g., `physical`, `deeplearn`). |
| `gpu_number`  | Number of GPUs installed on the node. |

> 📌 This configuration enables topology-aware and resource-aware allocation strategies. Nodes can be grouped or filtered by features, GPU presence, or partition membership as required by different scheduling experiments.

## Topology Data Format

HPCsim uses a topology configuration file to model the hierarchical network structure of HPC clusters. This information is essential for simulating topology-aware scheduling and allocation decisions. The topology file is a plain-text file with switch-level and node-level connections specified line by line.

Each entry defines either a high-level switch-to-switch connection or a low-level switch-to-node mapping.

### Example Format

SwitchName=sp-266-p16-1 Level=1 LinkSpeed=1 Switches=le-266-q11-1-res,le-266-q14-1-res,le-266-r17-1-res,le-266-r19-1-res

SwitchName=le-266-q11-1-res Level=0 LinkSpeed=1 Nodes=spartan-bm[053-066] SwitchName=le-266-q14-1-res Level=0 LinkSpeed=1 Nodes=spartan-bm[087-125] SwitchName=le-266-r17-1-res Level=0 LinkSpeed=1 Nodes=spartan-bm[001-020] SwitchName=le-266-r19-1-res Level=0 LinkSpeed=1 Nodes=spartan-bm[021-029],spartan-bm[039-043]

### Field Descriptions

| **Field**     | **Description** |
|---------------|-----------------|
| `SwitchName`  | Name or identifier of the switch. |
| `Level`       | Topology level (e.g., core switch = 1, edge switch = 0). |
| `LinkSpeed`   | Speed of the network links (used for simulation scaling). |
| `Switches`    | (Optional) List of connected child switches (used in higher-level switches). |
| `Nodes`       | (Optional) List or range of node hostnames directly connected to the switch. |

> 🔗 Switch-to-switch and switch-to-node mappings allow HPCsim to simulate realistic network distances and bottlenecks in topology-aware scheduling experiments.
