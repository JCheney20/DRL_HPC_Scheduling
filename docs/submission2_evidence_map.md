# Submission 2 Evidence Map (Template)

Use this file to map every claim in Submission 2 to concrete project evidence.

## 1. Usage

- one row per claim;
- include exact source file path;
- include generated artefact path when applicable.

## 2. Claim Mapping Table

| Claim ID | Paper Section | Claim Summary | Evidence Type | Source Path | Artefact Path | Status |
|---|---|---|---|---|---|---|
| C-001 | Methodology | 6 algorithms selected for comparison | protocol | `Project_Github/docs/methodology_protocol.md` | | in_progress |
| C-002 | Methodology | time-aware holdout policy enforced | policy doc | `docs/data_split_policy.md` | `data/splits/logs/<trace>_r70.json` | in_progress |
| C-003 | Methods Workflow | local maskable smoke gate completed | run log | `docs/workflow_local.md` | `trained_model/<trace>/<seed>/<algo>/selector/<step>.zip` | in_progress |
| C-004 | Methods Workflow | pipeline automated with Snakemake | workflow doc | `docs/snakemake_pipeline.md` | | in_progress |
| C-005 | Evaluation | deterministic eval outputs generated | result files | `src/evaluate_agents.py` | `result/<trace>/eval_runs/runs/*.csv` | pending |
| C-006 | Statistics | Friedman + Kendall's W + Nemenyi + Wilcoxon CIs + Page trend implemented | analysis output | `src/statistical_test.py` | `result/<trace>/stats/` | pending |

Notes:

- C-002 includes script-level holdout training guard verification (negative test fail-fast on holdout trace).
- C-003 currently reflects interim maskable gate (A2C mask-on, DQN mask-on/off); full 6-algorithm smoke gate remains pending.

## 3. Required Evidence Buckets

- Methodology protocol
- Data split policy
- Local workflow run logs
- HPC runbook (prepared)
- Snakemake DAG and config
- Aggregated metrics
- Statistical outputs
- Figures/tables for manuscript

## 4. Review Checklist

- [ ] every major claim has at least one source
- [ ] file paths are valid and current
- [ ] all figure/table references resolve
- [ ] no claim depends on undocumented output
