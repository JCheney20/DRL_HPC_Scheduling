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
| C-002 | Methodology | time-aware holdout policy enforced | policy doc | `Project_Github/docs/data_split_policy.md` | `github_repos/herasched/data/splits/logs/<split_id>.json` | in_progress |
| C-003 | Methods Workflow | interim local maskable smoke gate completed | run log | `Project_Github/docs/workflow_local.md` | `github_repos/herasched/trained_model/smoke_a2c_mask_on/selector/1000.zip`; `github_repos/herasched/trained_model/smoke_dqn_mask_on/selector/1000.zip`; `github_repos/herasched/trained_model/smoke_dqn_mask_off/selector/1000.zip` | in_progress |
<<<<<<< HEAD
| C-004 | Methods Workflow | pipeline automated with Snakemake | workflow doc | | | pending |
| C-005 | Evaluation | deterministic eval outputs generated | result files | | | pending |
| C-006 | Statistics | Friedman/Nemenyi flow implemented | analysis output | | | pending |
=======
| C-004 | Methods Workflow | pipeline automated with Snakemake | workflow doc | `Project_Github/docs/snakemake_pipeline.md` | | in_progress |
| C-005 | Evaluation | deterministic eval outputs generated | result files | | | pending |
| C-006 | Statistics | Friedman/Conover + Kendall's W + VDA flow implemented | analysis output | | | pending |
>>>>>>> e7ed95b (update:ver1.1)

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
