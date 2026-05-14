# Snakemake Pipeline Specification (Template)

This file documents the DAG contract used for local and HPC execution.

## 1. Pipeline Goal

- provide a reproducible end-to-end workflow from smoke tests to statistical outputs.

## 2. Rule Graph (Target)

`smoke -> train -> eval -> aggregate -> stats -> plots`

## 3. Rule Contracts

### `smoke`

- Inputs:
- Outputs:
- Command:

### `train`

- Inputs:
- Outputs:
- Command:

### `eval`

- Inputs:
- Outputs:
- Command:

### `aggregate`

- Inputs:
- Outputs:
- Command:

### `stats`

- Inputs:
- Outputs:
- Command:

### `plots`

- Inputs:
- Outputs:
- Command:

## 4. Config Schema

- algorithms:
- seeds:
- split_id:
- timesteps:
- output_root:
- mode (smoke/full):

## 5. Profiles

- local profile path:
<<<<<<< HEAD
- hpc profile path:
=======
- hpc profile path (Slurm):
>>>>>>> e7ed95b (update:ver1.1)
- cluster submit command:

## 6. Failure and Resume Policy

- failed jobs rerun behavior:
- skip completed behavior:
- idempotency constraints:

## 7. Validation Checklist

- [ ] DAG resolves
- [ ] smoke mode finishes
- [ ] full mode dry run passes
- [ ] outputs match expected schema

## 8. Example Commands

```bash
snakemake -n
snakemake --cores 4
snakemake --config mode=smoke
<<<<<<< HEAD
=======
snakemake --profile profiles/slurm --config mode=full
>>>>>>> e7ed95b (update:ver1.1)
```
